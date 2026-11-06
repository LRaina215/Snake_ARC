"""Run the gravity-moment crossing (paper Sec. II-A) on the 18-joint tee scenes; saves ACTUAL states.

The controller sees only time, mean |torque| of enabled joints and measured joint angles.
Pipe/branch geometry and contacts are evaluation only.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import math
import time
from dataclasses import asdict, replace
import numpy as np
import mujoco
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import backbone
from snake_control.tee_simulation import TeeSimulation, PIPE_HEIGHT
from snake_control.gravity_moment import GravityMomentController, GravityMomentOptions

BRANCH_X, BRANCH_R, LINK_R = .35, .03, .029


def angle(p):
    return float(np.degrees(np.arctan2(p[2] - PIPE_HEIGHT, p[1])))


def top_crossings(pts):
    th = np.degrees(np.unwrap(np.radians([angle(p_) for p_ in pts])))
    return sorted(pts[k, 0] + (pts[k + 1, 0] - pts[k, 0]) * (c - th[k]) / (th[k + 1] - th[k])
                  for k in range(len(pts) - 1) for c in (90 + 360 * n for n in range(-6, 7))
                  if (th[k] - c) * (th[k + 1] - c) < 0)


def run(output, label, scene, options, duration, x_head, detect=True, friction=.5, triggers=(), every=None,
        align_head_deg=None, align_x=.414, joints=18, stop_after_cross=False, u_on_top=False,
        hang_angle_deg=135., stop_by_eye=False, release_by_moment=False):
    p = ArcParameters(joints=joints, first_joint=1, last_joint=joints)
    sim = TeeSimulation(scene, p, 0., friction)
    ctl = GravityMomentController(joints, options, detect, triggers)
    q0, lh = ctl.g.targets(0., options.r_h1)
    nh = int(lh / ctl.g.l)
    fit_r = sim.place_shape(q0, nh, x_head)
    if align_head_deg is not None:
        # initial placement only: head at the given angle about the pipe, front-most top crossing at align_x
        beta = math.radians(align_head_deg - angle(backbone(sim.model, sim.data)[0]))
        sim.place_shape(q0, nh, x_head, beta)
        front = top_crossings(backbone(sim.model, sim.data))[-1]
        sim.place_shape(q0, nh, x_head + align_x - front, beta)
    head = sim.model.body('head').id
    passed_x = BRANCH_X + BRANCH_R + LINK_R
    choices = []
    if u_on_top:
        def operator_choice(t_now):
            """Operator: on the CURRENT body, pick the unwound length so the point where the helix ends
            (the U start, where the disabled tail will hang from) sits high on the non-branch side
            (target hang_angle_deg, 90 = top, 180 = -y horizontal). Keep >= 1.3 front turns."""
            cur = backbone(sim.model, sim.data)
            g = ctl.g
            best = None
            for extra in np.arange(-4.5, 4.51, .25) * g.l:
                ls = g.turn_length(options.r_h1) - math.pi * options.r_c + extra
                lh = g.L - ls - math.pi * options.r_c
                if ls < 0 or lh < 1.3 * g.turn_length(options.r_h1):
                    continue
                k, f = int(lh // g.l), (lh % g.l) / g.l
                ang = angle(cur[k] + f * (cur[k + 1] - cur[k]))
                score = abs(((ang - hang_angle_deg) + 180) % 360 - 180) + 2 * abs(extra / g.l)
                if best is None or score < best[0]:
                    best = (score, float(extra), ang, lh / g.turn_length(options.r_h1))
            choices.append(dict(t=round(t_now, 2), extra_links=round(best[1] / g.l, 2),
                                hang_angle_deg=round(best[2], 1), helix_turns_left=round(best[3], 2)))
            return best[1]
        ctl.unwind_hook = operator_choice
    if stop_by_eye:
        ctl.unwind_hook = None

        def operator_stop(t_now, lh_now):
            cur = backbone(sim.model, sim.data)
            g = ctl.g
            k, f = int(lh_now // g.l), (lh_now % g.l) / g.l
            ang = angle(cur[k] + f * (cur[k + 1] - cur[k]))
            hit = abs(((ang - hang_angle_deg) + 180) % 360 - 180) <= 10
            if hit:
                choices.append(dict(t=round(t_now, 2), stopped_ls_links=round((g.L - math.pi * options.r_c - lh_now) / g.l, 2),
                                    hang_angle_deg=round(ang, 1), helix_turns_left=round(lh_now / g.turn_length(options.r_h1), 2)))
            return hit
        ctl.line_stop = operator_stop
    if release_by_moment:
        def operator_release(t_now, n_helix):
            """Operator: on the CURRENT body, choose how many front joints stay enabled so the released
            part's actual gravity moment about the pipe axis is forward (positive), keeping >= 1.4 turns."""
            g = ctl.g
            n_min = int(math.ceil(1.4 * g.turn_length(options.r_h2) / g.l))
            best = None
            for n_keep in range(max(n_min, n_helix - 4), min(joints - 3, n_helix + 4) + 1):
                b = head + n_keep + 1
                m, y = sim.model.body_mass[b:], sim.data.xipos[b:, 1]
                tau = float(-9.81 * np.sum(m * y))
                if best is None or tau > best[1]:
                    best = (n_keep, tau)
            choices.append(dict(t=round(t_now, 2), release_after_joint=best[0], helix_joints=n_helix,
                                actual_release_moment_nm=round(best[1], 3)))
            return best[0]
        ctl.release_hook = operator_release
    rows, qpos, qvel, tlog, mlog = [], [], [], [], []
    torque_samples = []
    next_sample, crossed_at = 0., None
    wall = time.monotonic()
    while sim.data.time < duration:
        t = sim.data.time
        q = sim.data.qpos[sim.drive.qadr].copy()
        mask_prev = ctl.prev_mask
        mean_tau = float(np.mean(np.abs(sim.data.actuator_force[mask_prev]))) if mask_prev.any() else 0.
        if every and ctl.stage == 'approach' and not ctl.manual_triggers and ctl.cycles >= 1:
            ctl.manual_triggers.append(ctl.t0 + every)
        target, mask = ctl.update(t, q, mean_tau)
        sim.drive.apply_targets(sim.data, target, mask)
        mujoco.mj_step(sim.model, sim.data)
        if not np.isfinite(sim.data.qpos).all():
            raise RuntimeError('nonfinite state')
        if sim.data.time + 1e-9 >= next_sample:
            next_sample += .05
            row = sim.measure(ctl.roll)
            pts = backbone(sim.model, sim.data)
            force = sim.data.actuator_force
            mass, com = sim.model.body_mass[head:], sim.data.xipos[head:]
            row.update(stage=ctl.stage, cycle=ctl.cycles, roll_rad=ctl.roll, mean_abs_torque_nm=mean_tau,
                       x_min_m=float(pts[:, 0].min()), x_max_m=float(pts[:, 0].max()), head_x_m=float(pts[0, 0]),
                       front_angle_deg=angle(pts[4]), tau_x_nm=float(-9.81 * np.sum(mass * com[:, 1])),
                       enabled_count=int(mask.sum()),
                       saturated=bool(np.any(np.abs(force[mask]) >= p.torque_limit_nm - 1e-6)),
                       max_abs_joint_deg=float(np.degrees(np.abs(q).max())))
            row['branch_links'], row['branch_force_n'] = sim.branch_contacts()
            for key in ('qpos_rad', 'target_rad', 'actuator_force_nm', 'passive_actuator_force_max_nm'):
                row.pop(key)
            rows.append(row)
            qpos.append(sim.data.qpos.copy()); qvel.append(sim.data.qvel.copy())
            tlog.append(target.copy()); mlog.append(mask.copy())
            if ctl.stage == 'approach':
                torque_samples.append(mean_tau)
            if crossed_at is None and row['x_min_m'] > passed_x:
                crossed_at = row['time_s']
                if stop_after_cross:
                    ctl.stopped = True                 # operator: no further unwinding
            if stop_after_cross and ctl.stopped and ctl.stage == 'cycle_hold' and sim.data.time - ctl.t0 > 3.:
                break
            if row['min_height_m'] < .2:          # fell to the floor
                break
    output.mkdir(parents=True, exist_ok=True)
    times = np.array([r['time_s'] for r in rows])
    front = np.degrees(np.unwrap(np.radians([r['front_angle_deg'] for r in rows])))
    stages = []
    ev = ctl.events + [dict(t=float(times[-1]) + .05, stage='end')]
    for a, b in zip(ev[:-1], ev[1:]):
        i, k = int(np.argmin(abs(times - a['t']))), int(np.argmin(abs(times - (b['t'] - .05))))
        seg = rows[i:k + 1]
        if not seg:
            continue
        stages.append(dict(stage=a['stage'], t0=a['t'], t1=b['t'], cycle=seg[0]['cycle'],
                           dx_center_mm=(seg[-1]['center_m'][0] - seg[0]['center_m'][0]) * 1000,
                           x_min_end_mm=seg[-1]['x_min_m'] * 1000, head_x_end_mm=seg[-1]['head_x_m'] * 1000,
                           front_rotation_deg=float(front[k] - front[i]),
                           tau_x_nm=[round(min(r['tau_x_nm'] for r in seg), 3), round(max(r['tau_x_nm'] for r in seg), 3)],
                           saturated_fraction=float(np.mean([r['saturated'] for r in seg])),
                           unsupported=int(sum(r['normal_force_n'] < 1e-6 for r in seg)),
                           min_height_m=float(min(r['min_height_m'] for r in seg)),
                           branch_links=sorted({b2 for r in seg for b2 in r['branch_links']}),
                           branch_force_max_n=float(max(r['branch_force_n'] for r in seg))))
    first_contact = next((r['time_s'] for r in rows if r['branch_links']), None)
    summary = dict(label=label, scene=scene, friction=friction, x_head_start_m=x_head, fitted_start_radius_m=fit_r,
                   options=asdict(options), detect=detect, operator_triggers=list(triggers), repeat_trigger_s=every, align_head_deg=align_head_deg, joints=joints,
                   operator_unwind_choices=choices, duration_s=float(times[-1]),
                   first_branch_contact_s=first_contact, crossed_at_s=crossed_at,
                   unwind_cycles=ctl.cycles, events=ctl.events,
                   min_height_m=float(min(r['min_height_m'] for r in rows)),
                   max_ground_force_n=float(max(r['ground_contact_force_n'] for r in rows)),
                   unsupported_samples=int(sum(r['normal_force_n'] < 1e-6 for r in rows)),
                   max_abs_joint_deg=float(max(r['max_abs_joint_deg'] for r in rows)),
                   approach_torque=dict(p50=float(np.percentile(torque_samples, 50)) if torque_samples else None,
                                        p95=float(np.percentile(torque_samples, 95)) if torque_samples else None,
                                        p99=float(np.percentile(torque_samples, 99)) if torque_samples else None,
                                        max=float(max(torque_samples)) if torque_samples else None),
                   x_min_start_mm=rows[0]['x_min_m'] * 1000, x_min_end_mm=rows[-1]['x_min_m'] * 1000,
                   head_x_end_mm=rows[-1]['head_x_m'] * 1000, stages=stages,
                   environment=dict(mujoco=mujoco.__version__), wall_s=time.monotonic() - wall)
    (output / f'{label}.json').write_text(json.dumps(dict(summary=summary, samples=rows), indent=1), encoding='utf-8')
    np.savez_compressed(output / f'{label}_trace.npz', time=times, qpos=np.array(qpos), qvel=np.array(qvel),
                        targets=np.array(tlog), masks=np.array(mlog))
    return summary


def print_summary(s):
    keys = ('duration_s', 'first_branch_contact_s', 'crossed_at_s', 'unwind_cycles', 'min_height_m',
            'max_ground_force_n', 'unsupported_samples', 'max_abs_joint_deg', 'approach_torque',
            'x_min_start_mm', 'x_min_end_mm', 'head_x_end_mm', 'fitted_start_radius_m', 'operator_unwind_choices')
    print(json.dumps({k: s[k] for k in keys}, ensure_ascii=False))
    for st in s['stages']:
        print(f"c{st['cycle']} {st['stage']:12s} {st['t0']:6.1f}-{st['t1']:6.1f}s dx={st['dx_center_mm']:+7.1f} "
              f"xmin={st['x_min_end_mm']:5.0f} head={st['head_x_end_mm']:5.0f} rot={st['front_rotation_deg']:+7.1f} "
              f"tau={st['tau_x_nm']} sat={st['saturated_fraction']:.2f} unsup={st['unsupported']} "
              f"branch={st['branch_links']}/{st['branch_force_max_n']:.1f}N")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--scene', default='up60')
    parser.add_argument('--duration', type=float, default=150.)
    parser.add_argument('--x-head', type=float, default=.20)
    parser.add_argument('--r1', type=float, default=.092)
    parser.add_argument('--r2', type=float)
    parser.add_argument('--roll-rate', type=float, default=.6)
    parser.add_argument('--t1', type=float, default=1.)
    parser.add_argument('--no-detect', action='store_true')
    parser.add_argument('--trigger', type=float, nargs='*', default=[], help='operator trigger times (s)')
    parser.add_argument('--joints', type=int, default=18, choices=[18, 20, 26])
    parser.add_argument('--no-crossing-roll', action='store_true')
    parser.add_argument('--direct-cycle', action='store_true')
    parser.add_argument('--keep-links', type=float, default=2.)
    parser.add_argument('--rewind-s', type=float, default=6.)
    parser.add_argument('--extend-s', type=float, default=2.)
    parser.add_argument('--extend-hold-s', type=float, default=1.)
    parser.add_argument('--u-on-top', action='store_true', help='operator picks the unwound length so the U starts on top')
    parser.add_argument('--rewind-loose', action='store_true')
    parser.add_argument('--cycle-hold-s', type=float, default=1.)
    parser.add_argument('--stop-by-eye', action='store_true')
    parser.add_argument('--shift-fit', action='store_true', help='phase 4 by shift control (no roll compensation)')
    parser.add_argument('--shift-rate', type=float, default=0., help='phase 4 whole-body shift on a fixed curve (m/s)')
    parser.add_argument('--release-by-moment', action='store_true',
                        help='operator picks the released joints from the actual tail gravity moment')
    parser.add_argument('--hang-angle', type=float, default=135.)
    parser.add_argument('--stop-after-cross', action='store_true')
    parser.add_argument('--align-x', type=float, default=.414)
    parser.add_argument('--align-head', type=float, help='initial head angle about the pipe (deg); head-end top crossing at 0.414 m')
    parser.add_argument('--repeat-trigger', type=float, help='re-trigger this long after returning to phase 1')
    parser.add_argument('--friction', type=float, default=.5)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'reports/gm_reproduction')
    args = parser.parse_args()
    o = replace(GravityMomentOptions(), r_h1=args.r1, r_h2=args.r2 if args.r2 else args.r1 + .004,
                roll_rate=args.roll_rate, t1=args.t1, t2=1.125 * args.t1, crossing_roll=not args.no_crossing_roll,
                direct_cycle=args.direct_cycle, keep_links=args.keep_links, rewind_s=args.rewind_s, extend_s=args.extend_s,
                extend_hold_s=args.extend_hold_s, rewind_loose=args.rewind_loose, cycle_hold_s=args.cycle_hold_s,
                shift_fit=args.shift_fit, shift_rate=args.shift_rate)
    print_summary(run(args.output, args.label, args.scene, o, args.duration, args.x_head,
                      detect=not args.no_detect, friction=args.friction, triggers=args.trigger, every=args.repeat_trigger, align_head_deg=args.align_head,
                      align_x=args.align_x, joints=args.joints, stop_after_cross=args.stop_after_cross,
                      u_on_top=args.u_on_top, hang_angle_deg=args.hang_angle,
                      stop_by_eye=args.stop_by_eye, release_by_moment=args.release_by_moment))
