"""Simulate the confirmed gravity-moment gait on the horizontal tee."""
import argparse
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.gravity_global_shift import MeasuredGravityShift
from snake_control.gravity_moment import GravityMomentController, smooth
from snake_control.simulation import backbone
from snake_control.tee_simulation import PIPE_HEIGHT
from validation.gm_experiment import BRANCH_X, BRANCH_R, LINK_R, angle, top_crossings
from validation.gm_replay_wrap_experiment import phase_from_front, save
from validation.gm_shift_cycle_experiment import CycleTrial
from validation.gm_wrap_first_experiment import pipe_contacts


REPORTS = Path(__file__).resolve().parent/'reports'
APPROACH_SETUP = dict(
    scene='up60', joints=18, friction=.5, x_head_start_m=.2, align_head_deg=130.,
    detect=True, operator_triggers=[3.],
    options=dict(r_h1=.094, r_h2=.098, t1=.97, t2=1.09125),
)


def drop_metrics(rows):
    start, end = rows[0], rows[-1]
    front = np.unwrap(np.radians([row['front_angle_deg'] for row in rows]))
    com = np.array([row['released_tail_com_m'] for row in rows])
    end_centered = bool(abs(com[-1, 1]) <= .015 and com[-1, 2] <= PIPE_HEIGHT-.10)
    rotation = float(np.degrees(front[-1]-front[0]))
    advance = end['head_x_m']-start['head_x_m']
    return dict(centered=end_centered, tail_com_y_end_m=float(com[-1, 1]),
                tail_com_y_closest_m=float(np.min(abs(com[:, 1]))),
                tail_com_z_end_m=float(com[-1, 2]), front_rotation_deg=rotation, head_advance_m=advance,
                axial_body_advance_m=end['center_x_m']-start['center_x_m'],
                passive_force_max_nm=max(row['passive_force_max_nm'] for row in rows),
                supported_samples=sum(row['support_n'] > .1 for row in rows),
                branch_contact_samples=sum(bool(row['branch_links']) for row in rows),
                passed=bool(end_centered and rotation >= 15. and advance >= .01))


def recovery_metrics(rows):
    """Assess the full recovery interval, including contact changes as it rolls."""
    if not rows:
        return dict(passed=False, reason='recovery not reached')
    start, end = rows[0], rows[-1]
    clearance = BRANCH_X+BRANCH_R+LINK_R
    gripped = sum(bool(row['distal_contacts']) and row['tail_span_deg'] >= 75. and
                  row['tail_max_radius_m'] < .14 for row in rows)
    all_clear = all(row['x_min_m'] > clearance and not row['branch_links'] for row in rows)
    supported = all(row['support_n'] > .1 and row['ground_force_n'] < 1. for row in rows)
    duration = end['time_s']-start['time_s']
    advance = end['x_min_m']-start['x_min_m']
    return dict(passed=bool(all_clear and supported and gripped == len(rows) and
                            duration >= 10. and advance > .05),
                duration_s=duration, distal_engaged_samples=gripped, samples=len(rows),
                clear_boundary_m=clearance, all_clear=all_clear, supported=supported,
                trailing_advance_m=advance, min_x_end_m=end['x_min_m'],
                max_tail_radius_m=max(row['tail_max_radius_m'] for row in rows),
                min_tail_span_deg=min(row['tail_span_deg'] for row in rows))


def rewrap(trial, support_count, roll_rate):
    """Shift the measured body along a single curve while rolling its front."""
    g, d, m = trial.gait, trial.data, trial.model
    angles = d.qpos[trial.sim.drive.qadr].copy()
    radius = g.o.r_h2
    trial.phase = phase_from_front(g, radius, angles, support_count)
    # Extend the upstream helix so the last joint can leave the measured
    # transition as well. The short straight tip is retained during pickup.
    curve = MeasuredGravityShift(g, radius, trial.phase, angles, support_count,
                                 upstream_m=(support_count+1)*g.l)
    offset = roll = 0.
    start = d.time
    command = angles.copy()
    roll_start = None
    short_tip = 2*g.l
    partial = (len(angles)-support_count)*g.l-short_tip
    crossed_partial = None

    def coupled(t):
        nonlocal offset, roll, command, roll_start, crossed_partial
        error = max(abs(command-d.qpos[trial.sim.drive.qadr]))
        # Move the curve interval at a speed the joints can track.
        rate = .05*float(np.clip((math.radians(19)-error)/math.radians(9), 0., 1.))
        offset = min(curve.shift_limit_m, offset+rate*m.opt.timestep)
        if offset >= partial and crossed_partial is None:
            crossed_partial = t
        if offset >= .012:
            if roll_start is None:
                roll_start = t
            roll += roll_rate*smooth((t-roll_start)/.6)*m.opt.timestep
        command = np.clip(curve.targets(offset, roll), -np.pi/2, np.pi/2)
        return command

    trial.phase_run(1, 'coupled_rewrap', curve.shift_limit_m/.05+10., coupled)
    if offset == curve.shift_limit_m and not trial.stopped:
        # The shift is now pure helix. Continue at the same material phase,
        # then return to the tighter approach radius and normal rolling rate.
        phase_at_contact = trial.phase-curve.torsion*offset+roll

        def recover(t):
            blend = smooth(t/2.)
            phase = phase_at_contact+roll_rate*t+(g.o.roll_rate-roll_rate)*max(0., t-1.)
            return g.targets(phase, radius+(g.o.r_h1-radius)*blend)[0]

        trial.phase_run(1, 'recover_roll', 2.+2.*math.pi/g.o.roll_rate, recover)
    return dict(start_s=start, roll_start_s=None if roll_start is None else start+roll_start,
                shift_m=offset, short_tip_shift_m=partial,
                short_tip_passed_s=None if crossed_partial is None else start+crossed_partial,
                full_shift_m=curve.shift_limit_m, active_roll_rad=roll)


def approach_and_unwind(trial, summary, release_time):
    """Start on the pipe once, then run the existing approach and unwind."""
    g, sim, m, d = trial.gait, trial.sim, trial.model, trial.data
    controller = GravityMomentController(sim.p.joints, g.o, summary['detect'], summary['operator_triggers'])
    target, helix_length = g.targets(0., g.o.r_h1)
    nh = int(helix_length/g.l)
    x = summary['x_head_start_m']
    sim.place_shape(target, nh, x)
    if summary.get('align_head_deg') is not None:
        beta = math.radians(summary['align_head_deg']-angle(backbone(m, d)[0]))
        sim.place_shape(target, nh, x, beta)
        top = top_crossings(backbone(m, d))[-1]
        sim.place_shape(target, nh, x+.414-top, beta)

    def command(_):
        mean_torque = float(np.mean(abs(d.actuator_force[controller.prev_mask])))
        q, mask = controller.update(d.time, d.qpos[sim.drive.qadr], mean_torque)
        if not np.all(mask):
            raise RuntimeError('Approach replay reached the obsolete release sequence')
        trial.phase = controller.roll
        return q

    trial.phase_run(1, lambda: controller.stage, release_time, command)
    return trial.targets[-1].copy()


def run(checkpoint, checkpoint_time, output, slip_radius, opening_s, axial_offset, keep=None, damping=None,
        roll_rate=None, from_approach=False):
    if from_approach:
        summary = (APPROACH_SETUP if checkpoint is None else
                   json.loads(checkpoint.read_text(encoding='utf-8'))['summary'])
        trial = CycleTrial(passive_tail_damping=damping, setup=summary)
    elif checkpoint.suffix == '.json':
        record = json.loads(checkpoint.read_text(encoding='utf-8'))
        trial = CycleTrial(checkpoint, passive_tail_damping=damping)
        trace_path = checkpoint.with_name(checkpoint.stem+'_trace.npz')
    else:
        record = None
        trial = CycleTrial(REPORTS/'gm_wrap_first_20261106/global_shift.json',
                           REPORTS/'gm_reproduction/round2/r2_t1.json', damping)
        trace_path = checkpoint
    g, m, d = trial.gait, trial.model, trial.data
    if from_approach:
        if axial_offset:
            raise ValueError('Continuous approach requires zero axial offset')
        start_target = approach_and_unwind(trial, summary, checkpoint_time)
        initial = None
    else:
        with np.load(trace_path) as saved:
            i = int(np.argmin(abs(saved['time']-checkpoint_time)))
            initial = float(saved['time'][i])
            d.qpos[:] = saved['qpos'][i]
            d.qvel[:] = saved['qvel'][i]
            start_target = saved['targets'][i].copy()
        trial.phase = 0. if record is None else record['samples'][i]['roll_rad']
    ls = g.turn_length(g.o.r_h1)-math.pi*g.o.r_c+.25*g.l
    _, lh = g.targets(trial.phase, slip_radius, ls=ls, th1=math.pi)
    helix_mask = g.helix_joints(lh)
    mask = helix_mask if keep is None else np.arange(trial.full_mask.size) < keep
    opened = start_target.copy()
    opened[helix_mask] = g.targets(trial.phase, slip_radius)[0][helix_mask]
    if not from_approach:
        d.qpos[0] += axial_offset
        mujoco.mj_forward(m, d)
    main, branch, _ = pipe_contacts(trial.sim)
    y, moment = trial.release_moment(mask)
    points = backbone(m, d)
    release = dict(time_s=float(d.time), main_links=main, branch_links=branch,
                   tail_com_y_m=y, gravity_moment_nm=moment,
                   tail_supported=any(link > int(mask.sum()) for link in main),
                   front_supported=any(link <= int(mask.sum()) for link in main),
                   tail_start_z_m=float(points[-1, 2]), support_count=int(mask.sum()))

    def passive_drop(t):
        # Only relax the front curvature. The front roll phase is constant
        # and every released joint receives zero motor torque.
        return start_target+smooth(t/opening_s)*(opened-start_target)

    drop_start = len(trial.samples)
    release_time = d.time

    def hanging(_):
        # The geometry check stands in for an operator watching the drop.
        # The motor command remains fixed phase throughout the drop.
        if d.time-release_time < 2. or not trial.samples or trial.samples[-1]['time_s']-release_time < 2.:
            return False
        rows = trial.samples[max(drop_start, len(trial.samples)-3):]
        if len(rows) < 3 or any(abs(row['released_tail_y_m']) > .015 for row in rows):
            return False
        return drop_metrics(trial.samples[drop_start:])['passed']

    if from_approach and (trial.stopped or not release['tail_supported'] or
                          not release['front_supported'] or y >= -.02):
        save(trial, output, 'locked_gait', initial, checkpoint=str(checkpoint),
             release=release, failure='right U not supported at proposed release')
        return
    trial.phase_run(1, 'drop', 5., passive_drop, mask, stop_when=hanging if roll_rate is not None else None)
    metrics = drop_metrics(trial.samples[drop_start:])
    if roll_rate is not None and metrics['passed'] and not trial.stopped:
        trial.rewrap_checks.append(rewrap(trial, int(mask.sum()), roll_rate))
    recovery = recovery_metrics([row for row in trial.samples if row['stage'] == 'recover_roll'])
    save(trial, output, 'locked_gait', initial, checkpoint=None if checkpoint is None else str(checkpoint),
         slip_radius_m=slip_radius,
         axial_offset_m=axial_offset, opening_s=opening_s, passive_tail_damping=damping,
         release=release, drop=metrics, recovery=recovery, from_approach=from_approach,
         release_events=trial.release_events)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-time', type=float)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--slip-radius', type=float)
    parser.add_argument('--opening-s', type=float, default=.40)
    parser.add_argument('--axial-offset', type=float, default=0.)
    parser.add_argument('--keep', type=int)
    parser.add_argument('--damping', type=float)
    parser.add_argument('--roll-rate', type=float, help='Continue from the centered drop into coupled rewrap')
    parser.add_argument('--from-approach', action='store_true',
                        help='Run approach and unwind continuously; checkpoint time becomes release time')
    args = parser.parse_args()
    checkpoint = args.checkpoint if args.from_approach or args.checkpoint is not None else (
        REPORTS/'gm_right_drop_20261106/shift_cycle_trace.npz')
    release_time = args.checkpoint_time if args.checkpoint_time is not None else (10.85 if args.from_approach else .1)
    slip_radius = args.slip_radius if args.slip_radius is not None else (.098 if args.from_approach else .106)
    keep = args.keep if args.keep is not None else (8 if args.from_approach else None)
    roll_rate = args.roll_rate if args.roll_rate is not None else (.35 if args.from_approach else None)
    run(checkpoint, release_time, args.output, slip_radius, args.opening_s, args.axial_offset,
        keep, args.damping, roll_rate, args.from_approach)
