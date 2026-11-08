"""One uninterrupted centre -> left end -> right end -> centre attempt.

Left/right use the tail-to-head camera: +Y is left (tail), -Y right (head).
J/H are dispatched unchanged. Limits mean the existing profile bounds,
never a stall relabeled as a mechanical limit. No pose reset after t=0.
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.optimize import brentq

from snake_control.rolling_arc import ArcParameters
from snake_control.section_geometry import forward
from snake_control.simulation import backbone
from snake_control.tee_simulation import TeeSimulation
from validation.gm_arc_handoff_experiment import place_arc_for_adjustment
from validation.mode_handoff_experiment import ExperimentController
from .validate_mujoco import real, BASE, CURVE, RecordingBus
from .review import actual_lengths, review_one

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'validation/reports/curve_feeding_full_stroke_20261108/normal'
LABEL = 'full_stroke_curve'
LEG_NAMES = ['左侧调至边界', '右侧调至边界', '返回中间', '中间保持']


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output/(LABEL+'.json')).exists():
        raise FileExistsError('Use a new output directory; keep previous evidence')
    real.ContinuousFeeding = CURVE
    p = ArcParameters(joints=18, first_joint=7, last_joint=12)
    options = real.ContinuousFeedingOptions(first_joint=7, last_joint=12,
        amplitude_deg=60., requested_transfer_m=-.0645, automatic_return=False,
        tracking_tolerance_deg=8.)
    core = CURVE(real.LINK_LENGTHS_M, options)

    def seed_at(position):
        return core.profile(position, -.12*np.sin(np.pi*position))

    # Define a balanced initial material shape using angles/known lengths.
    # The two straight hanging ends have equal height when their Xs match.
    centre = brentq(lambda q: forward(seed_at(q)[0], real.LINK_LENGTHS_M)[-1, 0], -.5, 1.5)
    seed, focus, _, _ = seed_at(centre)
    seed_mask = core.helper.mask(focus)

    class SeedParameters(ArcParameters):
        def targets(self, phase):
            return seed.copy()

    sim = TeeSimulation('straight', SeedParameters(**asdict(p)), friction=.5)
    m, d = sim.model, sim.data
    place_arc_for_adjustment(SimpleNamespace(model=m, data=d, sim=sim), sim.p, .1)
    for _ in range(2500):
        sim.drive.apply_targets(d, seed, seed_mask)
        mujoco.mj_step(m, d)
    sim.p = sim.drive.p = p
    d.time = 0.
    initial_points = backbone(m, d).copy()
    np.savez_compressed(output/'initial.npz', qpos=d.qpos, qvel=d.qvel, warmstart=d.qacc_warmstart)
    c = ExperimentController(RecordingBus(sim))
    c.first, c.last = focus
    c.configured_window = c.feeding_origin = (7, 12)
    c.phase = c.command_phase = 0.
    c.mode, c.touched_hardware = 'hold', True
    c.targets, c.enabled = np.degrees(seed).tolist(), seed_mask.tolist()
    core.position, core.bias = centre, -.12*np.sin(np.pi*centre)
    core.targets, core.enabled, core.window = seed.copy(), seed_mask.copy(), focus
    core.stage, core.stage_since = 'acquire', 0.
    c.feeding, c.feeding_selected, c.feeding_paused = core, True, True
    c.feeding_clock_origin = c.feeding_pause_clock = 0.
    c.bus.prepare_enabled(np.degrees(d.qpos[sim.drive.qadr]), seed_mask)
    c.bus.send_targets(c.targets, c.enabled)

    commands, controls, rows, events, goals = [], [], [], [], []
    states, velocities, targets, masks, forces, points = ([] for _ in range(6))
    leg = action = 0
    destinations = [core.position_bounds[0], core.position_bounds[1], centre]
    stable_since = terminal = None
    failure = None
    next_feedback = next_sample = 0.
    progress = 15.
    stats = dict(passive_force_max_nm=0., saturated_joint_steps=0, joint_steps=0,
        limit_excess_max_rad=0., self_contact_steps=0, floor_contact_steps=0,
        max_shared_target_increment_deg=0.)
    limits = m.jnt_range[m.actuator_trnid[:, 0]]
    floor = m.geom('floor').id
    previous_targets, previous_mask = c.bus.targets_deg.copy(), c.bus.enabled.copy()
    previous_stage = None

    def press(key):
        event = dict(time_s=float(d.time), key='Ctrl+'+key, leg=leg,
            stage_before=core.stage, position_joints=core.position)
        try:
            with redirect_stdout(io.StringIO()):
                c.command(real.decode_hotkey(ord(key), True))
            event['result'] = 'accepted'
        except Exception as exc:
            event.update(result='rejected', reason=str(exc))
            commands.append(event)
            raise
        commands.append(event)

    press('H')
    try:
        for frame in range(20000):  # 400s physical ceiling, not an imposed dwell.
            t = float(d.time)
            if t+1e-9 >= next_feedback:
                c.refresh_compliance_feedback(); next_feedback += .2
            if core.stage != previous_stage:
                events.append(dict(time_s=t, leg=leg, stage=core.stage, position_joints=core.position))
                previous_stage = core.stage
            if core.fault:
                failure = core.fault
                press('G') if not c.feeding_paused else None
                terminal = t if terminal is None else terminal
            if leg < 3 and failure is None:
                destination = destinations[leg]
                reached = (core.position <= destination+1e-7 if leg in (0, 2)
                           else core.position >= destination-1e-7)
                if leg == 2 and reached:
                    press('G')
                    events.append(dict(time_s=t, kind='centre_coordinate_reached',
                        leg=leg, position_joints=core.position))
                    leg, terminal = 3, t
                elif leg < 2:
                    error = float(np.max(abs(np.degrees(core.profile(core.position, core.bias)[0])[core.enabled]
                                              -np.asarray(c.measured_angles())[core.enabled])))
                    stable_since = (t if stable_since is None else stable_since) if reached and error <= 8. else None
                    if stable_since is not None and t-stable_since >= .6:
                        press('G') if not c.feeding_paused else None
                        events.append(dict(time_s=t, kind='profile_boundary_reached', leg=leg,
                            position_joints=core.position, actual_lengths_m=actual_lengths(backbone(m, d)).tolist(),
                            qpos=d.qpos.tolist(), enabled=c.enabled.copy()))
                        leg += 1; stable_since = None; action += 1
                        press('J' if leg == 1 else 'H')
                        previous_stage = None
                if core.completed and leg < 3:
                    goals.append(dict(kind='relative_goal', action=action, leg=leg, time_s=t, snapshot=core.snapshot()))
                    action += 1
                    press('J' if leg == 1 else 'H')
                    previous_stage = None
            with redirect_stdout(io.StringIO()):
                c.tick(.02)
            mask, target = c.bus.enabled.copy(), c.bus.targets_deg.copy()
            shared = mask & previous_mask
            stats['max_shared_target_increment_deg'] = max(stats['max_shared_target_increment_deg'],
                float(np.max(abs(target[shared]-previous_targets[shared]), initial=0.)))
            previous_targets, previous_mask = target, mask
            controls.append(dict(time_s=t, action=action, leg=leg, mode=c.mode,
                paused=c.feeding_paused, feeding=core.snapshot(), motor_input=c.feeding_inputs,
                target_deg=c.targets.copy(), enabled=c.enabled.copy()))
            for _ in range(20):
                c.bus.apply_drive(); mujoco.mj_step(m, d)
                stats['joint_steps'] += 18
                stats['passive_force_max_nm'] = max(stats['passive_force_max_nm'],
                    float(np.max(abs(d.actuator_force[~mask]), initial=0.)))
                stats['saturated_joint_steps'] += int(np.count_nonzero(abs(d.actuator_force) >= p.torque_limit_nm-1e-6))
                q = d.qpos[sim.drive.qadr]
                stats['limit_excess_max_rad'] = max(stats['limit_excess_max_rad'],
                    float(np.maximum(np.maximum(limits[:, 0]-q, q-limits[:, 1]), 0.).max()))
                for contact in d.contact:
                    if floor in (contact.geom1, contact.geom2) and {int(contact.geom1), int(contact.geom2)} & sim.robot_geoms:
                        stats['floor_contact_steps'] += 1
                    a, b = m.geom_bodyid[[contact.geom1, contact.geom2]]-m.body('head').id
                    if min(a, b) >= 0 and abs(a-b) > 2 and contact.dist <= 0:
                        stats['self_contact_steps'] += 1
            if d.time+1e-9 >= next_sample:
                next_sample += .05
                pts = backbone(m, d)
                measurement = sim.measure(0.)
                lengths = actual_lengths(pts)
                rows.append(dict(time_s=float(d.time), stage=LEG_NAMES[leg], leg=leg, action=action,
                    feeding_stage=core.stage, operator_paused=c.feeding_paused,
                    position_joints=core.position, reference_roll_rad=0.,
                    x_min_m=float(pts[:, 0].min()), center_x_m=float(pts[:, 0].mean()),
                    support_n=measurement['normal_force_n'], ground_force_n=measurement['ground_contact_force_n'],
                    actual_head_hanging_m=float(lengths[0]), actual_tail_hanging_m=float(lengths[1]),
                    tracking_max_deg=float(np.max(abs(target[mask]-np.degrees(d.qpos[sim.drive.qadr])[mask]), initial=0.))))
                states.append(d.qpos.copy()); velocities.append(d.qvel.copy()); targets.append(np.radians(target))
                masks.append(mask); forces.append(d.actuator_force.copy()); points.append(pts.copy())
            if stats['floor_contact_steps']:
                failure = failure or 'floor contact'; terminal = terminal if terminal is not None else t
            if t >= progress:
                print(f't={t:.1f}s leg={leg} {core.stage} p={core.position:.3f} action={action} fault={core.fault}', flush=True)
                progress += 15.
            if terminal is not None and t-terminal >= 3.:
                break
        if terminal is None:
            failure = 'time ceiling before complete stroke'
    except Exception as exc:
        failure = repr(exc)
    finally:
        np.savez_compressed(output/(LABEL+'_trace.npz'), time=[r['time_s'] for r in rows],
            qpos=states, qvel=velocities, targets=targets, masks=masks,
            actuator_force=forces, backbone_world=points)
        with gzip.open(output/(LABEL+'_control.jsonl.gz'), 'wt', encoding='utf-8') as stream:
            for row in controls:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        summary = dict(case='full_stroke', backend='curve', scene='straight', joints=18,
            parameters=asdict(p), start_x=.1, binary_enable=True, feedback_period_s=.2,
            physics_period_s=.001, control_period_s=.02, hardware_run=False,
            commands=commands, stage_events=events, completed_actions=goals, handovers=c.bus.handovers,
            statistics=stats, final_time_s=float(d.time), final=core.snapshot(),
            completed=leg == 3 and failure is None, failure=failure,
            centre_position_joints=centre, destinations_joints=destinations,
            initial_actual_lengths_m=actual_lengths(initial_points).tolist(),
            final_actual_lengths_m=actual_lengths(backbone(m,d)).tolist(),
            left_means='+Y tail in tail-to-head view; H grows left',
            geometry=dict(link_lengths_m=real.LINK_LENGTHS_M, pipe_geom_size=m.geom('pipe_main').size.tolist(),
                pipe_geom_pos=m.geom('pipe_main').pos.tolist(), friction=m.geom('pipe_main').friction.tolist()))
        (output/(LABEL+'.json')).write_text(json.dumps(dict(summary=summary, samples=rows), ensure_ascii=False, indent=1), encoding='utf-8')
        print(json.dumps({k:summary[k] for k in ('completed','failure','final_time_s','centre_position_joints','destinations_joints','initial_actual_lengths_m','final_actual_lengths_m')}, ensure_ascii=False), flush=True)
        real.ContinuousFeeding = BASE
    if rows:
        review_one(output, LABEL)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    run(parser.parse_args().output)
