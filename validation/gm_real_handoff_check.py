"""Replay an 18-joint T-pipe checkpoint using the standalone real controller's core.

Only this harness uses world geometry to stand in for the operator at Ctrl+L.
The motor core receives sampled joint angles and commands binary torque enable.
"""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codes.Rolling_Arc_Real import GravityCrossing, GravityMomentOptions
from snake_control.simulation import backbone
from snake_control.tee_simulation import PIPE_HEIGHT
from validation.gm_arc_handoff_experiment import helix_alignment_ready
from validation.gm_locked_experiment import APPROACH_SETUP, recovery_metrics, right_u_ready
from validation.gm_replay_wrap_experiment import save
from validation.gm_shift_cycle_experiment import CycleTrial
from validation.gm_wrap_first_experiment import pipe_contacts


def run(source, output, from_right_u):
    record = json.loads((source/'locked_gait.json').read_text(encoding='utf-8'))
    trace = np.load(source/'locked_gait_trace.npz')
    if from_right_u:
        start = max(i for i, row in enumerate(record['samples']) if row['stage'] == 'unwind_line')
    else:
        start = next(i for i, row in enumerate(record['samples']) if row['stage'] == 'front_only_coil')
    trial = CycleTrial(setup=APPROACH_SETUP)
    model, data = trial.model, trial.data
    qadr = trial.sim.drive.qadr
    data.qpos[:] = trace['qpos'][start]
    data.qvel[:] = trace['qvel'][start]
    mujoco.mj_forward(model, data)
    gait = GravityMomentOptions(**APPROACH_SETUP['options'])
    angle = np.degrees(data.qpos[qadr])
    phase = record['samples'][start]['reference_roll_rad']
    if from_right_u:
        # The U is the entry to the release stage, not a new ARC or Helix.
        helix = np.degrees(trial.gait.targets(phase, gait.r_h1)[0])
        core = GravityCrossing([.0645]*19, trace['masks'][start], helix, gait, from_helix=True)
        core.phase = phase
        core._targets = trace['targets'][start].copy()
        core._enter('right_u_hold')
    else:
        # The first lead frame already preloads one hinge. Start from its underlying ARC.
        core = GravityCrossing([.0645]*19, trace['stiffness_scale'][start] >= .999, angle, gait)
    keep = np.arange(18) < core.options.retain_joints
    core.running = not from_right_u
    last_tick = last_feedback = -1.
    operator_events = []

    def observe():
        if core.stage == 'front' and core.front_extended:
            last = core.front.window[1]
            p = backbone(model, data)
            yz = p[:last+1, 1:]-[0., PIPE_HEIGHT]
            turn = np.ptp(np.unwrap(np.arctan2(yz[:, 1], yz[:, 0])))
            main, branch, _ = pipe_contacts(trial.sim)
            return (turn >= .65*2.*np.pi and max(np.linalg.norm(yz, axis=1)) <= .148 and
                    p[0, 0] > .380 and any(j <= last for j in main) and any(j <= last for j in branch))
        if core.stage == 'align':
            return helix_alignment_ready(trial, (90., 108.), 1.035, .10)
        if core.stage == 'right_u_hold' or (core.stage == 'unwind' and core.elapsed >= 3.):
            return right_u_ready(trial, keep) and trial.release_moment(keep)[1] >= .63
        if core.stage == 'drop_hold':
            tail = slice(model.body('head').id+keep.sum()+1, model.body('head').id+19)
            com = np.average(data.xipos[tail], axis=0, weights=model.body_mass[tail])
            return abs(com[1]) <= .015 and com[2] <= PIPE_HEIGHT-.10
        return False

    def control(_):
        nonlocal last_tick, last_feedback, angle
        if data.time-last_tick >= .02-1e-8:
            if data.time-last_feedback >= .2-1e-8:
                last_feedback, angle = data.time, np.degrees(data.qpos[qadr])
            if observe():
                operator_events.append(dict(stage=core.stage, time_s=float(data.time)))
                core.pause()
                last_feedback, angle = data.time, np.degrees(data.qpos[qadr])
                core.confirm(last_feedback, angle)
                core.running = True
            elif core.stage in ('front', 'align'):
                core.running = True
            core.advance(.02, data.time, last_feedback, angle)
            trial.phase, last_tick = core.phase, data.time
        return core.targets

    trial.phase_run(1, lambda: core.stage, 285., control, stiffness=lambda: core.enabled.astype(float),
                    stop_when=lambda _: core.stage == 'roll' and
                    sum(r['stage'] == 'roll' for r in trial.samples) >= 211)
    rolling = [r for r in trial.samples if r['stage'] == 'roll']
    masks, forces = np.asarray(trial.masks), np.asarray(trial.forces)
    save(trial, output, 'locked_gait', record['samples'][start]['time_s'],
         final_stage=core.stage, events=core.events, operator_events=operator_events,
         binary_enable=True, feedback_period_s=.2, recovery=recovery_metrics(rolling),
         passive_torque_max_nm=float(np.max(abs(forces[~masks]), initial=0.)),
         final_min_x_m=trial.samples[-1]['x_min_m'], dropped_to_ground=trial.stopped)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/gm_arc_shift_rolling/gravity_continuous'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--from-right-u', action='store_true')
    args = parser.parse_args()
    run(args.source, args.output, args.from_right_u)
