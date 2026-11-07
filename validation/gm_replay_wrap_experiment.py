"""Resume the successful rewrap phase of the saved second cycle."""
import argparse
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.gm_shift_cycle_experiment import CycleTrial


def phase_from_front(gait, radius, angles, support_count=12):
    """Fit phase using the powered portion of the helical reference."""
    references = [gait.targets(phase, radius)[0][:support_count] for phase in (0., math.pi/2)]
    coeff, *_ = np.linalg.lstsq(np.column_stack(references), angles[:support_count], rcond=None)
    return math.atan2(coeff[1], coeff[0])


def save(trial, output, label, source_time, **extra):
    rows = trial.samples
    summary = dict(scene=trial.sim.scene, joints=trial.sim.p.joints, source_time_s=source_time, dynamics_s=trial.data.time,
        rewrap_checks=trial.rewrap_checks, wrapped_samples=sum(r['tail_wrapped'] for r in rows),
        clear_samples=sum(r['x_min_m'] > .409 for r in rows),
        aux_roll_started=any(r['stage']=='aux_roll' for r in rows), **extra)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/(label+'_trace.npz'), time=[r['time_s'] for r in rows], qpos=trial.states,
        qvel=trial.velocities, targets=trial.targets, masks=trial.masks, actuator_force=trial.forces,
        stiffness_scale=trial.stiffness_scales)
    with (output/(label+'.json')).open('w', encoding='utf-8') as stream:
        json.dump(dict(summary=summary, samples=rows), stream, indent=1)
    print(json.dumps(summary, indent=2))


def run(root, output, right_drop=False, keep_original_support=False, passive_tail_damping=.15):
    trial = CycleTrial(root/'gm_wrap_first_20261106/global_shift.json', root/'gm_reproduction/round2/r2_t1.json')
    source = root/'gm_wrap_first_20261106/shift_cycle.json'
    rows = json.loads(source.read_text(encoding='utf-8'))['samples']
    drop = max(i for i, r in enumerate(rows) if r['cycle'] == 2 and r['stage'] == 'drop')
    fit = max(i for i, r in enumerate(rows) if r['cycle'] == 2 and r['stage'] == 'fit')
    with np.load(source.with_name('shift_cycle_trace.npz')) as trace:
        trial.phase = phase_from_front(trial.gait, trial.gait.o.r_h2, trace['targets'][fit])
        if right_drop:
            # Keep the working global rewrap orientation. Prepare the right U
            # in the same root pose before any dynamics and start loosened.
            trial.data.qpos[:] = trace['qpos'][fit]
            trial.data.qvel[:] = 0.
            target, lh = trial.gait.targets(trial.phase, trial.gait.o.r_h2, ls=.527, th1=math.pi)
            trial.data.qpos[trial.sim.drive.qadr] = target
            release_mask = trial.gait.helix_joints(lh)
            if keep_original_support:
                release_mask |= trace['masks'][drop]
            trial.released_support_count = int(release_mask.sum())
            trial.passive_tail_damping = passive_tail_damping
        else:
            trial.data.qpos[:] = trace['qpos'][drop]
            trial.data.qvel[:] = trace['qvel'][drop]
            trial.released_support_count = int(trace['masks'][drop].sum())
    mujoco.mj_forward(trial.model, trial.data)
    phase = trial.phase
    release = None
    if right_drop:
        center_y, moment = trial.release_moment(release_mask)
        if center_y >= -.02:
            raise ValueError('Prepared tail does not lie on the right')
        release = dict(tail_center_y_m=center_y, gravity_torque_nm=moment)
        trial.phase_run(2, 'drop', 3., lambda t: target, release_mask)
    trial.rewrap(2)
    save(trial, output, 'right_then_rewrap' if right_drop else 'old_rewrap',
         rows[fit if right_drop else drop]['time_s'], initial_phase_rad=phase, release=release,
         release_events=trial.release_events)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=Path(__file__).resolve().parent/'reports')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--right-drop', action='store_true')
    parser.add_argument('--keep-original-support', action='store_true')
    parser.add_argument('--passive-tail-damping', type=float, default=.15)
    args = parser.parse_args()
    if not math.isfinite(args.passive_tail_damping) or args.passive_tail_damping < 0:
        parser.error('--passive-tail-damping must be finite and non-negative')
    run(args.reports, args.output, args.right_drop, args.keep_original_support, args.passive_tail_damping)
