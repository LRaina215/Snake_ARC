"""Compare complete tail intake from one saved, physically supported fore helix."""
import argparse
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import sys
import types

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation import progressive_tail_candidate as candidate
from snake_control.simulation import backbone
from snake_control.tee_simulation import PIPE_HEIGHT
from validation.gm_locked_experiment import APPROACH_SETUP
from validation.gm_replay_wrap_experiment import save
from validation.gm_shift_cycle_experiment import CycleTrial


def baseline_module():
    source = subprocess.run(['git', 'show', '6c2f71c:codes/Rolling_Arc_Real.py'],
                            cwd=ROOT, capture_output=True, check=True).stdout.decode('utf-8')
    module = types.ModuleType('tail_intake_baseline')
    module.__file__ = str(ROOT/'codes/Rolling_Arc_Real.py')
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def run(module, output, label, source):
    archive = json.loads((source/'locked_gait.json').read_text(encoding='utf-8'))
    trace = np.load(source/'locked_gait_trace.npz')
    index = max(i for i, row in enumerate(archive['samples']) if row['stage'] == 'front_hold')
    initial = archive['samples'][index]
    trial = CycleTrial(setup=dict(APPROACH_SETUP, scene='straight'))
    sim, model, data = trial.sim, trial.model, trial.data
    data.qpos[:], data.qvel[:] = trace['qpos'][index], trace['qvel'][index]
    mujoco.mj_forward(model, data)
    mask, target = trace['masks'][index].copy(), trace['targets'][index].copy()
    front_count = int(mask.sum())
    options = module.GravityMomentOptions(r_h1=.094, r_h2=.098)
    gait = module.GravityMomentGait(18, options, .0645)
    seed = np.degrees(gait.targets(0., .094)[0])
    core = module.GravityCrossing([.0645]*19, [True]*18, seed, options,
                                  from_helix=True, helix_only=True)
    core.front_count, core.stage, core.started = front_count, 'front_hold', True
    core.phase = initial['reference_roll_rad']
    core._targets, core._enabled = target.copy(), mask.copy()
    measured = np.degrees(data.qpos[sim.drive.qadr])
    core.confirm(0., measured)
    core.running = True  # One Ctrl+G after the operator's physical-support confirmation.
    locked = target[:front_count].copy()
    last_control = last_feedback = -1.
    terminal = None
    collisions = []
    diagnostics = []
    front_target_error = max_passive_force = 0.
    root_x = float(np.mean(backbone(model, data)[:, 0]))

    def control(_):
        nonlocal last_control, last_feedback, measured, terminal, front_target_error, max_passive_force
        head = model.body('head').id
        for contact in data.contact:
            a, b = model.geom_bodyid[[contact.geom1, contact.geom2]]-head
            if 0 <= min(a, b) and max(a, b) <= 18 and abs(a-b) > 2 and contact.dist <= 0.:
                if not collisions:
                    collisions.append(dict(time_s=float(data.time), links=[int(a), int(b)], penetration_m=float(-contact.dist)))
        if data.time-last_control >= .02-1e-8:
            if data.time-last_feedback >= .2-1e-8:
                measured = np.degrees(data.qpos[sim.drive.qadr])
                last_feedback = float(data.time)
            core.advance(.02, float(data.time), last_feedback, measured)
            last_control = float(data.time)
            front_target_error = max(front_target_error, float(np.max(abs(core.targets[:front_count]-locked))))
            if core.stage == 'complete' and terminal is None:
                terminal = float(data.time)
            points = backbone(model, data)
            rho = np.linalg.norm(points[:front_count+2, 1:]-[0., PIPE_HEIGHT], axis=1)
            diagnostics.append(dict(time_s=float(data.time), stage=core.stage,
                offset_m=core.offset, accepted=core.accepted,
                active_end=int(core.enabled.sum()), front_radius_max_m=float(max(rho)),
                front_radius_min_m=float(min(rho)), **{k:v for k,v in core.status().items()
                    if k in ('tail_rate_m_s', 'tail_error_deg', 'tail_wait_reason')}))
        passive = ~core.enabled
        max_passive_force = max(max_passive_force, float(np.max(abs(data.actuator_force[passive]), initial=0.)))
        trial.phase = core.phase
        return core.targets

    def stop(_):
        return bool(collisions) or terminal is not None and data.time-terminal >= 3.

    trial.phase_run(1, lambda: core.stage, 120., control,
                    stiffness=lambda: core.enabled.astype(float), stop_when=stop)
    final = trial.samples[-1]
    report = dict(label=label, source=str(source), checkpoint_index=index,
        checkpoint_time_s=float(trace['time'][index]), completed_at_s=terminal,
        elapsed_s=float(data.time), max_front_target_change_rad=front_target_error,
        max_passive_force_nm=max_passive_force, collisions=collisions,
        front_radius_max_m=max(row['front_radius_max_m'] for row in diagnostics),
        max_ground_force_n=max(row['ground_force_n'] for row in trial.samples),
        min_front_contacts=min(len([j for j in row['main_links'] if j <= front_count]) for row in trial.samples),
        final_tail_radius_m=final['tail_max_radius_m'], final_tail_span_deg=final['tail_span_deg'],
        final_contacts=final['main_links'], axial_center_change_m=final['center_x_m']-root_x,
        saturated_sample_count=sum(bool(row['saturated_joints']) for row in trial.samples),
        final_core_stage=core.stage)
    report['passed'] = bool(terminal is not None and not collisions and front_target_error < 1e-12
        and max_passive_force < 1e-12 and report['max_ground_force_n'] < 1.
        and report['min_front_contacts'] >= 2
        and report['front_radius_max_m'] <= .13 and report['final_tail_radius_m'] <= .14
        and report['final_tail_span_deg'] >= 75. and final['tail_wrapped'])
    with contextlib.redirect_stdout(io.StringIO()):
        save(trial, output, label, report['checkpoint_time_s'], comparison=report,
             parameters=asdict(sim.p), start_x=.2)
    (output/(label+'_control.json')).write_text(json.dumps(diagnostics, indent=1), encoding='utf-8')
    (output/(label+'_review.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT/'artifacts/sequential_handoff/round18/basic5')
    parser.add_argument('--candidate-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = ((candidate, 'progressive'),) if args.candidate_only else ((baseline_module(), 'baseline'), (candidate, 'progressive'))
    for module, label in cases:
        run(module, args.output, label, args.source)


if __name__ == '__main__':
    main()
