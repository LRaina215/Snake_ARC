"""Actual dynamics acceptance and small mechanism controls; no imposed root path."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from dataclasses import asdict, replace
import json
import platform
import time
import numpy as np
import mujoco
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation, ROOT, backbone


CASES = [('straight', 'straight', 30., -.5, {}),
         ('bend90', 'bend90', 70., -.35, {}),
         ('elbow90', 'elbow90', 60., -.25, {}),
         ('hold', 'straight', 20., -.5, {'omega_rad_s': 0.}),
         ('reverse', 'straight', 20., -.5, {'direction': -1}),
         ('low_damping_bend90', 'bend90', 70., -.35, {'passive_damping': .05}),
         ('low_damping_elbow90', 'elbow90', 60., -.25, {'passive_damping': .05})]


def experiment(label, scene, duration, start_x, parameters, output):
    sim = ArcSimulation(scene, parameters, start_x)
    records, states, velocities, points = [], [], [], []
    initial = sim.measure()
    next_sample, work = 0., 0.
    passive_max, force_max, ground_max, joint_violation = 0., 0., 0., 0.
    warnings_before = np.array([w.number for w in sim.data.warning])
    wall = time.monotonic()
    while sim.data.time < duration:
        sim.step()
        passive_max = max(passive_max, np.max(np.abs(sim.data.actuator_force[~parameters.enabled]), initial=0))
        force_max = max(force_max, float(np.max(np.abs(sim.data.actuator_force))))
        work += np.maximum(sim.data.actuator_force * sim.data.qvel[sim.drive.vadr], 0).sum() * sim.model.opt.timestep
        q = sim.data.qpos[sim.drive.qadr]
        joint_violation = max(joint_violation, float(np.max(np.maximum(abs(q)-np.pi/2, 0))))
        if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all():
            raise RuntimeError('Nonfinite physical state')
        if sim.data.time >= next_sample:
            row = sim.measure()
            ground_max = max(ground_max, row['ground_contact_force_n'])
            records.append(row)
            states.append(sim.data.qpos.copy()); velocities.append(sim.data.qvel.copy())
            points.append(backbone(sim.model, sim.data))
            next_sample += .1
    final = records[-1]
    delta = final['progress_m'] - initial['progress_m']
    rolling = [r for r in records if r['time_s'] >= parameters.settle_s]
    max_unsupported, streak = 0., 0.
    for r in rolling:
        streak = streak + .1 if r['normal_force_n'] < .1 else 0.
        max_unsupported = max(max_unsupported, streak)
    checks = dict(passive_drive_zero=bool(passive_max < 1e-12),
                  torque_bounded=bool(force_max <= parameters.torque_limit_nm + 1e-9),
                  no_ground_support=bool(ground_max < .1),
                  support_retained=bool(max_unsupported < .5),
                  joint_limits=bool(joint_violation < .02),
                  no_new_solver_warnings=bool(np.all(np.array([w.number for w in sim.data.warning]) == warnings_before)))
    if label == 'hold':
        checks['stationary'] = bool(abs(delta) < .025)
    elif label == 'reverse':
        checks['reversed_progress'] = bool(delta < -.2)
    else:
        checks['forward_progress'] = bool(delta > .3)
        if scene != 'straight':
            checks['whole_body_beyond_bend'] = bool(final['min_body_progress_m'] > sim.feature[-1] + .05)
    summary = dict(label=label, scene=scene, passed=all(checks.values()), checks=checks,
                   simulation_duration_s=final['time_s'], wall_duration_s=time.monotonic()-wall,
                   parameters=asdict(parameters), initial_progress_m=initial['progress_m'],
                   final_progress_m=final['progress_m'], progress_delta_m=delta,
                   min_body_final_progress_m=final['min_body_progress_m'], feature_span_m=sim.feature.tolist(),
                   final_center_m=final['center_m'], passive_actuator_force_max_nm=float(passive_max),
                   max_ground_contact_force_n=ground_max, max_unsupported_s=max_unsupported,
                   maximum_torque_nm=force_max, positive_actuator_work_j=float(work),
                   note='Work is simulated positive mechanical actuator work, not electrical energy. Contact does not feed gait.')
    output.mkdir(parents=True, exist_ok=True)
    (output/f'{label}.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    fields = ['time_s','phase_rad','progress_m','min_body_progress_m','normal_force_n',
              'ground_contact_force_n','passive_actuator_force_max_nm','min_height_m']
    np.savez_compressed(output/f'{label}_trace.npz', qpos=np.array(states), qvel=np.array(velocities),
                        backbone=np.array(points), metrics=np.array([[r[k] for k in fields] for r in records]),
                        metric_fields=np.array(fields), route=sim.route,
                        active_mask=parameters.enabled, config=json.dumps(asdict(parameters)),
                        targets_rad=np.array([r['target_rad'] for r in records]),
                        actuator_force_nm=np.array([r['actuator_force_nm'] for r in records]))
    print(json.dumps(dict(case=label, passed=summary['passed'], delta_m=round(delta,4), checks=checks)), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=[c[0] for c in CASES])
    parser.add_argument('--output', type=Path, default=ROOT/'validation/reports/step1')
    args = parser.parse_args()
    base = ArcParameters.load(ROOT/'configs/step1.json')
    results = [experiment(label, scene, duration, start, replace(base, **overrides), args.output)
               for label, scene, duration, start, overrides in CASES if not args.cases or label in args.cases]
    report = dict(environment=dict(python=platform.python_version(), mujoco=mujoco.__version__,
                                  numpy=np.__version__, platform=platform.platform()),
                  passed=all(r['passed'] for r in results), cases=results)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
