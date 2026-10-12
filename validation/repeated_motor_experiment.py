"""Bounded repeated-window validation; no scripted window requests."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from dataclasses import asdict
from datetime import datetime
import gzip
import json
import numpy as np
import mujoco
from snake_control.rolling_arc import ArcParameters
from snake_control.motor_feedback_simulation import MotorFeedbackSimulation
from snake_control.simulation import ROOT
from validation.motor_feedback_experiment import snapshot

REPORT = ROOT/'validation/reports/repeated_motor_sliding'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['natural', 'return'], default='natural')
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    if len(list(REPORT.glob('*/status.json'))) >= 3:
        raise RuntimeError('Three planned attempts used; inspect evidence instead of another blind run')
    if args.case == 'return':
        review = json.loads((REPORT/'natural_review.json').read_text(encoding='utf8'))
        if not review['passed']:
            raise RuntimeError('Natural repeated handover has not passed review')
    output = REPORT/(args.case+('_repair' if args.repair else '')+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    p = ArcParameters(first_joint=6, last_joint=11)
    sim = MotorFeedbackSimulation(p, execute=True, start_x=-.9, feedback_period_s=.2)
    count, error, pull_sign, switch_count, last_switch_phase = 0, None, None, 0, None
    stats = dict(physical_steps=0, passive_torque_max_nm=0., active_torque_max_nm=0.,
        joint_limit_excess_max_rad=0., saturation_joint_steps=0, mask_violation_steps=0,
        floor_contact_steps=0, maximum_external_force_n=0., finite=True)
    limits = sim.model.jnt_range[sim.model.actuator_trnid[:, 0]]
    stopped_early = False
    try:
        with gzip.open(output/'actual_trace.jsonl.gz', 'wt', encoding='utf8') as stream:
            stream.write(json.dumps(snapshot(sim))+'\n'); count += 1
            for frame in range(4100):
                for _ in range(20):
                    sim.data.xfrc_applied[:] = 0.
                    if args.case == 'return' and 36. <= sim.data.time < 39.:
                        if args.repair:
                            # Same 20N sum of force magnitudes and same net
                            # vertical load as the failed diagonal stimulus.
                            # Remove its cancelling horizontal moment, apply
                            # the opposite vertical load at fixed module J10.
                            down, up = (20+20/np.sqrt(2))/2, (20-20/np.sqrt(2))/2
                            sim.data.xfrc_applied[sim.model.body('head').id, 2] = -down
                            bank = sim.model.jnt_bodyid[sim.model.joint('joint10').id]
                            sim.data.xfrc_applied[bank, 2] = up
                        else:
                            if pull_sign is None:
                                pull_sign = 1 if sim.data.body('head').xpos[1] >= 0 else -1
                            sim.data.xfrc_applied[sim.model.body('head').id, :3] = np.array([0., pull_sign, -1.])*20/np.sqrt(2)
                    sim.step()
                    q, tau = sim.data.qpos[sim.drive.qadr], sim.data.actuator_force
                    stats['physical_steps'] += 1
                    stats['mask_violation_steps'] += int(sim.enabled.sum() != 6)
                    stats['passive_torque_max_nm'] = max(stats['passive_torque_max_nm'], float(abs(tau[~sim.enabled]).max()))
                    stats['active_torque_max_nm'] = max(stats['active_torque_max_nm'], float(abs(tau[sim.enabled]).max()))
                    stats['saturation_joint_steps'] += int(np.count_nonzero(abs(tau[sim.enabled]) >= p.torque_limit_nm-1e-6))
                    stats['joint_limit_excess_max_rad'] = max(stats['joint_limit_excess_max_rad'],
                        float(np.maximum(np.maximum(limits[:, 0]-q, q-limits[:, 1]), 0.).max()))
                    stats['finite'] &= bool(np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all())
                    floor = sim.model.geom('floor').id
                    stats['floor_contact_steps'] += int(any(floor in (int(c.geom1), int(c.geom2)) and
                        bool({int(c.geom1), int(c.geom2)} & sim.robot_geoms) for c in sim.data.contact))
                    stats['maximum_external_force_n'] = max(stats['maximum_external_force_n'],
                        float(np.linalg.norm(sim.data.xfrc_applied[:, :3], axis=1).max()))
                row = snapshot(sim)
                row['stimulus'] = 'head_pull' if np.any(sim.data.xfrc_applied) else 'unforced'
                stream.write(json.dumps(row, ensure_ascii=False)+'\n'); count += 1
                completed = [e for e in sim.controller.events if e['kind'] == 'complete']
                if len(completed) > switch_count:
                    switch_count, last_switch_phase = len(completed), sim.controller.phase
                visited = [tuple((6, 11))]+[tuple(e['to']) for e in completed]
                returned = len(visited) >= 3 and visited[-1] in visited[:-1] and any(
                    completed[j]['to'][0] > completed[j]['old'][0] for j in range(len(completed)))
                fulfilled = switch_count >= 2 if args.case == 'natural' else returned
                if fulfilled and sim.controller.phase-last_switch_phase >= 4*np.pi and (
                        args.case == 'natural' or sim.data.time >= 39.):
                    stopped_early = True
                    break
                if frame % 500 == 499:
                    print(f"{args.case}: {sim.data.time:.2f}s, window={sim.controller.window}, switches={switch_count}, "
                          f"observer={None if sim.controller.observation is None else sim.controller.observation['state']}", flush=True)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        geom = sim.model.geom('pipe_straight')
        status = dict(status='complete' if error is None else 'interrupted', error=error,
            samples=count, final_time_s=float(sim.data.time), case=args.case, stopped_after_coverage=stopped_early,
            parameters=asdict(p), start_x_m=-.9, scene='straight', closed_loop=True,
            control_period_s=.02, motor_feedback_period_s=.2, statistics_1ms=stats,
            scene_geometry=dict(pipe_radius_m=float(geom.size[0]), pipe_half_length_m=float(geom.size[1]),
                pipe_position_world=sim.data.geom('pipe_straight').xpos.tolist(),
                pipe_axis_world=sim.data.geom('pipe_straight').xmat.reshape(3, 3)[:, 2].tolist(),
                pipe_friction=geom.friction.tolist(), gravity=sim.model.opt.gravity.tolist()),
            passive_model=dict(damping=sim.model.dof_damping[sim.drive.vadr].tolist(),
                armature=sim.model.dof_armature[sim.drive.vadr].tolist(), calibrated_to_hardware=False),
            stimulation=dict(kind='none' if args.case == 'natural' else
                'head down plus fixed J10 up load difference' if args.repair else 'head outward/down pulse',
                interval_s=[36., 39.], total_force_n=0. if args.case == 'natural' else 20., sign=pull_sign,
                head_down_n=(20+20/np.sqrt(2))/2 if args.repair else None,
                j10_up_n=(20-20/np.sqrt(2))/2 if args.repair else None,
                enters_controller=False), events=sim.controller.events,
            motor_observer=sim.controller.observer.configuration())
        (output/'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf8')
        print('Actual-state archive:', output, flush=True)


if __name__ == '__main__':
    main()
