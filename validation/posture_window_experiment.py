"""Two posture-window main cases and one documented same-case repair."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from dataclasses import asdict
from datetime import datetime
import gzip
import json
import math
import numpy as np
from snake_control.simulation import ROOT, backbone
from snake_control.adaptive_arc_length import HalfArcParameters
from snake_control.motor_shape_observer import PhaseShapeReference
from snake_control.posture_simulation import PostureSimulation
from snake_control.posture_window import mechanically_feasible
from validation.motor_feedback_experiment import snapshot

REPORT = ROOT/'validation/reports/posture_window'


def references():
    return {w: PhaseShapeReference.from_dict(json.loads((ROOT/f'configs/{name}').read_text(encoding='utf8')))
            for w, name in ((6, 'motor_shape_reference.json'), (8, 'motor_shape_reference8.json'))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=['straight', 'acute'])
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    records = list(REPORT.glob('*/status.json'))
    if len(records) >= 3:
        raise RuntimeError('Three dynamics attempts exhausted; review the retained failures')
    if args.repair and not (REPORT/'repair_plan.md').exists():
        raise RuntimeError('Write the concrete failure and repair before retrying')
    if not args.repair and any(json.loads(p.read_text())['case'] == args.case for p in records):
        raise RuntimeError('The main case has already run')
    if args.case == 'acute' and not args.repair and not (REPORT/'straight_review.json').exists():
        raise RuntimeError('Review the straight main case first')
    acute = args.case == 'acute'
    p = HalfArcParameters(first_joint=8 if acute else 6, last_joint=13 if acute else 11)
    scene = 'elbow90_taper110_210' if acute else 'straight'
    start_x = -.25 if acute else -.9
    sim = PostureSimulation(p, references(), scene, start_x)
    folder = REPORT/(args.case+('_repair' if args.repair else '')+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    duration = 100. if acute else 82.
    count, error, exit_phase = 0, None, None
    termination = 'duration_limit'
    stats = dict(physical_steps=0, passive_torque_max_nm=0., active_torque_max_nm=0.,
                 joint_limit_excess_max_rad=0., saturation_joint_steps=0, mask_violation_steps=0,
                 floor_contact_steps=0, maximum_external_force_sum_n=0., finite=True)
    limits = sim.model.jnt_range[sim.model.actuator_trnid[:, 0]]
    floor = sim.model.geom('floor').id
    try:
        with gzip.open(folder/'actual_trace.jsonl.gz', 'wt', encoding='utf8') as stream:
            stream.write(json.dumps(snapshot(sim))+'\n'); count += 1
            for frame in range(round(duration/.02)):
                for _ in range(20):
                    sim.data.xfrc_applied[:] = 0.
                    sim.step()
                    q, tau = sim.data.qpos[sim.drive.qadr], sim.data.actuator_force
                    stats['physical_steps'] += 1
                    stats['passive_torque_max_nm'] = max(stats['passive_torque_max_nm'], float(abs(tau[~sim.enabled]).max()))
                    stats['active_torque_max_nm'] = max(stats['active_torque_max_nm'], float(abs(tau).max()))
                    stats['mask_violation_steps'] += int(not mechanically_feasible(sim.lengths, sim.controller.window) or np.any(np.diff(np.flatnonzero(sim.enabled)) != 1))
                    stats['saturation_joint_steps'] += int(np.count_nonzero(abs(tau[sim.enabled]) >= p.torque_limit_nm-1e-6))
                    stats['joint_limit_excess_max_rad'] = max(stats['joint_limit_excess_max_rad'],
                        float(np.maximum(np.maximum(limits[:, 0]-q, q-limits[:, 1]), 0.).max()))
                    stats['finite'] &= bool(np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all())
                    stats['floor_contact_steps'] += int(any(floor in (int(c.geom1), int(c.geom2)) and
                        bool({int(c.geom1), int(c.geom2)} & sim.robot_geoms) for c in sim.data.contact))
                row = snapshot(sim)
                route = sim.route_coordinates(backbone(sim.model, sim.data))[0]
                row['whole_body_min_route_m'] = float(route.min())
                stream.write(json.dumps(row, ensure_ascii=False)+'\n'); count += 1
                controller = sim.controller
                changes = [e for e in controller.events if e['kind'] in ('complete', 'resize_complete')]
                last_phase = changes[-1]['phase_rad'] if changes else p.initial_phase_rad
                if not stats['finite'] or stats['floor_contact_steps'] or controller.fault or controller.resize_fault:
                    termination = 'nonfinite' if not stats['finite'] else 'floor_contact' if stats['floor_contact_steps'] else 'controller_fault'
                    break
                settled = controller.pending is None and controller.trial is None
                after_change = controller.phase-last_phase >= 2*math.pi
                if not acute and sum(e['kind'] == 'complete' for e in changes) >= 2 and after_change and settled:
                    termination = 'position_coverage_and_one_cycle'; break
                if acute:
                    if route.min() > sim.feature[-1]+.05:
                        if exit_phase is None: exit_phase = controller.phase
                    else:
                        exit_phase = None
                    if exit_phase is not None and controller.phase-exit_phase >= 2*math.pi and after_change and settled:
                        termination = 'whole_body_exit_and_one_cycle'; break
                if frame % 500 == 499:
                    print(f'{args.case}: {sim.data.time:.2f}s, window={controller.window}, '
                          f"selection={None if controller.selection is None else controller.selection['state']}, "
                          f'fault={controller.fault or controller.resize_fault}', flush=True)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        status = dict(status='complete' if error is None else 'interrupted', error=error, case=args.case, repair=args.repair,
            final_time_s=float(sim.data.time), samples=count, scene=scene, start_x_m=start_x,
            parameters=asdict(p), control_period_s=.02, motor_feedback_period_s=.2, statistics_1ms=stats,
            scene_geometry=dict(centerline_world_m_and_radius=sim.model.numeric('pipe_centerline').data.reshape(-1, 4).tolist(),
                                feature_span_m=sim.feature.tolist(), gravity=sim.model.opt.gravity.tolist()),
            passive_model=dict(damping=sim.model.dof_damping[sim.drive.vadr].tolist(),
                               armature=sim.model.dof_armature[sim.drive.vadr].tolist(), calibrated_to_hardware=False),
            stimulation=dict(kind='none', enters_controller=False), events=sim.controller.events,
            final_controller=sim.controller.snapshot(), hardware_run=False, termination_reason=termination,
            post_exit_cycles=None if exit_phase is None else (sim.controller.phase-exit_phase)/(2*math.pi),
            preferred_count=None, stable_width_whitelist=None, frozen_references={str(w): r.to_dict() for w, r in references().items()})
        (folder/'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf8')
        print('Actual-state archive:', folder, flush=True)


if __name__ == '__main__':
    main()
