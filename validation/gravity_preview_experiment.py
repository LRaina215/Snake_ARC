"""One frozen 25s display validation; control stays v3, 4N pull during 10..12s."""
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
from snake_control.posture_simulation import PostureSimulation
from snake_control.posture_window import mechanically_feasible
from validation.motor_feedback_experiment import snapshot
from validation.posture_window_experiment import references

REPORT = ROOT/'validation/reports/gravity_preview/physics'


def main():
    from types import SimpleNamespace
    args = SimpleNamespace(case='display_pull', repair=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    if list(REPORT.glob('*/status.json')):
        raise RuntimeError('The one planned dynamics case already ran; inspect its evidence')
    scene, start_x, duration = 'straight', -.5, 25.
    p = HalfArcParameters(first_joint=8, last_joint=13)
    sim = PostureSimulation(p, references(), scene, start_x)
    folder = REPORT/(args.case+('_repair' if args.repair else '')+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
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
                    if 10. <= sim.data.time < 12.:
                        # Pull the physical head outward/downward; no root/angle reset.
                        sign = 1. if sim.data.body('head').xpos[1] >= 0 else -1.
                        sim.data.xfrc_applied[sim.model.body('head').id, :3] = np.array([0., sign, -1.])*4./math.sqrt(2.)
                    stats['maximum_external_force_sum_n'] = max(stats['maximum_external_force_sum_n'],
                        float(np.linalg.norm(sim.data.xfrc_applied[:, :3], axis=1).sum()))
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
                if frame % 500 == 499:
                    count_o = controller.count_observation or {}
                    phase_s = count_o.get('phase_statistics', {})
                    print(f"{args.case}: {sim.data.time:.2f}s, window={controller.window}, "
                          f"balanced_count={phase_s.get('balanced_count_joints')}, "
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
            stimulation=dict(kind='head_outward_downward_4N', interval_s=[10.,12.], enters_controller=False), events=sim.controller.events,
            final_controller=sim.controller.snapshot(), hardware_run=False, termination_reason=termination,
            post_exit_cycles=None if exit_phase is None else (sim.controller.phase-exit_phase)/(2*math.pi),
            preferred_count=None, stable_width_whitelist=None, frozen_references={str(w): r.to_dict() for w, r in references().items()})
        (folder/'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf8')
        print('Actual-state archive:', folder, flush=True)


if __name__ == '__main__':
    main()
