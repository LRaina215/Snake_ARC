"""Four planned dynamics cases, at most one documented repair; no sweep."""
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
from snake_control.adaptive_arc_length import HalfArcParameters
from snake_control.length_simulation import LengthSimulation
from snake_control.motor_shape_observer import PhaseShapeReference
from snake_control.simulation import ROOT, backbone
from validation.motor_feedback_experiment import snapshot

REPORT = ROOT/'validation/reports/adaptive_arc_length'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=['calibration', 'baseline', 'adaptive', 'combined'])
    parser.add_argument('--repair', action='store_true')
    parser.add_argument('--radius-only', action='store_true', help='Historical failed passive-radius selector, for explicit comparison only')
    args = parser.parse_args()
    if len(list(REPORT.glob('*/status.json'))) >= 5:
        raise RuntimeError('Five-attempt budget exhausted; review instead of more dynamics')
    if not json.loads((REPORT/'preflight.json').read_text())['passed']:
        raise RuntimeError('Geometry preflight has not passed')
    refs = {6: PhaseShapeReference.from_dict(json.loads((ROOT/'configs/motor_shape_reference.json').read_text()))}
    if args.case in ('adaptive', 'combined'):
        refs[8] = PhaseShapeReference.from_dict(json.loads((ROOT/'configs/motor_shape_reference8.json').read_text()))
    if args.case == 'combined' and not json.loads((REPORT/'adaptive_review.json').read_text())['resize_pass']:
        raise RuntimeError('Do not combine with migration before the resizing case passes')
    if args.repair and not (REPORT/'repair_plan.md').exists():
        raise RuntimeError('Write the specific repair cause and hypothesis first')
    width8 = args.case == 'calibration'
    p = HalfArcParameters(first_joint=7 if width8 else 8, last_joint=14 if width8 else 13)
    scene = 'length_calibration210' if width8 else 'elbow90_taper110_210' if args.case == 'combined' else 'length_taper_return'
    start = -.25 if args.case == 'combined' else -.5
    sim = LengthSimulation(p, refs, scene, start, resize=args.case in ('adaptive', 'combined'), sliding=args.case == 'combined',
                          feasibility_trial=args.case in ('adaptive', 'combined') and not args.radius_only)
    folder = REPORT/(args.case+('_repair' if args.repair else '')+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    duration = 35. if width8 else 150. if args.case == 'combined' else 180.
    count, error, exit_phase = 0, None, None
    change_count, last_change_phase = 0, 0.
    stats = dict(physical_steps=0, passive_torque_max_nm=0., active_torque_max_nm=0.,
        joint_limit_excess_max_rad=0., saturation_joint_steps=0, mask_violation_steps=0,
        floor_contact_steps=0, finite=True)
    limits = sim.model.jnt_range[sim.model.actuator_trnid[:, 0]]
    floor = sim.model.geom('floor').id
    try:
        with gzip.open(folder/'actual_trace.jsonl.gz', 'wt', encoding='utf8') as stream:
            stream.write(json.dumps(snapshot(sim))+'\n'); count += 1
            for frame in range(round(duration/.02)):
                for _ in range(20):
                    sim.step()
                    q, tau = sim.data.qpos[sim.drive.qadr], sim.data.actuator_force
                    stats['physical_steps'] += 1
                    stats['passive_torque_max_nm'] = max(stats['passive_torque_max_nm'], float(abs(tau[~sim.enabled]).max()))
                    stats['active_torque_max_nm'] = max(stats['active_torque_max_nm'], float(abs(tau).max()))
                    stats['mask_violation_steps'] += int(sim.enabled.sum() not in (6, 7, 8) or np.any(np.diff(np.flatnonzero(sim.enabled)) != 1))
                    stats['saturation_joint_steps'] += int(np.count_nonzero(abs(tau[sim.enabled]) >= p.torque_limit_nm-1e-6))
                    stats['joint_limit_excess_max_rad'] = max(stats['joint_limit_excess_max_rad'],
                        float(np.maximum(np.maximum(limits[:, 0]-q, q-limits[:, 1]), 0.).max()))
                    stats['finite'] &= bool(np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all())
                    stats['floor_contact_steps'] += int(any(floor in (int(c.geom1), int(c.geom2)) and
                        bool({int(c.geom1), int(c.geom2)} & sim.robot_geoms) for c in sim.data.contact))
                row = snapshot(sim)
                progress = sim.route_coordinates(backbone(sim.model, sim.data))[0]
                row['whole_body_min_route_m'] = float(progress.min())
                row['evaluation_radius_m'] = float(np.interp(progress.mean(), sim.route_s,
                    sim.model.numeric('pipe_centerline').data.reshape(-1, 4)[:, 3]))
                stream.write(json.dumps(row, ensure_ascii=False)+'\n'); count += 1
                changes = [e for e in sim.controller.events if e['kind'] in ('resize_complete', 'complete')]
                if len(changes) > change_count:
                    change_count, last_change_phase = len(changes), sim.controller.phase
                if not width8 and progress.min() > sim.feature[-1]+.05:
                    if exit_phase is None:
                        exit_phase = sim.controller.phase
                    if sim.controller.phase-exit_phase >= 4*np.pi and sim.controller.phase-last_change_phase >= 4*np.pi and sim.controller.pending is None:
                        break
                elif not width8:
                    exit_phase = None  # Re-entry starts continuous exit observation again.
                if frame % 500 == 499:
                    c = sim.controller
                    print(f"{args.case}: {sim.data.time:.1f}s, window={c.window}, pending={None if c.pending is None else c.pending['kind']}, "
                          f"length={c.length_observation}, fault={c.resize_fault or c.fault}", flush=True)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        points = sim.model.numeric('pipe_centerline').data.reshape(-1, 4)
        status = dict(status='complete' if error is None else 'interrupted', error=error, case=args.case, repair=args.repair,
            final_time_s=float(sim.data.time), samples=count, scene=scene, start_x_m=start, parameters=asdict(p),
            control_period_s=.02, motor_feedback_period_s=.2, statistics_1ms=stats,
            scene_geometry=dict(centerline_world_m_and_radius=points.tolist(), feature_span_m=sim.feature.tolist(),
                gravity=sim.model.opt.gravity.tolist(), mesh_cross_section_vertices=48),
            controller_options=dict(length=asdict(sim.controller.length_options), handover=asdict(sim.controller.options)),
            events=sim.controller.events, exit_phase_rad=exit_phase,
            post_exit_cycles=None if exit_phase is None else (sim.controller.phase-exit_phase)/(2*np.pi),
            final_controller=sim.controller.snapshot(), hardware_run=False)
        status['driver_revision'] = 'continuous_exit_and_final_handover_phase'
        (folder/'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf8')
        print('Actual-state archive:', folder, flush=True)


if __name__ == '__main__':
    main()
