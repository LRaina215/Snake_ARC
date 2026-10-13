"""Two bounded dynamics cases for measured half-circle window control."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
import gzip
import json
import math
import numpy as np
import mujoco
from snake_control.rolling_arc import ArcParameters
from snake_control.posture_simulation import PostureSimulation
from snake_control.posture_window import measured_half_circle
from snake_control.section_geometry import local_section
from snake_control.simulation import backbone
from validation.posture_window_experiment import references
from validation.motor_feedback_experiment import snapshot

REPORT = ROOT/'validation/reports/half_circle_window'


@dataclass(frozen=True)
class MeasuredParameters(ArcParameters):
    mechanical_lengths_m: tuple = (.0645,)*21
    manual_tuning: bool = True  # Same policy as the default manual/real entry.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=['fixed', 'return'])
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    if len(list(REPORT.glob('*/status.json'))) >= 3:
        raise RuntimeError('Two main runs plus one targeted repair exhausted; review before further experiments')
    if args.repair and not (REPORT/'repair_plan.md').exists():
        raise RuntimeError('Document the observed failure and repair first')
    if not args.repair and list(REPORT.glob(args.case+'_*/status.json')):
        raise RuntimeError('This main case already ran')
    scene = 'straight' if args.case == 'fixed' else 'length_taper_return'
    start = -.5 if args.case == 'fixed' else -.35
    p = MeasuredParameters(amplitude_deg=45. if args.case == 'fixed' else 60.)
    sim = PostureSimulation(p, references(), scene, start)
    sim.controller.automatic = args.case != 'fixed'
    folder = REPORT/(args.case+('_repair' if args.repair else '')+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder.mkdir(parents=True)
    duration = 25. if args.case == 'fixed' else 160.
    stats = dict(steps=0, passive_torque_max_nm=0., active_torque_max_nm=0.,
                 joint_limit_excess_max_rad=0., saturation_joint_steps=0, floor_contact_steps=0)
    limits = sim.model.jnt_range[sim.model.actuator_trnid[:, 0]]
    floor = sim.model.geom('floor').id
    reason, error, exit_phase = 'duration', None, None
    evaluations = []
    try:
        with gzip.open(folder/'actual_trace.jsonl.gz', 'wt', encoding='utf8') as f:
            for frame in range(round(duration/.02)):
                for _ in range(20):
                    sim.step()
                    q, tau = sim.data.qpos[sim.drive.qadr], sim.data.actuator_force
                    stats['steps'] += 1
                    stats['passive_torque_max_nm'] = max(stats['passive_torque_max_nm'], float(np.max(abs(tau[~sim.enabled]), initial=0.)))
                    stats['active_torque_max_nm'] = max(stats['active_torque_max_nm'], float(np.max(abs(tau))))
                    stats['saturation_joint_steps'] += int(np.count_nonzero(abs(tau) >= p.torque_limit_nm-1e-6))
                    stats['joint_limit_excess_max_rad'] = max(stats['joint_limit_excess_max_rad'], float(np.maximum(np.maximum(limits[:, 0]-q, q-limits[:, 1]), 0.).max()))
                    stats['floor_contact_steps'] += int(any(floor in (int(c.geom1), int(c.geom2)) and
                        bool({int(c.geom1), int(c.geom2)} & sim.robot_geoms) for c in sim.data.contact))
                row = snapshot(sim)
                points = backbone(sim.model, sim.data)
                first, last = sim.controller.window
                arc = points[first:last+2]
                polar = np.unwrap(np.arctan2(arc[:, 2]-1.2, arc[:, 1]))
                truth_wrap = math.degrees(abs(polar[-1]-polar[0]))
                geometry = measured_half_circle(local_section(q, sim.lengths, first, last))
                record = dict(time_s=float(sim.data.time), phase_rad=sim.controller.phase, window=[first, last],
                    whole_body_min_x_m=float(points[:, 0].min()), body_mean_x_m=float(points[:, 0].mean()),
                    true_pipe_wrap_deg=truth_wrap, support_up_n=sum(row['support_up_n']),
                    radius_m=geometry['radius_m'], coverage_deg=geometry['coverage_deg'],
                    circle_rms_m=geometry['circle_rms_m'],
                    actual_turn_deg=geometry['turn_deg'], min_z_m=float(points[:, 2].min()),
                    inverse=sim.controller.core.inverse_diagnostic)
                evaluations.append(record)
                row['evaluation'] = record
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
                if not np.isfinite(sim.data.qpos).all() or stats['floor_contact_steps'] or sim.controller.fault or sim.controller.resize_fault:
                    reason = 'floor_or_controller_fault'; break
                if args.case == 'return':
                    if points[:, 0].min() > .95:
                        if exit_phase is None: exit_phase = sim.controller.phase
                    else: exit_phase = None
                    if exit_phase is not None and sim.controller.phase-exit_phase >= 2*math.pi:
                        reason = 'whole_body_exit_and_one_cycle'; break
                if frame % 500 == 499:
                    print(f"{args.case} {sim.data.time:.1f}s {sim.controller.window} x={record['body_mean_x_m']:.3f} "
                          f"coverage={record['coverage_deg']:.1f} wrap={truth_wrap:.1f} R={geometry['radius_m']:.3f}", flush=True)
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        (folder/'evaluation.json').write_text(json.dumps(evaluations), encoding='utf8')
        status = dict(case=args.case, repair=args.repair, scene=scene, start_x_m=start,
            parameters=asdict(p), statistics=stats, termination=reason, error=error,
            final_time_s=float(sim.data.time), events=sim.controller.events, final_controller=sim.controller.snapshot(),
            centerline_world_m_and_radius=sim.model.numeric('pipe_centerline').data.reshape(-1, 4).tolist(),
            gravity=sim.model.opt.gravity.tolist(), passive_damping=p.passive_damping,
            control_period_s=.02, feedback_period_s=.2, physics_step_s=sim.model.opt.timestep,
            hardware_run=False, nominal_restoring=False)
        (folder/'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf8')
        print(folder, reason, error, flush=True)


if __name__ == '__main__':
    main()
