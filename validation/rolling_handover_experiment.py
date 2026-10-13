"""Three planned moving-exchange cases and at most one documented repair."""
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
from snake_control.rolling_simulation import RollingSimulation
from validation.motor_feedback_experiment import snapshot

REPORT = ROOT/'validation/reports/rolling_handover'


def pulse(time_s, begin, end, ramp=2.):
    if not begin <= time_s < end:
        return 0.
    fraction = min(1., (time_s-begin)/ramp, (end-time_s)/ramp)
    return .5-.5*math.cos(math.pi*fraction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=['natural', 'short_return', 'combined'])
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    if len(list(REPORT.glob('*/status.json'))) >= 5:
        raise RuntimeError('Five-attempt amended budget exhausted; review instead of another run')
    if args.repair and not (REPORT/'repair_plan.md').exists():
        raise RuntimeError('Specific repair hypothesis must be written first')
    if args.case != 'natural' and not json.loads((REPORT/'natural_review.json').read_text())['moving_exchange_pass']:
        raise RuntimeError('Review the moving six-joint experiment first')
    refs = {w: PhaseShapeReference.from_dict(json.loads((ROOT/f'configs/motor_shape_reference{suffix}.json').read_text()))
            for w, suffix in ((6, ''), (8, '8'))}
    short, combined = args.case == 'short_return', args.case == 'combined'
    p = HalfArcParameters(first_joint=3 if short else 8 if combined else 6,
                          last_joint=8 if short else 13 if combined else 11)
    scene = 'rolling_straight_long' if short else 'elbow90_taper110_210' if combined else 'straight'
    sim = RollingSimulation(p, refs, scene, -.25 if combined else -.9, resize=combined)
    folder = REPORT/(args.case+('_repair' if args.repair else '')+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    duration = 150. if combined else 120. if short else 82.
    count, error, change_count, last_phase, exit_phase = 0, None, 0, None, None
    termination_reason = 'duration_limit'
    stats = dict(physical_steps=0, passive_torque_max_nm=0., active_torque_max_nm=0.,
        joint_limit_excess_max_rad=0., saturation_joint_steps=0, mask_violation_steps=0,
        floor_contact_steps=0, maximum_external_force_sum_n=0., finite=True)
    limits = sim.model.jnt_range[sim.model.actuator_trnid[:, 0]]
    floor, head, tail = sim.model.geom('floor').id, sim.model.body('head').id, sim.model.body('tail').id
    try:
        with gzip.open(folder/'actual_trace.jsonl.gz', 'wt', encoding='utf8') as stream:
            stream.write(json.dumps(snapshot(sim))+'\n'); count += 1
            for frame in range(round(duration/.02)):
                for _ in range(20):
                    sim.data.xfrc_applied[:] = 0.
                    if short:
                        load = 30.*(pulse(sim.data.time, 24., 36.)-pulse(sim.data.time, 60., 72.))
                        sim.data.xfrc_applied[head, 2], sim.data.xfrc_applied[tail, 2] = -load, load
                    sim.step()
                    q, tau = sim.data.qpos[sim.drive.qadr], sim.data.actuator_force
                    stats['physical_steps'] += 1
                    stats['passive_torque_max_nm'] = max(stats['passive_torque_max_nm'], float(abs(tau[~sim.enabled]).max()))
                    stats['active_torque_max_nm'] = max(stats['active_torque_max_nm'], float(abs(tau).max()))
                    stats['mask_violation_steps'] += int(sim.enabled.sum() not in ((6, 7, 8) if combined else (6,)) or np.any(np.diff(np.flatnonzero(sim.enabled)) != 1))
                    stats['saturation_joint_steps'] += int(np.count_nonzero(abs(tau[sim.enabled]) >= p.torque_limit_nm-1e-6))
                    stats['joint_limit_excess_max_rad'] = max(stats['joint_limit_excess_max_rad'],
                        float(np.maximum(np.maximum(limits[:, 0]-q, q-limits[:, 1]), 0.).max()))
                    stats['finite'] &= bool(np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all())
                    stats['floor_contact_steps'] += int(any(floor in (int(c.geom1), int(c.geom2)) and
                        bool({int(c.geom1), int(c.geom2)} & sim.robot_geoms) for c in sim.data.contact))
                    stats['maximum_external_force_sum_n'] = max(stats['maximum_external_force_sum_n'],
                        float(np.linalg.norm(sim.data.xfrc_applied[:, :3], axis=1).sum()))
                row = snapshot(sim)
                route = sim.route_coordinates(backbone(sim.model, sim.data))[0]
                row['whole_body_min_route_m'] = float(route.min())
                stream.write(json.dumps(row, ensure_ascii=False)+'\n'); count += 1
                changes = [e for e in sim.controller.events if e['kind'] in ('complete', 'resize_complete')]
                if len(changes) > change_count:
                    change_count, last_phase = len(changes), sim.controller.phase
                slides = [e for e in changes if e['kind'] == 'complete']
                if stats['floor_contact_steps'] or sim.controller.fault or sim.controller.resize_fault:
                    termination_reason = 'floor_contact' if stats['floor_contact_steps'] else 'controller_fault'
                    break
                if args.case == 'natural' and len(slides) >= 2 and sim.controller.phase-last_phase >= 2*np.pi and sim.controller.pending is None:
                    termination_reason = 'coverage_and_one_cycle'
                    break
                if combined:
                    if route.min() > sim.feature[-1]+.05:
                        if exit_phase is None: exit_phase = sim.controller.phase
                    else:
                        exit_phase = None
                    if (exit_phase is not None and last_phase is not None and sim.controller.pending is None and
                            sim.controller.phase-exit_phase >= 2*np.pi and sim.controller.phase-last_phase >= 2*np.pi and
                            sim.controller.six_trial is None and
                            any(e['kind'] == 'resize_complete' and len(range(e['to'][0], e['to'][1]+1)) == 6 for e in changes)):
                        termination_reason = 'continuous_exit_and_one_cycle'
                        break
                if frame % 500 == 499:
                    c = sim.controller
                    print(f"{args.case}: {sim.data.time:.2f}s, window={c.window}, transfers={len(slides)}, "
                          f"observer={None if c.observation is None else c.observation['state']}, fault={c.fault or c.resize_fault}", flush=True)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        status = dict(status='complete' if error is None else 'interrupted', error=error, case=args.case, repair=args.repair,
            final_time_s=float(sim.data.time), samples=count, scene=scene, start_x_m=-.25 if combined else -.9,
            parameters=asdict(p), control_period_s=.02, motor_feedback_period_s=.2, statistics_1ms=stats,
            scene_geometry=dict(centerline_world_m_and_radius=sim.model.numeric('pipe_centerline').data.reshape(-1, 4).tolist(),
                                feature_span_m=sim.feature.tolist(), gravity=sim.model.opt.gravity.tolist()),
            stimulation=dict(kind='opposite endpoint vertical load difference' if short else 'none',
                interval_s=[24., 72.], pulses_s=[[24., 36.], [60., 72.]], ramp_s=2., each_endpoint_n=30. if short else 0.,
                net_force_n=0., enters_controller=False),
            events=sim.controller.events, final_controller=sim.controller.snapshot(), hardware_run=False,
            termination_reason=termination_reason,
            post_exit_cycles=None if exit_phase is None else (sim.controller.phase-exit_phase)/(2*np.pi),
            controller_options=dict(length=asdict(sim.controller.length_options), handover=asdict(sim.controller.options)),
            driver_revision='moving_phase_continuous_exit_one_cycle')
        (folder/'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf8')
        print('Actual-state archive:', folder, flush=True)


if __name__ == '__main__':
    main()
