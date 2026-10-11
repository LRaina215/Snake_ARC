"""One declared experiment; actual variable-window trace and separate verdicts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import gzip
import time
from dataclasses import asdict
import numpy as np
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation, backbone, ROOT
from snake_control.fixed_section_simulation import FixedSectionSimulation
from snake_control.fixed_section_compliance import SectionOptions
from snake_control.sliding_section_simulation import SlidingSectionSimulation, upward_support
from snake_control.sliding_section_compliance import open_geometry
from snake_control.arc_shape import ArcShapeEvaluator


def run(scene, label, output, duration=62., mode='sliding', start=-.5,
        manual_direction=0, initial_window=None, payload=False, exit_cycles=0):
    p = ArcParameters.load(ROOT/'configs/step1.json')
    options = SectionOptions()
    if mode == 'baseline':
        sim = ArcSimulation(scene, p, start)
    elif mode == 'fixed':
        sim = FixedSectionSimulation(scene, p, start, options)
    else:
        sim = SlidingSectionSimulation(scene, p, start, options, automatic=not manual_direction,
                                       initial_window=initial_window)
    lengths = np.linalg.norm(np.diff(backbone(sim.model, sim.data), axis=0), axis=1)
    evaluator = ArcShapeEvaluator(sim.route)
    records, states, velocities, bones = [], [], [], []
    warnings = np.array([w.number for w in sim.data.warning])
    work = passive_max = torque_max = limit_error = 0.
    saturated_s = 0.
    wall = time.monotonic()
    requested = set()
    manual_requests = []
    exit_time = None
    exit_phase = None
    next_sample = 0.
    period = 2*np.pi/p.omega_rad_s
    pipe_radii = sim.model.numeric('pipe_centerline').data.reshape(-1, 4)[:, 3]

    def record():
        row = sim.measure()
        pts = backbone(sim.model, sim.data)
        q = sim.data.qpos[sim.drive.qadr]
        geometry = row.get('geometry', {})
        bounds = geometry.get('window')
        if bounds is None:
            active = np.flatnonzero(getattr(sim, 'enabled', p.enabled))
            bounds = [int(active[0])+1, int(active[-1])+1] if len(active) else [p.first_joint, p.last_joint]
        shape = evaluator.measure(pts, np.degrees(q), *bounds)
        body = open_geometry(q, lengths, bounds)
        sent = open_geometry(np.asarray(row['target_rad']), lengths, bounds)
        row.update(shape=shape, body_geometry=body, sent_geometry=sent, window=bounds,
                   enabled=np.asarray(getattr(sim, 'enabled', p.enabled)).tolist(),
                   progress_m=float(sim.route_coordinates(pts.mean(axis=0))[0][0]),
                   weights=np.asarray(getattr(sim, 'weights', p.enabled.astype(float))).tolist(),
                   support_up_n=upward_support(sim).tolist(), geometry=geometry,
                   payload_n=float(-sim.data.xfrc_applied[sim.model.body('link14').id, 2]))
        local = pts[bounds[0]:bounds[1]+2]
        row['local_pipe_radius_m'] = float(np.interp(
            sim.route_coordinates(local.mean(axis=0))[0][0], sim.route_s, pipe_radii))
        records.append(row)
        states.append(sim.data.qpos.copy())
        velocities.append(sim.data.qvel.copy())
        bones.append(pts)

    record()
    initial_progress = records[0]['progress_m']
    while sim.data.time < duration:
        now = float(sim.data.time)
        if mode == 'sliding':
            sim.paused = bool(manual_direction and 8 <= now < 14)
            completed_now = [e for e in sim.controller.events if e['kind'] == 'complete']
            return_time = max(24., completed_now[0]['time_s']+6) if completed_now else float('inf')
            for when, direction in ((9., manual_direction), (return_time, -manual_direction)):
                key = 0 if direction == manual_direction else 1
                if manual_direction and now >= when and key not in requested:
                    requested.add(key)
                    accepted = sim.request_neighbour(direction)
                    manual_requests.append(dict(time_s=now, direction=direction, accepted=accepted))
        if payload:
            magnitude = 4.*min(max(now-14, 0.), max(24-now, 0.), 1.) if 14 <= now <= 24 else 0.
            sim.data.xfrc_applied[sim.model.body('link14').id, 2] = -magnitude
        sim.step()
        enabled = np.asarray(getattr(sim, 'enabled', p.enabled))
        force = sim.data.actuator_force
        passive_max = max(passive_max, float(np.max(abs(force[~enabled]), initial=0)))
        torque_max = max(torque_max, float(np.max(abs(force))))
        saturated_s += bool(np.any(abs(force) >= p.torque_limit_nm*.99))*sim.model.opt.timestep
        work += float(np.maximum(force*sim.data.qvel[sim.drive.vadr], 0).sum())*sim.model.opt.timestep
        limit_error = max(limit_error, float(np.max(np.maximum(abs(sim.data.qpos[sim.drive.qadr])-np.pi/2, 0))))
        if not np.isfinite(sim.data.qpos).all():
            raise RuntimeError('Nonfinite physical state')
        if sim.data.time+1e-10 >= next_sample:
            record()
            next_sample += .1
            if manual_direction:
                completed_now = [e for e in sim.controller.events if e['kind'] == 'complete']
                if len(completed_now) == 2 and sim.data.time >= completed_now[-1]['time_s']+3*period:
                    break
            if scene != 'straight':
                if exit_time is None and records[-1]['min_body_progress_m'] > sim.feature[-1]+.05:
                    exit_time = float(sim.data.time)
                    exit_phase = records[-1]['phase_rad']
                if exit_cycles and exit_phase is not None and records[-1]['phase_rad'] >= exit_phase+exit_cycles*2*np.pi:
                    break
    audited = [r for r in records if r['time_s'] >= p.settle_s+1]
    valid = [r for r in audited if r['shape']['wrap_valid']]
    wrap = [r['shape']['wrap_deg'] for r in valid]
    unsupported = streak = 0.
    for row in audited:
        streak = streak+.1 if sum(row['support_up_n']) < 2 else 0.
        unsupported = max(unsupported, streak)
    final = records[-1]
    events = sim.controller.events if mode == 'sliding' else []
    completed = [e for e in events if e['kind'] == 'complete']
    geometry_rows = [r for r in audited if r['geometry'].get('target_section_turn_deg') is not None]
    checks = dict(
        passive_zero=passive_max < 1e-12, torque_bounded=torque_max <= p.torque_limit_nm+1e-9,
        no_ground_support=max(r['ground_contact_force_n'] for r in audited) < .1,
        support_retained=unsupported < .5, limits=limit_error < .02,
        solver_stable=bool(np.array_equal([w.number for w in sim.data.warning], warnings)),
        body_open=all(150 <= r['body_geometry']['turn_deg'] <= 218 for r in audited),
        opening=min(r['body_geometry']['gap_fraction'] for r in audited) >= .4,
        near_planar=max(r['body_geometry']['fit_rms_m'] for r in audited) <= .015,
        sent_target_open=all(r['sent_geometry']['turn_deg'] <= 190.01 and r['sent_geometry']['gap_fraction'] >= .4 for r in audited),
        progress=final['progress_m']-initial_progress > .3)
    pipe_checks = dict(shape_observable=len(valid) == len(audited),
                       actual_pipe_open=bool(wrap) and min(wrap) >= 150 and max(wrap) <= 218)
    if mode != 'baseline':
        checks['reference_feasible'] = bool(geometry_rows) and all(r['geometry']['status'] == 'running' for r in geometry_rows)
    if manual_direction:
        checks['two_handoffs'] = len(completed) == 2 and all(r['accepted'] for r in manual_requests)
        checks['post_handoff_cycles'] = bool(completed) and final['time_s']-completed[-1]['time_s'] >= 3*period
    if payload:
        checks['automatic_migration_observed'] = any(e.get('reason') == 'persistent observed load displacement' for e in events if e['kind'] in ('prepare', 'begin')) and bool(completed)
    if mode == 'sliding':
        checks['acquired'] = sim.controller.window is not None
        checks['no_unfinished_handover'] = sim.controller.transfer is None and sim.controller.preparation is None
        checks['no_handover_abort'] = not any(e['kind'] in ('abort', 'rollback', 'reject') for e in events)
        checks['online_sent_target_open'] = sim.controller.sent_constraint_violations == 0
    if scene != 'straight':
        checks['whole_body_beyond_feature'] = final['min_body_progress_m'] > sim.feature[-1]+.05
        checks['post_exit_cycles'] = exit_phase is not None and final['phase_rad']-exit_phase >= exit_cycles*2*np.pi-1e-8
    # Sharp-corner body openness and pipe-frame observability are reported
    # separately. Never silently turn an unobservable pipe angle into a pass.
    checks = {k: bool(v) for k, v in checks.items()}
    pipe_checks = {k: bool(v) for k, v in pipe_checks.items()}
    report = dict(label=label, scene=scene, mode=mode, parameters=asdict(p), options=asdict(options),
                  start_x=start, initial_window=initial_window, payload=payload,
                  manual_direction=manual_direction, manual_requests=manual_requests,
                  passed=all(checks.values()) and all(pipe_checks.values()),
                  motion_and_body_passed=all(checks.values()), checks=checks, pipe_checks=pipe_checks,
                  duration_s=final['time_s'], maximum_duration_s=duration, exit_time_s=exit_time,
                  post_exit_s=None if exit_time is None else final['time_s']-exit_time,
                  post_exit_phase_cycles=None if exit_phase is None else (final['phase_rad']-exit_phase)/(2*np.pi),
                  progress_m=final['progress_m']-initial_progress,
                  minimum_wrap_deg=None if not wrap else min(wrap), maximum_wrap_deg=None if not wrap else max(wrap),
                  unobservable_samples=len(audited)-len(valid),
                  maximum_body_turn_deg=max(r['body_geometry']['turn_deg'] for r in audited),
                  minimum_body_turn_deg=min(r['body_geometry']['turn_deg'] for r in audited),
                  minimum_gap_fraction=min(r['body_geometry']['gap_fraction'] for r in audited),
                  maximum_fit_rms_m=max(r['body_geometry']['fit_rms_m'] for r in audited),
                  maximum_sent_turn_deg=max(r['sent_geometry']['turn_deg'] for r in audited),
                  passive_max_nm=passive_max, maximum_torque_nm=torque_max,
                  maximum_unsupported_s=unsupported, positive_work_j=work, saturated_s=saturated_s,
                  events=events, final_geometry=final['geometry'], wall_s=time.monotonic()-wall,
                  acceptance=dict(body_turn_deg=[150, 218], pipe_wrap_deg=[150, 218], gap_min=.4,
                                  plane_rms_max_m=.015, sent_turn_max_deg=190., command_audit_period_s=.02))
    output.mkdir(parents=True, exist_ok=True)
    (output/f'{label}.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    with gzip.open(output/f'{label}_records.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(records, stream, separators=(',', ':'), allow_nan=False)
    np.savez_compressed(output/f'{label}_trace.npz', qpos=states, qvel=velocities, backbone=bones,
                        time_s=[r['time_s'] for r in records], windows=[r['window'] for r in records],
                        targets_rad=[r['target_rad'] for r in records], enabled=[r['enabled'] for r in records],
                        weights=[r['weights'] for r in records], support_up_n=[r['support_up_n'] for r in records],
                        actuator_force_nm=[r['actuator_force_nm'] for r in records])
    print(json.dumps({k: report[k] for k in ('label', 'motion_and_body_passed', 'pipe_checks', 'checks', 'progress_m', 'maximum_body_turn_deg', 'events', 'wall_s')}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='straight')
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'validation/reports/sliding_section')
    parser.add_argument('--duration', type=float, default=62.)
    parser.add_argument('--mode', choices=['baseline', 'fixed', 'sliding'], default='sliding')
    parser.add_argument('--start', type=float, default=-.5)
    parser.add_argument('--manual-direction', type=int, choices=[-1, 0, 1], default=0)
    parser.add_argument('--initial-window', type=lambda text: tuple(map(int, text.split(':'))))
    parser.add_argument('--payload', action='store_true')
    parser.add_argument('--exit-cycles', type=int, default=0)
    args = parser.parse_args()
    run(**vars(args))
