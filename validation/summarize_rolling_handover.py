"""Collect case-specific scope, supported intervals and failure timing."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import math
import numpy as np
import mujoco
from snake_control.simulation import ROOT
from snake_control.section_geometry import local_section

REPORT = ROOT/'validation/reports/rolling_handover'


def load(path):
    with gzip.open(path, 'rt', encoding='utf8') as f: return [json.loads(line) for line in f]


def main():
    collected = []
    folders = sorted((p for p in REPORT.iterdir() if p.is_dir() and (p/'status.json').exists()),
                     key=lambda p: '_'.join(p.name.split('_')[-3:]))
    for folder in folders:
        r = json.loads((folder/'review.json').read_text())
        status = json.loads((folder/'status.json').read_text())
        rows, evaluation = load(folder/'actual_trace.jsonl.gz'), load(folder/'evaluation.jsonl.gz')
        failure = next((row for row in rows if row['controller']['fault'] or row['controller'].get('resize_fault')), None)
        support_loss = next((v for v in evaluation if v['time_s'] >= 2. and v['support_n'] < 2.), None)
        first_floor = None
        if status['statistics_1ms']['floor_contact_steps']:
            model = mujoco.MjModel.from_xml_path(str(ROOT/f"mujoco_model/Snake_Robot_o20_Rolling_Arc_{status['scene']}.xml"))
            data, floor = mujoco.MjData(model), model.geom('floor').id
            robot = set(np.flatnonzero(model.geom_bodyid >= model.body('head').id))
            for row in rows:
                if row['time_s'] < 2.: continue
                data.qpos[:], data.qvel[:], data.time = row['actual_qpos'], row['actual_qvel'], row['time_s']
                mujoco.mj_forward(model, data)
                if any(floor in (int(c.geom1), int(c.geom2)) and bool({int(c.geom1), int(c.geom2)} & robot) for c in data.contact):
                    first_floor = row['time_s']; break
        prefix = [v for v in evaluation if 2. <= v['time_s'] < 24.] if status['case'] == 'short_return' else None
        if prefix:
            prefix = dict(time_interval_s=[2., 24.], body_turn_range_deg=[min(v['body_geometry']['turn_deg'] for v in prefix),
                max(v['body_geometry']['turn_deg'] for v in prefix)], minimum_support_n=min(v['support_n'] for v in prefix),
                minimum_gap=min(v['body_geometry']['gap_fraction'] for v in prefix),
                pipe_wrap_range_deg=[min(v['pipe_shape']['wrap_deg'] for v in prefix), max(v['pipe_shape']['wrap_deg'] for v in prefix)],
                body_violations=sum(not v['actual_open'] for v in prefix),
                maximum_plane_rms_m=max(v['body_geometry']['fit_rms_m'] for v in prefix))
        s = r['summary']
        changes = [e for e in status['events'] if e['kind'] in ('complete', 'resize_complete')]
        if changes:
            s['post_last_change_cycles'] = (rows[-1]['controller']['phase_rad']-changes[-1]['phase_rad'])/(2*math.pi)
            s['post_last_change_phase_basis'] = 'recorded_event_phase'
            (folder/'review.json').write_text(json.dumps(r, indent=2), encoding='utf8')
        radii, minimum_local_radius, worst_curvature_time, curvature_unknown = [], None, None, 0
        for row, ev in zip(rows, evaluation):
            if row['time_s'] < 2.: continue
            radii.append(ev['body_circle']['radius_m'])
            try:
                w = row['controller']['window']
                section = local_section(np.asarray(row['actual_qpos'])[7:], [.0645]*21, *w)
                local = [abs(segment.radius) for segment in section['curve']
                         if 1 <= segment.joint <= w[1]-w[0]+1 and segment.radius > 0]
                if local and (minimum_local_radius is None or min(local) < minimum_local_radius):
                    minimum_local_radius, worst_curvature_time = min(local), row['time_s']
            except ValueError:
                curvature_unknown += 1
        item = dict(source=folder.name, case=r['case'], time_s=status['final_time_s'],
            moving_case_pass=r['passed'], body_shape_pass=r['shape_pass'], mechanics_pass=r['mechanics_pass'],
            strict_pipe_pass=r['strict_pipe_pass'], continuous_phase_pass=r['phase_continuity_pass'],
            position_transfers=r['transfers'], visited_windows=r['visited_windows'],
            bidirectional_return_on_pipe=r['bidirectional_return_on_pipe_observed'],
            full_phase_reliable_position_pass=r['full_phase_reliable_position_pass'],
            minimum_passive_joint_count=r['minimum_passive_joint_count'],
            final_six_trial=status['final_controller'].get('six_trial'),
            prefer_feasible_six=status['final_controller'].get('prefer_feasible_six'),
            first_support_below_2n_s=None if not support_loss else support_loss['time_s'],
            first_floor_contact_saved_time_s=first_floor,
            first_controller_fault_s=None if not failure else failure['time_s'],
            initial_short_unforced_prefix=prefix, actual_width_chain=r['completed_widths'],
            continuous_exit_cycles=s['continuous_exit_cycles'], post_last_change_cycles=s['post_last_change_cycles'])
        item['curvature_review'] = dict(fitted_body_radius_range_m=[min(radii), max(radii)],
            minimum_projected_smooth_curve_radius_m=minimum_local_radius,
            worst_local_curvature_time_s=worst_curvature_time, unobservable_frames=curvature_unknown,
            definition='actual angle reconstructed plane/smooth-curve proxy; not pipe radius or solid-link bend radius')
        collected.append(item)
        # Latest chronological case wins, including a repair after its failure.
        (REPORT/(r['case']+'_review.json')).write_text(json.dumps(r, indent=2), encoding='utf8')
    result = dict(dynamics_attempts=len(collected), amended_budget_maximum=5, hardware_runs=0,
                  real_and_manual_entries_updated=False, all_requested_scope_complete=False,
                  future_driver_termination_revision_dynamics_verified=False, cases=collected)
    (REPORT/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__': main()
