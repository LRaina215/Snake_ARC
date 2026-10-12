"""Replay two archived cases with a strictly angle-only observer.

The observer is called before evaluating matched MuJoCo pose/contact records.
No mj_step, drive, target generation, serial, or window update is performed.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import math
from collections import Counter
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.angle_shape_observer import AngleShapeObserver
from snake_control.section_geometry import forward
from snake_control.simulation import backbone

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'validation' / 'reports' / 'debug_modes'
OUTPUT = ROOT / 'validation' / 'reports' / 'angle_observer'
LENGTHS = [.0645] * 21
# Canonical FK x/y/z -> MuJoCo head-local z/y/-x, fixed mechanical convention.
CANONICAL_TO_MODEL = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])


def read_trace(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def motor_angles(row):
    # Archive schema is a named motor dictionary, not a numerically ordered list.
    measured = row['measured_controller_deg']
    return [float(measured[f'J{j}']) for j in range(1, 21)]


def percentiles(values):
    values = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not values:
        return None
    return dict(samples=len(values), minimum=min(values), median=float(np.median(values)),
                p90=float(np.percentile(values, 90)), maximum=max(values))


def coverage(support, window):
    a, b = window
    return float(sum(support[a - 1:b + 1]) / max(sum(support), 1e-12))


def evaluate(row, matched_state, observer, result, model, data):
    if matched_state is None:
        return dict(matched=False, reason='feedback precedes first archived physical state')
    qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    actual_q = np.array(matched_state['actual_qpos'])[qadr]
    measured_q = np.radians(motor_angles(row))
    motor_match = float(np.max(abs(actual_q - measured_q)))
    if motor_match > 1e-8:
        raise ValueError(f'Motor feedback and matched physical angles differ by {motor_match} rad')
    data.qpos[:] = matched_state['actual_qpos']
    data.qvel[:] = matched_state['actual_qvel']
    data.time = matched_state['time_s']
    mujoco.mj_forward(model, data)
    world = backbone(model, data)
    points = forward(measured_q, LENGTHS)
    rotation = data.body('head').xmat.reshape(3, 3) @ CANONICAL_TO_MODEL
    reconstructed = points @ rotation.T + data.site('s_head').xpos
    fk_residual = float(np.max(np.linalg.norm(world - reconstructed, axis=1)))
    if fk_residual > 1e-8:
        raise ValueError(f'Canonical FK does not match actual bone geometry: {fk_residual} m')
    true_down = rotation.T @ (model.opt.gravity / np.linalg.norm(model.opt.gravity))
    window = tuple(result['current_window'])
    support = matched_state['support_up_n']
    support_center = float(np.dot(support, np.arange(1, observer.n + 1)) / sum(support))
    neighbours = [(a, a + 5) for a in (window[0] - 1, window[0], window[0] + 1)
                  if a >= 2 and a + 5 < observer.n]
    candidate_coverage = [{'window': list(w), 'coverage': coverage(support, w)} for w in neighbours]
    best = max(candidate_coverage, key=lambda r: r['coverage'])
    current_coverage = coverage(support, window)
    evaluation = dict(matched=True, state_time_s=matched_state['time_s'], motor_match_max_rad=motor_match,
                      fk_match_max_m=fk_residual, true_down_head=true_down.tolist(),
                      support_center_joint_bin=support_center,
                      support_center_body_coordinate=support_center + .5,
                      support_cycle_center_joint_bin=matched_state['sliding']['cycle_support_center_joint'],
                      support_sum_n=sum(support), current_coverage=current_coverage,
                      neighbour_coverage=candidate_coverage, best_neighbour=best,
                      suggestion_coverage=(None if result['suggested_window'] is None else
                                           coverage(support, result['suggested_window'])),
                      original_window=matched_state['window'], original_geometry=matched_state['actual_body_geometry'])
    if 'down_head' in result:
        evaluation['down_error_deg'] = math.degrees(math.acos(float(np.clip(true_down @ result['down_head'], -1., 1.))))
    _, _, section = observer.geometry(measured_q, window)
    try:
        oracle = observer.crest(points, window, section, true_down)
        evaluation['true_gravity_smooth_crest_joint'] = oracle['joint_coordinate']
        if 'reference_joint_coordinate' in result:
            evaluation['true_gravity_raw_error_joints'] = oracle['joint_coordinate'] - result['reference_joint_coordinate']
        if 'crest' in result:
            evaluation['crest_position_error_joints'] = result['crest']['joint_coordinate'] - oracle['joint_coordinate']
    except ValueError as exc:
        evaluation['true_gravity_geometry_error'] = str(exc)
    start = max(0, window[0] - observer.options.boundary_links)
    stop = min(len(world), window[1] + observer.options.boundary_links + 2)
    coordinates, physical = [], []
    for j in range(start, stop - 1):
        for u in np.linspace(0., 1., 20, endpoint=False):
            coordinates.append(j + u)
            physical.append((1 - u) * world[j] + u * world[j + 1])
    z = np.asarray(physical)[:, 2]
    weight = np.maximum(0., z - z.max() + np.median(LENGTHS) * observer.options.top_band_link_fraction)
    evaluation['true_rigid_backbone_crest_joint'] = float(weight @ coordinates / weight.sum())
    return evaluation


def recommendations(rows):
    spans = []
    active = None
    for row in rows:
        o = row['observation']
        key = None if not o['direction'] else (o['direction'], tuple(o['suggested_window']))
        if key is None or active is None or key != active['key']:
            if active is not None:
                spans.append(active)
                active = None
            if key is not None:
                active = dict(key=key, start_s=row['feedback_time_s'], end_s=row['feedback_time_s'],
                              frames=0, support_coverages=[])
        if active is not None:
            active['end_s'] = row['feedback_time_s']
            active['frames'] += 1
            c = row['evaluation'].get('suggestion_coverage')
            if c is not None:
                active['support_coverages'].append(c)
    if active is not None:
        spans.append(active)
    return [dict(direction=s['key'][0], window=list(s['key'][1]), start_s=s['start_s'], end_s=s['end_s'],
                 frames=s['frames'], support_coverage=percentiles(s['support_coverages'])) for s in spans]


def subset_summary(rows):
    valid = [r for r in rows if r['observation']['valid']]
    matched = [r for r in rows if r['evaluation']['matched']]
    valid_matched = [r for r in valid if r['evaluation']['matched']]
    suggested = [r for r in rows if r['observation']['direction']]
    return dict(frames=len(rows), valid_frames=len(valid), valid_fraction=len(valid) / max(len(rows), 1),
                recommendation_frames=len(suggested),
                states=dict(Counter(r['observation']['state'] for r in rows)),
                rejection_reasons=dict(Counter(reason for r in rows for reason in r['observation']['reasons'])),
                raw_error_joints=percentiles(r['observation'].get('raw_error_joints') for r in rows),
                filtered_error_joints=percentiles(r['observation'].get('filtered_error_joints') for r in rows),
                down_error_all_deg=percentiles(r['evaluation'].get('down_error_deg') for r in matched),
                down_error_valid_deg=percentiles(r['evaluation'].get('down_error_deg') for r in valid_matched),
                absolute_crest_error_all_joints=percentiles(abs(r['evaluation']['crest_position_error_joints'])
                    for r in matched if 'crest_position_error_joints' in r['evaluation']),
                absolute_crest_error_valid_joints=percentiles(abs(r['evaluation']['crest_position_error_joints'])
                    for r in valid_matched if 'crest_position_error_joints' in r['evaluation']),
                true_gravity_error_joints=percentiles(r['evaluation'].get('true_gravity_raw_error_joints') for r in matched),
                support_cycle_center_joint_bin=percentiles(r['evaluation'].get('support_cycle_center_joint_bin') for r in matched),
                current_support_coverage=percentiles(r['evaluation'].get('current_coverage') for r in matched),
                recommendation_support_coverage=percentiles(r['evaluation'].get('suggestion_coverage') for r in suggested),
                recommendation_spans=recommendations(rows))


def replay(label, path, model, data):
    raw = read_trace(path)
    observer = AngleShapeObserver(LENGTHS)
    states = {round(r['time_s'], 8): r for r in raw}
    phase_history = {}
    derived = []
    for row in raw:
        time_s = float(row['feedback_time_s'])
        phase = phase_history.get(round(time_s, 8), 0.)
        motor_input = dict(angles_deg=motor_angles(row), phase_rad=phase,
                           enabled=row['enabled'], feedback_time_s=time_s)
        # This call has no access to row, states, stage, contacts, or root pose.
        result = observer.update(**motor_input)
        matched = states.get(round(time_s, 8))
        assessment = evaluate(row, matched, observer, result, model, data)
        derived.append(dict(source_snapshot_time_s=row['time_s'], feedback_time_s=time_s,
                            stage=row['stage'], input=motor_input,
                            observation=result, evaluation=assessment,
                            before_original_handover=row['window'] == [8, 13]))
        phase_history[round(row['time_s'], 8)] = row['phase_rad']
    summary = dict(source=str(path.relative_to(ROOT)), frames=len(derived), config=observer.configuration(),
                   original_feedback_delay_s=percentiles(r['source_snapshot_time_s'] - r['feedback_time_s'] for r in derived),
                   stages={stage: subset_summary([r for r in derived if r['stage'] == stage])
                           for stage in ('no_force', 'right_force', 'recovery')},
                   before_original_handover=subset_summary([r for r in derived if r['before_original_handover']]),
                   after_original_handover=subset_summary([r for r in derived if not r['before_original_handover']]),
                   max_motor_matching_error_rad=max(r['evaluation'].get('motor_match_max_rad', 0.) for r in derived),
                   max_fk_world_matching_error_m=max(r['evaluation'].get('fk_match_max_m', 0.) for r in derived),
                   matched_frames=sum(r['evaluation']['matched'] for r in derived),
                   first_external_window_change_s=next((r['feedback_time_s'] for r in derived
                       if r['observation']['external_window_change']), None),
                   old_controller_events=raw[-1]['sliding_events'])
    with gzip.open(OUTPUT / f'{label}_observer_trace.jsonl.gz', 'wt', encoding='utf8') as stream:
        for row in derived:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return derived, summary


def graph(traces):
    fig, axes = plt.subplots(5, 2, figsize=(14, 13), sharex=True)
    for column, (title, rows) in enumerate(traces):
        t = np.array([r['feedback_time_s'] for r in rows])
        o = [r['observation'] for r in rows]
        e = [r['evaluation'] for r in rows]
        value = lambda collection, key: [r.get(key, np.nan) for r in collection]
        axes[0, column].plot(t, value(o, 'raw_error_joints'), alpha=.45, label='Angle crest offset (raw)')
        axes[0, column].plot(t, value(o, 'filtered_error_joints'), label='Valid filtered offset')
        axes[0, column].plot(t, value(e, 'true_gravity_raw_error_joints'), '--', label='True-gravity diagnostic')
        axes[0, column].axhline(.5, color='gray', ls=':')
        axes[0, column].axhline(-.5, color='gray', ls=':')
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel('Offset (joints)')
        axes[1, column].plot(t, value(e, 'down_error_deg'), label='Inferred down vs actual gravity')
        axes[1, column].plot(t, [r.get('end_quality', {}).get('end_disagreement_deg', np.nan) for r in o], label='End-direction disagreement')
        axes[1, column].axhline(15., color='gray', ls=':', label='Evaluation 15 deg')
        axes[1, column].set_ylabel('Angle (deg)')
        axes[2, column].plot(t, [r.get('crest', {}).get('joint_coordinate', np.nan) for r in o], label='Angle crest coordinate')
        axes[2, column].plot(t, value(e, 'true_gravity_smooth_crest_joint'), label='True-gravity crest')
        cycle = [np.nan if r.get('support_cycle_center_joint_bin') is None else r['support_cycle_center_joint_bin'] + .5 for r in e]
        axes[2, column].plot(t, cycle, label='Cycle support center +0.5')
        axes[2, column].set_ylabel('Body coordinate')
        axes[3, column].plot(t, value(e, 'current_coverage'), label='Current-window support coverage')
        axes[3, column].plot(t, value(e, 'suggestion_coverage'), 'o', ms=2, label='Suggestion coverage')
        axes[3, column].axhline(.8, color='gray', ls=':')
        axes[3, column].set_ylabel('Support fraction')
        axes[4, column].step(t, [int(r['valid']) for r in o], label='Observation valid', where='post')
        axes[4, column].step(t, [r['direction'] for r in o], label='Suggestion: head -1, tail +1', where='post')
        axes[4, column].set_ylabel('State')
        axes[4, column].set_xlabel('Motor-feedback time (s)')
        for ax in axes[:, column]:
            ax.axvspan(14.2, 28.2, color='#f3c867', alpha=.14)
            if title.startswith('Endpoint'):
                ax.axvline(33.88, color='#ad4b78', ls='--', alpha=.7)
            ax.grid(alpha=.2)
            ax.legend(loc='best', fontsize=8)
    fig.suptitle('One frozen angle-only observer, two saved cases; no window execution\nOrange: force stage. Magenta: original controller handover (not observer execution).')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'observer_full_process.png', dpi=150)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'mujoco_model' / 'Snake_Robot_o20_Rolling_Arc_straight.xml'))
    data = mujoco.MjData(model)
    metadata = model.numeric('pipe_centerline').data.reshape(-1, 4)
    traces, summaries = [], {}
    for label, title, filename in (
        ('whole_body', 'Whole-body horizontal force', 'whole_body_trace.jsonl.gz'),
        ('endpoint', 'Endpoint outward/downward pull', 'endpoint_pull_trace.jsonl.gz')):
        rows, summary = replay(label, SOURCE / filename, model, data)
        traces.append((title, rows))
        summaries[label] = summary
    result = dict(analysis='one frozen observer; two complete archived traces; zero new dynamics runs',
                  controller_input='20 measured angles, known phase, current enable mask, feedback timestamp',
                  evaluation_only='matched free pose, actual gravity, contacts, old events, stage labels',
                  environment=dict(python=sys.version.split()[0], mujoco=mujoco.__version__, numpy=np.__version__),
                  pipe_diameter_min_max_m=[float(2 * metadata[:, 3].min()), float(2 * metadata[:, 3].max())],
                  support_coordinate_convention='contact bin j represents downstream link j, whose center is body coordinate j+0.5',
                  normal_prefix_identical=all(a['input'] == b['input'] for a, b in zip(traces[0][1][:600], traces[1][1][:600])),
                  cases=summaries)
    (OUTPUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    graph(traces)
    for label, summary in summaries.items():
        text = dict(case=label, matched_frames=summary['matched_frames'],
                    motor_match=summary['max_motor_matching_error_rad'], fk_match=summary['max_fk_world_matching_error_m'],
                    stages=summary['stages'], handover_cutoff=summary['first_external_window_change_s'])
        print(json.dumps(text, ensure_ascii=False))


if __name__ == '__main__':
    main()
