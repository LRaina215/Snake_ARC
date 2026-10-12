"""Explain the frozen observer's saved outputs; do not rerun or retune it."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.angle_shape_observer import AngleShapeObserver
from snake_control.section_geometry import forward
from validation.angle_observer_replay import OUTPUT, LENGTHS, percentiles, subset_summary


def load(label):
    with gzip.open(OUTPUT / f'{label}_observer_trace.jsonl.gz', 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def angle(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1., 1.))))


def describe(rows):
    matched = [r for r in rows if r['evaluation']['matched']]
    head = [angle(r['observation']['end_quality']['head_direction'], r['evaluation']['true_down_head']) for r in matched]
    tail = [angle(r['observation']['end_quality']['tail_direction'], r['evaluation']['true_down_head']) for r in matched]
    covers = [r['evaluation']['neighbour_coverage'] for r in matched]
    return dict(frames=len(rows), head_end_down_error_deg=percentiles(head), tail_end_down_error_deg=percentiles(tail),
                both_ends_same_common_bias_frames=sum(h > 15 and t > 15 for h, t in zip(head, tail)),
                valid_but_down_error_over_15_frames=sum(r['observation']['valid'] and
                    r['evaluation'].get('down_error_deg', 0.) > 15 for r in matched),
                three_windows_all_cover_at_least_80pct_frames=sum(len(c) == 3 and all(x['coverage'] >= .8 for x in c) for c in covers),
                three_windows_all_cover_all_support_frames=sum(len(c) == 3 and all(x['coverage'] >= 1 - 1e-8 for x in c) for c in covers),
                current_to_best_coverage_gain=percentiles(r['evaluation']['best_neighbour']['coverage'] -
                    r['evaluation']['current_coverage'] for r in matched),
                latent_offset_over_half_joint_frames=sum(abs(r['observation'].get('filtered_error_joints', 0.)) > .5 for r in rows),
                false_normal_proposals=sum(bool(r['observation']['direction']) for r in rows if r['stage'] == 'no_force'))


def persistent_signal_spans(rows, key, threshold=.5):
    """Evaluation-only diagnostic of a geometric signal, not a new observer."""
    filtered, last_time, mask, pending = None, None, None, None
    spans = []
    for row in rows:
        value = row['evaluation'].get(key)
        t = row['feedback_time_s']
        window = tuple(row['observation']['current_window'])
        if value is None:
            continue
        if window != mask:
            filtered, pending = None, None
        dt = 0. if last_time is None else t - last_time
        filtered = value if filtered is None else filtered + (1 - math.exp(-dt / .2)) * (value - filtered)
        direction = 0 if abs(filtered) <= threshold else (1 if filtered > 0 else -1)
        if not direction or (pending and pending['direction'] != direction):
            if pending and pending['end_s'] - pending['start_s'] >= .5:
                spans.append(pending)
            pending = None
        if direction:
            if pending is None:
                pending = dict(start_s=t, end_s=t, direction=direction)
            pending['end_s'] = t
        last_time, mask = t, window
    if pending and pending['end_s'] - pending['start_s'] >= .5:
        spans.append(pending)
    return spans


def snapshots(selected):
    observer = AngleShapeObserver(LENGTHS)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, (title, row) in zip(axes.ravel(), selected):
        q = np.radians(row['input']['angles_deg'])
        p = forward(q, LENGTHS)
        o, e = row['observation'], row['evaluation']
        w = tuple(o['current_window'])
        _, _, section = observer.geometry(q, w)
        origin, basis = section['axis']['point'], section['axis']['basis'][:, :2]
        down_true = np.array(e['true_down_head'])
        vertical = -(down_true @ basis)
        vertical /= np.linalg.norm(vertical)
        horizontal = np.array([vertical[1], -vertical[0]])
        plane = basis @ np.column_stack([horizontal, vertical])
        xy = (p - origin) @ plane
        ax.plot(xy[:, 0], xy[:, 1], '.-', color='#9b9b9b', label='Actual full-body projection')
        current = xy[w[0]:w[1] + 2]
        ax.plot(current[:, 0], current[:, 1], 'o-', color='#237cb7', lw=2, label='Current enabled arc')
        support = np.array(row['evaluation_support'])
        centres = (xy[1:-1] + xy[2:]) / 2
        positive = support > 1e-6
        ax.scatter(centres[positive, 0], centres[positive, 1], s=15 + 4 * support[positive],
                   c='#c33b36', alpha=.7, label='Support bins at link centres')
        proxy = (np.array(o['crest']['point_head_m']) - origin) @ plane
        true = observer.crest(p, w, section, down_true)
        oracle = (np.array(true['point_head_m']) - origin) @ plane
        ax.scatter(*proxy, marker='o', c='black', s=65, label='Angle-inferred crest')
        ax.scatter(*oracle, marker='x', c='#e58c22', s=85, label='True-gravity geometric crest')
        start = np.array([.19, .18])
        for direction, color, name in ((down_true, '#6a48a0', 'Actual gravity (evaluation)'),
                                        (np.array(o['down_head']), 'black', 'Angle-inferred down')):
            d = direction @ plane
            ax.arrow(*start, *(d * .18), width=.003, color=color, length_includes_head=True, label=name)
        ax.text(*xy[0], 'Head', fontsize=9)
        ax.text(*xy[-1], 'Tail', fontsize=9)
        ax.set_title(f'{title}\nfeedback {row["feedback_time_s"]:.2f} s | {o["state"]}', fontsize=10)
        ax.set_xlabel('Section horizontal (m)')
        ax.set_ylabel('Up along actual gravity (m), evaluation only')
        ax.axis('equal')
        ax.grid(alpha=.2)
        ax.margins(x=.12, y=.12)
    fig.suptitle('Selected actual states: causal observer estimates vs evaluation-only gravity/support\nProjection uses actual gravity for viewing; the observer never receives it.')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8)
    fig.tight_layout(rect=[0., .10, 1., .94])
    fig.savefig(OUTPUT / 'diagnostic_actual_states.png', dpi=150)
    plt.close(fig)


def main():
    traces = {label: load(label) for label in ('whole_body', 'endpoint')}
    # These are previously measured support arrays, only used to draw evaluation bins.
    for label, source_name in (('whole_body', 'whole_body_trace.jsonl.gz'), ('endpoint', 'endpoint_pull_trace.jsonl.gz')):
        source = OUTPUT.parent / 'debug_modes' / source_name
        with gzip.open(source, 'rt', encoding='utf8') as stream:
            original = {round(r['time_s'], 8): r for r in (json.loads(line) for line in stream)}
        for row in traces[label]:
            matched = original.get(round(row['feedback_time_s'], 8))
            if matched:
                row['evaluation_support'] = matched['support_up_n']
    result = dict(analysis='post-hoc cause audit of saved outputs; no observer rerun, retuning, or actuation', cases={})
    for label, rows in traces.items():
        result['cases'][label] = dict(
            stages={stage: describe([r for r in rows if r['stage'] == stage])
                    for stage in ('no_force', 'right_force', 'recovery')},
            true_gravity_signal_spans=persistent_signal_spans(rows, 'true_gravity_raw_error_joints'))
    endpoint = traces['endpoint']
    result['endpoint_time_partitions'] = {
        'before_old_prepare': subset_summary([r for r in endpoint if r['feedback_time_s'] < 27.46]),
        'old_prepare_only_before_phase_hold': subset_summary([r for r in endpoint if 27.46 <= r['feedback_time_s'] < 30.96]),
        'old_phase_hold_before_mask_change': subset_summary([r for r in endpoint if 30.96 <= r['feedback_time_s'] < 33.88]),
        'after_old_mask_change': subset_summary([r for r in endpoint if r['feedback_time_s'] >= 33.88])}
    selected = [
        ('Normal phase-reference offset blocked by candidate shape', max(
            (r for r in traces['whole_body'] if r['stage'] == 'no_force' and r['evaluation']['matched']),
            key=lambda r: abs(r['observation']['filtered_error_joints']))),
        ('Both ends agree but inferred down is wrong', max(
            (r for r in traces['whole_body'] if r['stage'] == 'right_force'),
            key=lambda r: r['evaluation']['down_error_deg'])),
        ('Pulled tail violates the hanging-end assumption', max(
            (r for r in endpoint if r['stage'] == 'right_force'),
            key=lambda r: r['observation']['end_quality']['end_disagreement_deg'])),
        ('First headward suggestion after releasing the pull', next(
            r for r in endpoint if r['observation']['direction']))]
    result['selected_actual_states'] = [{k: v for k, v in row.items() if k != 'evaluation_support'} for _, row in selected]
    (OUTPUT / 'cause_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    snapshots(selected)
    print(json.dumps({label: result['cases'][label]['stages'] for label in traces}, ensure_ascii=False))
    print(json.dumps({'true_gravity_signal_spans': {k: v['true_gravity_signal_spans'] for k, v in result['cases'].items()},
                      'first_suggestion_feedback_s': selected[-1][1]['feedback_time_s']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
