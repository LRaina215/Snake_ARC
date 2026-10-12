"""Freeze one normal-cycle reference, then replay the two existing records.

No mj_step, simulated feedback, parameter scan, serial or actuator calls.
All training fields are motor angles, commanded phase and enable mask only.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import math
from collections import Counter
import numpy as np
from snake_control.motor_shape_observer import MotorShapeObserver, PhaseShapeReference, bending_features, local_phase

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'validation/reports/angle_observer'
REPORT = ROOT / 'validation/reports/motor_feedback_sliding'
REFERENCE = ROOT / 'configs/motor_shape_reference.json'


def load(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def build_reference(rows):
    normal = [r for r in rows if r['stage'] == 'no_force']
    first_phase = normal[0]['input']['phase_rad']
    cycle = [r for r in normal if first_phase <= r['input']['phase_rad'] < first_phase+2*math.pi]
    bins = [[] for _ in range(24)]
    for row in cycle:
        motor = row['input']
        active = np.flatnonzero(motor['enabled'])+1
        window = (int(active[0]), int(active[-1]))
        shape = row['observation']
        if not shape['valid']:
            raise ValueError('Calibration cycle has invalid passive/open shape')
        f = bending_features(motor['angles_deg'], [.0645]*21, window)
        middle = sum(window)/2
        value = [shape['crest']['joint_coordinate']-middle,
                 f['bending_center_joint']-middle,
                 f['tail_boundary_turn_deg']-f['head_boundary_turn_deg']]
        index = min(23, int(local_phase(motor['phase_rad'], window[0])/(2*math.pi)*24))
        bins[index].append(value)
    if min(map(len, bins)) < 3:
        raise ValueError('Not enough feedback in every phase bin')
    values = [np.median(bucket, axis=0).tolist() for bucket in bins]
    reference = PhaseShapeReference(values, dict(
        source='validation/reports/angle_observer/whole_body_observer_trace.jsonl.gz',
        stage='shared no_force prefix; first full commanded cycle only', samples=len(cycle),
        start_time_s=cycle[0]['feedback_time_s'], end_time_s=cycle[-1]['feedback_time_s'],
        phase_span_rad=cycle[-1]['input']['phase_rad']-first_phase,
        start_phase_rad=first_phase, bins=24, bin_counts=list(map(len, bins)),
        first_joint=8, calibration_window=[8, 13], link_lengths_m=[.0645]*21,
        adaptation='frozen offline; no online learning',
        fields=['angle-only crest relative to window center', 'angle bending center relative to window center',
                'tail minus head passive-boundary absolute turns'],
        unused_inputs=['contact', 'root pose', 'gravity truth', 'motor torque', 'side-pull labels']))
    return reference


def summary(rows):
    result = dict(samples=len(rows), states=dict(Counter(r['observation']['state'] for r in rows)),
        valid_samples=sum(r['observation']['valid'] for r in rows),
        suggested_samples=sum(bool(r['observation']['direction']) for r in rows), spans=[])
    active = None
    for r in rows:
        o = r['observation']
        if not o['direction']:
            active = None
            continue
        key = tuple(o['suggested_window'])
        if active is None or active['window'] != list(key):
            active = dict(window=list(key), start_s=r['feedback_time_s'], end_s=r['feedback_time_s'], samples=0)
            result['spans'].append(active)
        active['end_s'], active['samples'] = r['feedback_time_s'], active['samples']+1
    for key, getter in [('crest_residual', lambda r: r['observation'].get('filtered_feature_error', [np.nan]*3)[0]),
                        ('bending_residual', lambda r: r['observation'].get('filtered_feature_error', [np.nan]*3)[1]),
                        ('boundary_residual_deg', lambda r: r['observation'].get('filtered_feature_error', [np.nan]*3)[2])]:
        values = np.array([getter(r) for r in rows])
        values = values[np.isfinite(values)]
        result[key] = None if not len(values) else dict(minimum=float(values.min()), maximum=float(values.max()),
                                                      absolute_p90=float(np.percentile(abs(values), 90)))
    return result


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    whole = load(SOURCE / 'whole_body_observer_trace.jsonl.gz')
    reference = build_reference(whole)
    REFERENCE.write_text(json.dumps(reference.to_dict(), ensure_ascii=False, indent=2), encoding='utf8')
    all_summaries = {}
    for label in ('whole_body', 'endpoint'):
        original = whole if label == 'whole_body' else load(SOURCE / 'endpoint_observer_trace.jsonl.gz')
        observer, derived = MotorShapeObserver([.0645]*21, reference), []
        for row in original:
            output = observer.update(**row['input'])
            derived.append(dict(feedback_time_s=row['feedback_time_s'], stage=row['stage'], input=row['input'],
                                observation=output, evaluation=row['evaluation']))
        with gzip.open(REPORT / (label+'_motor_observer.jsonl.gz'), 'wt', encoding='utf8') as stream:
            for row in derived:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        holdout = [r for r in derived if r['stage'] == 'no_force' and
                   r['input']['phase_rad'] >= reference.metadata['start_phase_rad']+2*math.pi]
        all_summaries[label] = dict(stages={s: summary([r for r in derived if r['stage'] == s])
            for s in ('no_force', 'right_force', 'recovery')}, normal_holdout=summary(holdout))
    result = dict(frozen_reference=reference.to_dict(), offline_passes=2, new_dynamics_runs=0,
                  results=all_summaries)
    (REPORT / 'offline_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: {s: dict(samples=v['samples'], valid=v['valid_samples'], suggested=v['suggested_samples'],
                                spans=v['spans']) for s, v in value['stages'].items()}
                      for k, value in all_summaries.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
