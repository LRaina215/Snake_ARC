"""Diagnose the existing count oscillation using identical actual angles."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import math
import numpy as np
from snake_control.simulation import ROOT
from snake_control.posture_window import nominal_angles, PostureWindowOptions
from snake_control.section_geometry import local_section
from snake_control.adaptive_arc_length import circle_shape
from snake_control.phase_balanced_count import PhaseBalancedCount

REPORT = ROOT/'validation/reports/phase_balanced_window'


def check(path):
    seen, last_window, ema = set(), None, None
    history, records = PhaseBalancedCount(), []
    with gzip.open(path/'actual_trace.jsonl.gz', 'rt', encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line); f = row['motor_input']
            if not f or f['feedback_time_s'] < 2. or f['feedback_time_s'] in seen:
                continue
            seen.add(f['feedback_time_s'])
            indices = np.flatnonzero(f['enabled'])+1
            window = (int(indices[0]), int(indices[-1])); width = len(indices)
            if window != last_window:
                history.reset(); last_window, ema = window, None
            q = np.radians(f['angles_deg'])
            actual = circle_shape(local_section(q, [.0645]*21, *window))
            nominal = q.copy(); nominal[indices-1] = nominal_angles([.0645]*21, window, f['phase_rad'])[indices-1]
            reference = circle_shape(local_section(nominal, [.0645]*21, *window))
            raw = width*.5*(actual['radius_m']/reference['radius_m']+reference['turn_deg']/actual['turn_deg'])
            ema = raw if ema is None else ema+(1-math.exp(-.2/.5))*(raw-ema)
            stats = history.update(raw, actual['turn_deg'], f['phase_rad'], f['feedback_time_s'], reference['turn_deg'])
            records.append(dict(time_s=f['feedback_time_s'], width=width, phase_rad=f['phase_rad'],
                                raw_count=raw, old_ema=ema, phase_statistics=stats,
                                actual_body_circle=actual, nominal_body_circle=reference))
    summary = []
    for width in sorted({r['width'] for r in records}):
        subset = [r for r in records if r['width'] == width and r['phase_statistics']['ready']]
        if not subset: continue
        summary.append(dict(width=width, comparable_ready_samples=len(subset),
            raw_range=[min(r['raw_count'] for r in subset), max(r['raw_count'] for r in subset)],
            ema_range=[min(r['old_ema'] for r in subset), max(r['old_ema'] for r in subset)],
            balanced_range=[min(r['phase_statistics']['balanced_count_joints'] for r in subset),
                            max(r['phase_statistics']['balanced_count_joints'] for r in subset)],
            expands=sum(r['phase_statistics']['balanced_count_joints'] >= width+PostureWindowOptions().count_hysteresis_joints and
                        r['phase_statistics']['balanced_turn_error_deg'] <= PostureWindowOptions().expansion_turn_error_deg for r in subset),
            shorter_trial_evidence=sum(r['phase_statistics']['maximum_group_count_joints'] <= width+.25 and
                                       r['phase_statistics']['balanced_turn_error_deg'] >= PostureWindowOptions().shorter_trial_turn_error_deg for r in subset)))
    with gzip.open(REPORT/(path.name+'_count_replay.jsonl.gz'), 'wt', encoding='utf8') as stream:
        for record in records: stream.write(json.dumps(record)+'\n')
    return dict(source=path.name, feedbacks=len(records), by_width=summary, physics_steps=0,
                interpretation='same actual angle inputs; does not simulate a changed mask response')


if __name__ == '__main__':
    REPORT.mkdir(parents=True, exist_ok=True)
    sources = [ROOT/'validation/reports/posture_window/acute_20261013_134937_702758',
               ROOT/'validation/reports/adaptive_arc_length/adaptive_20261013_081901_624965']
    result = [check(p) for p in sources]
    (REPORT/'count_preflight.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))
