"""Audit all sampled actual states, mask commands, phase and exit cycle peaks."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import gzip
import json
import numpy as np
from snake_control.simulation import ROOT


def audit(label, path):
    report = json.loads((path/f'{label}.json').read_text(encoding='utf-8'))
    compressed = path/f'{label}_records.json.gz'
    if compressed.exists():
        with gzip.open(compressed, 'rt', encoding='utf-8') as stream:
            rows = json.load(stream)
    else:
        rows = json.loads((path/f'{label}_records.json').read_text(encoding='utf-8'))
    audited = [r for r in rows if r['time_s'] >= 3]
    with np.load(path/f'{label}_trace.npz') as trace:
        enabled = trace['enabled'].astype(bool)
        weights = trace['weights']
        forces = trace['actuator_force_nm']
        t = trace['time_s']
        phase = np.array([r['phase_rad'] for r in rows])
        dq = np.diff(phase)
        phase_continuous = bool(np.all(dq >= -1e-10) and np.all(dq <= .6*np.diff(t)+1e-8))
        inactive_force = float(np.max(abs(np.where(enabled, 0., forces))))
        enabled_count = sorted(set(enabled.sum(axis=1).tolist()))
        # Existing baseline traces are checked by the same diagnostics.
        weight_consistent = bool(np.all(enabled == (weights > 1e-12)))
        cycle_rows = []
        start = report['exit_time_s']
        if start is None:
            completions = [e['time_s'] for e in report['events'] if e['kind'] == 'complete']
            start = completions[-1] if completions else 3.
        start_index = int(np.searchsorted(t, start))
        start_phase = phase[min(start_index, len(phase)-1)]
        for lower in np.arange(start_phase, phase[-1]-2*np.pi+1e-6, 2*np.pi):
            selected = [r for r in rows if lower <= r['phase_rad'] < lower+2*np.pi and r['time_s'] >= start]
            if selected:
                observable = [r['shape']['wrap_deg'] for r in selected if r['shape']['wrap_valid']]
                cycle_rows.append(dict(start_s=selected[0]['time_s'], start_phase_rad=float(lower), body_peak_deg=max(r['body_geometry']['turn_deg'] for r in selected),
                                       pipe_peak_deg=max(observable) if observable else None,
                                       gap_min=min(r['body_geometry']['gap_fraction'] for r in selected)))
        worst = max(audited, key=lambda r: r['body_geometry']['turn_deg'])
        smallest = min(audited, key=lambda r: r['body_geometry']['turn_deg'])
        gap_worst = min(audited, key=lambda r: r['body_geometry']['gap_fraction'])
        result = dict(label=label, full_saved_samples=len(rows), enabled_counts=enabled_count,
                      inactive_force_max_nm=inactive_force, mask_matches_weights=weight_consistent,
                      phase_continuous=phase_continuous, post_event_cycles=cycle_rows,
                      post_event_phase_cycles=float((phase[-1]-start_phase)/(2*np.pi)),
                      maximum_body_turn_at_s=worst['time_s'], minimum_body_turn_at_s=smallest['time_s'],
                      minimum_gap_at_s=gap_worst['time_s'],
                      body_failed_samples=sum(not (150 <= r['body_geometry']['turn_deg'] <= 218 and
                                                  r['body_geometry']['gap_fraction'] >= .4 and
                                                  r['body_geometry']['fit_rms_m'] <= .015) for r in audited),
                      pipe_failed_samples=sum(r['shape']['wrap_valid'] and not 150 <= r['shape']['wrap_deg'] <= 218 for r in audited),
                      pipe_unobservable_samples=sum(not r['shape']['wrap_valid'] for r in audited),
                      sent_failed_samples=sum(r['sent_geometry']['turn_deg'] > 190.01 or r['sent_geometry']['gap_fraction'] < .4 for r in audited))
        (path/f'{label}_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--path', type=Path, default=ROOT/'validation/reports/sliding_section')
    args = parser.parse_args()
    for label in args.labels:
        audit(label, args.path)
