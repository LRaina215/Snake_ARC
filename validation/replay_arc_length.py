"""Reproduce all length-controller outputs from archived motor input only."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import numpy as np
from snake_control.adaptive_arc_length import HalfArcParameters, AdaptiveArcLength
from snake_control.motor_shape_observer import PhaseShapeReference
from snake_control.simulation import ROOT

REPORT = ROOT/'validation/reports/adaptive_arc_length'


def main():
    references = {n: PhaseShapeReference.from_dict(json.loads((ROOT/'configs'/
        ('motor_shape_reference.json' if n == 6 else 'motor_shape_reference8.json')).read_text())) for n in (6, 8)}
    results = []
    for path in sorted(p for p in REPORT.iterdir() if p.is_dir() and (p/'status.json').exists()):
        status = json.loads((path/'status.json').read_text())
        # A live run is not replayed until its immutable archive is complete.
        if status['status'] != 'complete':
            continue
        case = status['case']
        refs = references if case in ('adaptive', 'combined') else {6: references[6]}
        c = AdaptiveArcLength(HalfArcParameters(**status['parameters']), [.0645]*21, refs,
            resize=case in ('adaptive', 'combined'), sliding=case == 'combined',
            feasibility_trial=status['final_controller']['feasibility_trial'])
        target_error, phase_error, masks, frames = 0., 0., 0, 0
        with gzip.open(path/'actual_trace.jsonl.gz', 'rt', encoding='utf8') as stream:
            for row in map(json.loads, stream):
                f = row['motor_input']
                if f is None:
                    continue
                target, enabled = c.advance(f['angles_deg'], f['feedback_time_s'], row['controller']['control_time_s'],
                                            phase_at_feedback=f['phase_rad'])
                target_error = max(target_error, float(abs(target-row['target_rad']).max()))
                phase_error = max(phase_error, abs(c.phase-row['controller']['phase_rad']))
                masks += int(not np.array_equal(enabled, row['enabled']))
                frames += 1
        signature = lambda events: [(e['kind'], round(e['time_s'], 6), e.get('to')) for e in events]
        events_match = signature(c.events) == signature(status['events'])
        results.append(dict(source=path.name, frames=frames, maximum_target_error_rad=target_error,
            maximum_phase_error_rad=phase_error, mask_mismatches=masks, events_match=events_match,
            passed=bool(target_error < 1e-9 and phase_error < 1e-9 and masks == 0 and events_match)))
    (REPORT/'input_replay.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    print(json.dumps(results))
    if not all(r['passed'] for r in results):
        raise SystemExit('Archived outputs changed; inspect before claiming validation')


if __name__ == '__main__':
    main()
