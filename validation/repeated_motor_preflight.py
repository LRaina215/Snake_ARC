"""Replay previous raw angles through the split-clock core; no dynamics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import numpy as np
from snake_control.rolling_arc import ArcParameters
from snake_control.motor_shape_observer import PhaseShapeReference
from snake_control.motor_feedback_sliding import MotorFeedbackSliding

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'validation/reports/repeated_motor_sliding'
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT/'validation/reports/motor_feedback_sliding'
    reference = PhaseShapeReference.from_dict(json.loads((ROOT/'configs/motor_shape_reference.json').read_text(encoding='utf8')))
    result = {}
    for path in sorted(list(source.glob('observe_*'))+list(source.glob('closed_*'))):
        status = json.loads((path/'status.json').read_text(encoding='utf8'))
        c = MotorFeedbackSliding(ArcParameters(**status['parameters']), [.0645]*21, reference,
                                  execute=status['closed_loop'])
        max_target, max_phase, count, masks = 0., 0., 0, 0
        with gzip.open(path/'actual_trace.jsonl.gz', 'rt', encoding='utf8') as stream:
            for row in map(json.loads, stream):
                f = row['motor_input']
                if f is None:
                    continue
                target, enabled = c.update(f['angles_deg'], f['feedback_time_s'])
                count += 1
                max_target = max(max_target, float(np.max(abs(target-row['target_rad']))))
                max_phase = max(max_phase, abs(c.phase-row['controller']['phase_rad']))
                masks += int(not np.array_equal(enabled, row['enabled']))
        events_match = [(e['kind'], e['time_s']) for e in c.events] == [(e['kind'], e['time_s']) for e in status['events']]
        result[path.name] = dict(fresh_samples=count, maximum_target_error_rad=max_target,
            maximum_phase_error_rad=max_phase, mask_mismatches=masks, events_match=events_match,
            final_observer_state=c.observation['state'], passed=bool(max_target < 1e-8 and max_phase < 1e-8 and masks == 0 and events_match))
    (output/'prior_core_replay.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))
    if not all(r['passed'] for r in result.values()):
        raise SystemExit('Prior input regression changed decisions; inspect before new dynamics')


if __name__ == '__main__':
    main()
