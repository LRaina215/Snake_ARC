"""Verify final controller against the two archived motor-input sequences.

Does not load MuJoCo or advance dynamics. Privileged actual states/forces
are not passed to the controller; only its saved input boundary is replayed.
"""
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
REPORT = ROOT/'validation/reports/motor_feedback_sliding'


def main():
    reference = PhaseShapeReference.from_dict(json.loads((ROOT/'configs/motor_shape_reference.json').read_text(encoding='utf8')))
    result = {}
    for path in sorted(list(REPORT.glob('observe_*'))+list(REPORT.glob('closed_*'))):
        status = json.loads((path/'status.json').read_text(encoding='utf8'))
        controller = MotorFeedbackSliding(ArcParameters(**status['parameters']), [.0645]*21,
                                          reference, execute=status['closed_loop'])
        count, target_error, phase_error, mask_mismatches = 0, 0., 0., 0
        with gzip.open(path/'actual_trace.jsonl.gz', 'rt', encoding='utf8') as stream:
            for row in map(json.loads, stream):
                motor = row['motor_input']
                if motor is None:
                    continue
                target, enabled = controller.update(motor['angles_deg'], motor['feedback_time_s'])
                count += 1
                target_error = max(target_error, float(np.max(abs(target-row['target_rad']))))
                phase_error = max(phase_error, abs(controller.phase-row['controller']['phase_rad']))
                mask_mismatches += int(not np.array_equal(enabled, row['enabled']))
        old_events = [(e['kind'], e['time_s']) for e in status['events']]
        events = [(e['kind'], e['time_s']) for e in controller.events]
        match = old_events == events
        result[path.name] = dict(fresh_feedbacks=count, maximum_target_error_rad=target_error,
            maximum_phase_error_rad=phase_error, enable_mask_mismatches=mask_mismatches,
            event_kinds_and_times_match=match, passed=bool(target_error < 1e-10 and phase_error < 1e-10 and
                                                         mask_mismatches == 0 and match))
    (REPORT/'final_core_replay.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))
    if not all(r['passed'] for r in result.values()):
        raise SystemExit('Final core does not reproduce archived commands; investigate without another dynamics run')


if __name__ == '__main__':
    main()
