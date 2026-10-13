"""Reproduce commands from archived motor inputs; zero dynamics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import numpy as np
from snake_control.adaptive_arc_length import HalfArcParameters
from snake_control.motor_shape_observer import PhaseShapeReference
from snake_control.posture_window import PostureWindow
from validation.posture_window_experiment import REPORT


def replay(folder):
    status = json.loads((folder/'status.json').read_text(encoding='utf8'))
    refs = {int(w): PhaseShapeReference.from_dict(r) for w, r in status['frozen_references'].items()}
    c = PostureWindow(HalfArcParameters(**status['parameters']), [.0645]*21, refs, feedback_period_s=.2)
    frames, target_error, phase_error, mask_errors = 0, 0., 0., 0
    with gzip.open(folder/'actual_trace.jsonl.gz', 'rt', encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line); incoming = row['motor_input']
            if not incoming:
                continue
            target, mask = c.advance(incoming['angles_deg'], incoming['feedback_time_s'],
                                     row['controller']['control_time_s'], phase_at_feedback=incoming['phase_rad'])
            target_error = max(target_error, float(abs(target-row['target_rad']).max()))
            phase_error = max(phase_error, abs(c.phase-row['controller']['phase_rad']))
            mask_errors += int(not np.array_equal(mask, row['enabled'])); frames += 1
    events = lambda items: [(e['kind'], e['time_s'], e.get('old'), e.get('to'), e.get('operation')) for e in items]
    event_match = events(c.events) == events(status['events'])
    return dict(source=folder.name, control_frames=frames, maximum_target_error_rad=target_error,
                maximum_phase_error_rad=phase_error, mask_mismatches=mask_errors, event_sequence_matches=event_match,
                input_fields=['angles_deg', 'feedback_time_s', 'phase_rad', 'control_time_s'], physics_steps=0,
                passed=bool(target_error < 1e-10 and phase_error < 1e-10 and mask_errors == 0 and event_match))


if __name__ == '__main__':
    result = [replay(p.parent) for p in sorted(REPORT.glob('*/status.json'))]
    (REPORT/'input_replay.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))
    if not all(r['passed'] for r in result):
        raise SystemExit(1)
