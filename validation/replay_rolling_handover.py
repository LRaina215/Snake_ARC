"""Replay each archived strategy version using angles/timestamps/phase only."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import importlib.util
import json
import numpy as np
from snake_control.simulation import ROOT
from snake_control.adaptive_arc_length import HalfArcParameters
from snake_control.motor_shape_observer import PhaseShapeReference
import snake_control.rolling_handover as current

REPORT = ROOT/'validation/reports/rolling_handover'


def recorded_module(name, path):
    spec = importlib.util.spec_from_file_location('snake_control.'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay(folder):
    status = json.loads((folder/'status.json').read_text(encoding='utf8'))
    refs = {w: PhaseShapeReference.from_dict(json.loads((ROOT/f'configs/motor_shape_reference{suffix}.json').read_text()))
            for w, suffix in ((6, ''), (8, '8'))}
    cls = current.RollingHandover
    original_observer = current.RollingShapeObserver
    if folder.name == 'natural_20261013_112105_172339':
        current.RollingShapeObserver = recorded_module('recorded_intrinsic', REPORT/'intrinsic_only_version/rolling_shape_observer.py').RollingShapeObserver
    elif status['case'] == 'combined' and not status['repair']:
        cls = recorded_module('recorded_trial', REPORT/'before_trial_fix/rolling_handover.py').RollingHandover
    c = cls(HalfArcParameters(**status['parameters']), [.0645]*21, refs, resize=status['case'] == 'combined',
            sliding=True, feedback_period_s=.2)
    frames, target_error, phase_error, mask_errors = 0, 0., 0., 0
    try:
        with gzip.open(folder/'actual_trace.jsonl.gz', 'rt', encoding='utf8') as f:
            for line in f:
                row = json.loads(line); incoming = row.get('motor_input')
                if not incoming: continue
                target, mask = c.advance(incoming['angles_deg'], incoming['feedback_time_s'], row['controller']['control_time_s'],
                                         phase_at_feedback=incoming['phase_rad'])
                target_error = max(target_error, float(abs(target-row['target_rad']).max()))
                phase_error = max(phase_error, abs(c.phase-row['controller']['phase_rad']))
                mask_errors += int(not np.array_equal(mask, row['enabled'])); frames += 1
        events = lambda items: [(e['kind'], e['time_s'], e.get('old'), e.get('to')) for e in items]
        event_match = events(c.events) == events(status['events'])
    finally:
        current.RollingShapeObserver = original_observer
    return dict(source=folder.name, frames=frames, maximum_target_error_rad=target_error,
        maximum_phase_error_rad=phase_error, mask_mismatches=mask_errors, command_event_sequence_match=event_match,
        passed=bool(target_error < 1e-10 and phase_error < 1e-10 and not mask_errors and event_match),
        input_fields=['angles_deg', 'feedback_time_s', 'phase_rad', 'control_time_s'])


if __name__ == '__main__':
    results = []
    for folder in sorted(p for p in REPORT.iterdir() if p.is_dir() and (p/'status.json').exists()):
        result = replay(folder); results.append(result); print(json.dumps(result), flush=True)
    (REPORT/'input_replay.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    if not all(r['passed'] for r in results): raise SystemExit(1)
