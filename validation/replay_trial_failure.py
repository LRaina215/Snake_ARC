"""Reproduce the old acute-bend prefix through its false trial failure."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import numpy as np
from snake_control.simulation import ROOT
from snake_control.adaptive_arc_length import HalfArcParameters
from snake_control.motor_shape_observer import PhaseShapeReference
from snake_control.rolling_handover import RollingHandover

REPORT = ROOT/'validation/reports/rolling_handover'


def main():
    folder = REPORT/'combined_20261013_115002_906774'
    status = json.loads((folder/'status.json').read_text())
    refs = {w: PhaseShapeReference.from_dict(json.loads((ROOT/f'configs/motor_shape_reference{suffix}.json').read_text()))
            for w, suffix in ((6, ''), (8, '8'))}
    c = RollingHandover(HalfArcParameters(**status['parameters']), [.0645]*21, refs, resize=True,
                        sliding=True, feedback_period_s=.2)
    old_failure = next(e for e in status['events'] if e['kind'] == 'six_feasibility_review' and not e['accepted'])
    target_error, phase_error, mask_errors, frames = 0., 0., 0, 0
    last_processed_time = None
    with gzip.open(folder/'actual_trace.jsonl.gz', 'rt') as f:
        for line in f:
            r = json.loads(line); incoming = r.get('motor_input')
            if not incoming: continue
            t = r['controller']['control_time_s']
            if t > old_failure['time_s']+1e-8: break
            target, enabled = c.advance(incoming['angles_deg'], incoming['feedback_time_s'], t,
                                       phase_at_feedback=incoming['phase_rad'])
            target_error = max(target_error, float(abs(target-r['target_rad']).max()))
            phase_error = max(phase_error, abs(c.phase-r['controller']['phase_rad']))
            mask_errors += int(not np.array_equal(enabled, r['enabled'])); frames += 1
            last_processed_time = t
    result = dict(frames=frames, through_control_time_s=last_processed_time, old_failure=old_failure,
        maximum_target_error_rad=target_error, maximum_phase_error_rad=phase_error, mask_mismatches=mask_errors,
        new_resize_fault=c.resize_fault, new_six_trial=c.six_trial,
        dynamics_after_old_failure_verified=False,
        passed=bool(target_error < 1e-10 and phase_error < 1e-10 and not mask_errors and c.resize_fault is None and c.six_trial is not None))
    (REPORT/'trial_failure_prefix_replay.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
