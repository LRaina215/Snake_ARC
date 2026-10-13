"""Read-only check of the general observer on existing measured feedback."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import math
import numpy as np
from snake_control.simulation import ROOT
from snake_control.posture_window import GeneralPhaseReference, mechanically_feasible, nominal_angles
from snake_control.rolling_shape_observer import RollingShapeObserver
from snake_control.motor_shape_observer import MotorObserverOptions
from validation.posture_window_experiment import REPORT, references


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    refs, dimensions = references(), [.0645]*21
    mechanical = []
    for width in (5, 6, 7, 8, 9, 12, 18):
        window = (2, width+1)
        mechanical.append(dict(width=width, feasible=mechanically_feasible(dimensions, window),
                               peak_joint_deg=float(np.degrees(np.maximum(abs(nominal_angles(dimensions, window, 0.)),
                                                                            abs(nominal_angles(dimensions, window, math.pi/2)))).max())))
    sources = ['natural_repair_20261013_113711_001298', 'combined_repair_20261013_120622_382379']
    result = []
    for name in sources:
        path = ROOT/'validation/reports/rolling_handover'/name
        last_feedback, window, observer = None, None, None
        widths = {}
        with gzip.open(path/'actual_trace.jsonl.gz', 'rt', encoding='utf8') as stream:
            for line in stream:
                row = json.loads(line); f = row['motor_input']
                if not f or f['feedback_time_s'] < 2. or f['feedback_time_s'] == last_feedback:
                    continue
                last_feedback = f['feedback_time_s']
                active = np.flatnonzero(f['enabled'])+1
                current = (int(active[0]), int(active[-1]))
                width = len(active)
                if current != window:
                    window = current
                    observer = RollingShapeObserver(dimensions, GeneralPhaseReference(dimensions, width, refs), width=width,
                                                     options=MotorObserverOptions(maximum_feedback_gap_s=.5))
                o = observer.update(f['angles_deg'], f['phase_rad'], f['enabled'], f['feedback_time_s'])
                data = widths.setdefault(width, dict(samples=0, valid=0, trusted_down=0, recommendations=0, no_reference=0))
                data['samples'] += 1; data['valid'] += bool(o['valid']); data['trusted_down'] += 'down_head' in o
                data['recommendations'] += bool(o['direction']); data['no_reference'] += not o.get('position_valid', False)
        result.append(dict(source=name, by_width=widths, execution='observation only; not a new motion'))
    output = dict(mechanical=mechanical, feedback_checks=result, physics_steps=0, hardware_runs=0)
    (REPORT/'preflight.json').write_text(json.dumps(output, indent=2), encoding='utf8')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
