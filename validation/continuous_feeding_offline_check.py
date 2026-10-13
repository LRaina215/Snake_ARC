"""Existing angle records only: estimator check and unchanged valid-path replay."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gzip
import json
import numpy as np
from snake_control.continuous_feeding import ContinuousFeeding, ContinuousFeedingOptions
from snake_control.feeding_length_observer import FeedingLengthObserver

report = ROOT/'validation/reports/continuous_feeding'
results = dict(hardware_run=False, dynamics_run=False, observer_existing_record_checks=[])
observer = FeedingLengthObserver(np.full(21, .0645))
for file in sorted((ROOT/'validation/reports/in_place_feeding').glob('*/actual_trace.jsonl.gz')):
    rows = [json.loads(line) for line in gzip.open(file, 'rt', encoding='utf8')]
    comparisons = []
    for r in rows[::10]:
        if r['time_s'] < 6.:
            continue
        o = observer.observe(r['actual_angles_deg'], tuple(r['controller']['window']))
        m = r['evaluation']['material']
        if o['valid'] and m['valid']:
            comparisons.append([o['transfer_coordinate_m'], .5*(m['head_hanging_m']-m['tail_hanging_m'])])
    a = np.array(comparisons)
    errors = ((a[:, 0]-a[0, 0])-(a[:, 1]-a[0, 1]))*1000
    results['observer_existing_record_checks'].append(dict(record=file.parent.name, samples=len(a),
        delta_error_rmse_mm=float(np.sqrt(np.mean(errors**2))), delta_error_maximum_mm=float(max(abs(errors))),
        input='actual angles resampled 0.2s; no world inputs'))

status_file = next(report.glob('small_*/status.json'))
status = json.loads(status_file.read_text(encoding='utf8'))
controller = ContinuousFeeding(np.full(21, .0645), ContinuousFeedingOptions(**status['options']))
maximum = 0.; mismatch = 0; samples = 0
for line in gzip.open(status_file.parent/'actual_trace.jsonl.gz', 'rt', encoding='utf8'):
    row = json.loads(line)
    feedback = row['motor_input']
    target, enabled = controller.advance(feedback['angles_deg'], feedback['feedback_time_s'], row['time_s']-.02)
    maximum = max(maximum, float(np.degrees(abs(target-np.array(row['target_rad']))).max()))
    mismatch += int(not np.array_equal(enabled, row['enabled']))
    samples += 1
results['small_valid_path_replay'] = dict(record=status_file.parent.name, samples=samples,
    target_maximum_error_deg=maximum, mask_mismatch_frames=mismatch,
    passed=maximum<1e-8 and mismatch==0, new_physics=False)
(report/'offline_checks.json').write_text(json.dumps(results, indent=2), encoding='utf8')
print(json.dumps(results, indent=2), flush=True)
