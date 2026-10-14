"""Replay final controller inputs from existing records; no new dynamics.

Historical methods must use their corresponding frozen source. The default
round contains the current controller, not the obsolete small-transfer case.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from dataclasses import replace
import gzip
import json
import numpy as np
from snake_control.continuous_feeding import ContinuousFeeding, ContinuousFeedingOptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-root', type=Path,
        default=ROOT/'validation/reports/active_feeding_continuation/round_09')
    parser.add_argument('--small-equivalence-record', type=Path,
        default=ROOT/'validation/reports/active_feeding_continuation/round_07/small_20261014_140308/status.json')
    args = parser.parse_args()
    results = dict(hardware_run=False, new_physics=False, replay=[])
    files = sorted(args.report_root.glob('*/status.json'))
    files.append(args.small_equivalence_record)
    for file in files:
        status = json.loads(file.read_text(encoding='utf8'))
        options = ContinuousFeedingOptions(**status['options'])
        equivalent_small = file.resolve()==args.small_equivalence_record.resolve()
        if equivalent_small:
            current = ContinuousFeedingOptions()
            options = replace(options, fractional_middle_boost_deg=current.fractional_middle_boost_deg,
                middle_boost_activation_joints=current.middle_boost_activation_joints,
                middle_boost_ramp_joints=current.middle_boost_ramp_joints)
        controller = ContinuousFeeding(np.full(21, .0645), options)
        maximum = 0.; mismatch = samples = 0
        for line in gzip.open(file.parent/'actual_trace.jsonl.gz', 'rt', encoding='utf8'):
            row = json.loads(line)
            feedback = row['motor_input']
            if set(feedback) != {'angles_deg', 'feedback_time_s'}:
                raise ValueError('Replay must contain only motor angles/timestamp')
            target, enabled = controller.advance(feedback['angles_deg'], feedback['feedback_time_s'],
                                                 row['time_s']-status['control_period_s'])
            maximum = max(maximum, float(np.degrees(abs(target-np.array(row['target_rad']))).max()))
            mismatch += int(not np.array_equal(enabled, row['enabled']))
            samples += 1
        results['replay'].append(dict(record=file.parent.name, samples=samples,
            target_maximum_error_deg=maximum, mask_mismatch_frames=mismatch,
            small_allocation_replayed_with_current_default=equivalent_small,
            passed=maximum<1e-8 and mismatch==0,
            inputs='recorded motor angles, motor timestamp, control timestamp'))
    results['all_passed'] = bool(results['replay']) and all(r['passed'] for r in results['replay'])
    (args.report_root/'offline_checks.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    print(json.dumps(results, indent=2), flush=True)
    if not results['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
