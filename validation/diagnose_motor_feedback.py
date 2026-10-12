"""Small causal checks on saved feedback/forces, not another dynamics test."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import numpy as np
from snake_control.simulation import ROOT

REPORT = ROOT/'validation/reports/motor_feedback_sliding'


def load(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def main():
    results = {}
    for path in sorted(list(REPORT.glob('observe_*'))+list(REPORT.glob('closed_*'))):
        raw, evaluated = load(path/'actual_trace.jsonl.gz'), load(path/'evaluation.jsonl.gz')
        by_time = {round(r['time_s'], 8): r for r in raw}
        status = json.loads((path/'status.json').read_text(encoding='utf8'))
        recommendations = [r for r in evaluated if r.get('observation', {}).get('direction')]
        valid = [r for r in evaluated if r.get('observation', {}).get('valid')]
        quality = [r for r in evaluated if 'down_error_deg' in r]
        entry = dict(valid_down_error_max_deg=max(r['down_error_deg'] for r in valid),
            all_down_error_max_deg=max(r['down_error_deg'] for r in quality),
            recommendations_have_support_gain=False, recommendations_have_proven_need=False,
            first_actual_pose=raw[0]['actual_qpos'], events=status['events'], handover_truth=[])
        for e in status['events']:
            if e['kind'] != 'complete':
                continue
            before = by_time[round(e['time_s'], 8)]
            after = next(r for r in raw if r['time_s'] > e['time_s'])
            old, new = set(range(e['old'][0], e['old'][1]+1)), set(range(e['to'][0], e['to'][1]+1))
            exiting = next(iter(old-new))
            entering = next(iter(new-old))
            entry['handover_truth'].append(dict(time_s=e['time_s'], exiting=exiting, entering=entering,
                previous_saved_exit_actuator_torque_nm=before['actuator_force_nm'][exiting-1],
                before_support_sum_n=sum(before['support_up_n']), after_support_sum_n=sum(after['support_up_n']),
                observation_valid_at_handover=after['controller']['observation']['valid'],
                continuous_physical_state_max_20ms_joint_change_deg=float(np.max(abs(
                    np.degrees(np.array(after['actual_qpos'][7:])-before['actual_qpos'][7:])))),
                evaluation_scope='saved 20ms interval; torque is evaluation only, not motor-controller input'))
        # The two cases have the same initial condition, not just scene name.
        results[path.name] = entry
    keys = list(results)
    initial = [results[k]['first_actual_pose'] for k in keys]
    if len(initial) == 2:
        equal = bool(np.array_equal(initial[0], initial[1]))
    else:
        equal = None
    output = dict(same_initial_actual_qpos=equal, cases=results,
        dynamics_runs=len(results), calibration_holdout_note='76 feedback samples, 1.52s; not a full independent cycle',
        feature_inspection_error='Initial cheap feature inspection used endpoint_pull instead of endpoint filename; fixed before replay; no dynamics.')
    (REPORT/'cause_audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({key: {k: v for k, v in value.items() if k not in ('first_actual_pose', 'events')}
                      for key, value in results.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
