"""Summarize this bounded round and plot same-input count evidence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validation.phase_window_experiment import REPORT


def main():
    cases = []
    for path in sorted(REPORT.glob('*/status.json')):
        status = json.loads(path.read_text(encoding='utf8'))
        review = json.loads((path.parent/'review.json').read_text(encoding='utf8'))
        s = review['summary']
        cases.append(dict(source=path.parent.name, case=status['case'], repair=status['repair'],
            final_time_s=status['final_time_s'], physical_steps=status['statistics_1ms']['physical_steps'],
            saved_states=status['samples'], passed=review['passed'], shape_pass=review['shape_pass'],
            mechanics_pass=review['mechanics_pass'], strict_pipe_pass=review['strict_pipe_pass'],
            phase_continuity_pass=review['phase_continuity_pass'], body_turn_range_deg=s['body_turn_range_deg'],
            pipe_wrap_range_deg=s['pipe_wrap_range_deg'], pipe_violation_frames=s['pipe_violations'],
            pipe_unobservable_frames=s['pipe_unobservable'], count_chain=s['count_chain'],
            operation_counts=s['operation_counts'], post_exit_cycles=status['post_exit_cycles'],
            post_last_change_cycles=s['post_last_change_cycles']))
    source = REPORT/'acute_20261013_134937_702758_count_replay.jsonl.gz'
    records = [json.loads(l) for l in gzip.open(source, 'rt', encoding='utf8')]
    data = [r for r in records if r['width'] == 7 and r['phase_statistics']['ready']]
    t = [r['time_s'] for r in data]
    values = {'Instantaneous proxy': [r['raw_count'] for r in data],
              'Temporal EMA formula': [r['old_ema'] for r in data],
              'Equal phase groups': [r['phase_statistics']['balanced_count_joints'] for r in data]}
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for label, v in values.items(): ax.plot(t, v, label=label)
    ax.axhline(7., color='grey', ls=':', label='Current enabled count')
    ax.axhline(7.5, color='grey', ls='--', label='Adjacent expansion threshold')
    ax.set(xlabel='Feedback time / s', ylabel='Shape-based count proxy',
           title='Same archived actual angle feedback; no new dynamics')
    ax.grid(alpha=.2); ax.legend(loc='upper center', ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(REPORT/'count_phase_comparison.png', dpi=150); plt.close(fig)
    comparison = dict(source=source.name, comparable_samples=len(data),
                      ranges={name: [min(v), max(v)] for name, v in values.items()},
                      interpretation='decision smoothing, not a measured optimal joint count')
    summary = dict(dynamics_attempts=len(cases), budget_maximum=3,
        physical_steps=sum(c['physical_steps'] for c in cases), saved_actual_states=sum(c['saved_states'] for c in cases),
        replayed_control_frames=sum(c['saved_states']-1 for c in cases), hardware_runs=0,
        real_and_manual_entries_updated=False, all_requested_acceptance_complete=False,
        tests_and_repair_plan_executed=True, cases=cases, count_comparison=comparison)
    (REPORT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
