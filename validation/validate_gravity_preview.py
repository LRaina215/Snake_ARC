"""Display-only gravity evaluation. World truth is accessed AFTER observer output."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import math
from collections import Counter
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.gravity_preview import DualEndGravityPreview, gravity_angle
from validation.angle_observer_replay import CANONICAL_TO_MODEL

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'validation/reports/gravity_preview'


def evaluate(path):
    status = json.loads((path/'status.json').read_text(encoding='utf8'))
    with gzip.open(path/'actual_trace.jsonl.gz', 'rt', encoding='utf8') as f:
        rows = list(map(json.loads, f))
    actual = {round(r['time_s'], 8): r for r in rows}
    observer = DualEndGravityPreview([.0645]*21)
    seen, results, previous = set(), [], None
    for row in rows:
        f = row.get('motor_input')
        if not f or f['feedback_time_s'] in seen:
            continue
        t = f['feedback_time_s']; seen.add(t)
        origin = actual[round(t, 8)]
        # Use the window at acquisition, never evaluation contacts/world position.
        window = origin['controller']['window']
        out = observer.update(f['angles_deg'], window, t)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, np.array(origin['actual_qpos'][3:7]))
        rotation = rotation.reshape(3, 3)@CANONICAL_TO_MODEL
        error = step = raw_error = raw_step = None
        world = raw_world = None
        if out['down_head'] is not None:
            world = rotation@out['down_head']
            error = gravity_angle(world, [0, 0, -1])
            if previous is not None and previous['world'] is not None:
                step = gravity_angle(world, previous['world'])
            basis = np.array(out['basis'])
            if not np.allclose(basis@basis.T, np.eye(3), atol=1e-9) or np.linalg.det(basis) < .999999:
                raise AssertionError('Display transform must preserve physical link lengths')
        if out['raw_down_head'] is not None:
            raw_world = rotation@out['raw_down_head']
            raw_error = gravity_angle(raw_world, [0, 0, -1])
            if previous is not None and previous['raw_world'] is not None:
                raw_step = gravity_angle(raw_world, previous['raw_world'])
        individual = {name: gravity_angle(rotation@c['direction'], [0, 0, -1]) for name, c in out.get('candidates', {}).items()}
        results.append(dict(time_s=t, estimate=out, truth_error_deg=error, world_step_deg=step,
                            raw_error_deg=raw_error, raw_step_deg=raw_step, individual_error_deg=individual))
        previous = dict(world=world, raw_world=raw_world)
    def stats(values):
        a = np.array([v for v in values if v is not None])
        return None if not len(a) else dict(samples=len(a), median=float(np.median(a)), p95=float(np.percentile(a, 95)), maximum=float(a.max()))
    dual = [r for r in results if r['time_s'] >= 2. and r['estimate']['state'] == 'dual']
    summary = dict(source=path.name, feedbacks=len(results), states=dict(Counter(r['estimate']['state'] for r in results)),
        dual_error_deg=stats([r['truth_error_deg'] for r in dual]),
        all_error_deg=stats([r['truth_error_deg'] for r in results]),
        world_step_deg=stats([r['world_step_deg'] for r in results]),
        raw_world_step_deg=stats([r['raw_step_deg'] for r in results]),
        raw_dual_error_deg=stats([r['raw_error_deg'] for r in dual]),
        no_flip=all(r['world_step_deg'] is None or r['world_step_deg'] < 90. for r in results),
        last_state=results[-1])
    summary['passed'] = bool(summary['no_flip'] and summary['dual_error_deg'] and summary['dual_error_deg']['p95'] <= 12.)
    interval = status.get('stimulation', {}).get('interval_s')
    if interval:
        start, end = interval
        groups = {'before_pull': [r for r in results if 2. <= round(r['time_s'], 8) < start],
                  'during_pull': [r for r in results if start <= round(r['time_s'], 8) < end],
                  'after_pull': [r for r in results if round(r['time_s'], 8) >= end]}
        summary['segments'] = {k: stats([r['truth_error_deg'] for r in v]) for k, v in groups.items()}
        summary['full_run_dual_p95_under_12'] = summary['passed']
        summary['passed'] = bool(summary['no_flip'] and summary['world_step_deg']['maximum'] < 15.
            and summary['segments']['before_pull']['p95'] <= 12. and summary['segments']['after_pull']['p95'] <= 12.)
        summary['acceptance_scope'] = 'Frozen normal accuracy and no-flip/recovery test; no 12deg promise while externally pulled'
    REPORT.mkdir(parents=True, exist_ok=True)
    with gzip.open(REPORT/(path.name+'_gravity.jsonl.gz'), 'wt', encoding='utf8') as f:
        for r in results: f.write(json.dumps(r)+'\n')
    (REPORT/(path.name+'_review.json')).write_text(json.dumps(summary, indent=2), encoding='utf8')
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = [r['time_s'] for r in results]
    axes[0].plot(t, [r['truth_error_deg'] for r in results], label='History + dual ends')
    axes[0].plot(t, [r['raw_error_deg'] for r in results], alpha=.5, label='Instantaneous dual-end fusion')
    axes[0].axhline(12, color='grey', ls=':'); axes[0].set_ylabel('Error to true down / deg'); axes[0].legend()
    axes[1].plot(t, [r['world_step_deg'] for r in results], label='Filtered')
    axes[1].plot(t, [r['raw_step_deg'] for r in results], alpha=.5, label='Instantaneous')
    axes[1].set(xlabel='Actual feedback time / s', ylabel='World direction step / deg'); axes[1].legend()
    fig.suptitle(path.name+' | truth used only for evaluation')
    fig.tight_layout(); fig.savefig(REPORT/(path.name+'_timeline.png'), dpi=120); plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != 'last_state'}), flush=True)
    return summary


if __name__ == '__main__':
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else next((ROOT/'validation/reports/phase_balanced_window').glob('return_repair_*'))
    evaluate(source)
