"""Read-only real-log initialization replay, with optional embedded-core check.

No bus, MuJoCo step or hardware connection. New commands are counterfactual;
the source log only contains the old controller's actual run.
"""
from pathlib import Path
import argparse
import gzip
import importlib.util
import json
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from snake_control.whole_feeding_observer import WholeFeedingObserver
from snake_control.continuous_feeding import ContinuousFeeding, ContinuousFeedingOptions

REPORT = ROOT/'validation/reports/passive_segment_selection_20261021'
LOG = ROOT/'codes/logs/rolling_arc_20261021_123930_695019.jsonl'


def load_baseline():
    path = REPORT/'baseline_observer.py'
    if not path.exists():
        path.write_bytes(subprocess.check_output(['git','show','HEAD:snake_control/whole_feeding_observer.py'],cwd=ROOT))
    spec = importlib.util.spec_from_file_location('snake_control._passive_baseline', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WholeFeedingObserver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync-check', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    old_type = load_baseline()
    archive = REPORT/'actual_log.jsonl.gz'
    if not archive.exists():
        with LOG.open('rb') as src, gzip.open(archive, 'wb') as out:
            shutil.copyfileobj(src, out)
    groups = {}
    with gzip.open(archive, 'rt', encoding='utf8') as src:
        for line in src:
            row = json.loads(line)
            inp = row.get('feeding_motor_input')
            if inp and row.get('feeding') is not None:
                task = row['feeding_action_id']
                # A held row can repeat the previous command input while
                # manual metadata has already changed. Keep its first capture.
                groups.setdefault(task, {}).setdefault(inp['control_time_s'], row)
    embedded_type = None
    if args.sync_check:
        from codes import Rolling_Arc_Real as real
        embedded_type = real.ContinuousFeeding
    results, trace = [], []
    for task, entries in groups.items():
        rows = list(entries.values())
        first = rows[0]
        window = tuple(first['window'])
        lengths = np.full(21, .0645)
        old = old_type(lengths, window)
        options = ContinuousFeedingOptions(**first['feeding']['options'])
        new = ContinuousFeeding(lengths, options)
        port = embedded_type(lengths, real.ContinuousFeedingOptions(**options.__dict__)) if embedded_type else None
        last_feedback = None
        summary = dict(task=task, control_inputs=len(rows), processed_control_inputs=0, fresh_inputs=0, old_valid=0,
            new_valid=0, first_valid_s=None, baseline_obtained=False, max_target_port_error_deg=0.,
            mask_port_mismatches=0, recorded_fault=rows[-1]['feeding']['fault'], counterfactual=True)
        for row in rows:
            summary['processed_control_inputs'] += 1
            inp = row['feeding_motor_input']
            q, feedback, t = inp['angles_deg'], inp['feedback_time_s'], inp['control_time_s']
            # Preserve recorded preparation-deadline extensions on resume.
            new.options = ContinuousFeedingOptions(**row['feeding']['options'])
            # Match the manual wrapper's formation-time calibration behavior.
            if new.stage == 'prepare':
                new.length_observer.cut_depth_m = None
            targets, mask = new.advance(q, feedback, t)
            if port:
                port.options = real.ContinuousFeedingOptions(**row['feeding']['options'])
                if port.stage == 'prepare':
                    port.length_observer.cut_depth_m = None
                target_port, mask_port = port.advance(q, feedback, t)
                summary['max_target_port_error_deg'] = max(summary['max_target_port_error_deg'],
                    float(np.degrees(abs(targets-target_port)).max()))
                summary['mask_port_mismatches'] += int(not np.array_equal(mask, mask_port))
            if feedback != last_feedback:
                baseline = old.observe(q, tuple(row['window']))
                summary['fresh_inputs'] += 1
                summary['old_valid'] += int(baseline['valid'])
                summary['new_valid'] += int(new.observation['valid'])
                if new.observation['valid'] and summary['first_valid_s'] is None:
                    summary['first_valid_s'] = feedback
                trace.append(dict(task=task, input=inp, recorded_enabled=row['enabled'],
                    recorded_targets_deg=row['target_controller_deg'],
                    old_observation=baseline, new_observation=new.observation,
                    new_stage=new.stage, new_baseline_m=new.baseline))
                last_feedback = feedback
            if new.baseline is not None:
                summary['baseline_obtained'] = True
                # The old real run never executed the new feed-out commands.
                # Stop its controller replay here, rather than claim a roundtrip.
                break
            if new.fault:
                break
        summary.update(final_stage=new.stage, new_fault=new.fault,
            baseline_m=new.baseline, final_quality=new.observation.get('end_quality'))
        results.append(summary)
    with gzip.open(REPORT/'observer_replay.jsonl.gz', 'wt', encoding='utf8') as out:
        for row in trace:
            out.write(json.dumps(row,ensure_ascii=False)+'\n')
    result=dict(source=str(LOG.relative_to(ROOT)), new_hardware_run=False, new_dynamics=False,
        embedded_core_checked=args.sync_check, tasks=results)
    (REPORT/'real_replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    for item in results:
        print(json.dumps({k:v for k,v in item.items() if k!='final_quality'},ensure_ascii=False))
    print('Last diagnostics:', json.dumps(results[-1]['final_quality'],ensure_ascii=False))
    assert all(r['old_valid']==0 for r in results if r['recorded_fault']=='initial differential length unavailable')
    assert all(r['new_fault'] is None for r in results)
    assert all(r['baseline_obtained'] for r in results if r['recorded_fault']=='initial differential length unavailable')
    assert all(r['max_target_port_error_deg']==0 and r['mask_port_mismatches']==0 for r in results)


if __name__=='__main__':
    main()
