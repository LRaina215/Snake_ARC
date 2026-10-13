"""Keep mechanism evidence separate from full-duration/pipe acceptance."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'validation/reports/adaptive_arc_length'


def main():
    collected = []
    for path in sorted(p for p in REPORT.iterdir() if p.is_dir() and (p/'review.json').exists()):
        r = json.loads((path/'review.json').read_text(encoding='utf8'))
        status = json.loads((path/'status.json').read_text(encoding='utf8'))
        s = r['summary']
        mechanism = (r['completed_widths'] == [8, 6] and r['shape_pass'] and r['mechanics_pass'] and
                     status['final_controller']['pending'] is None and status['final_controller']['resize_fault'] is None and
                     s.get('orphan_seven_frames', 0) == 0)
        r['resize_mechanism_pass'] = bool(mechanism)
        r['exit_duration_pass'] = bool((s.get('continuous_exit_cycles') or 0) >= 2 and s.get('post_last_change_cycles', 0) >= 2)
        (path/'review.json').write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding='utf8')
        (REPORT/(r['case']+'_review.json')).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding='utf8')
        collected.append(dict(source=path.name, case=r['case'], full_pass=r['passed'], shape_pass=r['shape_pass'],
            mechanics_pass=r['mechanics_pass'], resize_mechanism_pass=r['resize_mechanism_pass'],
            strict_pipe_pass=r['strict_pipe_pass'], continuous_exit_cycles=s.get('continuous_exit_cycles'),
            post_last_change_cycles=s.get('post_last_change_cycles'), actual_width_chain=r['completed_widths'],
            position_transfers=sum(e['kind'] == 'complete' for e in s['events'])))
    result = dict(dynamics_attempts=len(collected), planned_maximum=5, actual_hardware_runs=0,
        all_requested_acceptance_complete=False, current_user_debug_and_real_entries_updated=False,
        cases=collected, timing_fix_dynamics_verified=False,
        conclusion='6/8 resizing and serialised migration mechanisms observed; complete exit-duration and '
                   'sharp-corner pipe-axis validation remain incomplete. No efficiency or diameter-identification claim.')
    (REPORT/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
