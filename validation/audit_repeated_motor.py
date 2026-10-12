"""Review repeated automatic handovers against full archived actual motion."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import gzip
import json
import numpy as np
from snake_control.simulation import ROOT
from validation.audit_motor_feedback import audit

REPORT = ROOT/'validation/reports/repeated_motor_sliding'


def read(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    rows, evaluation = audit(args.path)
    status = json.loads((args.path/'status.json').read_text(encoding='utf8'))
    summary = json.loads((args.path/'audit.json').read_text(encoding='utf8'))
    completed = summary['complete_events']
    current, visited, transitions = (6, 11), [(6, 11)], []
    for e in completed:
        old, new = tuple(e['old']), tuple(e['to'])
        before = max((r for r in evaluation if r['time_s'] <= e['time_s']+1e-8), key=lambda r: r['time_s'])
        after = next(r for r in rows if r['time_s'] > e['time_s']+1e-8)
        o = after['controller']['observation']
        matched = min(evaluation, key=lambda r: abs(r['time_s']-after['time_s']))
        direction = new[0]-old[0]
        outgoing = next(iter(set(range(old[0], old[1]+1))-set(range(new[0], new[1]+1))))
        prior = min(rows, key=lambda r: abs(r['time_s']-e['time_s']))
        truth_correct = (o is not None and o['direction'] == direction and o['valid'] and
            matched.get('down_error_deg', np.inf) <= 15 and
            direction*matched.get('true_gravity_offset_joints', 0.) >= .5 and
            matched.get('suggested_coverage', 0.) >= .8)
        transitions.append(dict(time_s=e['time_s'], old=list(old), new=list(new),
            chained=old == current, adjacent=abs(direction) == 1 and new[1]-new[0] == 5,
            evidence_agrees_with_evaluation=bool(truth_correct),
            feedback_time_s=after['controller']['feedback_time_s'],
            returned_to_prior=new in visited,
            exit_torque_nm_evaluation=prior['actuator_force_nm'][outgoing-1],
            true_gravity_offset=matched.get('true_gravity_offset_joints'),
            suggested_support_coverage=matched.get('suggested_coverage')))
        current = new; visited.append(new)
    frames = [r for r in rows if r['controller'].get('new_feedback_this_tick') and r['stage'] == 'rolling']
    states = {r['motor_input']['feedback_time_s'] for r in rows if r['motor_input'] is not None}
    directions = [t['new'][0]-t['old'][0] for t in transitions]
    actual_return = any(t['returned_to_prior'] for t in transitions) and min(directions, default=0) < 0 < max(directions, default=0)
    morphology = (summary['actual_open_violations'] == 0 and summary['command_open_violations'] == 0 and
                  summary['pipe_angular_span_range_deg'][1] <= 218.)
    stats = summary['statistics_1ms']
    invariant = stats['mask_violation_steps'] == 0 and stats['passive_torque_max_nm'] == 0. and stats['finite']
    mechanics = stats['floor_contact_steps'] == 0 and stats['joint_limit_excess_max_rad'] == 0. and summary['minimum_support_n'] >= 2.
    repeated = len(transitions) >= 2 and all(t['chained'] and t['adjacent'] and t['evidence_agrees_with_evaluation'] for t in transitions)
    recovery = summary.get('post_last_handover_phase_cycles', 0.) >= 2.-1e-8
    required_motion = repeated and (status['case'] == 'natural' or actual_return)
    passed = bool(required_motion and morphology and invariant and mechanics and recovery)
    result = dict(case=status['case'], source=args.path.name, passed=passed,
        transitions=transitions, unique_feedback_samples=len(states), rolling_new_feedback_frames=len(frames),
        morphology_pass=morphology, six_and_disabled_invariant_pass=invariant,
        mechanics_pass=mechanics, repeated_motion_pass=repeated, actual_return_pass=bool(actual_return),
        post_handover_cycles_pass=recovery, full_audit=summary,
        notes=['No scripted window requests or qpos resets', 'All handover gates use new motor feedback',
               'High support coverage does not prove migration necessity or efficiency gain'])
    (args.path/'review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    (REPORT/(status['case']+'_review.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k != 'full_audit'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
