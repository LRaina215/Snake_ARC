"""Review every actual state and feedback, keeping truth outside control."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter
import contextlib
import gzip
import io
import json
import math
import numpy as np
import mujoco
from snake_control.simulation import ROOT
from snake_control.angle_shape_observer import AngleShapeObserver
from snake_control.posture_window import GeneralPhaseReference
from snake_control.motor_shape_observer import PhaseShapeReference, local_phase
from snake_control.section_geometry import forward, local_section
from validation.angle_observer_replay import CANONICAL_TO_MODEL
from validation.posture_window_experiment import REPORT
import validation.audit_arc_length as base_audit


def load(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def audit(path):
    # Reuse the full actual body/pipe/torque audit, replace its old 6/8 verdict.
    report_folder = path.parent
    base_audit.REPORT = report_folder
    with contextlib.redirect_stdout(io.StringIO()):
        base_audit.audit(path)
    status = json.loads((path/'status.json').read_text(encoding='utf8'))
    review = json.loads((path/'review.json').read_text(encoding='utf8'))
    rows = load(path/'actual_trace.jsonl.gz')
    states = {round(r['time_s'], 8): r for r in rows}
    references = {int(w): PhaseShapeReference.from_dict(r) for w, r in status['frozen_references'].items()}
    model = mujoco.MjModel.from_xml_path(str(ROOT/f"mujoco_model/Snake_Robot_o20_Rolling_Arc_{status['scene']}.xml"))
    data, helper = mujoco.MjData(model), AngleShapeObserver([.0645]*21)
    general_refs, observed, seen = {}, [], set()
    event_observation = {round(e['time_s'], 8): e.get('observation') for e in status['events'] if e['kind'] in ('complete', 'resize_complete')}
    for row in rows:
        f = row.get('motor_input')
        if not f or f['feedback_time_s'] < 2. or f['feedback_time_s'] in seen:
            continue
        o = row['controller']['observation'] or event_observation.get(round(f['feedback_time_s'], 8))
        if not o:
            continue
        seen.add(f['feedback_time_s'])
        origin = states[round(f['feedback_time_s'], 8)]
        data.qpos[:], data.qvel[:], data.time = origin['actual_qpos'], origin['actual_qvel'], origin['time_s']
        mujoco.mj_forward(model, data)
        q = np.radians(f['angles_deg'])
        if np.max(abs(data.qpos[7:]-q)) > 1e-8:
            raise ValueError('Feedback and independent state disagree')
        window = tuple(o['current_window']); width = window[1]-window[0]+1
        general_refs.setdefault(width, GeneralPhaseReference([.0645]*21, width, references))
        down = (data.body('head').xmat.reshape(3, 3)@CANONICAL_TO_MODEL).T@np.array([0., 0., -1.])
        true_offset = None
        try:
            _, _, section = helper.geometry(q, window)
            crest = helper.crest(forward(q, [.0645]*21), window, section, down)['joint_coordinate']
            true_offset = crest-sum(window)/2-general_refs[width].at(f['phase_rad'], window[0])[0]
        except (ValueError, np.linalg.LinAlgError):
            pass
        estimate = o.get('filtered_error_joints')
        direction = o.get('direction', 0)
        support = float(sum(origin['support_up_n']))
        observed.append(dict(time_s=f['feedback_time_s'], width=width,
            phase_bin=int(local_phase(f['phase_rad'], window[0])/(2*math.pi)*24) % 24,
            valid=o['valid'], end_source=o.get('end_quality', {}).get('source'),
            trusted_down='down_head' in o, true_gravity_offset_joints=true_offset,
            estimate_joints=estimate,
            error_joints=None if true_offset is None or estimate is None else abs(estimate-true_offset),
            suggested_direction=direction, opposite=bool(direction and true_offset is not None and direction*true_offset < 0),
            support_n=support, reference_transferred=width not in references))
    changes = [e for e in status['events'] if e['kind'] in ('complete', 'resize_complete')]
    phase_errors = [abs(r['controller']['phase_rad']-(status['parameters']['initial_phase_rad']+.6*max(0., r['controller']['control_time_s']-2.)))
                    for r in rows if r['controller']['control_time_s'] is not None]
    normal = max(phase_errors, default=0.) < 1e-7 and not any(r['controller']['phase_hold'] for r in rows)
    summary = review['summary']
    last_phase = changes[-1]['phase_rad'] if changes else status['parameters']['initial_phase_rad']
    summary['post_last_change_cycles'] = (rows[-1]['controller']['phase_rad']-last_phase)/(2*math.pi)
    summary.pop('orphan_seven_frames', None)
    summary['stable_odd_width_frames'] = sum(sum(r['enabled']) % 2 == 1 and r['controller']['pending'] is None and r['time_s'] >= 2. for r in rows)
    summary['count_chain'] = [status['parameters']['last_joint']-status['parameters']['first_joint']+1]+[e['to'][1]-e['to'][0]+1 for e in changes]
    summary['operation_counts'] = dict(Counter(e['operation'] for e in changes))
    summary['shape_trials'] = [e for e in status['events'] if e['kind'] in ('shape_trial_complete', 'rollback_request')]
    curves = []
    for row in rows:
        section = local_section(np.asarray(row['actual_qpos'][7:]), [.0645]*21, *row['controller']['window'])
        radii = [abs(s.radius) for s in section['curve'] if s.radius]
        if radii:
            curves.append((min(radii), row['time_s']))
    summary['curvature_review'] = dict(minimum_projected_smooth_curve_radius_m=min(curves)[0],
        worst_time_s=min(curves)[1], definition='actual projected curve proxy, not pipe or solid-link radius')
    summary['maximum_phase_error_rad'] = max(phase_errors, default=0.)
    trusted = [o for o in observed if o['trusted_down'] and o['error_joints'] is not None]
    location = dict(new_feedbacks=len(observed), sources=dict(Counter(o['end_source'] for o in observed)),
                    all_phase_bins_sampled=len({o['phase_bin'] for o in observed}) == 24,
                    maximum_trusted_error_joints=max((o['error_joints'] for o in trusted), default=None),
                    recommendations=sum(bool(o['suggested_direction']) for o in observed),
                    opposite_recommendations=sum(o['opposite'] for o in observed),
                    transferred_reference_feedbacks=sum(o['reference_transferred'] for o in observed))
    final = status['final_controller']
    finished = final['pending'] is None and final['trial'] is None and final['fault'] is None and final['resize_fault'] is None
    basic = review['shape_pass'] and review['mechanics_pass'] and normal and finished
    continuation = summary['post_last_change_cycles'] >= 1.
    if status['case'] == 'straight':
        passed = basic and review['strict_pipe_pass'] and continuation and sum(e['operation'] == 'slide' for e in changes) >= 2
    elif status['case'] == 'return':
        expansions = [e for e in changes if e['operation'] == 'expand']
        contractions = [e for e in changes if e['operation'] == 'contract']
        expand_then_contract = expansions and any(e['time_s'] > expansions[0]['time_s'] for e in contractions)
        passed = basic and review['strict_pipe_pass'] and continuation and (summary['continuous_exit_cycles'] or 0.) >= 1. and expand_then_contract
    else:
        passed = basic and continuation and (summary['continuous_exit_cycles'] or 0.) >= 1. and any(e['operation'] in ('expand', 'contract') for e in changes)
    for key in ('resize_pass', 'completed_widths', 'resize_mechanism_pass', 'exit_duration_pass'):
        review.pop(key, None)
    review.update(passed=bool(passed), phase_continuity_pass=bool(normal), location=location,
                  no_fixed_width_whitelist=status['stable_width_whitelist'] is None,
                  count_adjustment_covered=any(e['operation'] in ('expand', 'contract') for e in changes),
                  odd_stable_width_covered=bool(summary['stable_odd_width_frames']),
                  complete_single_pipe_axis_validation=review['strict_pipe_pass'],
                  supports_measured_in_controller=False)
    (path/'review.json').write_text(json.dumps(review, indent=2), encoding='utf8')
    (report_folder/(status['case']+'_review.json')).write_text(json.dumps(review, indent=2), encoding='utf8')
    with gzip.open(path/'position_review.jsonl.gz', 'wt', encoding='utf8') as stream:
        for o in observed: stream.write(json.dumps(o)+'\n')
    print(json.dumps({k: v for k, v in review.items() if k != 'summary'}))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('events', 'shape_trials')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('path', type=Path)
    audit(parser.parse_args().path)
