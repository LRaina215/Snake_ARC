"""Audit every archived actual state; never step dynamics or feed truth back."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter
import gzip
import json
import math
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.simulation import ROOT, backbone
from snake_control.arc_shape import ArcShapeEvaluator
from snake_control.angle_shape_observer import AngleShapeObserver
from snake_control.motor_shape_observer import PhaseShapeReference, MotorShapeObserver, MotorObserverOptions, bending_features, local_phase
from snake_control.adaptive_arc_length import circle_shape
from validation.angle_observer_replay import CANONICAL_TO_MODEL

REPORT = ROOT/'validation/reports/adaptive_arc_length'


def exit_review(rows, status):
    threshold = status['scene_geometry']['feature_span_m'][-1]+.05
    index = len(rows)-1
    while index > 0 and rows[index].get('whole_body_min_route_m', -1.) > threshold:
        index -= 1
    suffix = rows[index+1:]
    final_phase = rows[-1]['controller']['phase_rad']
    changes = [e for e in status['events'] if e['kind'] in ('resize_complete', 'complete')]
    last = 0. if not changes else changes[-1]['time_s']
    after = next(r['controller']['phase_rad'] for r in rows if r['time_s'] >= last)
    return dict(continuous_exit_start_s=None if not suffix else suffix[0]['time_s'],
        continuous_exit_cycles=None if not suffix else (final_phase-suffix[0]['controller']['phase_rad'])/(2*math.pi),
        post_last_change_cycles=(final_phase-after)/(2*math.pi),
        orphan_seven_frames=sum(sum(r['enabled']) == 7 and (r['controller']['pending'] or {}).get('kind') != 'resize' for r in rows))


def contact_axis_conflict(model, data, evaluator, points, window):
    """Evaluation-only: nearest anchors must agree with contacting pipe axes.

    Includes the physical boundary modules of the reconstructed arc. A sharp
    corner can contact the outgoing pipe while all anchors project nearest
    the incoming line; its apparent single-axis span is then not reliable.
    """
    pts = points[window[0]:window[1]+2]
    frac = np.clip(np.sum((pts[:, None]-evaluator.a)*evaluator.delta, axis=2)/evaluator.length2, 0, 1)
    index = np.argmin(np.linalg.norm(pts[:, None]-(evaluator.a+frac[:, :, None]*evaluator.delta), axis=2), axis=1)
    nearest = evaluator.tangent[index]
    head, pipe_body = model.body('head').id, model.body('pipe_network').id
    for contact in data.contact:
        pair = [int(contact.geom1), int(contact.geom2)]
        pipe = [g for g in pair if model.geom_bodyid[g] == pipe_body]
        robot = [g for g in pair if model.geom_bodyid[g] >= head]
        if not pipe or not robot:
            continue
        module = model.geom_bodyid[robot[0]]-head-1
        if not window[0]-2 <= module <= window[1]:
            continue
        suffix = model.geom(pipe[0]).name.rsplit('_', 1)[-1]
        if not suffix.isdigit():
            continue
        axis = evaluator.tangent[int(suffix)]
        if np.max(abs(nearest@axis)) < .5:
            return True
    return False


def load(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def calibration(path, rows, review):
    first, last = 7, 14
    unique = {r['motor_input']['feedback_time_s']: r for r in rows if r['motor_input'] and r['controller']['observation']}
    bins = [[] for _ in range(24)]
    for r in unique.values():
        f, o = r['motor_input'], r['controller']['observation']
        if not (0 <= f['phase_rad'] < 2*np.pi) or not o['valid']:
            continue
        features = bending_features(f['angles_deg'], [.0645]*21, (first, last))
        index = int(local_phase(f['phase_rad'], first)/(2*np.pi)*24) % 24
        bins[index].append([o['crest']['joint_coordinate']-10.5, features['bending_center_joint']-10.5,
                           features['tail_boundary_turn_deg']-features['head_boundary_turn_deg']])
    cal = dict(first_cycle_counts=[len(b) for b in bins], all_bins_observed=all(bins), holdout_new_feedbacks=0,
               false_recommendations=0, valid_fraction=0., freeze_reference=False)
    if all(bins):
        reference = PhaseShapeReference([np.median(b, axis=0).tolist() for b in bins],
            dict(width=8, source=path.name, control_feedback_period_s=.2, phase_interval_rad=[0., 2*math.pi],
                 fields=['crest', 'same-winding bending', 'same-winding tail minus head boundary'],
                 link_lengths_m=[.0645]*21, adaptation='frozen first cycle; next cycle held out'))
        observer = MotorShapeObserver([.0645]*21, reference, width=8,
                                      options=MotorObserverOptions(maximum_feedback_gap_s=.5))
        holdout = []
        for r in unique.values():
            f = r['motor_input']
            if 2*np.pi <= f['phase_rad'] < 4*np.pi:
                holdout.append(observer.update(f['angles_deg'], f['phase_rad'], f['enabled'], f['feedback_time_s']))
        cal.update(holdout_new_feedbacks=len(holdout), false_recommendations=sum(bool(o['suggested_window']) for o in holdout),
            valid_fraction=sum(o['valid'] for o in holdout)/max(1, len(holdout)))
        cal['freeze_reference'] = bool(review['shape_pass'] and review['mechanics_pass'] and
            len(holdout) >= 40 and cal['false_recommendations'] == 0 and cal['valid_fraction'] >= .7)
        if cal['freeze_reference']:
            (ROOT/'configs/motor_shape_reference8.json').write_text(json.dumps(reference.to_dict(), indent=2), encoding='utf8')
    return cal


def audit(path):
    status = json.loads((path/'status.json').read_text(encoding='utf8'))
    rows = load(path/'actual_trace.jsonl.gz')
    model = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model'/f"Snake_Robot_o20_Rolling_Arc_{status['scene']}.xml"))
    data = mujoco.MjData(model)
    helper = AngleShapeObserver([.0645]*21)
    evaluator = ArcShapeEvaluator(model.numeric('pipe_centerline').data.reshape(-1, 4)[:, :3])
    states = {round(r['time_s'], 8): r for r in rows}
    details = []
    for r in rows:
        data.qpos[:], data.qvel[:], data.time = r['actual_qpos'], r['actual_qvel'], r['time_s']
        mujoco.mj_forward(model, data)
        w = tuple(r['controller']['window'])
        good, g, section = helper.geometry(data.qpos[7:], w)
        _, sent, _ = helper.geometry(np.asarray(r['target_rad']), w)
        points = backbone(model, data)
        pipe = evaluator.measure(points, np.degrees(data.qpos[7:]), *w)
        pipe['nearest_wrap_valid'] = pipe['wrap_valid']
        pipe['contact_axis_conflict'] = contact_axis_conflict(model, data, evaluator, points, w)
        pipe['wrap_valid'] &= not pipe['contact_axis_conflict']
        support = np.asarray(r['support_up_n'])
        o = r['controller']['observation']
        d = dict(time_s=r['time_s'], body_geometry=g, actual_open=good, command_geometry=sent,
            command_open=sent['turn_deg'] <= 190.01 and sent['gap_fraction'] >= .4,
            pipe_shape=pipe, support_n=float(support.sum()), support_coverage=float(support[w[0]-1:w[1]+1].sum()/max(support.sum(), 1e-12)),
            width=int(sum(r['enabled'])), window=list(w), body_circle=circle_shape(section),
            observation=o, length_observation=r['controller'].get('length_observation'))
        if o and o.get('down_head'):
            f = r['motor_input']
            source = states[round(f['feedback_time_s'], 8)]
            data.qpos[:], data.qvel[:], data.time = source['actual_qpos'], source['actual_qvel'], source['time_s']
            mujoco.mj_forward(model, data)
            if np.max(abs(data.qpos[7:]-np.radians(f['angles_deg']))) > 1e-8:
                raise ValueError('Observation feedback does not match saved actual state')
            true_down = (data.body('head').xmat.reshape(3, 3)@CANONICAL_TO_MODEL).T@np.array([0., 0., -1.])
            d['down_error_deg'] = math.degrees(math.acos(float(np.clip(true_down@o['down_head'], -1., 1.))))
        details.append(d)
    rolling = [r for r in details if r['time_s'] >= 2.]
    stats = status['statistics_1ms']
    summary = dict(samples=len(rows), body_turn_range_deg=[min(r['body_geometry']['turn_deg'] for r in rolling), max(r['body_geometry']['turn_deg'] for r in rolling)],
        minimum_gap=min(r['body_geometry']['gap_fraction'] for r in rolling), maximum_plane_rms_m=max(r['body_geometry']['fit_rms_m'] for r in rolling),
        body_violations=sum(not r['actual_open'] for r in rolling), command_violations=sum(not r['command_open'] for r in rolling),
        pipe_unobservable=sum(not r['pipe_shape']['wrap_valid'] for r in rolling),
        contact_axis_conflict_frames=sum(r['pipe_shape']['contact_axis_conflict'] for r in rolling),
        pipe_wrap_range_deg=[min(r['pipe_shape']['wrap_deg'] for r in rolling if r['pipe_shape']['wrap_valid']), max(r['pipe_shape']['wrap_deg'] for r in rolling if r['pipe_shape']['wrap_valid'])],
        pipe_violations=sum(r['pipe_shape']['wrap_valid'] and not 150 <= r['pipe_shape']['wrap_deg'] <= 218 for r in rolling),
        minimum_support_n=min(r['support_n'] for r in rolling), minimum_support_coverage=min(r['support_coverage'] for r in rolling),
        widths=dict(Counter(r['width'] for r in rolling)), observation_states=dict(Counter(r['observation']['state'] if r['observation'] else 'none' for r in rolling)),
        length_states=dict(Counter(r['length_observation']['state'] if r['length_observation'] else 'none' for r in rolling)),
        displacement_m=(np.asarray(rows[-1]['body_center_m'])-rows[0]['body_center_m']).tolist(),
        events=status['events'], statistics_1ms=stats, post_exit_cycles=status['post_exit_cycles'])
    summary.update(exit_review(rows, status))
    completions = [e for e in status['events'] if e['kind'] == 'resize_complete']
    transitions = [len(range(e['to'][0], e['to'][1]+1)) for e in completions]
    mechanics = (stats['finite'] and stats['floor_contact_steps'] == 0 and stats['joint_limit_excess_max_rad'] <= .02 and
                 stats['passive_torque_max_nm'] == 0 and stats['mask_violation_steps'] == 0 and summary['minimum_support_n'] >= 2.)
    shape = summary['body_violations'] == 0 and summary['command_violations'] == 0
    pipe_pass = summary['pipe_unobservable'] == 0 and summary['pipe_violations'] == 0
    resize_pass = (transitions == [8, 6] and status['final_controller']['pending'] is None and
                   status['final_controller']['resize_fault'] is None and shape and mechanics and pipe_pass and
                   (summary['continuous_exit_cycles'] or 0) >= 2. and summary['post_last_change_cycles'] >= 2. and
                   summary['orphan_seven_frames'] == 0)
    review = dict(case=status['case'], source=path.name, shape_pass=shape, mechanics_pass=mechanics,
        strict_pipe_pass=pipe_pass, resize_pass=bool(resize_pass), completed_widths=transitions, summary=summary)
    review['resize_mechanism_pass'] = bool(transitions == [8, 6] and shape and mechanics and
        status['final_controller']['pending'] is None and status['final_controller']['resize_fault'] is None and
        summary['orphan_seven_frames'] == 0)
    review['exit_duration_pass'] = bool((summary['continuous_exit_cycles'] or 0) >= 2. and summary['post_last_change_cycles'] >= 2.)
    if status['case'] == 'calibration':
        review['calibration'] = calibration(path, rows, review)
        review['passed'] = review['calibration']['freeze_reference']
    elif status['case'] == 'baseline':
        review['passed'] = bool(shape and mechanics and pipe_pass and (summary['continuous_exit_cycles'] or 0) >= 2.)
    elif status['case'] == 'combined':
        review['passed'] = bool(resize_pass and any(e['kind'] == 'complete' for e in status['events']))
    else:
        review['passed'] = bool(resize_pass)
    with gzip.open(path/'evaluation.jsonl.gz', 'wt', encoding='utf8') as stream:
        for r in details:
            stream.write(json.dumps(r, ensure_ascii=False)+'\n')
    (path/'review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding='utf8')
    (REPORT/(status['case']+'_review.json')).write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding='utf8')
    t = [r['time_s'] for r in details]
    fig, ax = plt.subplots(7, 1, figsize=(13, 14), sharex=True)
    ax[0].plot(t, [r['width'] for r in details], label='Actual enabled count')
    ax[0].plot(t, [r['window'][0] for r in details], label='First joint'); ax[0].legend()
    ax[1].plot(t, [r['body_geometry']['turn_deg'] for r in details], label='Actual body turn')
    ax[1].plot(t, [r['command_geometry']['turn_deg'] for r in details], label='Sent target turn'); ax[1].legend()
    for y in (150, 218): ax[1].axhline(y, color='grey', ls=':')
    ax[2].plot(t, [r['pipe_shape']['wrap_deg'] if r['pipe_shape']['wrap_valid'] else np.nan for r in details], label='Actual pipe span (truth only)'); ax[2].legend()
    for y in (150, 218): ax[2].axhline(y, color='grey', ls=':')
    ax[3].plot(t, [r['body_geometry']['gap_fraction'] for r in details]); ax[3].axhline(.4, color='grey', ls=':'); ax[3].set_ylabel('Opening')
    ax[4].plot(t, [r['support_n'] for r in details], label='Support N (evaluation)'); ax[4].legend()
    ax[5].plot(t, [r['body_circle']['radius_m'] for r in details], label='Fitted BODY radius'); ax[5].legend()
    ax[6].plot(t, [r.get('down_error_deg', np.nan) for r in details], label='Down-proxy error (truth only)'); ax[6].legend()
    for a in ax: a.grid(alpha=.2); a.axvspan(0, 2, color='grey', alpha=.1)
    ax[-1].set_xlabel('Simulation time / s'); fig.suptitle(path.name+' - every archived actual state')
    fig.tight_layout(); fig.savefig(path/'full_timeline.png', dpi=140); plt.close(fig)
    print(json.dumps({k: v for k, v in review.items() if k != 'summary'}, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != 'events'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('path', type=Path)
    audit(parser.parse_args().path)
