"""Review every saved actual state; render without advancing dynamics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import gzip
import json
import math
from collections import Counter
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.simulation import ROOT, backbone
from snake_control.angle_shape_observer import AngleShapeObserver
from snake_control.section_geometry import local_section, forward
from validation.angle_observer_replay import CANONICAL_TO_MODEL

REPORT = ROOT/'validation/reports/motor_feedback_sliding'


def load(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def window(mask):
    active = np.flatnonzero(mask)+1
    return (int(active[0]), int(active[-1]))


def coverage(support, bounds):
    return float(sum(support[bounds[0]-1:bounds[1]+1])/max(sum(support), 1e-12))


def audit(path):
    rows = load(path/'actual_trace.jsonl.gz')
    status = json.loads((path/'status.json').read_text(encoding='utf8'))
    model = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Rolling_Arc_straight.xml'))
    data = mujoco.MjData(model)
    qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    observer = AngleShapeObserver([.0645]*21)
    states = {round(r['time_s'], 8): r for r in rows}
    detailed = []
    for r in rows:
        data.qpos[:], data.qvel[:], data.time = r['actual_qpos'], r['actual_qvel'], r['time_s']
        mujoco.mj_forward(model, data)
        q = data.qpos[qadr]
        w = window(r['enabled'])
        good, g, section = observer.geometry(q, w)
        points = backbone(model, data)
        arc = points[w[0]:w[1]+2]
        # Straight pipe's verified axis is X. Sample actual rigid links about
        # the pipe centerline; this is independent of body-plane total turn.
        samples = np.concatenate([(1-u[:, None])*a+u[:, None]*b for a, b in zip(arc[:-1], arc[1:])
                                  for u in [np.linspace(0., 1., 12)]])
        theta = np.unwrap(np.arctan2(samples[:, 2]-1.2, samples[:, 1]))
        wrap = float(np.degrees(theta.max()-theta.min()))
        radii = [s.radius for s in section['curve'] if 1 <= s.joint <= 6 and s.radius > 0]
        valid_command, command_g, _ = observer.geometry(np.asarray(r['target_rad']), w)
        item = dict(time_s=r['time_s'], stage=r['stage'], window=list(w), open_actual=good,
                    body_geometry=g, pipe_angular_span_deg=wrap,
                    smooth_radius_min_m=min(radii, default=None),
                    backbone_height_min_m=float(points[:, 2].min()),
                    actual_support_coverage=coverage(r['support_up_n'], w),
                    support_sum_n=float(sum(r['support_up_n'])),
                    command_geometry=command_g,
                    command_open=command_g['turn_deg'] <= 190.01 and command_g['gap_fraction'] >= .4)
        o = r['controller']['observation']
        if o is not None:
            item['observation'] = o
            feedback = r['motor_input']
            matched = states.get(round(feedback['feedback_time_s'], 8))
            if matched is not None:
                data.qpos[:], data.qvel[:], data.time = matched['actual_qpos'], matched['actual_qvel'], matched['time_s']
                mujoco.mj_forward(model, data)
                measured = np.radians(feedback['angles_deg'])
                if np.max(abs(measured-data.qpos[qadr])) > 1e-8:
                    raise ValueError('Motor observation is not matched to its archived state')
                ow = tuple(o['current_window'])
                rotation = data.body('head').xmat.reshape(3, 3) @ CANONICAL_TO_MODEL
                true_down = rotation.T @ np.array([0., 0., -1.])
                _, _, os = observer.geometry(measured, ow)
                oracle = observer.crest(forward(measured, [.0645]*21), ow, os, true_down)
                if 'down_head' in o:
                    item['down_error_deg'] = math.degrees(math.acos(float(np.clip(true_down@o['down_head'], -1., 1.))))
                if 'crest' in o:
                    item['crest_error_joints'] = o['crest']['joint_coordinate']-oracle['joint_coordinate']
                item['true_gravity_crest_joint'] = oracle['joint_coordinate']
                if 'phase_reference' in o:
                    item['true_gravity_offset_joints'] = oracle['joint_coordinate']-sum(ow)/2-o['phase_reference'][0]
                if o['suggested_window'] is not None:
                    item['suggested_coverage'] = coverage(matched['support_up_n'], o['suggested_window'])
        detailed.append(item)
    rolling = [r for r in detailed if r['stage'] == 'rolling']
    rec = [r for r in rolling if r.get('observation', {}).get('direction')]
    good_rec = [r for r in rec if r.get('down_error_deg', np.inf) <= 15 and r.get('suggested_coverage', 0.) >= .8 and
                r['observation']['direction']*r.get('true_gravity_offset_joints', 0.) >= .5]
    events = status['events']
    summary = dict(samples=len(rows), rolling_samples=len(rolling),
        observation_states=dict(Counter(r.get('observation', {}).get('state') for r in rolling)),
        rejection_reasons=dict(Counter(reason for r in rolling for reason in r.get('observation', {}).get('reasons', []))),
        actual_open_violations=sum(not r['open_actual'] for r in rolling),
        body_turn_range_deg=[min(r['body_geometry']['turn_deg'] for r in rolling), max(r['body_geometry']['turn_deg'] for r in rolling)],
        minimum_gap=min(r['body_geometry']['gap_fraction'] for r in rolling),
        maximum_plane_rms_m=max(r['body_geometry']['fit_rms_m'] for r in rolling),
        pipe_angular_span_range_deg=[min(r['pipe_angular_span_deg'] for r in rolling), max(r['pipe_angular_span_deg'] for r in rolling)],
        smooth_radius_min_m=min(r['smooth_radius_min_m'] for r in rolling if r['smooth_radius_min_m'] is not None),
        minimum_backbone_height_m=min(r['backbone_height_min_m'] for r in detailed),
        command_open_violations=sum(not r['command_open'] for r in rolling),
        minimum_support_n=min(r['support_sum_n'] for r in rolling),
        minimum_support_coverage=min(r['actual_support_coverage'] for r in rolling),
        suggested_frames=len(rec), evaluation_consistent_suggested_frames=len(good_rec),
        first_suggestion_s=None if not rec else rec[0]['time_s'],
        suggested_down_error_max_deg=max([r.get('down_error_deg', 0.) for r in rec], default=None),
        valid_fraction=sum(r.get('observation', {}).get('valid', False) for r in rolling)/len(rolling),
        displacement_m=(np.array(rows[-1]['body_center_m'])-rows[0]['body_center_m']).tolist(),
        statistics_1ms=status['statistics_1ms'], complete_events=[e for e in events if e['kind'] == 'complete'],
        reject_events=[e for e in events if e['kind'] == 'reject'],
        final_time_s=rows[-1]['time_s'], total_phase_cycles=(rows[-1]['controller']['phase_rad'])/(2*math.pi))
    complete = summary['complete_events']
    if complete:
        first_after = next(r for r in rows if r['time_s'] > complete[-1]['time_s'])
        summary['post_last_handover_phase_cycles'] = (rows[-1]['controller']['phase_rad']-first_after['controller']['phase_rad'])/(2*math.pi)
    summary['phase_hold_spans_s'] = [[e['time_s'], next((end['time_s'] for end in events
        if end['time_s'] >= e['time_s'] and end['kind'] in ('complete', 'reject')), rows[-1]['time_s'])]
        for e in events if e['kind'] == 'phase_hold']
    with gzip.open(path/'evaluation.jsonl.gz', 'wt', encoding='utf8') as stream:
        for item in detailed:
            stream.write(json.dumps(item, ensure_ascii=False)+'\n')
    (path/'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    if not status['closed_loop']:
        # The second case is a test of the whole controller, not evidence of
        # perfect observer availability. At least one trustworthy interval is
        # required, while all invalid intervals stay visible in the report.
        allow = (len(good_rec) >= 25 and len(good_rec)/max(len(rec), 1) >= .95 and
                 summary['actual_open_violations'] == 0 and summary['minimum_support_n'] >= 2 and
                 status['statistics_1ms']['floor_contact_steps'] == 0)
        review = dict(allow_closed_loop=bool(allow), source=path.name, summary=summary,
            reason='consistent actual migration interval permits one same-condition trial' if allow else
                   'observation/shape/support conditions insufficient; do not add blind dynamics runs')
        (REPORT/'observation_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding='utf8')
    plot(detailed, path)
    print(json.dumps(dict(path=path.name, summary=summary), ensure_ascii=False))
    return rows, detailed


def plot(rows, output):
    t = [r['time_s'] for r in rows]
    fig, axes = plt.subplots(6, 1, figsize=(13, 14), sharex=True)
    obs = [r.get('observation', {}) for r in rows]
    axes[0].plot(t, [o.get('filtered_error_joints', np.nan) for o in obs], label='Motor-only phase-referenced crest')
    axes[0].plot(t, [r.get('true_gravity_offset_joints', np.nan) for r in rows], '--', label='True-gravity diagnostic')
    axes[0].axhline(.5, color='gray', ls=':'); axes[0].axhline(-.5, color='gray', ls=':')
    axes[0].set_ylabel('Offset (joints)')
    axes[1].plot(t, [o.get('filtered_feature_error', [np.nan]*3)[1] for o in obs], label='Same-winding bending migration')
    axes[1].plot(t, [int(o.get('valid', False)) for o in obs], label='Geometry gate available')
    axes[1].set_ylabel('Bending / valid')
    axes[2].plot(t, [r.get('down_error_deg', np.nan) for r in rows], label='Passive-end proxy vs true gravity')
    axes[2].plot(t, [100*abs(r.get('crest_error_joints', np.nan)) for r in rows], label='Crest error x100 (joints)')
    axes[2].set_ylabel('Error (deg / x100)')
    axes[3].plot(t, [r['body_geometry']['turn_deg'] for r in rows], label='Actual body section turn')
    axes[3].plot(t, [r['pipe_angular_span_deg'] for r in rows], label='Actual pipe angular span')
    axes[3].axhline(218, color='gray', ls=':'); axes[3].axhline(150, color='gray', ls=':')
    axes[3].set_ylabel('Angle (deg)')
    axes[4].plot(t, [r['actual_support_coverage'] for r in rows], label='Actual-window support coverage (evaluation only)')
    axes[4].plot(t, [r['suggested_coverage'] if 'suggested_coverage' in r else np.nan for r in rows], '.', label='Suggested coverage')
    axes[4].plot(t, [r['body_geometry']['gap_fraction'] for r in rows], label='Actual opening fraction')
    axes[4].set_ylabel('Coverage / opening')
    axes[5].step(t, [r['window'][0] for r in rows], label='First enabled', where='post')
    axes[5].step(t, [r['window'][1] for r in rows], label='Last enabled', where='post')
    axes[5].plot(t, [o.get('direction', 0) for o in obs], label='Suggestion direction')
    axes[5].set_ylabel('Window / direction'); axes[5].set_xlabel('Simulation time (s)')
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(fontsize=8, loc='best'); ax.axvspan(0, 2, color='gray', alpha=.1)
    fig.suptitle(output.name+' — all archived actual states; contacts only used in evaluation')
    fig.tight_layout(); fig.savefig(output/'full_timeline.png', dpi=140); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    audit(args.path)


if __name__ == '__main__':
    main()
