"""Read-only spatial/material audit and rendering of the two recorded cases."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gzip
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from validation.review_active_transfer import audit

REPORT = ROOT/'validation/reports/in_place_feeding'


def review(folder):
    if not all((folder/name).exists() for name in ('actual_full.mp4', 'review.json', 'timeline.png')):
        audit(folder)  # Replay saved qpos/qvel; never advance dynamics.
    rows = [json.loads(s) for s in gzip.open(folder/'actual_trace.jsonl.gz', 'rt', encoding='utf8')]
    status = json.loads((folder/'status.json').read_text(encoding='utf8'))
    rr = [r for r in rows if r['time_s'] >= 6.]
    base = rr[0]
    bm = base['evaluation']['material']
    result = dict(baseline_time_s=base['time_s'], all_feedback_angles_only=True,
                  hardware_run=False, physics_statistics=status['statistics'])
    for key in ('support_point_world_m', 'crown_world_m', 'mean_body_world_m'):
        deltas = np.array([r['evaluation']['spatial'][key+'_delta'] for r in rr
                           if key+'_delta' in r['evaluation']['spatial']])
        result[key] = dict(maximum_displacement_mm=float(np.linalg.norm(deltas, axis=1).max()*1000),
                           maximum_absolute_axis_displacement_mm=(abs(deltas).max(0)*1000).tolist(),
                           final_delta_mm=(deltas[-1]*1000).tolist(),
                           missing_samples=len(rr)-len(deltas))
    result['stages'] = []
    for stage in ('feed_out', 'review_out', 'feed_return', 'review_return', 'complete'):
        group = [r for r in rr if r['controller']['stage'] == stage]
        if not group:
            continue
        last = group[-1]
        delta = {k:last['evaluation']['material'][k]-bm[k]
                 for k in ('crown_joint', 'head_hanging_m', 'tail_hanging_m')}
        transfer_mm = .5*(delta['head_hanging_m']-delta['tail_hanging_m'])*1000
        goal_delta = last['controller']['goal_joint']-last['controller']['baseline_crest_joint']
        result['stages'].append(dict(stage=stage, end_s=last['time_s'],
            last_material_delta=delta, goal_delta_joints=goal_delta,
            net_hanging_transfer_mm=transfer_mm,
            total_hanging_length_change_mm=(delta['head_hanging_m']+delta['tail_hanging_m'])*1000,
            final_true_goal_error_joints=delta['crown_joint']-goal_delta,
            true_goal_error_range_joints=[min(r['evaluation']['material']['crown_joint']-bm['crown_joint']-goal_delta for r in group),
                                         max(r['evaluation']['material']['crown_joint']-bm['crown_joint']-goal_delta for r in group)],
            final_progress=last['controller']['progress']))
    result['goal_events'] = []
    for event in status['events']:
        if event['kind'] != 'material_goal_observed':
            continue
        row = min(rr, key=lambda r:abs(r['time_s']-event['time_s']))
        goal_delta = event['goal_joint']-status['controller']['baseline_crest_joint']
        actual_delta = row['evaluation']['material']['crown_joint']-bm['crown_joint']
        result['goal_events'].append(dict(time_s=event['time_s'], goal_delta_joints=goal_delta,
            actual_delta_joints=actual_delta, true_goal_error_joints=actual_delta-goal_delta,
            observed_error_joints=event['observed_joint']-event['goal_joint']))
    invalid = [r for r in rr if not (r['controller'].get('observation') or {}).get('valid')]
    result['invalid_observation_samples'] = len(invalid)
    result['all_feedback_angles_only'] = all(set(r['motor_input']) == {'angles_deg', 'feedback_time_s'} for r in rows)
    result['external_force_max'] = float(max(np.max(abs(np.array(r['external_force_world']))) for r in rows))
    result['phase_max_absolute_rad'] = float(max(abs(r['controller']['phase_rad']) for r in rows))
    result['actual_bend'] = {}
    for key in ('wrap_deg', 'gap_fraction', 'total_turn_deg'):
        values = [r['evaluation']['actual_bend'][key] for r in rr if r['evaluation']['actual_bend']['valid']]
        result['actual_bend'][key] = dict(min=min(values), max=max(values))
    result['actual_bend']['invalid_samples'] = sum(not r['evaluation']['actual_bend']['valid'] for r in rr)
    worst_wrap = max(rr, key=lambda r:r['evaluation']['actual_bend'].get('wrap_deg', 0.))
    result['actual_bend']['maximum_wrap_time_s'] = worst_wrap['time_s']
    result['acceptance'] = dict(
        support_stationary=result['support_point_world_m']['maximum_displacement_mm'] <= 5.,
        support_axial_stationary=result['support_point_world_m']['maximum_absolute_axis_displacement_mm'][0] <= 1.,
        body_axial_stationary=result['mean_body_world_m']['maximum_absolute_axis_displacement_mm'][0] <= 1.,
        material_outbound=abs(next(s for s in result['stages'] if s['stage']=='review_out')['final_true_goal_error_joints']) <= .2,
        material_return=abs(result['stages'][-1]['final_true_goal_error_joints']) <= .2,
        open_smooth_section=all(150. <= r['evaluation']['section_turn_deg'] <= 218.
                                and r['evaluation']['enabled_gap_fraction'] >= .4 for r in rr),
        actual_pipe_wrap_le218=all(r['evaluation']['actual_bend']['valid'] and
                                  r['evaluation']['actual_bend']['wrap_deg'] <= 218. for r in rr),
        no_floor_or_limits=status['statistics']['floor_contact_steps'] == 0 and status['statistics']['joint_limit_excess_max_rad'] == 0,
        passive_active_torque_zero=status['statistics']['passive_torque_max_nm'] == 0.)
    result['acceptance']['all_passed'] = all(result['acceptance'].values())
    (folder/'spatial_review.json').write_text(json.dumps(result, indent=2), encoding='utf8')

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    t = np.array([r['time_s'] for r in rr])
    for key, label in (('support_point_world_m', 'up-force weighted contact'), ('crown_world_m', 'highest band'), ('mean_body_world_m', 'body mean')):
        values = np.array([r['evaluation']['spatial'].get(key+'_delta', [np.nan]*3) for r in rr])*1000
        axes[0].plot(t, np.linalg.norm(values, axis=1), label=label)
        if key == 'support_point_world_m':
            for i, name in enumerate(('pipe axis X', 'transverse Y', 'vertical Z')):
                axes[1].plot(t, values[:, i], label=name)
    axes[0].axhline(5., ls='--', color='red', label='5 mm stationarity goal')
    axes[0].set_ylabel('3D excursion mm'); axes[1].set_ylabel('support delta mm')
    axes[2].plot(t, [r['evaluation']['material']['crown_joint']-bm['crown_joint'] for r in rr], label='world-z material crown')
    axes[2].plot(t, [r['controller']['observation']['crest_joint']-r['controller']['baseline_crest_joint']
                    if r['controller']['observation']['valid'] and r['controller']['baseline_crest_joint'] is not None else np.nan for r in rr], label='encoder conditional crown')
    axes[2].plot(t, [r['controller']['goal_joint']-r['controller']['baseline_crest_joint']
                    if r['controller']['baseline_crest_joint'] is not None else np.nan for r in rr], ls='--', label='material goal')
    axes[2].set_ylabel('joint coordinate change')
    axes[3].plot(t, [r['controller']['progress'] for r in rr], label='curvature exchange progress')
    axes[3].set_ylabel('progress'); axes[3].set_xlabel('physical time s')
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(folder/'spatial_timeline.png', dpi=120); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for ax, row in zip(axes, (base, worst_wrap)):
        points = np.array(row['backbone_world_m'])
        bend = row['evaluation']['actual_bend']; first, last = bend['interval']
        arc = np.vstack([.5*(points[first-1]+points[first]), points[first:last+1],
                         .5*(points[last]+points[last+1])])
        ax.add_patch(Circle((0., 1.2), .0565, color='gray', alpha=.4))
        ax.plot(points[:, 1], points[:, 2], '-o', color='purple', ms=3, label='actual backbone')
        ax.plot(arc[:, 1], arc[:, 2], '-o', color='royalblue', ms=3, label='bend interval (>5 deg)')
        for contact in row['evaluation']['spatial']['contacts']:
            ax.plot(contact['position_world_m'][1], contact['position_world_m'][2], 'x', color='red')
        ax.set_title(f"t={row['time_s']:.2f}s  wrap={bend['wrap_deg']:.1f} deg\nactual bend turn={bend['total_turn_deg']:.1f} deg")
        ax.set_xlim(-.2, .2); ax.set_ylim(1., 1.4); ax.set_aspect('equal')
        ax.set_xlabel('world transverse Y m'); ax.set_ylabel('world Z m'); ax.legend(fontsize=8)
    fig.savefig(folder/'worst_pipe_wrap.png', dpi=130); plt.close(fig)
    print(folder.name, json.dumps(dict(spatial=result['support_point_world_m'], acceptance=result['acceptance'], stages=result['stages']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    for p in sorted(REPORT.glob('*/status.json')):
        review(p.parent)
