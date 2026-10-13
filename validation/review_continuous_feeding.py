"""Audit/render complete recorded motion. This file never advances dynamics."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import gzip
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2
import mujoco
from snake_control.simulation import ArcSimulation
from snake_control.rolling_arc import ArcParameters

REPORT = ROOT/'validation/reports/continuous_feeding'


def transfer(row):
    m = row['evaluation']['material']
    return .5*(m['head_hanging_m']-m['tail_hanging_m']) if m['valid'] else np.nan


def whole_body_bend(row):
    """Repeat the actual bend metric over ALL joints, beyond old J5-J16 scope.

    Report positive bend and reverse folds separately. A threshold interval
    and its polar wrap are boundary-dependent and cannot alone certify a
    fitted circle. Raw full backbone and the worst section plot are retained.
    """
    xy = np.array(row['backbone_world_m'])[:, [1, 2]]-np.array([0., 1.2])
    d = np.diff(xy, axis=0)
    turns = np.arctan2(d[:-1, 0]*d[1:, 1]-d[:-1, 1]*d[1:, 0], np.sum(d[:-1]*d[1:], axis=1))
    peak = int(np.argmax(xy[:, 1]))
    winding = 1 if turns[max(0, peak-3):min(len(turns), peak+2)].sum() >= 0 else -1
    select = np.flatnonzero(winding*turns > math.radians(5.))+1
    if not len(select):
        return dict(valid=False)
    first, last = int(select[0]), int(select[-1])
    arc = np.vstack([.5*(xy[first-1]+xy[first]), xy[first:last+1], .5*(xy[last]+xy[last+1])])
    polar = np.unwrap(np.arctan2(arc[:, 1], arc[:, 0]))
    return dict(valid=True, interval=[first, last], threshold_deg=5.,
        positive_turn_deg=math.degrees(float(np.maximum(winding*turns[first-1:last], 0.).sum())),
        reverse_turn_deg=math.degrees(float(np.maximum(-winding*turns, 0.).sum())),
        pipe_wrap_deg=math.degrees(abs(polar[-1]-polar[0])),
        gap_fraction=float(np.linalg.norm(arc[-1]-arc[0])/np.linalg.norm(np.diff(arc, axis=0), axis=1).sum()))


def review(folder, render=False):
    status = json.loads((folder/'status.json').read_text(encoding='utf8'))
    rows = [json.loads(s) for s in gzip.open(folder/'actual_trace.jsonl.gz', 'rt', encoding='utf8')]
    for row in rows:
        row['review_whole_bend'] = whole_body_bend(row)
    rr = [r for r in rows if r['time_s']>=6.]
    base = rr[0]; baseline = transfer(base)
    t = np.array([r['time_s'] for r in rows])
    true = np.array([transfer(r)-baseline for r in rows])
    summary = dict(termination=status['termination'], fault=status['controller']['fault'],
                   final_time_s=status['final_time_s'], statistics=status['statistics'],
                   baseline_time_s=base['time_s'], requested_transfer_m=status['options']['requested_transfer_m'],
                   hardware_run=False, stage_results=[])
    for stage in ('feed_out', 'review_out', 'feed_return', 'review_return', 'complete'):
        group = [r for r in rr if r['controller']['stage']==stage]
        if not group:
            continue
        last = group[-1]; m = last['evaluation']['material']; bm = base['evaluation']['material']
        errors = [(transfer(r)-baseline-r['controller']['goal_delta_m'])*1000 for r in group]
        summary['stage_results'].append(dict(stage=stage, start_s=group[0]['time_s'], end_s=last['time_s'],
            last_head_change_mm=(m['head_hanging_m']-bm['head_hanging_m'])*1000,
            last_tail_change_mm=(m['tail_hanging_m']-bm['tail_hanging_m'])*1000,
            last_transfer_delta_mm=(transfer(last)-baseline)*1000,
            true_goal_error_mm=errors[-1], true_goal_error_range_mm=[min(errors), max(errors)],
            last_position_joints=last['controller']['position_joints']))
    summary['spatial'] = {}
    for key in ('support_point_world_m', 'crown_world_m', 'mean_body_world_m'):
        deltas = np.array([r['evaluation']['spatial'][key+'_delta'] for r in rr if key+'_delta' in r['evaluation']['spatial']])*1000
        summary['spatial'][key] = dict(maximum_displacement_mm=float(np.linalg.norm(deltas, axis=1).max()),
            maximum_absolute_axis_displacement_mm=abs(deltas).max(0).tolist(), final_delta_mm=deltas[-1].tolist(),
            missing_samples=len(rr)-len(deltas))
    for key in ('section_turn_deg', 'enabled_gap_fraction', 'plane_rms_m', 'maximum_curvature_per_m', 'support_up_n', 'legacy_enabled_wrap_deg', 'min_z_m'):
        values = [r['evaluation'][key] for r in rr]
        summary[key] = dict(min=min(values), max=max(values))
    summary['whole_body_bend'] = {}
    for key in ('positive_turn_deg', 'reverse_turn_deg', 'pipe_wrap_deg', 'gap_fraction'):
        values = [r['review_whole_bend'][key] for r in rr if r['review_whole_bend']['valid']]
        summary['whole_body_bend'][key] = dict(min=min(values), max=max(values))
    invalid = [r for r in rr if not r['controller']['observation']['valid']]
    summary['invalid_length_samples'] = len(invalid)
    summary['zero_support_samples'] = sum(r['evaluation']['support_up_n']<=1e-9 for r in rr)
    summary['first_invalid_time_s'] = invalid[0]['time_s'] if invalid else None
    bound = status['options']['bearing_bias_limit']
    summary['bearing_bias_at_limit_s'] = sum(abs(r['controller']['bearing_bias'])>=bound-1e-8 for r in rr)*.02
    errors = []
    for r in rr:
        o = r['controller']['observation']
        if not o['valid'] or r['controller']['baseline_m'] is None:
            continue
        k = int(np.argmin(abs(t-r['motor_input']['feedback_time_s'])))
        actual = transfer(rows[k])-baseline
        errors.append((o['transfer_coordinate_m']-r['controller']['baseline_m']-actual)*1000)
    summary['feedback_aligned_delta_error_mm'] = dict(rmse=float(np.sqrt(np.mean(np.array(errors)**2))), maximum=float(max(abs(np.array(errors)))))
    summary['mask_handovers'] = [e for e in status['events'] if e['kind']=='mask_handover']
    summary['phase_max_absolute_rad'] = max(abs(r['controller']['phase_rad']) for r in rows)
    summary['external_force_max'] = float(max(np.max(abs(np.array(r['external_force_world']))) for r in rows))
    summary['all_inputs_angles_only'] = all(set(r['motor_input'])=={'angles_deg', 'feedback_time_s'} for r in rows)
    summary['finite_actual_states'] = all(np.isfinite(r['actual_qpos']).all() and np.isfinite(r['actual_qvel']).all() for r in rows)
    stages = {s['stage']:s for s in summary['stage_results']}
    summary['acceptance'] = dict(
        completed=status['termination']=='task_complete',
        outbound_sustained_length='review_out' in stages and max(abs(np.array(stages['review_out']['true_goal_error_range_mm'])))<=12.,
        return_sustained_length='review_return' in stages and max(abs(np.array(stages['review_return']['true_goal_error_range_mm'])))<=12.,
        final_length='complete' in stages and abs(stages['complete']['true_goal_error_mm'])<=12.,
        open_enabled_section=all(150.<=r['evaluation']['section_turn_deg']<=218. and r['evaluation']['enabled_gap_fraction']>=.4 for r in rr),
        whole_bend_turn_le218=summary['whole_body_bend']['positive_turn_deg']['max']<=218.,
        whole_pipe_wrap_le218=summary['whole_body_bend']['pipe_wrap_deg']['max']<=218.,
        no_floor_or_limits=status['statistics']['floor_contact_steps']==0 and status['statistics']['joint_limit_excess_max_rad']==0.,
        continuous_support_at_recorded_samples=summary['zero_support_samples']==0,
        passive_active_torque_zero=status['statistics']['passive_torque_max_nm']==0.,
        body_axial_displacement_le10mm=summary['spatial']['mean_body_world_m']['maximum_absolute_axis_displacement_mm'][0]<=10.)
    summary['acceptance'] = {key:bool(value) for key, value in summary['acceptance'].items()}
    summary['acceptance']['all_passed'] = all(summary['acceptance'].values())
    (folder/'review.json').write_text(json.dumps(summary, indent=2), encoding='utf8')

    fig, axes = plt.subplots(6, 1, figsize=(12, 15), sharex=True, constrained_layout=True)
    axes[0].plot(t, true*1000, label='true net hanging transfer')
    axes[0].plot(t, [(r['controller']['observation'].get('transfer_coordinate_m', np.nan)-r['controller']['baseline_m'])*1000
                    if r['controller']['baseline_m'] is not None and r['controller']['observation']['valid'] else np.nan for r in rows], label='encoder proxy')
    axes[0].plot(t, [r['controller']['goal_delta_m']*1000 if r['controller']['goal_delta_m'] is not None else np.nan for r in rows], '--', label='goal')
    axes[0].set_ylabel('transfer mm')
    for key in ('head_hanging_m', 'tail_hanging_m'):
        axes[1].plot(t, [(r['evaluation']['material'][key]-base['evaluation']['material'][key])*1000 for r in rows], label=key)
    axes[1].set_ylabel('actual length change mm')
    axes[2].plot(t, [r['evaluation']['section_turn_deg'] for r in rows], label='enabled smooth section turn')
    axes[2].plot(t, [r['review_whole_bend'].get('positive_turn_deg', np.nan) for r in rows], label='whole-body bend positive turn')
    axes[2].plot(t, [r['review_whole_bend'].get('pipe_wrap_deg', np.nan) for r in rows], label='whole-body pipe polar wrap')
    axes[2].axhline(218., color='red', ls='--'); axes[2].set_ylabel('degrees')
    axes[3].plot(t, [(r['evaluation']['mean_x_m']-base['evaluation']['mean_x_m'])*1000 for r in rows], label='body mean pipe axis mm')
    for key, label in (('support_point_world_m', 'support 3D excursion'), ('crown_world_m', 'crown 3D excursion')):
        axes[3].plot(t, [np.linalg.norm(r['evaluation']['spatial'].get(key+'_delta', [np.nan]*3))*1000 for r in rows], label=label)
    axes[3].set_ylabel('mm')
    axes[4].plot(t, [r['controller']['position_joints'] for r in rows], label='travelling profile position')
    axes[4].plot(t, [r['controller']['bearing_bias'] for r in rows], label='middle curvature bias')
    axes[4].step(t, [int(r['controller']['observation']['valid']) for r in rows], label='length valid')
    axes[5].plot(t, [r['evaluation']['support_up_n'] for r in rows], label='true support N')
    axes[5].step(t, [sum(r['enabled']) for r in rows], label='enabled count')
    axes[5].set_xlabel('physical time s')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(folder/'timeline.png', dpi=110); plt.close(fig)
    outbound = [r for r in rows if r['controller']['stage']=='review_out']
    selected_rows = [outbound[-1] if outbound else base,
                     max(rr, key=lambda r:r['review_whole_bend'].get('positive_turn_deg', 0.)), rows[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 6), constrained_layout=True)
    for ax, row in zip(axes, selected_rows):
        points = np.array(row['backbone_world_m'])
        ax.add_patch(Circle((0., 1.2), .0565, color='gray', alpha=.35))
        ax.plot(points[:, 1], points[:, 2], '-o', color='purple', ms=3, label='actual full backbone')
        first, last = row['controller']['window']
        ax.plot(points[first:last+2, 1], points[first:last+2, 2], '-o', color='royalblue', ms=3, label='enabled region')
        for contact in row['evaluation']['spatial']['contacts']:
            ax.plot(contact['position_world_m'][1], contact['position_world_m'][2], 'x', color='red')
        ax.set_title(f"t={row['time_s']:.2f}s\nnet transfer={(transfer(row)-baseline)*1000:.1f}mm")
        ax.set_xlim(-.3, .3); ax.set_ylim(.3, 1.4); ax.set_aspect('equal')
        ax.set_xlabel('world transverse Y m'); ax.set_ylabel('world Z m'); ax.legend(fontsize=7)
    fig.savefig(folder/'actual_sections.png', dpi=120); plt.close(fig)
    if render and not (folder/'actual_full.mp4').exists():
        render_actual(folder, rows, status, baseline)
    print(folder.name, json.dumps(dict(stages=summary['stage_results'], acceptance=summary['acceptance']), ensure_ascii=False), flush=True)


def render_actual(folder, rows, status, baseline):
    sim = ArcSimulation('straight', ArcParameters(**status['parameters']), -.5)
    for geom in sim.pipe_geoms:
        sim.model.geom_rgba[geom, 3] = .38
    camera = mujoco.MjvCamera(); camera.distance = 1.8
    times = [r['time_s'] for r in rows]
    selected = [0., 6., times[-1]]+[e['time_s'] for e in status['events']]
    selected += [max(rows, key=lambda r:r['evaluation']['section_turn_deg'])['time_s'],
                 max(rows, key=lambda r:r['review_whole_bend'].get('positive_turn_deg', 0.))['time_s']]
    indices = sorted(set(int(np.argmin(abs(np.array(times)-t)))//5*5 for t in selected))
    strip = []
    writer = cv2.VideoWriter(str(folder/'actual_full.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (1000, 300))
    if not writer.isOpened():
        raise RuntimeError('Video writer unavailable')
    try:
        with mujoco.Renderer(sim.model, height=300, width=500) as renderer:
            for i in range(0, len(rows), 5):
                r = rows[i]
                sim.data.qpos[:] = r['actual_qpos']; sim.data.qvel[:] = r['actual_qvel']
                mujoco.mj_forward(sim.model, sim.data)
                for geom in sim.robot_geoms:
                    link = int(sim.model.geom_bodyid[geom])-sim.model.body('head').id
                    joint = min(max(link-1, 0), 19)
                    sim.model.geom_rgba[geom] = [.12, .5, .9, 1] if r['enabled'][joint] else [.55, .3, .7, 1]
                camera.lookat[:] = [r['evaluation']['mean_x_m'], 0., .90]
                images = []
                for azimuth, elevation in ((75, -15), (0, 0)):
                    camera.azimuth, camera.elevation = azimuth, elevation
                    renderer.update_scene(sim.data, camera=camera)
                    images.append(renderer.render().copy())
                frame = np.concatenate(images, axis=1)
                controller = r['controller']
                label = f"{status['case']} t={r['time_s']:.1f}s {controller['stage']} active={sum(r['enabled'])} phase=0"
                cv2.putText(frame, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .53, (255, 255, 255), 1)
                delta = (transfer(r)-baseline)*1000
                cv2.putText(frame, f"actual net transfer {delta:.1f} mm; encoder length valid={controller['observation']['valid']}",
                            (10, 44), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if i in indices:
                    strip.append(frame)
    finally:
        writer.release()
    for page in range(math.ceil(len(strip)/5)):
        images = strip[page*5:(page+1)*5]
        fig, axes = plt.subplots(len(images), 1, figsize=(12, 3.5*len(images)), squeeze=False, constrained_layout=True)
        for ax, frame in zip(axes[:, 0], images):
            ax.imshow(frame); ax.axis('off')
        fig.savefig(folder/f'filmstrip_{page+1}.png', dpi=95); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    for path in sorted(REPORT.glob('*/status.json')):
        review(path.parent, args.render)
