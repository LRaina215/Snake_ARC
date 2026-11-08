"""Review EVERY saved actual state; replay/plots do not run new physics."""
import gzip
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from validation.review_continuous_feeding import upper_arc, whole_body_bend
from snake_control.section_geometry import forward
from .validate_mujoco import OUTPUT, CASES


def actual_lengths(points, height=1.2):
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    h = points[:, 2]-height
    peak = int(np.argmax(h))
    left = right = peak
    while left > 0 and h[left-1] > 0:
        left -= 1
    while right < len(h)-1 and h[right+1] > 0:
        right += 1
    if left == 0 or right == len(h)-1 or h[peak] <= 0:
        return np.full(3, np.nan)
    a = s[left-1]-h[left-1]*(s[left]-s[left-1])/(h[left]-h[left-1])
    b = s[right]-h[right]*(s[right+1]-s[right])/(h[right+1]-h[right])
    tail = s[-1]-b
    return np.array([a, tail, .5*(a-tail)])


def review_one(output, label):
    data = json.loads((output/(label+'.json')).read_text(encoding='utf-8'))
    report, rows = data['summary'], data['samples']
    trace = np.load(output/(label+'_trace.npz'))
    with gzip.open(output/(label+'_control.jsonl.gz'), 'rt', encoding='utf-8') as stream:
        controls = [json.loads(line) for line in stream]
    times, points = trace['time'], trace['backbone_world']
    length = np.array([actual_lengths(p) for p in points])
    arc = [upper_arc(dict(backbone_world_m=p.tolist())) for p in points]
    whole = [whole_body_bend(dict(backbone_world_m=p.tolist())) for p in points]
    turning = np.array([a.get('turn_deg', np.nan) for a in arc])
    gap = np.array([a.get('smooth_gap_fraction', np.nan) for a in arc])
    whole_turn = np.array([a.get('positive_turn_deg', np.nan) for a in whole])
    mask = trace['masks']
    transitions = np.diff(mask.astype(int), axis=0)
    first_time = times[0]
    motion_end = report['completed_actions'][-1]['time_s'] if report['completed_actions'] else times[-1]
    centre_events = [e for e in report.get('stage_events', []) if e.get('kind') == 'centre_coordinate_reached']
    if centre_events:
        motion_end = centre_events[-1]['time_s']
    held = times >= motion_end
    center = points.mean(axis=1)
    all_raw_turns = []
    for p in points:
        v = np.diff(p, axis=0)
        dot = np.sum(v[:-1]*v[1:], axis=1)/np.linalg.norm(v[:-1], axis=1)/np.linalg.norm(v[1:], axis=1)
        turn = np.arccos(np.clip(dot, -1., 1.))
        all_raw_turns.append(turn/(.5*(np.linalg.norm(v[:-1], axis=1)+np.linalg.norm(v[1:], axis=1))))
    curvature = np.array(all_raw_turns)
    completions = []
    for event in report['completed_actions']:
        f = event['snapshot']
        if event['kind'] != 'relative_goal':
            continue
        start = next(r for r in controls if r['action'] == event['action'] and
                     r['feeding']['baseline_m'] is not None)
        start_index = int(np.argmin(abs(times-start['time_s'])))
        end_index = int(np.argmin(abs(times-event['time_s'])))
        delta = 1000*(length[end_index]-length[start_index])
        completions.append(dict(action=event['action'], requested_mm=f['goal_delta_m']*1000,
            true_head_change_mm=float(delta[0]), true_tail_change_mm=float(delta[1]),
            true_transfer_mm=float(delta[2]),
            encoder_delta_mm=(f['observation']['transfer_coordinate_m']-f['baseline_m'])*1000,
            true_error_mm=float(delta[2]-1000*f['goal_delta_m'])))
    handover = report['handovers']
    instantaneous_preload_difference = max((abs(a-b) for event in handover[1:]
        for a,b in zip(event['preload_deg'], event['actual_deg'])), default=0.)
    # Hardware preloads the last measured sample, not an unavailable zero-age
    # angle at the instant of enabling. Keep both differences explicit.
    control_times = np.array([r['time_s'] for r in controls])
    preload_errors = []
    for event in handover[1:]:
        row = controls[int(np.argmin(abs(control_times-event['time_s'])))]
        inp = row['motor_input']
        if inp:
            preload_errors.extend(abs(value-inp['angles_deg'][j-1])
                for value,j in zip(event['preload_deg'], event['entering']))
    preload_error = max(preload_errors, default=0.)
    direction_errors, reconstruction_residuals = [], []
    for row in controls:
        inp, obs = row['motor_input'], row['feeding']['observation']
        if not inp or not obs or not obs['valid']:
            continue
        index = int(np.argmin(abs(times-inp['feedback_time_s'])))
        local = forward(np.radians(inp['angles_deg']), report['geometry']['link_lengths_m'])
        actual = points[index]
        a, b = local-local.mean(axis=0), actual-actual.mean(axis=0)
        u, _, vt = np.linalg.svd(a.T@b)
        correction = np.eye(3); correction[-1, -1] = np.linalg.det(u@vt)
        rotation = u@correction@vt
        residual = np.max(np.linalg.norm(a@rotation-b, axis=1))
        reconstruction_residuals.append(float(residual))
        down = np.asarray(obs['down_head'])@rotation
        direction_errors.append(float(np.degrees(np.arccos(np.clip(-down[2]/np.linalg.norm(down), -1., 1.)))))
    acceptance = dict(
        task_complete=report['completed'] and report['failure'] is None,
        actual_length_direction=all(e['true_transfer_mm']*e['requested_mm'] > 0 for e in completions),
        actual_length_error_le12mm=bool(completions) and all(abs(e['true_error_mm']) <= 12. for e in completions),
        open_arc=all(a['valid'] and a['crown_in_interval'] and 150. <= a['turn_deg'] <= 218. and
                     a['smooth_gap_fraction'] >= .4 for a in arc),
        whole_positive_turn_le218=bool(np.isfinite(whole_turn).all() and np.max(whole_turn) <= 218.),
        continuous_sampled_contact=all(r['support_n'] > .1 for r in rows),
        no_floor_or_self_contact=report['statistics']['floor_contact_steps'] == 0 and report['statistics']['self_contact_steps'] == 0,
        no_limit_excess=report['statistics']['limit_excess_max_rad'] == 0.,
        passive_motor_torque_zero=report['statistics']['passive_force_max_nm'] == 0.,
        entering_preloaded_measured_angle=preload_error < 1e-10,
        slew_limit=report['statistics']['max_shared_target_increment_deg'] <= .900001)
    acceptance = {k: bool(v) for k,v in acceptance.items()}
    result = dict(label=label, acceptance=acceptance, all_passed=all(acceptance.values()),
        length_completions=completions,
        upper_turn_deg=[float(np.nanmin(turning)), float(np.nanmax(turning))],
        minimum_open_gap=float(np.nanmin(gap)), whole_positive_turn_max_deg=float(np.nanmax(whole_turn)),
        curvature_polyline_max_per_m=float(curvature.max()),
        active_count=[int(mask.sum(axis=1).min()), int(mask.sum(axis=1).max())],
        enable_events_per_joint=(transitions > 0).sum(axis=0).tolist(),
        disable_events_per_joint=(transitions < 0).sum(axis=0).tolist(),
        preload_error_max_deg=preload_error,
        preload_vs_instantaneous_angle_max_deg=instantaneous_preload_difference,
        saturated_joint_step_fraction=report['statistics']['saturated_joint_steps']/report['statistics']['joint_steps'],
        max_axis_drift_mm=float(np.max(abs(center[:, 0]-center[0, 0]))*1000),
        final_hold_length_range_mm=(np.ptp(length[held], axis=0)*1000).tolist(),
        final_hold_axis_drift_mm=float(np.ptp(center[held, 0])*1000),
        max_tracking_error_deg=max(r['tracking_max_deg'] for r in rows),
        angle_down_vs_world_gravity_max_deg=max(direction_errors, default=None),
        feedback_rigid_fit_residual_max_m=max(reconstruction_residuals, default=None),
        worst_open_shape_time_s=float(times[int(np.nanargmin(gap))]),
        review_samples=len(times), actual_duration_s=float(times[-1]-first_time),
        measured_link_length_range_m=[float(np.min(np.linalg.norm(np.diff(points, axis=1), axis=2))),
                                      float(np.max(np.linalg.norm(np.diff(points, axis=1), axis=2)))],
        scope='full saved trajectory and 3s final physical hold; contact sampled at 50ms, torque/limits/collisions at 1ms; no hardware',
        notes=['Motor-off still has the original model passive damping 1.5; this is not proof of hardware friction.',
               'World-z shoulder crossings evaluate actual hanging length, independent of the angle-only controller.',
               'No T crossing or compliance change in this experiment.'])
    (output/(label+'_review.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True, constrained_layout=True)
    ax[0].plot(times, (length-length[0])*1000, label=['head', 'tail', 'transfer'])
    ax[0].set_ylabel('Actual change / mm'); ax[0].legend(ncol=3)
    ax[1].plot(times, turning, label='actual upper smooth arc')
    ax[1].plot(times, whole_turn, label='whole positive turn', alpha=.7)
    ax[1].axhline(218., color='red', ls='--'); ax[1].set_ylabel('Turn / deg'); ax[1].legend()
    ax[2].plot(times, gap, label='actual smooth opening / length')
    ax[2].axhline(.4, color='red', ls='--'); ax[2].set_ylabel('Opening'); ax[2].legend()
    ax[3].imshow(mask.T, aspect='auto', origin='lower', interpolation='nearest',
        extent=(times[0], times[-1], .5, 18.5), cmap='Blues', vmin=0, vmax=1)
    ax[3].set_ylabel('Enabled joint'); ax[3].set_yticks(range(1, 19, 2))
    ax[4].plot(times, [r['support_n'] for r in rows], label='pipe normal force / N')
    ax[4].plot(times, np.max(abs(trace['actuator_force']), axis=1), label='max actuator torque / Nm')
    ax[4].legend(); ax[4].set_xlabel('Simulation time / s')
    for event in report['commands']:
        for axis in ax:
            axis.axvline(event['time_s'], color='grey', alpha=.25)
        ax[0].text(event['time_s'], ax[0].get_ylim()[1], event['key'], rotation=90, va='top', fontsize=8)
    fig.suptitle(label+' — ACTUAL motion, all saved states')
    fig.savefig(output/(label+'_review.png'), dpi=130)
    plt.close(fig)
    # Complete-process section filmstrip, including worst opening and end hold.
    indexes = np.unique(np.r_[np.linspace(0, len(times)-1, 8).astype(int), np.nanargmin(gap)])
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    for axis, i in zip(axes.flat, indexes):
        theta = np.linspace(0, 2*np.pi, 129)
        axis.fill(.055*np.cos(theta), 1.2+.055*np.sin(theta), color='.8')
        for j in range(18):
            axis.plot(points[i, j:j+2, 1], points[i, j:j+2, 2],
                color='#1287b5' if mask[i, j] else '#9a58b4', marker='.', lw=2)
        axis.plot(points[i, -2:, 1], points[i, -2:, 2], color='#9a58b4', marker='.')
        axis.set_aspect('equal'); axis.set_xlim(-.3, .3); axis.set_ylim(.2, 1.45)
        axis.set_title(f'{times[i]:.2f}s turn={turning[i]:.1f} gap={gap[i]:.3f}')
    fig.savefig(output/(label+'_sections.png'), dpi=130)
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=OUTPUT)
    output = parser.parse_args().input
    results = []
    pairs = []
    for case in CASES:
        if not (output/(case+'_curve.json')).exists():
            continue
        for backend in ('legacy', 'curve'):
            results.append(review_one(output, case+'_'+backend))
        a, b = [np.load(output/(case+'_'+name+'_trace.npz')) for name in ('legacy', 'curve')]
        np.testing.assert_array_equal(a['time'], b['time'])
        pairs.append(dict(case=case,
            max_actual_joint_angle_difference_deg=float(np.degrees(np.max(abs(a['qpos'][:, 7:]-b['qpos'][:, 7:])))),
            max_actual_backbone_difference_m=float(np.max(np.linalg.norm(a['backbone_world']-b['backbone_world'], axis=2))),
            target_difference_max_deg=float(np.degrees(np.max(abs(a['targets']-b['targets'])))),
            masks_identical=bool(np.array_equal(a['masks'], b['masks'])),
            qvel_difference_max=float(np.max(abs(a['qvel']-b['qvel'])))))
    compatibility = bool(results) and all(p['masks_identical'] and
        p['max_actual_backbone_difference_m'] < .00001 and
        p['max_actual_joint_angle_difference_deg'] < .001 for p in pairs) and all(
        all(value for key,value in r['acceptance'].items() if key != 'actual_length_error_le12mm')
        for r in results)
    report = dict(runs=results, paired_comparison=pairs,
        compatibility_passed=bool(compatibility),
        absolute_physical_acceptance_passed=bool(results) and all(r['all_passed'] for r in results),
        distinction='Curve refactor reproduces the baseline; baseline length accuracy is separately reported, never relabeled as passed.')
    (output/'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
