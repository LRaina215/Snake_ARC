"""Audit saved actual states and render them; never advance dynamics.

Run from the existing mujoco Conda environment. The archived traces remain
unchanged; mj_forward below is only for rendering and geometric clearance.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
from types import SimpleNamespace
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
from codes.Rolling_Arc_Debug_Mujoco import DebugSession, DebugViewer
from snake_control.simulation import backbone
from snake_control.section_geometry import local_section
from codes import Rolling_Arc_Real as real

REPORT = Path(__file__).resolve().parent / 'reports' / 'debug_modes'


def load(name):
    with gzip.open(REPORT / name, 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def audit(rows, session):
    sim = session.sim
    joint_ids = sim.model.actuator_trnid[:, 0]
    limits = sim.model.jnt_range[joint_ids]
    q = np.array([r['actual_qpos'] for r in rows])[:, sim.drive.qadr]
    tau = np.array([r['actuator_force_nm'] for r in rows])
    enabled = np.array([r['enabled'] for r in rows], bool)
    support = np.array([r['support_up_n'] for r in rows])
    floor_id = sim.model.geom('floor').id
    minimum_height, minimum_curve_radius, floor_contacts = np.inf, np.inf, 0
    for row in rows:
        sim.data.qpos[:] = row['actual_qpos']
        sim.data.qvel[:] = row['actual_qvel']
        mujoco.mj_forward(sim.model, sim.data)
        section = local_section(sim.data.qpos[sim.drive.qadr], real.LINK_LENGTHS_M, *row['window'])
        radii = [segment.radius for segment in section['curve']
                 if 1 <= segment.joint <= row['window'][1] - row['window'][0] + 1 and segment.radius > 0]
        minimum_curve_radius = min(minimum_curve_radius, min(radii, default=np.inf))
        minimum_height = min(minimum_height, backbone(sim.model, sim.data)[:, 2].min())
        floor_contacts += any(
            floor_id in (int(c.geom1), int(c.geom2)) and
            bool({int(c.geom1), int(c.geom2)} & sim.robot_geoms)
            for c in sim.data.contact)
    return dict(
        samples=len(rows), finite_actual_state=all(
            np.isfinite(r['actual_qpos']).all() and np.isfinite(r['actual_qvel']).all() for r in rows),
        enabled_counts=sorted(set(enabled.sum(axis=1).tolist())),
        passive_actuator_force_max_nm=float(np.max(abs(tau[~enabled]), initial=0)),
        active_actuator_force_max_nm=float(np.max(abs(tau[enabled]))),
        torque_limit_nm=sim.p.torque_limit_nm,
        saturated_saved_samples=int(np.count_nonzero(np.any(
            (abs(tau) >= sim.p.torque_limit_nm - 1e-6) & enabled, axis=1))),
        saturation_scope='20 ms saved states; not all 1 ms physical substeps',
        joint_angle_abs_max_deg=float(np.degrees(abs(q).max())),
        joint_limit_excess_max_deg=float(np.degrees(np.maximum(
            np.maximum(limits[:, 0] - q, q - limits[:, 1]), 0)).max()),
        joint_limit_margin_min_deg=float(np.degrees(np.minimum(
            q - limits[:, 0], limits[:, 1] - q)).min()),
        support_up_sum_min_max_n=[float(support.sum(axis=1).min()), float(support.sum(axis=1).max())],
        body_turn_min_max_deg=[min(r['actual_body_geometry']['turn_deg'] for r in rows),
                               max(r['actual_body_geometry']['turn_deg'] for r in rows)],
        body_gap_min=min(r['actual_body_geometry']['gap_fraction'] for r in rows),
        body_plane_rms_max_m=max(r['actual_body_geometry']['fit_rms_m'] for r in rows),
        body_projected_smooth_curve_radius_min_m=float(minimum_curve_radius),
        sent_turn_max_deg=max(r['sliding']['sent_geometry']['turn_deg'] for r in rows),
        sent_gap_min=min(r['sliding']['sent_geometry']['gap_fraction'] for r in rows),
        sent_constraint_violations=rows[-1]['sliding']['sent_constraint_violations'],
        minimum_backbone_height_m=float(minimum_height),
        replay_floor_contact_samples=floor_contacts,
        replay_scope='geometric contact/height after mj_forward of saved qpos; no original floor force log',
        displacement_start_end_m=(np.array(rows[-1]['body_center_m']) - rows[0]['body_center_m']).tolist(),
        stage_y_peak_to_peak_m={stage: float(np.ptp([r['body_center_m'][1] for r in rows if r['stage'] == stage]))
                               for stage in ('no_force', 'right_force', 'recovery')},
        complete_events=[e for e in rows[-1]['sliding_events'] if e['kind'] == 'complete'])


def timeline(traces):
    fig, axes = plt.subplots(5, 2, figsize=(13, 12), sharex=True)
    for col, (title, rows) in enumerate(traces):
        t = np.array([r['time_s'] for r in rows])
        y = np.array([r['body_center_m'][1] for r in rows])
        axes[0, col].plot(t, 1000 * (y - y[0]), label='Actual backbone centroid Y')
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel('Side motion (mm)')
        cycles = [np.nan if r['sliding']['cycle_support_center_joint'] is None else
                  r['sliding']['cycle_support_center_joint'] for r in rows]
        axes[1, col].plot(t, cycles, label='Cycle-averaged support center')
        mid = np.array([sum(r['window']) / 2 for r in rows])
        axes[1, col].plot(t, mid - 1.25, '--', color='gray', label='Selection deadband')
        axes[1, col].plot(t, mid + 1.25, '--', color='gray')
        axes[1, col].set_ylabel('Joint index')
        axes[2, col].plot(t, [r['actual_body_geometry']['turn_deg'] for r in rows], label='Actual body turn')
        axes[2, col].axhline(150, color='gray', ls='--')
        axes[2, col].axhline(218, color='gray', ls='--')
        axes[2, col].set_ylabel('Body turn (deg)')
        for i, name in enumerate(('First enabled', 'Last enabled')):
            axes[3, col].step(t, [r['window'][i] for r in rows], label=name, where='post')
        axes[3, col].set_ylabel('Window joints')
        axes[4, col].plot(t, [r['perturb_force_world_n'][1] for r in rows], label='Force Y (N)')
        axes[4, col].plot(t, [3 * int(r['sliding']['phase_hold']) for r in rows], label='Handover phase hold x3')
        axes[4, col].set_ylabel('Force / hold')
        axes[4, col].set_xlabel('Simulation time (s)')
        for ax in axes[:, col]:
            ax.axvspan(14.2, 28.2, color='#f3c867', alpha=.15)
            ax.grid(alpha=.25)
            ax.legend(loc='best', fontsize=8)
    fig.suptitle('Saved actual motion: whole-body force vs endpoint pull (two diagnostics, not a baseline comparison)')
    fig.tight_layout()
    fig.savefig(REPORT / 'full_timeline.png', dpi=150)
    plt.close(fig)


def restore(session, row):
    sim, c = session.sim, session.controller
    sim.data.qpos[:] = row['actual_qpos']
    sim.data.qvel[:] = row['actual_qvel']
    sim.data.time = row['time_s']
    c.mode, c.phase = row['mode'], row['phase_rad']
    c.first, c.last = row['window']
    c.enabled, c.targets = row['enabled'], row['target_controller_deg']
    saved = row['sliding']
    c.sliding = SimpleNamespace(snapshot=lambda: saved)
    session.physics_started = True
    session.push_keys.clear()
    if row['perturb_active_bodies']:
        sign = 1 if row['perturb_force_world_n'][1] > 0 else -1
        session.push_keys.add(sign)
        session.push_targets[sign] = (sim.model.body(row['perturb_active_bodies'][0]).id, np.zeros(3))
    session.bus.enabled[:] = row['enabled']
    mujoco.mj_forward(sim.model, sim.data)
    session.colour_active()


def render_snapshots(rows, session):
    viewer = DebugViewer(session, visible=False)
    times = [14.2, rows[int(np.argmin([r['actual_body_geometry']['turn_deg'] for r in rows]))]['time_s'],
             next(r['time_s'] for r in rows if r['window'] == [7, 12]), 36.2]
    panels = []
    try:
        for index, time_s in enumerate(times):
            row = min(rows, key=lambda r: abs(r['time_s'] - time_s))
            restore(session, row)
            # Swap twice so the back buffer holds the same saved frame on Windows.
            viewer.render()
            viewer.render()
            rgb = np.empty((800, 1280, 3), np.uint8)
            mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, 1280, 800), viewer.context)
            picture = Image.fromarray(np.flipud(rgb))
            picture.save(REPORT / f'actual_{index + 1}.png')
            panel = ImageOps.expand(picture.resize((768, 480)), border=(0, 26, 0, 0), fill='white')
            ImageDraw.Draw(panel).text((12, 7), f'Saved actual state: {row["time_s"]:.2f} s', fill='black')
            panels.append(panel)
    finally:
        viewer.close()
    montage = Image.new('RGB', (1536, 1012), 'white')
    for i, panel in enumerate(panels):
        montage.paste(panel, ((i % 2) * 768, (i // 2) * 506))
    montage.save(REPORT / 'actual_montage.png')


def render_video(rows, session):
    """Optional local movie, 10 fps, using saved states only."""
    import cv2
    destination = REPORT.parents[1] / 'results' / 'debug_modes' / 'endpoint_pull_full.mp4'
    destination.parent.mkdir(parents=True, exist_ok=True)
    viewer = DebugViewer(session, visible=False)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'), 10, (960, 600))
    if not writer.isOpened():
        viewer.close()
        raise RuntimeError('Movie writer could not be opened')
    try:
        for row in rows[::5] + [rows[-1]]:
            restore(session, row)
            viewer.render()
            viewer.render()
            rgb = np.empty((800, 1280, 3), np.uint8)
            mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, 1280, 800), viewer.context)
            frame = cv2.resize(np.flipud(rgb), (960, 600))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        viewer.close()
    print(f'Saved-state movie: {destination}')


def main():
    endpoint = load('endpoint_pull_trace.jsonl.gz')
    whole = load('whole_body_trace.jsonl.gz')
    session = DebugSession()
    try:
        meta = session.sim.model.numeric('pipe_centerline').data.reshape(-1, 4)
        result = dict(scene='straight', pipe_diameter_min_max_m=(2 * meta[:, 3][[np.argmin(meta[:, 3]), np.argmax(meta[:, 3])]]).tolist(),
                      model_joint_limits_deg=np.degrees(session.sim.model.jnt_range[session.sim.model.actuator_trnid[:, 0]]).tolist(),
                      whole_body=audit(whole, session), endpoint_pull=audit(endpoint, session))
        (REPORT / 'full_state_audit.json').write_text(json.dumps(result, indent=2), encoding='utf8')
        timeline([('Uniform whole-body +Y force: no handover', whole), ('Outward/downward tail pull: one handover', endpoint)])
        render_snapshots(endpoint, session)
        if '--video' in sys.argv:
            render_video(endpoint, session)
        print(json.dumps(result, ensure_ascii=False))
    finally:
        session.close()


if __name__ == '__main__':
    main()
