"""Matched 18-joint acute-elbow run. World state is used only for evaluation
and to timestamp the simulated operator's one bend-entry command.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.acute_arc_transition import AcuteArcTransition, AcuteOptions
from snake_control.curve_gait import CurveGait, helix
from snake_control.simulation import backbone
from snake_control.vertical_scene import VerticalScene
from validation.vertical_arc_experiment import json_default


def dump(path, value):
    path.write_text(json.dumps(value, indent=1, default=json_default), encoding='utf-8')


class AcuteReview:
    def __init__(self, sim):
        self.sim = sim
        model = sim.model
        horizontal = model.geom('pipe_horizontal').id
        data = sim.data
        mujoco.mj_kinematics(model, data)
        self.corner_z = float(data.geom_xpos[horizontal, 2])
        self.material = np.r_[0., np.cumsum(sim.lengths)]
        self.route = model.numeric('pipe_centerline').data.reshape(-1, 4)[:, :3]
        self.horizontal = horizontal
        self.vertical = model.geom('pipe_vertical').id
        self.bend_meshes = []
        for geom in sorted(sim.pipe_geoms - {self.vertical, horizontal}):
            idx = model.geom_dataid[geom]
            adr = model.mesh_vertadr[idx]
            vertices = model.mesh_vert[adr:adr + model.mesh_vertnum[idx]]
            points = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            self.bend_meshes.append(dict(name=model.geom(geom).name, bounds_min_m=points.min(0), bounds_max_m=points.max(0)))
        self.exit_x = max(float(mesh['bounds_max_m'][0]) for mesh in self.bend_meshes)
        self.entry_z = min(float(mesh['bounds_min_m'][2]) for mesh in self.bend_meshes)
        self.weight = float(model.body_mass[sim.head:].sum() * abs(model.opt.gravity[2]))
        self.robot = set(np.flatnonzero(model.geom_bodyid >= sim.head))
        self.shells = []
        for geom in sorted(self.robot):
            if not model.geom_contype[geom]:
                continue
            mesh = model.geom_dataid[geom]
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                adr = model.mesh_vertadr[mesh]
                verts = model.mesh_vert[adr:adr + model.mesh_vertnum[mesh]]
            elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                verts = None
            else:
                raise ValueError(f'Unexpected robot envelope geometry {model.geom(geom).name}')
            self.shells.append((geom, verts))

    def last_envelope_x(self):
        m, d = self.sim.model, self.sim.data
        values = []
        for geom, verts in self.shells:
            axis = d.geom_xmat[geom].reshape(3, 3)[0]
            if verts is None:
                r, half, _ = m.geom_size[geom]
                offset = -abs(axis[2]) * half - r * np.linalg.norm(axis[:2])
            else:
                offset = (verts @ axis).min()
            values.append(d.geom_xpos[geom, 0] + offset)
        return float(min(values))

    def support(self):
        m, d = self.sim.model, self.sim.data
        normal = dict(vertical=0., elbow=0., horizontal=0.)
        up = dict(vertical=0., elbow=0., horizontal=0.)
        links = set()
        for i, c in enumerate(d.contact):
            a, b = int(c.geom1), int(c.geom2)
            if a in self.robot and b in self.sim.pipe_geoms:
                robot, pipe, sign = a, b, -1.
            elif b in self.robot and a in self.sim.pipe_geoms:
                robot, pipe, sign = b, a, 1.
            else:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            if force[0] <= .05:
                continue
            links.add(int(m.geom_bodyid[robot] - self.sim.head))
            key = 'horizontal' if pipe == self.horizontal else 'vertical' if pipe == self.vertical else 'elbow'
            normal[key] += force[0]
            up[key] += float(sign * (c.frame.reshape(3, 3).T @ force[:3])[2])
        return normal, up, sorted(links)

    def wrap_shape(self, points, start, stop, pre_bend, horizontal):
        # Dense samples are points on the *actual* robot's link polyline.
        coords = np.linspace(start, stop, 65)
        ring = np.column_stack([np.interp(coords, self.material, points[:, k]) for k in range(3)])
        mean = ring.mean(0)
        tilt = np.linalg.svd(ring - mean, full_matrices=False)[2][-1]
        if tilt[0] + tilt[2] < 0:
            tilt = -tilt
        if horizontal:
            axis = np.array([1., 0., 0.])
            center = np.array([mean[0], 0., self.corner_z])
        elif pre_bend:
            axis = np.array([0., 0., 1.])
            center = np.array([0., 0., mean[2]])
        else:
            # Find where the fitted ring plane intersects the known collision
            # route. Near the elbow the plane is allowed to tilt with the body.
            axis = tilt
            signed = (self.route - mean) @ axis
            intervals = np.flatnonzero((signed[:-1] * signed[1:] <= 0.) & (np.abs(np.diff(signed)) > 1e-8))
            if len(intervals):
                fraction = -signed[intervals] / np.diff(signed)[intervals]
                intersections = self.route[intervals] + fraction[:, None] * np.diff(self.route, axis=0)[intervals]
                center = intersections[np.argmin(np.linalg.norm(intersections - mean, axis=1))]
            else:
                center = self.route[np.argmin(abs(signed))]
        e = np.cross([0., 1., 0.], axis)
        e /= max(np.linalg.norm(e), 1e-9)
        f = np.cross(axis, e)
        radial = np.column_stack([(ring - center) @ e, (ring - center) @ f])
        angle = np.unwrap(np.arctan2(radial[:, 1], radial[:, 0]))
        axial = (ring - center) @ axis
        radius = np.linalg.norm(radial, axis=1)
        turn = abs(float(np.degrees(angle[-1] - angle[0])))
        span = float(np.ptp(axial))
        rms = float(np.std((ring - mean) @ tilt))
        gap = float(np.linalg.norm(ring[0] - ring[-1]))
        straight_pitch = float(abs(np.linalg.lstsq(np.c_[angle, np.ones(len(angle))], axial, rcond=None)[0][0]) * 2 * np.pi)
        return dict(wrap_deg=turn, span_m=span, plane_rms_m=rms, end_gap_m=gap,
                    minimum_radius_m=float(radius.min()), maximum_radius_m=float(radius.max()),
                    fitted_normal=tilt.tolist(), ring_center_m=mean.tolist(), fitted_pitch_m=straight_pitch,
                    ring_shape=bool(325. <= turn <= 395. and span <= .100 and rms <= .027 and radius.max() < .170))

    def sample(self, core, target, mask):
        m, d = self.sim.model, self.sim.data
        points = backbone(m, d)
        normal, up, links = self.support()
        last_x = self.last_envelope_x()
        row = self.sim.evaluate(target, mask)
        clear = last_x >= self.exit_x and normal['elbow'] + normal['vertical'] <= .1 and row['floor_force_n'] <= .1
        full_support = clear and up['horizontal'] >= .5 * self.weight
        wrap_end = core.wrap_start + core.helix_length
        end = int(np.searchsorted(self.material, wrap_end))
        pre_bend = bool(points[:end + 1, 2].max() < self.entry_z and normal['elbow'] < .1)
        ring = self.wrap_shape(points, core.wrap_start, wrap_end, pre_bend, clear)
        tangents = np.diff(points, axis=0) / self.sim.lengths[:, None]
        turns = np.arccos(np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1., 1.))
        return dict(time_s=float(d.time), stage=core.stage, last_envelope_x_m=last_x,
                    head_xyz_m=points[0].tolist(), tail_xyz_m=points[-1].tolist(),
                    pre_bend=pre_bend,
                    normal_n=normal, up_n=up, supporting_links=links,
                    clear_supported=bool(full_support), exit_attached=bool(clear and normal['horizontal'] > 1.),
                    maximum_curvature_per_m=float(np.max(turns / ((self.sim.lengths[:-1] + self.sim.lengths[1:]) * .5))),
                    **row, **ring)


def summarize(rows, points, statuses, mode, weight):
    ring_mode = mode == 'arc'
    stable_ring = [i for i, r in enumerate(rows) if r['stage'] in ('vertical_ring', 'bend_shift') and r['pre_bend']
                   and r['ring_shape'] and r['up_n']['vertical'] >= .5 * weight]
    last_stable = None
    for seq in np.split(stable_ring, np.flatnonzero(np.diff(stable_ring) != 1) + 1):
        if len(seq) >= 5:
            last_stable = seq[-1]
    physical_reference = None
    first_clear = None
    progress = 0.
    passed = None
    gait_clear = None
    full_pass = False
    episode_correct_crossing = episode_correct_continuation = False

    def marker_indices(status):
        start = status.get('wrap_start_m', 0.)
        s = np.cumsum(status['lengths_m'])
        return np.flatnonzero((s >= start) & (s <= start + status['wrap_length_m'])) + 1

    for i, row in enumerate(rows):
        attached = row.get('exit_attached', row['clear_supported'])
        correct_shape = row['ring_shape'] if ring_mode else row['fitted_pitch_m'] >= .095 and row['wrap_deg'] >= 365.
        frame_ok = correct_shape and not row['self_contacts']
        if row['clear_supported'] and physical_reference is None:
            physical_reference = i
            physical_markers = marker_indices(statuses[i])
            if first_clear is None:
                first_clear = i
            episode_correct_continuation = frame_ok
            episode_correct_crossing = frame_ok and (not ring_mode or (
                last_stable is not None and last_stable <= i and
                all(r['ring_shape'] and not r['self_contacts'] for r in rows[last_stable:i + 1])))
            if episode_correct_crossing and gait_clear is None:
                gait_clear = i
        if attached and physical_reference is not None:
            advance = float(np.mean(points[i][physical_markers, 0] - points[physical_reference][physical_markers, 0]))
            row['physical_progress_m'] = advance
            episode_correct_continuation &= frame_ok
            if episode_correct_continuation:
                progress = max(progress, advance)
                if advance >= .072:
                    passed = i if passed is None else passed
                    full_pass |= episode_correct_crossing
        else:
            physical_reference = None
            row['physical_progress_m'] = None
    stages = {}
    for name in dict.fromkeys(r['stage'] for r in rows):
        frames = [r for r in rows if r['stage'] == name]
        stages[name] = dict(time_s=[frames[0]['time_s'], frames[-1]['time_s']],
                            head_xz_delta_m=[frames[-1][key] - frames[0][key] for key in ('head_x_m', 'head_z_m')],
                            wrap_deg=[min(r['wrap_deg'] for r in frames), max(r['wrap_deg'] for r in frames)],
                            span_m=[min(r['span_m'] for r in frames), max(r['span_m'] for r in frames)],
                            ring_frames=sum(r['ring_shape'] for r in frames),
                            maximum_curvature_per_m=max(r['maximum_curvature_per_m'] for r in frames))
    return dict(physically_cleared=first_clear is not None,
                clear_time_s=None if first_clear is None else rows[first_clear]['time_s'],
                physical_progress_m=max((r['physical_progress_m'] for r in rows if r['physical_progress_m'] is not None), default=0.),
                correct_gait_cleared=gait_clear is not None,
                advance_m=progress, continued_72mm=passed is not None,
                complete_sequence=full_pass,
                continuation_time_s=None if passed is None else rows[passed]['time_s'],
                vertical_supported_ring_frames=len(stable_ring),
                last_stable_ring_time_s=None if last_stable is None else rows[last_stable]['time_s'], stages=stages,
                furthest_last_envelope_m=max(r['last_envelope_x_m'] for r in rows),
                max_tracking_deg=max(r['tracking_deg'] for r in rows), max_joint_deg=max(r['max_joint_deg'] for r in rows),
                max_unlimited_reference_deg=max(s['requested_max_deg'] for s in statuses),
                passive_max_nm=max(r['passive_actuator_nm'] for r in rows),
                floor_frames=sum(r['floor_force_n'] > .1 for r in rows),
                self_contact_frames=sum(bool(r['self_contacts']) for r in rows),
                saturated_frames=sum(bool(r['saturated_joints']) for r in rows))


def review_saved(folder):
    """Reapply the acceptance gates without changing raw records or rerunning physics."""
    folder = Path(folder)
    hypothesis = json.loads((folder / 'hypothesis.json').read_text(encoding='utf-8'))
    original = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    statuses = json.loads((folder / 'controller.json').read_text(encoding='utf-8'))
    model = folder / 'model.xml'
    sim = VerticalScene(hypothesis.get('scene', 'acute'), model if model.exists() else folder.parent / 'model.xml')
    evaluator = AcuteReview(sim)
    rows = original['samples']
    with np.load(folder / 'trace.npz') as trace:
        points = trace['points']
        enabled_count = np.count_nonzero(trace['masks'], axis=1)
    for row, body, status in zip(rows, points, statuses):
        start = status.get('wrap_start_m', 0.)
        stop = start + status['wrap_length_m']
        end = np.searchsorted(evaluator.material, stop)
        elbow, vertical = row['normal_n']['elbow'], row['normal_n']['vertical']
        before = body[:end + 1, 2].max() < evaluator.entry_z and elbow < .1
        clear = row['last_envelope_x_m'] >= evaluator.exit_x and elbow + vertical <= .1 and row['floor_force_n'] <= .1
        row.update(pre_bend=bool(before), clear_supported=bool(clear and row['up_n']['horizontal'] >= .5 * evaluator.weight),
                   exit_attached=bool(clear and row['normal_n']['horizontal'] > 1.))
        row.update(evaluator.wrap_shape(body, start, stop, before, clear))
    result = summarize(rows, points, statuses, folder.name, evaluator.weight)
    result.update(exit_x_m=evaluator.exit_x, failure=original['summary']['failure'],
                  enabled_count_min_max=[int(enabled_count.min()), int(enabled_count.max())],
                  total_pipe_normal_peak_n=max(sum(r['normal_n'].values()) for r in rows),
                  last_s=rows[-1]['time_s'], samples=len(rows))
    dump(folder / 'review.json', dict(summary=result, evaluation=[
        {key: r[key] for key in ('time_s', 'wrap_deg', 'span_m', 'plane_rms_m', 'ring_shape', 'clear_supported',
                                'exit_attached', 'last_envelope_x_m', 'physical_progress_m')} for r in rows]))
    return result


def run(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    sim = VerticalScene(args.scene)
    model, data = sim.model, sim.data
    review = AcuteReview(sim)
    source = root / 'source'
    source.mkdir()
    for name in ('snake_control/acute_arc_transition.py', 'snake_control/bent_helix_curve.py',
                 'snake_control/curve_gait.py', 'tools/build_vertical_arc_scene.py', __file__):
        shutil.copyfile(name, source / Path(name).name)
    options = replace(AcuteOptions(), ring_pitch_m=args.ring_pitch, ring_turns=args.ring_turns,
                      ring_radius_delta_m=args.ring_radius_delta, ring_feed_m_s=args.ring_feed,
                      reference_bend_radius_m=args.reference_radius, miter_section=args.miter_section,
                      lead_length_m=args.lead_length, corner_pitch_m=args.corner_pitch, exit_pitch_m=args.exit_pitch,
                      corner_turns=args.corner_turns, section_compression=args.section_compression)
    shutil.copyfile(sim.model_path, root / 'model.xml')
    dump(root / 'hypothesis.json', dict(case='acute', scene=args.scene,
         hypothesis='Only the compact ring may be able to clear the matched elbow',
         diameter_m=2 * model.geom_size[review.horizontal, 0], vertical_axis=[0, 0, 1], horizontal_axis=[1, 0, 0],
         corner_z_m=review.corner_z, entry_z_m=review.entry_z, exit_x_m=review.exit_x,
         meshes=review.bend_meshes, lengths_m=sim.lengths,
         friction=model.geom_friction[review.horizontal], mass_kg=model.body_mass[sim.head:].sum(),
         phase_rad=args.phase, options=options.__dict__,
         acceptance='actual last shell clears and supported by horizontal pipe; 72 mm after exit with correct shape',
         shape_limits=dict(turn_deg=[325,395], axial_span_m=.100, plane_rms_m=.027, max_radius_m=.170)))
    gait = CurveGait(18, lengths=sim.lengths)
    q = gait.targets([helix(gait.body_length, .094, .13)], args.phase)
    initial = sim.place_helix(q, 1., 0.)
    core = AcuteArcTransition(q, sim.lengths, options)
    mask = np.ones(18, bool)
    measured = q.copy()
    state = mujoco.mjtState.mjSTATE_INTEGRATION
    saved = np.empty(mujoco.mj_stateSize(model, state))
    common_records = ([], [], [], [], [], [], [], [])

    def sample(core, target, mask, records):
        rows, states, speeds, targets, masks, forces, statuses, points = records
        rows.append(review.sample(core, target, mask))
        states.append(data.qpos.copy())
        speeds.append(data.qvel.copy())
        targets.append(target.copy())
        masks.append(mask.copy())
        forces.append(data.actuator_force.copy())
        statuses.append(dict(core.status(), lengths_m=sim.lengths.tolist(), measured_rad=core.measured.tolist()))
        points.append(backbone(model, data))

    timestep = model.opt.timestep
    for tick in range(round(10. / timestep)):
        if tick % 200 == 0:
            measured = data.qpos[sim.drive.qadr].copy()
        if tick % 20 == 0:
            q, mask = core.step(.02, measured)
        sim.drive.apply_targets(data, q, mask)
        mujoco.mj_step(model, data)
        if tick % 50 == 0:
            sample(core, q, mask, common_records)
    mujoco.mj_getState(model, data, saved, state)
    np.save(root / 'matched_start.npy', saved)
    common_core, common_q, common_mask = core, q, mask
    common_climb = common_records[0][-1]['head_z_m'] - common_records[0][40]['head_z_m']
    print(f'common climb after settle: {common_climb:.3f} m', flush=True)
    for mode in args.modes:
        core = deepcopy(common_core)
        core.branch(mode)
        mujoco.mj_setState(model, data, saved, state)
        mujoco.mj_forward(model, data)
        q, mask = common_q.copy(), common_mask.copy()
        records = tuple(list(stream) for stream in common_records)
        events = []
        failure = None
        previous = None
        low_since = None
        for tick in range(round(10. / timestep), round(args.duration / timestep)):
            if tick % 200 == 0:
                measured = data.qpos[sim.drive.qadr].copy()
            if tick % 20 == 0:
                try:
                    nominal_entry = review.corner_z - options.reference_bend_radius_m - .055
                    if core.stage in ('vertical_helix', 'vertical_ring') and data.site('s_head').xpos[2] >= nominal_entry:
                        core.begin_bend(-.055)
                        events.append(dict(time_s=float(data.time), command='begin_bend',
                                           source='simulated operator observing the marked approach height'))
                    q, mask = core.step(.02, measured)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    failure = str(exc)
                    break
                if core.stage != previous:
                    print(mode, round(data.time, 2), core.stage, np.round(data.site('s_head').xpos, 3), flush=True)
                    previous = core.stage
            sim.drive.apply_targets(data, q, mask)
            mujoco.mj_step(model, data)
            if tick % 50 == 0:
                sample(core, q, mask, records)
                row = records[0][-1]
                if row['floor_force_n'] > .1:
                    failure = 'ground_contact'
                    break
                if row['head_z_m'] < .65:
                    low_since = float(data.time) if low_since is None else low_since
                    if data.time - low_since > 1.:
                        failure = 'lost_vertical_height'
                        break
                else:
                    low_since = None
        rows, states, speeds, targets, masks, forces, statuses, points = records
        out = root / mode
        out.mkdir()
        summary = summarize(rows, points, statuses, mode, review.weight)
        summary.update(failure=failure, common_climb_m=common_climb, initial=initial)
        dump(out / 'run.json', dict(summary=summary, samples=rows))
        dump(out / 'controller.json', statuses)
        dump(out / 'operator.json', events)
        shutil.copyfile(root / 'model.xml', out / 'model.xml')
        shutil.copyfile(root / 'hypothesis.json', out / 'hypothesis.json')
        np.savez_compressed(out / 'trace.npz', qpos=states, qvel=speeds, targets=targets, masks=masks,
                            forces=forces, time=[r['time_s'] for r in rows], points=points)
        print(mode, json.dumps(summary, default=json_default), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--review', nargs='+', type=Path)
    p.add_argument('--scene', choices=['acute', 'short'], default='acute')
    p.add_argument('--duration', type=float, default=185.)
    p.add_argument('--phase', type=float, default=.4)
    p.add_argument('--ring-pitch', type=float, default=AcuteOptions.ring_pitch_m)
    p.add_argument('--ring-turns', type=float, default=AcuteOptions.ring_turns)
    p.add_argument('--ring-radius-delta', type=float, default=AcuteOptions.ring_radius_delta_m)
    p.add_argument('--ring-feed', type=float, default=AcuteOptions.ring_feed_m_s)
    p.add_argument('--reference-radius', type=float, default=AcuteOptions.reference_bend_radius_m)
    p.add_argument('--miter-section', action='store_true')
    p.add_argument('--lead-length', type=float, default=0.)
    p.add_argument('--corner-pitch', type=float)
    p.add_argument('--exit-pitch', type=float)
    p.add_argument('--corner-turns', type=float)
    p.add_argument('--section-compression', type=float, default=0.)
    p.add_argument('--modes', nargs='+', choices=['helix', 'arc'], default=['helix', 'arc'])
    args = p.parse_args()
    if args.review:
        for folder in args.review:
            print(folder, json.dumps(review_saved(folder), default=json_default), flush=True)
    elif args.output:
        run(args)
    else:
        p.error('supply --output for a new run or --review for saved runs')
