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
from time import perf_counter

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.acute_arc_transition import AcuteArcTransition, AcuteOptions
from snake_control.curve_gait import CurveGait, helix
from snake_control.simulation import backbone
from snake_control.vertical_scene import VerticalScene
from snake_control.vertical_ring_manual import VerticalRingManual
from tools.build_vertical_arc_scene import GENERATED_STRAIGHT_SCENES, SHORT_BEND_RADIUS, build
from validation.vertical_arc_experiment import json_default
from validation.tro_helix_baseline import TroHelixBaseline, REFERENCE

STRAIGHT_SCENES = ('straight',) + GENERATED_STRAIGHT_SCENES
VERTICAL_FEATURES = ('variable', 'variable_return', 'round_to_square', 'round_to_ellipse', 'collar', 'twin')


def dump(path, value):
    path.write_text(json.dumps(value, indent=1, default=json_default), encoding='utf-8')


class AcuteReview:
    def __init__(self, sim):
        self.sim = sim
        model = sim.model
        self.straight = sim.scene in STRAIGHT_SCENES
        self.square = next((geom for geom in sim.pipe_geoms
                            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX), None)
        horizontal = None if self.straight else model.geom('pipe_horizontal').id
        data = sim.data
        mujoco.mj_kinematics(model, data)
        self.twin = tuple(model.geom(name).id for name in ('pipe_twin_small', 'pipe_twin_large')) if sim.scene == 'twin' else ()
        self.twin_centers = data.geom_xpos[list(self.twin), :2].copy()
        self.corner_z = None if self.straight else float(data.geom_xpos[horizontal, 2])
        self.material = np.r_[0., np.cumsum(sim.lengths)]
        self.route = model.numeric('pipe_centerline').data.reshape(-1, 4)[:, :3]
        self.horizontal = horizontal
        self.vertical = model.geom('pipe_vertical').id
        self.bend_meshes = []
        for geom in ([] if self.straight else sorted(sim.pipe_geoms - {self.vertical, horizontal})):
            idx = model.geom_dataid[geom]
            adr = model.mesh_vertadr[idx]
            vertices = model.mesh_vert[adr:adr + model.mesh_vertnum[idx]]
            points = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            self.bend_meshes.append(dict(name=model.geom(geom).name, bounds_min_m=points.min(0), bounds_max_m=points.max(0)))
        self.exit_x = None if self.straight else max(float(mesh['bounds_max_m'][0]) for mesh in self.bend_meshes)
        self.entry_z = None if self.straight else min(float(mesh['bounds_min_m'][2]) for mesh in self.bend_meshes)
        self.continuous_elbow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'pipe_bend_sdf')
        if self.continuous_elbow >= 0:
            self.entry_z, self.exit_x = map(float, model.numeric('bend_planes').data)
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

    def last_envelope(self, coordinate):
        m, d = self.sim.model, self.sim.data
        values = []
        for geom, verts in self.shells:
            axis = d.geom_xmat[geom].reshape(3, 3)[coordinate]
            if verts is None:
                r, half, _ = m.geom_size[geom]
                offset = -abs(axis[2]) * half - r * np.linalg.norm(axis[:2])
            else:
                offset = (verts @ axis).min()
            values.append(d.geom_xpos[geom, coordinate] + offset)
        return float(min(values))

    def support(self):
        m, d = self.sim.model, self.sim.data
        normal = dict(vertical=0., elbow=0., horizontal=0.)
        up = dict(vertical=0., elbow=0., horizontal=0.)
        normal_up = dict.fromkeys(normal, 0.)
        link_normal = {}
        corners = set()
        twin_normals = dict.fromkeys(self.twin, 0.)
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
            link = int(m.geom_bodyid[robot] - self.sim.head)
            link_normal[link] = link_normal.get(link, 0.) + force[0]
            if pipe == self.square and min(abs(c.pos[:2])) >= .65 * m.geom_size[pipe, 0]:
                corners.add((int(c.pos[0] > 0), int(c.pos[1] > 0)))
            key = 'vertical' if self.straight or pipe == self.vertical else 'horizontal' if pipe == self.horizontal else 'elbow'
            if pipe == self.continuous_elbow:
                # The swept collider also includes straight sections. The
                # tangent planes, rather than its mesh bounds, mark the bend.
                if c.pos[2] < self.entry_z:
                    key = 'vertical'
                elif c.pos[0] >= self.exit_x:
                    key = 'horizontal'
            frame = c.frame.reshape(3, 3)
            normal[key] += force[0]
            up[key] += float(sign * frame[:, 2] @ force[:3])
            normal_up[key] += float(sign * frame[0, 2] * force[0])
            if pipe in twin_normals:
                twin_normals[pipe] += force[0]
        return normal, up, normal_up, link_normal, sorted(corners), list(twin_normals.values())

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
        twin_turns = []
        if self.twin:
            for xy in self.twin_centers:
                x, y = (ring[:, :2] - xy).T
                winding = np.unwrap(np.arctan2(y, x))
                twin_turns.append(abs(float(np.degrees(winding[-1] - winding[0]))))
        return dict(wrap_deg=turn, span_m=span, plane_rms_m=rms, end_gap_m=gap, twin_turn_deg=twin_turns,
                    minimum_radius_m=float(radius.min()), maximum_radius_m=float(radius.max()),
                    fitted_normal=tilt.tolist(), ring_center_m=mean.tolist(), fitted_pitch_m=straight_pitch,
                    ring_shape=bool(325. <= turn <= 395. and span <= .100 and rms <= .027 and radius.max() < .170))

    def sample(self, core, target, mask):
        m, d = self.sim.model, self.sim.data
        points = backbone(m, d)
        normal, up, normal_up, link_normal, corners, twin_normals = self.support()
        last_x = self.last_envelope(0)
        row = self.sim.evaluate(target, mask)
        clear = (not self.straight and last_x >= self.exit_x and normal['elbow'] + normal['vertical'] <= .1
                 and row['floor_force_n'] <= .1)
        full_support = clear and up['horizontal'] >= .5 * self.weight
        wrap_end = core.wrap_start + core.helix_length
        end = int(np.searchsorted(self.material, wrap_end))
        pre_bend = self.straight or bool(points[:end + 1, 2].max() < self.entry_z and normal['elbow'] < .1)
        ring = self.wrap_shape(points, core.wrap_start, wrap_end, pre_bend, clear)
        tangents = np.diff(points, axis=0) / self.sim.lengths[:, None]
        turns = np.arccos(np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1., 1.))
        return dict(time_s=float(d.time), stage=core.stage, last_envelope_x_m=last_x,
                    last_envelope_z_m=self.last_envelope(2) if self.straight else None,
                    head_xyz_m=points[0].tolist(), tail_xyz_m=points[-1].tolist(),
                    pre_bend=pre_bend,
                    normal_n=normal, up_n=up, normal_up_n=normal_up,
                    supporting_links=sorted(link_normal), link_normal_n=link_normal,
                    square_corner_contacts=corners, twin_normal_n=twin_normals,
                    clear_supported=bool(full_support), exit_attached=bool(clear and normal['horizontal'] > 1.),
                    maximum_curvature_per_m=float(np.max(turns / ((self.sim.lengths[:-1] + self.sim.lengths[1:]) * .5))),
                    **row, **ring)


def record_self_contacts(sim, episodes):
    model, data = sim.model, sim.data
    timestep = model.opt.timestep
    for contact in data.contact:
        if contact.dist >= 0.:
            continue
        first, second = sorted(int(model.geom_bodyid[geom] - sim.head) for geom in contact.geom)
        if first < 0 or second - first <= 2:
            continue
        intervals = episodes.setdefault((first + 1, second + 1), [])
        if intervals and data.time - intervals[-1][1] <= 1.01 * timestep:
            intervals[-1][1] = float(data.time)
            intervals[-1][2] = max(intervals[-1][2], float(-contact.dist))
        else:
            intervals.append([float(data.time), float(data.time), float(-contact.dist)])


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
    physical_72mm = None
    physical_72mm_reference = None
    first_exit_72mm = None
    progress = 0.
    passed = None
    gait_clear = None
    full_pass = False
    episode_correct_crossing = episode_correct_continuation = False
    bend_start = next((i for i, r in enumerate(rows) if r['stage'] == 'bend_shift'), None)
    audit_from = last_stable if ring_mode else bend_start

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
            episode_clean_crossing = (audit_from is not None and audit_from <= i and
                all(not r['self_contacts'] and r['floor_force_n'] <= .1 for r in rows[audit_from:i + 1]))
            episode_correct_crossing = frame_ok and episode_clean_crossing and (
                not ring_mode or all(r['ring_shape'] for r in rows[audit_from:i + 1]))
            if episode_correct_crossing and gait_clear is None:
                gait_clear = i
        if attached and physical_reference is not None:
            advance = float(np.mean(points[i][physical_markers, 0] - points[physical_reference][physical_markers, 0]))
            row['physical_progress_m'] = advance
            if physical_72mm is None and advance >= .072:
                physical_72mm = i
                physical_72mm_reference = physical_reference
            episode_correct_continuation &= frame_ok
            if episode_correct_continuation:
                progress = max(progress, advance)
                if advance >= .072:
                    passed = i if passed is None else passed
                    if physical_reference == first_clear and first_exit_72mm is None:
                        first_exit_72mm = i
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
    crossing_collisions = (None if bend_start is None or physical_72mm is None else
                           sum(bool(r['self_contacts']) for r in rows[bend_start:physical_72mm + 1]))
    return dict(physically_cleared=first_clear is not None,
                clear_time_s=None if first_clear is None else rows[first_clear]['time_s'],
                physical_72mm_time_s=None if physical_72mm is None else rows[physical_72mm]['time_s'],
                physical_72mm_reference_s=(None if physical_72mm_reference is None else
                                            rows[physical_72mm_reference]['time_s']),
                crossing_self_contact_frames_to_physical_72mm=crossing_collisions,
                physical_progress_m=max((r['physical_progress_m'] for r in rows if r['physical_progress_m'] is not None), default=0.),
                correct_gait_cleared=gait_clear is not None,
                advance_m=progress, continued_72mm=passed is not None,
                complete_sequence=full_pass,
                continuation_time_s=None if passed is None else rows[passed]['time_s'],
                first_exit_continuation_time_s=(None if first_exit_72mm is None else rows[first_exit_72mm]['time_s']),
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
        masks = trace['masks']
        enabled_count = np.count_nonzero(masks, axis=1)
    for row, body, status in zip(rows, points, statuses):
        if evaluator.straight:
            continue
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
    if evaluator.straight:
        result['vertical_continuation'] = summarize_straight(
            rows, hypothesis['options']['roll_rate_rad_s'], hypothesis['feature_exit_z_m'], folder.name,
            square=evaluator.square is not None, twin=bool(evaluator.twin))
    result.update(exit_x_m=evaluator.exit_x, failure=original['summary']['failure'],
                  enabled_count_min_max=[int(enabled_count.min()), int(enabled_count.max())],
                  total_pipe_normal_peak_n=max(sum(r['normal_n'].values()) for r in rows),
                  last_s=rows[-1]['time_s'], samples=len(rows))
    if hypothesis['options'].get('progressive_tail_release') and folder.name == 'arc':
        changes = []
        for index in np.flatnonzero(np.any(masks[1:] != masks[:-1], axis=1)) + 1:
            changes.append(dict(time_s=rows[index]['time_s'], stage=rows[index]['stage'],
                                off=(np.flatnonzero(masks[index-1] & ~masks[index]) + 1).tolist(),
                                on=(np.flatnonzero(~masks[index-1] & masks[index]) + 1).tolist()))
        result['enable_changes'] = changes
    dump(folder / 'review.json', dict(summary=result, evaluation=[
        {key: r[key] for key in ('time_s', 'wrap_deg', 'span_m', 'plane_rms_m', 'ring_shape', 'clear_supported',
                                'exit_attached', 'last_envelope_x_m', 'physical_progress_m')} for r in rows]))
    return result


def summarize_straight(rows, rate, feature_exit, mode, square=False, twin=False):
    stage = 'vertical_ring' if mode == 'arc' else 'vertical_helix'
    ring = [r for r in rows if r['stage'] == stage]
    if not ring:
        return dict(ring_reached=False)
    height = np.array([r['ring_center_m'][2] for r in ring])
    clear = next((r for r in ring if feature_exit is not None and r['last_envelope_z_m'] >= feature_exit), None)
    period = 2. * np.pi / abs(rate)
    times = np.array([r['time_s'] for r in ring])
    shape = np.array([r['ring_shape'] if mode == 'arc' else
                      r['fitted_pitch_m'] >= .095 and r['wrap_deg'] >= 365. for r in ring])
    if twin:
        shape &= np.array([min(r['twin_turn_deg']) >= 325. for r in ring])
    above = np.array([feature_exit is None or r['last_envelope_z_m'] >= feature_exit for r in ring])
    contact = np.array([bool(r['supporting_links']) for r in ring])
    sustained_clear_time = None
    if feature_exit is not None and above[-1] and contact[-1]:
        interruptions = np.flatnonzero(~(above & contact))
        clear_index = int(interruptions[-1]) + 1 if len(interruptions) else 0
        sustained_clear_time = float(times[clear_index])
    collision_free = np.array([not r['self_contacts'] and r['floor_force_n'] <= .1 for r in ring])
    allowed = np.flatnonzero(shape) if feature_exit is None else np.flatnonzero(above)
    continuation = None
    sustained = None
    for begin in allowed:
        end = int(np.searchsorted(times, times[begin] + period))
        if end >= len(ring):
            break
        window = ring[begin:end + 1]
        corners = sorted({tuple(corner) for r in window for corner in r.get('square_corner_contacts', [])})
        twin_contact = bool(np.all(np.max([r['twin_normal_n'] for r in window], axis=0) >= 1.)) if twin else None
        candidate = dict(time_s=[float(times[begin]), float(times[end])],
                            net_up_m=float(height[end] - height[begin]), gait_shape_all=bool(shape[begin:end + 1].all()),
                            contact_all=bool(contact[begin:end + 1].all()),
                            square_corners=corners if square else None, twin_contact_both=twin_contact,
                            collision_free_all=bool(collision_free[begin:end + 1].all()),
                            above_feature_all=bool(above[begin:end + 1].all()))
        valid = (candidate['gait_shape_all'] and candidate['contact_all'] and candidate['collision_free_all'] and
                 candidate['above_feature_all'] and candidate['net_up_m'] > 0. and
                 (not square or len(corners) == 4) and (not twin or twin_contact))
        if feature_exit is not None and continuation is None:
            continuation = candidate
        if valid:
            sustained = candidate
            if feature_exit is None:
                continuation = candidate
            break
    return dict(ring_reached=True, ring_time_s=times[0], ring_net_up_m=float(height[-1] - height[0]),
                maximum_backslide_m=float(np.max(np.maximum.accumulate(height) - height)),
                minimum_normal_n=min(r['normal_n']['vertical'] for r in ring),
                maximum_normal_n=max(r['normal_n']['vertical'] for r in ring),
                end_envelope_z_m=ring[-1]['last_envelope_z_m'],
                first_clear_time_s=None if clear is None else clear['time_s'],
                sustained_clear_time_s=sustained_clear_time, continuation=continuation,
                sustained_continuation=sustained)


class ReturnSeamTrial(AcuteArcTransition):
    """Expose the existing wider curve search while the measured ring is enlarged."""

    @classmethod
    def attach(cls, core):
        """Keep the controller installed by a paired experiment."""
        class WiderReturnSeam(type(core)):
            extra_vertical_tip_choices = cls.extra_vertical_tip_choices

        core.__class__ = WiderReturnSeam

    def extra_vertical_tip_choices(self, vertical):
        return (vertical and self.tip_gap_m < self.section_ring.shell_diameter and
                (self.ring_radius_offset < .001 or self.vertical_advance_trim < .001))


def run(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    model_copy = root / 'model.xml'
    short_radius = SHORT_BEND_RADIUS if args.short_bend_radius is None else args.short_bend_radius
    if args.recorded_model is not None:
        shutil.copyfile(args.recorded_model, model_copy)
        if args.short_elbow_sdf:
            shutil.copyfile(args.recorded_model.with_suffix('.obj'), model_copy.with_suffix('.obj'))
    elif args.scene in GENERATED_STRAIGHT_SCENES:
        build(args.scene, model_copy, taper_length_m=args.taper_length, upper_diameter_m=args.upper_pipe_diameter,
              lower_diameter_m=args.lower_pipe_diameter,
              outlet_height_m=args.vertical_outlet_height, embed_return_outlet=args.embed_return_outlet)
    elif args.short_bend_radius is not None or args.short_elbow_sdf:
        build(args.scene, model_copy, short_bend_radius_m=short_radius,
              short_elbow_sdf=args.short_elbow_sdf)
    sim = VerticalScene(args.scene, recorded_model=model_copy if model_copy.exists() else None)
    model, data = sim.model, sim.data
    review = AcuteReview(sim)
    source = root / 'source'
    source.mkdir()
    for name in ('snake_control/acute_arc_transition.py', 'snake_control/bent_helix_curve.py',
                 'snake_control/curve_gait.py', 'snake_control/vertical_ring_manual.py', 'snake_control/vertical_scene.py',
                 'snake_control/section_ring_curve.py',
                 'snake_control/section_geometry.py', 'snake_control/section_reach.py', 'snake_control/phase_balanced_count.py',
                 'snake_control/section_effort.py',
                 'tools/build_vertical_arc_scene.py', __file__):
        shutil.copyfile(name, source / Path(name).name)
    if 'tro' in args.modes:
        shutil.copyfile(REFERENCE, source / 'tro_original_pole_compliance.py')
        shutil.copyfile(Path(__file__).with_name('tro_helix_baseline.py'), source / 'tro_helix_baseline.py')
    options = replace(AcuteOptions(), ring_pitch_m=args.ring_pitch, ring_turns=args.ring_turns,
                      ring_radius_delta_m=args.ring_radius_delta, ring_feed_m_s=args.ring_feed,
                      reference_bend_radius_m=args.reference_radius, miter_section=args.miter_section,
                      lead_length_m=args.lead_length, corner_pitch_m=args.corner_pitch, exit_pitch_m=args.exit_pitch,
                      corner_turns=args.corner_turns, section_compression=args.section_compression,
                      corner_shape_delay_m=args.corner_shape_delay, exit_shape_delay_m=args.exit_shape_delay,
                      encoder_spacing=args.encoder_spacing,
                      encoder_compaction=args.encoder_compaction,
                      encoder_tip_clearance=args.encoder_tip_clearance, encoder_feed_sync=args.encoder_feed_sync,
                      section_shell_diameter_m=args.section_shell_diameter,
                      section_endpoint_allowance=args.section_endpoint_allowance,
                      section_handoff_from_tilt=args.section_handoff_from_tilt,
                      section_keep_curvature=args.section_keep_curvature,
                      progressive_tail_release=args.progressive_tail_release,
                      section_before_bend=args.section_before_bend,
                      section_seam_clearance=args.section_seam_clearance,
                      outlet_end_correction=args.outlet_end_correction or args.outlet_use_spare_length,
                      outlet_use_spare_length=args.outlet_use_spare_length,
                      section_feedback_weight=args.section_feedback_weight, section_feedback_gain=args.section_feedback_gain,
                      section_measured_carrier=args.section_measured_carrier,
                      section_measured_tightening=args.section_measured_tightening,
                      section_measured_smoothing_links=args.section_measured_smoothing_links,
                      section_measured_phase_pair=args.section_measured_phase_pair,
                      section_measured_end_splice=args.section_measured_end_splice,
                      section_measured_reach=args.section_measured_reach,
                      section_bearing_outline_feedback=args.section_bearing_outline_feedback,
                      section_bearing_outline_phase_balance=args.section_bearing_outline_phase_balance,
                      section_measured_closed_outline=args.section_measured_closed_outline,
                      vertical_use_spare_length=args.vertical_use_spare_length,
                      vertical_adaptive_material=args.vertical_adaptive_material,
                      vertical_material_radius_gain=args.vertical_material_radius_gain,
                      vertical_material_handoff_m=args.vertical_material_handoff,
                      vertical_material_minimum_wrap_deg=args.vertical_material_minimum_wrap,
                      vertical_material_max_extra_m=args.vertical_material_max_extra,
                      vertical_material_budget_links=args.vertical_material_budget_links,
                      vertical_extend_overlap_ramp=args.vertical_extend_overlap_ramp,
                      vertical_overlap_redistribution_m=args.vertical_overlap_redistribution,
                      vertical_phase_material_handoff=args.vertical_phase_material_handoff,
                      vertical_material_wrap_only=args.vertical_material_wrap_only,
                      vertical_material_return_phase_rad=args.vertical_material_return_phase,
                      vertical_return_advance_m=args.vertical_return_advance,
                      vertical_return_overlap_trim_m=args.vertical_return_overlap_trim,
                      vertical_return_crown_m=args.vertical_return_crown,
                      section_bearing_advance_m=args.section_bearing_advance,
                      section_flat_advance_weight=args.section_flat_advance_weight,
                      vertical_end_clearance=args.vertical_end_clearance,
                      section_overlap_clearance=args.section_overlap_clearance,
                      section_head_transition_links=args.section_head_transition,
                      section_feedback_during_release=args.section_feedback_during_release,
                      section_preserve_mean_turn=args.section_preserve_mean_turn,
                      section_local_handoff=args.section_local_handoff,
                      bend_section_radius_trim_m=args.bend_section_radius_trim)
    recorded_options = getattr(args, 'recorded_options', None)
    if recorded_options is not None:
        saved = json.loads(recorded_options.read_text(encoding='utf-8'))
        options = AcuteOptions(**saved['options'])
    if not model_copy.exists():
        shutil.copyfile(sim.model_path, model_copy)
    route = model.numeric('pipe_centerline').data.reshape(-1, 4)
    feature_exit = float(route[-2, 2]) if args.scene in VERTICAL_FEATURES else None
    ellipse_axes = model.numeric('pipe_ellipse_semiaxes').data.copy() if args.scene == 'round_to_ellipse' else None
    pipe = review.vertical if review.straight else review.horizontal
    dump(root / 'hypothesis.json', dict(case='straight' if review.straight else 'acute', scene=args.scene,
         hypothesis=('The measured section maintains ring shape and vertical grip through the vertical course' if review.straight
                     else 'The selected curve can carry the whole robot through the elbow and keep its gait on the outlet'),
         diameter_m=2 * route[0, 3] if review.square != pipe else None,
         square_side_m=None if review.square is None else 2 * model.geom_size[review.square, 0],
         ellipse_diameters_m=None if ellipse_axes is None else 2 * ellipse_axes,
         twin_diameters_m=[2. * model.geom_size[i, 0] for i in review.twin], twin_centers_m=review.twin_centers,
         route_radii_m=np.unique(route[:, 3]), feature_exit_z_m=feature_exit,
         vertical_outlet_height_m=float(route[-1, 2]) if review.straight else None,
         embed_return_outlet=args.embed_return_outlet,
         short_bend_radius_m=short_radius if args.scene == 'short' else None,
         short_elbow_sdf=args.short_elbow_sdf,
         start_head_height_m=args.start_head_height,
         vertical_axis=[0, 0, 1], horizontal_axis=[1, 0, 0],
         corner_z_m=review.corner_z, entry_z_m=review.entry_z, exit_x_m=review.exit_x,
         meshes=review.bend_meshes, lengths_m=sim.lengths,
         friction=model.geom_friction[pipe], mass_kg=model.body_mass[sim.head:].sum(),
         phase_rad=args.phase, initial_helix_radius_m=args.initial_helix_radius, options=options.__dict__,
         manual_takeover=args.manual_takeover,
         return_seam_trial=args.return_seam_trial,
         acceptance=('positive net ascent and a supported intact ring for one rolling period after formation/feature exit'
                     if review.straight else
                     'actual last shell clears and supported by horizontal pipe; 72 mm after exit with correct shape'),
         shape_limits=dict(turn_deg=[325,395], axial_span_m=.100, plane_rms_m=.027, max_radius_m=.170)))
    gait = CurveGait(18, lengths=sim.lengths)
    q = gait.targets([helix(gait.body_length, args.initial_helix_radius, .13)], args.phase)
    initial = sim.place_helix(q, args.start_head_height, 0.)
    core = AcuteArcTransition(q, sim.lengths, options)
    mask = np.ones(18, bool)
    measured = q.copy()
    state = mujoco.mjtState.mjSTATE_INTEGRATION
    saved = np.empty(mujoco.mj_stateSize(model, state))
    common_records = ([], [], [], [], [], [], [], [])
    inspect_contacts = getattr(args, 'audit_self_contacts', False)
    contact_ranges = {}

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
    common_work = 0.
    for tick in range(round(10. / timestep)):
        if tick % 200 == 0:
            measured = data.qpos[sim.drive.qadr].copy()
        if tick % 20 == 0:
            q, mask = core.step(.02, measured)
        sim.drive.apply_targets(data, q, mask)
        mujoco.mj_step(model, data)
        if inspect_contacts:
            record_self_contacts(sim, contact_ranges)
        common_work += timestep * float(np.maximum(0., data.actuator_force * data.qvel[sim.drive.vadr]).sum())
        if tick % 50 == 0:
            sample(core, q, mask, common_records)
    mujoco.mj_getState(model, data, saved, state)
    np.save(root / 'matched_start.npy', saved)
    common_core, common_q, common_mask = core, q, mask
    initial_contacts = deepcopy(contact_ranges)
    common_climb = common_records[0][-1]['head_z_m'] - common_records[0][40]['head_z_m']
    print(f'common climb after settle: {common_climb:.3f} m', flush=True)
    branches = [(mode, effort) for mode in args.modes
                for effort in (args.section_effort if mode == 'arc' and args.section_effort else (None,))]
    for mode, effort in branches:
        contact_ranges = deepcopy(initial_contacts)
        label = mode if effort is None else f'{mode}_{effort}'
        branch_options = replace(options, section_effort_mode=effort)
        mujoco.mj_setState(model, data, saved, state)
        mujoco.mj_forward(model, data)
        q, mask = common_q.copy(), common_mask.copy()
        records = tuple(list(stream) for stream in common_records)
        events = []
        manual = None
        tro = TroHelixBaseline(sim.lengths) if mode == 'tro' else None
        if args.manual_takeover and mode == 'arc':
            initial_feedback = np.degrees(data.qpos[sim.drive.qadr])
            manual = VerticalRingManual(initial_feedback, sim.lengths, branch_options, helix_targets_deg=np.degrees(q))
            if args.return_seam_trial:
                ReturnSeamTrial.attach(manual.core)
            manual.toggle_continuous()
            core = manual.core
            events.append(dict(time_s=float(data.time), command='takeover_and_continuous',
                               source='hardware-facing command adapter'))
        else:
            core = deepcopy(common_core)
            core.branch(mode)
        failure = None
        previous = None
        low_since = None
        feedback_time = float(data.time)
        motor_effort = None
        effort_sum = np.zeros(model.nu)
        effort_samples = np.zeros(model.nu, dtype=int)
        branch_work = 0.
        control_us = []
        for tick in range(round(10. / timestep), round(args.duration / timestep)):
            if tick % 200 == 0:
                measured = data.qpos[sim.drive.qadr].copy()
                feedback_time = float(data.time)
                if effort is not None:
                    motor_effort = np.divide(effort_sum, effort_samples * sim.drive.p.torque_limit_nm,
                                             out=np.zeros_like(effort_sum), where=effort_samples > 0)
                    effort_sum.fill(0.)
                    effort_samples.fill(0)
            if tick % 20 == 0:
                try:
                    if (not review.straight and tro is None and core.stage in ('vertical_helix', 'vertical_ring') and
                            data.site('s_head').xpos[2] >= review.corner_z - options.reference_bend_radius_m - .055):
                        if manual is None:
                            core.begin_bend(-.055)
                        else:
                            manual.begin_bend()
                        events.append(dict(time_s=float(data.time), command='begin_bend',
                                           source='simulated operator observing the marked approach height'))
                    if core.stage == 'bend_shift' and options.outlet_end_correction and mode == 'arc':
                        latest = records[0][-1]
                        normal = latest['normal_n']
                        if (latest['head_x_m'] > review.exit_x + core.wrap_radius and
                                normal['horizontal'] > 1. and normal['vertical'] + normal['elbow'] <= .1):
                            confirmed = core.confirm_outlet() if manual is None else manual.confirm_outlet()
                            if confirmed:
                                events.append(dict(time_s=float(data.time), command='confirm_outlet',
                                                   source='simulated operator observing ring on horizontal straight'))
                    control_start = perf_counter()
                    if tro is not None:
                        core.measured = measured
                        core.phase += core.handedness * options.roll_rate_rad_s * .02
                        desired = tro.step(measured, core.phase, q, feedback_time, float(data.time))
                        if desired is not None:
                            core.target_angle_max = float(np.degrees(max(abs(desired))))
                            core.apply_reference(desired, .02, mask)
                            q = core.targets.copy()
                    elif manual is None:
                        q, mask = core.step(.02, measured, feedback_time)
                    else:
                        manual.advance(.02, float(data.time), feedback_time, np.degrees(measured),
                                       motor_effort=motor_effort)
                        q, mask = np.radians(manual.targets), manual.enabled
                    control_us.append((perf_counter() - control_start) * 1.e6)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    failure = str(exc)
                    break
                if core.stage != previous:
                    print(label, round(data.time, 2), core.stage, np.round(data.site('s_head').xpos, 3), flush=True)
                    previous = core.stage
            sim.drive.apply_targets(data, q, mask)
            try:
                mujoco.mj_step(model, data)
            except mujoco.FatalError as exc:
                failure = f'physics_error: {exc}'
                events.append(dict(time_s=float(data.time), command='physics_failure', reason=str(exc)))
                break
            if inspect_contacts:
                record_self_contacts(sim, contact_ranges)
            if effort is not None:
                effort_sum[mask] += abs(data.actuator_force[mask])
                effort_samples += mask
            branch_work += timestep * float(np.maximum(0., data.actuator_force * data.qvel[sim.drive.vadr]).sum())
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
        out = root / label
        out.mkdir()
        summary = summarize(rows, points, statuses, mode, review.weight)
        if review.straight:
            summary['vertical_continuation'] = summarize_straight(rows, options.roll_rate_rad_s, feature_exit, mode,
                                                                  square=review.square is not None, twin=bool(review.twin))
        if tro is not None:
            summary['tro_status'] = tro.controller.status
        summary.update(failure=failure, common_climb_m=common_climb, initial=initial,
                       positive_mechanical_work_j=common_work + branch_work, branch_positive_work_j=branch_work,
                       control_tick_us=None if not control_us else np.percentile(control_us, [50, 95, 99]))
        dump(out / 'run.json', dict(summary=summary, samples=rows))
        dump(out / 'controller.json', statuses)
        dump(out / 'operator.json', events)
        if inspect_contacts:
            dump(out / 'self_contact_episodes.json',
                 [dict(links=links, intervals_s_depth_m=intervals)
                  for links, intervals in sorted(contact_ranges.items())])
        if options.progressive_tail_release and mode == 'arc':
            dump(out / 'tail_release.json', [event for event in core.events if 'released_joint' in event])
        shutil.copyfile(root / 'model.xml', out / 'model.xml')
        if args.short_elbow_sdf:
            shutil.copyfile(root / 'model.obj', out / 'model.obj')
        if effort is None:
            shutil.copyfile(root / 'hypothesis.json', out / 'hypothesis.json')
        else:
            meta = json.loads((root / 'hypothesis.json').read_text(encoding='utf-8'))
            meta['options'] = branch_options.__dict__
            meta['simulated_effort'] = 'mean abs actuator torque over preceding 200 ms of enabled samples, divided by torque limit'
            dump(out / 'hypothesis.json', meta)
        np.savez_compressed(out / 'trace.npz', qpos=states, qvel=speeds, targets=targets, masks=masks,
                            forces=forces, time=[r['time_s'] for r in rows], points=points)
        print(label, json.dumps(summary, default=json_default), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--recorded-model', type=Path, help='Reuse the exact collision XML from a prior trial')
    p.add_argument('--audit-self-contacts', action='store_true')
    p.add_argument('--review', nargs='+', type=Path)
    p.add_argument('--scene', choices=('acute', 'short') + STRAIGHT_SCENES, default='acute')
    p.add_argument('--duration', type=float, default=185.)
    p.add_argument('--phase', type=float, default=.4)
    p.add_argument('--initial-helix-radius', type=float, default=.094)
    p.add_argument('--start-head-height', type=float, default=1.,
                   help='Starting height of the placed vertical Helix head, in metres')
    p.add_argument('--recorded-options', type=Path,
                   help='Reuse the complete AcuteOptions from a saved hypothesis.json')
    p.add_argument('--taper-length', type=float, default=.15)
    p.add_argument('--upper-pipe-diameter', type=float, default=.140)
    p.add_argument('--lower-pipe-diameter', type=float, default=.110,
                   help='Inlet diameter of a tapered vertical pipe in metres')
    p.add_argument('--short-bend-radius', type=float,
                   help='Generate a 90-degree short elbow with this centreline radius in metres')
    p.add_argument('--short-elbow-sdf', action='store_true',
                   help='Use one continuous collision surface across the tight elbow and its tangent joins')
    p.add_argument('--vertical-outlet-height', type=float, default=4.,
                   help='Top of the straight outlet above a variable-diameter transition, in metres')
    p.add_argument('--embed-return-outlet', action='store_true',
                   help='Place the outlet cylinder cap inside the shrinking transition')
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
    p.add_argument('--corner-shape-delay', type=float, default=0.)
    p.add_argument('--exit-shape-delay', type=float, default=0.)
    p.add_argument('--encoder-spacing', action='store_true')
    p.add_argument('--encoder-compaction', action='store_true')
    p.add_argument('--encoder-tip-clearance', action='store_true')
    p.add_argument('--encoder-feed-sync', action='store_true')
    p.add_argument('--section-shell-diameter', type=float)
    p.add_argument('--section-endpoint-allowance', action='store_true')
    p.add_argument('--section-handoff-from-tilt', action='store_true')
    p.add_argument('--section-keep-curvature', action='store_true')
    p.add_argument('--progressive-tail-release', action='store_true')
    p.add_argument('--section-before-bend', action='store_true')
    p.add_argument('--section-seam-clearance', action='store_true')
    p.add_argument('--outlet-end-correction', action='store_true')
    p.add_argument('--outlet-use-spare-length', action='store_true')
    p.add_argument('--section-feedback-weight', type=float, default=0.)
    p.add_argument('--section-feedback-gain', type=float, default=.02)
    p.add_argument('--section-measured-carrier', action='store_true',
                   help='Reconstruct the spatial carrier from the measured planar section')
    p.add_argument('--section-measured-tightening', type=float, default=AcuteOptions.section_measured_tightening,
                   help='Fractional turn increase after normalizing the measured carrier winding')
    p.add_argument('--section-measured-smoothing-links', type=float, default=0.,
                   help='Measured turn averaging width in nominal joint spacings')
    p.add_argument('--section-measured-phase-pair', action='store_true',
                   help='Compensate alternating-axis projection bias with quadrature reference profiles')
    p.add_argument('--section-measured-end-splice', action='store_true',
                   help='Join measured curvature to the axial transfer geometry near each end')
    p.add_argument('--section-measured-reach', action='store_true',
                   help='Allocate measured ring curvature and material within the joint travel available')
    p.add_argument('--section-bearing-outline-feedback', action='store_true',
                   help='Deform the existing bearing outline with matched encoder feedback')
    p.add_argument('--section-bearing-outline-phase-balance', action='store_true',
                   help='Fit the persistent outline separately from roll harmonics')
    p.add_argument('--section-measured-closed-outline', action='store_true',
                   help='Rebuild a periodic measured outline with radial preload')
    p.add_argument('--section-effort', nargs='+', choices=('local', 'global', 'fused'),
                   help='Compare optional torque-proxy feedback modes on the same vertical ARC start')
    p.add_argument('--vertical-use-spare-length', action='store_true')
    p.add_argument('--vertical-adaptive-material', action='store_true')
    p.add_argument('--vertical-material-radius-gain', type=float, default=AcuteOptions.vertical_material_radius_gain)
    p.add_argument('--vertical-material-handoff', type=float, default=0.)
    p.add_argument('--vertical-material-minimum-wrap', type=float, default=0.)
    p.add_argument('--vertical-material-max-extra', type=float, default=AcuteOptions.vertical_material_max_extra_m)
    p.add_argument('--vertical-material-budget-links', type=float, default=AcuteOptions.vertical_material_budget_links)
    p.add_argument('--vertical-extend-overlap-ramp', action='store_true')
    p.add_argument('--vertical-overlap-redistribution', type=float, default=0.,
                   help='Maximum radius change in metres when trading overlap for circumference')
    p.add_argument('--vertical-phase-material-handoff', action='store_true')
    p.add_argument('--vertical-material-wrap-only', action='store_true')
    p.add_argument('--vertical-material-return-phase', type=float, default=0.)
    p.add_argument('--vertical-return-advance', type=float)
    p.add_argument('--vertical-return-overlap-trim', type=float, default=0.,
                   help='Shorten the material overlap in metres as the adaptive ring returns to the smaller section')
    p.add_argument('--vertical-return-crown', type=float, default=0.,
                   help='Lower the ring body by this many metres at the return roll phase with maximum span')
    p.add_argument('--section-bearing-advance', type=float)
    p.add_argument('--section-flat-advance-weight', type=float, default=0.)
    p.add_argument('--vertical-end-clearance', action='store_true')
    p.add_argument('--section-overlap-clearance', action='store_true',
                   help='Compare local end-curve candidates when measured overlapping spans approach')
    p.add_argument('--section-head-transition', type=float, default=AcuteOptions.section_head_transition_links,
                   help='Length of the head end ramp in link spacings')
    p.add_argument('--section-feedback-during-release', action='store_true')
    p.add_argument('--section-preserve-mean-turn', action='store_true')
    p.add_argument('--section-local-handoff', action='store_true')
    p.add_argument('--bend-section-radius-trim', type=float, default=0.)
    p.add_argument('--manual-takeover', action='store_true')
    p.add_argument('--return-seam-trial', action='store_true',
                   help='Try the wider local curve candidates before the return to the smaller tube')
    p.add_argument('--modes', nargs='+', choices=['helix', 'arc', 'tro'], default=['helix', 'arc'])
    args = p.parse_args()
    if args.section_measured_carrier and (not args.section_feedback_weight or args.section_effort):
        p.error('--section-measured-carrier requires angle feedback without a motor effort trial')
    if args.section_effort and (len(set(args.section_effort)) != len(args.section_effort) or
                                not (args.manual_takeover and args.vertical_adaptive_material and 'arc' in args.modes)):
        p.error('--section-effort requires distinct modes, manual takeover, adaptive ring material and ARC')
    if args.return_seam_trial and not (args.manual_takeover and args.scene == 'variable_return' and
                                       args.section_overlap_clearance and args.vertical_return_advance is not None):
        p.error('--return-seam-trial requires a measured overlap and material-return run')
    if not np.isfinite(args.start_head_height) or args.start_head_height <= 0.:
        p.error('--start-head-height must be finite and positive')
    if (args.short_bend_radius is not None or args.short_elbow_sdf) and args.scene != 'short':
        p.error('--short-bend-radius and --short-elbow-sdf require --scene short')
    if args.vertical_outlet_height != 4. and args.scene not in ('variable', 'variable_return', 'round_to_ellipse'):
        p.error('--vertical-outlet-height requires a variable-diameter or circle-to-ellipse scene')
    if not np.isfinite(args.vertical_outlet_height):
        p.error('--vertical-outlet-height must be finite')
    if args.embed_return_outlet and args.scene != 'variable_return':
        p.error('--embed-return-outlet requires --scene variable_return')
    maximum_taper_length = ((args.vertical_outlet_height - 2.2) / 2. if args.scene == 'variable_return'
                            else args.vertical_outlet_height - 1.9)
    if not 0. < args.taper_length < maximum_taper_length:
        p.error('--taper-length must fit between the taper start and pipe top')
    if args.lower_pipe_diameter != .110 and args.scene not in ('variable', 'variable_return'):
        p.error('--lower-pipe-diameter requires --scene variable or variable_return')
    if not 0. < args.lower_pipe_diameter < args.upper_pipe_diameter < np.inf:
        p.error('The upper and lower pipe diameters must be positive and finite, with the upper larger')
    if args.section_before_bend and not (args.progressive_tail_release and args.section_shell_diameter):
        p.error('--section-before-bend requires --progressive-tail-release and --section-shell-diameter')
    if args.review:
        for folder in args.review:
            print(folder, json.dumps(review_saved(folder), default=json_default), flush=True)
    elif args.output:
        run(args)
    else:
        p.error('supply --output for a new run or --review for saved runs')
