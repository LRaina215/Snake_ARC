"""Run the existing angle-only horizontal controller on a declared course."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import shutil
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.adaptive_arc_length import HalfArcParameters
from snake_control.arc_shape import ArcShapeEvaluator
from snake_control.fixed_section_compliance import SectionOptions
from snake_control.posture_simulation import PostureSimulation
from snake_control.posture_window import PostureWindowOptions
from snake_control.simulation import ROOT, backbone
from snake_control.sliding_section_compliance import open_geometry
from validation.motor_feedback_experiment import snapshot
from validation.posture_window_experiment import references


COURSES = {
    'elbow': ('elbow90_section_exit', -.25, 85.),
    'taper_elbow': ('elbow90_taper110_210', -.25, 100.),
    'taper_straight': ('length_taper_return', -.50, 130.),
}


class Outlet:
    def __init__(self, sim):
        self.sim = sim
        self.axis = sim.route[-1] - sim.route[-2]
        self.axis /= np.linalg.norm(self.axis)
        self.point = np.array([np.interp(sim.feature[-1], sim.route_s, coord) for coord in sim.route.T])
        model = sim.model
        self.shells = []
        for geom in sorted(sim.robot_geoms):
            if not model.geom_contype[geom]:
                continue
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                index = model.geom_dataid[geom]
                offset = model.mesh_vertadr[index]
                vertices = model.mesh_vert[offset:offset + model.mesh_vertnum[index]]
            elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                vertices = None
            else:
                raise ValueError(f'Unsupported collision shell {model.geom(geom).name}')
            self.shells.append((geom, vertices))

    def distance(self):
        model, data = self.sim.model, self.sim.data
        distances = []
        for geom, vertices in self.shells:
            direction = data.geom_xmat[geom].reshape(3, 3).T @ self.axis
            if vertices is None:
                radius, half_length, _ = model.geom_size[geom]
                local = -abs(direction[2]) * half_length - radius * np.linalg.norm(direction[:2])
            else:
                local = np.min(vertices @ direction)
            distances.append((data.geom_xpos[geom] - self.point) @ self.axis + local)
        return float(min(distances))


def run(case, output, expansion_turn=None, anticipate=False):
    scene, start, duration = COURSES[case]
    parameters = HalfArcParameters()
    options = SectionOptions(project_open_goal=True)
    posture = PostureWindowOptions(minimum_loaded_turn_deg=expansion_turn, anticipate_tangent_loss=anticipate)
    sim = PostureSimulation(parameters, references(), scene, start, section_options=options,
                            posture_options=posture)
    outlet = Outlet(sim)
    evaluator = ArcShapeEvaluator(sim.route)
    model, data = sim.model, sim.data
    floor = model.geom('floor').id
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    source.mkdir()
    for file in ('section_geometry.py', 'fixed_section_compliance.py', 'posture_window.py',
                 'posture_simulation.py', 'motor_feedback_simulation.py'):
        shutil.copyfile(ROOT / 'snake_control' / file, source / file)
    shutil.copyfile(__file__, source / Path(__file__).name)
    shutil.copyfile(ROOT / 'mujoco_model' / f'Snake_Robot_o20_Rolling_Arc_{scene}.xml', output / 'model.xml')
    weight = float(model.body_mass[model.body('head').id:].sum() * abs(model.opt.gravity[2]))
    rows, traces = [], []
    markers = None
    first_clear = clear_phase = clear_time = origin = None
    reason = 'duration_limit'
    exit_checks = None
    with gzip.open(output / 'records.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for sample in range(round(duration / .05) + 1):
            if sample:
                for _ in range(50):
                    sim.step()
            row = snapshot(sim)
            points = backbone(model, data)
            angles = data.qpos[sim.drive.qadr]
            first, last = sim.controller.window
            shape = open_geometry(angles, sim.lengths, (first, last))
            pipe_shape = evaluator.measure(points, np.degrees(angles), first, last)
            ground = False
            self_collision = False
            for contact in data.contact:
                a, b = int(contact.geom1), int(contact.geom2)
                ground |= (a == floor and b in sim.robot_geoms) or (b == floor and a in sim.robot_geoms)
                if a in sim.robot_geoms and b in sim.robot_geoms and contact.dist < 0.:
                    self_collision |= abs(model.geom_bodyid[a] - model.geom_bodyid[b]) > 2
            open_arc = 150. <= shape['turn_deg'] <= 218. and shape['gap_fraction'] >= .4 and shape['fit_rms_m'] <= .015
            support = sum(row['support_up_n']) >= .5 * weight
            clear = outlet.distance()
            phase = sim.controller.phase
            if clear >= 0. and support and not ground:
                if clear_time is None:
                    clear_phase, clear_time = phase, row['time_s']
                    first_clear = clear_time if first_clear is None else first_clear
                    markers = slice(first, last + 2)
                    origin = float(points[markers].mean(0) @ outlet.axis)
                    exit_checks = True
                advance = float(points[markers].mean(0) @ outlet.axis - origin)
                exit_checks &= open_arc and not self_collision
            else:
                clear_phase = clear_time = origin = advance = exit_checks = None
            row.update(body_shape=shape, pipe_shape=pipe_shape, open_arc=open_arc, support=support, ground=ground,
                       self_collision=self_collision, last_shell_outlet_m=clear, outlet_advance_m=advance,
                       route_progress_m=float(sim.route_coordinates(points.mean(0))[0][0]))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            rows.append(row)
            traces.append(points.copy())
            if ground or sim.controller.fault or sim.controller.resize_fault:
                reason = 'ground_contact' if ground else 'controller_fault'
                break
            if clear_phase is not None and phase - clear_phase >= 2. * np.pi and advance >= .072:
                reason = 'sustained_whole_body_exit'
                break

    audited = [row for row in rows if row['time_s'] >= parameters.settle_s + 1.]
    valid_pipe = [row['pipe_shape']['wrap_deg'] for row in audited if row['pipe_shape']['wrap_valid']]
    last = rows[-1]
    report = dict(case=case, scene=scene, parameters=asdict(parameters), section_options=asdict(options),
                  posture_options=asdict(posture),
                  start_x_m=start, maximum_time_s=duration,
                  command_period_s=.02, angle_period_s=.2, outlet_point_m=outlet.point.tolist(),
                  outlet_axis=outlet.axis.tolist(), route=model.numeric('pipe_centerline').data.reshape(-1, 4).tolist(),
                  termination=reason, last_time_s=last['time_s'], first_clear_s=first_clear,
                  sustained_clear_s=clear_time, continued_advance_m=last['outlet_advance_m'],
                  sustained_exit_open=bool(reason == 'sustained_whole_body_exit' and exit_checks),
                  route_progress_m=last['route_progress_m'] - rows[0]['route_progress_m'],
                  furthest_shell_outlet_m=max(row['last_shell_outlet_m'] for row in rows),
                  minimum_upward_contact_n=min((sum(row['support_up_n']) for row in audited), default=None),
                  open_shape_failures=sum(not row['open_arc'] for row in audited),
                  body_turn_deg=[min(row['body_shape']['turn_deg'] for row in audited),
                                 max(row['body_shape']['turn_deg'] for row in audited)] if audited else None,
                  minimum_gap=min((row['body_shape']['gap_fraction'] for row in audited), default=None),
                  maximum_plane_rms_m=max((row['body_shape']['fit_rms_m'] for row in audited), default=None),
                  pipe_wrap_deg=[min(valid_pipe), max(valid_pipe)] if valid_pipe else None,
                  pipe_wrap_unobservable=len(audited) - len(valid_pipe),
                  self_collision_frames=sum(row['self_collision'] for row in rows),
                  passive_torque_peak_nm=max(np.max(np.abs(np.asarray(row['actuator_force_nm'])[~np.asarray(row['enabled'])]),
                                                     initial=0.) for row in rows),
                  fault=sim.controller.fault or sim.controller.resize_fault, events=sim.controller.events)
    (output / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(output / 'trace.npz', time_s=[row['time_s'] for row in rows], backbone=traces,
                        qpos=[row['actual_qpos'] for row in rows], qvel=[row['actual_qvel'] for row in rows],
                        enabled=[row['enabled'] for row in rows], targets_rad=[row['target_rad'] for row in rows])
    print(case, reason, 'clear', first_clear, 'exit m', last['outlet_advance_m'], 'shape failures', report['open_shape_failures'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=COURSES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expansion-turn', type=float, help='Use the angle-derived tangent turn to anticipate window growth.')
    parser.add_argument('--anticipate', action='store_true')
    args = parser.parse_args()
    run(args.case, args.output, args.expansion_turn, args.anticipate)
