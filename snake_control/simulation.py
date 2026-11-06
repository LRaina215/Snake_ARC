"""Dynamic simulator: place once, then advance only with mj_step."""
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .rolling_arc import LocalMotorDrive

ROOT = Path(__file__).resolve().parents[1]


def backbone(model, data):
    return np.array([data.site('s_head').xpos.copy()] +
                    [data.xanchor[model.joint(f'joint{j}').id].copy() for j in range(1, model.nu + 1)] +
                    [data.site('s_tail').xpos.copy()])


class ArcSimulation:
    def __init__(self, scene, parameters, start_x=-.5, *, model=None):
        self.scene, self.p = scene, parameters
        self.model = (mujoco.MjModel.from_xml_path(str(ROOT / 'mujoco_model' / f'Snake_Robot_o20_Rolling_Arc_{scene}.xml'))
                      if model is None else model)
        self.data = mujoco.MjData(self.model)
        self.drive = LocalMotorDrive(self.model, parameters)
        self.model.dof_damping[self.drive.vadr] = parameters.passive_damping
        self.model.geom_rgba[self.model.geom_bodyid >= self.model.body('head').id] = [.50, .26, .70, 1]
        for j in np.flatnonzero(parameters.enabled) + 1:
            bid = self.model.jnt_bodyid[self.model.joint(f'joint{j}').id]
            self.model.geom_rgba[self.model.geom_bodyid == bid] = [.10, .48, .85, 1]
        self.pipe_geoms = set(np.flatnonzero(self.model.geom_bodyid == self.model.body('pipe_network').id))
        self.robot_geoms = set(np.flatnonzero(self.model.geom_bodyid >= self.model.body('head').id))
        raw = self.model.numeric_data[self.model.numeric_adr[self.model.numeric('pipe_centerline').id]:]
        n = self.model.numeric_size[self.model.numeric('pipe_centerline').id]
        self.route = raw[:n].reshape(-1, 4)[:, :3].copy()
        self.route_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.route, axis=0), axis=1))]
        self.feature = self.model.numeric('feature_span').data.copy()
        self.last_phase = parameters.initial_phase_rad
        self.place(start_x)

    def place(self, start_x):
        # Phase-zero U in the local yz plane -> an inverted U over a horizontal x pipe.
        self.data.qpos[:] = 0
        self.data.qpos[3] = 1
        self.data.qpos[self.drive.qadr] = self.p.targets(self.p.initial_phase_rad)
        mujoco.mj_forward(self.model, self.data)
        points = backbone(self.model, self.data)
        rot = Rotation.from_euler('x', 180, degrees=True)
        rotated = rot.apply(points)
        arc = rotated[self.p.first_joint:self.p.last_joint + 2]
        # Center the opening laterally, then place its highest centerline point above the pipe.
        self.data.qpos[:3] = [start_x, -(arc[:, 1].min() + arc[:, 1].max()) / 2,
                             1.2 + .0565 + .040 - arc[:, 2].max()]
        quat = rot.as_quat()
        self.data.qpos[3:7] = quat[[3, 0, 1, 2]]
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def step(self, phase=None):
        phase = self.p.phase(self.data.time) if phase is None else phase
        self.last_phase = phase
        target = self.drive.apply(self.data, phase)
        mujoco.mj_step(self.model, self.data)
        return phase, target

    def route_coordinates(self, points):
        points = np.atleast_2d(points)
        a, delta = self.route[:-1], np.diff(self.route, axis=0)
        length2 = np.sum(delta * delta, axis=1)
        t = np.clip(np.sum((points[:, None] - a) * delta, axis=2) / np.maximum(length2, 1e-12), 0, 1)
        nearest = a + t[:, :, None] * delta
        distance = np.linalg.norm(nearest - points[:, None], axis=2)
        index = np.argmin(distance, axis=1)
        row = np.arange(len(points))
        return self.route_s[index] + t[row, index] * np.sqrt(length2[index]), distance[row, index]

    def measure(self, phase=None):
        mujoco.mj_forward(self.model, self.data)
        points = backbone(self.model, self.data)
        center = points[self.p.first_joint:self.p.last_joint + 2].mean(axis=0)
        progress = self.route_coordinates(center)[0][0]
        body_s = self.route_coordinates(points)[0]
        contact_links = set()
        support, ground_force = 0.0, 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & self.pipe_geoms and pair & self.robot_geoms:
                robot_geom = next(iter(pair & self.robot_geoms))
                contact_links.add(int(self.model.geom_bodyid[robot_geom]) - self.model.body('head').id)
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                support += abs(float(f[0]))
            if self.model.geom('floor').id in pair and pair & self.robot_geoms:
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                ground_force += abs(float(f[0]))
        phase = self.last_phase if phase is None else phase
        return dict(time_s=float(self.data.time), phase_rad=float(phase),
                    progress_m=float(progress), center_m=center.tolist(), min_height_m=float(points[:, 2].min()),
                    min_body_progress_m=float(body_s.min()), max_body_progress_m=float(body_s.max()),
                    ground_contact_force_n=ground_force,
                    contact_links=sorted(contact_links), normal_force_n=support,
                    passive_actuator_force_max_nm=float(np.max(np.abs(self.data.actuator_force[~self.p.enabled]), initial=0)),
                    qpos_rad=self.data.qpos[self.drive.qadr].tolist(), target_rad=self.p.targets(phase).tolist(),
                    actuator_force_nm=self.data.actuator_force.tolist())
