"""18/20-joint simulation on the bare-pipe tee scenes; same drive and placement idea as ArcSimulation."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .rolling_arc import LocalMotorDrive
from .simulation import ArcSimulation, ROOT, backbone

PIPE_RADIUS = .055
PIPE_HEIGHT = 1.2


class TeeSimulation(ArcSimulation):
    """Loads `Snake_Robot_o<joints>_Tee_<scene>.xml`; route/feature/measure are inherited."""

    def __init__(self, scene, parameters, start_x=0., friction=None):
        self.scene, self.p = scene, parameters
        path = ROOT / 'mujoco_model' / f'Snake_Robot_o{parameters.joints}_Tee_{scene}.xml'
        self.model_path = str(path)
        self.model = mujoco.MjModel.from_xml_path(str(path))
        if friction is not None:
            for name in ('pipe_main', 'pipe_branch'):
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid >= 0:
                    self.model.geom_friction[gid, 0] = friction
        self.data = mujoco.MjData(self.model)
        self.drive = LocalMotorDrive(self.model, parameters)
        self.model.dof_damping[self.drive.vadr] = parameters.passive_damping
        self.model.geom_rgba[self.model.geom_bodyid >= self.model.body('head').id] = [.50, .26, .70, 1]
        for j in np.flatnonzero(parameters.enabled) + 1:
            bid = self.model.jnt_bodyid[self.model.joint(f'joint{j}').id]
            self.model.geom_rgba[self.model.geom_bodyid == bid] = [.10, .48, .85, 1]
        self.pipe_geoms = set(np.flatnonzero(self.model.geom_bodyid == self.model.body('pipe_network').id))
        self.robot_geoms = set(np.flatnonzero(self.model.geom_bodyid >= self.model.body('head').id))
        self.branch_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'pipe_branch')
        nid = self.model.numeric('pipe_centerline').id
        raw = self.model.numeric_data[self.model.numeric_adr[nid]:]
        self.route = raw[:self.model.numeric_size[nid]].reshape(-1, 4)[:, :3].copy()
        self.route_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.route, axis=0), axis=1))]
        self.feature = self.model.numeric('feature_span').data.copy()
        self.last_phase = parameters.initial_phase_rad
        self.place(start_x)

    def place(self, start_x):
        self.data.qpos[:] = 0
        self.data.qpos[3] = 1
        self.data.qpos[self.drive.qadr] = self.p.targets(self.p.initial_phase_rad)
        mujoco.mj_forward(self.model, self.data)
        rot = Rotation.from_euler('x', 180, degrees=True)
        arc = rot.apply(backbone(self.model, self.data))[self.p.first_joint:self.p.last_joint + 2]
        self.data.qpos[:3] = [start_x, -(arc[:, 1].min() + arc[:, 1].max()) / 2,
                              PIPE_HEIGHT + PIPE_RADIUS + .040 - arc[:, 2].max()]
        self.data.qpos[3:7] = rot.as_quat()[[3, 0, 1, 2]]
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def place_shape(self, q, helix_points, x_head, beta=0.):
        """Initial placement only: put a helix-shaped body (joint angles q) on the main pipe.

        The axis fitted to backbone points [0, helix_points] is laid on the pipe axis, head toward +x
        at x_head, then rotated by beta about the axis. Used once before mj_step; never during control.
        """
        q = np.asarray(q, float)
        self.data.qpos[:] = 0
        self.data.qpos[3] = 1
        self.data.qpos[self.drive.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        pts = backbone(self.model, self.data)
        hp = pts[:helix_points + 1]
        t = np.diff(hp, axis=0)
        t /= np.linalg.norm(t, axis=1)[:, None]
        a = np.linalg.eigh(np.cov(t.T))[1][:, 0]
        if np.dot(hp[0] - hp[-1], a) < 0:
            a = -a
        rel = (hp - hp.mean(0)) - np.outer((hp - hp.mean(0)) @ a, a)
        e1 = np.linalg.svd(rel)[2][0]
        e1 -= np.dot(e1, a) * a
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        x, y = rel @ e1, rel @ e2
        cx, cy, c0 = np.linalg.lstsq(np.column_stack([x, y, np.ones_like(x)]), x * x + y * y, rcond=None)[0]
        centre = hp.mean(0) + cx / 2 * e1 + cy / 2 * e2
        rot = Rotation.from_euler('x', beta).as_matrix() @ np.vstack([a, e1, e2])
        head_axial = float(np.dot(pts[0] - centre, a))
        pos = np.array([x_head - head_axial, 0., PIPE_HEIGHT]) - rot @ centre
        self.data.qpos[:3] = pos
        self.data.qpos[3:7] = Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]]
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return float(np.sqrt(c0 + (cx / 2) ** 2 + (cy / 2) ** 2))

    def branch_contacts(self):
        """Robot links touching the branch geom, with total normal force (evaluation only)."""
        links, force = set(), 0.
        if self.branch_geom < 0:
            return [], 0.
        head = self.model.body('head').id
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            if self.branch_geom in pair and pair & self.robot_geoms:
                g = next(iter(pair & self.robot_geoms))
                links.add(int(self.model.geom_bodyid[g]) - head)
                f = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, f)
                force += abs(float(f[0]))
        return sorted(links), force
