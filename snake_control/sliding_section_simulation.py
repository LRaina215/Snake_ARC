"""Simulation-only adapter. No real motor or legacy adaptive imports."""
from dataclasses import replace
import numpy as np
import mujoco
from .simulation import ArcSimulation, backbone
from .sliding_section_compliance import SlidingSectionCompliance, WindowOptions
from .fixed_section_compliance import SectionOptions


def upward_support(sim):
    """Actual contact wrench on each robot link, positive world vertical part."""
    force = np.zeros(sim.model.nu)
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        pair = {int(contact.geom1), int(contact.geom2)}
        if not pair & sim.pipe_geoms or not pair & sim.robot_geoms:
            continue
        robot = next(iter(pair & sim.robot_geoms))
        wrench = np.zeros(6)
        mujoco.mj_contactForce(sim.model, sim.data, i, wrench)
        world = np.asarray(contact.frame).reshape(3, 3).T @ wrench[:3]
        if robot == contact.geom1:
            world *= -1
        link = int(sim.model.geom_bodyid[robot])-sim.model.body('head').id
        force[min(max(link-1, 0), sim.model.nu-1)] += max(0., float(world[2]))
    return force


class SlidingSectionSimulation(ArcSimulation):
    def __init__(self, scene, parameters, start_x=-.5, section_options=SectionOptions(),
                 window_options=WindowOptions(), automatic=True, initial_window=None):
        initial = parameters if initial_window is None else replace(
            parameters, first_joint=initial_window[0], last_joint=initial_window[1])
        super().__init__(scene, initial, start_x)
        self.initial_parameters = initial
        self.p = parameters
        self.lengths = np.linalg.norm(np.diff(backbone(self.model, self.data), axis=0), axis=1)
        self.controller = SlidingSectionCompliance(parameters, self.lengths, section_options, window_options, automatic)
        self.weights = initial.enabled.astype(float)
        self.enabled = self.weights > 0
        self.target = initial.targets(initial.initial_phase_rad)
        self.phase = parameters.initial_phase_rad
        self.paused = False
        self.support_up = np.zeros(parameters.joints)

    def step(self, phase=None):
        self.support_up = upward_support(self)
        if phase is not None:
            self.phase = phase
        if self.data.time < self.p.settle_s:
            self.target = self.initial_parameters.targets(self.phase)
        else:
            self.target, self.weights = self.controller.update(
                self.data.qpos[self.drive.qadr], self.phase, self.support_up,
                float(self.data.time), self.model.opt.timestep, self.data.actuator_force.copy())
            self.enabled = self.weights > 1e-12
        self.last_phase = self.phase
        self.drive.apply_targets(self.data, self.target, self.enabled)
        self.data.ctrl[:] *= self.weights
        mujoco.mj_step(self.model, self.data)
        if phase is None and not self.paused and not self.controller.phase_hold and self.data.time >= self.p.settle_s:
            self.phase += self.p.omega_rad_s*self.model.opt.timestep
        return self.last_phase, self.target

    def request_neighbour(self, direction):
        c = self.controller
        if c.window is None:
            return False
        return c.request((c.window[0]+direction, c.window[1]+direction),
                         self.data.qpos[self.drive.qadr], self.phase, float(self.data.time))

    def measure(self, phase=None):
        row = super().measure(self.last_phase)
        points = backbone(self.model, self.data)
        row.update(progress_m=float(self.route_coordinates(points.mean(axis=0))[0][0]),
                   enabled=self.enabled.tolist(), weights=self.weights.tolist(),
                   target_rad=self.target.tolist(), support_up_n=self.support_up.tolist(),
                   geometry=self.controller.snapshot(),
                   passive_actuator_force_max_nm=float(np.max(abs(self.data.actuator_force[~self.enabled]), initial=0)))
        return row
