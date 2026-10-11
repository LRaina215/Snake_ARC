"""MuJoCo adapter: only measured joint angles and pipe support enter the core."""
import math
import statistics
from dataclasses import replace
import numpy as np
import mujoco
from .simulation import ArcSimulation, backbone
from .arc_compliance import SlidingArcCompliance, SlidingArcOptions, support_seed


def support_observations(sim):
    force = np.zeros(sim.model.nu)
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        pair = {int(contact.geom1),int(contact.geom2)}
        if not pair & sim.pipe_geoms or not pair & sim.robot_geoms:
            continue
        robot = next(iter(pair & sim.robot_geoms))
        f = np.zeros(6)
        mujoco.mj_contactForce(sim.model,sim.data,i,f)
        # Contact force acts geom1 -> geom2; frame rows are contact basis.
        world = np.asarray(contact.frame).reshape(3,3).T @ f[:3]
        if robot == contact.geom1:
            world *= -1
        link = int(sim.model.geom_bodyid[robot])-sim.model.body('head').id
        force[min(max(link-1,0),sim.model.nu-1)] += max(0.,float(world[2]))
    return force


class AdaptiveArcSimulation(ArcSimulation):
    def __init__(self, scene, parameters, start_x=-.5, options=SlidingArcOptions(), takeover_s=2.):
        super().__init__(scene,parameters,start_x)
        self.options = options
        self.takeover_s = takeover_s
        self.controller = None
        self.phase = parameters.direction*parameters.initial_phase_rad
        self.command_phase = self.phase
        self.enabled = parameters.enabled.copy()
        self.target = parameters.targets(parameters.initial_phase_rad)
        self.support_force = np.zeros(parameters.joints)
        self.next_control = takeover_s
        self.paused = False
        self.compliance_enabled = True
        self.pre_pause_s = 0.

    def toggle_compliance(self):
        self.compliance_enabled = not self.compliance_enabled
        if self.compliance_enabled:
            self.controller = None
        else:
            # The basic wave must cover the CURRENT effective window, including
            # an unfinished handover, rather than driving its new joints to zero.
            active = np.flatnonzero(self.enabled)
            if len(active):
                self.p = replace(self.p,first_joint=int(active[0])+1,last_joint=int(active[-1])+1)

    def adopt_supported_pose(self, qpos, qvel=None):
        """One-time experimental placement, then support-only capture and mj_step.

        Use for existing supported poses rather than forming a phase-zero arc.
        No stored actuator target or route coordinate is imported as control.
        """
        qpos = np.asarray(qpos,float)
        qvel = np.zeros(self.model.nv) if qvel is None else np.asarray(qvel,float)
        if qpos.shape!=(self.model.nq,) or qvel.shape!=(self.model.nv,) or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError('Invalid initial physical state')
        if abs(np.linalg.norm(qpos[3:7])-1)>1e-3:
            raise ValueError('Initial root quaternion must be normalized')
        self.data.qpos[:]=qpos;self.data.qvel[:]=qvel;self.data.ctrl[:]=0.;self.data.time=0.
        self.enabled=np.zeros(self.model.nu,bool)
        self.target=self.data.qpos[self.drive.qadr].copy()
        self.takeover_s=0.;self.controller=None;self.next_control=0.
        self.capture_phase=True
        mujoco.mj_forward(self.model,self.data)

    def step(self, phase=None):
        if self.data.time < self.takeover_s or not self.compliance_enabled:
            if not self.compliance_enabled:
                target = self.p.targets(self.phase * self.p.direction)
                limit = math.radians(self.options.target_rate_deg_s) * self.model.opt.timestep
                self.target += np.clip(target-self.target,-limit,limit)
                self.drive.apply_targets(self.data,self.target,self.enabled)
                mujoco.mj_step(self.model,self.data)
                if not self.paused:
                    self.phase += self.p.direction*self.p.omega_rad_s*self.model.opt.timestep
                self.command_phase = self.phase
                return self.phase,self.target
            if phase is None:
                if self.paused:self.pre_pause_s += self.model.opt.timestep
                phase = self.p.phase(self.data.time-self.pre_pause_s)
            result = super().step(phase)
            self.phase, self.target = self.p.direction*result[0],result[1]
            self.command_phase = self.phase
            return result
        if self.controller is None:
            forces = support_observations(self)
            seed = support_seed(np.where(forces>.5,forces,0).tolist(),self.options.minimum_width,
                                self.options.maximum_width,self.options.guard_joints)
            if seed is None:
                # No support at switch-on: hold the current driven shape; do not
                # invent contacts or automatically grip an unsupported body.
                self.drive.apply_targets(self.data,self.target,self.enabled)
                mujoco.mj_step(self.model,self.data)
                return self.phase,self.target
            measured = np.degrees(self.data.qpos[self.drive.qadr]).tolist()
            if getattr(self,'capture_phase',False):
                first,last=seed
                odd=[measured[i] for i in range(first-1,last) if i%2==0]
                even=[measured[i] for i in range(first-1,last) if i%2==1]
                self.phase=math.atan2(statistics.mean(even),statistics.mean(odd))
                self.capture_phase=False
            self.controller = SlidingArcCompliance(measured,self.phase,seed,self.options,self.enabled.tolist())
            self.next_control = self.data.time
        if self.data.time + 1e-10 >= self.next_control:
            dt = .02
            self.support_force = support_observations(self)
            measured = np.degrees(self.data.qpos[self.drive.qadr]).tolist()
            targets, enabled = self.controller.update(measured,(self.support_force>.5).tolist(),dt,self.phase)
            self.command_phase = self.phase
            self.target, self.enabled = np.radians(targets),np.array(enabled,bool)
            self.next_control += dt
            if phase is None and not self.paused and not self.controller.phase_paused:
                self.phase += self.p.direction * self.p.omega_rad_s * dt
        self.last_phase = self.command_phase
        self.drive.apply_targets(self.data,self.target,self.enabled)
        mujoco.mj_step(self.model,self.data)
        return self.command_phase,self.target

    def color_window(self):
        """Display only: blue enabled, green observed support, purple passive."""
        for geom in self.robot_geoms:
            link = int(self.model.geom_bodyid[geom])-self.model.body('head').id
            i = min(max(link-1,0),self.model.nu-1)
            self.model.geom_rgba[geom] = ([.15,.72,.40,1] if self.support_force[i]>.5 else
                                         [.10,.48,.85,1] if self.enabled[i] else [.50,.26,.70,1])

    def measure(self,phase=None):
        row = super().measure(self.command_phase if phase is None else phase)
        # Fixed body points for evaluation, independent of a moving control window.
        points = backbone(self.model,self.data)
        center = points.mean(axis=0)
        row.update(progress_m=float(self.route_coordinates(center)[0][0]),center_m=center.tolist(),
                   passive_actuator_force_max_nm=float(np.max(np.abs(self.data.actuator_force[~self.enabled]),initial=0)),
                   target_rad=self.target.tolist(),enabled=self.enabled.tolist(),
                   support_force_n=self.support_force.tolist(),
                   compliance=None if self.controller is None else self.controller.snapshot())
        return row
