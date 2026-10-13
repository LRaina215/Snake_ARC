"""Six-joint automatic sliding using only fresh motor angles and a clock.

Geometry-based handover is a trial; it cannot measure contact or unloading.
The simulation adapter is responsible for executing this angle/mask command.
"""
from dataclasses import dataclass, replace, asdict
import math
import numpy as np
from .fixed_section_compliance import FixedSectionCompliance, SectionOptions
from .motor_shape_observer import MotorShapeObserver, MotorObserverOptions


@dataclass(frozen=True)
class MotorSlidingOptions:
    cooldown_s: float = 5.
    phase_zero_tolerance: float = .015
    switch_angle_deg: float = 4.
    switch_velocity_deg_s: float = 8.
    stable_feedback_s: float = .06
    minimum_hold_s: float = .1
    handover_timeout_s: float = 20.
    maximum_feedback_age_s: float = .8


class MotorFeedbackSliding:
    def __init__(self, parameters, lengths, reference, execute=True,
                 section_options=SectionOptions(), options=MotorSlidingOptions(), feedback_period_s=.02,
                 allowed_widths=(6,)):
        width = parameters.last_joint-parameters.first_joint+1
        if width not in allowed_widths:
            raise ValueError('Enabled width is outside the explicitly allowed set')
        self.allowed_widths = tuple(allowed_widths)
        self.p, self.lengths, self.options = parameters, np.asarray(lengths), options
        self.section_options, self.execute = section_options, execute
        self.window = (parameters.first_joint, parameters.last_joint)
        self.core = FixedSectionCompliance(parameters, lengths, section_options)
        self.observer = MotorShapeObserver(lengths, reference, parameters.amplitude_deg,
            MotorObserverOptions(maximum_feedback_gap_s=max(.1, 2.5*feedback_period_s)), width=width)
        self.feedback_period_s = feedback_period_s
        self.enabled = parameters.enabled.copy()
        self.phase = parameters.initial_phase_rad
        self.phase_hold = False
        self.targets = parameters.targets(self.phase)
        self.previous_angles, self.last_time = None, None
        self.last_control_time = None
        self.feedback_phase = self.phase
        self.fresh_feedback = False
        self.pending = None
        self.cooldown_until = parameters.settle_s
        self.events = []
        self.observation = None
        self.angle_velocity_deg_s = np.zeros(parameters.joints)
        self.angle_velocity_available = False
        self.last_gate = None
        self.fault = None

    def _handover(self, angles, time_s):
        request = self.pending
        new = tuple(request['to'])
        new_mask = (np.arange(1, self.p.joints+1) >= new[0]) & (np.arange(1, self.p.joints+1) <= new[1])
        entering = np.flatnonzero(new_mask & ~self.enabled)
        exiting = np.flatnonzero(self.enabled & ~new_mask)
        changed = np.r_[entering, exiting]
        factor = math.cos(self.phase) if exiting[0] % 2 == 0 else math.sin(self.phase)
        if not self.phase_hold and abs(factor) <= self.options.phase_zero_tolerance:
            self.phase_hold = True
            request['hold_since'] = time_s
            self.events.append(dict(kind='phase_hold', time_s=time_s, phase_rad=self.phase,
                                    entering=(entering+1).tolist(), exiting=(exiting+1).tolist()))
        if self.phase_hold and self.fresh_feedback:
            q = np.radians(angles)
            good, geometry, _ = self.observer.geometry_observer.geometry(q, new)
            preload = self.targets.copy()
            preload[entering] = q[entering]
            preload[~new_mask] = q[~new_mask]
            next_core = FixedSectionCompliance(replace(self.p, first_joint=new[0], last_joint=new[1]),
                                              self.lengths, self.section_options)
            command_good, _ = next_core._check_goal(preload)
            self.last_gate = dict(time_s=time_s, actual_candidate=geometry,
                switch_angles_deg=angles[changed].tolist(),
                switch_velocity_deg_s=self.angle_velocity_deg_s[changed].tolist(),
                velocity_from_continuous_fresh_feedback=self.angle_velocity_available,
                candidate_open=good, command_open=bool(command_good),
                unloading_observed=False)
            direction = new[0]-self.window[0]
            evidence = self.observation is not None and self.observation['valid'] and self.observation['direction'] == direction
            ready = (good and command_good and evidence and self.angle_velocity_available and
                np.max(abs(angles[changed])) <= self.options.switch_angle_deg and
                np.max(abs(self.angle_velocity_deg_s[changed])) <= self.options.switch_velocity_deg_s and
                time_s-request['hold_since'] >= self.options.minimum_hold_s)
            if ready:
                if request['ready_since'] is None:
                    request['ready_since'] = self.last_time
                if self.last_time-request['ready_since']+1e-9 >= self.options.stable_feedback_s:
                    old = self.window
                    next_core.targets = preload.copy()
                    next_core.previous_axis = self.core.previous_axis
                    self.core, self.window, self.enabled = next_core, new, new_mask
                    self.targets = preload.copy()
                    self.events.append(dict(kind='complete', time_s=time_s, old=list(old), to=list(new),
                                            gate=self.last_gate, strategy='motor-angle geometric trial'))
                    self.pending, self.phase_hold = None, False
                    self.cooldown_until = time_s+self.options.cooldown_s
            else:
                request['ready_since'] = None
        if self.pending is not None and time_s-request['time_s'] >= self.options.handover_timeout_s:
            self.events.append(dict(kind='reject', time_s=time_s, to=list(new),
                                    reason='motor-angle handover timed out', last_gate=self.last_gate))
            self.pending, self.phase_hold = None, False
            self.cooldown_until = time_s+self.options.cooldown_s

    @property
    def automatic(self):
        return self.execute

    @automatic.setter
    def automatic(self, value):
        self.execute = bool(value)
        if not self.execute and self.pending is not None:
            self.events.append(dict(kind='cancel', time_s=self.last_control_time,
                                    to=self.pending['to'], reason='automatic selection disabled'))
            self.pending, self.phase_hold = None, False

    def update(self, angles_deg, feedback_time_s, commanded_phase_rad=None):
        """Compatibility: one fresh feedback and one control tick together."""
        if self.last_time is not None and feedback_time_s <= self.last_time:
            raise ValueError('Fresh increasing motor-feedback time required')
        return self.advance(angles_deg, feedback_time_s, feedback_time_s, commanded_phase_rad)

    def advance(self, angles_deg, feedback_time_s, control_time_s,
                commanded_phase_rad=None, phase_at_feedback=None, freeze_targets=False):
        """Control tick may hold a motor sample; observation may not invent one.

        Both times use the caller's same clock. The known phase at acquisition
        is separate from the next commanded phase. Mask changes require NEW
        motor feedback; target filtering and phase gating run at control rate.
        """
        angles = np.asarray(angles_deg, float)
        if angles.shape != (self.p.joints,) or not np.isfinite(angles).all():
            raise ValueError('One finite actual motor angle per joint required')
        if not all(math.isfinite(t) for t in (feedback_time_s, control_time_s)) or feedback_time_s > control_time_s+1e-8:
            raise ValueError('Finite, nonfuture feedback/control times required')
        if self.last_time is not None and feedback_time_s < self.last_time:
            raise ValueError('Motor-feedback time cannot run backwards')
        if self.last_control_time is not None and control_time_s < self.last_control_time:
            raise ValueError('Control clock cannot run backwards')
        if self.last_control_time == control_time_s:
            # A duplicate timestamp has no elapsed time. Do not invent dt,
            # feedback, phase progress or another window transaction.
            self.fresh_feedback = False
            return self.targets.copy(), self.enabled.copy()
        if control_time_s-feedback_time_s > self.options.maximum_feedback_age_s:
            raise TimeoutError('Motor angles too old for sliding control')
        dt = .02 if self.last_control_time is None else min(.02, control_time_s-self.last_control_time)
        previous_time = control_time_s-dt if self.last_control_time is None else self.last_control_time
        observed_phase = self.phase
        self.fresh_feedback = self.last_time is None or feedback_time_s > self.last_time
        if self.fresh_feedback:
            feedback_dt = .02 if self.last_time is None else feedback_time_s-self.last_time
            gap = feedback_dt > self.observer.options.maximum_feedback_gap_s
            self.angle_velocity_available = self.previous_angles is not None and not gap
            if gap and self.pending is not None:
                self.pending['ready_since'] = None
            if self.previous_angles is not None:
                self.angle_velocity_deg_s = (angles-self.previous_angles)/feedback_dt
            self.previous_angles, self.last_time = angles.copy(), feedback_time_s
            self.feedback_phase = observed_phase if phase_at_feedback is None else phase_at_feedback
            if control_time_s >= self.p.settle_s:
                self.observation = self.observer.update(angles, self.feedback_phase, self.enabled, feedback_time_s)
        elif not np.array_equal(angles, self.previous_angles):
            raise ValueError('Changed angles with an unchanged feedback timestamp')
        self.last_control_time = control_time_s
        if freeze_targets:
            if self.pending is not None:
                self.pending['ready_since'] = None
            return self.targets.copy(), self.enabled.copy()
        if commanded_phase_rad is not None and not math.isfinite(commanded_phase_rad):
            raise ValueError('Finite commanded phase required')
        if commanded_phase_rad is not None:
            if not self.phase_hold:
                self.phase = float(commanded_phase_rad)
        elif not self.phase_hold:
            rolling_dt = min(dt, max(0., control_time_s-max(previous_time, self.p.settle_s)))
            self.phase += self.p.omega_rad_s*rolling_dt
        if control_time_s < self.p.settle_s:
            self.targets = self.p.targets(self.phase)
            return self.targets.copy(), self.enabled.copy()
        if (self.execute and self.observation is not None and self.pending is None and control_time_s >= self.cooldown_until
                and self.observation['suggested_window'] is not None and self.fault is None):
            self.pending = dict(to=self.observation['suggested_window'], time_s=control_time_s,
                                ready_since=None, hold_since=None)
            self.events.append(dict(kind='prepare', time_s=control_time_s, old=list(self.window),
                                    to=self.pending['to'], observation=self.observation))
        if self.pending is not None:
            self._handover(angles, control_time_s)
        previous_targets = self.targets.copy()
        proposed = self.core.update(np.radians(angles), self.phase, control_time_s, dt)
        valid, _ = self.core._check_goal(proposed)
        if self.core.status != 'running' or not valid:
            if self.fault is None:
                self.events.append(dict(kind='shape_hold', time_s=control_time_s, reason=self.core.status))
            self.fault = self.core.status if self.core.status != 'running' else 'filtered target outside opening'
            self.targets[self.enabled] = previous_targets[self.enabled]
            self.targets[~self.enabled] = np.radians(angles)[~self.enabled]
        else:
            self.fault = None
            self.targets = proposed
        if self.enabled.sum() not in self.allowed_widths:
            raise RuntimeError('Allowed-width mask invariant violated')
        return self.targets.copy(), self.enabled.copy()

    def snapshot(self):
        return dict(window=list(self.window), phase_rad=self.phase, phase_hold=self.phase_hold,
                    pending=self.pending, observation=self.observation, last_gate=self.last_gate,
                    fault=self.fault, compliance=self.core.snapshot(), options=asdict(self.options),
                    feedback_time_s=self.last_time, control_time_s=self.last_control_time,
                    new_feedback_this_tick=self.fresh_feedback, feedback_phase_rad=self.feedback_phase,
                    feedback_period_s=self.feedback_period_s,
                    feedback_source='measured joint angles; qdot from fresh angle differences',
                    observes_contact_or_unloading=False, execute=self.execute)
