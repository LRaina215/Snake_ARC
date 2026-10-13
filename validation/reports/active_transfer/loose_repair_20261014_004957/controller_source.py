"""Active material transfer over a pipe. No MuJoCo or compliance inputs.

Two research actions: loose rolling with observed drift; prescribed arc
transfer with frozen (or small reciprocating) rolling phase. The caller owns
torque-off and passes only sampled angles/time. Shape/support truth is NOT
available here. Temporary adjacent overlap is intentional, not a resize.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from .section_geometry import forward, local_section
from .angle_shape_observer import AngleShapeObserver
from .rolling_shape_observer import RollingShapeObserver


def ease(value):
    value = float(np.clip(value, 0., 1.))
    return value*value*(3.-2.*value)


@dataclass(frozen=True)
class TransferOptions:
    first_joint: int = 8
    last_joint: int = 13
    amplitude_deg: float = 60.
    loose_amplitude_deg: float = 45.
    hold_phase_rad: float = .7
    preparation_s: float = 6.
    transfer_s: float = 5.
    review_s: float = 3.
    rolling_speed_rad_s: float = .6
    exit_angle_deg: float = 12.
    target_rate_deg_s: float = 45.
    desired_direction: int = -1
    desired_shift_joints: float = 1.
    drift_persistence_s: float = .6
    maximum_transfer_wait_s: float = 20.
    rocking_amplitude_rad: float = .12
    rocking_period_s: float = 2.


class ActiveArcTransfer:
    def __init__(self, lengths, route, options=TransferOptions(), rocking=False):
        if route not in ('loose', 'transfer'):
            raise ValueError('route must be loose or transfer')
        self.lengths = np.asarray(lengths, float)
        self.n = len(self.lengths)-1
        self.options, self.route, self.rocking = options, route, rocking
        self.window = (options.first_joint, options.last_joint)
        self.initial_window = self.window
        self.enabled = self.mask(self.window)
        self.targets = self.template(self.window, 0., options.amplitude_deg)
        self.phase = 0.
        self.stage = 'prepare'
        self.last_control_time = self.last_feedback_time = None
        self.observation = None
        self.events = []
        self.transaction = None
        self.stage_since = None
        self.baseline = None
        self.drift_since = None
        self.direction_index = 0
        self.completed = False
        self.fault = None
        self.last_gate = None
        self.geometry = AngleShapeObserver(self.lengths)
        self.end_helper = RollingShapeObserver(self.lengths, None)

    def mask(self, window):
        if not 2 <= window[0] <= window[1] < self.n:
            raise ValueError('Active arc needs passive boundary joints')
        j = np.arange(1, self.n+1)
        return (j >= window[0]) & (j <= window[1])

    def template(self, window, phase, amplitude):
        factors = np.where(np.arange(self.n) % 2 == 0, math.cos(phase), math.sin(phase))
        return np.where(self.mask(window), math.radians(amplitude)*factors, 0.)

    def position(self, angles, window):
        """Conditional gravity crest in material coordinates, not world position."""
        try:
            q = np.radians(angles)
            points = forward(q, self.lengths)
            section = local_section(q, self.lengths, *window)
            down, quality = self.end_helper.passive_down(points, window, section)
            if down is None:
                return dict(valid=False, reason=quality['source'], end_quality=quality)
            crest = self.geometry.crest(points, window, section, down)
            return dict(valid=True, crest_joint=crest['joint_coordinate'],
                        end_quality=quality, down_head=down.tolist(),
                        plane_rms_m=section['fit_rms'], observes_world_position=False)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return dict(valid=False, reason=str(exc))

    def ideal_crest(self, phase, amplitude):
        q = self.template(self.initial_window, phase, amplitude)
        points = forward(q, self.lengths)
        section = local_section(q, self.lengths, *self.initial_window)
        arc = section['points'][1:-1]
        down = (arc[0]+arc[-1])/2-arc.mean(axis=0)
        down /= np.linalg.norm(down)
        return self.geometry.crest(points, self.initial_window, section, down)['joint_coordinate']

    def begin_transfer(self, direction, angles, time_s, reason):
        old = self.window
        new = (old[0]+direction, old[1]+direction)
        new_mask = self.mask(new)
        old_mask = self.enabled.copy()
        entering = new_mask & ~old_mask
        preload = self.targets.copy()
        preload[entering] = np.radians(angles)[entering]
        self.targets = preload
        self.enabled = old_mask | new_mask
        self.transaction = dict(old=list(old), new=list(new), start_s=time_s,
                                entering=np.flatnonzero(entering).tolist(),
                                exiting=np.flatnonzero(old_mask & ~new_mask).tolist(),
                                preload=preload.tolist())
        self.stage, self.stage_since = 'transfer', time_s
        self.events.append(dict(kind='begin_transfer', time_s=time_s, direction=direction,
                                old=list(old), new=list(new), reason=reason,
                                overlap_count=int(self.enabled.sum()), phase_rad=self.phase))

    def advance(self, angles_deg, feedback_time_s, control_time_s):
        angles = np.asarray(angles_deg, float)
        if angles.shape != (self.n,) or not np.isfinite(angles).all():
            raise ValueError('Finite actual motor angles required')
        if feedback_time_s > control_time_s+1e-8 or control_time_s-feedback_time_s > .5:
            raise ValueError('Feedback is future or stale')
        if self.last_control_time is not None and control_time_s <= self.last_control_time:
            raise ValueError('Increasing control time required')
        dt = .02 if self.last_control_time is None else control_time_s-self.last_control_time
        self.last_control_time = control_time_s
        fresh = self.last_feedback_time is None or feedback_time_s > self.last_feedback_time
        if fresh:
            self.last_feedback_time = feedback_time_s
            focus = self.window if self.transaction is None else (
                min(self.transaction['old'][0], self.transaction['new'][0]),
                max(self.transaction['old'][1], self.transaction['new'][1]))
            self.observation = self.position(angles, focus)
        opt = self.options
        desired = self.targets.copy()
        if self.stage == 'prepare':
            self.phase = opt.hold_phase_rad*ease((control_time_s-2.)/2.)
            desired = self.template(self.window, self.phase, opt.amplitude_deg)
            if control_time_s >= opt.preparation_s and fresh:
                if not self.observation['valid']:
                    self.fault = 'initial material position unobservable'
                else:
                    self.baseline = self.observation['crest_joint']
                    self.events.append(dict(kind='baseline', time_s=control_time_s,
                        crest_joint=self.baseline, phase_rad=self.phase))
                    if self.route == 'loose':
                        self.stage, self.stage_since = 'loose_roll', control_time_s
                    else:
                        self.begin_transfer(opt.desired_direction, angles, control_time_s, 'active_command')
                        desired = self.targets.copy()
        elif self.stage == 'loose_roll':
            age = control_time_s-self.stage_since
            self.phase = opt.hold_phase_rad+opt.rolling_speed_rad_s*age
            amplitude = opt.amplitude_deg+(opt.loose_amplitude_deg-opt.amplitude_deg)*ease(age/2.)
            desired = self.template(self.window, self.phase, amplitude)
            if fresh and self.observation['valid']:
                predicted = self.ideal_crest(self.phase, amplitude)-self.ideal_crest(opt.hold_phase_rad, opt.amplitude_deg)
                shift = self.observation['crest_joint']-self.baseline-predicted
                self.observation.update(phase_corrected_shift_joints=shift,
                                        ideal_phase_offset_joints=predicted)
                reached = opt.desired_direction*shift >= opt.desired_shift_joints
                self.drift_since = (control_time_s if self.drift_since is None else self.drift_since) if reached else None
                if self.drift_since is not None and control_time_s-self.drift_since >= opt.drift_persistence_s-1e-8:
                    self.events.append(dict(kind='observed_drift', time_s=control_time_s,
                                            shift_joints=shift, observation=self.observation.copy()))
                    self.begin_transfer(opt.desired_direction, angles, control_time_s, 'follow_observed_drift')
                    desired = self.targets.copy()
        elif self.stage == 'transfer':
            tr = self.transaction
            age = control_time_s-tr['start_s']
            if self.rocking:
                self.phase = opt.hold_phase_rad+opt.rocking_amplitude_rad*math.sin(2*math.pi*age/opt.rocking_period_s)
            old = self.template(tuple(tr['old']), self.phase, opt.amplitude_deg)
            new = self.template(tuple(tr['new']), self.phase, opt.amplitude_deg)
            amount = ease(age/opt.transfer_s)
            desired = (1.-amount)*old+amount*new
            entering = tr['entering']
            desired[entering] = (1.-amount)*np.array(tr['preload'])[entering]+amount*new[entering]
            exiting = tr['exiting']
            self.last_gate = dict(exit_angle_deg=angles[exiting].tolist(),
                                  exit_angle_ready=bool(np.max(abs(angles[exiting])) <= opt.exit_angle_deg),
                                  trajectory_finished=amount >= 1., fresh_feedback=fresh)
            if amount >= 1. and fresh and self.last_gate['exit_angle_ready']:
                self.window = tuple(tr['new'])
                self.enabled = self.mask(self.window)
                self.events.append(dict(kind='transfer_complete', time_s=control_time_s,
                    old=tr['old'], new=tr['new'], gate=self.last_gate.copy(),
                    observation=None if self.observation is None else self.observation.copy()))
                self.transaction = None
                self.stage, self.stage_since = 'review', control_time_s
                if self.rocking:
                    self.phase = opt.hold_phase_rad
                desired = self.template(self.window, self.phase, opt.amplitude_deg)
            elif age > opt.maximum_transfer_wait_s:
                self.fault = 'exit angle did not reach release condition'
        elif self.stage == 'review':
            desired = self.template(self.window, self.phase, opt.amplitude_deg)
            if control_time_s-self.stage_since >= opt.review_s:
                if self.route == 'transfer' and self.direction_index == 0:
                    self.direction_index += 1
                    self.begin_transfer(-opt.desired_direction, angles, control_time_s, 'active_return')
                    desired = self.targets.copy()
                else:
                    self.completed, self.stage = True, 'complete'
        delta = np.clip(desired-self.targets, -math.radians(opt.target_rate_deg_s)*dt,
                        math.radians(opt.target_rate_deg_s)*dt)
        self.targets[self.enabled] += delta[self.enabled]
        self.targets[~self.enabled] = np.radians(angles)[~self.enabled]
        return self.targets.copy(), self.enabled.copy()

    def snapshot(self):
        return dict(stage=self.stage, route=self.route, window=list(self.window), phase_rad=self.phase,
                    enabled=self.enabled.tolist(), transaction=self.transaction, observation=self.observation,
                    baseline_crest_joint=self.baseline, completed=self.completed, fault=self.fault,
                    gate=self.last_gate, options=asdict(self.options), rocking=self.rocking,
                    compliance=False, inputs=['angles_deg','feedback_time_s','control_time_s'])
