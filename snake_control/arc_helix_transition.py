"""ARC to helix using the sampled motor angles and the active ARC window."""
from dataclasses import dataclass
import math

import numpy as np

from .active_arc_transfer import ActiveArcTransfer, TransferOptions, ease
from .curve_gait import CurveGait, helix
from .gravity_global_shift import ArcHelixShift
from .section_geometry import forward, local_section


@dataclass(frozen=True)
class ArcHelixOptions:
    radius_m: float = .094
    pitch_m: float = .13
    shift_m_s: float = .026
    roll_rad_s: float = .23
    target_rate_deg_s: float = 23.
    intake_error_deg: float = 16.
    min_end_depth_links: float = 1.5
    max_feedback_age_s: float = .8
    step_s: float = 1.
    staged_shift_m_s: float = .022
    staged_roll_rad_s: float = .14
    staged_target_rate_deg_s: float = 14.
    prepare_front: bool = False
    front_entry_max_links: float = 3.15


class ArcHelixTransition:
    """One curve interval and adjacent motor handoffs, with optional ARC transfer."""

    def __init__(self, lengths, window, measured_deg, direction=1, options=ArcHelixOptions()):
        self.lengths = np.asarray(lengths, float)
        self.n = len(self.lengths)-1
        self.l = float(np.mean(self.lengths))
        self.options = options
        self.direction = direction
        mask = self._mask(window)
        q = np.radians(measured_deg)
        section = local_section(q, self.lengths, *window)
        self.arc_turn_deg = math.degrees(section['total_turn'])
        if not 125. <= self.arc_turn_deg <= 232. or section['fit_rms'] > .025:
            raise ValueError('当前使能区域没有可拟合的开放圆弧')
        head_length, tail_length, head_depth, tail_depth = self._end_geometry(q, section, window)
        self.end_lengths_m = (head_length, tail_length)
        self.end_depths_m = (head_depth, tail_depth)
        near_equal = abs(head_length-tail_length) <= 2*self.l
        self.staged = options.prepare_front or (near_equal and min(head_length, tail_length) >= 3*self.l)
        # Tail first for the near-equal mounting convention (tail on the left).
        self.reversed = not options.prepare_front and (self.staged or head_length > tail_length+2*self.l)
        self.roll_direction = -direction if self.reversed else direction
        if self.reversed:
            self.lengths = self.lengths[::-1].copy()
        self.window = self.physical_window(window)
        self._enabled = mask[::-1].copy() if self.reversed else mask
        self._targets = self._map_angles(q)
        self.measured = self._targets.copy()
        self.feedback_time = None
        self.stage = 'shift'
        self.running = False
        self.started = False
        self.step_left = 0.
        self.events = [dict(stage='entry', reversed_intake=self.reversed, staged=self.staged, window=list(window),
                            head_length_m=head_length, tail_length_m=tail_length,
                            head_depth_m=head_depth, tail_depth_m=tail_depth)]
        self.transfer = None
        odd = np.arange(self.n)%2 == 0
        components = [np.mean(self.measured[self._enabled & parity]) for parity in (odd, ~odd)]
        self.amplitude = float(np.linalg.norm(components))
        self.phase = math.atan2(components[1], components[0])
        leading_depth = tail_depth if self.reversed else head_depth
        if options.prepare_front:
            self._check_front_entry()
        else:
            self._plan_entry(leading_depth)

    def _end_geometry(self, q, section, window):
        """Estimate arm lengths and end depths using a cut through the fitted ARC center."""
        points = section['projected'][1:-1]
        system = np.column_stack((2.*points, np.ones(len(points))))
        center = np.linalg.lstsq(system, np.sum(points**2, axis=1), rcond=None)[0][:2]
        crown = points.mean(axis=0)-center
        direction = crown/np.linalg.norm(crown)
        backbone = forward(q, self.lengths)
        projected = (backbone-section['axis']['point'])@section['axis']['basis'][:,:2]
        height = (projected-center)@direction
        first, last = window
        top = first+int(np.argmax(height[first:last+2]))
        material = np.r_[0., np.cumsum(self.lengths)]
        head = self._length_after_cut(height[top::-1], material[top]-material[top::-1])
        tail = self._length_after_cut(height[top:], material[top:]-material[top])
        return head, tail, -float(height[0]), -float(height[-1])

    @staticmethod
    def _length_after_cut(height, material):
        below = np.flatnonzero(height <= 0.)
        if not len(below):
            return 0.
        i = below[0]
        if i == 0:
            return float(material[-1])
        crossing = material[i-1]+(material[i]-material[i-1])*height[i-1]/(height[i-1]-height[i])
        return float(material[-1]-crossing)

    def _mask(self, window):
        j = np.arange(1, self.n+1)
        if not 2 <= window[0] < window[1] < self.n:
            raise ValueError('圆弧窗口应连续且两端至少保留一个失能关节')
        return (j >= window[0]) & (j <= window[1])

    def _map_angles(self, angles):
        angles = np.asarray(angles)
        if not self.reversed:
            return angles.copy()
        out = angles[::-1].copy()
        if self.n%2:
            out[1::2] *= -1.
        return out

    def physical_window(self, window):
        return ((self.n+1-window[1],self.n+1-window[0]) if self.reversed else tuple(window))

    @property
    def targets(self):
        return self._map_angles(self._targets)

    @property
    def enabled(self):
        return self._enabled[::-1].copy() if self.reversed else self._enabled.copy()

    def _plan_entry(self, leading_depth):
        first, last = self.window
        # Add a hinge at the leading end if its measured section has little
        # room for the first helical shoulder.
        if leading_depth < self.options.min_end_depth_links*self.l and last < self.n-1:
            self._prepare_transfer(1)
        elif last == self.n-1 and first > 2:
            self._prepare_transfer(-1)
        else:
            self._prepare_shift()

    def _check_front_entry(self):
        section = local_section(self.measured, self.lengths, *self.window)
        head_length, _, head_depth, _ = self._end_geometry(self.measured, section, self.window)
        self.events.append(dict(stage='front_check', window=list(self.window),
                                head_length_m=head_length, head_depth_m=head_depth))
        if head_length <= self.options.front_entry_max_links*self.l:
            self._prepare_shift()
        elif self.window[0] > 2:
            self._prepare_transfer(-1)
        else:
            self.stage = 'blocked'
            self.pause()
            self.events.append(dict(stage=self.stage, reason='front geometry unreachable at the transfer boundary'))

    def _prepare_transfer(self, direction):
        first, last = self.window
        self.stage = 'adjust'
        options = TransferOptions(first_joint=first, last_joint=last, hold_phase_rad=self.phase,
                                  amplitude_deg=math.degrees(self.amplitude), desired_direction=direction,
                                  transfer_s=6. if self.options.prepare_front else 5.,
                                  loose_amplitude_deg=min(45., math.degrees(self.amplitude)))
        self.transfer = ActiveArcTransfer(self.lengths, 'transfer', options)
        self.transfer.phase = self.phase
        self.transfer.targets = self._targets.copy()
        self.transfer_clock = 0.
        self.transfer_feedback_sample = None
        reason = 'shorten the leading arm' if self.options.prepare_front else 'leave two receiving hinges'
        self.transfer.begin_transfer(direction, np.degrees(self.measured), 0., reason)
        self.events.append(dict(stage='adjust', before=list(self.physical_window(self.window)),
                                after=list(self.physical_window((first+direction,last+direction)))))

    def _prepare_shift(self):
        self.stage = 'lead' if self.staged else 'shift'
        self.shift = self.roll = self.shift_time = 0.
        self.leading_extension = 0.
        self.leading_pose = None
        self.receive_start = 0.
        self.receive_shift = 0.
        self.accepted = self.window[1]
        self.confirmed = 0
        self.ready = 0
        opt = self.options
        self.curve = ArcHelixShift(CurveGait(self.n, self.l), self.window, self.amplitude,
                                   opt.radius_m, opt.pitch_m, self.direction if self.reversed else -self.direction)
        self.curve.front_extension_rate = max(1.75, 1.+self.curve.head_length/max(.40,self.curve.tail_length))
        nominal, _, _ = self.curve.shifted_window(0., self.curve.arc_length, self.phase)
        self.residual = np.where(self._enabled, self.measured-nominal, 0.)
        self.events.append(dict(stage=self.stage, window=list(self.physical_window(self.window))))

    def step_once(self):
        if self.stage not in ('complete','blocked'):
            self.running = False
            self.step_left = self.options.step_s

    def pause(self):
        self.running = False
        self.step_left = 0.

    def status(self):
        accepted = None if self.staged and not self.options.prepare_front else getattr(self, 'accepted', None)
        if accepted is not None and self.reversed:
            accepted = self.n+1-accepted
        return dict(stage=self.stage, arc_window=list(self.physical_window(self.window)),
                    leading_end='tail' if self.reversed else 'head',
                    intake_end='head' if self.reversed else 'tail',
                    accepted_joint=accepted,
                    entry_end_lengths_m=self.end_lengths_m, entry_end_depths_m=self.end_depths_m,
                    shift_m=getattr(self, 'shift', 0.), step_remaining_s=self.step_left,
                    receive_shift_m=getattr(self, 'receive_shift', 0.),
                    leading_pose=getattr(self, 'leading_pose', None),
                    transfer_stage=None if self.transfer is None or self.stage != 'adjust' else self.transfer.stage,
                    events=self.events.copy())

    def advance(self, dt, now, feedback_time, measured_deg):
        fresh = self.feedback_time is None or feedback_time > self.feedback_time
        if fresh:
            self.measured = self._map_angles(np.radians(measured_deg))
            self.feedback_time = feedback_time
        max_age = min(.5, self.options.max_feedback_age_s) if self.stage == 'adjust' else self.options.max_feedback_age_s
        if (self.stage in ('complete','blocked') or (not self.running and self.step_left <= 0.) or
                now-feedback_time > max_age):
            return False
        self.started = True
        dt = dt if self.running else min(dt,self.step_left)
        self.step_left = max(0., self.step_left-dt)
        if self.step_left < 1e-9:
            self.step_left = 0.
        if self.stage == 'adjust':
            self._advance_transfer(dt)
        elif self.stage == 'front_settle':
            self.front_settle -= dt
            if self.front_settle <= 0. and fresh:
                self._check_front_entry()
        elif self.options.prepare_front and self.stage == 'lead':
            self._advance_prepared_lead(dt, fresh)
        elif self.staged and not self.options.prepare_front:
            self._advance_staged(dt, fresh)
        else:
            self._advance_shift(dt, fresh)
        return True

    def _advance_transfer(self, dt):
        transfer = self.transfer
        self.transfer_clock += dt
        clock = self.transfer_clock
        if getattr(self, 'transfer_feedback_sample', None) != self.feedback_time:
            self.transfer_feedback_clock = clock
            self.transfer_feedback_sample = self.feedback_time
        transfer.advance(np.degrees(self.measured), self.transfer_feedback_clock, clock)
        self._targets = transfer.targets.copy()
        self._enabled = transfer.enabled.copy()
        if transfer.fault:
            self.stage = 'blocked'
            self.events.append(dict(stage=self.stage, reason=transfer.fault))
        elif transfer.stage == 'review':
            self.window = transfer.window
            if self.options.prepare_front:
                self.stage = 'front_settle'
                self.front_settle = .4
            else:
                self._prepare_shift()

    def _tracking_pace(self):
        error = np.max(np.abs(self._targets[self._enabled]-self.measured[self._enabled]))
        return float(np.clip((math.radians(25.)-error)/math.radians(18.), .15, 1.))

    def _advance_prepared_lead(self, dt, fresh):
        opt, curve = self.options, self.curve
        first, last = self.window
        pace = self._tracking_pace()
        distance = opt.staged_shift_m_s*pace*dt
        self.shift_time += dt
        self.roll += self.roll_direction*opt.staged_roll_rad_s*ease(self.shift_time/1.5)*pace*dt
        phase = self.phase+self.roll
        if self.leading_extension < curve.head_length:
            self.leading_extension = min(curve.head_length, self.leading_extension+distance)
            goal, _ = curve.targets(self.leading_extension, 0., curve.arc_length, phase)
            coverage = np.clip(self.leading_extension/self.l+np.arange(1,self.n+1)-first+.999, 0., 1.)
        else:
            self.shift = min(1.7*self.l, self.shift+distance)
            arc_length = curve.arc_length*(1.-ease(self.shift/.16))
            goal, _, coverage = curve.shifted_window(self.shift, arc_length, phase, self.leading_extension,
                                                    self._prepared_helix())
        coverage[last:] = 0.
        goal = self._apply_curve(goal, coverage, opt.staged_target_rate_deg_s, dt)
        if not fresh:
            return
        self.leading_pose = self._leading_pose(goal)
        error = np.max(abs(goal[:last]-self.measured[:last]))
        ready = (self.leading_extension == curve.head_length and self.leading_pose['axial_m'] >= .022
                 and -.024 <= self.leading_pose['radial_offset_m'] <= -.0135
                 and self.leading_pose['axial_tangent'] >= .35
                 and error <= math.radians(opt.intake_error_deg))
        self.ready = self.ready+1 if ready else 0
        if self.ready >= 3:
            self.ready = 0
            self.stage = 'receive'
            self.receive_start = self.shift
            self.events.append(dict(stage=self.stage, shift_m=self.shift, leading_pose=self.leading_pose.copy()))

    def _leading_pose(self, goal):
        first, last = self.window
        section = local_section(self.measured, self.lengths, first, last)
        points = section['projected'][1:-1]
        system = np.column_stack((2.*points, np.ones(len(points))))
        center = np.linalg.lstsq(system, np.sum(points**2, axis=1), rcond=None)[0][:2]
        radius = np.median(np.linalg.norm(points-center, axis=1))
        points = forward(self.measured, self.lengths)
        head_link = (points[:2]-section['axis']['point'])@section['axis']['basis']
        head = head_link.mean(axis=0)
        shoulder = (points[first]-section['axis']['point'])@section['axis']['basis']
        reference = forward(goal, self.lengths)
        axis = section['axis']['basis'][:,2]
        expected_axial = (reference[:2].mean(axis=0)-reference[first])@axis
        forward_sign = math.copysign(1., expected_axial)
        return dict(axial_m=float(forward_sign*(head[2]-shoulder[2])),
                    radial_offset_m=float(np.linalg.norm(head[:2]-center)-radius),
                    axial_tangent=float(forward_sign*(head_link[0,2]-head_link[1,2])/self.lengths[0]))

    def _prepared_helix(self):
        if not self.options.prepare_front:
            return None
        amount = ease(self.shift/.075)
        if self.stage == 'receive':
            amount *= 1.-ease((self.shift-self.receive_start)/.14)
        opt = self.options
        return helix(0., opt.radius_m-.012*amount, opt.pitch_m, -self.direction)

    def _advance_staged(self, dt, fresh):
        opt, curve = self.options, self.curve
        first, last = self.window
        leading = self.stage == 'lead'
        pace = self._tracking_pace()
        distance = opt.staged_shift_m_s*pace*dt
        self.shift_time += dt
        self.roll += self.roll_direction*opt.staged_roll_rad_s*ease(self.shift_time/1.5)*pace*dt
        if leading:
            self.shift = min(curve.head_length, self.shift+distance)
        else:
            self.receive_shift = min(curve.tail_length, self.receive_shift+distance)
        # Keep the initial supporting ARC while only the first straight arm
        # is converted. Shorten the ARC as the second arm enters the curve.
        arc_length = curve.arc_length*(1.-ease(self.receive_shift/min(.12,curve.tail_length)))
        goal, _ = curve.targets(self.shift, self.receive_shift, arc_length, self.phase+self.roll)
        j = np.arange(1,self.n+1)
        coverage = np.clip(np.where(j < first, self.shift/self.l+j-first,
                                    np.where(j > last, self.receive_shift/self.l+last-j, 1.))+.999, 0., 1.)
        goal = self._apply_curve(goal, coverage, opt.staged_target_rate_deg_s, dt)
        if not fresh:
            return
        if leading:
            error = np.max(abs(goal[:last]-self.measured[:last]))
            settled = self.shift == curve.head_length and error <= math.radians(opt.intake_error_deg)
        else:
            error = np.max(abs(goal-self.measured))
            settled = self.receive_shift == curve.tail_length and error <= math.radians(10.)
        self.ready = self.ready+1 if settled else 0
        if self.ready >= 3:
            self.ready = 0
            self.stage = 'receive' if leading else 'complete'
            self.events.append(dict(stage=self.stage, shift_m=self.shift, receive_shift_m=self.receive_shift))
            if not leading:
                self.pause()

    def _advance_shift(self, dt, fresh):
        opt, curve = self.options, self.curve
        pace = self._tracking_pace()
        limit = (self.accepted-self.window[1]+1.5)*self.l if self.accepted < self.n else math.inf
        self.shift_time += dt
        if opt.prepare_front:
            self.shift += min(max(0., limit-self.shift), opt.shift_m_s*pace*dt)
            self.receive_shift = self.shift-self.receive_start
        else:
            self.shift = min(limit, self.shift+opt.shift_m_s*pace*dt)
        self.roll += self.roll_direction*opt.roll_rad_s*ease(self.shift_time/1.5)*pace*dt
        arc_length = curve.arc_length*(1.-ease(self.shift/.16))
        goal, _, coverage = curve.shifted_window(self.shift, arc_length, self.phase+self.roll,
                                                self.leading_extension, self._prepared_helix())
        if self.options.prepare_front:
            coverage[self.accepted+1:] = 0.
        goal = self._apply_curve(goal, coverage, opt.target_rate_deg_s, dt)
        if fresh and self.accepted < self.n:
            i = self.accepted
            follows = np.max(np.abs(goal[i-1:i+1]-self.measured[i-1:i+1])) <= math.radians(opt.intake_error_deg)
            self.confirmed = self.confirmed+1 if coverage[i] >= .65 and follows else 0
            if self.confirmed >= 2:
                self.accepted += 1
                self.confirmed = 0
                joint = self.n+1-self.accepted if self.reversed else self.accepted
                self.events.append(dict(stage='intake', joint=joint, shift_m=self.shift))
        if fresh and self.accepted == self.n and np.all(coverage >= .999):
            self.ready = self.ready+1 if np.max(np.abs(goal-self.measured)) <= math.radians(10.) else 0
            if self.ready >= 3:
                self.stage = 'complete'
                self.pause()
                self.events.append(dict(stage=self.stage, shift_m=self.shift))

    def _apply_curve(self, goal, coverage, rate_deg_s, dt):
        goal = np.clip(goal+self.residual*(1.-ease(self.shift_time/2.)), -math.radians(88.), math.radians(88.))
        entering = (coverage > 1.e-6) & ~self._enabled
        self._targets[entering] = self.measured[entering]
        self._enabled |= entering
        # Apply the overlap as a position blend. Both backends use the same
        # binary enable mask; no simulated stiffness or contact is supplied.
        receive = self._enabled & (coverage < .999)
        blended = goal.copy()
        blended[receive] = self.measured[receive]+coverage[receive]*(goal[receive]-self.measured[receive])
        rate = math.radians(rate_deg_s)*dt
        self._targets += np.clip(blended-self._targets,-rate,rate)
        self._targets[~self._enabled] = self.measured[~self._enabled]
        return goal
