"""ARC to helix using the sampled motor angles and the active ARC window."""
from dataclasses import dataclass
import math

import numpy as np

from .active_arc_transfer import ActiveArcTransfer, TransferOptions, ease
from .curve_gait import CurveGait
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
        head_depth, tail_depth = self._end_depth(q, section)
        self.reversed = bool(head_depth > tail_depth+2*self.l)
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
        self.step_left = 0.
        self.events = [dict(stage='entry', reversed_intake=self.reversed, window=list(window),
                            head_depth_m=float(head_depth), tail_depth_m=float(tail_depth))]
        self.transfer = None
        odd = np.arange(self.n)%2 == 0
        components = [np.mean(self.measured[self._enabled & parity]) for parity in (odd, ~odd)]
        self.amplitude = float(np.linalg.norm(components))
        self.phase = math.atan2(components[1], components[0])
        short_depth = tail_depth if self.reversed else head_depth
        self._plan_entry(short_depth)

    def _end_depth(self, q, section):
        """End distance below the fitted ARC center along its crown direction."""
        points = section['projected'][1:-1]
        system = np.column_stack((2.*points, np.ones(len(points))))
        center = np.linalg.lstsq(system, np.sum(points**2, axis=1), rcond=None)[0][:2]
        crown = points.mean(axis=0)-center
        direction = crown/np.linalg.norm(crown)
        ends = forward(q, self.lengths)[[0,-1]]
        projected = (ends-section['axis']['point'])@section['axis']['basis'][:,:2]
        return -(projected-center)@direction

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

    def _plan_entry(self, short_depth):
        first, last = self.window
        # Add a hinge at the leading end if its measured section has little
        # room for the first helical shoulder.
        if short_depth < self.options.min_end_depth_links*self.l and last < self.n-1:
            self._prepare_transfer(1)
        elif last == self.n-1 and first > 2:
            self._prepare_transfer(-1)
        else:
            self._prepare_shift()

    def _prepare_transfer(self, direction):
        first, last = self.window
        self.stage = 'adjust'
        options = TransferOptions(first_joint=first, last_joint=last, hold_phase_rad=self.phase,
                                  amplitude_deg=math.degrees(self.amplitude), desired_direction=direction,
                                  loose_amplitude_deg=min(45., math.degrees(self.amplitude)))
        self.transfer = ActiveArcTransfer(self.lengths, 'transfer', options)
        self.transfer.phase = self.phase
        self.transfer.targets = self._targets.copy()
        self.transfer_clock = 0.
        self.transfer_feedback_sample = None
        self.transfer.begin_transfer(direction, np.degrees(self.measured), 0., 'leave two receiving hinges')
        self.events.append(dict(stage='adjust', before=list(self.physical_window(self.window)),
                                after=list(self.physical_window((first+direction,last+direction)))))

    def _prepare_shift(self):
        self.stage = 'shift'
        self.shift = self.roll = self.shift_time = 0.
        self.accepted = self.window[1]
        self.confirmed = 0
        self.ready = 0
        opt = self.options
        self.curve = ArcHelixShift(CurveGait(self.n, self.l), self.window, self.amplitude,
                                   opt.radius_m, opt.pitch_m, self.direction if self.reversed else -self.direction)
        self.curve.front_extension_rate = max(1.75, 1.+self.curve.head_length/max(.40,self.curve.tail_length))
        nominal, _, _ = self.curve.shifted_window(0., self.curve.arc_length, self.phase)
        self.residual = np.where(self._enabled, self.measured-nominal, 0.)
        self.events.append(dict(stage='shift', window=list(self.physical_window(self.window))))

    def step_once(self):
        if self.stage not in ('complete','blocked'):
            self.running = False
            self.step_left = self.options.step_s

    def pause(self):
        self.running = False
        self.step_left = 0.

    def status(self):
        accepted = getattr(self, 'accepted', None)
        if accepted is not None and self.reversed:
            accepted = self.n+1-accepted
        return dict(stage=self.stage, arc_window=list(self.physical_window(self.window)),
                    intake_end='head' if self.reversed else 'tail',
                    accepted_joint=accepted,
                    shift_m=getattr(self, 'shift', 0.), step_remaining_s=self.step_left,
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
        dt = dt if self.running else min(dt,self.step_left)
        self.step_left = max(0., self.step_left-dt)
        if self.step_left < 1e-9:
            self.step_left = 0.
        if self.stage == 'adjust':
            self._advance_transfer(dt)
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
            self._prepare_shift()

    def _advance_shift(self, dt, fresh):
        opt = self.options
        curve = self.curve
        error = np.max(np.abs(self._targets[self._enabled]-self.measured[self._enabled]))
        pace = float(np.clip((math.radians(25.)-error)/math.radians(18.), .15, 1.))
        limit = (self.accepted-self.window[1]+1.5)*self.l if self.accepted < self.n else math.inf
        self.shift_time += dt
        self.shift = min(limit, self.shift+opt.shift_m_s*pace*dt)
        self.roll += self.roll_direction*opt.roll_rad_s*ease(self.shift_time/1.5)*pace*dt
        arc_length = curve.arc_length*(1.-ease(self.shift/.16))
        goal, _, coverage = curve.shifted_window(self.shift, arc_length, self.phase+self.roll)
        goal = np.clip(goal+self.residual*(1.-ease(self.shift_time/2.)), -math.radians(88.), math.radians(88.))
        entering = (coverage > 1.e-6) & ~self._enabled
        self._targets[entering] = self.measured[entering]
        self._enabled |= entering
        # Apply the overlap as a position blend. Both backends use the same
        # binary enable mask; no simulated stiffness or contact is supplied.
        receive = self._enabled & (coverage < .999)
        blended = goal.copy()
        blended[receive] = self.measured[receive]+coverage[receive]*(goal[receive]-self.measured[receive])
        rate = math.radians(opt.target_rate_deg_s)*dt
        self._targets += np.clip(blended-self._targets,-rate,rate)
        self._targets[~self._enabled] = self.measured[~self._enabled]
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
