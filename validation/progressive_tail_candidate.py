"""FAILED progressive-tail candidates retained for reproduction; not a real controller."""
from dataclasses import dataclass
import math
import numpy as np
from snake_control.gravity_handoff import GravityCrossing as BaselineCrossing, GravityCrossingOptions as BaselineOptions
from snake_control.gravity_moment import GravityMomentGait, GravityMomentOptions

class ProgressiveHelixTail:
    """Move helix/arc/line joins, taking over each hinge from its measured pose.

    The supported front never contributes a transported fitting residual.
    Each entering hinge's initial correction disappears over two link lengths
    of join travel. This avoids both straightening the tail on entry and
    transporting its incidental bends as a wave through the whole robot.
    """

    def __init__(self, gait, radius_m, phase_rad, support_count, measured_rad):
        if not 1 <= support_count < gait.gait.joints:
            raise ValueError('Expected a supported front and a nonempty tail')
        self.gait, self.radius, self.phase = gait, radius_m, phase_rad
        self.support_count = support_count
        self.helix_start_length = (support_count+1)*gait.l
        self.shift_limit_m = (gait.gait.joints-support_count+1)*gait.l
        self.tangent_length = gait.o.r_c*gait.lead_angle(radius_m)
        self.initial_correction = np.asarray(measured_rad, float)-self.nominal_targets(0.)
        self.initial_correction[:support_count] = 0.
        self.entry_coordinates = np.maximum(0., np.arange(gait.gait.joints)-support_count)*gait.l

    def geometry(self, shift_m):
        if not math.isfinite(shift_m) or not 0 <= shift_m <= self.shift_limit_m+1e-10:
            raise ValueError('Tail join lies outside the robot')
        helix_length = min(self.gait.L, self.helix_start_length+shift_m)
        arc_length = min(self.tangent_length, max(0., self.gait.L-helix_length))
        straight_length = max(0., self.gait.L-helix_length-arc_length)
        return helix_length, arc_length, straight_length

    def targets(self, shift_m):
        progress = np.clip((shift_m-self.entry_coordinates)/(2*self.gait.l), 0., 1.)
        blend = progress*progress*(3.-2.*progress)
        return self.nominal_targets(shift_m)+(1.-blend)*self.initial_correction

    def nominal_targets(self, shift_m):
        lh, la, ls = self.geometry(shift_m)
        return self.gait.targets(self.phase, self.radius, ls=ls, th2=la/self.gait.o.r_c)[0]

    def enabled(self, shift_m):
        lh, la, _ = self.geometry(shift_m)
        joint = np.arange(1, self.gait.gait.joints+1)
        # Each hinge integrates two links; enable it when its first link enters
        # the non-straight curve. This includes the necessary alternating axis.
        active_end = min(self.gait.gait.joints, self.support_count+2+int(shift_m/self.gait.l+1e-9))
        return joint <= active_end

    def helix_prefix(self, shift_m):
        lh, _, _ = self.geometry(shift_m)
        return min(self.gait.gait.joints, max(self.support_count,
                   int(math.floor(lh/self.gait.l+1e-9))-1))


@dataclass(frozen=True)
class GravityCrossingOptions(BaselineOptions):
    tail_shift_rate_m_s: float = .04
    tail_target_rate_deg_s: float = 45.


class GravityCrossing(BaselineCrossing):
    def __init__(self, *args, options=None, **kwargs):
        super().__init__(*args, options=options or GravityCrossingOptions(), **kwargs)
        self.tail_rate_m_s = self.tail_error_deg = 0.
        self.tail_wait_reason = None

    def _start_tail_intake(self):
        self.accepted = self.initial_prefix = self.front_count
        # 固定前部相位；仅移动螺旋—切线弧—被动尾段的交界。
        self.curve = ProgressiveHelixTail(self.gait, self.gait.o.r_h1, self.phase, self.front_count, self.measured)
        self.locked_front = self._targets[:self.front_count].copy()
        self.offset = 0.
        self.tracking_since = None
        self._enter('first_shift')


    def _advance_tail_intake(self, dt, fresh):
        """增长螺旋段、缩短被动尾段；曲线交界和逐段接管使用同一个坐标。"""
        goal = self.curve.targets(self.offset)
        tail_mask = self.curve.enabled(self.offset)
        tail_mask[:self.front_count] = False
        error = np.max(abs(goal[tail_mask]-self.measured[tail_mask]), initial=0.)
        self.tail_error_deg = float(np.degrees(error))
        # 已发目标和实际形状未跟上时，仍向当前曲线目标收敛，但不把交界甩到前面。
        pace = float(np.clip((math.radians(20.)-error)/math.radians(14.), 0., 1.))
        previous = self.offset
        self.offset = min(self.curve.shift_limit_m, self.offset+self.options.tail_shift_rate_m_s*pace*dt)
        self.tail_rate_m_s = (self.offset-previous)/dt
        self.tail_wait_reason = 'tail_curve_tracking' if pace < 1. else None
        enabled = self.curve.enabled(self.offset)
        entering = enabled & ~self._enabled
        self._targets[entering] = self.measured[entering]
        self._enabled = enabled
        goal = self.curve.targets(self.offset)
        goal[:self.front_count] = self.locked_front
        self._set_targets(goal, dt, rate=self.options.tail_target_rate_deg_s)
        self._targets[:self.front_count] = self.locked_front
        accepted = self.curve.helix_prefix(self.offset)
        if accepted != self.accepted:
            self.accepted = accepted
            self.events.append(dict(stage='intake', joint=accepted, shift_m=self.offset))
        if fresh and self.offset >= self.curve.shift_limit_m-1e-9:
            error = np.max(abs(goal[self.front_count:]-self.measured[self.front_count:]), initial=0.)
            follows = error <= math.radians(5.)
            self.tracking_since = (self.elapsed if self.tracking_since is None else self.tracking_since) if follows else None
            if self.tracking_since is not None and self.elapsed-self.tracking_since >= .2:
                self._enter('complete' if self.helix_only else 'align')
                self.pause()


    def status(self):
        result = super().status()
        result.update(tail_rate_m_s=self.tail_rate_m_s, tail_error_deg=self.tail_error_deg,
                      tail_wait_reason=self.tail_wait_reason)
        return result
