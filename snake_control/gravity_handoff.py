"""实机 T 管越障。几何事件由操作者确认，进给只使用全身实测关节角。"""
from dataclasses import dataclass
import math

import numpy as np

from .arc_helix_transition import ArcHelixOptions, ArcHelixTransition
from .gravity_global_shift import MeasuredGravityShift
from .gravity_moment import GravityMomentGait, GravityMomentOptions, smooth
from .section_geometry import local_section


class GravityFrontTransition(ArcHelixTransition):
    """保持承载 ARC，只延长头端螺旋，直至操作者确认贴管。"""

    def _check_front_entry(self):
        section = local_section(self.measured, self.lengths, *self.window)
        head, tail, _, _ = self.estimate_end_geometry(self.measured, self.lengths, section, self.window)
        self.events.append(dict(stage='front_check', head_length_m=head, tail_length_m=tail))
        if head+.5*self.l <= tail:
            self._prepare_shift()
        elif self.window[0] > 2:
            self._prepare_transfer(-1)
        else:
            self.stage = 'blocked'
            self.pause()
            self.events.append(dict(stage='blocked', reason='调到头侧边界后仍未形成头短尾长的起卷姿态'))

    def _advance_prepared_lead(self, dt, fresh):
        first, last = self.window
        pace = self._tracking_pace()
        self.shift_time += dt
        self.leading_extension = min(self.curve.head_length, self.leading_extension+.035*pace*dt)
        self.roll += .23*smooth(self.shift_time/1.5)*pace*dt
        goal, _ = self.curve.targets(self.leading_extension, 0., self.curve.arc_length, self.phase+self.roll)
        coverage = np.clip(self.leading_extension/self.l+np.arange(1, self.n+1)-first+1., 0., 1.)
        coverage[last:] = 0.
        self._apply_curve(goal, coverage, 23., dt)


@dataclass(frozen=True)
class GravityCrossingOptions:
    retain_joints: int = 8
    release_line_m: float = .32201388
    release_radius_m: float = .09554165
    alignment_rate_rad_s: float = .14
    shift_rate_m_s: float = .012
    shift_roll_rad_s: float = .35
    intake_error_deg: float = 16.
    step_s: float = 1.
    max_feedback_age_s: float = .8


class GravityCrossing:
    """头侧成形 → Shift Rolling → 解旋成右 U → 失能 → 回卷。"""

    def __init__(self, lengths, enabled, measured_deg, gait_options=GravityMomentOptions(r_h1=.094, r_h2=.098),
                 options=GravityCrossingOptions(), from_helix=False):
        lengths = np.asarray(lengths, float)
        self.n, self.l = len(lengths)-1, float(np.mean(lengths))
        if not np.allclose(lengths, self.l):
            raise ValueError('T 管模式的曲线积分需要等间距关节')
        if not 4 <= options.retain_joints < self.n:
            raise ValueError('请检查 T 管模式的前部保留关节数')
        self.gait = GravityMomentGait(self.n, gait_options, self.l)
        self.gait.segments(options.release_radius_m, ls=options.release_line_m, th1=math.pi)
        self.options = options
        self._enabled = np.asarray(enabled, bool).copy()
        self.measured = np.radians(measured_deg)
        self._targets = self.measured.copy()
        self.running = self.started = False
        self.reversed = False  # 本模式固定 J1 为前端，右 U 的方向与此绑定。
        self.step_left = self.elapsed = self.phase = self.offset = self.roll = 0.
        self.feedback_time = None
        self.accepted = None
        self.front = None
        self.front_extended = False
        self.align_direction = 1
        self.events = []
        if from_helix:
            if not self._enabled.all():
                raise ValueError('从 Helix 接入 T 管模式需要已成形的全身螺旋')
            self.phase = self._fit_phase(self.n, gait_options.r_h1)
            error = np.sqrt(np.mean((self._helix(self.phase, gait_options.r_h1)-self.measured)**2))
            if error > math.radians(16.):
                raise ValueError('实测 Helix 的绕向或形状与 J1 在前的 T 管曲线不符')
            self._enter('align')
        else:
            active = np.flatnonzero(self._enabled)+1
            if len(active) < 3 or np.any(np.diff(active) != 1):
                raise ValueError('T 管模式需要先形成连续的局部承载圆弧')
            arc_options = ArcHelixOptions(radius_m=gait_options.r_h1, pitch_m=gait_options.pitch,
                                         prepare_front=True)
            self.front = GravityFrontTransition(lengths, (int(active[0]), int(active[-1])),
                                                measured_deg, options=arc_options)
            self._enter('front')

    @property
    def targets(self):
        return self._targets.copy()

    @property
    def enabled(self):
        return self._enabled.copy()

    def _enter(self, stage):
        self.stage, self.elapsed = stage, 0.
        self.events.append(dict(stage=stage, phase_rad=self.phase, enabled=(np.flatnonzero(self._enabled)+1).tolist()))

    def pause(self):
        self.running = False
        self.step_left = 0.

    def step_once(self):
        self.running = False
        self.step_left = self.options.step_s

    def _helix(self, phase, radius):
        return self.gait.targets(phase, radius)[0]

    def _fit_phase(self, count, radius):
        basis = np.column_stack([self._helix(p, radius)[:count] for p in (0., math.pi/2)])
        weights = np.linalg.lstsq(basis, self.measured[:count], rcond=None)[0]
        return math.atan2(weights[1], weights[0])

    def _start_shift(self, support, stage):
        radius = self.gait.o.r_h2
        self.initial_prefix = self.accepted = support
        self.phase = self._fit_phase(support, radius)
        self.curve = MeasuredGravityShift(self.gait, radius, self.phase, self.measured, support,
                                         upstream_m=(support+1)*self.l)
        self.offset = self.roll = 0.
        self.tracking_since = None
        self._enter(stage)

    def confirm(self, feedback_time, measured_deg):
        """Ctrl+L 标记操作者看到的几何事件，之后仍保持到 J/G。"""
        if self.running or self.step_left > 0.:
            raise ValueError('请先暂停，再按 Ctrl+L 确认实际姿态')
        self.measured = np.radians(measured_deg)
        self.feedback_time = feedback_time
        self.started = True
        if self.stage == 'front':
            if self.front.stage != 'lead' or self.front.leading_extension < self.front.curve.head_length:
                raise ValueError('前部曲线还在形成，请先用 Ctrl+J/G 完成前卷')
            self._start_shift(self.front.window[1], 'first_shift')
        elif self.stage == 'align':
            self.anchor = self._targets.copy()
            self._enter('unwind')
        elif self.stage == 'right_u_hold' or (self.stage == 'unwind' and self.elapsed >= 3.):
            self.anchor = self._targets.copy()
            self._enter('drop')
        elif self.stage == 'drop_hold':
            self._start_shift(self.options.retain_joints, 'rewrap')
        else:
            raise ValueError('当前阶段用 Ctrl+J 单步或 Ctrl+G 连续推进')

    def _set_targets(self, goal, dt, rate=18.):
        limit = math.radians(rate)*dt
        goal = np.clip(goal, -math.radians(88.), math.radians(88.))
        self._targets += np.clip(goal-self._targets, -limit, limit)
        self._targets[~self._enabled] = self.measured[~self._enabled]

    def advance(self, dt, now, feedback_time, measured_deg):
        if (feedback_time is None or now-feedback_time > self.options.max_feedback_age_s or
                (not self.running and self.step_left <= 0.)):
            return False
        fresh = self.feedback_time is None or feedback_time > self.feedback_time
        if fresh:
            self.measured = np.radians(measured_deg)
            self.feedback_time = feedback_time
        self.started = True
        dt = dt if self.running else min(dt, self.step_left)
        self.step_left = max(0., self.step_left-dt)
        if self.step_left < 1e-9:
            self.step_left = 0.
        self.elapsed += dt

        if self.stage == 'front':
            self.front.running = True
            self.front.advance(dt, now, feedback_time, measured_deg)
            self._enabled, self._targets = self.front.enabled, self.front.targets
            self.phase = self.front.phase+self.front.roll if self.front.stage == 'lead' else self.front.phase
            if self.front.stage == 'blocked':
                self.pause()
                self._enter('blocked')
            elif (not self.front_extended and self.front.stage == 'lead' and
                  self.front.leading_extension >= self.front.curve.head_length):
                self.front_extended = True
                self.pause()
        elif self.stage in ('first_shift', 'rewrap'):
            self._advance_shift(dt, fresh)
        elif self.stage == 'align':
            self.phase += self.align_direction*self.options.alignment_rate_rad_s*dt
            self._set_targets(self._helix(self.phase, self.gait.o.r_h1), dt)
        elif self.stage == 'unwind':
            self._advance_unwind(dt)
        elif self.stage == 'drop':
            self._enabled[self.options.retain_joints:] = False
            opened = self._helix(self.phase, self.gait.o.r_h2)
            self._targets = self.anchor+smooth(self.elapsed/.40)*(opened-self.anchor)
            self._targets[~self._enabled] = self.measured[~self._enabled]
            if self.elapsed >= 1.-1e-9:
                self._enter('drop_hold')
                self.pause()
        elif self.stage in ('first_recover', 'recover'):
            self._advance_recover(dt)
        elif self.stage == 'roll':
            self.phase += self.gait.o.roll_rate*dt
            self._set_targets(self._helix(self.phase, self.gait.o.r_h1), dt, rate=45.)
        return True

    def _advance_unwind(self, dt):
        t, opt, gait = self.elapsed, self.options, self.gait
        radius = gait.o.r_h1+(opt.release_radius_m-gait.o.r_h1)*smooth((t-7.)/.85)
        goal = gait.targets(self.phase, radius, ls=opt.release_line_m*smooth((t-3.)/4.),
                            th1=math.pi*smooth(t/3.))[0]
        residual = self.anchor-self._helix(self.phase, gait.o.r_h1)
        self._set_targets(goal+(1.-smooth(t))*residual, dt, rate=30.)
        if t >= 7.85:
            self._enter('right_u_hold')
            self.pause()

    def _advance_shift(self, dt, fresh):
        """采样窗向前移动，滚动相位同时推进，尾侧从已接管边界逐节加入。"""
        opt, count = self.options, self.accepted
        stop = min(self.n, count+3)
        error = np.max(abs(self._targets[self._enabled]-self.measured[self._enabled]))
        pace = float(np.clip((math.radians(23.)-error)/math.radians(17.), .10, 1.))
        limit = (count-self.initial_prefix+2)*self.l if count < self.n else self.curve.shift_limit_m
        advance = min(opt.shift_rate_m_s*pace*dt, max(0., limit-self.offset))
        self.offset += advance
        turn = advance*opt.shift_roll_rad_s/.05
        if count < self.n and self.offset >= limit:
            turn = max(turn, .055*pace*dt)
        self.roll += turn
        self.phase = self.curve.phase_rad-self.curve.torsion*self.offset+self.roll
        goal = self.curve.targets(self.offset, self.roll)
        nominal = self.curve.helical_targets(self.offset, self.roll)
        blend = smooth(self.offset/self.gait.turn_length(self.gait.o.r_h2))
        goal[:count] += blend*(nominal[:count]-goal[:count])

        # 舵机是二值使能。边界后最多三节以位置目标渐入，未到边界的尾段仍失能。
        for i in range(count, stop):
            reach = smooth((self.offset/self.l-(i-self.initial_prefix)+1.)/.75)
            if reach > 0.:
                if not self._enabled[i]:
                    self._targets[i] = self.measured[i]
                    self._enabled[i] = True
                weight = reach*[.55, .45, .30][i-count]
                goal[i] = self.measured[i]+weight*(goal[i]-self.measured[i])
        self._set_targets(goal, dt)
        if not fresh:
            return
        if count < self.n:
            arrived = self.offset >= (count-self.initial_prefix+.75)*self.l
            follows = np.max(abs(nominal[count-1:count+1]-self.measured[count-1:count+1])) <= math.radians(opt.intake_error_deg)
            ready = arrived and follows
        else:
            ready = np.max(abs(nominal-self.measured)) <= math.radians(10.)
        self.tracking_since = (self.elapsed if self.tracking_since is None else self.tracking_since) if ready else None
        if self.tracking_since is not None and self.elapsed-self.tracking_since >= .30:
            self.tracking_since = None
            if count < self.n:
                self.accepted += 1
                self.events.append(dict(stage='intake', joint=self.accepted, shift_m=self.offset))
            else:
                self.recover_phase, self.anchor = self.phase, self._targets.copy()
                self._enter('first_recover' if self.stage == 'first_shift' else 'recover')

    def _advance_recover(self, dt):
        t, gait = self.elapsed, self.gait
        start_rate = self.options.shift_rate_m_s*(self.options.shift_roll_rad_s/.05-self.curve.torsion)
        rate = start_rate+(gait.o.roll_rate-start_rate)*smooth((t-3.)/2.)
        self.phase += rate*dt
        radius = gait.o.r_h2+(gait.o.r_h1-gait.o.r_h2)*smooth(t/3.)
        residual = self.anchor-self._helix(self.recover_phase, gait.o.r_h2)
        self._set_targets(self._helix(self.phase, radius)+(1.-smooth(t/2.))*residual, dt, rate=45.)
        if t >= 5.:
            if self.stage == 'first_recover':
                self._enter('align')
                self.pause()
            else:
                self._enter('roll')

    def status(self):
        front = None if self.front is None else self.front.status()
        return dict(stage=self.stage, elapsed_s=self.elapsed, phase_rad=math.remainder(self.phase, math.tau),
                    accepted_joint=self.accepted, shift_m=self.offset, step_remaining_s=self.step_left,
                    front=front, events=self.events.copy())
