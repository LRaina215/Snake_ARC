"""实机 T 管越障。几何事件由操作者确认，进给只使用全身实测关节角。"""
from dataclasses import dataclass
import math

import numpy as np

from .arc_helix_transition import ArcHelixOptions, ArcHelixTransition
from .curve_gait import helix
from .continuous_feeding import ContinuousFeeding, ContinuousFeedingOptions
from .gravity_global_shift import MeasuredGravityShift
from .gravity_moment import GravityMomentGait, GravityMomentOptions, smooth
from .section_geometry import local_section
from .posture_initialization import choose_feeding_arc


class GravityFrontTransition(ArcHelixTransition):
    """保持承载 ARC，只延长头端螺旋，直至操作者确认贴管。"""

    def __init__(self, *args, minimum_front_turns=1., **kwargs):
        self.minimum_front_turns = minimum_front_turns
        super().__init__(*args, **kwargs)

    def _check_front_entry(self):
        section = local_section(self.measured, self.lengths, *self.window)
        head, tail, _, _ = self.estimate_end_geometry(self.measured, self.lengths, section, self.window)
        self.events.append(dict(stage='front_check', head_length_m=head, tail_length_m=tail))
        turn_length = math.hypot(2.*math.pi*self.options.radius_m, self.options.pitch_m)
        front_count = math.ceil(self.minimum_front_turns*turn_length/self.l)-1
        self.support_joint_count = front_count
        width = self.window[1]-self.window[0]+1
        # 留出两种交替转轴所需的头端长度，已有较短头端不再被无谓拉长。
        minimum_start = 3
        first_turn_count = math.ceil(turn_length/self.l)-1
        maximum_start = max(minimum_start, first_turn_count-width+1)
        start = int(np.clip(self.window[0], minimum_start, min(self.n-width, maximum_start)))
        self.desired_start = start
        self.events.append(dict(stage='front_plan', desired_window=[start, start+width-1],
                                front_material_m=(front_count+1)*self.l))
        if self.window[0] < start:
            self._prepare_transfer(1)
        elif self.window[0] > start:
            self._prepare_transfer(-1)
        else:
            self._prepare_shift()

    def _prepare_transfer(self, direction):
        """自动调长复用模式4的曲线分配、相邻折弯接管与目标限速。"""
        first, last = self.window
        goal = self.desired_start-first
        options = ContinuousFeedingOptions(first_joint=first, last_joint=last,
            amplitude_deg=math.degrees(self.amplitude), requested_transfer_m=goal*self.l,
            automatic_return=False, tracking_tolerance_deg=8.)
        self.transfer = ContinuousFeeding(self.lengths, options)
        self.transfer.phase = self.phase
        self.transfer.targets = self.measured.copy()
        self.transfer.enabled = self._enabled.copy()
        self.transfer_goal = float(goal)
        self.transfer_clock = 0.
        self.transfer_feedback_sample = None
        self.transfer_ready_since = None
        self.stage = 'adjust'
        self.events.append(dict(stage='adjust', before=list(self.window),
                                desired_start=self.desired_start, generator='mode4_profile'))

    def _advance_transfer(self, dt):
        helper = self.transfer
        self.transfer_clock += dt
        clock = self.transfer_clock
        fresh = self.transfer_feedback_sample != self.feedback_time
        if fresh:
            self.transfer_feedback_sample = self.feedback_time
            self.transfer_feedback_clock = clock
        helper.advance(np.degrees(self.measured), self.transfer_feedback_clock, clock)
        if helper.forced_step is None:
            error = np.max(abs(helper.targets[helper.enabled]-self.measured[helper.enabled]))
            remaining = abs(self.transfer_goal-helper.position)
            if remaining > 1e-9:
                pace = float(np.clip((math.radians(23.)-error)/math.radians(17.), .1, 1.))
                increment = min(remaining, .1*pace*dt)
                _, focus, _, _ = helper.profile(helper.position, helper.bias)
                required = helper.required_mask(focus, np.degrees(self.measured), helper.position)
                entering = required & ~helper.enabled
                helper.targets[entering] = self.measured[entering]
                helper.enabled |= required
                helper.start_forced_step(np.degrees(self.measured), increment, clock)
            elif fresh:
                follows = error <= math.radians(8.)
                self.transfer_ready_since = (clock if self.transfer_ready_since is None else
                                             self.transfer_ready_since) if follows else None
                if self.transfer_ready_since is not None and clock-self.transfer_ready_since >= .4:
                    _, focus, _, _ = helper.profile(helper.position, helper.bias)
                    selection = choose_feeding_arc(np.degrees(self.measured), self.lengths, helper.enabled, focus)
                    self.window = tuple(selection['chosen'])
                    self.phase = helper.phase
                    self.stage = 'transfer_done'
        self._targets = helper.targets.copy()
        self._enabled = helper.enabled.copy()

    def _advance_prepared_lead(self, dt, fresh):
        first, last = self.window
        pace = self._tracking_pace()
        self.shift_time += dt
        self.leading_extension = min(self.curve.head_length, self.leading_extension+.035*pace*dt)
        # 承载 ARC 的相位保持，前部只沿拼接曲线逐渐成形。
        leading_helix = helix(0., self.options.radius_m, self.options.pitch_m, -self.direction)
        goal, self.head_phase, _ = self.curve.shifted_window(0., self.curve.arc_length, self.phase+self.roll,
                                                          self.leading_extension, leading_helix)
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
    return_roll_rad_s: float = .084
    step_s: float = 1.
    max_feedback_age_s: float = .8
    min_front_turns: float = 1.
    front_seat_rad: float = .14


class GravityCrossing:
    """自主调长 → 前部成形保持 → 逐节接尾；可继续重力矩释放与回卷。"""

    def __init__(self, lengths, enabled, measured_deg, gait_options=GravityMomentOptions(r_h1=.094, r_h2=.098),
                 options=GravityCrossingOptions(), from_helix=False, helix_only=False):
        lengths = np.asarray(lengths, float)
        self.n, self.l = len(lengths)-1, float(np.mean(lengths))
        if not np.allclose(lengths, self.l):
            raise ValueError('T 管模式的曲线积分需要等间距关节')
        if not 4 <= options.retain_joints < self.n:
            raise ValueError('请检查 T 管模式的前部保留关节数')
        self.gait = GravityMomentGait(self.n, gait_options, self.l)
        self.gait.segments(options.release_radius_m, ls=options.release_line_m, th1=math.pi)
        self.options = options
        self.helix_only = helix_only
        self._enabled = np.asarray(enabled, bool).copy()
        self.measured = np.radians(measured_deg)
        self._targets = self.measured.copy()
        self.running = self.started = False
        self.reversed = False  # 本模式固定 J1 为前端，右 U 的方向与此绑定。
        self.step_left = self.elapsed = self.phase = self.offset = 0.
        self.feedback_time = None
        self.accepted = None
        self.front = None
        self.front_extended = False
        self.front_ready_since = None
        self.front_tracking_error_deg = None
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
                                                measured_deg, options=arc_options,
                                                minimum_front_turns=options.min_front_turns)
            self._plan_arc_reference(lengths)

    @staticmethod
    def _section_radius(q, lengths, window):
        xy = local_section(q, lengths, *window)['projected'][1:-1]
        fit = np.linalg.lstsq(np.column_stack((2.*xy, np.ones(len(xy)))), (xy*xy).sum(axis=1), rcond=None)[0]
        return math.sqrt(max(0., fit[2]+fit[:2]@fit[:2]))

    def _plan_arc_reference(self, lengths):
        """将送链留下的不均匀圆弧接到同半径、同总弯曲的曲线基元。"""
        first, last = self.front.window
        section = self.measured[first-1:last]
        parity = np.arange(first-1, last)%2
        budget = np.array([section[parity == axis].sum() for axis in (0, 1)])
        phase = math.atan2(budget[1], budget[0])
        radius = self._section_radius(self.measured, lengths, (first, last))
        count = last-first+1
        candidates = []
        for size in sorted({max(4, count-count%2), count+count%2}):
            amplitude = np.linalg.norm(budget)/(size/2.)
            for start in range(max(2, first-1), min(self.n-size, first+1)+1):
                window = (start, start+size-1)
                mask = (np.arange(1, self.n+1) >= window[0]) & (np.arange(1, self.n+1) <= window[1])
                target = np.where(np.arange(self.n)%2 == 0, amplitude*math.cos(phase), amplitude*math.sin(phase))
                target[~mask] = 0.
                fitted_radius = self._section_radius(target, lengths, window)
                score = abs(fitted_radius-radius)/.01+.05*(abs(start-first)+abs(window[1]-last))
                candidates.append((score, window, target, mask))
        _, window, target, mask = min(candidates, key=lambda item: item[0])
        self.arc_reference = dict(window=window, target=target, mask=mask, lengths=lengths.copy())
        shared = self._enabled | mask
        change = np.max(abs(target[shared]-self.measured[shared]))
        if window != (first, last) or change > math.radians(8.):
            self._enabled |= mask
            self.arc_reference_ready_since = None
            self._enter('arc_settle')
        else:
            self._enabled = self.front.enabled
            self._enter('front')

    def _advance_arc_reference(self, dt, fresh):
        reference = self.arc_reference
        self._set_targets(reference['target'], dt, rate=15.)
        error = np.max(abs(reference['target'][self._enabled]-self.measured[self._enabled]))
        if fresh:
            ready = error <= math.radians(8.)
            self.arc_reference_ready_since = (self.elapsed if self.arc_reference_ready_since is None else
                                              self.arc_reference_ready_since) if ready else None
            if self.arc_reference_ready_since is not None and self.elapsed-self.arc_reference_ready_since >= .4:
                self._enabled = reference['mask'].copy()
                self.front = GravityFrontTransition(reference['lengths'], reference['window'], np.degrees(self.measured),
                    options=ArcHelixOptions(radius_m=self.gait.o.r_h1, pitch_m=self.gait.o.pitch, prepare_front=True),
                    minimum_front_turns=self.options.min_front_turns)
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

    def _start_front_finish(self):
        """尾端失能，将原承载弧接入完整的前部螺旋。"""
        count = max(self.front.window[1], self.front.support_joint_count)
        phase_count = max(2, self.front.window[0]-1)
        self.phase = self.front.head_phase
        self.curve = MeasuredGravityShift(self.gait, self.gait.o.r_h1, self.phase, self.measured, phase_count,
                                         upstream_m=(phase_count+1)*self.l)
        self.front_count = count
        self._enabled = np.arange(self.n) < count
        self.offset = 0.
        self.tracking_since = None
        self._enter('front_finish')

    def _advance_front_finish(self, dt, fresh):
        observed_error = np.max(abs(self._targets[self._enabled]-self.measured[self._enabled]))
        pace = float(np.clip((math.radians(23.)-observed_error)/math.radians(17.), .10, 1.))
        self.offset = min(self.curve.shift_limit_m, self.offset+self.options.shift_rate_m_s*pace*dt)
        goal = self.curve.targets(self.offset, self.curve.torsion*self.offset)
        self._set_targets(goal, dt)
        if fresh:
            nominal = self.curve.helical_targets(self.offset, self.curve.torsion*self.offset)
            follows = np.max(abs(nominal[:self.front_count]-self.measured[:self.front_count])) <= math.radians(3.)
            self.tracking_since = (self.elapsed if self.tracking_since is None else self.tracking_since) if follows else None
            if self.tracking_since is not None and self.elapsed-self.tracking_since >= .4:
                self.seating_phase = self.phase
                self.tracking_since = None
                self._enter('front_seat')

    def _advance_front_seat(self, dt, fresh):
        """只滚动前部一小段使其就位；后部仍失能，然后停下确认支撑。"""
        travel = min(self.options.front_seat_rad, self.options.alignment_rate_rad_s*self.elapsed)
        self.phase = self.seating_phase+travel
        self._set_targets(self._helix(self.phase, self.gait.o.r_h1), dt)
        if fresh and travel >= self.options.front_seat_rad:
            follows = np.max(abs(self._targets[self._enabled]-self.measured[self._enabled])) <= math.radians(5.)
            self.tracking_since = (self.elapsed if self.tracking_since is None else self.tracking_since) if follows else None
            if self.tracking_since is not None and self.elapsed-self.tracking_since >= .4:
                self._enter('front_hold')
                self.pause()

    def _start_tail_intake(self):
        self.accepted = self.initial_prefix = self.front_count
        # 用已经下发并搭稳的前部相位，不再用受载角偏差重新拟合它。
        self.curve = MeasuredGravityShift(self.gait, self.gait.o.r_h1, self.phase, self.measured, self.front_count,
                                         upstream_m=(self.front_count+1)*self.l)
        self.locked_front = self._targets[:self.front_count].copy()
        self.offset = 0.
        self.tracking_since = None
        self._enter('first_shift')

    def _advance_tail_intake(self, dt, fresh):
        """相位补偿保持前部目标，只把后段实测曲线逐节接入螺旋。"""
        count = self.accepted
        if count < self.n and not self._enabled[count]:
            self._enabled[count] = True
            self._targets[count] = self.measured[count]
        stop = min(self.n, count+1)
        error = np.max(abs(self._targets[self._enabled]-self.measured[self._enabled]), initial=0.)
        pace = float(np.clip((math.radians(23.)-error)/math.radians(17.), .1, 1.))
        self.offset = min(self.curve.shift_limit_m, self.offset+self.options.shift_rate_m_s*pace*dt)
        goal = self.curve.targets(self.offset, self.curve.torsion*self.offset)
        goal[:self.front_count] = self.locked_front
        self._set_targets(goal, dt)
        # 送尾期间前部已发目标逐帧保持完全相同。
        self._targets[:self.front_count] = self.locked_front
        if not fresh:
            return
        nominal = self.curve.helical_targets(self.offset, self.curve.torsion*self.offset)
        # count 是已接受的前缀；当前新关节是相邻转轴的前瞻接管。
        # 先等当前关节完全到位才开放下一转轴，会把3D曲线退化成单轴掰尾。
        if count == self.front_count:
            checking = slice(0, count)
        elif count < self.n:
            checking = slice(max(self.front_count, count-2), count)
        else:
            checking = slice(None)
        follows = np.max(abs(nominal[checking]-self.measured[checking])) <= math.radians(3.)
        self.tracking_since = (self.elapsed if self.tracking_since is None else self.tracking_since) if follows else None
        if self.tracking_since is not None and self.elapsed-self.tracking_since >= .4:
            self.tracking_since = None
            if count < self.n:
                self.accepted += 1
                self.events.append(dict(stage='intake', joint=self.accepted, shift_m=self.offset))
            else:
                self._enter('complete' if self.helix_only else 'align')
                self.pause()

    def _helix(self, phase, radius):
        return self.gait.targets(phase, radius)[0]

    def _fit_phase(self, count, radius):
        basis = np.column_stack([self._helix(p, radius)[:count] for p in (0., math.pi/2)])
        weights = np.linalg.lstsq(basis, self.measured[:count], rcond=None)[0]
        return math.atan2(weights[1], weights[0])


    def confirm(self, feedback_time, measured_deg):
        """Ctrl+L 标记操作者看到的几何事件，之后仍保持到 J/G。"""
        if self.running or self.step_left > 0.:
            raise ValueError('请先暂停，再按 Ctrl+L 确认实际姿态')
        self.measured = np.radians(measured_deg)
        self.feedback_time = feedback_time
        self.started = True
        if self.stage == 'front_hold':
            self._start_tail_intake()
        elif self.stage == 'align':
            self.anchor = self._targets.copy()
            self._enter('unwind')
        elif self.stage == 'right_u_hold' or (self.stage == 'unwind' and self.elapsed >= 3.):
            self.anchor = self._targets.copy()
            self._enter('drop')
        elif self.stage == 'drop_hold':
            self._start_source_return()
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

        if self.stage == 'arc_settle':
            self._advance_arc_reference(dt, fresh)
        elif self.stage == 'front':
            self.front.running = True
            self.front.advance(dt, now, feedback_time, measured_deg)
            self._enabled, self._targets = self.front.enabled, self.front.targets
            self.phase = self.front.phase+self.front.roll if self.front.stage == 'lead' else self.front.phase
            if self.front.stage == 'transfer_done':
                self.events.append(dict(stage='auto_adjust_done', measured_window=list(self.front.window),
                    material_shift_joints=self.front.transfer.position,
                    requested_start=self.front.desired_start))
                lengths = self.arc_reference['lengths']
                self.front = GravityFrontTransition(lengths, self.front.window, np.degrees(self.measured),
                    options=ArcHelixOptions(radius_m=self.gait.o.r_h1, pitch_m=self.gait.o.pitch, prepare_front=True),
                    minimum_front_turns=self.options.min_front_turns)
                self._plan_arc_reference(lengths)
            elif self.front.stage == 'blocked':
                self.pause()
                self._enter('blocked')
            elif (not self.front_extended and self.front.stage == 'lead' and
                  self.front.leading_extension >= self.front.curve.head_length):
                self.front_tracking_error_deg = float(np.degrees(np.max(
                    abs(self._targets[self._enabled]-self.measured[self._enabled]))))
                if fresh:
                    settled = self.front_tracking_error_deg <= 10.
                    self.front_ready_since = (self.elapsed if self.front_ready_since is None else
                                              self.front_ready_since) if settled else None
                    if self.front_ready_since is not None and self.elapsed-self.front_ready_since >= .4:
                        self.front_extended = True
                        self._start_front_finish()
        elif self.stage == 'front_finish':
            self._advance_front_finish(dt, fresh)
        elif self.stage == 'front_seat':
            self._advance_front_seat(dt, fresh)
        elif self.stage == 'front_hold':
            self.phase += self.options.alignment_rate_rad_s*dt
            self._set_targets(self._helix(self.phase, self.gait.o.r_h1), dt)
        elif self.stage == 'first_shift':
            self._advance_tail_intake(dt, fresh)
        elif self.stage in ('return_seat', 'return_wind', 'return_finish'):
            self._advance_source_return(dt)
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
        elif self.stage == 'recover':
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

    def _start_source_return(self):
        """原实机版2：逐个接管直尾，再缩短直线段，补偿前部螺旋相位。"""
        self.return_straight = (self.n-self.options.retain_joints+1)*self.l
        self.return_lead = self.gait.lead_angle(self.gait.o.r_h2)
        self.return_seating = self.gait.targets(self.phase, self.gait.o.r_h2,
            ls=self.return_straight, th2=self.return_lead)[0]
        self.return_fixed = self._targets[:self.options.retain_joints-4].copy()
        self.offset = 0.
        self._enter('return_seat')

    def _advance_source_return(self, dt):
        keep = self.options.retain_joints-4
        gait, radius = self.gait, self.gait.o.r_h2
        if self.stage == 'return_seat':
            end = min(self.n, keep+int(self.elapsed)+1)
            entering = ~self._enabled[:end]
            self._targets[:end][entering] = self.measured[:end][entering]
            self._enabled[:end] = True
            goal = self._targets.copy()
            goal[keep:end] = self.return_seating[keep:end]
            goal[:keep] = self.return_fixed
            self._set_targets(goal, dt)
            if self.elapsed >= self.n-keep:
                self._enter('return_wind')
        elif self.stage == 'return_wind':
            self.offset = min(self.return_straight, self.options.shift_rate_m_s*self.elapsed)
            straight = self.return_straight-self.offset
            # 初次ARC接尾时前部锁定；重力矩回卷时前部随曲线继续滚动推进。
            self.phase += self.options.return_roll_rad_s*dt
            self._set_targets(gait.targets(self.phase, radius, ls=straight, th2=self.return_lead)[0], dt)
            if self.elapsed >= self.return_straight/self.options.shift_rate_m_s+2.:
                self._enter('return_finish')
        else:
            lead = self.return_lead*(1.-min(1., self.elapsed/3.))
            self._set_targets(gait.targets(self.phase, radius, th2=lead)[0], dt)
            if self.elapsed >= 3.:
                self.recover_phase, self.anchor = self.phase, self._targets.copy()
                self._enter('recover')


    def _advance_recover(self, dt):
        t, gait = self.elapsed, self.gait
        self.phase += gait.o.roll_rate*dt
        radius = gait.o.r_h2+(gait.o.r_h1-gait.o.r_h2)*smooth(t/3.)
        residual = self.anchor-self._helix(self.recover_phase, gait.o.r_h2)
        self._set_targets(self._helix(self.phase, radius)+(1.-smooth(t/2.))*residual, dt, rate=45.)
        if t >= 5.:
            self._enter('roll')

    def status(self):
        front = None if self.front is None else self.front.status()
        return dict(stage=self.stage, elapsed_s=self.elapsed, phase_rad=math.remainder(self.phase, math.tau),
                    planned_window=None if self.front is None else
                        [self.front.desired_start, self.front.desired_start+self.front.window[1]-self.front.window[0]],
                    front_count=getattr(self, 'front_count', None),
                    helix_only=self.helix_only, front_tracking_error_deg=self.front_tracking_error_deg,
                    accepted_joint=self.accepted, shift_m=self.offset, step_remaining_s=self.step_left,
                    front=front, events=self.events.copy())
