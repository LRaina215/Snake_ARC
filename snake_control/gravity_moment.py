"""Gravity-assisted helical locomotion for crossing a branch on a horizontal pipe (paper Sec. II-A).

Curve (Table I), head -> tail: helix (seg 4) -> arc theta2 (seg 3, twist +pi/2) -> arc theta1
(seg 2, twist -pi) -> straight l_s (seg 1). The curve starts at the head, so changing the tail
segment lengths leaves the helix part in place (the paper's roll compensation, eqs. (3) and (7)).

Inputs at run time: elapsed time, mean |joint torque| of the enabled joints (stands in for motor
current) and measured joint angles. No pipe/branch geometry is used by the controller.
"""
from dataclasses import dataclass
import math
import numpy as np
from .curve_gait import CurveGait, Segment, helix


@dataclass(frozen=True)
class GravityMomentOptions:
    r_h1: float = .092
    r_h2: float = .096
    pitch: float = .13
    r_c: float = .09
    handed: int = -1                 # torsion sign; with direction=+1 this is a right-handed helix
    roll_rate: float = .6            # rad/s on psi(0); sign chosen so the robot advances toward +x
    t1: float = 1.0                  # contact thresholds on mean |torque| (N*m), set by calibration
    t2: float = 1.125
    settle_s: float = 2.
    detect_after_s: float = 1.5      # ignore the detector right after (re)entering phase 1
    arc_s: float = 3.                # phase 2: theta1 0 -> pi
    line_s: float = 4.               # phase 2: l_s 0 -> one helix turn - pi*r_c
    loosen_s: float = 2.             # phase 2: r_h1 -> r_h2
    drop_s: float = 3.               # phase 2: tail disabled, hangs under gravity
    extend_s: float = 2.             # phase 3: theta2 = alpha_h, l_s = (n_ext+1) l
    extend_hold_s: float = 1.
    rewind_s: float = 6.             # phase 4: l_s -> keep_links*l, theta2 -> 0, r_h2 -> r_h1, rolling
    keep_links: float = 2.
    straighten_s: float = 2.         # back in phase 1: l_s -> 0
    blend_s: float = 1.5             # re-enabled joints blend from measured angles
    crossing_roll: bool = True       # False: no rolling during phases 2-4 (user 11-06: advance comes from
                                     # re-winding against the branch, not from rolling)
    direct_cycle: bool = False       # True: after settle and after each rewind go straight to phase 2
    cycle_hold_s: float = 1.
    rewind_loose: bool = False       # keep r_h2 while re-forming the helix; tighten to r_h1 in cycle_hold
    line_rate: float = .1            # m/s, l_s growth when the operator stops the unwinding by eye
    shift_fit: bool = False          # phase 4 by shift control: the helix phase stays fixed at the tail
                                     # junction (the front is an extended helix), so as the tail re-forms
                                     # the front helix moves along the curve. Phases 2-3 stay compensated
                                     # (eqs. (3), (7)); phase 4 has no compensation in the paper.
    shift_rate: float = 0.           # m/s > 0: phase 4 slides the body along ONE fixed curve (helix, arc
                                     # theta2 = alpha_h kept, line) at this rate, radius kept at r_h2;
                                     # implies shift_fit. 0: old timed rewind (arc shrinks, rewind_s).
    max_target_deg: float = 88.


def smooth(x):
    x = min(max(x, 0.), 1.)
    return x * x * (3 - 2 * x)


class GravityMomentGait:
    def __init__(self, joints=18, options=GravityMomentOptions(), link_length=.0645):
        self.o = options
        self.gait = CurveGait(joints, link_length)
        self.l = link_length
        self.L = (joints + 1) * link_length

    def turn_length(self, r):
        return math.hypot(2 * math.pi * r, self.o.pitch)

    def lead_angle(self, r):
        return math.atan(self.o.pitch / (2 * math.pi * r))

    def n_ext(self, r):
        return int(self.turn_length(r) // self.l)

    def segments(self, r_h, ls=0., th1=0., th2=0.):
        """Return (segments, helix_length). Seg 3's +pi/2 twist also applies when theta2 = 0."""
        o = self.o
        a2, a3 = o.r_c * th1, o.r_c * th2
        lh = self.L - ls - a2 - a3
        if lh <= 0:
            raise ValueError('tail segments longer than the body')
        h = helix(1., r_h, o.pitch, o.handed)
        segs = [Segment(lh, h.kappa, h.tau)]
        carry = math.pi / 2
        if a3 > 0:
            segs.append(Segment(a3, 1 / o.r_c, 0., carry))
            carry = 0.
        if a2 > 0:
            segs.append(Segment(a2, 1 / o.r_c, 0., carry - math.pi))
        if ls > 0:
            segs.append(Segment(ls))
        return segs, lh

    def targets(self, roll, r_h, ls=0., th1=0., th2=0.):
        segs, lh = self.segments(r_h, ls, th1, th2)
        q = self.gait.targets(segs, roll)
        lim = math.radians(self.o.max_target_deg)
        return np.clip(q, -lim, lim), lh

    def helix_joints(self, lh):
        """Joints whose two neighbouring links lie entirely on the helix part."""
        j = np.arange(1, self.gait.joints + 1)
        return (j + 1) * self.l <= lh + 1e-9


class ContactDetector:
    """Paper: >8 of the last 15 samples above t1, or >5 of the last 10 above t2 (100 ms samples)."""

    def __init__(self, t1, t2):
        self.t1, self.t2 = t1, t2
        self.samples = []

    def reset(self):
        self.samples = []

    def add(self, value):
        self.samples.append(value)
        self.samples = self.samples[-15:]
        last15, last10 = self.samples[-15:], self.samples[-10:]
        return (len(last15) == 15 and sum(v > self.t1 for v in last15) > 8) or \
               (len(last10) == 10 and sum(v > self.t2 for v in last10) > 5)


class GravityMomentController:
    """Phase machine: approach -> arc -> line -> loosen -> drop -> extend -> extend_hold -> rewind
    -> straighten -> approach ... Outputs joint targets (rad) and an enable mask."""

    def __init__(self, joints=18, options=GravityMomentOptions(), detect=True, manual_triggers=()):
        self.o = options
        self.g = GravityMomentGait(joints, options)
        self.n = joints
        self.detect = detect
        # operator triggers (absolute times): enter phase 2 at the first approach sample after each time
        self.manual_triggers = sorted(manual_triggers)
        self.detector = ContactDetector(options.t1, options.t2)
        self.roll = 0.
        self.stage, self.t0 = 'settle', 0.
        self.last_t = 0.
        self.next_sample = 0.
        self.cycles = 0
        self.events = []
        self.prev_mask = np.ones(joints, bool)
        self.stopped = False             # operator stop: stay wrapped after the current rewind
        self.unwind_extra = 0.           # operator choice: extra unwound length (m) so the small U starts on top
        self.unwind_hook = None          # operator callback(t) -> unwind_extra, called when phase 2 starts
        self.line_stop = None            # operator callback(t, helix_length) -> True to stop growing l_s
        self.release_hook = None         # operator callback(t, enabled joint count) -> joints kept enabled at drop
        self.release_n = None
        self.ls_now = self.ls_hold = None
        self.lh_fit = None               # helix length at the previous phase-4 step (shift_fit)
        self.entered, self.q_enter = {}, {}

    def _go(self, stage, t):
        if stage == 'arc' and self.unwind_hook is not None:
            self.unwind_extra = float(self.unwind_hook(t))
        self.events.append(dict(t=round(t, 3), stage=stage, unwind_extra_m=round(self.unwind_extra, 4)))
        self.stage, self.t0 = stage, t
        self.lh_fit = None
        if stage == 'approach':
            self.detector.reset()

    def _u(self, t, duration):
        return smooth((t - self.t0) / duration)

    def update(self, t, q_measured, mean_abs_torque):
        o, g = self.o, self.g
        dt, self.last_t = t - self.last_t, t
        r1, r2 = o.r_h1, o.r_h2
        turn1 = g.turn_length(r1)
        ls_unwind = turn1 - math.pi * o.r_c + self.unwind_extra
        ls_ext = (g.n_ext(r2) + 1) * g.l
        ah = g.lead_angle(r2)
        elapsed = t - self.t0
        mask = np.ones(self.n, bool)
        st = self.stage
        if st == 'settle':
            q, _ = g.targets(self.roll, r1)
            if elapsed >= o.settle_s:
                if o.direct_cycle:
                    self.cycles += 1
                    self._go('arc', t)
                else:
                    self._go('approach', t)
        elif st == 'cycle_hold':
            r_ch = r2 + (r1 - r2) * self._u(t, o.cycle_hold_s) if o.rewind_loose else r1
            q, _ = g.targets(self.roll, r_ch)
            if elapsed >= o.cycle_hold_s and not self.stopped:
                self.cycles += 1
                self._go('arc', t)
        elif st == 'approach':
            self.roll += o.roll_rate * dt
            q, _ = g.targets(self.roll, r1)
            if t + 1e-9 >= self.next_sample:
                self.next_sample = t + .1
                hit = self.detector.add(mean_abs_torque)
                manual = self.manual_triggers and t >= self.manual_triggers[0]
                if (self.detect and elapsed >= o.detect_after_s and hit) or manual:
                    if manual:
                        self.manual_triggers.pop(0)
                        self.events.append(dict(t=round(t, 3), stage='operator_trigger'))
                    self.cycles += 1
                    self._go('arc', t)
        elif st == 'arc':
            q, _ = g.targets(self.roll, r1, th1=math.pi * self._u(t, o.arc_s))
            if elapsed >= o.arc_s:
                self._go('line', t)
        elif st == 'line':
            if self.line_stop is None:
                self.ls_now = ls_unwind * self._u(t, o.line_s)
                done = elapsed >= o.line_s
            else:
                # operator watches the body and stops unwinding when the U start sits where wanted
                ls_max = g.L - math.pi * o.r_c - 1.3 * turn1
                self.ls_now = min(o.line_rate * elapsed, ls_max)
                lh_now = g.L - math.pi * o.r_c - self.ls_now
                done = self.ls_now >= ls_max or (self.ls_now > .5 * g.l and self.line_stop(t, lh_now))
            q, _ = g.targets(self.roll, r1, ls=self.ls_now, th1=math.pi)
            if done:
                self.ls_hold = self.ls_now
                self._go('loosen', t)
        elif st == 'loosen':
            r = r1 + (r2 - r1) * self._u(t, o.loosen_s)
            ls_u = ls_unwind if self.ls_hold is None else self.ls_hold
            q, _ = g.targets(self.roll, r, ls=ls_u, th1=math.pi)
            if elapsed >= o.loosen_s:
                self._go('drop', t)
        elif st == 'drop':
            q, lh = g.targets(self.roll, r2, ls=ls_unwind if self.ls_hold is None else self.ls_hold, th1=math.pi)
            mask = g.helix_joints(lh)
            if self.release_hook is not None:
                if self.release_n is None:
                    self.release_n = int(self.release_hook(t, int(mask.sum())))
                    self.events.append(dict(t=round(t, 3), stage='operator_release', enabled=self.release_n))
                mask = np.arange(self.n) < self.release_n
            if elapsed >= o.drop_s:
                self.release_n = None
            if elapsed >= o.drop_s:
                self._go('extend', t)
        elif st == 'extend':
            u = self._u(t, o.extend_s)
            q, _ = g.targets(self.roll, r2, ls=ls_ext, th2=ah)
            if elapsed >= o.extend_s:
                self._go('extend_hold', t)
        elif st == 'extend_hold':
            q, _ = g.targets(self.roll, r2, ls=ls_ext, th2=ah)
            if elapsed >= o.extend_hold_s:
                self._go('rewind', t)
        elif st == 'rewind':
            u = self._u(t, o.rewind_s)
            if o.crossing_roll:
                self.roll += o.roll_rate * dt
            if o.shift_rate > 0:
                # whole-body shift on the fixed curve: the arc stays, the line feeds through it into the helix
                ls = max(ls_ext - o.shift_rate * elapsed, o.keep_links * g.l)
                lh = g.L - ls - o.r_c * ah
                if self.lh_fit is not None:
                    self.roll -= helix(1., r2, o.pitch, o.handed).tau * (lh - self.lh_fit)
                self.lh_fit = lh
                q, _ = g.targets(self.roll, r2, ls=ls, th2=ah)
                if ls <= o.keep_links * g.l + 1e-9:
                    self._go('cycle_hold' if o.direct_cycle else 'straighten', t)
                return self._finish(t, q, mask, q_measured)
            ls = ls_ext + (o.keep_links * g.l - ls_ext) * u
            r_rw = r2 if o.rewind_loose else r2 + (r1 - r2) * u
            if o.shift_fit:
                # head-anchored curve: keeping the junction phase fixed means roll -= tau * d(helix length)
                lh = g.L - ls - o.r_c * ah * (1 - u)
                if self.lh_fit is not None:
                    self.roll -= helix(1., r_rw, o.pitch, o.handed).tau * (lh - self.lh_fit)
                self.lh_fit = lh
            q, _ = g.targets(self.roll, r_rw, ls=ls, th2=ah * (1 - u))
            if elapsed >= o.rewind_s:
                self._go('cycle_hold' if o.direct_cycle else 'straighten', t)
        elif st == 'straighten':
            u = self._u(t, o.straighten_s)
            self.roll += o.roll_rate * dt
            q, _ = g.targets(self.roll, r1, ls=o.keep_links * g.l * (1 - u))
            if elapsed >= o.straighten_s:
                self._go('approach', t)
        else:
            raise RuntimeError(st)
        return self._finish(t, q, mask, q_measured)

    def _finish(self, t, q, mask, q_measured):
        o = self.o
        # joints entering the enabled set start from their measured angle
        for j in np.flatnonzero(mask & ~self.prev_mask):
            self.entered[j], self.q_enter[j] = t, q_measured[j]
        for j in list(self.entered):
            if not mask[j]:
                self.entered.pop(j); self.q_enter.pop(j)
                continue
            b = smooth((t - self.entered[j]) / o.blend_s)
            q[j] = (1 - b) * self.q_enter[j] + b * q[j]
            if b >= 1:
                self.entered.pop(j); self.q_enter.pop(j)
        self.prev_mask = mask.copy()
        return np.where(mask, q, 0.), mask
