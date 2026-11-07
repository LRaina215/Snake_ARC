"""Temporarily straighten the non-bearing joints, then explicitly disable them."""
import numpy as np


class PassiveEndReset:
    def __init__(self, measured_deg, bearing_targets_deg, bearing_mask, *, rate_deg_s=60., hold_s=.25, max_s=2.):
        self.bearing = np.asarray(bearing_mask, bool).copy()
        self.targets = np.asarray(measured_deg, float).copy()
        self.targets[self.bearing] = np.asarray(bearing_targets_deg)[self.bearing]
        self.enabled = np.ones(len(self.bearing), bool)
        self.rate, self.hold_s, self.max_s = rate_deg_s, hold_s, max_s
        self.elapsed = self.zero_hold = 0.
        self.done = False
        self.reason = None
        self.actual_max_deg = None

    def advance(self, dt, measured_deg):
        if self.done:
            return
        self.elapsed += dt
        passive = ~self.bearing
        self.targets[passive] += np.clip(-self.targets[passive], -self.rate*dt, self.rate*dt)
        self.actual_max_deg = float(np.max(abs(np.asarray(measured_deg)[passive]), initial=0.))
        target_zero = np.max(abs(self.targets[passive]), initial=0.) < 1e-8
        self.zero_hold = self.zero_hold+dt if target_zero else 0.
        ready = self.zero_hold >= self.hold_s and self.actual_max_deg <= 5.
        if ready or self.elapsed >= self.max_s:
            self.enabled = self.bearing.copy()
            self.done = True
            self.reason = 'zero_then_disabled' if ready else 'timeout_then_disabled'

    def status(self):
        return dict(elapsed_s=self.elapsed, done=self.done, actual_max_deg=self.actual_max_deg,
                    reason=self.reason, temporary_enabled=self.enabled.tolist())
