"""Optional effort feedback around the encoder-derived sectional curvature."""
import math

import numpy as np


class SectionEffort:
    """Consume normalized motor effort in [0, 1] for currently enabled joints."""

    def __init__(self, mode):
        self.mode = mode
        self.time = None
        self.global_error = 0.
        self.local_error = None
        self.reference_effort = None
        self.reference_sum = 0.
        self.reference_time = 0.
        self.mean_effort = None

    def observe(self, sample, time, enabled, reference, measured, period):
        if time is None or (self.time is not None and time <= self.time):
            return
        dt = .2 if self.time is None else time - self.time
        self.time = time
        effort = None if sample is None else np.asarray(sample, dtype=float)
        if (effort is None or effort.shape != enabled.shape or not enabled.any() or
                not np.isfinite(effort[enabled]).all()):
            self.global_error = 0.
            self.local_error = None
            self.mean_effort = None
            return
        effort = np.clip(effort, 0., 1.)
        mean = float(effort[enabled].mean())
        self.mean_effort = mean
        if self.reference_effort is None:
            duration = min(dt, period - self.reference_time)
            self.reference_sum += mean * duration
            self.reference_time += duration
            if self.reference_time >= period:
                self.reference_effort = self.reference_sum / self.reference_time
        error = 0. if self.reference_effort is None else mean - self.reference_effort
        # Only yield locally when the position error calls for more bending
        # in the commanded direction. A holding or opening torque is ambiguous.
        bending = np.clip(abs(reference) / math.radians(25.), 0., 1.)
        closing = (reference - measured) * reference > 0.
        local = effort - mean
        local = np.where(local > 0., local * closing, local) * bending
        alpha = 1. - math.exp(-dt / .6)
        self.global_error += alpha * (error - self.global_error)
        previous = np.zeros_like(local) if self.local_error is None else self.local_error
        self.local_error = np.where(enabled, previous + alpha * (local - previous), 0.)

    def scale(self, bounds, field, joints, enabled, taper, observed_wrap):
        if self.local_error is None:
            return field
        weights = np.diff(bounds) * np.linalg.norm(field, axis=1) * taper
        if not np.any(weights):
            return field
        common = 0.
        if self.mode in ('global', 'fused'):
            error = self.global_error
            common = -.02 * np.sign(error) * np.clip((abs(error) - .04) / .10, 0., 1.)
            if common < 0. and (observed_wrap is None or observed_wrap < 325.):
                common = 0.
        change = np.full(len(field), common)
        if self.mode in ('local', 'fused'):
            mid = .5 * (bounds[:-1] + bounds[1:])
            local = np.interp(mid, joints[enabled], self.local_error[enabled])
            local = -np.sign(local) * np.clip((abs(local) - .06) / .30, 0., 1.)
            # Redistribute turn without changing its curvature-weighted integral.
            local -= weights @ local / weights.sum()
            change += .06 * local / max(1., float(max(abs(local))))
        return field * (1. + change * taper)[:, None]
