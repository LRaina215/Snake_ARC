"""Hardware-facing command clock for the vertical Helix to full-ring ARC.

The existing motor loop owns serial I/O, torque order and emergency release.
This module consumes complete angle reads in controller degrees and emits
target degrees and a boolean torque mask. Timestamps share a monotonic clock.
"""
import math
from dataclasses import replace

import numpy as np

from .acute_arc_transition import AcuteArcTransition, AcuteOptions


def vertical_short_options():
    """Shared profile verified on the 18-joint, 110 mm short-elbow scene."""
    return AcuteOptions(reference_bend_radius_m=.13, ring_turns=1.12, ring_radius_delta_m=-.003,
                        section_shell_diameter_m=.058, section_keep_curvature=True,
                        progressive_tail_release=True, section_before_bend=True,
                        outlet_end_correction=True, outlet_use_spare_length=True)


class VerticalRingManual:
    def __init__(self, measured_deg, lengths, options=AcuteOptions(), *, helix_targets_deg=None,
                 step_s=1., max_feedback_age_s=.8):
        measured = np.asarray(measured_deg, dtype=float)
        reference = measured if helix_targets_deg is None else np.asarray(helix_targets_deg, dtype=float)
        if measured.ndim != 1 or reference.shape != measured.shape or not np.isfinite(measured).all():
            raise ValueError('Expected one complete Helix encoder sample and matching targets')
        self.core = AcuteArcTransition(np.radians(reference), lengths, replace(options, initial_hold_s=0.))
        self.core.measured = np.radians(measured)
        self.core.branch('arc')
        self.step_s = step_s
        self.max_feedback_age_s = max_feedback_age_s
        self.running = False
        self.step_remaining_s = 0.
        self.feedback_wait = False
        self.latest_sample_s = None

    @property
    def targets(self):
        return np.degrees(self.core.targets)

    @property
    def enabled(self):
        return self.core.enabled.copy()

    @property
    def stage(self):
        return self.core.stage

    def pause(self):
        self.running = False
        self.step_remaining_s = 0.

    def toggle_continuous(self):
        if self.running:
            self.pause()
        else:
            self.step_remaining_s = 0.
            self.running = True

    def step_once(self):
        self.running = False
        self.step_remaining_s = self.step_s

    def begin_bend(self, nominal_head_u=-.055):
        if self.core.stage != 'vertical_ring':
            return False
        self.core.begin_bend(nominal_head_u)
        return True

    def confirm_outlet(self):
        return self.core.confirm_outlet()

    def advance(self, dt, now, feedback_time, measured_deg):
        if not math.isfinite(dt) or dt <= 0.:
            raise ValueError('Expected a positive control step')
        if not self.running and self.step_remaining_s <= 0.:
            return False
        if (feedback_time is None or not math.isfinite(feedback_time) or not math.isfinite(now) or
                not 0. <= now - feedback_time <= self.max_feedback_age_s or
                (self.latest_sample_s is not None and feedback_time < self.latest_sample_s)):
            self.feedback_wait = True
            return False
        measured = np.asarray(measured_deg, dtype=float)
        if measured.shape != self.core.targets.shape or not np.isfinite(measured).all():
            self.feedback_wait = True
            return False
        self.feedback_wait = False
        self.latest_sample_s = feedback_time
        used = dt if self.running else min(dt, self.step_remaining_s)
        self.core.step(used, np.radians(measured), feedback_time)
        self.step_remaining_s = max(0., self.step_remaining_s - used)
        if self.step_remaining_s < 1e-9:
            self.step_remaining_s = 0.
        return True

    def status(self):
        return dict(self.core.status(), running=self.running, step_remaining_s=self.step_remaining_s,
                    feedback_wait=self.feedback_wait, enabled_joints=(np.flatnonzero(self.enabled) + 1).tolist())
