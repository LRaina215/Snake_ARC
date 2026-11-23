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


def vertical_roll_options():
    """Carrier geometry used in the manual 110→140→110 mm validation."""
    return replace(vertical_short_options(), vertical_use_spare_length=True, section_bearing_advance_m=.014)


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

    def advance(self, dt, now, feedback_time, measured_deg, *, motor_effort=None):
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
        self.core.step(used, np.radians(measured), feedback_time, motor_effort=motor_effort)
        self.step_remaining_s = max(0., self.step_remaining_s - used)
        if self.step_remaining_s < 1e-9:
            self.step_remaining_s = 0.
        return True

    def status(self):
        return dict(self.core.status(), running=self.running, step_remaining_s=self.step_remaining_s,
                    feedback_wait=self.feedback_wait, enabled_joints=(np.flatnonzero(self.enabled) + 1).tolist())


class VerticalRollManual(VerticalRingManual):
    """Select either vertical roll on the same carrier and command clock."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compliant = False
        self.basic_options = self.core.acute
        self.compliant_options = replace(
            self.basic_options,
            section_feedback_weight=.35, vertical_use_spare_length=True, vertical_adaptive_material=True,
            vertical_material_radius_gain=.85, vertical_material_minimum_wrap_deg=350.,
            vertical_material_max_extra_m=.105, vertical_material_handoff_m=.036,
            vertical_end_clearance=True, vertical_material_return_phase_rad=2.*math.pi,
            vertical_return_advance_m=.012,
        )

    def select_roll(self, compliant):
        if self.stage != 'vertical_ring':
            raise ValueError('请在竖直成环并完成退尾后、入弯前使用 Ctrl+1/3')
        if self.compliant == compliant:
            return
        self.compliant = compliant
        if compliant:
            # Retain the current carrier dimensions when returning to ordinary rolling.
            self.basic_options = replace(self.basic_options, vertical_use_spare_length=True)
            self.core.section_feedback = self.core.section_feedback_time = None
            self.core.section_feedback_diagnostic = None
            if self.core.ring_base_radius is None:
                self.core.ring_radius_samples.clear()
                self.core.ring_radius_capture_phase = None
        self.core.acute = self.compliant_options if compliant else self.basic_options

    def status(self):
        status = super().status()
        mode = 'compliant' if self.compliant else 'basic'
        status.update(vertical_roll_mode=mode if self.stage == 'vertical_ring' else None,
                      reference_radius_m=self.core.wrap_radius + self.core.ring_radius_offset)
        return status
