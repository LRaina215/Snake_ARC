"""Stitched curves for the vertical acute-elbow comparison.

The bend has a finite *reference* radius for frame transport. It does not
describe the collision mesh. A bend-entry command supplies the nominal origin.
"""
from dataclasses import asdict, dataclass
import math
from typing import Optional

import numpy as np

from .bent_helix_curve import BentHelixCurve
from .curve_gait import arc, helix, helix_turn_length, line
from .helix_arc_transition import HelixArcOptions, HelixArcTransition


@dataclass(frozen=True)
class AcuteOptions:
    ring_pitch_m: float = .075
    ring_turns: float = 1.055
    ring_radius_delta_m: float = 0.
    reference_bend_radius_m: float = .155
    roll_rate_rad_s: float = -.5
    feed_m_s: float = .012
    ring_feed_m_s: float = .007
    release_m_s: float = .024
    shoulder_angle_deg: float = 40.
    miter_section: bool = False
    lead_length_m: float = 0.
    corner_pitch_m: Optional[float] = None
    exit_pitch_m: Optional[float] = None
    corner_turns: Optional[float] = None
    section_compression: float = 0.


def elliptic_field(radius, major_scale, pitch, turns, azimuth):
    """Curvature in the transported frame of an elliptic, shallow-pitch turn."""
    a, b = radius * major_scale, radius
    c = pitch / (2. * math.pi)
    theta = np.linspace(azimuth, azimuth + 2. * math.pi * turns, 481)
    sine, cosine = np.sin(theta), np.cos(theta)
    speed = np.sqrt((a * sine)**2 + (b * cosine)**2 + c*c)
    cross_sq = (b*c*sine)**2 + (a*c*cosine)**2 + (a*b)**2
    curvature = np.sqrt(cross_sq) / speed**3
    torsion = a*b*c / cross_sq
    ds = (speed[:-1] + speed[1:]) * np.diff(theta) * .5
    bounds = np.r_[0., np.cumsum(ds)]
    psi = np.r_[0., np.cumsum((torsion[:-1] + torsion[1:]) * ds * .5)]
    field = curvature[:, None] * np.column_stack([np.cos(psi), np.sin(psi)])
    return bounds, (field[:-1] + field[1:]) * .5, psi[-1]


class AcuteArcTransition(HelixArcTransition):
    def __init__(self, measured, lengths, options=AcuteOptions()):
        super().__init__(measured, lengths, HelixArcOptions(target_rate_deg_s=45.))
        self.acute = options
        self.stage = 'common_climb'
        self.mode = 'common'
        self.nominal_head_u = None
        self.bent = None
        self.field = None
        self.next_field_s = 0.
        self.wrap_radius = self.radius
        self.pitch = self.fit['pitch']
        self.shoulder = 0.
        self.target_angle_max = 0.
        self.major_scale = 1.
        self.wrap_start = 0.
        self.compression = 0.

    def branch(self, mode):
        self.mode = mode
        self.enter('form_ring' if mode == 'arc' else 'vertical_helix')

    def begin_bend(self, nominal_head_u):
        self.nominal_head_u = nominal_head_u
        if self.mode != 'arc' or not (self.acute.miter_section or self.acute.lead_length_m):
            self.bent = BentHelixCurve(self.wrap_radius, self.pitch, self.acute.reference_bend_radius_m)
        self.enter('bend_shift')

    def form_ring(self, dt):
        opt = self.acute
        # First transfer material from the long helix to the outgoing shoulder
        # and tail. Change pitch while the wrap still spans more than one turn.
        w = min(self.elapsed / 8., 1.)
        w = w * w * (3. - 2. * w)
        self.wrap_radius = self.radius + w * opt.ring_radius_delta_m
        self.pitch = self.fit['pitch'] + w * (opt.ring_pitch_m - self.fit['pitch'])
        self.shoulder = w * math.radians(opt.shoulder_angle_deg) * self.wrap_radius
        retained = opt.ring_turns * helix_turn_length(self.wrap_radius, self.pitch)
        self.helix_length = max(retained, min(self.L - self.shoulder, self.helix_length - opt.release_m_s * dt))
        if w == 1. and self.helix_length <= retained + 1e-9:
            self.enter('vertical_ring')

    def straight_reference(self):
        lead_plane = -math.pi / 2.
        segments = [arc(self.wrap_start, 1. / self.wrap_radius, lead_plane),
                    helix(self.helix_length, self.wrap_radius, self.pitch, psi=-lead_plane),
                    arc(self.shoulder, 1. / self.wrap_radius, math.pi / 2.),
                    line(max(0., self.L - self.wrap_start - self.helix_length - self.shoulder))]
        return self.curve.targets(segments, self.phase - segments[1].tau * self.wrap_start)

    def update_corner_section(self):
        opt = self.acute
        u = self.nominal_head_u
        exit_u = opt.reference_bend_radius_m * math.pi / 2.

        def ramp(start, stop):
            w = float(np.clip((u - start) / (stop - start), 0., 1.))
            return w*w*(3. - 2.*w)

        enter = ramp(-.045, .045)
        leave = ramp(exit_u - .015, exit_u + .025)
        turns = opt.ring_turns
        if opt.corner_turns is not None:
            turns += (enter - leave) * (opt.corner_turns - opt.ring_turns)
        if opt.corner_pitch_m is not None:
            self.pitch = opt.ring_pitch_m + enter * (opt.corner_pitch_m - opt.ring_pitch_m)
            if opt.exit_pitch_m is not None:
                self.pitch += leave * (opt.exit_pitch_m - opt.corner_pitch_m)
        self.bent.pitch = self.pitch
        self.helix_length = turns * helix_turn_length(self.wrap_radius, self.pitch)

    def bent_reference(self, dt):
        rate = self.acute.ring_feed_m_s if self.mode == 'arc' else self.acute.feed_m_s
        self.nominal_head_u = min(self.nominal_head_u + rate * dt,
                                  self.acute.reference_bend_radius_m * math.pi / 2. + .65)
        if self.time >= self.next_field_s - 1e-9:
            if self.mode == 'arc' and (self.acute.corner_pitch_m is not None or self.acute.corner_turns is not None):
                self.update_corner_section()
            if self.mode == 'arc' and self.acute.section_compression:
                exit_u = self.acute.reference_bend_radius_m * math.pi / 2.
                enter = np.clip((self.nominal_head_u + .015) / .07, 0., 1.)
                leave = np.clip((self.nominal_head_u - exit_u + .015) / .04, 0., 1.)
                self.compression = self.acute.section_compression * (enter*enter*(3.-2.*enter) - leave*leave*(3.-2.*leave))
            bounds, field, end_phase = self.bent.field(self.nominal_head_u, self.helix_length,
                                                       section_compression=self.compression)
            if self.mode == 'helix':
                self.field = bounds, field
            else:
                shoulder_phase = end_phase + math.pi / 2.
                shoulder = np.array([math.cos(shoulder_phase), math.sin(shoulder_phase)]) / self.wrap_radius
                self.field = (np.r_[bounds, self.helix_length + self.shoulder, self.L],
                              np.vstack([field, shoulder, [0., 0.]]))
            self.next_field_s = self.time + .1
        return self.curve.sampled_targets(*self.field, self.phase)

    def miter_reference(self):
        # At 45 degrees a cylinder cuts an ellipse with major radius sqrt(2) r.
        # Widen only during the corner transfer. The added section length is
        # taken from the free tail, then returned to it on the horizontal leg.
        ramp = np.clip((self.elapsed - 10.) / 14., 0., 1.)
        retreat = np.clip((self.elapsed - 46.) / 14., 0., 1.)
        angle = math.pi / 4. * (ramp - retreat)
        self.major_scale = 1. / math.cos(angle)
        bounds, field, psi = elliptic_field(self.wrap_radius, self.major_scale, self.pitch,
                                             self.acute.ring_turns, -2.1)
        self.helix_length = bounds[-1]
        shoulder_phase = psi + math.pi / 2.
        shoulder = np.array([math.cos(shoulder_phase), math.sin(shoulder_phase)]) / self.wrap_radius
        bounds = np.r_[bounds, self.helix_length + self.shoulder, self.L]
        return self.curve.sampled_targets(bounds, np.vstack([field, shoulder, [0., 0.]]), self.phase)

    def lead_reference(self):
        # Shift the retained ring toward the tail to allocate material to a
        # short, out-of-plane lead. The lead seats toward the outgoing leg
        # while the ring supplies the main support.
        grow = np.clip((self.elapsed - 10.) / 7., 0., 1.)
        shrink = np.clip((self.elapsed - 42.) / 9., 0., 1.)
        self.wrap_start = self.acute.lead_length_m * (grow - shrink)
        return self.straight_reference()

    def step(self, dt, measured):
        self.measured = measured
        self.time += dt
        self.elapsed += dt
        if self.time > 2.:
            self.phase += self.acute.roll_rate_rad_s * dt
        if self.stage == 'form_ring':
            self.form_ring(dt)
        if self.bent is not None:
            desired = self.bent_reference(dt)
        elif self.stage == 'bend_shift' and self.acute.lead_length_m:
            desired = self.lead_reference()
        elif self.stage == 'bend_shift':
            desired = self.miter_reference()
        else:
            desired = self.straight_reference()
        mask = (self.curve.s[:-2] < self.wrap_start + self.helix_length + self.shoulder - 1e-8
                if self.mode == 'arc' else np.ones(self.n, dtype=bool))
        self.target_angle_max = float(np.degrees(np.max(abs(desired))))
        self.apply_reference(desired, dt, mask)
        return self.targets.copy(), self.enabled.copy()

    def status(self):
        return dict(stage=self.stage, phase_rad=self.phase, nominal_head_u_m=self.nominal_head_u,
                    wrap_start_m=self.wrap_start, wrap_length_m=self.helix_length, shoulder_m=self.shoulder,
                    free_tail_m=max(0., self.L - self.wrap_start - self.helix_length - self.shoulder),
                    pitch_m=self.pitch, radius_m=self.wrap_radius, section_major_scale=self.major_scale,
                    section_compression=self.compression,
                    requested_max_deg=self.target_angle_max,
                    curvature_frame_rad=None if self.bent is None else self.bent.gamma,
                    options=asdict(self.acute))
