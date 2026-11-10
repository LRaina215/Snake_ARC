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
from .section_geometry import forward
from .section_ring_curve import SectionRingCurve


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
    corner_shape_delay_m: float = 0.
    exit_shape_delay_m: float = 0.
    encoder_spacing: bool = False
    encoder_spacing_start_m: float = .210
    encoder_compaction: bool = False
    initial_hold_s: float = 2.
    encoder_tip_clearance: bool = False
    encoder_feed_sync: bool = False
    section_shell_diameter_m: Optional[float] = None
    section_endpoint_allowance: bool = False
    section_handoff_from_tilt: bool = False
    section_keep_curvature: bool = False
    progressive_tail_release: bool = False
    section_before_bend: bool = False
    section_seam_clearance: bool = False


def observed_ring(measured, lengths, material, start, stop):
    """Shape of the active interval in the head frame, from encoders only."""
    points = forward(measured, lengths)
    s = np.linspace(start, stop, 65)
    ring = np.column_stack([np.interp(s, material, axis) for axis in points.T])
    centered = ring - ring.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ axes[:2].T
    radius_sq = np.sum(xy**2, axis=1)
    center = np.linalg.lstsq(np.c_[2.*xy, np.ones(len(xy))], radius_sq, rcond=None)[0][:2]
    x, y = (xy - center).T
    azimuth = np.unwrap(np.arctan2(y, x))
    tail = points[-1] - points[-5]
    tail_length = np.linalg.norm(tail)
    tail_straightness = tail_length / sum(lengths[-4:])
    tilt = None
    if tail_straightness >= .90:
        tilt = float(np.degrees(np.arccos(np.clip(abs(axes[2] @ tail) / tail_length, 0., 1.))))
    return dict(turn_deg=float(abs(np.degrees(azimuth[-1] - azimuth[0]))),
                width_m=float(np.ptp(centered @ axes[2])),
                end_gap_m=float(np.linalg.norm(ring[-1] - ring[0])),
                tilt_from_tail_deg=tilt)


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
        if options.section_before_bend and (not options.progressive_tail_release
                                            or not options.section_shell_diameter_m):
            raise ValueError('Pre-bend section transfer needs progressive release and shell dimensions')
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
        self.handedness = 1 if self.tau >= 0. else -1
        self.encoder_shape = None
        self.last_feedback_time = None
        self.spacing_trim = 0.
        self.compaction_trim = 0.
        self.tip_turn = 0.
        self.tip_demand = 0.
        self.tip_sample_time = None
        self.tip_gap_m = None
        self.feed_fraction = 1.
        self.section_ring = None
        self.section_weight = 0.
        self.section_handoff_u = None
        self.release_feedback_time = None
        self.release_joint = None
        self.release_count = 0
        self.vertical_section = None

    def branch(self, mode):
        self.mode = mode
        self.enter('form_ring' if mode == 'arc' else 'vertical_helix')

    def begin_bend(self, nominal_head_u):
        if self.mode == 'arc' and self.acute.progressive_tail_release and self.stage != 'vertical_ring':
            raise ValueError('The tail release must finish before the ring enters the bend')
        self.nominal_head_u = nominal_head_u
        if self.mode != 'arc' or not (self.acute.miter_section or self.acute.lead_length_m):
            self.bent = BentHelixCurve(self.wrap_radius, self.pitch, self.acute.reference_bend_radius_m,
                                       handedness=self.handedness)
            if self.mode == 'arc' and self.acute.section_shell_diameter_m is not None and self.section_ring is None:
                self.section_ring = self.make_section()
        self.enter('bend_shift')

    def make_section(self):
        return SectionRingCurve(self.wrap_radius, self.pitch, self.acute.reference_bend_radius_m,
                                self.l, self.acute.section_shell_diameter_m, handedness=self.handedness,
                                endpoint_allowance=self.acute.section_endpoint_allowance,
                                seam_clearance=self.acute.section_seam_clearance)

    def ring_profile(self):
        opt = self.acute
        w = min(self.elapsed / 8., 1.)
        w = w * w * (3. - 2. * w)
        self.wrap_radius = self.radius + w * opt.ring_radius_delta_m
        self.pitch = self.fit['pitch'] + w * (opt.ring_pitch_m - self.fit['pitch'])
        return w

    def form_ring(self, dt):
        opt = self.acute
        # First transfer material from the long helix to the outgoing shoulder
        # and tail. Change pitch while the wrap still spans more than one turn.
        w = self.ring_profile()
        self.shoulder = w * math.radians(opt.shoulder_angle_deg) * self.wrap_radius
        retained = opt.ring_turns * helix_turn_length(self.wrap_radius, self.pitch)
        self.helix_length = max(retained, min(self.L - self.shoulder, self.helix_length - opt.release_m_s * dt))
        if w == 1. and self.helix_length <= retained + 1e-9:
            self.enter('vertical_ring')

    def progressive_ring(self, dt, feedback_time):
        """Establish the front turn, then move the release boundary tail-first."""
        opt = self.acute
        fresh = feedback_time is not None and (
            self.release_feedback_time is None or feedback_time > self.release_feedback_time)
        if fresh:
            self.release_feedback_time = feedback_time
        if self.stage == 'form_ring':
            w = self.ring_profile()
        retained = opt.ring_turns * helix_turn_length(self.wrap_radius, self.pitch)
        if self.stage == 'form_ring':
            if fresh and w == 1.:
                self.encoder_shape = observed_ring(self.measured, self.lengths, self.curve.s, 0., retained)
                if 325. <= self.encoder_shape['turn_deg'] <= 395.:
                    if opt.section_before_bend:
                        self.section_ring = self.make_section()
                        self.vertical_section = self.section_ring.section_field(-.055)[:2]
                    self.enter('release_tail')
            return self.enabled.copy()

        # Use the same curvature integral for the moving boundary. The pending
        # prefix follows its curve until the distal joint's integral vanishes.
        if self.vertical_section is not None:
            retained = self.vertical_section[0][-1]
        self.helix_length = max(retained, self.helix_length - opt.release_m_s * dt)
        mask = self.enabled.copy()
        tail = int(np.flatnonzero(mask)[-1])
        self.release_joint = None
        if fresh and self.curve.s[tail] >= self.helix_length - 1e-8:
            mask[tail] = False
            self.release_joint = tail + 1
            self.release_count += 1
            self.events.append(dict(time_s=self.time, stage=self.stage, released_joint=self.release_joint,
                                    feedback_time_s=feedback_time))
        if self.helix_length <= retained + 1e-9 and np.array_equal(mask, self.ring_mask()):
            self.enter('vertical_ring')
        return mask

    def ring_mask(self):
        # A joint owns the integral of curvature over its two adjacent links.
        start = self.wrap_start
        end = start + self.helix_length + self.shoulder
        return (self.curve.s[2:] > start + 1e-8) & (self.curve.s[:-2] < end - 1e-8)

    def straight_reference(self):
        lead_plane = -self.handedness * math.pi / 2.
        segments = [arc(self.wrap_start, 1. / self.wrap_radius, lead_plane),
                    helix(self.helix_length, self.wrap_radius, self.pitch, handed=self.handedness, psi=-lead_plane),
                    arc(self.shoulder, 1. / self.wrap_radius, -lead_plane),
                    line(max(0., self.L - self.wrap_start - self.helix_length - self.shoulder))]
        return self.curve.targets(segments, self.phase - segments[1].tau * self.wrap_start)

    def vertical_section_reference(self):
        cut, compact = self.vertical_section
        nominal = self.acute.ring_turns * helix_turn_length(self.wrap_radius, self.pitch)
        lead = min(nominal, self.helix_length)
        blend = float(np.clip((nominal + 2.*self.l - self.helix_length) / (2.*self.l), 0., 1.))
        blend = blend*blend*(3. - 2.*blend)
        self.section_weight = blend
        helix_segment = helix(lead, self.wrap_radius, self.pitch, handed=self.handedness)
        bounds = np.linspace(0., lead, 481)
        angle = helix_segment.tau * (bounds[:-1] + bounds[1:]) * .5
        swept = helix_segment.kappa * np.column_stack([np.cos(angle), np.sin(angle)])
        bounds, field = self.blend_fields(bounds, swept, cut, compact, blend)
        if self.helix_length > bounds[-1] + 1e-8:
            rest = np.linspace(bounds[-1], self.helix_length, 81)
            end = math.atan2(field[-1, 1], field[-1, 0])
            angle = end + helix_segment.tau*((rest[1:] + rest[:-1])*.5 - rest[0])
            tail = helix_segment.kappa * np.column_stack([np.cos(angle), np.sin(angle)])
            bounds = np.r_[bounds, rest[1:]]
            field = np.vstack([field, tail])
        return self.curve.sampled_targets(np.r_[bounds, self.L], np.vstack([field, [0., 0.]]), self.phase)

    def update_corner_section(self):
        opt = self.acute
        u = self.nominal_head_u
        exit_u = opt.reference_bend_radius_m * math.pi / 2.

        def ramp(start, stop):
            w = float(np.clip((u - start) / (stop - start), 0., 1.))
            return w*w*(3. - 2.*w)

        enter = ramp(-.045 + opt.corner_shape_delay_m, .045 + opt.corner_shape_delay_m)
        leave = ramp(exit_u - .015 + opt.exit_shape_delay_m, exit_u + .025 + opt.exit_shape_delay_m)
        turns = opt.ring_turns
        if opt.corner_turns is not None:
            turns += (enter - leave) * (opt.corner_turns - opt.ring_turns)
        self.pitch = opt.ring_pitch_m
        if opt.corner_pitch_m is not None:
            self.pitch += enter * (opt.corner_pitch_m - opt.ring_pitch_m)
            if opt.exit_pitch_m is not None:
                self.pitch += leave * (opt.exit_pitch_m - opt.corner_pitch_m)
        if opt.encoder_spacing:
            if opt.exit_pitch_m is not None:
                self.pitch = min(self.pitch + self.spacing_trim * (1. - leave), opt.exit_pitch_m + .004)
            else:
                self.pitch += self.spacing_trim
            self.pitch = float(np.clip(self.pitch, .024, .105))
        self.bent.pitch = self.pitch
        self.helix_length = turns * helix_turn_length(self.wrap_radius, self.pitch)

    def observe_encoders(self, feedback_time):
        previous = self.last_feedback_time
        if feedback_time is None or (previous is not None and feedback_time <= previous):
            return
        self.last_feedback_time = feedback_time
        self.encoder_shape = observed_ring(self.measured, self.lengths, self.curve.s, self.wrap_start,
                                           self.wrap_start + self.helix_length)
        if self.stage != 'bend_shift' or previous is None:
            return
        if self.section_ring is not None and self.acute.section_handoff_from_tilt and self.section_handoff_u is None:
            tilt = self.encoder_shape['tilt_from_tail_deg']
            if tilt is not None and tilt >= 22.5 and not self.enabled[-4:].any():
                self.section_handoff_u = self.nominal_head_u
        width = self.encoder_shape['width_m']
        gap = self.encoder_shape['end_gap_m']
        elapsed = min(feedback_time - previous, .25)
        if self.acute.encoder_feed_sync:
            u = self.nominal_head_u
            deficit = max((width - .100) / .035, (340. - self.encoder_shape['turn_deg']) / 35.)
            desired = 1. - .55 * float(np.clip(deficit, 0., 1.)) if .050 < u < .25 else 1.
            self.feed_fraction += float(np.clip(desired - self.feed_fraction, -2. * elapsed, elapsed))
        if self.acute.encoder_compaction:
            u = self.nominal_head_u
            demand = np.clip(8. * (width - .094), 0., .22) if .04 < u < .22 and gap > .096 else 0.
            self.compaction_trim += float(np.clip(demand - self.compaction_trim, -.6 * elapsed, .6 * elapsed))
        if not self.acute.encoder_spacing or self.nominal_head_u < self.acute.encoder_spacing_start_m:
            return
        # Whole-ring pitch is part of the approach grip. Adjust it only as
        # the reference seats on the outlet, where the two ends get close.
        correction = np.clip(.25 * min(0., .093 - width) + 1.6 * max(0., .090 - gap), -.006, .030)
        change = np.clip(correction - self.spacing_trim, -.040 * elapsed, .065 * elapsed)
        self.spacing_trim += float(change)

    def tip_field(self, bounds, field, turn):
        mid = (bounds[:-1] + bounds[1:]) * .5
        angle = turn * np.clip(1. - mid / .15, 0., 1.)
        c, s = np.cos(angle), np.sin(angle)
        return np.column_stack([c*field[:, 0] - s*field[:, 1], s*field[:, 0] + c*field[:, 1]])

    def tip_clearance(self, q):
        points = forward(q, self.lengths)
        front = np.linspace(0., .100, 13)
        rear = self.helix_length + np.linspace(-.070, .060, 17)
        a = np.column_stack([np.interp(front, self.curve.s, axis) for axis in points.T])
        b = np.column_stack([np.interp(rear, self.curve.s, axis) for axis in points.T])
        return float(np.linalg.norm(a[:, None] - b, axis=2).min())

    def steer_tip(self, bounds, field, dt):
        limit = math.radians(88.)
        if not .195 <= self.nominal_head_u <= .33:
            self.tip_demand = 0.
        elif self.tip_sample_time != self.last_feedback_time:
            self.tip_sample_time = self.last_feedback_time
            current = self.tip_field(bounds, field, self.tip_turn)
            base = np.clip(self.curve.sampled_targets(bounds, current, self.phase), -limit, limit)
            self.tip_gap_m = self.tip_clearance(self.measured)
            choices = (0., -.35, .35, -.70, .70)
            scores = []
            for turn in choices:
                trial = self.tip_field(bounds, field, turn)
                q = np.clip(self.curve.sampled_targets(bounds, trial, self.phase), -limit, limit)
                predicted = np.clip(self.measured + q - base, -limit, limit)
                scores.append(min(self.tip_clearance(predicted), .073) - .003 * abs(turn))
            self.tip_demand = choices[int(np.argmax(scores))]
        self.tip_turn += float(np.clip(self.tip_demand - self.tip_turn, -1.4 * dt, 1.4 * dt))
        return self.tip_field(bounds, field, self.tip_turn)

    def transfer_to_section(self, bounds, field, shoulder_phase):
        section = self.section_ring
        cut, compact, compact_shoulder = section.section_field(self.nominal_head_u)
        # Start blending a half link before the plane starts to turn. The
        # blend takes one link of nominal travel, at unchanged roll/feed rates.
        if self.acute.section_handoff_from_tilt:
            if self.section_handoff_u is None:
                self.helix_length = bounds[-1]
                return bounds, field, shoulder_phase
            w = float(np.clip((self.nominal_head_u - self.section_handoff_u) / (.5 * self.l), 0., 1.))
        else:
            w = float(np.clip(self.nominal_head_u / self.l + .5, 0., 1.))
        w = w*w*(3. - 2.*w)
        self.section_weight = w
        s, values = self.blend_fields(bounds, field, cut, compact, w)
        angle = math.atan2(math.sin(compact_shoulder - shoulder_phase), math.cos(compact_shoulder - shoulder_phase))
        self.helix_length = s[-1]
        return s, values, shoulder_phase + w*angle

    def blend_fields(self, bounds, field, cut, compact, w):
        sampled = np.linspace(0., 1., 481)

        def resample(s, values):
            u = (s[1:] + s[:-1]) / (2.*s[-1])
            return np.column_stack([np.interp(sampled, u, axis) for axis in values.T])

        length = (1. - w)*bounds[-1] + w*cut[-1]
        swept = bounds[-1]*resample(bounds, field)
        fixed = cut[-1]*resample(cut, compact)
        if self.acute.section_keep_curvature:
            # Blend the turning budget and the direction separately. Mixing
            # vectors can cancel their curvature and briefly open the ring.
            direction = np.arctan2(swept[:, 1], swept[:, 0])
            delta = np.arctan2(fixed[:, 1], fixed[:, 0]) - direction
            delta = np.unwrap(np.arctan2(np.sin(delta), np.cos(delta)))
            direction += w*delta
            turning = (1. - w)*np.linalg.norm(swept, axis=1) + w*np.linalg.norm(fixed, axis=1)
            integrated = turning[:, None]*np.column_stack([np.cos(direction), np.sin(direction)])
        else:
            integrated = (1. - w)*swept + w*fixed
        return sampled*length, (integrated[1:] + integrated[:-1]) / (2.*length)

    def bent_reference(self, dt):
        rate = self.acute.ring_feed_m_s if self.mode == 'arc' else self.acute.feed_m_s
        if self.mode == 'arc':
            rate *= self.feed_fraction
        self.nominal_head_u = min(self.nominal_head_u + rate * dt,
                                  self.acute.reference_bend_radius_m * math.pi / 2. + .65)
        if self.time >= self.next_field_s - 1e-9:
            if self.mode == 'arc' and (self.acute.corner_pitch_m is not None or self.acute.corner_turns is not None
                                       or self.acute.encoder_spacing):
                self.update_corner_section()
            if self.mode == 'arc' and self.acute.section_compression:
                exit_u = self.acute.reference_bend_radius_m * math.pi / 2.
                enter = np.clip((self.nominal_head_u + .015) / .07, 0., 1.)
                leave = np.clip((self.nominal_head_u - exit_u + .015) / .04, 0., 1.)
                self.compression = self.acute.section_compression * (enter*enter*(3.-2.*enter) - leave*leave*(3.-2.*leave))
            length = self.helix_length
            if self.section_ring is not None:
                length = self.acute.ring_turns * helix_turn_length(self.wrap_radius, self.pitch)
            if self.vertical_section is not None:
                bounds, field, end_phase = self.section_ring.section_field(self.nominal_head_u)
            else:
                bounds, field, end_phase = self.bent.field(self.nominal_head_u, length,
                                                           section_compression=self.compression + self.compaction_trim)
            if self.mode == 'helix':
                self.field = bounds, field
            else:
                shoulder_phase = end_phase + self.handedness * math.pi / 2.
                if self.section_ring is not None and self.vertical_section is None:
                    bounds, field, shoulder_phase = self.transfer_to_section(bounds, field, shoulder_phase)
                if self.shoulder > 0.:
                    shoulder = np.array([math.cos(shoulder_phase), math.sin(shoulder_phase)]) / self.wrap_radius
                    bounds = np.r_[bounds, self.helix_length + self.shoulder]
                    field = np.vstack([field, shoulder])
                self.field = np.r_[bounds, self.L], np.vstack([field, [0., 0.]])
            self.next_field_s = self.time + .1
        bounds, field = self.field
        if self.mode == 'arc' and self.acute.encoder_tip_clearance:
            field = self.steer_tip(bounds, field, dt)
        return self.curve.sampled_targets(bounds, field, self.phase)

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

    def step(self, dt, measured, feedback_time=None):
        self.measured = measured
        self.time += dt
        self.elapsed += dt
        if self.time > self.acute.initial_hold_s:
            self.phase += self.handedness * self.acute.roll_rate_rad_s * dt
        mask = None
        self.release_joint = None
        if self.acute.progressive_tail_release and self.stage in ('form_ring', 'release_tail'):
            mask = self.progressive_ring(dt, feedback_time)
        elif self.stage == 'form_ring':
            self.form_ring(dt)
        if ((self.acute.encoder_spacing or self.acute.encoder_compaction or self.acute.encoder_tip_clearance
             or self.acute.encoder_feed_sync or self.acute.section_handoff_from_tilt) and
                self.stage in ('vertical_ring', 'bend_shift') and self.mode == 'arc'):
            self.observe_encoders(feedback_time)
        if self.bent is not None:
            desired = self.bent_reference(dt)
        elif self.stage == 'bend_shift' and self.acute.lead_length_m:
            desired = self.lead_reference()
        elif self.stage == 'bend_shift':
            desired = self.miter_reference()
        elif self.vertical_section is not None:
            desired = self.vertical_section_reference()
        else:
            desired = self.straight_reference()
        if mask is None:
            if self.mode != 'arc':
                mask = np.ones(self.n, dtype=bool)
            elif self.acute.progressive_tail_release:
                mask = self.enabled & self.ring_mask()
            else:
                mask = self.curve.s[:-2] < self.wrap_start + self.helix_length + self.shoulder - 1e-8
        self.target_angle_max = float(np.degrees(np.max(abs(desired))))
        self.apply_reference(desired, dt, mask)
        return self.targets.copy(), self.enabled.copy()

    def status(self):
        return dict(stage=self.stage, phase_rad=self.phase, nominal_head_u_m=self.nominal_head_u,
                    wrap_start_m=self.wrap_start, wrap_length_m=self.helix_length, shoulder_m=self.shoulder,
                    free_tail_m=max(0., self.L - self.wrap_start - self.helix_length - self.shoulder),
                    pitch_m=self.pitch, radius_m=self.wrap_radius, section_major_scale=self.major_scale,
                    section_compression=self.compression,
                    encoder_shape=self.encoder_shape, spacing_trim_m=self.spacing_trim,
                    encoder_compaction=self.compaction_trim,
                    tip_turn_rad=self.tip_turn, tip_gap_m=self.tip_gap_m,
                    feed_fraction=self.feed_fraction,
                    section_weight=self.section_weight,
                    section_handoff_u_m=self.section_handoff_u,
                    released_tail_joint=self.release_joint, released_tail_count=self.release_count,
                    requested_max_deg=self.target_angle_max,
                    curvature_frame_rad=None if self.bent is None else self.bent.gamma,
                    options=asdict(self.acute))
