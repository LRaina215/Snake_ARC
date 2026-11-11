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
from .section_geometry import forward, local_section, section_circle, section_curvature_field, section_integral
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
    outlet_end_correction: bool = False
    outlet_use_spare_length: bool = False
    section_feedback_weight: float = 0.
    section_feedback_gain: float = .02
    vertical_use_spare_length: bool = False
    vertical_adaptive_material: bool = False
    vertical_material_radius_gain: float = .85
    vertical_material_handoff_m: float = 0.
    vertical_material_minimum_wrap_deg: float = 0.
    vertical_material_max_extra_m: float = .060
    vertical_material_wrap_only: bool = False
    vertical_material_return_phase_rad: float = 0.
    vertical_return_advance_m: Optional[float] = None
    section_bearing_advance_m: Optional[float] = None
    section_flat_advance_weight: float = 0.
    section_feedback_during_release: bool = False
    section_preserve_mean_turn: bool = False
    section_local_handoff: bool = False
    bend_section_radius_trim_m: float = 0.


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
        if not 0. <= options.section_feedback_weight <= 1. or not 0. <= options.section_feedback_gain < .2:
            raise ValueError('Invalid section feedback weight or gain')
        if options.section_feedback_weight and not options.section_before_bend:
            raise ValueError('Ring section feedback requires the material curve formed before the bend')
        if options.vertical_adaptive_material and not (options.section_feedback_weight and options.vertical_use_spare_length):
            raise ValueError('Vertical material adjustment needs measured section feedback and an end transition')
        if not 0. <= options.vertical_material_radius_gain <= 1.:
            raise ValueError('Invalid ring radius response gain')
        if options.vertical_material_handoff_m < 0.:
            raise ValueError('Invalid material handoff length')
        if not 0. <= options.vertical_material_minimum_wrap_deg <= 360.:
            raise ValueError('Material allocation requires a valid wrap target')
        if not math.isfinite(options.vertical_material_max_extra_m) or options.vertical_material_max_extra_m <= 0.:
            raise ValueError('Invalid limit for added ring material')
        if options.vertical_material_wrap_only and not (options.vertical_adaptive_material and
                                                        options.vertical_material_minimum_wrap_deg):
            raise ValueError('Wrap-based allocation requires adaptive material and a wrap target')
        if not math.isfinite(options.vertical_material_return_phase_rad) or options.vertical_material_return_phase_rad < 0.:
            raise ValueError('Invalid material return phase span')
        if options.section_local_handoff and not options.section_before_bend:
            raise ValueError('Local handoff requires a separately constructed bearing ring')
        if options.bend_section_radius_trim_m < 0. or (options.bend_section_radius_trim_m and
                                                       not options.vertical_use_spare_length):
            raise ValueError('Bend reshaping requires a fixed material interval with spare arc length')
        if options.section_bearing_advance_m is not None and not 0. <= options.section_bearing_advance_m <= options.ring_pitch_m:
            raise ValueError('Bearing advance must fit inside the available axial offset')
        if not 0. <= options.section_flat_advance_weight < 1. or (options.section_flat_advance_weight and
                not (options.section_feedback_weight and options.vertical_use_spare_length)):
            raise ValueError('Shape-dependent advance needs a measured section and weight in [0, 1)')
        if options.vertical_return_advance_m is not None and (not options.vertical_adaptive_material or
                not 0. < options.vertical_return_advance_m < (options.section_bearing_advance_m or 0.)):
            raise ValueError('Return advance requires an adaptive ring and a smaller bearing advance')
        super().__init__(measured, lengths, HelixArcOptions(target_rate_deg_s=45.))
        if options.bend_section_radius_trim_m >= self.radius + min(0., options.ring_radius_delta_m):
            raise ValueError('Bend reshaping must retain a positive ring radius')
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
        self.ring_formed_time = None
        self.outlet_weight = 0.
        self.outlet_length_budget = 0.
        self.section_feedback_time = None
        self.section_feedback = None
        self.section_feedback_axis = None
        self.section_feedback_diagnostic = None
        self.ring_base_radius = None
        self.ring_radius_offset = 0.
        self.ring_radius_samples = []
        self.ring_radius_filtered = None
        self.ring_turn_error_filtered = 0.
        self.ring_overlap_offset = 0.
        self.ring_narrow_since_phase = None
        self.ring_material_extended = False
        self.vertical_advance_trim = 0.

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

    def confirm_outlet(self):
        """Operator cue that the active ring is on the outgoing straight."""
        if self.stage != 'bend_shift' or not self.acute.outlet_end_correction or self.vertical_section is None:
            return False
        last = int(np.flatnonzero(self.enabled)[-1])
        if last + 1 < self.n:
            baseline_end = self.vertical_section[0][-1] if self.acute.vertical_use_spare_length else self.helix_length
            self.outlet_length_budget = max(0., self.curve.s[last + 1] - baseline_end - .001)
        self.enter('outlet_roll')
        return True

    def make_section(self):
        pitch = self.acute.ring_pitch_m if self.acute.section_local_handoff else self.pitch
        return SectionRingCurve(self.wrap_radius, pitch, self.acute.reference_bend_radius_m,
                                self.l, self.acute.section_shell_diameter_m, handedness=self.handedness,
                                endpoint_allowance=self.acute.section_endpoint_allowance,
                                seam_clearance=self.acute.section_seam_clearance,
                                bearing_advance_m=self.acute.section_bearing_advance_m)

    def ring_profile(self):
        opt = self.acute
        w = min(self.elapsed / 8., 1.)
        w = w * w * (3. - 2. * w)
        self.wrap_radius = self.radius + w * opt.ring_radius_delta_m
        if not opt.section_local_handoff:
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
            self.ring_formed_time = self.time
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

    def ring_spare_length(self):
        end = self.vertical_section[0][-1]
        first_unused = int(np.searchsorted(self.curve.s, end, side='left'))
        return max(0., self.curve.s[first_unused] - end - .001)

    def ring_spare_overlap(self):
        if not self.acute.vertical_use_spare_length or self.ring_formed_time is None:
            return 0.
        age = self.time - self.ring_formed_time
        w = np.clip(age * self.wrap_radius * abs(self.acute.roll_rate_rad_s) / self.l, 0., 1.)
        return self.ring_spare_length() * w*w*(3. - 2.*w)

    def vertical_section_reference(self, feedback_time=None):
        cut, compact = self.vertical_section
        if self.acute.vertical_use_spare_length and self.stage == 'vertical_ring':
            # Extend the end of the curve within the existing torque mask.
            # Keep both axial ramps, including the clearance across the overlap.
            overlap = self.ring_spare_overlap() + self.ring_overlap_offset
            if overlap > 0.:
                bearing = self.section_ring.bearing_advance_m - self.vertical_advance_trim
                profile = None
                if self.acute.section_flat_advance_weight and self.section_feedback is not None:
                    # Preserve the net advance and the shoulder clearance.
                    # Give the flat spans more of the advance so that the
                    # actual corners can round the pipe with less axial twist.
                    positions, correction = self.section_feedback
                    edge = np.minimum(positions, self.helix_length - positions)
                    taper = np.clip(edge / (2. * self.l), 0., 1.)
                    normalized = np.clip(correction * self.wrap_radius, -1., 1.)
                    density = 1. - self.acute.section_flat_advance_weight * normalized * taper
                    profile = positions, density
                cut, compact, _ = self.section_ring.section_field(
                    -.055, overlap, self.wrap_radius + self.ring_radius_offset,
                    bearing_advance_m=bearing, bearing_profile=profile)
            # The carrier has already taken the load. Subsequent material
            # growth changes its circumference, not its takeover stage.
            self.helix_length = cut[-1]
            self.section_weight = 1.
            bounds = np.r_[cut, self.L]
            field = np.vstack([compact, [0., 0.]])
            return self.section_targets(bounds, field, feedback_time)
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
        return self.section_targets(np.r_[bounds, self.L], np.vstack([field, [0., 0.]]), feedback_time)

    def section_targets(self, bounds, field, feedback_time):
        """Lift the measured planar turn to the existing transported curve frame."""
        weight = self.acute.section_feedback_weight
        releasing = self.acute.section_feedback_during_release and self.stage == 'release_tail'
        if not weight or (not releasing and self.stage not in ('vertical_ring', 'bend_shift', 'outlet_roll')):
            return self.curve.sampled_targets(bounds, field, self.phase)
        coverage_end = self.helix_length
        active = self.enabled
        if releasing:
            coverage_end = self.vertical_section[0][-1]
            active = active & (self.curve.s[:-2] < coverage_end)
            weight *= min(1., self.elapsed * self.wrap_radius * abs(self.acute.roll_rate_rad_s) / self.l)
        first, last = np.flatnonzero(active)[[0, -1]] + 1
        offset = self.curve.s[first - 1]
        spacing = self.l
        if feedback_time is not None and (self.section_feedback_time is None or feedback_time > self.section_feedback_time):
            # Register the planar residual against the undeformed 3-D curve
            # projected through the same alternating-joint discretization.
            reference = self.measured.copy()
            nominal = self.curve.sampled_targets(bounds, field, self.phase)
            reference[active] = nominal[active]
            # Fit the carrier rather than the two spatial transitions. In
            # particular the trailing projected point may already lie on a
            # passive link outside the bearing material interval.
            fit = (.9 * spacing, coverage_end - .9 * spacing)
            geometry = local_section(self.measured, self.lengths, first, last,
                                     self.section_feedback_axis, fit_interval_m=fit)
            template = local_section(reference, self.lengths, first, last, geometry['axis'], fit_interval_m=fit)
            self.section_feedback_axis = geometry['axis']
            lengths = self.lengths[first-1:last+2]
            count = last - first + 1
            measured = section_curvature_field(geometry['curve'], lengths, count, self.acute.section_feedback_gain)
            baseline = section_curvature_field(template['curve'], lengths, count)
            # Smooth within one joint pitch; retaining longer flat portions and
            # separate corner peaks is important on a noncircular section.
            positions = np.linspace(0., sum(lengths), 97)
            half = .5 * spacing
            lo, hi = positions - half, positions + half
            correction = ((section_integral(measured, hi) - section_integral(measured, lo)) -
                          (section_integral(baseline, hi) - section_integral(baseline, lo))) / spacing
            quality = float(np.clip((.040 - geometry['fit_rms']) / .025, 0., 1.))
            sample_dt = .2 if self.section_feedback_time is None else feedback_time - self.section_feedback_time
            previous = (np.zeros_like(correction) if self.section_feedback is None else
                        np.interp(positions, *self.section_feedback, right=0.))
            filtered = previous + (1. - math.exp(-sample_dt / .45)) * (quality * correction - previous)
            self.section_feedback_time = feedback_time
            self.section_feedback = positions, filtered
            self.section_feedback_diagnostic = dict(fit_rms_m=geometry['fit_rms'], quality=quality,
                observed_turn_deg=math.degrees(geometry['total_turn']), peak_delta_m_inv=float(max(abs(filtered))))
            if self.acute.vertical_adaptive_material and self.stage == 'vertical_ring':
                self.observe_ring_material(geometry, lengths, coverage_end - offset, sample_dt)
        if self.section_feedback is not None:
            positions, correction = self.section_feedback
            mid = (bounds[1:] + bounds[:-1]) * .5
            curvature = np.linalg.norm(field, axis=1)
            # The two end transitions and their axial clearance remain in the
            # spatial field. Fade the planar deformation into them smoothly.
            taper = np.clip(np.minimum(mid - offset, coverage_end - mid) / (1.5 * spacing), 0., 1.)
            taper = taper * taper * (3. - 2. * taper)
            delta = weight * taper * np.interp(mid - offset, positions, correction, left=0., right=0.)
            limit = .30 * curvature
            if self.acute.section_preserve_mean_turn:
                # Allow curvature to move around the circumference. Avoid
                # reducing the net closing turn of the whole carrier on the
                # strength of a single shape sample. The clearance ramps and
                # material allocation determine the carrier's spatial scale.
                limit *= taper
                delta = np.clip(delta, -limit, limit)
                widths = np.diff(bounds)
                loss = float(widths @ delta)
                if loss < 0.:
                    capacity = limit - delta
                    delta += (-loss / (widths @ capacity)) * capacity
            scale = 1. + np.clip(delta, -limit, limit) / np.maximum(curvature, 1.e-9)
            field = field * scale[:, None]
        return self.curve.sampled_targets(bounds, field, self.phase)

    def observe_ring_material(self, geometry, lengths, coverage, dt):
        """Estimate material demand of the active ring from its own shape."""
        radius, residual, center = section_circle(geometry['fit_projected'])
        material = np.r_[0., np.cumsum(lengths)]
        coords = np.linspace(0., coverage, 65)
        xy = np.array([np.interp(coords, material, p) for p in geometry['projected'].T]).T - center
        azimuth = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
        turn = abs(azimuth[-1] - azimuth[0])
        self.section_feedback_diagnostic.update(observed_radius_m=radius, circle_residual_m=residual,
                                                observed_ring_wrap_deg=math.degrees(turn))
        if self.acute.vertical_material_wrap_only:
            # The centre only sets the winding origin. On a noncircular
            # section use the measured arc length per radian, not a radius
            # change, to decide how much material the ring needs.
            if .04 < radius < .18 and residual <= .30 * radius and turn > math.pi:
                weight = 1. - math.exp(-dt / 1.8)
                error = math.radians(self.acute.vertical_material_minimum_wrap_deg) - turn
                self.ring_turn_error_filtered += weight * (error - self.ring_turn_error_filtered)
                change = .25 * coverage * self.ring_turn_error_filtered / turn
                limit = min(self.acute.vertical_material_max_extra_m, 2. * self.l - self.ring_spare_length())
                rate = .008 if change > 0. else .002
                increment = float(np.clip(change, -rate * dt, rate * dt))
                self.ring_overlap_offset = float(np.clip(self.ring_overlap_offset + increment, 0., limit))
            return
        if not .04 < radius < .18 or residual > .12 * radius:
            return
        phase = self.phase % (2. * math.pi)
        period = 2. * math.pi / abs(self.acute.roll_rate_rad_s)
        if self.ring_base_radius is None:
            self.ring_radius_samples.append((phase, radius, turn))
            if self.elapsed >= period:
                self.ring_base_radius = np.asarray(self.ring_radius_samples).T
                self.ring_radius_filtered = 0.
            return
        phases, radii, turns = self.ring_base_radius
        phase_radius = np.interp(phase, phases, radii, period=2. * math.pi)
        phase_turn = np.interp(phase, phases, turns, period=2. * math.pi)
        difference = radius - phase_radius
        weight = 1. - math.exp(-dt / 1.8)
        self.ring_radius_filtered += weight * (difference - self.ring_radius_filtered)
        target = max(phase_turn, math.radians(self.acute.vertical_material_minimum_wrap_deg))
        self.ring_turn_error_filtered += weight * (target - turn - self.ring_turn_error_filtered)
        desired = np.clip(self.acute.vertical_material_radius_gain * (self.ring_radius_filtered - .001),
                          0., 2. * self.l / (2. * math.pi))
        if self.acute.vertical_material_minimum_wrap_deg:
            # Accumulate material before opening the target radius. Keep a
            # half-spacing reserve for an end whose contact is still on the
            # smaller section of the taper.
            available = max(0., self.ring_overlap_offset - .5 * self.l)
            desired = min(desired, available / (2. * math.pi))
        self.ring_radius_offset += float(np.clip(desired - self.ring_radius_offset, -.003*dt, .003*dt))
        # Widening the reference is only part of the handover. If measured
        # coverage has shortened, extend material without forcing a further
        # global radius release. Use the same ring's phase-calibrated shape.
        gap = self.ring_turn_error_filtered
        wider = self.ring_radius_filtered >= .002
        if wider:
            self.ring_narrow_since_phase = None
            # Distribute the correction over several encoder samples. The
            # ring should not repeatedly give back material at a single roll
            # phase while the fitted section remains wider than at entry.
            adjustment = .25 if self.acute.vertical_material_minimum_wrap_deg else 1.
            requested = self.ring_overlap_offset + adjustment * radius * gap
        else:
            if self.ring_narrow_since_phase is None:
                self.ring_narrow_since_phase = self.phase
            travel = abs(self.phase - self.ring_narrow_since_phase)
            # A part of the ring can encounter the smaller diameter before
            # the rest clears the slope. Require narrow evidence across roll
            # phases before returning material to the passive end.
            requested = self.ring_overlap_offset if travel < self.acute.vertical_material_return_phase_rad else 0.
        budget = max(0., 2. * self.l - 2. * math.pi * self.ring_radius_offset - self.ring_spare_length())
        requested = float(np.clip(requested, 0., min(self.acute.vertical_material_max_extra_m, budget)))
        withdrawal = .002 if wider and self.acute.vertical_material_minimum_wrap_deg else .008
        self.ring_overlap_offset += float(np.clip(requested - self.ring_overlap_offset, -withdrawal*dt, .008*dt))
        if self.ring_overlap_offset >= .012:
            self.ring_material_extended = True
        if self.ring_material_extended and self.acute.vertical_return_advance_m is not None:
            returned = float(np.clip(1. - self.ring_overlap_offset / .012, 0., 1.))
            difference = self.section_ring.bearing_advance_m - self.acute.vertical_return_advance_m
            wanted_trim = returned * difference
            self.vertical_advance_trim += float(np.clip(wanted_trim - self.vertical_advance_trim, -.006*dt, .006*dt))

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

    def bent_reference(self, dt, feedback_time=None):
        rate = self.acute.ring_feed_m_s if self.mode == 'arc' else self.acute.feed_m_s
        if self.mode == 'arc':
            rate *= self.feed_fraction
        self.nominal_head_u = min(self.nominal_head_u + rate * dt,
                                  self.acute.reference_bend_radius_m * math.pi / 2. + .65)
        if self.stage == 'outlet_roll':
            # Blend the end transitions over one link of circumferential travel.
            w = float(np.clip(self.elapsed * self.wrap_radius * abs(self.acute.roll_rate_rad_s) / self.l, 0., 1.))
            self.outlet_weight = w*w*(3. - 2.*w)
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
                reserve = self.ring_spare_overlap()
                outlet = self.outlet_weight * self.outlet_length_budget if self.acute.outlet_use_spare_length else 0.
                overlap = max(reserve, outlet)
                trim = self.acute.bend_section_radius_trim_m
                if trim:
                    # Reallocate material from axial separation to planar
                    # coverage while the bearing ring straddles the bend.
                    # Maintain the J1–J9 interval and restore the normal outlet.
                    entry = np.clip((self.nominal_head_u + .055) / (.5 * self.l), 0., 1.)
                    reshaping = entry*entry*(3. - 2.*entry) * (1. - self.outlet_weight)
                    delta = trim * reshaping
                    # The ring is widest while the transported section first
                    # tilts. Restore axial clearance before the ends overlap.
                    radius = self.acute.reference_bend_radius_m
                    centres = (radius * math.pi / 10., radius * 1.05)
                    distance = min(abs(self.nominal_head_u - centre) for centre in centres)
                    bump = np.clip(1. - distance / (.30 * self.l), 0., 1.)
                    compaction = bump*bump*(3. - 2.*bump)
                    if self.encoder_shape is not None:
                        shell = self.section_ring.shell_diameter
                        gap = self.encoder_shape['end_gap_m']
                        clearance = np.clip((gap - 1.12 * shell) / (.35 * shell), 0., 1.)
                        entry_bump = np.clip(1. - abs(self.nominal_head_u + .05 * radius) / (.25 * self.l), 0., 1.)
                        compaction = max(compaction, .6 * clearance * entry_bump**2 * (3. - 2.*entry_bump))
                    separation = self.section_ring.pitch - 2. * math.pi * delta * compaction
                    bounds, field, end_phase = self.section_ring.section_field(
                        self.nominal_head_u, overlap + 2. * math.pi * delta, self.wrap_radius - delta, separation)
                else:
                    bounds, field, end_phase = self.section_ring.section_field(self.nominal_head_u, overlap)
                if overlap or self.acute.vertical_use_spare_length:
                    self.helix_length = min(bounds[-1], self.vertical_section[0][-1] + max(reserve, self.outlet_length_budget))
                    bounds *= self.helix_length / bounds[-1]
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
        if self.stage == 'outlet_roll' and not self.acute.outlet_use_spare_length:
            # Shape the two end ramps without changing the material interval or
            # the drive clock. Use two link spans to blend into the bearing arc.
            mid = (bounds[1:] + bounds[:-1])*.5
            distance = np.minimum(mid, np.maximum(0., self.helix_length-mid))
            end = np.clip(1. - distance / (2.*self.l), 0., 1.)
            end[mid >= self.helix_length] = 0.
            field = field * (1. + .30*self.outlet_weight*end*end)[:, None]
        if self.mode == 'arc' and self.acute.encoder_tip_clearance:
            field = self.steer_tip(bounds, field, dt)
        if self.mode == 'arc' and self.vertical_section is not None:
            return self.section_targets(bounds, field, feedback_time)
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
             or self.acute.encoder_feed_sync or self.acute.section_handoff_from_tilt or
             self.acute.bend_section_radius_trim_m) and
                self.stage in ('vertical_ring', 'bend_shift') and self.mode == 'arc'):
            self.observe_encoders(feedback_time)
        if self.bent is not None:
            desired = self.bent_reference(dt, feedback_time)
        elif self.stage == 'bend_shift' and self.acute.lead_length_m:
            desired = self.lead_reference()
        elif self.stage == 'bend_shift':
            desired = self.miter_reference()
        elif self.vertical_section is not None:
            desired = self.vertical_section_reference(feedback_time)
        else:
            desired = self.straight_reference()
        if mask is None:
            if self.mode != 'arc':
                mask = np.ones(self.n, dtype=bool)
            elif self.acute.progressive_tail_release:
                if self.acute.vertical_adaptive_material and self.stage == 'vertical_ring':
                    ends = self.curve.s[:-2]
                    occupancy = self.helix_length - ends
                    mask = (occupancy >= .008) | (self.enabled & (occupancy >= .003))
                    handoff = self.acute.vertical_material_handoff_m
                    if handoff:
                        # A newly selected boundary hinge begins by following
                        # its measured angle. Give it the full target only
                        # when the carrier reaches across its material span.
                        joining = mask & (ends >= self.vertical_section[0][-1])
                        weight = np.clip((occupancy[joining] - .008) / handoff, 0., 1.)
                        weight = weight*weight*(3. - 2.*weight)
                        desired[joining] = self.measured[joining] + weight * (desired[joining] - self.measured[joining])
                else:
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
                    section_feedback=self.section_feedback_diagnostic,
                    ring_radius_offset_m=self.ring_radius_offset,
                    ring_overlap_offset_m=self.ring_overlap_offset,
                    vertical_advance_trim_m=self.vertical_advance_trim,
                    section_handoff_u_m=self.section_handoff_u,
                    outlet_weight=self.outlet_weight,
                    released_tail_joint=self.release_joint, released_tail_count=self.release_count,
                    requested_max_deg=self.target_angle_max,
                    curvature_frame_rad=None if self.bent is None else self.bent.gamma,
                    options=asdict(self.acute))
