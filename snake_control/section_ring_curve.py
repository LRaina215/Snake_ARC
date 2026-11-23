"""One transported section with two separated end transitions."""
import math

import numpy as np

from .bent_helix_curve import BentHelixCurve
from .section_geometry import section_integral


class SectionRingCurve(BentHelixCurve):
    def __init__(self, radius, pitch, bend_radius, link_spacing, shell_diameter, *, handedness=1,
                 endpoint_allowance=False, seam_clearance=False, bearing_advance_m=None,
                 head_transition_links=2.):
        super().__init__(radius, pitch, bend_radius, handedness=handedness)
        self.link_spacing = link_spacing
        self.shell_diameter = shell_diameter
        self.endpoint_allowance = endpoint_allowance
        self.seam_clearance = seam_clearance
        if not math.isfinite(head_transition_links) or head_transition_links <= 0.:
            raise ValueError('Head transition must occupy a positive number of link spacings')
        self.head_transition_links = head_transition_links
        self.bearing_advance_m = shell_diameter * .4 if bearing_advance_m is None else bearing_advance_m
        self.measured_turns = None
        self.measured_tightening = 0.
        self.measured_smoothing_m = 0.
        self.measured_end_splice = False
        self.measured_outline = None
        self.bearing_outline_residual = None

    def section_field(self, head_u, outlet_overlap_m=0., radius_m=None, separation_m=None,
                      bearing_advance_m=None, bearing_profile=None, tail_ramp_extension_m=0.,
                      overlap_radial_clearance_m=0., clearance_ramp_m=0., axial_crown_m=0.,
                      radial_offset=None):
        # The lead and return share one moving frame. The nearly planar body
        # spans the full section. Most axial displacement is allocated to the
        # ends, over two link spacings, to keep the shell envelopes apart.
        # A half link covers the joint-centred discretisation at the two ends.
        # The resulting angular overlap is separated by the two axial ramps.
        radius = self.r if radius_m is None else radius_m
        overlap = self.link_spacing / 2. if self.endpoint_allowance else outlet_overlap_m
        extent = 2. * math.pi * radius + overlap
        fraction = np.linspace(0., 1., 481)
        theta = math.pi - self.handedness * (extent / radius) * fraction
        if self.seam_clearance:
            transition = 2. * self.link_spacing
            plateau = overlap + self.link_spacing / 2.

            def end_weight(distance):
                w = np.clip(1. - (distance - plateau) / transition, 0., 1.)
                return w**3 * (10. - 15.*w + 6.*w*w)

            # Keep the axial separation across the overlap. The ramps meet
            # the clearance plateaus and the central ring with zero slope.
            ends = end_weight(fraction * extent) - end_weight((1. - fraction) * extent)
        else:
            transition = 2. * self.link_spacing / extent
            head_transition = self.head_transition_links * self.link_spacing / extent
            tail_transition = transition + tail_ramp_extension_m / extent
            ends = (np.maximum(0., 1. - fraction / head_transition)**3
                    - np.maximum(0., 1. - (1. - fraction) / tail_transition)**3)
        # Use the measured shell diameter for tip clearance. Leave the shallow
        # material advance of one head length on the bearing segment itself.
        bearing_advance = min(self.pitch, self.bearing_advance_m if bearing_advance_m is None else bearing_advance_m)
        separation = max(self.pitch if separation_m is None else separation_m, self.shell_diameter)
        advance_fraction = fraction
        if bearing_profile is not None:
            positions, density = bearing_profile
            mid_fraction = .5 * (fraction[:-1] + fraction[1:])
            steps = np.interp(mid_fraction * extent, positions, density, left=1., right=1.)
            advance_fraction = np.r_[0., np.cumsum(steps)]
            advance_fraction /= advance_fraction[-1]
        axial = bearing_advance * (.5 - advance_fraction) + .5 * (separation - bearing_advance) * ends
        if axial_crown_m:
            # Shape the middle of the carrier without changing either end.
            crown = np.maximum(0., 1. - ((fraction - .53) / .29)**2)**3
            axial += axial_crown_m * crown
        u = head_u - separation * .5
        center, normal = self.axis(np.array([u]))
        normal = normal[0]
        tangent = np.array([-normal[2], 0., normal[0]])
        radial = np.cos(theta)[:, None] * normal + np.sin(theta)[:, None] * [0., 1., 0.]
        radial_size = radius
        if self.measured_outline is not None:
            angle = (extent / radius) * fraction
            modes = angle[:, None] * np.array([2., 3., 4.])
            profile = 1. + np.column_stack((np.cos(modes), np.sin(modes))) @ self.measured_outline
            # Radial preload changes the perimeter of the contour, while
            # its tangent returns to the same direction each revolution.
            radial_size = (radius * profile / (1. + self.measured_tightening))[:, None]
        if radial_offset is not None:
            radial_size = radial_size + radial_offset(fraction)[:, None]
        if overlap_radial_clearance_m:
            if not math.isfinite(clearance_ramp_m) or clearance_ramp_m <= 0.:
                raise ValueError('Radial clearance requires a positive material ramp length')
            ramp = np.clip(1. - (1. - fraction) * extent / clearance_ramp_m, 0., 1.)
            radial_size = radial_size + overlap_radial_clearance_m * (ramp**2 * (3. - 2. * ramp))[:, None]
        points = center[0] + radial_size * radial + axial[:, None] * tangent
        material = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if self.bearing_outline_residual is not None:
            # Change the bearing outline while keeping the two end splices
            # and the material interval of the original carrier.
            phase = (extent / radius) * fraction
            terms = phase[:, None] * np.array([2., 3., 4.])
            profile = np.column_stack((np.cos(terms), np.sin(terms))) @ self.bearing_outline_residual
            distance = np.minimum(material, material[-1] - material)
            blend = np.clip(distance / (2. * self.link_spacing), 0., 1.)
            blend = blend**3 * (10. + blend * (-15. + blend * 6.))
            points, material = self.deform_bearing(points, material, radial * blend[:, None], radius * profile)
        samples = np.linspace(0., material[-1], 481)
        if self.measured_turns is None or self.measured_outline is not None:
            points = np.column_stack([np.interp(samples, material, axis) for axis in points.T])
        else:
            # Build planar positions from the measured turning profile, then
            # lift them by the existing axial advance and end ramps.
            height = np.interp(samples, material, axial)
            step = np.sqrt(np.maximum(0., np.diff(samples)**2 - np.diff(height)**2))
            mid = (samples[:-1] + samples[1:]) / 2.
            winding = (1. + self.measured_tightening) * extent / radius
            if self.measured_smoothing_m:
                half_width = self.measured_smoothing_m / 2.
                # Redistribute the measured bend over material length before
                # sampling it with the alternating joint axes.
                hi = section_integral(self.measured_turns, mid + half_width)
                lo = section_integral(self.measured_turns, mid - half_width)
                local_turn = (hi - lo) * np.diff(samples) / (2. * half_width)
                cumulative = np.r_[0., np.cumsum(local_turn)]
                turn = np.r_[.5 * (cumulative[:-1] + cumulative[1:]), cumulative[-1]]
            else:
                turn = section_integral(self.measured_turns, np.r_[mid, samples[-1]])
            if self.measured_end_splice:
                # One axis spacing at each end belongs to the axial seam.
                # Keep its original turning and let the measured ring carry
                # the remaining winding and shape change.
                weight = np.clip(np.minimum(mid, samples[-1] - mid) / self.link_spacing, 0., 1.)
                weight = weight**2 * (3. - 2. * weight)
                nominal_turn = (extent / radius) * np.diff(np.interp(samples, material, fraction))
                ends_turn = (1. - weight) * nominal_turn
                if not self.measured_smoothing_m:
                    local_turn = np.diff(section_integral(self.measured_turns, samples))
                local_turn *= weight
                local_turn = ends_turn + (winding - ends_turn.sum()) * local_turn / local_turn.sum()
                cumulative = np.r_[0., np.cumsum(local_turn)]
                turn = np.r_[.5 * (cumulative[:-1] + cumulative[1:]), cumulative[-1]]
            heading = self.handedness * (math.pi / 2. - winding * turn[:-1] / turn[-1])
            transverse = np.column_stack((np.cos(heading), np.sin(heading)))
            xy = np.vstack((np.zeros(2), np.cumsum(step[:, None] * transverse, axis=0)))
            xy[:, 0] -= radius
            points = center[0] + xy[:, :1] * normal + xy[:, 1:] * [0., 1., 0.] + height[:, None] * tangent
        if self.measured_turns is not None or self.measured_outline is not None:
            bounds, field, end_phase = self.material_field(samples, points, section_axis=tangent)
        else:
            bounds, field, end_phase = self.material_field(samples, points)
        tangents = np.gradient(points, samples, axis=0, edge_order=2)
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        tip = tangents[-1]
        curvature = (tangents[-1] - tangents[-2]) / (samples[-1] - samples[-2])
        curvature -= tip * (tip @ curvature)
        # The outgoing shoulder curves toward the freely hanging tail. This
        # direction is part of the nominal vertical-to-horizontal geometry.
        down = np.array([0., 0., -1.])
        shoulder_angle = math.atan2(np.cross(curvature, down) @ tip, curvature @ down)
        return bounds, field, end_phase + shoulder_angle

    @staticmethod
    def deform_bearing(points, material, directions, displacement):
        """Move the outline normals without extending the bearing material."""
        original_length = material[-1]
        uniform_shift = 0.
        for _ in range(3):
            deformed = points + directions * (displacement - uniform_shift)[:, None]
            chords = np.diff(deformed, axis=0)
            lengths = np.linalg.norm(chords, axis=1)
            derivative = np.sum(chords * np.diff(directions, axis=0) / lengths[:, None])
            uniform_shift += (lengths.sum() - original_length) / derivative
        deformed = points + directions * (displacement - uniform_shift)[:, None]
        arc_length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(deformed, axis=0), axis=1))]
        return deformed, arc_length * (original_length / arc_length[-1])
