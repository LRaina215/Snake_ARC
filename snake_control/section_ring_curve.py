"""One transported section with two separated end transitions."""
import math

import numpy as np

from .bent_helix_curve import BentHelixCurve


class SectionRingCurve(BentHelixCurve):
    def __init__(self, radius, pitch, bend_radius, link_spacing, shell_diameter, *, handedness=1,
                 endpoint_allowance=False, seam_clearance=False):
        super().__init__(radius, pitch, bend_radius, handedness=handedness)
        self.link_spacing = link_spacing
        self.shell_diameter = shell_diameter
        self.endpoint_allowance = endpoint_allowance
        self.seam_clearance = seam_clearance

    def section_field(self, head_u, outlet_overlap_m=0.):
        # The lead and return share one moving frame. The nearly planar body
        # spans the full section. Most axial displacement is allocated to the
        # ends, over two link spacings, to keep the shell envelopes apart.
        # A half link covers the joint-centred discretisation at the two ends.
        # The resulting angular overlap is separated by the two axial ramps.
        overlap = self.link_spacing / 2. if self.endpoint_allowance else outlet_overlap_m
        extent = 2. * math.pi * self.r + overlap
        fraction = np.linspace(0., 1., 481)
        theta = math.pi - self.handedness * (extent / self.r) * fraction
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
            ends = (np.maximum(0., 1. - fraction / transition)**3
                    - np.maximum(0., 1. - (1. - fraction) / transition)**3)
        # Use the measured shell diameter for tip clearance. Leave the shallow
        # material advance of one head length on the bearing segment itself.
        bearing_advance = min(self.pitch, self.shell_diameter * .4)
        separation = max(self.pitch, self.shell_diameter)
        axial = bearing_advance * (.5 - fraction) + .5 * (separation - bearing_advance) * ends
        u = head_u - separation * .5
        center, normal = self.axis(np.array([u]))
        normal = normal[0]
        tangent = np.array([-normal[2], 0., normal[0]])
        radial = np.cos(theta)[:, None] * normal + np.sin(theta)[:, None] * [0., 1., 0.]
        points = center[0] + self.r * radial + axial[:, None] * tangent
        material = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        samples = np.linspace(0., material[-1], 481)
        points = np.column_stack([np.interp(samples, material, axis) for axis in points.T])
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
