"""Tail-first curve and shift compensation from the supplied 26-servo code."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SourceShape:
    radius: float
    straight: float
    bend: float
    phase: float = 0.


class SourceCurveGait:
    """Integrate the source curve over two links centred on each servo."""

    def __init__(self, joints, link, pitch, arc_radius=.090):
        self.joints, self.link, self.pitch, self.arc_radius = joints, link, pitch, arc_radius
        self.slope = pitch/(2.*math.pi)
        centers = (joints-np.arange(joints))*link
        self.lower, self.upper = centers-link, centers+link

    def torsion(self, radius):
        return self.slope/(radius**2+self.slope**2)

    def lead(self, radius):
        return math.atan2(self.slope, radius)

    def turn_length(self, radius):
        return math.hypot(2.*math.pi*radius, self.pitch)

    def targets(self, shape):
        """Tail to head: line, lead arc, opposite U arc, helix."""
        lead_end = shape.straight+self.arc_radius*max(0., shape.bend)
        u_end = lead_end+self.arc_radius*max(0., -shape.bend)
        angles = np.zeros(self.joints)
        for start, end, kappa, tau, phase in (
            (shape.straight, lead_end, 1./self.arc_radius, 0., shape.phase),
            (lead_end, u_end, 1./self.arc_radius, 0., shape.phase-math.pi),
            (u_end, math.inf, shape.radius/(shape.radius**2+self.slope**2),
             self.torsion(shape.radius), shape.phase-math.pi/2.),
        ):
            left = np.maximum(start, self.lower)
            right = np.minimum(end, self.upper)
            span = np.maximum(0., right-left)
            middle_phase = phase+tau*((left+right)*.5-start)
            integral = kappa*span*np.sinc(tau*span/(2.*math.pi))
            # Source servo order is pitch (−sin), yaw (+cos), then alternating.
            integral[::2] *= -np.sin(middle_phase[::2])
            integral[1::2] *= np.cos(middle_phase[1::2])
            angles += integral
        return angles

    def change_straight(self, shape, straight):
        """Move the helix boundary along the robot and preserve its phase."""
        phase = shape.phase+(straight-shape.straight)*self.torsion(shape.radius)
        return SourceShape(shape.radius, straight, shape.bend, phase)

    def prepare_return(self, shape, released):
        """Replace the U with a tangent lead arc and the straight hanging tail."""
        lead = self.lead(shape.radius)
        straight = (released+1)*self.link
        lost_arc = (abs(shape.bend)-lead)*self.arc_radius
        phase = shape.phase+(straight-shape.straight-lost_arc)*self.torsion(shape.radius)
        return SourceShape(shape.radius, straight, lead, phase)
