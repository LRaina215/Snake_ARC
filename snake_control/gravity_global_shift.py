"""Fixed composite curve for a whole-body shift after gravity-assisted drop."""
from dataclasses import replace
import math

import numpy as np

from .curve_gait import arc, helix, line


class ArcHelixShift:
    """Convert an ARC to a helix by moving the joins of a single body curve."""

    def __init__(self, gait, window, amplitude, radius, pitch, handed):
        self.gait = gait
        first, last = window
        self.head_length = (first-1)*gait.l
        self.tail_length = gait.body_length-(last+1)*gait.l
        self.arc_length = gait.body_length-self.head_length-self.tail_length
        self.arc_curvature = amplitude/(2.*gait.l)
        self.helix = helix(0., radius, pitch, handed)

    def targets(self, head_shift, tail_shift, arc_length, anchor_phase, *, tail_intake=False):
        """Keep the bending phase at the original ARC midpoint continuous."""
        head_line = self.head_length-head_shift
        tail_line = self.tail_length-tail_shift
        converted = self.arc_length-arc_length
        head_conversion = 0. if tail_intake else .5*converted
        head_helix = replace(self.helix, length=head_shift+head_conversion)
        tail_helix = replace(self.helix, length=tail_shift+(converted-head_conversion))
        # On right-side intake the rear junction alone consumes the old ARC.
        # Count its torsion if it has already passed the original midpoint.
        behind_anchor = max(0., .5*self.arc_length-arc_length) if tail_intake else 0.
        phase = anchor_phase-self.helix.tau*(head_helix.length+behind_anchor)
        segments = (line(head_line), head_helix, arc(arc_length, self.arc_curvature),
                    tail_helix, line(tail_line))
        return self.gait.targets(segments, phase), phase


class GravityGlobalShift:
    """Move the fitted body interval headward on one helix–arc–line curve.

    The incoming shape is the paper's extended-tail posture. Extending the
    helix ahead of the head provides space for subsequent shifts. The phase
    at the original head remains the same after adding that extension.
    """

    def __init__(self, gait, radius_m, phase_rad, straight_m):
        if not math.isfinite(phase_rad):
            raise ValueError('Phase must be finite')
        if not math.isfinite(straight_m) or straight_m <= 0:
            raise ValueError('Straight tail length must be finite and positive')
        segments, helix_length = gait.segments(radius_m, ls=straight_m,
                                               th2=gait.lead_angle(radius_m))
        self.gait = gait.gait
        self.shift_limit_m = gait.L-helix_length
        helix = segments[0]
        self.segments = (replace(helix, length=helix.length+self.shift_limit_m), *segments[1:])
        self.phase_at_curve_start = phase_rad-helix.tau*self.shift_limit_m

    def targets(self, shift_m, roll_rad=0.):
        """Angles for all joints, with positive shift toward the head."""
        if not math.isfinite(shift_m) or not 0 <= shift_m <= self.shift_limit_m:
            raise ValueError('Shift lies outside the available fixed curve')
        offset = self.shift_limit_m-shift_m
        return self.gait.targets(self.segments, self.phase_at_curve_start+roll_rad, offset)


class MeasuredGravityShift:
    """Translate one curve reconstructed from the measured hanging pose.

    The two alternating hinge families each measure a curvature integral over
    two links. A piecewise-constant correction to a helical reference matches
    those measured integrals. Upstream of the recorded body the correction is
    zero, so the new material coordinates enter the helix continuously.
    """

    def __init__(self, gait, radius_m, phase_rad, angles_rad, support_count, *, upstream_m=0.):
        self.link = gait.l
        n = gait.gait.joints
        angles = np.asarray(angles_rad, float)
        if angles.shape != (n,) or not np.isfinite(angles).all() or not 1 <= support_count < n:
            raise ValueError('Expected one valid hanging pose and a nonempty support and tail')
        if not math.isfinite(upstream_m) or upstream_m < 0:
            raise ValueError('Upstream extension must be finite and non-negative')
        self.shift_limit_m = (n-support_count)*self.link+upstream_m
        self.support_end_m = (support_count+1)*self.link
        self.joint_coordinates = (np.arange(n)+1)*self.link
        self.phase_rad = phase_rad
        self.direction = gait.gait.direction
        segment = helix(1., radius_m, gait.o.pitch, gait.o.handed)
        self.torsion = segment.tau
        self.angle_scale = segment.kappa*2*self.link*np.sinc(segment.tau*self.link/math.pi)
        residual = angles-self.helical_targets(0.)
        self.corrections = []
        for axis in (0, 1):
            coords = self.joint_coordinates[axis::2]
            boundaries = np.r_[coords-self.link, coords[-1]+self.link]
            self.corrections.append((boundaries, np.r_[0., np.cumsum(residual[axis::2])]))

    def helical_targets(self, shift_m, roll_rad=0.):
        phase = self.phase_rad+self.torsion*(self.joint_coordinates-shift_m)+roll_rad
        angles = np.cos(phase)
        angles[1::2] = self.direction*np.sin(phase[1::2])
        return self.angle_scale*angles

    def targets(self, shift_m, front_roll_rad=0.):
        if not math.isfinite(shift_m) or not 0 <= shift_m <= self.shift_limit_m:
            raise ValueError('Shift lies outside the available measured curve')
        q = self.helical_targets(shift_m)
        for axis, (boundaries, integral) in enumerate(self.corrections):
            centers = self.joint_coordinates[axis::2]-shift_m
            q[axis::2] += np.interp(centers+self.link, boundaries, integral)
            q[axis::2] -= np.interp(centers-self.link, boundaries, integral)
        if front_roll_rad:
            # Tie the transition to the curve coordinate. The hanging tail
            # follows the measured curve until it reaches the supported part.
            distance = self.support_end_m-(self.joint_coordinates-shift_m)
            blend = np.clip(distance/(2*self.link), 0., 1.)
            weight = blend**2*(3-2*blend)
            q += weight*(self.helical_targets(shift_m, front_roll_rad)-self.helical_targets(shift_m))
        return q
