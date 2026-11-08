"""Build a stitched curvature field FIRST, then integrate motor targets.

Compatibility model for the existing alternating-hinge feeder. Each pitch
and yaw basis occupies the two links adjacent to its hinge. Same-axis bases
tile the material coordinate; the two axis tilings are staggered. Combining
their breakpoints gives ONE piecewise curvature vector, including shoulders
and the legacy middle redistribution. No old angle template is called here.

This representation deliberately preserves the discrete old shape. It does
not claim to turn its shoulder corrections into a rigid circular translation.
"""
from dataclasses import dataclass
import math

import numpy as np

from snake_control.curve_gait import Segment


@dataclass(frozen=True)
class CurvePlan:
    segments: tuple
    focus: tuple
    cell: int = 0
    fraction: float = 0.

    def targets(self, lengths):
        """Exact integral of k*cos(psi), k*sin(psi) over adjacent-link spans.

        Uses individual link lengths, with the production parity convention.
        Constant torsion is supported so the same integral accepts helix
        segments later; this feeding backend currently uses tau=0 subarcs.
        """
        s = np.r_[0., np.cumsum(lengths)]
        lo, hi = s[:-2], s[2:]
        out = np.zeros(len(lengths)-1)
        boundary = phase = 0.
        pitch = np.arange(len(out)) % 2 == 0
        for segment in self.segments:
            phase += segment.psi
            a = np.maximum(lo, boundary)
            b = np.minimum(hi, boundary+segment.length)
            span = np.maximum(b-a, 0.)
            mid = phase+segment.tau*((a+b)/2-boundary)
            weight = segment.kappa*span*np.sinc(segment.tau*span/(2*math.pi))
            out += weight*np.where(pitch, np.cos(mid), np.sin(mid))
            phase += segment.tau*segment.length
            boundary += segment.length
        return out

    def as_dict(self):
        boundary = phase = 0.
        rows = []
        for segment in self.segments:
            phase += segment.psi
            rows.append(dict(start_m=boundary, length_m=segment.length,
                             curvature_per_m=segment.kappa,
                             bending_direction_rad=phase,
                             torsion_per_m=segment.tau))
            boundary += segment.length
            phase += segment.tau*segment.length
        return dict(focus=list(self.focus), cell=self.cell,
                    fraction=self.fraction, segments=rows)


class FeedingCurveBuilder:
    def __init__(self, lengths):
        self.lengths = np.asarray(lengths, float)
        if self.lengths.ndim != 1 or len(self.lengths) < 4 or not np.isfinite(self.lengths).all() or np.any(self.lengths <= 0):
            raise ValueError('Positive finite individual link lengths required')
        self.n = len(self.lengths)-1
        self.s = np.r_[0., np.cumsum(self.lengths)]

    def _basis(self, window, phase, amplitude_deg, bias):
        """Curvature densities keyed by axis and adjacent-link interval.

        Amplitude is the integrated turn budget of each nominal subarc, not
        an already generated motor vector. Bias redistributes the budget of
        the two shoulder subarcs within each bending-axis family.
        """
        first, last = window
        if not 2 <= first <= last < self.n:
            raise ValueError('Active arc needs passive boundary joints')
        basis = {}
        for parity, projection in enumerate((math.cos(phase), math.sin(phase))):
            family = [j for j in range(first, last+1) if (j-1) % 2 == parity]
            for j in family:
                shoulder = 1.
                if len(family) >= 2:
                    if j == family[0]:
                        shoulder *= 1.-bias
                    if j == family[-1]:
                        shoulder *= 1.+bias
                width = self.s[j+1]-self.s[j-1]
                basis[j] = math.radians(amplitude_deg)*projection*shoulder/width
        return basis

    def _stitch(self, densities, focus, cell=0, fraction=0.):
        # Every link boundary is a possible endpoint of a staggered basis.
        # Assemble the two projected components over their common partition.
        components = np.zeros((len(self.lengths), 2))
        for j, density in densities.items():
            components[j-1:j+1, (j-1) % 2] += density
        segments, phase = [], 0.
        for length, (kp, ky) in zip(self.lengths, components):
            curvature = math.hypot(kp, ky)
            direction = math.atan2(ky, kp) if curvature else phase
            segment = Segment(float(length), curvature, 0., direction-phase)
            # Merge identical neighboring subarcs. A zero-curvature extension
            # carries the frame onward; it does not enable any passive hinge.
            if segments and abs(segment.psi) < 1e-15 and segments[-1].kappa == curvature:
                old = segments[-1]
                segments[-1] = Segment(old.length+length, curvature, 0., old.psi)
            else:
                segments.append(segment)
            phase = direction
        return CurvePlan(tuple(segments), tuple(focus), cell, fraction)

    def template(self, window, phase, amplitude_deg, bias=0.):
        return self._stitch(self._basis(window, phase, amplitude_deg, bias), window)

    def travelling(self, origin, position, phase, bias, options, minimum_bin, maximum_bin):
        cell = int(np.clip(math.floor(position/2), minimum_bin, maximum_bin))
        fraction = float(np.clip(position/2-cell, 0., 1.))
        old = tuple(j+2*cell for j in origin)
        new = tuple(j+2 for j in old)
        left = self._basis(old, phase, options.amplitude_deg, bias)
        right = self._basis(new, phase, options.amplitude_deg, bias)
        densities, support = {}, []
        for j in sorted(left.keys() | right.keys()):
            a, b = left.get(j, 0.), right.get(j, 0.)
            width = self.s[j+1]-self.s[j-1]
            if (abs(a)+abs(b))*width > 1e-10:
                support.append(j)
            densities[j] = (1-fraction)*a+fraction*b
        if not support:
            raise ValueError('Zero bending budget has no feeding arc')
        allocation = float(np.clip((abs(position)-options.middle_boost_activation_joints)/
                                  options.middle_boost_ramp_joints, 0., 1.))
        allocation = allocation**2*(3.-2.*allocation)
        extra_turn = math.radians(options.fractional_middle_boost_deg)*allocation*math.sin(math.pi*fraction)**2
        if len(support) == 4:
            for j, sign in zip(support, (-1., 1., 1., -1.)):
                densities[j] += sign*extra_turn/(self.s[j+1]-self.s[j-1])
        return self._stitch(densities, (support[0], support[-1]), cell, fraction)
