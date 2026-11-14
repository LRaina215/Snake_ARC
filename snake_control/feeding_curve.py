"""Shift a continuous backbone curve in body arc-length coordinates."""
from dataclasses import dataclass
import math

import numpy as np

from .curve_gait import Segment


@dataclass(frozen=True)
class CurvePlan:
    segments: tuple
    focus: tuple
    cell: int = 0
    fraction: float = 0.
    roll_rad: float = 0.

    def targets(self, lengths):
        """Integrate each segment over its overlap with the two links at a joint."""
        s = np.r_[0., np.cumsum(lengths)]
        lo, hi = s[:-2], s[2:]
        out = np.zeros(len(lengths)-1)
        boundary, phase = 0., self.roll_rad
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
        boundary, phase = 0., self.roll_rad
        rows = []
        for segment in self.segments:
            phase += segment.psi
            rows.append(dict(start_m=boundary, length_m=segment.length,
                             curvature_per_m=segment.kappa,
                             bending_direction_rad=phase,
                             torsion_per_m=segment.tau))
            boundary += segment.length
            phase += segment.tau*segment.length
        return dict(focus=list(self.focus), cell=self.cell, roll_rad=self.roll_rad,
                    fraction=self.fraction, segments=rows)


class FeedingCurveBuilder:
    def __init__(self, lengths, window, amplitude_deg, phase, seed_curve=None):
        self.lengths = np.asarray(lengths, float)
        if self.lengths.ndim != 1 or len(self.lengths) < 4 or not np.isfinite(self.lengths).all() or np.any(self.lengths <= 0):
            raise ValueError('Positive finite individual link lengths required')
        self.n = len(self.lengths)-1
        self.s = np.r_[0., np.cumsum(self.lengths)]
        self.spacing = float(np.median(self.lengths[1:-1]))
        self.reference_amplitude = amplitude_deg
        self.reference_curve = (self.initial_arc(window, phase, amplitude_deg)
                                if seed_curve is None else tuple(seed_curve))
        first, *interior, last = self.reference_curve
        if first.kappa != 0. or last.kappa != 0. or not any(s.kappa > 0. for s in interior):
            raise ValueError('Feeding curve requires a curved span between straight arms')
        if not math.isclose(sum(s.length for s in self.reference_curve), self.s[-1], abs_tol=1e-9):
            raise ValueError('Feeding curve must span the robot body')
        # Keep the first and last hinges outside the curved span during motion.
        # Retain the initial position even for a seed already at an end.
        lower = min(first.length, self.s[2])-first.length
        upper = last.length-min(last.length, self.s[-1]-self.s[-3])
        self.position_bounds = (lower/self.spacing, upper/self.spacing)

    def initial_arc(self, window, phase, amplitude_deg):
        first, last = window
        if not 2 <= first < last < self.n:
            raise ValueError('Active arc needs passive boundary joints')
        # Place one continuous curve from the active interval and the entry
        # bending plane. This accounts for the staggered pitch/yaw axes only
        # at entry. During feeding every boundary moves by the same distance.
        first_axis = math.cos(phase) if first % 2 else math.sin(phase)
        last_axis = math.cos(phase) if last % 2 else math.sin(phase)
        head = max(self.s[2], self.s[first]-self.lengths[first-1]*first_axis**2)
        end = min(self.s[-3], self.s[last]+self.lengths[last]*last_axis**2)
        length = end-head
        turn = math.radians(amplitude_deg)*(last-first+1)/2
        return (Segment(head), Segment(length, turn/length), Segment(self.s[-1]-end))

    def plan(self, position, phase, amplitude_deg):
        """Change the straight lengths by +/- displacement, keeping the inner curve."""
        position = float(np.clip(position, *self.position_bounds))
        shift = position*self.spacing
        first, *interior, last = self.reference_curve
        # Amplitude is an explicit operating parameter, independent of shift.
        scale = amplitude_deg/self.reference_amplitude if self.reference_amplitude else 1.
        if scale != 1.:
            interior = [Segment(s.length, s.kappa*scale, s.tau, s.psi) for s in interior]
        segments = (Segment(max(0., first.length+shift), psi=first.psi), *interior,
                    Segment(max(0., last.length-shift), psi=last.psi))
        bounds = np.r_[0., np.cumsum([s.length for s in segments])]
        active = np.zeros(self.n, bool)
        for i, segment in enumerate(segments):
            if segment.kappa > 0.:
                active |= (self.s[2:] > bounds[i]+1e-9) & (self.s[:-2] < bounds[i+1]-1e-9)
        indices = np.flatnonzero(active)+1
        cell = math.floor(position)
        return CurvePlan(segments, (int(indices[0]), int(indices[-1])), cell, position-cell, phase)
