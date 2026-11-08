"""Piecewise backbone curves (line / arc / helix) mapped to joint angles.

The curve is parametrized by body arc length u from the head tip (u=0) to the
tail. Joint j (1-based) sits at u_j = sum(lengths[:j]). Its target integrates curvature over
its neighbouring links [u_{j-1}, u_{j+1}] exactly for piecewise-constant
curvature and torsion (Yamada's method):

    odd j:  theta_j =      ∫ kappa(u) cos(psi(u)) du
    even j: theta_j = d *  ∫ kappa(u) sin(psi(u)) du,   psi(u) = roll + ∫ tau du + psi_j

This is the same parity and phase convention as `ArcParameters.targets`; a
single arc of curvature A/(2l) reproduces the fixed-window ARC exactly.
No MuJoCo or hardware access here.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Segment:
    length: float      # body arc length, m
    kappa: float = 0.  # curvature, 1/m
    tau: float = 0.    # torsion, 1/m (sign = handedness)
    psi: float = 0.    # torsion-angle offset added at this segment's start, rad

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.length, self.kappa, self.tau, self.psi)):
            raise ValueError('nonfinite segment')
        if self.length < 0 or self.kappa < 0:
            raise ValueError('length and curvature must be non-negative')


def line(length):
    return Segment(length)


def arc(length, kappa, psi=0.):
    return Segment(length, kappa, 0., psi)


def helix(length, radius, pitch, handed=1, psi=0.):
    """Circular helix around a straight axis; pitch is axial advance per turn.

    `handed` is the sign of the torsion term in this joint convention. The
    physical handedness also depends on `CurveGait.direction`: with
    direction=+1, handed=+1 gives a LEFT-handed helix in the MuJoCo model
    (checked by forward kinematics); use `physical_handedness` to convert.
    """
    if radius <= 0 or pitch < 0 or handed not in (-1, 1):
        raise ValueError('invalid helix')
    c = pitch / (2 * math.pi)
    denom = radius * radius + c * c
    return Segment(length, radius / denom, handed * c / denom, psi)


def physical_handedness(handed, direction):
    """+1 right-handed, -1 left-handed (MuJoCo forward kinematics, 2026-11-06)."""
    return -handed * direction


def helix_turn_length(radius, pitch):
    return math.hypot(2 * math.pi * radius, pitch)


class CurveGait:
    def __init__(self, joints, link_length=.0645, direction=1, lengths=None):
        if joints < 3 or link_length <= 0 or direction not in (-1, 1):
            raise ValueError('invalid robot')
        self.lengths=np.full(joints+1,link_length) if lengths is None else np.asarray(lengths,float).copy()
        if self.lengths.shape!=(joints+1,) or not np.isfinite(self.lengths).all() or np.min(self.lengths)<=0:
            raise ValueError('Expected positive head, adjacent-axis and tail lengths')
        self.joints,self.direction=joints,direction
        self.l=link_length if lengths is None else float(np.median(self.lengths[1:-1]))
        self.s=np.r_[0.,np.cumsum(self.lengths)]
        self.body_length=float(self.s[-1])

    def targets(self, segments, roll=0., offset=0.):
        """Joint angles (rad). `offset` shifts the body along the curve (shift control):
        the head tip sits at curve coordinate `offset`."""
        bounds = np.r_[0., np.cumsum([s.length for s in segments])]
        # psi at each segment start: accumulate torsion and explicit offsets.
        psi0, starts = roll, []
        for s in segments:
            psi0 += s.psi
            starts.append(psi0)
            psi0 += s.tau * s.length
        out = np.zeros(self.joints)
        for j in range(1, self.joints + 1):
            lo, hi = offset+self.s[j-1],offset+self.s[j+1]
            cos_part = sin_part = 0.
            for k, s in enumerate(segments):
                a, b = max(lo, bounds[k]), min(hi, bounds[k + 1])
                if b <= a or s.kappa == 0:
                    continue
                span = b - a
                mid = starts[k] + s.tau * ((a + b) / 2 - bounds[k])
                weight = s.kappa * span * np.sinc(s.tau * span / (2 * math.pi))
                cos_part += weight * math.cos(mid)
                sin_part += weight * math.sin(mid)
            out[j - 1] = cos_part if j % 2 else self.direction * sin_part
        return out

    def window_arc(self, first, last, amplitude_rad, extra_links=0.):
        """ARC as one arc segment covering joints first..last (plus optional extra links)."""
        start = self.s[first-1]-extra_links*self.l
        stop = self.s[last+1]+extra_links*self.l
        kappa = amplitude_rad / (2 * self.l)
        return [line(max(start, 0.)), arc(stop - max(start, 0.), kappa), line(max(self.body_length - stop, 0.))]
