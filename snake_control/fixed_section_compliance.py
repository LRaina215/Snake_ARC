"""Fixed open-arc section compliance. Simulation implementation; no hardware IO.

2023 ingredients: measured shape, line/arc section, length-preserving local
radius/turn deformation, module normalization and Yamada integration.
Local adaptation: zero spatial twist, active-window boundary lengths and global
joint parity. Targets come directly from the measured section, without a
nominal-arc restoring term. Shape checks never substitute a nominal target.
"""
from dataclasses import dataclass,asdict
import math
import numpy as np
from .section_geometry import local_section, sample_section


@dataclass(frozen=True)
class SectionOptions:
    gain: float = .02
    yielding: float = .7  # Legacy configuration field; ignored by measured-section control.
    geometry_period_s: float = .05
    target_rate_deg_s: float = 45.
    target_filter_s: float = .15
    maximum_section_turn_deg: float = 190.
    minimum_gap_fraction: float = .4

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Nonfinite parameter')
        if not 0 <= self.gain < .2:
            raise ValueError('Invalid compliance gain')
        if min(self.geometry_period_s, self.target_rate_deg_s, self.target_filter_s) <= 0:
            raise ValueError('Geometry period, rate limit and filter time must be positive')
        if not 0 < self.minimum_gap_fraction < 1 or not 0 < self.maximum_section_turn_deg < 360:
            raise ValueError('Invalid open-shape bounds')


def yamada_section(curve, link_lengths, first, count, phase, gain=0., direction=1):
    """Exact piecewise κ integral over neighbouring links, τ=0 for Step1.

    θ=κ Δs is the planar limit of the reference's sinc integration. Global
    yaw/pitch indexing and phase match Step1, including windows starting Pitch.
    The same-module line/arc lengths are normalized together to physical length.
    """
    lengths=np.asarray(link_lengths)
    modules=np.r_[lengths[0]/2,(lengths[:-1]+lengths[1:])/2,lengths[-1]/2]
    winding=1 if sum(s.turn for s in curve if 1<=s.joint<=count)>=0 else -1
    total=[sum(s.length for s in curve if s.joint==j) for j in range(len(modules))]
    rows=[]; cursor=0.
    for s in curve:
        length=s.length*modules[s.joint]/max(total[s.joint],1e-12)
        taper=.5 if s.joint in (1,count) else 1.
        factor=1+gain*taper if 1<=s.joint<=count else 1.
        turn=s.turn*winding*factor
        rows.append((cursor,cursor+length,turn/length if s.radius else 0.))
        cursor+=length
    joints=np.r_[0,np.cumsum(lengths)]
    targets=np.zeros(count)
    for j in range(1,count+1):
        integral=sum(k*max(0,min(hi,joints[j+1])-max(lo,joints[j-1])) for lo,hi,k in rows)
        number=first+j-1
        targets[j-1]=integral*(math.cos(phase) if number%2 else direction*math.sin(phase))
    return targets,rows


class FixedSectionCompliance:
    def __init__(self, parameters, lengths, options=SectionOptions()):
        self.p = parameters
        self.lengths = np.asarray(lengths)
        self.options = options
        self.first, self.last = parameters.first_joint, parameters.last_joint
        self.count = self.last - self.first + 1
        # This implementation needs one passive boundary link at each end and
        # enough interior points to estimate an open section plane.
        if self.first < 2 or self.last >= parameters.joints or self.count < 3:
            raise ValueError('Section controller requires 3+ active joints inside J2..J(N-1)')
        if self.lengths.shape != (parameters.joints+1,) or not np.isfinite(self.lengths).all() or np.any(self.lengths <= 0):
            raise ValueError('Expected one positive physical length per backbone link')
        self.enabled = parameters.enabled.copy()
        self.targets = None
        self.previous_axis = None
        self.geometry = None
        self.target_geometry = None
        self.last_goal = None
        self.next_geometry_s = 0.
        self.status = 'initializing'
        self.updates = 0
        self.clipped = 0

    def _check_goal(self, angles):
        """Evaluate the open shape from angles, without any pipe information."""
        geometry = local_section(angles, self.lengths, self.first, self.last)
        active = geometry['points'][1:-1]
        length = np.linalg.norm(np.diff(active, axis=0), axis=1).sum()
        gap = np.linalg.norm(active[-1] - active[0]) / length
        valid = (
            geometry['total_turn'] <= math.radians(self.options.maximum_section_turn_deg) + 1e-8
            and gap >= self.options.minimum_gap_fraction
        )
        if getattr(self.p, 'manual_tuning', False):
            valid = True  # DIY只记录形态，不按研究阈值拒绝目标。
        return valid, geometry

    def update(self, measured, phase, time_s, dt, reference_targets=None):
        """Read actual shape and return one full-size angle command.

        Only self.enabled is driven by the adapter. Passive entries carry the
        measured angles so that later geometry checks use the actual boundary.
        reference_targets is accepted for old callers but deliberately ignored:
        an initial/nominal arc must not restore or replace measured geometry.
        """
        measured = np.asarray(measured)
        if self.targets is None:
            self.targets = measured.copy()
        if time_s + 1e-10 >= self.next_geometry_s:
            self.next_geometry_s = time_s + self.options.geometry_period_s
            self.updates += 1
            geometry = local_section(measured, self.lengths, self.first, self.last, self.previous_axis)
            self.geometry = geometry
            self.previous_axis = geometry['axis']
            links = self.lengths[self.first-1:self.last+2]
            observed, _ = yamada_section(
                geometry['curve'], links, self.first, self.count, phase,
                self.options.gain, self.p.direction,
            )
            # Reference repository: deform the ACTUAL section at constant arc
            # length, integrate its curvature, and use those angles directly.
            goal = measured.copy()
            goal[self.enabled] = observed
            if not getattr(self.p, 'manual_tuning', False):
                goal[self.enabled] = np.clip(goal[self.enabled], -math.pi/2 + .01, math.pi/2 - .01)
            valid, goal_geometry = self._check_goal(goal)
            if not valid:
                self.clipped += 1
            self.last_goal, self.target_geometry = goal, goal_geometry
            self.status = 'running' if valid else 'measured-section target outside opening'
        # Hold the candidate between geometry updates; filter it at every
        # physics/control tick. The next geometry update uses the new phase.
        if self.last_goal is not None and self.status == 'running':
            filter_gain = 1 - math.exp(-dt / self.options.target_filter_s)
            delta = filter_gain * (self.last_goal - self.targets)
            limit = math.radians(self.options.target_rate_deg_s) * dt
            self.targets[self.enabled] += np.clip(delta[self.enabled], -limit, limit)
        self.targets[~self.enabled] = measured[~self.enabled]
        return self.targets.copy()

    def snapshot(self):
        g=self.geometry
        if g is None:return dict(status=self.status)
        curve_points=sample_section(g['curve'])
        xy=g['projected'][1:-1]
        circle=np.linalg.lstsq(np.column_stack([2*xy,np.ones(len(xy))]),np.sum(xy*xy,axis=1),rcond=None)[0]
        radius=math.sqrt(max(0,circle[2]+circle[:2]@circle[:2]))
        return dict(status=self.status,window=[self.first,self.last],geometry_updates=self.updates,
                    control_law='direct_measured_section', nominal_restoring=False,
                    constrained_updates=self.clipped,fit_rms_m=g['fit_rms'],
                    measured_section_turn_deg=math.degrees(g['total_turn']),
                    target_section_turn_deg=math.degrees(self.target_geometry['total_turn']),
                    fitted_body_radius_m=radius,axis=g['axis']['direction'].tolist(),
                    axis_confidence=g['axis']['confidence'],projected_m=g['projected'].tolist(),
                    section_curve_m=curve_points.tolist())
