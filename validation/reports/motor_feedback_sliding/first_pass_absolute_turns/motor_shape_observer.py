"""Phase-referenced angle observer. No contact, root pose or actuator inputs.

The calibration is frozen from one explicitly supplied normal rolling cycle.
It is never learned again while slipping, holding phase, or changing masks.
The old observer remains available for reproducing the earlier experiments.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from .angle_shape_observer import AngleShapeObserver, ObserverOptions


def local_phase(phase, first):
    """Local first hinge is cos; an even first hinge swaps the two families."""
    return (phase if first % 2 else math.pi/2-phase) % (2*math.pi)


def bending_features(angles_deg, lengths, window):
    """Intrinsic turn distribution at physical hinges, including boundaries.

    Integrated curvature at each rigid hinge is its absolute measured angle.
    Length weighting converts this to a module curvature mass. The centroid
    describes bending location, not gravity direction or pipe support.
    """
    first, last = window
    n = len(angles_deg)
    indices = np.arange(max(1, first-2), min(n, last+2)+1)
    lengths = np.asarray(lengths)
    modules = (lengths[:-1]+lengths[1:])/2
    turns = np.abs(np.asarray(angles_deg)[indices-1])
    curvature = turns / modules[indices-1]
    mass = curvature * modules[indices-1]
    return dict(bending_center_joint=float(indices @ mass / max(mass.sum(), 1e-12)),
                head_boundary_turn_deg=float(turns[indices < first].sum()),
                tail_boundary_turn_deg=float(turns[indices > last].sum()))


@dataclass(frozen=True)
class MotorObserverOptions:
    crest_threshold_joints: float = .5
    bending_threshold_joints: float = .2
    boundary_threshold_deg: float = 8.
    filter_s: float = .2
    persistence_s: float = .5


class PhaseShapeReference:
    """Periodic shape offsets in relative body coordinates, with provenance."""
    def __init__(self, values, metadata):
        self.values = np.asarray(values, float)
        if self.values.ndim != 2 or self.values.shape[1] != 3 or not np.isfinite(self.values).all():
            raise ValueError('Reference requires finite crest/bending/boundary phase bins')
        self.metadata = dict(metadata)

    def at(self, phase, first):
        count = len(self.values)
        position = local_phase(phase, first)/(2*math.pi)*count-.5
        left = math.floor(position)
        fraction = position-left
        return (1-fraction)*self.values[left % count] + fraction*self.values[(left+1) % count]

    def to_dict(self):
        return dict(values=self.values.tolist(), metadata=self.metadata)

    @classmethod
    def from_dict(cls, value):
        return cls(value['values'], value['metadata'])


class MotorShapeObserver:
    """Angle-only corroborated recommendation; cannot operate any actuator."""
    def __init__(self, lengths, reference, amplitude_deg=60., options=MotorObserverOptions()):
        self.lengths = np.asarray(lengths, float)
        self.reference = reference
        self.options = options
        self.geometry_observer = AngleShapeObserver(lengths, amplitude_deg, ObserverOptions())
        self.filtered = None
        self.pending_direction = 0
        self.pending_since = None
        self.last_time = None

    def update(self, angles_deg, phase_rad, enabled, feedback_time_s):
        # The legacy object supplies geometry/down quality only. Its nominal
        # offset and recommendation are NOT used as migration evidence.
        original = self.geometry_observer.update(angles_deg, phase_rad, enabled, feedback_time_s)
        dt = 0. if self.last_time is None else feedback_time_s-self.last_time
        self.last_time = feedback_time_s
        window = tuple(original['current_window'])
        result = {key: original[key] for key in ('feedback_time_s', 'current_window',
                  'external_window_change', 'valid', 'reasons', 'execution', 'executes_window_update')}
        result.update(direction=0, suggested_window=None, state='unobservable')
        for key in ('down_head', 'end_quality', 'current_geometry', 'crest'):
            if key in original:
                result[key] = original[key]
        if not original['valid']:
            self.filtered, self.pending_direction, self.pending_since = None, 0, None
            return result
        f = bending_features(angles_deg, self.lengths, window)
        center = sum(window)/2
        values = np.array([original['crest']['joint_coordinate']-center,
                           f['bending_center_joint']-center,
                           f['tail_boundary_turn_deg']-f['head_boundary_turn_deg']])
        baseline = self.reference.at(phase_rad, window[0])
        error = values-baseline
        if self.filtered is None or original['external_window_change']:
            self.filtered, self.pending_direction, self.pending_since = error.copy(), 0, None
        else:
            self.filtered += (1-math.exp(-dt/self.options.filter_s))*(error-self.filtered)
        result.update(features=f, phase_reference=baseline.tolist(), feature_error=error.tolist(),
                      filtered_feature_error=self.filtered.tolist(),
                      filtered_error_joints=float(self.filtered[0]))
        if abs(self.filtered[0]) <= self.options.crest_threshold_joints:
            self.pending_direction, self.pending_since = 0, None
            result['state'] = 'retain'
            return result
        direction = 1 if self.filtered[0] > 0 else -1
        corroborated = (direction*self.filtered[1] >= self.options.bending_threshold_joints or
                        direction*self.filtered[2] >= self.options.boundary_threshold_deg)
        if not corroborated:
            self.pending_direction, self.pending_since = 0, None
            result['state'] = 'reject_crest_without_bending_migration'
            return result
        candidate = (window[0]+direction, window[1]+direction)
        # Both far ends need at least three passive links for the down proxy.
        if candidate[0] < 4 or candidate[1] > len(angles_deg)-3:
            self.pending_direction, self.pending_since = 0, None
            result['state'] = 'blocked_passive_end_length'
            return result
        good, geometry, _ = self.geometry_observer.geometry(np.radians(angles_deg), candidate)
        result.update(candidate_window=list(candidate), candidate_geometry=geometry)
        if not good:
            self.pending_direction, self.pending_since = 0, None
            result['state'] = 'blocked_candidate_shape'
            return result
        if self.pending_direction != direction:
            self.pending_direction, self.pending_since = direction, feedback_time_s
        persistent = feedback_time_s-self.pending_since
        result.update(state='confirming', persistence_s=persistent)
        if persistent+1e-9 >= self.options.persistence_s:
            result.update(state='recommend_head' if direction < 0 else 'recommend_tail',
                          direction=direction, suggested_window=list(candidate))
        return result

    def configuration(self):
        return dict(options=asdict(self.options), frozen_reference=self.reference.to_dict(),
                    gravity_assumption='unforced sufficiently straight passive ends',
                    inputs=['angles_deg', 'phase_rad', 'enabled', 'feedback_time_s'])
