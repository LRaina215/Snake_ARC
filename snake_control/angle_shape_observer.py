"""Angle-only open-arc observer. Returns suggestions; has no actuator API.

All positions are reconstructed in the head frame. The inferred down vector
assumes two sufficiently straight passive ends hang in a common direction.
No MuJoCo, world pose, contact, torque, or measured pipe data enters here.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from .section_geometry import forward, local_section, continuous_section


@dataclass(frozen=True)
class ObserverOptions:
    width: int = 6
    end_links: int = 4
    minimum_end_links: int = 3
    minimum_straightness: float = .98
    maximum_end_disagreement_deg: float = 25.
    minimum_turn_deg: float = 150.
    maximum_turn_deg: float = 218.
    minimum_gap: float = .4
    maximum_plane_rms_m: float = .015
    boundary_links: int = 2
    top_band_link_fraction: float = .125
    samples_per_module: int = 12
    minimum_projected_down: float = .25
    error_filter_s: float = .2
    offset_threshold_joints: float = .5
    persistence_s: float = .5


class AngleShapeObserver:
    """One causal observer; current enabled mask always comes from feedback.

    Angles use controller/model signs, in degrees. Phase is a known commanded
    phase in radians, and timestamps refer to actual motor feedback acquisition.
    A recommendation never updates the enabled window inside the observer.
    """
    def __init__(self, link_lengths_m, amplitude_deg=60., options=ObserverOptions()):
        self.lengths = np.asarray(link_lengths_m, float).copy()
        if self.lengths.ndim != 1 or len(self.lengths) < 9 or not np.isfinite(self.lengths).all() or np.any(self.lengths <= 0):
            raise ValueError('Finite positive mechanical link lengths required')
        self.n = len(self.lengths) - 1
        self.amplitude = math.radians(amplitude_deg)
        self.options = options
        self.last_time = None
        self.window = None
        self.filtered_error = None
        self.pending_direction = 0
        self.pending_since = None

    def infer_down(self, points, window):
        first, last = window
        head_count = min(self.options.end_links, first - 1)
        tail_count = min(self.options.end_links, self.n - last)
        if min(head_count, tail_count) < self.options.minimum_end_links:
            raise ValueError('insufficient passive end links')
        head = points[:head_count + 1][::-1]
        tail = points[-tail_count - 1:]
        vectors, straightness = [], []
        for end in (head, tail):
            chord = end[-1] - end[0]
            length = np.linalg.norm(chord)
            if length <= 1e-12:
                raise ValueError('degenerate end direction')
            vectors.append(chord / length)
            straightness.append(float(length / np.linalg.norm(np.diff(end, axis=0), axis=1).sum()))
        disagreement = math.degrees(math.acos(float(np.clip(vectors[0] @ vectors[1], -1., 1.))))
        summed = vectors[0] + vectors[1]
        if np.linalg.norm(summed) <= 1e-12:
            raise ValueError('opposite end directions')
        down = summed / np.linalg.norm(summed)
        quality = dict(head_straightness=straightness[0], tail_straightness=straightness[1],
                       end_disagreement_deg=disagreement, head_direction=vectors[0].tolist(),
                       tail_direction=vectors[1].tolist(), end_link_counts=[head_count, tail_count])
        return down, quality

    def geometry(self, q, window):
        section = local_section(q, self.lengths, *window)
        points = section['points'][1:-1]
        length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        info = dict(turn_deg=math.degrees(section['total_turn']), fit_rms_m=section['fit_rms'],
                    gap_fraction=float(np.linalg.norm(points[-1] - points[0]) / length))
        good = (self.options.minimum_turn_deg <= info['turn_deg'] <= self.options.maximum_turn_deg
                and info['gap_fraction'] >= self.options.minimum_gap
                and info['fit_rms_m'] <= self.options.maximum_plane_rms_m)
        return good, info, section

    def crest(self, points, window, section, down, return_curve=False):
        """Smooth projected curve with a continuous physical joint coordinate.

        Each curved mechanical module is mapped to j-0.5 .. j+0.5, including
        all A/B/C pieces. This prevents short-radius smoothing from changing
        its joint identity. End half-links retain their half-intervals.
        """
        start = max(0, window[0] - self.options.boundary_links)
        stop = min(len(points), window[1] + self.options.boundary_links + 2)
        axis = section['axis']
        basis = axis['basis'][:, :2]
        projected_down = down @ basis
        down_norm = float(np.linalg.norm(projected_down))
        if down_norm < self.options.minimum_projected_down:
            raise ValueError('down direction nearly normal to section')
        projected = (points[start:stop] - axis['point']) @ basis
        curve = continuous_section(projected)
        groups = []
        for segment in curve:
            group = (segment.joint, segment.kind == 'end')
            if not groups or groups[-1][0] != group:
                groups.append((group, []))
            groups[-1][1].append(segment)
        locations, coordinates, arc_weights = [], [], []
        last_point = stop - 1
        for (joint, is_end), pieces in groups:
            if is_end and joint == 0:
                lo, hi = start, start + .5
            elif is_end:
                lo, hi = last_point - .5, last_point
            else:
                lo, hi = start + joint - .5, start + joint + .5
            module_length = sum(piece.length for piece in pieces)
            traversed = 0.
            for piece in pieces:
                # Midpoint quadrature in arclength. Tiny A/B straight remnants
                # cannot receive the same total weight as a long curved piece.
                count = self.options.samples_per_module
                for fraction in (np.arange(count) + .5) / count:
                    coordinates.append(lo + (hi - lo) * (traversed + piece.length * fraction) / module_length)
                    locations.append(piece.at(fraction))
                    arc_weights.append(piece.length / count)
                traversed += piece.length
        locations = np.asarray(locations)
        coordinates = np.asarray(coordinates)
        heights = -locations @ projected_down
        band = float(np.median(self.lengths) * self.options.top_band_link_fraction)
        weight = np.maximum(0., heights - heights.max() + band) * np.asarray(arc_weights)
        crest_joint = float(weight @ coordinates / weight.sum())
        top_xy = (weight[:, None] * locations).sum(axis=0) / weight.sum()
        result = dict(joint_coordinate=crest_joint, point_head_m=(axis['point'] + basis @ top_xy).tolist(),
                      top_band_m=band, projected_down_norm=down_norm,
                      top_coordinate_span=coordinates[weight > 0][[0, -1]].tolist())
        if return_curve:
            result.update(curve_head_m=(locations @ basis.T + axis['point']).tolist(),
                          curve_joint_coordinates=coordinates.tolist())
        return result

    def reference_crest(self, window, phase):
        q = np.zeros(self.n)
        indices = np.arange(window[0] - 1, window[1])
        q[indices] = self.amplitude * np.where(indices % 2 == 0, math.cos(phase), math.sin(phase))
        points = forward(q, self.lengths)
        down, _ = self.infer_down(points, window)
        _, _, section = self.geometry(q, window)
        return self.crest(points, window, section, down)['joint_coordinate']

    def _clear_confirmation(self):
        self.filtered_error = None
        self.pending_direction = 0
        self.pending_since = None

    def update(self, angles_deg, phase_rad, enabled, feedback_time_s):
        angles = np.asarray(angles_deg, float)
        mask = np.asarray(enabled, bool)
        if angles.shape != (self.n,) or mask.shape != angles.shape or not np.isfinite(angles).all():
            raise ValueError('Expected one finite measured angle and enable flag per joint')
        if not math.isfinite(phase_rad) or not math.isfinite(feedback_time_s):
            raise ValueError('Finite phase and feedback timestamp required')
        if self.last_time is not None and feedback_time_s <= self.last_time:
            raise ValueError('Observer requires fresh, increasing feedback timestamps')
        active = np.flatnonzero(mask)
        if len(active) != self.options.width or np.any(np.diff(active) != 1):
            raise ValueError('Expected one contiguous six-joint enable mask')
        window = (int(active[0] + 1), int(active[-1] + 1))
        dt = 0. if self.last_time is None else feedback_time_s - self.last_time
        changed = self.window is not None and window != self.window
        if changed:
            self._clear_confirmation()
        self.window, self.last_time = window, feedback_time_s
        q = np.radians(angles)
        result = dict(execution='observation_only', executes_window_update=False,
                      feedback_time_s=feedback_time_s, current_window=list(window),
                      external_window_change=changed, valid=False, direction=0,
                      suggested_window=None, state='unobservable', reasons=[])
        try:
            points = forward(q, self.lengths)
            down, quality = self.infer_down(points, window)
            result.update(down_head=down.tolist(), end_quality=quality)
            good, geometry, section = self.geometry(q, window)
            result['current_geometry'] = geometry
            crest = self.crest(points, window, section, down)
            reference = self.reference_crest(window, phase_rad)
            error = crest['joint_coordinate'] - reference
            result.update(crest=crest, reference_joint_coordinate=reference, raw_error_joints=error)
            reasons = result['reasons']
            if min(quality['head_straightness'], quality['tail_straightness']) < self.options.minimum_straightness:
                reasons.append('passive end not straight enough')
            if quality['end_disagreement_deg'] > self.options.maximum_end_disagreement_deg:
                reasons.append('passive ends disagree on down direction')
            if not good:
                reasons.append('current section outside open-shape limits')
            if reasons:
                self._clear_confirmation()
                return result
            result['valid'] = True
            if self.filtered_error is None:
                self.filtered_error = error
            else:
                self.filtered_error += (1 - math.exp(-dt / self.options.error_filter_s)) * (error - self.filtered_error)
            result['filtered_error_joints'] = self.filtered_error
            if abs(self.filtered_error) <= self.options.offset_threshold_joints:
                self.pending_direction, self.pending_since = 0, None
                result['state'] = 'retain'
                return result
            direction = 1 if self.filtered_error > 0 else -1
            candidate = (window[0] + direction, window[1] + direction)
            if candidate[0] < 2 or candidate[1] >= self.n:
                result['state'] = 'blocked_boundary'
                self.pending_direction, self.pending_since = 0, None
                return result
            candidate_good, candidate_geometry, _ = self.geometry(q, candidate)
            result['candidate_window'] = list(candidate)
            result['candidate_geometry'] = candidate_geometry
            if not candidate_good:
                result['state'] = 'blocked_candidate_shape'
                self.pending_direction, self.pending_since = 0, None
                return result
            if self.pending_direction != direction:
                self.pending_direction, self.pending_since = direction, feedback_time_s
            persistent = feedback_time_s - self.pending_since
            result.update(state='confirming', pending_direction=direction, persistence_s=persistent)
            if persistent + 1e-9 >= self.options.persistence_s:
                result.update(state='recommend_head' if direction < 0 else 'recommend_tail',
                              direction=direction, suggested_window=list(candidate))
            return result
        except (ValueError, np.linalg.LinAlgError) as exc:
            self._clear_confirmation()
            result['reasons'].append(str(exc))
            return result

    def configuration(self):
        return dict(options=asdict(self.options), mechanical_lengths_m=self.lengths.tolist(),
                    nominal_amplitude_deg=math.degrees(self.amplitude), input_units='controller degrees; phase radians')
