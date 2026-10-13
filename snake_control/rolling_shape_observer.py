"""Conditional crest plus phase-calibrated bending, including short ends.

Only angles, phase, enable mask and time enter. The intrinsic bending centre
is NOT a gravity crest and does not measure pipe contact or world up. A
qualified passive end supplies a conditional down proxy, never a measured IMU.
"""
from dataclasses import asdict
import math
import numpy as np
from .motor_shape_observer import MotorObserverOptions, bending_features
from .angle_shape_observer import AngleShapeObserver, ObserverOptions
from .section_geometry import forward


class RollingShapeObserver:
    def __init__(self, lengths, reference, amplitude_deg=60.,
                 options=MotorObserverOptions(), width=6):
        self.lengths = np.asarray(lengths, float)
        self.reference, self.options = reference, options
        self.geometry_observer = AngleShapeObserver(lengths, amplitude_deg, ObserverOptions(width=width))
        self.width = width
        self.last_time = self.window = self.filtered = self.pending_since = None
        self.pending_direction = 0
        self.filtered_crest = None

    def passive_down(self, points, window, section):
        """Conditional gravity proxy; one straight long end is not a sensor.

        The open-side hemisphere rejects a pulled-up end, but cannot rule out
        every force/contact that tilts a straight arm. Truth is evaluated later.
        """
        ends = []
        arc = section['points'][1:-1]
        opening = (arc[0]+arc[-1])/2-arc.mean(axis=0)
        if np.linalg.norm(opening) < 1e-12:
            return None, dict(source='degenerate_opening_direction', candidates=[])
        opening /= np.linalg.norm(opening)
        for name, available in (('head', window[0]-1), ('tail', len(self.lengths)-1-window[1])):
            count = min(4, available)
            if count < 3: continue
            samples = points[:count+1][::-1] if name == 'head' else points[-count-1:]
            chord = samples[-1]-samples[0]
            length = np.linalg.norm(chord)
            if length < 1e-12: continue
            straightness = float(length/np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())
            direction = chord/length
            alignment = float(direction@opening)
            if straightness >= .98 and alignment >= .5:
                ends.append(dict(name=name, count=count, straightness=straightness,
                                 opening_alignment=alignment, direction=direction.tolist()))
        if not ends: return None, dict(source='no_trusted_passive_end', candidates=[])
        if len(ends) == 2:
            a, b = [np.asarray(e['direction']) for e in ends]
            disagreement = math.degrees(math.acos(float(np.clip(a@b, -1., 1.))))
            if disagreement <= 25.:
                direction = a+b; direction /= np.linalg.norm(direction)
                return direction, dict(source='two_consistent_ends', end_disagreement_deg=disagreement, candidates=ends)
        chosen = max(ends, key=lambda e: (e['count'], e['straightness']))
        return np.asarray(chosen['direction']), dict(source='single_'+chosen['name'], candidates=ends,
                                                   does_not_prove_gravity=True)

    def update(self, angles_deg, phase_rad, enabled, feedback_time_s):
        angles, mask = np.asarray(angles_deg, float), np.asarray(enabled, bool)
        active = np.flatnonzero(mask)
        if (angles.shape != (len(self.lengths)-1,) or mask.shape != angles.shape or
                not np.isfinite(angles).all() or not math.isfinite(phase_rad) or not math.isfinite(feedback_time_s)):
            raise ValueError('Finite measured angles, phase and timestamp required')
        if len(active) != self.width or np.any(np.diff(active) != 1):
            raise ValueError('Contiguous mask must match observer width')
        if self.last_time is not None and feedback_time_s <= self.last_time:
            raise ValueError('New increasing feedback required')
        window = (int(active[0]+1), int(active[-1]+1))
        dt = 0. if self.last_time is None else feedback_time_s-self.last_time
        changed = self.window is not None and window != self.window
        if changed or dt > self.options.maximum_feedback_gap_s:
            self.filtered, self.pending_since, self.pending_direction = None, None, 0
            self.filtered_crest = None
        self.window, self.last_time = window, feedback_time_s
        r = dict(feedback_time_s=feedback_time_s, current_window=list(window), external_window_change=changed,
                 execution='observation_only', executes_window_update=False, valid=False, reasons=[],
                 direction=0, suggested_window=None, state='unobservable',
                 localisation='intrinsic_same_winding_bending', observes_absolute_down=False)
        try:
            good, geometry, section = self.geometry_observer.geometry(np.radians(angles), window)
            r['current_geometry'] = geometry
            if not good:
                raise ValueError('Current body outside open-shape limits')
            features = bending_features(angles, self.lengths, window)
            center = sum(window)/2
            values = np.array([features['bending_center_joint']-center,
                               features['tail_boundary_turn_deg']-features['head_boundary_turn_deg']])
            baseline = self.reference.at(phase_rad, window[0])[1:]
            error = values-baseline
            if self.filtered is None:
                self.filtered = error.copy()
            else:
                self.filtered += (1-math.exp(-dt/self.options.filter_s))*(error-self.filtered)
            r.update(valid=True, features=features, phase_reference=baseline.tolist(), feature_error=error.tolist(),
                     filtered_feature_error=self.filtered.tolist(), filtered_error_joints=float(self.filtered[0]),
                     estimated_bending_center_joint=float(center+self.filtered[0]))
            r['position_valid'] = not self.reference.metadata.get('transition_only', False)
            if not r['position_valid']:
                r['state'] = 'transition_geometry_only'
                return r
            down, quality = self.passive_down(forward(np.radians(angles), self.lengths), window, section)
            r['end_quality'] = quality
            primary, corroborated = float(self.filtered[0]), False
            if down is not None:
                crest = self.geometry_observer.crest(forward(np.radians(angles), self.lengths), window, section, down)
                crest_error = crest['joint_coordinate']-center-self.reference.at(phase_rad, window[0])[0]
                if self.filtered_crest is None:
                    self.filtered_crest = crest_error
                else:
                    self.filtered_crest += (1-math.exp(-dt/self.options.filter_s))*(crest_error-self.filtered_crest)
                primary = float(self.filtered_crest)
                r.update(down_head=down.tolist(), crest=crest, localisation='conditional_crest_with_bending',
                         gravity_proxy_assumption='qualified passive end hangs down', filtered_error_joints=primary,
                         observes_absolute_down=False)
                sign = 1 if primary > 0 else -1
                corroborated = (sign*self.filtered[0] >= self.options.bending_threshold_joints or
                                sign*self.filtered[1] >= self.options.boundary_threshold_deg)
            else:
                self.filtered_crest = None
                sign = 1 if primary > 0 else -1
                corroborated = sign*self.filtered[1] >= self.options.boundary_threshold_deg
            direction = 1 if primary > 0 else -1
            if abs(primary) < self.options.crest_threshold_joints or not corroborated:
                self.pending_since, self.pending_direction = None, 0
                r['state'] = 'retain'
                return r
            candidate = (window[0]+direction, window[1]+direction)
            if candidate[0] < 2 or candidate[1] > len(angles)-1:
                self.pending_since, self.pending_direction = None, 0
                r['state'] = 'blocked_robot_boundary'
                return r
            candidate_good, candidate_geometry, _ = self.geometry_observer.geometry(np.radians(angles), candidate)
            r.update(candidate_window=list(candidate), candidate_geometry=candidate_geometry)
            if not candidate_good:
                self.pending_since, self.pending_direction = None, 0
                r['state'] = 'blocked_candidate_shape'
                return r
            if self.pending_direction != direction:
                self.pending_direction, self.pending_since = direction, feedback_time_s
            persistent = feedback_time_s-self.pending_since
            r.update(state='confirming', persistence_s=persistent)
            if persistent+1e-9 >= self.options.persistence_s:
                r.update(state='recommend_head' if direction < 0 else 'recommend_tail', direction=direction,
                         suggested_window=list(candidate))
            return r
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.filtered, self.pending_since, self.pending_direction = None, None, 0
            self.filtered_crest = None
            r['valid'], r['position_valid'] = False, False
            r['reasons'].append(str(exc))
            return r

    def configuration(self):
        return dict(options=asdict(self.options), frozen_reference=self.reference.to_dict(),
                    gravity_assumption='qualified long passive end; otherwise strong intrinsic bending only',
                    position_definition='conditional gravity crest with same-winding corroboration',
                    inputs=['angles_deg', 'phase_rad', 'enabled', 'feedback_time_s'])
