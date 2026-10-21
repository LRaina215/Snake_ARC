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
from .section_geometry import local_section
from .whole_feeding_observer import WholeFeedingObserver
from .feature_arc import feature_arc_candidates


def locate_measured_arc(angles_deg, lengths, width, down, current_window=None):
    """Compatibility entry: width is no longer a constraint on feature search."""
    return feature_arc_candidates(angles_deg,lengths,down=down,current_window=current_window)


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
        self.direction_observer = WholeFeedingObserver(lengths,(2,3))
        self.runtime_down = None
        self.runtime_quality = None
        self.feature_cache = None

    def passive_down(self, points, window, section):
        """Conditional gravity proxy; one straight long end is not a sensor.

        The open-side hemisphere rejects a pulled-up end, but cannot rule out
        every force/contact that tilts a straight arm. Truth is evaluated later.
        """
        if self.runtime_quality is not None:
            # Candidate evaluation may not borrow a reference it would enable.
            for name,links,_ in self.direction_observer.reference_recipe(self.runtime_quality) if self.runtime_down is not None else []:
                if not (links[1]-1 < window[0] or links[0] > window[1]):
                    return None,dict(source='candidate_would_enable_direction_reference')
            return self.runtime_down,self.runtime_quality
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
        points=forward(np.radians(angles),self.lengths)
        self.runtime_down,self.runtime_quality=self.direction_observer.down_direction(
            points,window,angles,feedback_time_s)
        r = dict(feedback_time_s=feedback_time_s, current_window=list(window), external_window_change=changed,
                 execution='observation_only', executes_window_update=False, valid=False, reasons=[],
                 direction=0, suggested_window=None, state='unobservable',
                 localisation='intrinsic_same_winding_bending', observes_absolute_down=False)
        try:
            key=(tuple(angles),window,None if self.runtime_down is None else tuple(self.runtime_down))
            if self.feature_cache is None or self.feature_cache[0]!=key:
                self.feature_cache=(key,feature_arc_candidates(angles,self.lengths,down=self.runtime_down,current_window=window))
            feature=self.feature_cache[1]
            r.update(global_arc=feature,end_quality=self.runtime_quality,
                     localisation='whole_body_measured_arc_feature')
            if feature['chosen'] is None:
                raise ValueError('No reconstructed open arc feature in full measured body')
            try:
                good,geometry,section=self.geometry_observer.geometry(np.radians(angles),window)
            except (ValueError,np.linalg.LinAlgError) as exc:
                good,geometry,section=False,dict(error=str(exc)),None
            r['current_geometry']=geometry
            center=sum(window)/2
            desired=tuple(feature['chosen']);offset=sum(desired)/2-center
            r.update(valid=True,position_valid=True,features={},
                     phase_reference=None,feature_error=[offset,0.],filtered_feature_error=[offset,0.],
                     filtered_error_joints=offset,estimated_bending_center_joint=sum(desired)/2)
            aligned=None
            if self.runtime_down is not None:
                r['down_head']=self.runtime_down.tolist()
                if section is not None:
                    arc=section['points'][1:-1]
                    opening=(arc[0]+arc[-1])/2-arc.mean(axis=0)
                    aligned=float(opening@self.runtime_down/max(np.linalg.norm(opening),1e-12))
                    r['current_opening_alignment']=aligned
            disjoint=desired[0]>window[1] or desired[1]<window[0]
            if desired!=window and (disjoint or not good or abs(offset)>2. or (aligned is not None and aligned<0.)):
                r.update(state='relocalize',relocalize_window=list(desired))
                return r
            # Half-joint changes can reflect alternate hinge families during
            # rolling. Position shifts require a full-joint centre difference;
            # width changes are separately persisted by the count estimator.
            direction=1 if offset>0 else -1
            if abs(offset)<1.-1e-8:
                self.pending_since,self.pending_direction=None,0
                r['state']='retain'
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
                    gravity_assumption='optional stable passive material segments',
                    position_definition='whole-body measured open-arc feature; phase reference is legacy metadata',
                    inputs=['angles_deg', 'phase_rad', 'enabled', 'feedback_time_s'])
