"""Angle-only, dual-end gravity proxy for DISPLAY, never for motor control.

Endpoint-frame transport is conditional on a qualified free end remaining
approximately gravity aligned. Joint encoders cannot observe whole-body rigid
rotation or common-mode swing. Confidence describes consistency, not accuracy.
"""
from dataclasses import dataclass
import math
import numpy as np
from .section_geometry import forward


def gravity_unit(v):
    return np.asarray(v, float)/max(float(np.linalg.norm(v)), 1e-12)


def gravity_angle(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(gravity_unit(a), gravity_unit(b)), -1., 1.))))


def gravity_link_frames(q):
    frames = [np.eye(3)]
    for i, angle in enumerate(q):
        a = angle if i % 2 == 0 else -angle
        c, s = math.cos(a), math.sin(a)
        rotation = (np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) if i % 2 == 0 else
                    np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]]))
        frames.append(frames[-1]@rotation)
    return frames


@dataclass(frozen=True)
class GravityPreviewOptions:
    maximum_end_links: int = 4
    minimum_end_links: int = 3
    minimum_straightness: float = .98
    maximum_disagreement_deg: float = 25.
    filter_time_s: float = .4
    correction_rate_deg_s: float = 30.
    maximum_feedback_gap_s: float = .8


class DualEndGravityPreview:
    def __init__(self, lengths, options=GravityPreviewOptions()):
        self.lengths = np.asarray(lengths, float)
        self.options = options
        self.down = self.basis = self.last_time = None
        self.previous_frames = self.previous_candidates = None
        self.result = dict(state='waiting', reliable=False, down_head=None, basis=None)

    def update(self, angles_deg, window, feedback_time_s, hint=None):
        q = np.radians(angles_deg)
        if q.shape != (len(self.lengths)-1,) or not np.isfinite(q).all() or not math.isfinite(feedback_time_s):
            raise ValueError('Finite full-body angles and sample time required')
        if self.last_time is not None and feedback_time_s <= self.last_time:
            return self.result  # Repainting an old sample must not advance the estimator.
        dt = 0. if self.last_time is None else feedback_time_s-self.last_time
        gap = dt > self.options.maximum_feedback_gap_s
        points, frames = forward(q, self.lengths), gravity_link_frames(q)
        arc = points[window[0]:window[1]+2]
        opening = (arc[0]+arc[-1])*.5-arc.mean(axis=0)
        opening_valid = np.linalg.norm(opening) > 1e-6
        opening = gravity_unit(opening)
        candidates = {}
        for name, available, anchor in (('head', window[0]-1, 0), ('tail', len(q)-window[1], len(q))):
            count = min(self.options.maximum_end_links, available)
            if count < self.options.minimum_end_links:
                continue
            samples = points[:count+1][::-1] if name == 'head' else points[-count-1:]
            chord = samples[-1]-samples[0]
            straight = float(np.linalg.norm(chord)/np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())
            direction = gravity_unit(chord)
            alignment = float(direction@opening)
            eligible = opening_valid and straight >= self.options.minimum_straightness and alignment >= .5
            weight = count/self.options.maximum_end_links * float(np.clip((straight-.97)/.03, 0., 1.)) if eligible else 0.
            # Curvature/bending change in that end's own frame discounts an unstable end.
            instability = 0.
            old = (self.previous_candidates or {}).get(name)
            if old and self.previous_frames is not None and not gap:
                transported = frames[anchor]@self.previous_frames[anchor].T@np.array(old['direction'])
                instability = gravity_angle(transported, direction)
                weight /= 1.+(instability/8.)**2
            candidates[name] = dict(direction=direction.tolist(), count=count, anchor=anchor,
                straightness=straight, alignment=alignment, weight=weight, instability_deg=instability)
        usable = [c for c in candidates.values() if c['weight'] > 0.]
        disagreement = None if len(usable) != 2 else gravity_angle(usable[0]['direction'], usable[1]['direction'])
        consistent = disagreement is None or disagreement <= self.options.maximum_disagreement_deg
        # Both ends predict the previous estimate in today's head frame. No root pose is used.
        predictions, weights = [], []
        if self.down is not None and self.previous_frames is not None and not gap:
            for name, c in candidates.items():
                old = (self.previous_candidates or {}).get(name)
                if old and c['weight'] > 0. and old['weight'] > 0.:
                    predictions.append(frames[c['anchor']]@self.previous_frames[c['anchor']].T@self.down)
                    weights.append(min(c['weight'], old['weight']))
        prediction = self.down
        if predictions and consistent:
            prediction = gravity_unit(np.average(predictions, axis=0, weights=weights))
        raw = gravity_unit(sum(c['weight']*np.array(c['direction']) for c in usable)) if usable and consistent else None
        state = 'conflict_hold' if not consistent else 'unobservable_hold'
        # A controller may already have qualified a straight passive subspan
        # (the distal tip itself can be bent). Copy that angle-only observation;
        # never ask the controller to run or change its state for the preview.
        hint_used = False
        if hint:
            direction = np.asarray(hint['down_head'], float)
            if direction.shape == (3,) and np.isfinite(direction).all() and np.linalg.norm(direction) > .5:
                raw = gravity_unit(direction)
                hint_used = True
        reliable = raw is not None
        if raw is not None:
            state = 'observer_'+hint['source'] if hint_used else 'dual' if len(usable) == 2 else 'single_degraded'
            if self.down is None:
                self.down = raw
            else:
                # After a gap, keep the visible history but explicitly reacquire slowly.
                reference = prediction if prediction is not None else self.down
                alpha = 1.-math.exp(-min(dt, .2)/self.options.filter_time_s)
                angle = gravity_angle(reference, raw)
                alpha = min(alpha, self.options.correction_rate_deg_s*min(dt, .2)/max(angle, 1e-9))
                self.down = gravity_unit((1.-alpha)*reference+alpha*raw)
                if gap:
                    state, reliable = 'reacquiring', False
        if self.down is not None:
            up = -self.down
            # Horizontal heading is an arbitrary display convention, NOT magnetic/world yaw.
            x = np.array([0., 1., 0.]) if self.basis is None else self.basis[0].copy()
            x -= (x@up)*up
            if np.linalg.norm(x) < .1:
                x = np.eye(3)[int(np.argmin(abs(up)))]; x -= (x@up)*up
            x = gravity_unit(x)
            self.basis = np.array([x, gravity_unit(np.cross(up, x)), up])
        self.last_time, self.previous_frames, self.previous_candidates = feedback_time_s, frames, candidates
        self.result = dict(state=state, reliable=reliable, down_head=None if self.down is None else self.down.tolist(),
            raw_down_head=None if raw is None else raw.tolist(), basis=None if self.basis is None else self.basis.tolist(),
            candidates=candidates, disagreement_deg=disagreement, feedback_time_s=feedback_time_s,
            observes_world_pose=False, conditional_transport=True)
        return self.result
