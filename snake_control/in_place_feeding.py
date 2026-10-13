"""Planar arc feeding: angles/time only, no compliance or spatial-position input.

Keep the rolling phase zero. Exchange curvature at opposite ends of an open
arc, rather than accumulating rotation or letting rolling drift set direction.
The conditional crown measures material position; it cannot certify world
stationarity. That must be assessed outside this controller.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from .active_arc_transfer import ActiveArcTransfer, TransferOptions


@dataclass(frozen=True)
class FeedingOptions:
    first_joint: int = 8
    last_joint: int = 13
    amplitude_deg: float = 60.
    direction: int = -1
    material_shift_joints: float = 1.
    preparation_s: float = 6.
    feed_gain_s: float = .3
    maximum_progress_rate_s: float = .12
    position_tolerance_joints: float = .06
    persistence_s: float = .4
    review_s: float = 3.
    maximum_feed_s: float = 25.
    target_rate_deg_s: float = 45.


class InPlaceFeeding:
    def __init__(self, lengths, options=FeedingOptions()):
        self.lengths = np.asarray(lengths, float)
        self.options = options
        if options.direction not in (-1, 1) or not all(math.isfinite(v) for v in asdict(options).values()):
            raise ValueError('Finite options and signed direction required')
        self.helper = ActiveArcTransfer(self.lengths, 'transfer', TransferOptions(
            first_joint=options.first_joint, last_joint=options.last_joint,
            amplitude_deg=options.amplitude_deg, hold_phase_rad=0.))
        self.old = (options.first_joint, options.last_joint)
        # A planar bend uses one alternating axis family. Move to its next
        # hinge pair in either direction; moving the zero-axis mask alone
        # would not feed material. The goal controls how far the wave travels.
        self.new = tuple(j+2*options.direction for j in self.old)
        self.old_mask, self.new_mask = self.helper.mask(self.old), self.helper.mask(self.new)
        old_curve = self.helper.template(self.old,0.,options.amplitude_deg)
        new_curve = self.helper.template(self.new,0.,options.amplitude_deg)
        curved = np.flatnonzero((abs(old_curve)+abs(new_curve))>1e-10)+1
        self.focus = (int(curved[0]),int(curved[-1]))
        self.wave_mask = self.helper.mask(self.focus)
        self.window = self.old
        self.enabled = self.old_mask.copy()
        self.targets = self.helper.template(self.old,0.,options.amplitude_deg)
        self.phase = 0.
        self.progress = 0.
        self.stage = 'prepare'
        self.stage_since = self.ready_since = None
        self.last_time = self.last_feedback = None
        self.previous_angles = self.observation = None
        self.baseline = self.goal = None
        self.completed = False
        self.fault = None
        self.events = []

    def advance(self, angles_deg, feedback_time_s, control_time_s):
        angles = np.asarray(angles_deg,float)
        if angles.shape != (len(self.lengths)-1,) or not np.isfinite(angles).all():
            raise ValueError('Finite full motor-angle sample required')
        if not all(math.isfinite(t) for t in (feedback_time_s,control_time_s)) or not 0 <= control_time_s-feedback_time_s <= .5:
            raise ValueError('Feedback must be finite, current and nonfuture')
        if self.last_time is not None and control_time_s <= self.last_time:
            raise ValueError('Increasing control time required')
        if self.last_feedback is not None and feedback_time_s < self.last_feedback:
            raise ValueError('Feedback time cannot run backwards')
        fresh = self.last_feedback is None or feedback_time_s > self.last_feedback
        if not fresh and not np.array_equal(angles,self.previous_angles):
            raise ValueError('Changed angles with same feedback time')
        dt = .02 if self.last_time is None else control_time_s-self.last_time
        self.last_time = control_time_s
        if fresh:
            self.last_feedback = feedback_time_s
            self.previous_angles = angles.copy()
            # Fixed focus avoids turning an enable-boundary change into a
            # purported material movement in the position observation.
            self.observation = self.helper.position(angles,self.focus)
        opt = self.options
        desired = self.targets.copy()
        if self.stage == 'prepare':
            desired = self.helper.template(self.old,0.,opt.amplitude_deg)
            if control_time_s >= opt.preparation_s and fresh:
                if not self.observation['valid']:
                    self.fault = 'initial conditional material position unavailable'
                else:
                    self.baseline = self.observation['crest_joint']
                    self.goal = self.baseline+opt.direction*opt.material_shift_joints
                    entering = self.wave_mask & ~self.enabled
                    self.targets[entering] = np.radians(angles)[entering]
                    self.enabled = self.wave_mask.copy()
                    self.window = self.focus
                    desired = self.targets.copy()
                    self.stage,self.stage_since = 'feed_out',control_time_s
                    self.events.append(dict(kind='start_feeding',time_s=control_time_s,
                        baseline_joint=self.baseline,goal_joint=self.goal,phase_rad=0.,
                        enabled_count=int(self.enabled.sum()),preload_joint=(np.flatnonzero(entering)+1).tolist()))
        elif self.stage in ('feed_out','review_out','feed_return','review_return'):
            if self.observation and self.observation['valid']:
                error = self.goal-self.observation['crest_joint']
                # The sign maps material shift to progression from old to new.
                velocity = np.clip(opt.feed_gain_s*opt.direction*error,
                                   -opt.maximum_progress_rate_s,opt.maximum_progress_rate_s)
                self.progress = float(np.clip(self.progress+velocity*dt,0.,1.))
                if fresh and self.stage in ('feed_out','feed_return'):
                    reached = abs(error) <= opt.position_tolerance_joints
                    self.ready_since = (feedback_time_s if self.ready_since is None else self.ready_since) if reached else None
                    if self.ready_since is not None and feedback_time_s-self.ready_since >= opt.persistence_s-1e-8:
                        self.stage = 'review_out' if self.stage == 'feed_out' else 'review_return'
                        self.stage_since = control_time_s
                        self.events.append(dict(kind='material_goal_observed',time_s=control_time_s,
                            goal_joint=self.goal,observed_joint=self.observation['crest_joint'],progress=self.progress,
                            uses_world_truth=False))
            old = self.helper.template(self.old,0.,opt.amplitude_deg)
            new = self.helper.template(self.new,0.,opt.amplitude_deg)
            desired = (1.-self.progress)*old+self.progress*new
            if self.stage in ('feed_out','feed_return') and control_time_s-self.stage_since > opt.maximum_feed_s:
                self.fault = 'material goal not reached within bounded active feeding'
            elif self.stage in ('review_out','review_return') and control_time_s-self.stage_since >= opt.review_s:
                if self.stage == 'review_out':
                    self.goal = self.baseline
                    self.stage,self.stage_since,self.ready_since = 'feed_return',control_time_s,None
                    self.events.append(dict(kind='start_return',time_s=control_time_s,progress=self.progress,goal_joint=self.goal))
                else:
                    self.stage,self.completed = 'complete',True
        delta = np.clip(desired-self.targets,-math.radians(opt.target_rate_deg_s)*dt,
                        math.radians(opt.target_rate_deg_s)*dt)
        self.targets[self.enabled] += delta[self.enabled]
        self.targets[~self.enabled] = np.radians(angles)[~self.enabled]
        return self.targets.copy(),self.enabled.copy()

    def snapshot(self):
        return dict(stage=self.stage,route='in_place_feeding',window=list(self.window),phase_rad=0.,
            progress=self.progress,goal_joint=self.goal,baseline_crest_joint=self.baseline,
            observation=self.observation,enabled=self.enabled.tolist(),options=asdict(self.options),
            completed=self.completed,fault=self.fault,compliance=False,
            fixed_world_position_observed=False,inputs=['angles_deg','feedback_time_s','control_time_s'])
