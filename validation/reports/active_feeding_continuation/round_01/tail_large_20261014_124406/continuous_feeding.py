"""Continuous planar material feeding with encoder-only length feedback.

Position is a travelling curvature profile in material joint coordinates,
not a world displacement. A new local transition follows each old one.
The left/right curvature allocation can change while its total stays pi.
All contact/world support and true length checks belong to the experiment.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from .active_arc_transfer import ActiveArcTransfer, TransferOptions
from .feeding_length_observer import FeedingLengthObserver
from .section_geometry import forward


@dataclass(frozen=True)
class ContinuousFeedingOptions:
    first_joint: int = 8
    last_joint: int = 13
    amplitude_deg: float = 60.
    requested_transfer_m: float = -.0645
    preparation_s: float = 6.
    length_gain_s: float = .8
    speed_damping: float = .7
    maximum_position_rate_joints_s: float = .3
    velocity_filter_s: float = .4
    length_tolerance_m: float = .008
    speed_tolerance_m_s: float = .004
    persistence_s: float = .4
    review_s: float = 2.
    maximum_feed_s: float = 60.
    target_rate_deg_s: float = 45.
    exit_angle_deg: float = 8.
    bearing_bias_limit: float = .18
    bearing_bias_rate_s: float = .125
    bearing_gain: float = .4
    maximum_untrusted_s: float = 3.
    trailing_buffer_joints: int = 2
    release_persistence_s: float = .4


class ContinuousFeeding:
    def __init__(self, lengths, options=ContinuousFeedingOptions()):
        self.lengths = np.asarray(lengths, float)
        self.n = len(self.lengths)-1
        self.options = options
        self.helper = ActiveArcTransfer(self.lengths, 'transfer', TransferOptions(
            first_joint=options.first_joint, last_joint=options.last_joint,
            amplitude_deg=options.amplitude_deg, hold_phase_rad=0.))
        self.length_observer = FeedingLengthObserver(self.lengths)
        self.origin = (options.first_joint, options.last_joint)
        # Each transition advances two alternating hinges. The last valid
        # pair gives a coarse profile range, not a guarantee of support.
        self.minimum_bin = math.ceil((2-options.first_joint)/2)
        self.maximum_bin = math.floor((self.n-1-options.last_joint-2)/2)
        self.position_bounds = (2*self.minimum_bin, 2*(self.maximum_bin+1))
        self.position = self.bias = self.phase = 0.
        self.enabled = self.helper.mask(self.origin)
        self.window = self.origin
        self.targets = self.helper.template(self.origin, 0., options.amplitude_deg)
        self.stage = 'prepare'
        self.stage_since = self.ready_since = None
        self.last_time = self.last_feedback = self.last_length_time = None
        self.previous_angles = self.previous_coordinate = None
        self.velocity_m_s = 0.
        self.observation = self.baseline = self.goal_m = None
        self.bearing_error_m = None
        self.untrusted_since = None
        self.release_below_peak_m = None
        self.release_since = np.full(self.n, np.nan)
        self.completed = False
        self.fault = None
        self.events = []

    def profile(self, position, bias):
        """Interpolate the travelling profile; totals of each axis stay fixed."""
        cell = int(np.clip(math.floor(position/2), self.minimum_bin, self.maximum_bin))
        fraction = float(np.clip(position/2-cell, 0., 1.))
        old = tuple(j+2*cell for j in self.origin)
        new = tuple(j+2 for j in old)
        left = self.helper.template(old, 0., self.options.amplitude_deg, bias)
        right = self.helper.template(new, 0., self.options.amplitude_deg, bias)
        curved = np.flatnonzero(abs(left)+abs(right)>1e-10)+1
        focus = (int(curved[0]), int(curved[-1]))
        return (1-fraction)*left+fraction*right, focus, cell, fraction

    @staticmethod
    def point_at(points, coordinate):
        i = min(int(coordinate), len(points)-2)
        return points[i]+(coordinate-i)*(points[i+1]-points[i])

    def bearing_error(self, q, down, focus):
        """Height difference of two curvature quantile shoulders.

        Distribute each odd-hinge turn over its two-link material cell. This
        avoids selecting a new endpoint just because a tiny bend crosses a
        hard threshold. It is an intrinsic shape constraint, not contact.
        """
        joints = np.arange(1, self.n+1)
        select = (joints%2 == 1) & (joints>=focus[0]) & (joints<=focus[1])
        js = joints[select]
        weights = np.maximum(q[select], 0.)
        total = weights.sum()
        if total < .1:
            return 0.
        points = forward(q, self.lengths)

        def shoulder(fraction):
            amount = total*fraction
            cumulative = np.cumsum(weights)
            k = min(int(np.searchsorted(cumulative, amount)), len(weights)-1)
            previous = 0. if k == 0 else cumulative[k-1]
            coordinate = js[k]-1+2*(amount-previous)/max(weights[k], 1e-12)
            return self.point_at(points, float(coordinate))

        return float(-(shoulder(.15)-shoulder(.85)) @ down)

    def update_bearing(self, angles, feedback_dt):
        down = np.array(self.observation['down_head'])
        q = np.radians(angles)
        target, focus, _, _ = self.profile(self.position, self.bias)
        error = self.bearing_error(q, down, focus)
        predictions = []
        for bias in (self.bias-.01, self.bias+.01):
            proposal, _, _, _ = self.profile(self.position, bias)
            predicted = q.copy()
            predicted[self.helper.mask(focus)] = proposal[self.helper.mask(focus)]
            predictions.append(self.bearing_error(predicted, down, focus))
        slope = (predictions[1]-predictions[0])/.02
        self.bearing_error_m = error
        if abs(slope) > .01:
            correction = -self.options.bearing_gain*error/slope
            limit = self.options.bearing_bias_rate_s*feedback_dt
            self.bias = float(np.clip(self.bias+np.clip(correction, -limit, limit),
                                      -self.options.bearing_bias_limit, self.options.bearing_bias_limit))

    def required_with_buffer(self, focus):
        direction = 1 if self.options.requested_transfer_m > 0 else -1
        if self.stage in ('feed_return', 'review_return'):
            direction *= -1
        first, last = focus
        if direction > 0:
            first = max(2, first-self.options.trailing_buffer_joints)
        else:
            last = min(self.n-1, last+self.options.trailing_buffer_joints)
        return self.helper.mask((first, last))

    def release_ready(self, angles, required, feedback_time, fresh):
        """Encoder geometry below the bearing arc; not a contact-force gate."""
        leaving = self.enabled & ~required
        if not self.observation['valid'] or self.release_below_peak_m is None:
            self.release_since[:] = np.nan
            return np.zeros(self.n, bool)
        points = forward(np.radians(angles), self.lengths)
        heights = -points @ np.array(self.observation['down_head'])
        level = float(heights.max()-self.release_below_peak_m)
        below = np.array([heights[max(0,j):min(len(heights),j+3)].max()<level for j in range(self.n)])
        eligible = leaving & (abs(angles)<=self.options.exit_angle_deg) & below
        if fresh:
            self.release_since[~eligible] = np.nan
            starting = eligible & np.isnan(self.release_since)
            self.release_since[starting] = feedback_time
        return eligible & (feedback_time-self.release_since>=self.options.release_persistence_s-1e-8)

    def advance(self, angles_deg, feedback_time_s, control_time_s):
        angles = np.asarray(angles_deg, float)
        if angles.shape != (self.n,) or not np.isfinite(angles).all():
            raise ValueError('Finite complete encoder sample required')
        age = control_time_s-feedback_time_s
        if not math.isfinite(age) or not -.00000001 <= age <= .5:
            raise ValueError('Current, nonfuture feedback required')
        if self.last_time is not None and control_time_s <= self.last_time:
            raise ValueError('Increasing control time required')
        if self.last_feedback is not None and feedback_time_s < self.last_feedback:
            raise ValueError('Feedback time cannot run backwards')
        fresh = self.last_feedback is None or feedback_time_s > self.last_feedback
        if not fresh and not np.array_equal(angles, self.previous_angles):
            raise ValueError('Changed angles with same feedback time')
        dt = .02 if self.last_time is None else control_time_s-self.last_time
        self.last_time = control_time_s
        opt = self.options
        if fresh:
            feedback_dt = .2 if self.last_feedback is None else feedback_time_s-self.last_feedback
            self.last_feedback = feedback_time_s
            self.previous_angles = angles.copy()
            self.observation = self.length_observer.observe(angles, self.window)
            crown = self.helper.position(angles, self.window)
            self.observation['crest_joint'] = crown.get('crest_joint')
            if self.observation['valid']:
                self.untrusted_since = None
                coordinate = self.observation['transfer_coordinate_m']
                if self.last_length_time is not None:
                    sample_dt = feedback_time_s-self.last_length_time
                    raw_velocity = (coordinate-self.previous_coordinate)/sample_dt
                    amount = 1-math.exp(-sample_dt/opt.velocity_filter_s)
                    self.velocity_m_s += amount*(raw_velocity-self.velocity_m_s)
                self.previous_coordinate = coordinate
                self.last_length_time = feedback_time_s
                if self.stage != 'prepare':
                    self.update_bearing(angles, feedback_dt)
            elif self.stage != 'prepare':
                self.ready_since = None
                if self.untrusted_since is None:
                    self.untrusted_since = control_time_s

        desired = self.targets.copy()
        if self.stage == 'prepare':
            desired = self.helper.template(self.origin, 0., opt.amplitude_deg)
            if control_time_s >= opt.preparation_s-1e-8 and fresh:
                if not self.observation['valid']:
                    self.fault = 'initial differential length unavailable'
                else:
                    self.baseline = self.observation['transfer_coordinate_m']
                    points = forward(np.radians(angles), self.lengths)
                    heights = -points @ np.array(self.observation['down_head'])
                    first, last = self.origin
                    arc_depth = float(heights.max()-heights[first:last+2].min())
                    self.release_below_peak_m = arc_depth+.5*float(np.mean(self.lengths))
                    self.goal_m = self.baseline+opt.requested_transfer_m
                    self.stage, self.stage_since = 'feed_out', control_time_s
                    self.events.append(dict(kind='start_feeding', time_s=control_time_s,
                        goal_transfer_m=opt.requested_transfer_m, baseline_m=self.baseline,
                        inputs='angles_and_time_only'))
        elif self.stage in ('feed_out', 'review_out', 'feed_return', 'review_return'):
            # Pausing the material coordinate must not freeze a newly enabled
            # joint at its preload angle. Finish the already commanded shape
            # while waiting for the length/gravity proxy to become usable.
            desired, _, _, _ = self.profile(self.position, self.bias)
            if self.observation['valid']:
                error = self.goal_m-self.observation['transfer_coordinate_m']
                speed = opt.length_gain_s*error-opt.speed_damping*self.velocity_m_s
                rate = np.clip(speed/np.mean(self.lengths),
                               -opt.maximum_position_rate_joints_s, opt.maximum_position_rate_joints_s)
                proposed_position = float(np.clip(self.position+rate*dt, *self.position_bounds))
                proposal, focus, _, _ = self.profile(proposed_position, self.bias)
                required = self.required_with_buffer(focus)
                leaving = self.enabled & ~required
                release = self.release_ready(angles, required, feedback_time_s, fresh)
                next_mask = required | (leaving & ~release)
                entering = next_mask & ~self.enabled
                if np.any(entering) or np.any(release):
                    self.targets[entering] = np.radians(angles)[entering]
                    self.events.append(dict(kind='mask_handover', time_s=control_time_s,
                        entering=(np.flatnonzero(entering)+1).tolist(),
                        released=(np.flatnonzero(release)+1).tolist(),
                        exit_angles_deg=angles[release].tolist(), enabled_count=int(next_mask.sum()),
                        release_means='persistent_angle_and_below_arc_geometry_not_measured_unloading'))
                self.position = proposed_position
                self.enabled = next_mask
                active = np.flatnonzero(self.enabled)+1
                self.window = (int(active[0]), int(active[-1]))
                desired = proposal
                if fresh and self.stage in ('feed_out', 'feed_return'):
                    reached = abs(error)<=opt.length_tolerance_m and abs(self.velocity_m_s)<=opt.speed_tolerance_m_s
                    self.ready_since = (feedback_time_s if self.ready_since is None else self.ready_since) if reached else None
                    if self.ready_since is not None and feedback_time_s-self.ready_since >= opt.persistence_s-1e-8:
                        self.stage = 'review_out' if self.stage == 'feed_out' else 'review_return'
                        self.stage_since = control_time_s
                        self.events.append(dict(kind='length_goal_observed', time_s=control_time_s,
                            goal_delta_m=self.goal_m-self.baseline, observed_delta_m=coordinate-self.baseline,
                            velocity_m_s=self.velocity_m_s, position_joints=self.position))
            if self.untrusted_since is not None and control_time_s-self.untrusted_since > opt.maximum_untrusted_s:
                self.fault = 'length observation did not recover during bounded shape completion'
            elif self.stage in ('feed_out', 'feed_return') and control_time_s-self.stage_since > opt.maximum_feed_s:
                self.fault = 'length goal not reached within bounded feeding'
            elif self.stage in ('review_out', 'review_return') and control_time_s-self.stage_since >= opt.review_s:
                if self.stage == 'review_out':
                    self.stage, self.stage_since, self.ready_since = 'feed_return', control_time_s, None
                    self.goal_m = self.baseline
                    self.events.append(dict(kind='start_return', time_s=control_time_s))
                else:
                    self.stage, self.completed = 'complete', True
        delta = np.clip(desired-self.targets, -math.radians(opt.target_rate_deg_s)*dt,
                        math.radians(opt.target_rate_deg_s)*dt)
        self.targets[self.enabled] += delta[self.enabled]
        self.targets[~self.enabled] = np.radians(angles)[~self.enabled]
        return self.targets.copy(), self.enabled.copy()

    def snapshot(self):
        return dict(stage=self.stage, route='continuous_feeding', window=list(self.window),
            phase_rad=0., position_joints=self.position, bearing_bias=self.bias,
            bearing_error_m=self.bearing_error_m, goal_m=self.goal_m, baseline_m=self.baseline,
            goal_delta_m=None if self.goal_m is None else self.goal_m-self.baseline,
            observation=self.observation, velocity_m_s=self.velocity_m_s, enabled=self.enabled.tolist(),
            untrusted_since_s=self.untrusted_since,
            release_below_peak_m=self.release_below_peak_m,
            completed=self.completed, fault=self.fault, options=asdict(self.options),
            profile_position_bounds_joints=list(self.position_bounds), compliance=False,
            completion_means='encoder_length_and_speed_goal_not_world_truth_acceptance',
            inputs=['angles_deg', 'feedback_time_s', 'control_time_s'])
