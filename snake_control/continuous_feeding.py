"""Continuous planar material feeding with encoder-only length feedback.

Position is a travelling curvature profile in material joint coordinates,
not a world displacement. A new local transition follows each old one.
The left/right curvature allocation can change while its total stays pi.
All contact/world support and true length checks belong to the experiment.
"""
from dataclasses import dataclass, asdict, replace
import math
import numpy as np
from .active_arc_transfer import ActiveArcTransfer, TransferOptions
from .manual_feeding_observer import ManualFeedingObserver
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
    maximum_position_rate_joints_s: float = .1
    velocity_filter_s: float = .4
    length_tolerance_m: float = .004
    speed_tolerance_m_s: float = .001
    persistence_s: float = .4
    review_s: float = 2.
    maximum_feed_s: float = 60.
    target_rate_deg_s: float = 45.
    exit_angle_deg: float = 8.
    bearing_bias_limit: float = .18
    # Retained for archived option dictionaries; current bias is periodic,
    # not the old integrated shoulder-height controller.
    bearing_bias_rate_s: float = .125
    bearing_gain: float = .4
    maximum_untrusted_s: float = 3.
    trailing_buffer_joints: int = 2
    release_persistence_s: float = .4
    tracking_tolerance_deg: float = 5.
    tracked_turn_soft_limit_deg: float = 192.
    maximum_tracking_wait_s: float = 10.
    travelling_bias_amplitude: float = .12
    fractional_middle_boost_deg: float = 15.
    middle_boost_activation_joints: float = 1.5
    middle_boost_ramp_joints: float = 1.
    adjacent_fold_angle_deg: float = 5.
    adjacent_fold_distance_joints: int = 4
    fold_retention_distance_joints: float = 1.
    release_down_alignment: float = .97
    minimum_passive_reference_links: int = 3
    automatic_return: bool = True  # Research round trips; manual entries set False.
    acquisition_stability_s: float = .6
    acquisition_angle_span_deg: float = 3.
    acquisition_length_span_m: float = .002


class ContinuousFeeding:
    def __init__(self, lengths, options=ContinuousFeedingOptions()):
        self.lengths = np.asarray(lengths, float)
        self.n = len(self.lengths)-1
        self.options = options
        self.reference_direction = 1 if options.requested_transfer_m > 0 else -1
        self.helper = ActiveArcTransfer(self.lengths, 'transfer', TransferOptions(
            first_joint=options.first_joint, last_joint=options.last_joint,
            amplitude_deg=options.amplitude_deg, hold_phase_rad=0.))
        self.length_observer = ManualFeedingObserver(self.lengths, (options.first_joint, options.last_joint))
        self.origin = (options.first_joint, options.last_joint)
        # Each transition advances two alternating hinges. The last valid
        # pair gives a coarse profile range, not a guarantee of support.
        self.minimum_bin = math.ceil((2-options.first_joint)/2)
        self.maximum_bin = math.floor((self.n-1-options.last_joint-2)/2)
        self.position_bounds = (2*self.minimum_bin, 2*(self.maximum_bin+1))
        self.position = self.bias = self.phase = 0.
        # Equal relative settling speed for one-link and larger actions.
        # This is a motor-proxy gate, not a claim of true length stability.
        self.goal_speed_tolerance_m_s = options.speed_tolerance_m_s*max(
            1., abs(options.requested_transfer_m)/float(np.mean(self.lengths)))
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
        self.fold_capture_position = np.full(self.n, np.nan)
        self.tracking_blocked_since = None
        self.tracking_error_deg = self.actual_enabled_turn_deg = 0.
        self.completed = False
        self.fault = None
        self.events = []
        self.acquisition_samples = []
        self.acquisition_reason = None
        self.acquisition_metrics = {}
        self.forced_step = None

    def request_step(self, distance_m, control_time_s):
        """Start one relative action without resetting the travelling profile.

        The start datum is measured after taking over/straightening neighbours.
        Existing targets, mask, profile position and release history survive.
        """
        if not math.isfinite(distance_m) or distance_m == 0:
            raise ValueError('A finite nonzero feeding distance is required')
        self.forced_step = None
        self.options = replace(self.options, requested_transfer_m=distance_m)
        self.goal_speed_tolerance_m_s = self.options.speed_tolerance_m_s*max(
            1., abs(distance_m)/float(np.mean(self.lengths)))
        self.stage, self.stage_since = 'acquire', control_time_s
        self.completed, self.fault = False, None
        self.baseline = self.goal_m = None
        self.ready_since = self.untrusted_since = self.tracking_blocked_since = None
        self.acquisition_samples = []
        self.acquisition_metrics = {}
        self.acquisition_reason = 'waiting_for_takeover_and_stability'
        self.velocity_m_s = 0.
        self.last_length_time = None
        self.events.append(dict(kind='manual_relative_step', time_s=control_time_s,
            requested_transfer_m=distance_m, position_joints=self.position))

    def start_forced_step(self, angles_deg, step_joints, control_time_s):
        """One manual profile increment, independent of length/tracking gates.

        Use the active task's intended direction, not the signed length error.
        Hold the existing torque mask; preload any newly needed joints. The
        target is sent gradually by advance(), without waiting for tracking.
        """
        angles = np.asarray(angles_deg, float)
        if angles.shape != (self.n,) or not np.isfinite(angles).all():
            raise ValueError('Finite complete encoder sample required')
        if not math.isfinite(step_joints) or step_joints <= 0:
            raise ValueError('Forced step must be a positive joint-coordinate increment')
        direction = 1 if self.options.requested_transfer_m > 0 else -1
        if self.stage in ('feed_return', 'review_return'):
            direction *= -1
        previous = self.position
        position = float(np.clip(previous+direction*step_joints, *self.position_bounds))
        if abs(position-previous) < 1e-12:
            self.events.append(dict(kind='forced_step_boundary', time_s=control_time_s,
                position_joints=previous, direction=direction))
            return False
        bias = -self.options.travelling_bias_amplitude*math.sin(math.pi*position)
        desired, focus, _, _ = self.profile(position, bias)
        # Only the new profile/buffer is needed for this explicit command;
        # do not require a passive direction or a measured length baseline.
        required = self.required_with_buffer(focus) | self.helper.mask(focus)
        entering = required & ~self.enabled
        self.targets[entering] = np.radians(angles)[entering]
        self.enabled |= required
        active = np.flatnonzero(self.enabled)+1
        self.window = (int(active[0]), int(active[-1]))
        self.position, self.bias = position, bias
        previous_fault = self.fault
        self.completed, self.fault = False, None
        if self.stage in ('prepare', 'complete'):
            self.stage = 'acquire'
            self.baseline = self.goal_m = None
        self.acquisition_samples = []
        self.acquisition_metrics = {}
        self.ready_since = self.untrusted_since = self.tracking_blocked_since = None
        self.last_length_time = None
        self.velocity_m_s = 0.
        self.stage_since = control_time_s
        self.last_time = control_time_s-.02
        self.forced_step = dict(target_rad=desired.tolist(), from_position_joints=previous,
            to_position_joints=position, direction=direction)
        self.events.append(dict(kind='forced_step_requested', time_s=control_time_s,
            from_position_joints=previous, to_position_joints=position,
            direction=direction, previous_fault=previous_fault,
            entering=(np.flatnonzero(entering)+1).tolist(), measured_distance_claim=False))
        return True

    def acquire_start(self, angles, feedback_time, control_time, fresh):
        """Settle the actual takeover before defining zero transfer.

        No commanded profile travel and no release while collecting the datum.
        Geometric readings during this stage are diagnostics, not progress.
        """
        desired, focus, _, _ = self.profile(self.position, self.bias)
        required = self.required_mask(focus, angles, self.position)
        entering = required & ~self.enabled
        if np.any(entering):
            self.targets[entering] = np.radians(angles)[entering]
            self.enabled |= required
            active = np.flatnonzero(self.enabled)+1
            self.window = (int(active[0]), int(active[-1]))
            self.acquisition_samples = []
            self.acquisition_metrics = {}
            self.events.append(dict(kind='acquisition_takeover', time_s=control_time,
                entering=(np.flatnonzero(entering)+1).tolist(), position_joints=self.position))
        opt = self.options
        self.tracking_error_deg = float(np.max(abs(np.degrees(desired[self.enabled])-angles[self.enabled])))
        self.acquisition_reason = 'waiting_for_stable_shape_and_length'
        if fresh:
            if entering.any() or not self.observation['valid'] or self.tracking_error_deg > opt.tracking_tolerance_deg:
                self.acquisition_samples = []
                self.acquisition_metrics = {}
                self.acquisition_reason = ('waiting_for_takeover_tracking' if entering.any() or self.observation['valid']
                                           else self.observation.get('reason', 'untrusted_direction'))
            else:
                coordinate = self.observation['transfer_coordinate_m']
                quality = self.observation.get('end_quality', {})
                # Changing the reference segments is a new measurement frame.
                # Ignore time-varying quality metrics; compare material spans.
                reference = (quality.get('source'), tuple(
                    (k, str(v.get('selected', {}).get('links')) if isinstance(v.get('selected'), dict) else str(v.get('selected')))
                    for k,v in quality.get('ends', {}).items()))
                if self.acquisition_samples and (feedback_time-self.acquisition_samples[-1][0]>.6 or
                                                  reference != self.acquisition_samples[-1][3]):
                    self.acquisition_samples = []
                self.acquisition_samples.append((feedback_time, angles.copy(), coordinate, reference))
                while len(self.acquisition_samples)>2 and feedback_time-self.acquisition_samples[1][0]>=opt.acquisition_stability_s:
                    self.acquisition_samples.pop(0)
                samples = self.acquisition_samples
                self.acquisition_metrics = dict(samples=len(samples),
                    stable_span_s=feedback_time-samples[0][0],
                    angle_span_deg=float(np.ptp(np.array([s[1] for s in samples]), axis=0).max()),
                    length_span_mm=float(1000*np.ptp([s[2] for s in samples])),
                    tracking_error_deg=self.tracking_error_deg)
                if (len(samples)>=3 and feedback_time-samples[0][0]>=opt.acquisition_stability_s-1e-8 and
                    np.ptp(np.array([s[1] for s in samples]), axis=0).max()<=opt.acquisition_angle_span_deg and
                    np.ptp([s[2] for s in samples])<=opt.acquisition_length_span_m):
                    self.baseline = coordinate
                    self.goal_m = coordinate+opt.requested_transfer_m
                    self.velocity_m_s = 0.
                    self.previous_coordinate, self.last_length_time = coordinate, feedback_time
                    points = forward(np.radians(angles), self.lengths)
                    heights = -points @ np.array(self.observation['down_head'])
                    first, last = focus
                    self.release_below_peak_m = float(heights.max()-heights[first:last+2].min()+.5*np.mean(self.lengths))
                    self.stage, self.stage_since = 'feed_out', control_time
                    self.acquisition_reason = None
                    self.events.append(dict(kind='start_feeding', time_s=control_time,
                        baseline_m=coordinate, goal_transfer_m=opt.requested_transfer_m,
                        position_joints=self.position, window=list(self.window),
                        datum='after_takeover_stable_encoder_geometry'))
        if self.stage=='acquire' and control_time-self.stage_since>opt.maximum_tracking_wait_s:
            self.fault = 'takeover shape/length did not settle; no transfer baseline recorded'
        return desired

    def profile(self, position, bias):
        """Interpolate the travelling profile; totals of each axis stay fixed."""
        cell = int(np.clip(math.floor(position/2), self.minimum_bin, self.maximum_bin))
        fraction = float(np.clip(position/2-cell, 0., 1.))
        old = tuple(j+2*cell for j in self.origin)
        new = tuple(j+2 for j in old)
        left = self.helper.template(old, self.phase, self.options.amplitude_deg, bias)
        right = self.helper.template(new, self.phase, self.options.amplitude_deg, bias)
        curved = np.flatnonzero(abs(left)+abs(right)>1e-10)+1
        focus = (int(curved[0]), int(curved[-1]))
        target = (1-fraction)*left+fraction*right
        # Four-axis interpolation can retain extra material in the top arc.
        # Coordinate both middle bends while preserving the pi turn budget.
        offset = float(np.clip((abs(position)-self.options.middle_boost_activation_joints)/
                               self.options.middle_boost_ramp_joints, 0., 1.))
        allocation = offset*offset*(3.-2.*offset)
        boost = math.radians(self.options.fractional_middle_boost_deg)*allocation*math.sin(math.pi*fraction)**2
        family = curved-1
        if len(family)==4:
            target[family[[0,-1]]] -= boost
            target[family[1:3]] += boost
        return target, focus, cell, fraction

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

    def required_mask(self, focus, angles, position):
        required = self.required_with_buffer(focus)
        actual_arc = self.observation.get('actual_arc_window')
        if actual_arc is not None:
            required |= self.helper.mask(tuple(actual_arc))
        active = np.flatnonzero(required)+1
        radius = self.options.adjacent_fold_distance_joints
        joints = np.arange(1,self.n+1)
        folds = joints[(abs(angles)>=self.options.adjacent_fold_angle_deg) &
                       (joints>=active[0]-radius) & (joints<=active[-1]+radius)]
        self.fold_capture_position[folds-1] = position
        # A straightened fold often returns under gravity immediately
        # after torque-off. Keep its temporary straightening control
        # until the travelling arc leaves, rather than cycling masks
        # at a stationary length goal. This is material distance,
        # never pipe contact or a world unloading observation.
        retained = joints[np.isfinite(self.fold_capture_position) &
            (abs(position-self.fold_capture_position)<self.options.fold_retention_distance_joints)]
        folds = np.union1d(folds, retained)
        if len(folds):
            first=max(2,min(int(active[0]),int(folds.min())))
            last=min(self.n-1,max(int(active[-1]),int(folds.max())))
            required |= self.helper.mask((first,last))
        # Keep one distal torque-off reference observable. The end
        # being lengthened on the outbound leg has the larger reserve;
        # use the same end through return to avoid switching the
        # observation geometry. Reserved folds remain passive and are
        # still fully included in external whole-body shape audits.
        reserve = self.options.minimum_passive_reference_links
        if self.reference_direction > 0:
            required[:reserve] = False
        else:
            required[self.n-reserve:] = False
        return required

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
        directions = np.diff(points,axis=0)
        directions /= np.maximum(np.linalg.norm(directions,axis=1)[:,None],1e-12)
        indices = np.arange(self.n)
        active = np.flatnonzero(required)
        towards_tip = directions[1:].copy()
        towards_tip[indices<active[0]] *= -1
        aligned = towards_tip @ np.array(self.observation['down_head']) >= self.options.release_down_alignment
        eligible = leaving & (abs(angles)<=self.options.exit_angle_deg) & below & aligned
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
            self.last_feedback = feedback_time_s
            self.previous_angles = angles.copy()
            self.observation = self.length_observer.observe(angles, self.window, feedback_time_s)
            if 'crest_joint' not in self.observation:
                self.observation['crest_joint'] = None
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
                # A periodic intrinsic allocation coordinates the middle
                # hinges without an accumulated orientation bias at rest.
            elif self.stage != 'prepare':
                self.ready_since = None
                if self.untrusted_since is None:
                    self.untrusted_since = control_time_s

        if self.forced_step is not None:
            desired = np.array(self.forced_step['target_rad'])
            limit = math.radians(opt.target_rate_deg_s)*min(dt, .02)
            self.targets[self.enabled] += np.clip(desired-self.targets, -limit, limit)[self.enabled]
            self.targets[~self.enabled] = np.radians(angles)[~self.enabled]
            if np.max(abs(desired[self.enabled]-self.targets[self.enabled])) < 1e-10:
                self.events.append(dict(kind='forced_step_targets_sent', time_s=control_time_s,
                    position_joints=self.position, actual_motion_verified=False))
                self.forced_step = None
                self.stage_since = control_time_s
                self.last_length_time = None
                self.velocity_m_s = 0.
            return self.targets.copy(), self.enabled.copy()

        desired = self.targets.copy()
        if self.stage == 'prepare':
            desired = self.helper.template(self.origin, self.phase, opt.amplitude_deg)
            if control_time_s >= opt.preparation_s-1e-8 and fresh:
                if not self.observation['valid']:
                    # A fresh sample at the nominal preparation boundary may
                    # still be gathering stable passive geometry. Use the same
                    # bounded observation wait as during feeding, not one frame.
                    if control_time_s >= opt.preparation_s+opt.maximum_untrusted_s:
                        self.fault = 'initial differential length unavailable'
                else:
                    points = forward(np.radians(angles), self.lengths)
                    heights = -points @ np.array(self.observation['down_head'])
                    first, last = self.origin
                    arc_depth = float(heights.max()-heights[first:last+2].min())
                    self.release_below_peak_m = arc_depth+.5*float(np.mean(self.lengths))
                    self.stage, self.stage_since = 'acquire', control_time_s
                    desired = self.acquire_start(angles, feedback_time_s, control_time_s, fresh)
        elif self.stage == 'acquire':
            desired = self.acquire_start(angles, feedback_time_s, control_time_s, fresh)
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
                current, _, _, _ = self.profile(self.position, self.bias)
                self.tracking_error_deg = float(np.max(abs(np.degrees(current[self.enabled])-angles[self.enabled])))
                self.actual_enabled_turn_deg = float(abs(angles[self.enabled]).sum())
                blocked = (self.tracking_error_deg>opt.tracking_tolerance_deg or
                           self.actual_enabled_turn_deg>opt.tracked_turn_soft_limit_deg)
                if blocked:
                    proposed_position = self.position
                    if self.tracking_blocked_since is None:
                        self.tracking_blocked_since = control_time_s
                else:
                    self.tracking_blocked_since = None
                self.bias = -opt.travelling_bias_amplitude*math.sin(math.pi*proposed_position)
                proposal, focus, _, _ = self.profile(proposed_position, self.bias)
                required = self.required_mask(focus, angles, proposed_position)
                leaving = self.enabled & ~required
                release = self.release_ready(angles, required, feedback_time_s, fresh)
                next_mask = required | (leaving & ~release)
                entering = next_mask & ~self.enabled
                # A completed release ends this temporary fold capture. Keep
                # no dormant record that can re-enable it when position moves
                # back across the retention boundary. Fresh nearby bends and
                # the current arc can still request a new takeover normally.
                self.fold_capture_position[release] = np.nan
                self.release_since[release] = np.nan
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
                    reached = abs(error)<=opt.length_tolerance_m and abs(self.velocity_m_s)<=self.goal_speed_tolerance_m_s
                    self.ready_since = (feedback_time_s if self.ready_since is None else self.ready_since) if reached else None
                    if self.ready_since is not None and feedback_time_s-self.ready_since >= opt.persistence_s-1e-8:
                        self.stage = 'review_out' if self.stage == 'feed_out' else 'review_return'
                        self.stage_since = control_time_s
                        if self.stage=='review_out' and not opt.automatic_return:
                            self.stage, self.completed = 'complete', True
                        self.events.append(dict(kind='length_goal_observed', time_s=control_time_s,
                            goal_delta_m=self.goal_m-self.baseline, observed_delta_m=coordinate-self.baseline,
                            velocity_m_s=self.velocity_m_s, position_joints=self.position))
            if self.tracking_blocked_since is not None and control_time_s-self.tracking_blocked_since>opt.maximum_tracking_wait_s:
                self.fault = 'measured shape tracking did not recover within bounded handover'
            elif self.untrusted_since is not None and control_time_s-self.untrusted_since > opt.maximum_untrusted_s:
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
            forced_step=None if self.forced_step is None else dict(self.forced_step),
            acquisition_reason=self.acquisition_reason,
            acquisition_samples=len(self.acquisition_samples),
            acquisition_metrics=self.acquisition_metrics,
            phase_rad=self.phase, position_joints=self.position, bearing_bias=self.bias,
            bearing_error_m=self.bearing_error_m, goal_m=self.goal_m, baseline_m=self.baseline,
            goal_delta_m=None if self.goal_m is None else self.goal_m-self.baseline,
            observation=self.observation, velocity_m_s=self.velocity_m_s, enabled=self.enabled.tolist(),
            goal_speed_tolerance_m_s=self.goal_speed_tolerance_m_s,
            untrusted_since_s=self.untrusted_since,
            release_below_peak_m=self.release_below_peak_m,
            tracking_blocked_since_s=self.tracking_blocked_since,
            tracking_error_deg=self.tracking_error_deg, actual_enabled_turn_deg=self.actual_enabled_turn_deg,
            retained_fold_joints=(np.flatnonzero(np.isfinite(self.fold_capture_position) &
                (abs(self.position-self.fold_capture_position)<self.options.fold_retention_distance_joints))+1).tolist(),
            completed=self.completed, fault=self.fault, options=asdict(self.options),
            profile_position_bounds_joints=list(self.position_bounds), compliance=False,
            passive_reference_end='head' if self.reference_direction>0 else 'tail',
            completion_means='encoder_length_and_speed_goal_not_world_truth_acceptance',
            inputs=['angles_deg', 'feedback_time_s', 'control_time_s'])
