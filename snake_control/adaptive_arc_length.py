"""Angle-only active-length supervisor: stable 6/8, transactional 7.

No model, pipe radius, contact, root orientation or torque is a controller input.
The fitted radius is a BODY-shape proxy; it cannot measure pipe diameter.
"""
from dataclasses import dataclass, replace, asdict
import math
import numpy as np
from .rolling_arc import ArcParameters
from .fixed_section_compliance import FixedSectionCompliance
from .motor_feedback_sliding import MotorFeedbackSliding, MotorSlidingOptions
from .motor_shape_observer import MotorShapeObserver, MotorObserverOptions, PhaseShapeReference
from .section_geometry import local_section


@dataclass(frozen=True)
class HalfArcParameters(ArcParameters):
    mechanical_lengths_m: tuple = (.0645,)*21

    def __post_init__(self):
        super().__post_init__()
        if self.amplitude_deg != 60. or self.direction != 1 or self.omega_rad_s != .6:
            raise ValueError('This experiment uses a fixed pi budget and the forward 0.6rad/s calibration')
        if len(self.mechanical_lengths_m) != self.joints+1 or not np.isfinite(self.mechanical_lengths_m).all() or min(self.mechanical_lengths_m) <= 0:
            raise ValueError('One positive physical length per backbone segment required')

    def targets(self, phase):
        if not math.isfinite(phase):
            raise ValueError('Finite phase required')
        lengths = np.asarray(self.mechanical_lengths_m)
        module = .5*(lengths[:-1]+lengths[1:])
        target = np.zeros(self.joints)
        active = np.flatnonzero(self.enabled)
        for parity, factor in ((0, math.cos(phase)), (1, math.sin(phase))):
            family = active[active % 2 == parity]
            target[family] = math.pi*module[family]/module[family].sum()*factor
        return target


@dataclass(frozen=True)
class LengthOptions:
    expansion_radius_ratio: float = 1.10
    expansion_turn_deg: float = 175.
    contraction_radius_ratio: float = .90
    contraction_turn_deg: float = 185.
    filter_s: float = .5
    persistence_s: float = 1.
    maximum_circle_rms_m: float = .012
    cooldown_s: float = 5.


def circle_shape(section):
    xy = section['projected'][1:-1]
    matrix = np.column_stack([2*xy, np.ones(len(xy))])
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.sum(xy*xy, axis=1), rcond=None)
    if rank < 3:
        raise ValueError('Circle fit degenerate')
    radius = math.sqrt(max(0., solution[2]+solution[:2]@solution[:2]))
    distance = np.linalg.norm(xy-solution[:2], axis=1)
    return dict(radius_m=radius, circle_rms_m=float(np.sqrt(np.mean((distance-radius)**2))),
                turn_deg=math.degrees(section['total_turn']))


class AdaptiveArcLength(MotorFeedbackSliding):
    def __init__(self, parameters, lengths, references, resize=True, sliding=False, feedback_period_s=.2,
                 length_options=LengthOptions(), feasibility_trial=True, **kwargs):
        self.references = dict(references)
        width = parameters.last_joint-parameters.first_joint+1
        if width not in (6, 8):
            raise ValueError('Start at a stable six/eight-joint window')
        reference = self.references.get(width)
        if reference is None:
            # Calibration observes quality/features but cannot migrate.
            if resize or sliding:
                raise ValueError('A frozen reference for the starting width is required')
            reference = PhaseShapeReference([[0., 0., 0.]]*24, {'calibration_only': True})
        super().__init__(parameters, lengths, reference, execute=False, feedback_period_s=feedback_period_s,
                         allowed_widths=(6, 7, 8), **kwargs)
        self.resize_automatic, self.sliding_automatic = bool(resize), bool(sliding)
        self.length_options = length_options
        self.length_observation = None
        self.length_filtered = None
        self.length_since, self.length_direction = None, 0
        self.resize_fault = None
        self.feasibility_trial = bool(feasibility_trial)
        self.eight_cycle_start = parameters.initial_phase_rad if width == 8 else None
        self.eight_cycle_good, self.eight_cycle_samples = True, 0
        self.six_trial = None
        self.prefer_feasible_six = False

    @property
    def automatic(self):
        return self.resize_automatic or self.sliding_automatic

    @automatic.setter
    def automatic(self, value):
        self.resize_automatic = self.sliding_automatic = bool(value)

    def _new_observer(self):
        width = self.window[1]-self.window[0]+1
        reference = self.references.get(width, PhaseShapeReference([[0., 0., 0.]]*24, {'transition_only': True}))
        self.observer = MotorShapeObserver(self.lengths, reference, options=MotorObserverOptions(
            maximum_feedback_gap_s=max(.1, 2.5*self.feedback_period_s)), width=width)
        self.observation = None
        self.length_filtered, self.length_since, self.length_direction = None, None, 0
        self.length_observation = None

    def _request_resize(self, wanted, time_s, measured):
        first, last = self.window
        width = last-first+1
        final = (first-1, last+1) if wanted > width else (first+1, last-1)
        if final[0] < 4 or final[1] > self.p.joints-3 or wanted not in self.references:
            self.length_observation['state'] = 'blocked_end_or_missing_reference'
            return
        choices = [(first-1, last), (first, last+1)] if wanted > width else [(first+1, last), (first, last-1)]
        # Choose the changed axis nearest its zero phase; neither side is a
        # preprogrammed migration. Stable final centre remains unchanged.
        def zero_distance(w):
            changed = next(iter(set(range(first, last+1)) ^ set(range(w[0], w[1]+1))))
            return abs(math.cos(self.phase) if changed % 2 else math.sin(self.phase))
        step = min(choices, key=zero_distance)
        self.pending = dict(kind='resize', to=list(step), final=list(final), origin=list(self.window),
                            time_s=time_s, ready_since=None, hold_since=None, wanted_width=wanted)
        self.events.append(dict(kind='resize_request', time_s=time_s, old=list(self.window),
            to=list(final), first_step=list(step), evidence=dict(self.length_observation)))

    def _observe_length(self, angles, time_s):
        o = self.observation
        if o is None or not o['valid']:
            self.length_since, self.length_direction, self.length_filtered = None, 0, None
            self.length_observation = dict(state='unobservable', reasons=[] if o is None else o['reasons'])
            return
        q = np.radians(angles)
        actual = circle_shape(local_section(q, self.lengths, *self.window))
        nominal = q.copy()
        nominal[self.enabled] = replace(self.p, first_joint=self.window[0], last_joint=self.window[1]).targets(self.feedback_phase)[self.enabled]
        reference = circle_shape(local_section(nominal, self.lengths, *self.window))
        ratio = actual['radius_m']/reference['radius_m']
        values = np.array([ratio, actual['turn_deg']])
        if self.length_filtered is None:
            self.length_filtered = values
        else:
            self.length_filtered += (1-math.exp(-self.feedback_period_s/self.length_options.filter_s))*(values-self.length_filtered)
        r, turn = self.length_filtered
        width = self.window[1]-self.window[0]+1
        option = self.length_options
        reliable = actual['circle_rms_m'] <= option.maximum_circle_rms_m
        expand_turn = 155. if self.prefer_feasible_six else option.expansion_turn_deg
        direction = (1 if width == 6 and reliable and r >= option.expansion_radius_ratio and turn <= expand_turn else
                     -1 if width == 8 and reliable and r <= option.contraction_radius_ratio and turn >= option.contraction_turn_deg else 0)
        self.length_observation = dict(state='retain', actual_body_circle=actual, nominal_body_circle=reference,
            filtered_radius_ratio=float(r), filtered_turn_deg=float(turn), direction=direction,
            observes_pipe_radius=False, observes_support=False)
        if not direction:
            self.length_since, self.length_direction = None, 0
            return
        if direction != self.length_direction:
            self.length_since, self.length_direction = self.last_time, direction
        elapsed = self.last_time-self.length_since
        self.length_observation.update(state='confirming', persistent_s=elapsed)
        if elapsed+1e-9 >= option.persistence_s:
            self.length_observation['state'] = 'recommend_expand' if direction > 0 else 'recommend_contract'
            self._request_resize(width+2*direction, time_s, angles)

    def _handover(self, angles, time_s):
        request = self.pending
        if request.get('kind') != 'resize':
            return super()._handover(angles, time_s)
        new = tuple(request['to'])
        p = replace(self.p, first_joint=new[0], last_joint=new[1])
        entering, exiting = np.flatnonzero(p.enabled & ~self.enabled), np.flatnonzero(self.enabled & ~p.enabled)
        changed = np.r_[entering, exiting]
        if len(changed) != 1:
            raise RuntimeError('Resize changes one boundary joint at a time')
        factor = math.cos(self.phase) if changed[0] % 2 == 0 else math.sin(self.phase)
        if not self.phase_hold and abs(factor) <= self.options.phase_zero_tolerance:
            self.phase_hold = True
            request['hold_since'] = time_s
            self.events.append(dict(kind='phase_hold', time_s=time_s, operation='resize', phase_rad=self.phase,
                                    changed=(changed+1).tolist()))
        if self.phase_hold and self.fresh_feedback:
            q = np.radians(angles)
            good, geometry, _ = self.observer.geometry_observer.geometry(q, new)
            preload = self.targets.copy()
            preload[entering], preload[~p.enabled] = q[entering], q[~p.enabled]
            core = FixedSectionCompliance(p, self.lengths, self.section_options)
            command_good = core._check_goal(preload)[0]
            evidence = self.observation is not None and self.observation['valid']
            ready = (good and command_good and evidence and self.angle_velocity_available and
                np.max(abs(angles[changed])) <= self.options.switch_angle_deg and
                np.max(abs(self.angle_velocity_deg_s[changed])) <= self.options.switch_velocity_deg_s and
                time_s-request['hold_since'] >= self.options.minimum_hold_s)
            self.last_gate = dict(time_s=time_s, candidate_geometry=geometry, command_open=bool(command_good),
                changed_joint=(changed+1).tolist(), changed_angle_deg=angles[changed].tolist(),
                changed_velocity_deg_s=self.angle_velocity_deg_s[changed].tolist(), evidence_valid=bool(evidence),
                unloading_observed=False)
            if ready:
                if request['ready_since'] is None:
                    request['ready_since'] = self.last_time
                if self.last_time-request['ready_since']+1e-9 >= self.options.stable_feedback_s:
                    old = self.window
                    core.targets, core.previous_axis = preload.copy(), self.core.previous_axis
                    self.core, self.window, self.enabled, self.targets = core, new, p.enabled.copy(), preload
                    final = new == tuple(request['final'])
                    self.events.append(dict(kind='resize_complete' if final else 'resize_step', time_s=time_s,
                        old=list(old), to=list(new), final=request['final'], gate=self.last_gate))
                    self.phase_hold = False
                    self._new_observer()
                    if final:
                        self.pending = None
                        self.cooldown_until = time_s+self.length_options.cooldown_s
                        if self.feasibility_trial:
                            if len(range(new[0], new[1]+1)) == 8:
                                self.eight_cycle_start = self.phase
                                self.eight_cycle_good, self.eight_cycle_samples = True, 0
                            else:
                                self.six_trial = dict(start_phase=self.phase, good=True, samples=0)
                                self.eight_cycle_start = None
                    else:
                        request.update(to=request['final'], ready_since=None, hold_since=None)
            else:
                request['ready_since'] = None
        if self.pending is not None and time_s-request['time_s'] >= self.options.handover_timeout_s:
            self.events.append(dict(kind='resize_reject', time_s=time_s, actual_window=list(self.window),
                                    reason='new angle/shape gate timed out', transaction=dict(request), gate=self.last_gate))
            self.pending, self.phase_hold = None, False
            if self.enabled.sum() == 7:
                self.resize_fault = 'incomplete resize; holding actual seven-joint state'
            self.cooldown_until = time_s+self.length_options.cooldown_s

    def advance(self, angles_deg, feedback_time_s, control_time_s, **kwargs):
        if self.resize_fault:
            kwargs['freeze_targets'] = True
        result = super().advance(angles_deg, feedback_time_s, control_time_s, **kwargs)
        if self.feasibility_trial and self.fresh_feedback and self.pending is None:
            good, geometry, _ = self.observer.geometry_observer.geometry(np.radians(angles_deg), self.window)
            if self.enabled.sum() == 8 and self.eight_cycle_start is not None:
                self.eight_cycle_good &= good
                self.eight_cycle_samples += 1
            if self.six_trial is not None:
                self.six_trial['good'] &= good
                self.six_trial['samples'] += 1
                if self.phase-self.six_trial['start_phase'] >= 2*math.pi:
                    self.prefer_feasible_six = bool(self.six_trial['good'] and self.six_trial['samples'] >= 40)
                    self.events.append(dict(kind='six_feasibility_review', time_s=control_time_s,
                        accepted=self.prefer_feasible_six, measured_only=True, observes_support=False,
                        actual_geometry=geometry, trial=dict(self.six_trial)))
                    if not self.prefer_feasible_six:
                        self.resize_fault = 'six-joint geometry trial failed'
                    self.six_trial = None
        if (self.fresh_feedback and self.pending is None and control_time_s >= self.cooldown_until and
                self.fault is None and self.resize_fault is None and not kwargs.get('freeze_targets', False)):
            if self.resize_automatic and self.six_trial is None:
                self._observe_length(np.asarray(angles_deg), control_time_s)
            if (self.feasibility_trial and self.resize_automatic and self.pending is None and self.enabled.sum() == 8 and
                    self.eight_cycle_start is not None and self.phase-self.eight_cycle_start >= 2*math.pi and
                    self.eight_cycle_good and self.eight_cycle_samples >= 40 and self.observation is not None and
                    self.observation['valid'] and abs(geometry['turn_deg']-180.) <= 10.):
                self.length_observation = dict(state='try_shorter_feasible_arc', direction=-1,
                    complete_phase_cycles=(self.phase-self.eight_cycle_start)/(2*math.pi),
                    actual_geometry=geometry, observes_pipe_radius=False, observes_support=False)
                self._request_resize(6, control_time_s, np.asarray(angles_deg))
            if self.pending is None and self.sliding_automatic and self.observation is not None and self.observation['suggested_window']:
                self.pending = dict(kind='slide', to=self.observation['suggested_window'], time_s=control_time_s,
                                    ready_since=None, hold_since=None)
                self.events.append(dict(kind='prepare', time_s=control_time_s, old=list(self.window),
                    to=self.pending['to'], observation=self.observation))
        return result

    def snapshot(self):
        result = super().snapshot()
        result.update(active_count=int(self.enabled.sum()), length_observation=self.length_observation,
            length_options=asdict(self.length_options), resize_automatic=self.resize_automatic,
            sliding_automatic=self.sliding_automatic, resize_fault=self.resize_fault)
        result.update(feasibility_trial=self.feasibility_trial, prefer_feasible_six=self.prefer_feasible_six,
                      six_trial=self.six_trial, eight_cycle_start=self.eight_cycle_start)
        return result
