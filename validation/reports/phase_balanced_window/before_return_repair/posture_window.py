"""Select both boundaries of one open rolling arc from motor-angle feedback.

No contact, world orientation, radius, qvel or measured torque enters here.
The phase reference for an uncalibrated width is a geometric transfer model,
not a measurement of gravity or evidence of support.
"""
import copy
from dataclasses import asdict, dataclass, replace
import math
import numpy as np

from .adaptive_arc_length import circle_shape
from .angle_shape_observer import AngleShapeObserver
from .fixed_section_compliance import FixedSectionCompliance
from .motor_feedback_sliding import MotorFeedbackSliding
from .motor_shape_observer import MotorObserverOptions, bending_features
from .rolling_shape_observer import RollingShapeObserver
from .section_geometry import forward, local_section
from .phase_balanced_count import PhaseBalancedCount


@dataclass(frozen=True)
class PostureWindowOptions:
    count_hysteresis_joints: float = .5
    expansion_turn_deg: float = 175.
    contraction_turn_deg: float = 185.
    circle_rms_max_m: float = .012
    persistence_s: float = 1.
    cooldown_s: float = 5.
    trial_grace_s: float = 1.
    trial_review_s: float = 5.
    nominal_joint_margin_rad: float = .01
    shorter_trial_count_margin_joints: float = .25
    shorter_trial_turn_deg: float = 180.


def nominal_angles(lengths, window, phase):
    """Distribute each alternating axis family's pi budget by module length."""
    lengths = np.asarray(lengths, float)
    module = .5*(lengths[:-1]+lengths[1:])
    indices = np.arange(window[0]-1, window[1])
    q = np.zeros(len(module))
    for parity, factor in ((0, math.cos(phase)), (1, math.sin(phase))):
        family = indices[indices % 2 == parity]
        if not len(family):
            raise ValueError('A rolling section requires both axis families')
        q[family] = math.pi*module[family]/module[family].sum()*factor
    return q


def mechanically_feasible(lengths, window, margin=.01):
    """A geometric boundary and full-cycle joint-limit check, not a width list."""
    n = len(lengths)-1
    if window[0] < 2 or window[1] >= n or window[1]-window[0]+1 < 3:
        return False
    try:
        peak = np.maximum(abs(nominal_angles(lengths, window, 0.)),
                          abs(nominal_angles(lengths, window, math.pi/2)))
        return bool(peak.max() <= math.pi/2-margin)
    except ValueError:
        return False


class GeneralPhaseReference:
    """Reuse any available calibration; transfer its residual to other widths.

    There is no stable-width whitelist. Unseen widths use their own nominal
    geometry plus an interpolated (or nearest) frozen calibration residual.
    Transfer across widths is an explicit assumption to audit in simulation.
    """
    def __init__(self, lengths, width, calibrated):
        self.lengths = np.asarray(lengths, float)
        self.width = width
        self.calibrated = calibrated
        self.helper = AngleShapeObserver(lengths)
        self.cache = {}
        self.metadata = dict(width=width, method='nominal geometry plus frozen residual',
                             calibrated_widths=sorted(calibrated),
                             transferred=width not in calibrated, online_learning=False)

    def ideal(self, width, phase, first):
        # Use the same starting parity, with enough space for boundary modules.
        start = 2 if first % 2 == 0 else 3
        if start+width-1 >= len(self.lengths)-1:
            start = 2
        window = (start, start+width-1)
        q = nominal_angles(self.lengths, window, phase)
        _, _, section = self.helper.geometry(q, window)
        points = forward(q, self.lengths)
        arc = section['points'][1:-1]
        down = (arc[0]+arc[-1])/2-arc.mean(axis=0)
        down /= np.linalg.norm(down)
        crest = self.helper.crest(points, window, section, down)['joint_coordinate']
        f = bending_features(np.degrees(q), self.lengths, window)
        center = sum(window)/2
        return np.array([crest-center, f['bending_center_joint']-center,
                         f['tail_boundary_turn_deg']-f['head_boundary_turn_deg']])

    def at(self, phase, first):
        if self.width in self.calibrated:
            return self.calibrated[self.width].at(phase, first)
        key = (round(phase % (2*math.pi), 8), first % 2)
        if key not in self.cache:
            widths = sorted(self.calibrated)
            residuals = np.array([self.calibrated[w].at(phase, first)-self.ideal(w, phase, first)
                                  for w in widths])
            residual = np.array([np.interp(self.width, widths, residuals[:, j]) for j in range(3)])
            self.cache[key] = self.ideal(self.width, phase, first)+residual
        return self.cache[key].copy()

    def to_dict(self):
        return dict(metadata=self.metadata,
                    calibrations={str(w): r.to_dict() for w, r in self.calibrated.items()})


class PostureWindow(MotorFeedbackSliding):
    """Adjacent position/length selection with continuous rolling handover."""
    def __init__(self, parameters, lengths, references, feedback_period_s=.2,
                 posture_options=PostureWindowOptions(), **kwargs):
        if not references:
            raise ValueError('At least one frozen normal-rolling reference is required')
        self.references = dict(references)
        self.posture_options = posture_options
        window = (parameters.first_joint, parameters.last_joint)
        if not mechanically_feasible(lengths, window, posture_options.nominal_joint_margin_rad):
            raise ValueError('Initial arc cannot meet the full-cycle mechanical limits')
        for phase in (0., math.pi/2):
            if not np.allclose(parameters.targets(phase), nominal_angles(lengths, window, phase), atol=1e-10):
                raise ValueError('Use length-normalized half-arc parameters for every width')
        width = window[1]-window[0]+1
        reference = GeneralPhaseReference(lengths, width, references)
        super().__init__(parameters, lengths, reference, execute=False,
                         allowed_widths=range(3, parameters.joints-1),
                         feedback_period_s=feedback_period_s, **kwargs)
        self.selection = None
        self.filtered_count = None
        self.intent = self.intent_since = None
        self.trial = None
        self.resize_fault = None
        self.adaptive_enabled = True
        self._new_observer()

    @property
    def automatic(self):
        return self.adaptive_enabled

    @automatic.setter
    def automatic(self, value):
        self.adaptive_enabled = bool(value)
        if not self.adaptive_enabled:
            self.pending = None
            self.intent = self.intent_since = None

    def _new_observer(self):
        width = self.window[1]-self.window[0]+1
        reference = GeneralPhaseReference(self.lengths, width, self.references)
        self.observer = RollingShapeObserver(self.lengths, reference, width=width,
            options=MotorObserverOptions(maximum_feedback_gap_s=2.5*self.feedback_period_s))
        self.observation = None
        self.filtered_count = None
        self.count_history = PhaseBalancedCount(maximum_feedback_gap_s=2.5*self.feedback_period_s)
        self.count_observation = None
        self.intent = self.intent_since = None

    def _candidate(self, angles, window):
        if not mechanically_feasible(self.lengths, window, self.posture_options.nominal_joint_margin_rad):
            return dict(window=list(window), eligible=False, reason='mechanical boundary or peak joint limit')
        try:
            good, geometry, section = self.observer.geometry_observer.geometry(np.radians(angles), window)
            down, quality = self.observer.passive_down(forward(np.radians(angles), self.lengths), window, section)
            return dict(window=list(window), eligible=bool(good and down is not None), geometry=geometry,
                        end_source=quality['source'], reason=None if good and down is not None else 'shape or passive reference')
        except (ValueError, np.linalg.LinAlgError) as exc:
            return dict(window=list(window), eligible=False, reason=str(exc))

    def _observe_count(self, angles, time_s):
        """Collect even while a handover, trial or cooldown is in progress."""
        o = self.observation
        if not o or tuple(o['current_window']) != self.window:
            return
        if not o['valid']:
            self.count_history.reset()
            self.count_observation = dict(valid=False, reason='current shape not reliable')
            return
        width = self.window[1]-self.window[0]+1
        q = np.radians(angles)
        actual = circle_shape(local_section(q, self.lengths, *self.window))
        nominal = q.copy()
        nominal[self.enabled] = nominal_angles(self.lengths, self.window, self.feedback_phase)[self.enabled]
        reference = circle_shape(local_section(nominal, self.lengths, *self.window))
        radius_ratio = actual['radius_m']/reference['radius_m']
        turn_ratio = reference['turn_deg']/max(actual['turn_deg'], 1e-8)
        estimated_count = width*.5*(radius_ratio+turn_ratio)
        if actual['circle_rms_m'] > self.posture_options.circle_rms_max_m:
            self.count_history.reset()
            self.count_observation = dict(valid=False, reason='body circle fit not reliable', actual_body_circle=actual)
            return
        phase_statistics = self.count_history.update(estimated_count, actual['turn_deg'], self.feedback_phase, self.last_time)
        self.filtered_count = phase_statistics['balanced_count_joints']
        self.count_observation = dict(valid=True, time_s=self.last_time, window=list(self.window),
            raw_count_joints=estimated_count, actual_body_circle=actual, nominal_body_circle=reference,
            phase_statistics=phase_statistics, observes_support=False, observes_pipe_radius=False)

    def _select(self, angles, time_s):
        o, count = self.observation, self.count_observation
        if not o or not o['valid'] or 'down_head' not in o:
            self.intent = self.intent_since = None
            self.selection = dict(state='retain_untrusted_shape_or_down', observes_support=False)
            return
        width = self.window[1]-self.window[0]+1
        option = self.posture_options
        stats = count.get('phase_statistics', {}) if count and count['valid'] else {}
        ready = stats.get('ready', False)
        actual = None if not count else count.get('actual_body_circle')
        reference = None if not count else count.get('nominal_body_circle')
        delta_width = 0
        length_reason = 'multi_phase_coverage_incomplete' if not ready else 'no_persistent_count_error'
        if ready and self.filtered_count >= width+option.count_hysteresis_joints and stats['balanced_turn_deg'] <= option.expansion_turn_deg:
            delta_width = 1
            length_reason = 'multi_phase_arc_deficit'
        elif ready:
            strongly_compact = self.filtered_count <= width-option.count_hysteresis_joints and stats['balanced_turn_deg'] >= option.contraction_turn_deg
            shorter_trial = (stats['maximum_group_count_joints'] <= width+option.shorter_trial_count_margin_joints and
                             stats['balanced_turn_deg'] >= option.shorter_trial_turn_deg)
            first, last = self.window
            can_shorten = any(mechanically_feasible(self.lengths, w, option.nominal_joint_margin_rad)
                              for w in ((first+1, last), (first, last-1)))
            if (strongly_compact or shorter_trial) and can_shorten:
                delta_width = -1
                length_reason = 'multi_phase_compact_arc' if strongly_compact else 'guarded_adjacent_shorter_trial'
        offset = float(o['filtered_error_joints'])
        sign = 1 if offset > 0 else -1
        bending = o['filtered_feature_error']
        corroborated = sign*bending[0] >= self.observer.options.bending_threshold_joints or sign*bending[1] >= self.observer.options.boundary_threshold_deg
        migration = sign if abs(offset) >= self.observer.options.crest_threshold_joints and corroborated else 0
        first, last = self.window
        if delta_width > 0:
            windows = [(first-1, last), (first, last+1)]
            operation = 'expand'
        elif delta_width < 0:
            windows = [(first+1, last), (first, last-1)]
            operation = 'contract'
        elif migration:
            windows = [(first+migration, last+migration)]
            operation = 'slide'
        else:
            windows, operation = [], 'retain'
        candidates = [self._candidate(angles, w) for w in windows]
        desired_center = sum(self.window)/2+(offset if migration else 0.)
        for c in candidates:
            w = c['window']
            c['center_error_joints'] = abs(sum(w)/2-desired_center)
            c['score'] = c['center_error_joints']+.01*abs(c.get('geometry', {}).get('turn_deg', 180.)-180.)
        usable = [c for c in candidates if c['eligible']]
        best = min(usable, key=lambda c: c['score']) if usable else None
        self.selection = dict(state=operation if best else 'retain' if not windows else 'blocked_candidates',
            current_window=list(self.window), estimated_count_joints=None if not count else count.get('raw_count_joints'),
            filtered_count_joints=self.filtered_count, actual_body_circle=actual, nominal_body_circle=reference,
            phase_statistics=stats, length_reason=length_reason,
            migration_direction=migration, conditional_crest_error_joints=offset,
            candidates=candidates, chosen=None if best is None else best['window'],
            observes_pipe_radius=False, observes_support=False)
        # A length decision persists across a phase-dependent choice of side.
        # The fresh candidate and continuity gate decide which side can switch.
        intent = None if best is None else (operation, width+delta_width) if delta_width else (operation, *best['window'])
        if intent != self.intent:
            self.intent, self.intent_since = intent, time_s
        if intent is not None and time_s-self.intent_since+1e-8 >= option.persistence_s:
            self.pending = dict(kind='slide' if operation == 'slide' else 'resize',
                                operation=operation, to=best['window'], time_s=time_s,
                                ready_since=None, evidence=copy.deepcopy(self.selection))
            self.events.append(dict(kind='prepare', time_s=time_s, old=list(self.window),
                                    to=best['window'], operation=operation, selection=copy.deepcopy(self.selection)))
            self.selection['state'] = 'waiting_moving_handover'

    def _handover(self, angles, time_s):
        request, new = self.pending, tuple(self.pending['to'])
        if self.fresh_feedback:
            p = replace(self.p, first_joint=new[0], last_joint=new[1])
            entering = np.flatnonzero(p.enabled & ~self.enabled)
            exiting = np.flatnonzero(self.enabled & ~p.enabled)
            changed = len(entering)+len(exiting)
            if (request['kind'] == 'slide' and (len(entering), len(exiting)) != (1, 1)) or (request['kind'] != 'slide' and changed != 1):
                raise RuntimeError('Only adjacent position or single-boundary length changes are allowed')
            candidate = self._candidate(angles, new)
            q = np.radians(angles)
            preload = self.targets.copy()
            preload[entering], preload[~p.enabled] = q[entering], q[~p.enabled]
            core = FixedSectionCompliance(p, self.lengths, self.section_options)
            core.targets, core.previous_axis = preload.copy(), self.core.previous_axis
            preview_core = copy.deepcopy(core)
            preview = preview_core.update(q, self.phase, time_s, .02)
            step_deg = float(np.degrees(abs(preview[p.enabled]-preload[p.enabled])).max())
            compatible = bool(np.max(abs(angles[exiting])) <= self.options.switch_angle_deg) if len(exiting) else abs(math.cos(self.phase) if entering[0] % 2 == 0 else math.sin(self.phase)) <= .1
            current_trusted = self.observation is not None and self.observation['valid'] and 'down_head' in self.observation
            evidence = request.get('rollback', False) or current_trusted
            shared = self.enabled & p.enabled
            self.last_gate = dict(time_s=time_s, candidate_geometry=candidate.get('geometry'),
                entering_joint=(entering+1).tolist(), exiting_joint=(exiting+1).tolist(),
                entering_angle_deg=angles[entering].tolist(), exiting_angle_deg=angles[exiting].tolist(),
                exiting_velocity_deg_s=self.angle_velocity_deg_s[exiting].tolist(),
                command_open=bool(core._check_goal(preload)[0]), next_command_open=bool(preview_core._check_goal(preview)[0]),
                next_target_step_deg=step_deg, phase_compatible=compatible, evidence_valid=bool(evidence),
                shared_preload_jump_rad=float(abs(preload[shared]-self.targets[shared]).max()),
                unloading_observed=False, phase_paused=False)
            ready = (candidate['eligible'] and evidence and compatible and self.angle_velocity_available and
                     self.last_gate['command_open'] and self.last_gate['next_command_open'] and preview_core.status == 'running' and
                     step_deg <= self.section_options.target_rate_deg_s*.02+1e-8)
            if ready:
                old = self.window
                observation = copy.deepcopy(self.observation)
                self.core, self.window, self.enabled, self.targets = core, new, p.enabled.copy(), preload
                kind = 'complete' if request['kind'] == 'slide' else 'resize_complete'
                self.events.append(dict(kind=kind, operation=request['operation'], time_s=time_s,
                    old=list(old), to=list(new), phase_rad=self.phase, observation=observation,
                    gate=copy.deepcopy(self.last_gate), selection=request['evidence'], rollback=request.get('rollback', False)))
                self.pending = None
                self.cooldown_until = time_s+self.posture_options.cooldown_s
                self.fault = None
                self.trial = None if request.get('rollback') else dict(old=list(old), new=list(new),
                    start_time_s=time_s, bad_samples=0, samples=0, good=True)
                self._new_observer()
        if self.pending is not None and time_s-request['time_s'] >= self.options.handover_timeout_s:
            self.events.append(dict(kind='reject', time_s=time_s, transaction=copy.deepcopy(request), gate=self.last_gate))
            if request.get('rollback'):
                self.resize_fault = 'shape rollback could not meet the moving gate'
            self.pending = None
            self.intent = self.intent_since = None
            self.cooldown_until = time_s+self.posture_options.cooldown_s

    def _review_trial(self, angles, time_s):
        if not self.trial or not self.fresh_feedback:
            return
        trial = self.trial
        good, geometry, _ = self.observer.geometry_observer.geometry(np.radians(angles), self.window)
        trial['samples'] += 1
        age = time_s-trial['start_time_s']
        if age >= self.posture_options.trial_grace_s:
            trial['good'] &= good
            trial['bad_samples'] = 0 if good else trial['bad_samples']+1
        if trial['bad_samples'] >= 2 or self.fault:
            old = trial['old']
            kind = 'slide' if old[1]-old[0] == self.window[1]-self.window[0] else 'resize'
            self.pending = dict(kind=kind, operation='rollback', to=old, time_s=time_s,
                                rollback=True, evidence=dict(actual_geometry=geometry, trial=copy.deepcopy(trial)))
            self.events.append(dict(kind='rollback_request', time_s=time_s, window=list(self.window), trial=copy.deepcopy(trial)))
            self.trial = None
        elif age >= self.posture_options.trial_review_s:
            self.events.append(dict(kind='shape_trial_complete', time_s=time_s, trial=copy.deepcopy(trial),
                                    observes_support=False, full_cycle_claim=False))
            self.trial = None

    def advance(self, angles_deg, feedback_time_s, control_time_s, commanded_phase_rad=None, **kwargs):
        rolling_back = self.pending is not None and self.pending.get('rollback', False)
        if self.resize_fault or (self.fault and not rolling_back):
            kwargs['freeze_targets'] = True
        result = super().advance(angles_deg, feedback_time_s, control_time_s, commanded_phase_rad, **kwargs)
        if self.fresh_feedback:
            self._observe_count(np.asarray(angles_deg), control_time_s)
        self._review_trial(np.asarray(angles_deg), control_time_s)
        if (self.automatic and self.fresh_feedback and self.pending is None and self.trial is None and
                control_time_s >= self.cooldown_until and not self.fault and not self.resize_fault and
                not kwargs.get('freeze_targets', False)):
            self._select(np.asarray(angles_deg), control_time_s)
        if self.phase_hold:
            raise RuntimeError('Normal posture-window exchanges must keep rolling')
        return result

    def snapshot(self):
        result = super().snapshot()
        result.update(active_count=int(self.enabled.sum()), selection=copy.deepcopy(self.selection),
                      length_observation=copy.deepcopy(self.selection), trial=copy.deepcopy(self.trial),
                      resize_fault=self.resize_fault, posture_options=asdict(self.posture_options),
                      observer_configuration=self.observer.configuration(),
                      automatic=self.automatic,
                      count_observation=copy.deepcopy(self.count_observation),
                      stable_width_whitelist=None, preferred_count=None,
                      controller_revision='phase_balanced_count_v2')
        return result
