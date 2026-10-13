"""Moving window exchange; existing paused strategy is preserved separately.

No contact, gravity, pipe geometry or physical torque enters this supervisor.
Targets being continuous does not establish unloading or torque continuity.
"""
import copy
from dataclasses import replace
import math
import numpy as np
from .adaptive_arc_length import AdaptiveArcLength
from .fixed_section_compliance import FixedSectionCompliance
from .motor_shape_observer import MotorObserverOptions, PhaseShapeReference
from .rolling_shape_observer import RollingShapeObserver
from .motor_feedback_sliding import MotorFeedbackSliding


class RollingHandover(AdaptiveArcLength):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._new_observer()

    def _new_observer(self):
        width = self.window[1]-self.window[0]+1
        reference = self.references.get(width, PhaseShapeReference([[0., 0., 0.]]*24, {'transition_only': True}))
        self.observer = RollingShapeObserver(self.lengths, reference, width=width,
            options=MotorObserverOptions(maximum_feedback_gap_s=max(.1, 2.5*self.feedback_period_s)))
        self.observation = None
        self.length_filtered, self.length_since, self.length_direction = None, None, 0
        self.length_observation = None

    def _handover(self, angles, time_s):
        request, new = self.pending, tuple(self.pending['to'])
        resize = request.get('kind') == 'resize'
        if self.fresh_feedback:
            p = replace(self.p, first_joint=new[0], last_joint=new[1])
            entering = np.flatnonzero(p.enabled & ~self.enabled)
            exiting = np.flatnonzero(self.enabled & ~p.enabled)
            if (resize and len(entering)+len(exiting) != 1) or (not resize and (len(entering), len(exiting)) != (1, 1)):
                raise RuntimeError('Only adjacent boundary exchanges are allowed')
            q = np.radians(angles)
            good, geometry, _ = self.observer.geometry_observer.geometry(q, new)
            preload = self.targets.copy()
            preload[entering], preload[~p.enabled] = q[entering], q[~p.enabled]
            next_core = FixedSectionCompliance(p, self.lengths, self.section_options)
            next_core.targets, next_core.previous_axis = preload.copy(), self.core.previous_axis
            command_good = next_core._check_goal(preload)[0]
            # Preview on a copy: the real core must integrate this tick once.
            preview_core = copy.deepcopy(next_core)
            preview = preview_core.update(q, self.phase, time_s, .02)
            preview_good = preview_core.status == 'running' and preview_core._check_goal(preview)[0]
            step_deg = float(np.degrees(abs(preview[p.enabled]-preload[p.enabled])).max())
            compatible_phase = True
            if len(exiting):
                compatible_phase = bool(np.max(abs(angles[exiting])) <= self.options.switch_angle_deg)
            elif len(entering):
                # 0.1 gives a ~0.334s opportunity at omega=.6, unlike the old
                # ~0.05s window which could be missed by 200ms angle acquisition.
                factor = math.cos(self.phase) if entering[0] % 2 == 0 else math.sin(self.phase)
                compatible_phase = abs(factor) <= .1
            o = self.observation
            direction = new[0]-self.window[0]
            evidence = o is not None and o['valid'] and (resize or o['direction'] == direction)
            shared = self.enabled & p.enabled
            self.last_gate = dict(time_s=time_s, candidate_geometry=geometry, command_open=bool(command_good),
                next_command_open=bool(preview_good), next_target_step_deg=step_deg,
                entering_joint=(entering+1).tolist(), exiting_joint=(exiting+1).tolist(),
                entering_angle_deg=angles[entering].tolist(), exiting_angle_deg=angles[exiting].tolist(),
                entering_velocity_deg_s=self.angle_velocity_deg_s[entering].tolist(),
                exiting_velocity_deg_s=self.angle_velocity_deg_s[exiting].tolist(),
                phase_compatible=compatible_phase, evidence_valid=bool(evidence),
                shared_preload_jump_rad=float(abs(preload[shared]-self.targets[shared]).max()),
                feedback_age_s=time_s-self.last_time, unloading_observed=False,
                phase_paused=False, strategy='rolling_angle_geometry')
            ready = (good and command_good and preview_good and evidence and compatible_phase and
                     self.angle_velocity_available and step_deg <= self.section_options.target_rate_deg_s*.02+1e-8)
            if ready:
                old = self.window
                self.core, self.window, self.enabled, self.targets = next_core, new, p.enabled.copy(), preload
                final = not resize or new == tuple(request['final'])
                kind = 'complete' if not resize else 'resize_complete' if final else 'resize_step'
                self.events.append(dict(kind=kind, time_s=time_s, old=list(old), to=list(new),
                    gate=self.last_gate, phase_rad=self.phase, observation=dict(o), strategy='rolling_angle_geometry'))
                self._new_observer()
                if final:
                    self.pending = None
                    self.cooldown_until = time_s+self.length_options.cooldown_s
                    if resize and self.feasibility_trial:
                        if self.enabled.sum() == 8:
                            self.eight_cycle_start = self.phase
                            self.eight_cycle_good, self.eight_cycle_samples = True, 0
                        else:
                            self.six_trial = dict(start_phase=self.phase, good=True, samples=0)
                            self.eight_cycle_start = None
                    elif self.six_trial is not None:
                        # A position exchange changes the actual six-joint
                        # section. Review a complete cycle of this new window.
                        self.six_trial = dict(start_phase=self.phase, good=True, samples=0)
                        self.events.append(dict(kind='six_trial_restart_after_slide', time_s=time_s,
                                                window=list(new), phase_rad=self.phase))
                else:
                    request.update(to=request['final'], ready_since=None, hold_since=None)
        if self.pending is not None and time_s-request['time_s'] >= self.options.handover_timeout_s:
            self.events.append(dict(kind='resize_reject' if resize else 'reject', time_s=time_s,
                actual_window=list(self.window), transaction=dict(request), gate=self.last_gate,
                reason='No valid moving exchange before timeout'))
            self.pending = None
            if self.enabled.sum() == 7:
                self.resize_fault = 'incomplete moving resize; actual seven-joint state held'
            self.length_since, self.length_direction = None, 0
            self.cooldown_until = time_s+self.length_options.cooldown_s

    def advance(self, angles_deg, feedback_time_s, control_time_s, commanded_phase_rad=None, **kwargs):
        if commanded_phase_rad is not None:
            kwargs['commanded_phase_rad'] = commanded_phase_rad
        if self.fault or self.resize_fault:
            kwargs['freeze_targets'] = True
        if not self.resize_automatic:
            result = super().advance(angles_deg, feedback_time_s, control_time_s, **kwargs)
        else:
            # Same inherited length policy, with moving six-window sampling.
            result = MotorFeedbackSliding.advance(self, angles_deg, feedback_time_s, control_time_s, **kwargs)
            geometry = None
            if self.feasibility_trial and self.fresh_feedback:
                good, geometry, _ = self.observer.geometry_observer.geometry(np.radians(angles_deg), self.window)
                # Preserve the old eight-joint timing. Six-joint observations
                # must also count while a slide is WAITING and phase advances.
                if self.pending is None and self.enabled.sum() == 8 and self.eight_cycle_start is not None:
                    self.eight_cycle_good &= good
                    self.eight_cycle_samples += 1
                if self.enabled.sum() == 6 and self.six_trial is not None:
                    self.six_trial['good'] &= good
                    self.six_trial['samples'] += 1
                    if self.phase-self.six_trial['start_phase'] >= 2*math.pi and self.six_trial['samples'] >= 40:
                        self.prefer_feasible_six = bool(self.six_trial['good'])
                        self.events.append(dict(kind='six_feasibility_review', time_s=control_time_s,
                            accepted=self.prefer_feasible_six, measured_only=True, observes_support=False,
                            actual_geometry=geometry, trial=dict(self.six_trial)))
                        if not self.prefer_feasible_six:
                            self.resize_fault = 'six-joint actual geometry trial failed'
                        self.six_trial = None
            if (self.fresh_feedback and self.pending is None and control_time_s >= self.cooldown_until and
                    self.fault is None and self.resize_fault is None and not kwargs.get('freeze_targets', False)):
                if self.six_trial is None:
                    self._observe_length(np.asarray(angles_deg), control_time_s)
                if (self.feasibility_trial and self.pending is None and self.enabled.sum() == 8 and
                        self.eight_cycle_start is not None and self.phase-self.eight_cycle_start >= 2*math.pi and
                        self.eight_cycle_good and self.eight_cycle_samples >= 40 and self.observation is not None and
                        self.observation['valid'] and abs(geometry['turn_deg']-180.) <= 10.):
                    self.length_observation = dict(state='try_shorter_feasible_arc', direction=-1,
                        complete_phase_cycles=(self.phase-self.eight_cycle_start)/(2*math.pi), actual_geometry=geometry,
                        observes_pipe_radius=False, observes_support=False)
                    self._request_resize(6, control_time_s, np.asarray(angles_deg))
                if self.pending is None and self.sliding_automatic and self.observation is not None and self.observation['suggested_window']:
                    self.pending = dict(kind='slide', to=self.observation['suggested_window'], time_s=control_time_s,
                                        ready_since=None, hold_since=None)
                    self.events.append(dict(kind='prepare', time_s=control_time_s, old=list(self.window),
                                            to=self.pending['to'], observation=self.observation))
        if self.phase_hold:
            raise RuntimeError('Normal rolling strategy must never pause phase')
        return result

    def snapshot(self):
        result = super().snapshot()
        result.update(handover_strategy='rolling_angle_geometry', emergency_target_freeze=bool(self.fault or self.resize_fault))
        result['observer_configuration'] = self.observer.configuration()
        result['controller_revision'] = 'moving_six_trial_samples_include_pending_slide'
        return result
