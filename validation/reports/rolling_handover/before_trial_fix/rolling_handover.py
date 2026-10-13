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

    def advance(self, *args, **kwargs):
        if self.fault:
            kwargs['freeze_targets'] = True
        result = super().advance(*args, **kwargs)
        if self.phase_hold:
            raise RuntimeError('Normal rolling strategy must never pause phase')
        return result

    def snapshot(self):
        result = super().snapshot()
        result.update(handover_strategy='rolling_angle_geometry', emergency_target_freeze=bool(self.fault or self.resize_fault))
        result['observer_configuration'] = self.observer.configuration()
        return result
