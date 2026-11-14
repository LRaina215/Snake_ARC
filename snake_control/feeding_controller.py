"""Continuous curve targets and their handoff through the shared feeder."""
from dataclasses import dataclass, replace
import numpy as np
from .continuous_feeding import ContinuousFeeding, ContinuousFeedingOptions
from .manual_feeding_observer import ManualFeedingDirectionOptions
from .feeding_curve import FeedingCurveBuilder


class _CurveTemplateHelper:
    def __init__(self, owner):
        self.owner = owner

    def mask(self, window):
        joints = np.arange(1, self.owner.curve_builder.n+1)
        return (joints >= window[0]) & (joints <= window[1])

    def template(self, window, phase, amplitude, bias=0.):
        plan = self.owner.curve_builder.plan(0., phase, amplitude)
        self.owner.last_curve = plan
        return plan.targets(self.owner.lengths)


def curve_feeding_class(base):
    """Use the shared motion state machine with a continuous curve reference."""
    if getattr(base, 'trajectory_backend', None) == 'continuous_curve_shift':
        return base

    class CurveContinuousFeeding(base):
        trajectory_backend = 'continuous_curve_shift'

        def __init__(self, lengths, options=ContinuousFeedingOptions(), *, seed_curve=None, phase_rad=0.):
            window = (options.first_joint, options.last_joint)
            self.curve_builder = FeedingCurveBuilder(
                lengths, window, options.amplitude_deg, phase_rad, seed_curve)
            self.last_curve = None
            super().__init__(lengths, options, geometry=_CurveTemplateHelper(self))
            self.phase = phase_rad
            self.position_bounds = self.curve_builder.position_bounds
            self.targets, focus, _, _ = self.profile(0., 0.)
            self.enabled = self.helper.mask(focus)
            self.window = focus

        def profile(self, position, bias):
            plan = self.curve_builder.plan(position, self.phase, self.options.amplitude_deg)
            self.last_curve = plan
            return plan.targets(self.lengths), plan.focus, plan.cell, plan.fraction

        def required_mask(self, focus, angles, position):
            required = super().required_mask(focus, angles, position)
            if self.goal_kind == 'relative_length':
                # Keep both distal references free wherever the curve permits.
                # A short arm can then disappear without taking away the
                # longer arm's existing observation history.
                reserve = self.options.minimum_passive_reference_links
                if reserve:
                    required[:reserve] = False
                    required[-reserve:] = False
                measured_arc = self.observation.get('actual_arc_window')
                if measured_arc is not None:
                    required |= self.helper.mask(measured_arc)
            # The passive reference reserve can remove only buffer joints.
            return required | self.helper.mask(focus)

        def include_fold_joints(self, required, active, folds):
            if self.goal_kind != 'relative_length':
                return super().include_fold_joints(required, active, folds)
            # Neither tip is driven by the travelling curve. A bent free
            # tip must not keep the intervening straight links energized.
            joints = np.arange(1, self.n+1)
            return required | ((joints > 1) & (joints < self.n) & np.isin(joints, folds))

        def release_candidates(self, angles, required):
            if self.goal_kind != 'relative_length':
                return super().release_candidates(angles, required)
            # The trailing buffer and any measured upper arc are already in
            # required. Once the whole two-link interval is on a straight arm,
            # a confirmed near-zero angle is enough to let that arm hang.
            return ~self.helper.mask(self.last_curve.focus)

        def snapshot(self):
            row = super().snapshot()
            row['trajectory_backend'] = self.trajectory_backend
            row['curve_reference'] = None if self.last_curve is None else self.last_curve.as_dict()
            return row

    CurveContinuousFeeding.__name__ = 'CurveContinuousFeeding'
    return CurveContinuousFeeding


CurveContinuousFeeding = curve_feeding_class(ContinuousFeeding)


@dataclass(frozen=True)
class FeedingSettings:
    options: ContinuousFeedingOptions=ContinuousFeedingOptions(automatic_return=False)
    direction: ManualFeedingDirectionOptions=ManualFeedingDirectionOptions()


def create_feeding_controller(lengths,window,settings=None,*,amplitude_deg=None,phase_rad=0.,
                              seed_curve=None,measured_deg=None,enabled=None,speed_scale=1.):
    """The sole factory for manual transfer, switch preparation and final balancing."""
    settings=FeedingSettings() if settings is None else settings
    options=replace(settings.options,first_joint=window[0],last_joint=window[1],
        maximum_position_rate_joints_s=settings.options.maximum_position_rate_joints_s*speed_scale,
        target_rate_deg_s=settings.options.target_rate_deg_s*speed_scale)
    if amplitude_deg is not None:options=replace(options,amplitude_deg=amplitude_deg)
    core=CurveContinuousFeeding(lengths,options,seed_curve=seed_curve,phase_rad=phase_rad)
    core.length_observer.options=settings.direction
    if measured_deg is not None:core.targets=np.radians(measured_deg).copy()
    if enabled is not None:core.enabled=np.asarray(enabled,bool).copy()
    return core
