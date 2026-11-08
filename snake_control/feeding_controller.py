"""Trajectory-only adapter; state machine and observation stay inherited."""
from dataclasses import dataclass, replace
import numpy as np
from .continuous_feeding import ContinuousFeeding, ContinuousFeedingOptions
from .manual_feeding_observer import ManualFeedingDirectionOptions
from .feeding_curve import FeedingCurveBuilder


class _CurveTemplateHelper:
    def __init__(self, original, owner):
        self.original, self.owner = original, owner

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'original'), name)

    def template(self, window, phase, amplitude, bias=0.):
        plan = self.owner.curve_builder.template(window, phase, amplitude, bias)
        self.owner.last_curve = plan
        return plan.targets(self.owner.lengths)


def curve_feeding_class(base):
    """Works with library or single-file embedded controller, unchanged above.

    Super initializes the old bookkeeping; its unused seed is immediately
    replaced before the object is exposed. Every returned motion reference,
    including preparation, comes through the curve integral thereafter.
    """
    if getattr(base, 'trajectory_backend', None) == 'stitched_curve_integral':
        return base

    class CurveContinuousFeeding(base):
        trajectory_backend = 'stitched_curve_integral'
        def __init__(self, *args, seed_curve=None,phase_rad=0., **kwargs):
            super().__init__(*args, **kwargs)
            self.curve_builder = FeedingCurveBuilder(self.lengths)
            self.phase=phase_rad
            if seed_curve is not None:
                self.curve_builder.seed(seed_curve,self.options.amplitude_deg)
            self.last_curve = None
            self.helper = _CurveTemplateHelper(self.helper, self)
            self.targets = self.helper.template(self.origin, self.phase, self.options.amplitude_deg)

        def profile(self, position, bias):
            plan = self.curve_builder.travelling(self.origin, position, self.phase, bias,
                self.options, self.minimum_bin, self.maximum_bin)
            self.last_curve = plan
            return plan.targets(self.lengths), plan.focus, plan.cell, plan.fraction

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
