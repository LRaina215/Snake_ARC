"""Trajectory-only adapter; state machine and observation stay inherited."""
from snake_control.continuous_feeding import ContinuousFeeding
from .curve_profile import FeedingCurveBuilder


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
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.curve_builder = FeedingCurveBuilder(self.lengths)
            self.last_curve = None
            self.helper = _CurveTemplateHelper(self.helper, self)
            self.targets = self.helper.template(self.origin, self.phase, self.options.amplitude_deg)

        def profile(self, position, bias):
            plan = self.curve_builder.travelling(self.origin, position, self.phase, bias,
                self.options, self.minimum_bin, self.maximum_bin)
            self.last_curve = plan
            return plan.targets(self.lengths), plan.focus, plan.cell, plan.fraction

    CurveContinuousFeeding.__name__ = 'CurveContinuousFeeding'
    return CurveContinuousFeeding


CurveContinuousFeeding = curve_feeding_class(ContinuousFeeding)


def install_in_debug_process():
    """Opt in ONLY inside this experimental process; no files are rewritten."""
    from codes import Rolling_Arc_Debug_Mujoco as debug
    debug.real.ContinuousFeeding = curve_feeding_class(debug.real.ContinuousFeeding)
    debug.SAVE_LOG = True
    from pathlib import Path
    debug.LOG_DIRECTORY = Path(__file__).resolve().parents[2]/'logs/curve_feeding_experiment'

    class ExperimentalSession(debug.DebugSession):
        def __init__(self, scene=None, start_x=None):
            super().__init__(debug.SCENE if scene is None else scene,
                             debug.START_X_M if start_x is None else start_x)

        def snapshot(self):
            row = super().snapshot()
            core = self.controller.feeding
            row['experimental_trajectory_backend'] = 'stitched_curve_integral'
            if core is not None and getattr(core, 'last_curve', None) is not None:
                row['experimental_curve'] = core.last_curve.as_dict()
            return row

    if not getattr(debug.DebugSession, '_curve_experiment', False):
        ExperimentalSession._curve_experiment = True
        debug.DebugSession = ExperimentalSession
    return debug
