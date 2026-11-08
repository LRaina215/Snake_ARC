"""Research opt-in; all feeding math comes from the production module."""
from snake_control.feeding_controller import (
    _CurveTemplateHelper, curve_feeding_class, CurveContinuousFeeding)


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
