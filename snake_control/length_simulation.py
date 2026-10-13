"""Simulation executor; only measured angles cross the control boundary."""
import numpy as np
import mujoco
from .simulation import ArcSimulation
from .motor_feedback_simulation import MotorFeedbackSimulation
from .adaptive_arc_length import AdaptiveArcLength


class LengthSimulation(MotorFeedbackSimulation):
    def __init__(self, parameters, references, scene, start_x, resize=False, sliding=False, **kwargs):
        ArcSimulation.__init__(self, scene, parameters, start_x)
        self.lengths = np.asarray(parameters.mechanical_lengths_m)
        # Correct the generic Step1 placement ONCE to this scene's initial
        # radius. This scene information never enters the controller.
        route = self.model.numeric('pipe_centerline').data.reshape(-1, 4)
        radius = float(np.interp(start_x, route[:, 0], route[:, 3])) if scene != 'elbow90_taper110_210' else .055
        self.data.qpos[2] += radius-.0565
        mujoco.mj_forward(self.model, self.data)
        self.controller = AdaptiveArcLength(parameters, self.lengths, references, resize=resize,
            sliding=sliding, feedback_period_s=.2, **kwargs)
        self.feedback_period_s = .2
        self.target, self.enabled = self.controller.targets.copy(), parameters.enabled.copy()
        self.next_feedback_s = self.next_control_s = 0.
        self.last_feedback = None
