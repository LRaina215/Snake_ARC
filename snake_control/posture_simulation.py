"""Reuse the robot, one-time placement and angle-only sampling adapter."""
import numpy as np
import mujoco
from .simulation import ArcSimulation
from .motor_feedback_simulation import MotorFeedbackSimulation
from .posture_window import PostureWindow


class PostureSimulation(MotorFeedbackSimulation):
    def __init__(self, parameters, references, scene, start_x):
        ArcSimulation.__init__(self, scene, parameters, start_x)
        self.lengths = np.asarray(parameters.mechanical_lengths_m)
        radius = .055 if scene == 'elbow90_taper110_210' else .0565
        self.data.qpos[2] += radius-.0565
        mujoco.mj_forward(self.model, self.data)
        self.controller = PostureWindow(parameters, self.lengths, references, feedback_period_s=.2)
        self.feedback_period_s = .2
        self.target, self.enabled = self.controller.targets.copy(), parameters.enabled.copy()
        self.next_feedback_s = self.next_control_s = 0.
        self.last_feedback = None
