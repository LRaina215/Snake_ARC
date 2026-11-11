"""Reuse the robot, one-time placement and angle-only sampling adapter."""
import numpy as np
import mujoco
from .simulation import ArcSimulation
from .fixed_section_compliance import SectionOptions
from .motor_feedback_simulation import MotorFeedbackSimulation
from .posture_window import PostureWindow, PostureWindowOptions


class PostureSimulation(MotorFeedbackSimulation):
    def __init__(self, parameters, references, scene, start_x, *, section_options=SectionOptions(),
                 posture_options=PostureWindowOptions()):
        ArcSimulation.__init__(self, scene, parameters, start_x)
        self.lengths = np.asarray(parameters.mechanical_lengths_m)
        route = self.model.numeric('pipe_centerline').data.reshape(-1, 4)
        radius = .055 if scene == 'elbow90_taper110_210' else float(np.interp(start_x, route[:, 0], route[:, 3]))
        self.data.qpos[2] += radius-.0565
        mujoco.mj_forward(self.model, self.data)
        self.controller = PostureWindow(parameters, self.lengths, references, feedback_period_s=.2,
                                        section_options=section_options, posture_options=posture_options)
        self.feedback_period_s = .2
        self.target, self.enabled = self.controller.targets.copy(), parameters.enabled.copy()
        self.next_feedback_s = self.next_control_s = 0.
        self.last_feedback = None
