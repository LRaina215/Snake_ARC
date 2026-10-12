"""MuJoCo executor with an angle-only boundary to the sliding controller."""
import json
import numpy as np
import mujoco
from .simulation import ArcSimulation, ROOT
from .motor_shape_observer import PhaseShapeReference
from .motor_feedback_sliding import MotorFeedbackSliding


class MotorFeedbackSimulation(ArcSimulation):
    def __init__(self, parameters, execute=False, scene='straight', start_x=-.5):
        super().__init__(scene, parameters, start_x)
        self.lengths = np.array([.0645]*(parameters.joints+1))
        reference = PhaseShapeReference.from_dict(json.loads(
            (ROOT/'configs/motor_shape_reference.json').read_text(encoding='utf8')))
        self.controller = MotorFeedbackSliding(parameters, self.lengths, reference, execute=execute)
        self.target, self.enabled = self.controller.targets.copy(), parameters.enabled.copy()
        self.next_feedback_s = 0.
        self.last_feedback = None

    def step(self, phase=None):
        if phase is not None:
            raise ValueError('Motor-feedback controller owns its commanded phase')
        if self.data.time+1e-9 >= self.next_feedback_s:
            self.next_feedback_s += .02
            # The boundary passes ONLY angles and time. In particular qvel,
            # root pose, contacts and actuator forces cannot enter this call.
            angles = np.degrees(self.data.qpos[self.drive.qadr]).copy()
            self.last_feedback = dict(angles_deg=angles.tolist(), feedback_time_s=float(self.data.time),
                phase_rad=self.controller.phase, enabled=self.enabled.tolist())
            self.target, self.enabled = self.controller.update(angles, float(self.data.time))
        self.last_phase = self.controller.phase
        self.drive.apply_targets(self.data, self.target, self.enabled)
        mujoco.mj_step(self.model, self.data)
        return self.last_phase, self.target
