"""Fixed local rolling arc, adapted from Snake_Robot_Gait/codes/Rolling_Arc.py.

Joint numbering is one-based in configuration. No contact identification or
compliance is performed in Step1. Zero target is NOT actuator disable.
"""
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class ArcParameters:
    joints: int = 20
    first_joint: int = 8
    last_joint: int = 13
    amplitude_deg: float = 60.0
    omega_rad_s: float = 0.6
    direction: int = 1
    initial_phase_rad: float = 0.0
    settle_s: float = 2.0
    kp: float = 127.0
    kd: float = 1.0
    torque_limit_nm: float = 2.355
    passive_damping: float = 0.05

    def __post_init__(self):
        if not 1 <= self.first_joint <= self.last_joint <= self.joints:
            raise ValueError('Invalid one-based active window')
        numeric = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not np.isfinite(numeric).all():
            raise ValueError('Parameters must be finite')
        if self.direction not in (-1, 1) or not 0 < self.amplitude_deg <= 90:
            raise ValueError('direction must be ±1; amplitude must be in (0,90]')
        if min(self.omega_rad_s, self.kp, self.torque_limit_nm) <= 0 or min(self.kd, self.settle_s, self.passive_damping) < 0:
            raise ValueError('Invalid frequency, gain or damping')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))

    @property
    def enabled(self):
        return (np.arange(1, self.joints + 1) >= self.first_joint) & (np.arange(1, self.joints + 1) <= self.last_joint)

    def targets(self, phase):
        if not math.isfinite(phase):
            raise ValueError('Phase must be finite')
        odd = np.arange(self.joints) % 2 == 0
        angles = math.radians(self.amplitude_deg) * np.where(odd, np.cos(phase), self.direction * np.sin(phase))
        return np.where(self.enabled, angles, 0.0)

    def phase(self, time_s):
        return self.initial_phase_rad + self.omega_rad_s * max(0.0, time_s - self.settle_s)


class LocalMotorDrive:
    """Motor PD only inside the fixed window; outside outputs exactly zero."""
    def __init__(self, model, parameters):
        self.p = parameters
        if model.nu != parameters.joints:
            raise ValueError('Actuator count differs from gait configuration')
        self.qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        self.vadr = model.jnt_dofadr[model.actuator_trnid[:, 0]]

    def apply(self, data, phase):
        target = self.p.targets(phase)
        torque = self.p.kp * (target - data.qpos[self.qadr]) - self.p.kd * data.qvel[self.vadr]
        data.ctrl[:] = np.where(self.p.enabled, np.clip(torque, -self.p.torque_limit_nm, self.p.torque_limit_nm), 0.0)
        return target
