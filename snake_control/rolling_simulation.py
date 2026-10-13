"""Angle-only moving-exchange executor, same reused robot and motor PD."""
from .length_simulation import LengthSimulation
from .rolling_handover import RollingHandover


class RollingSimulation(LengthSimulation):
    def __init__(self, parameters, references, scene='straight', start_x=-.9, resize=False):
        super().__init__(parameters, references, scene, start_x, resize=False, sliding=False)
        self.controller = RollingHandover(parameters, self.lengths, references, resize=resize,
                                         sliding=True, feedback_period_s=.2)
        self.target, self.enabled = self.controller.targets.copy(), parameters.enabled.copy()
