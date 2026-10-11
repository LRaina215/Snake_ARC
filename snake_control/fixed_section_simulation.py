"""MuJoCo adapter; contacts and pipe metadata never enter the controller."""
import numpy as np
import mujoco
from .simulation import ArcSimulation,backbone
from .fixed_section_compliance import FixedSectionCompliance,SectionOptions


class FixedSectionSimulation(ArcSimulation):
    def __init__(self,scene,parameters,start_x=-.5,options=SectionOptions(),hold_phase=None):
        super().__init__(scene,parameters,start_x)
        self.enabled=parameters.enabled.copy();self.hold_phase=hold_phase
        # Physical skeleton lengths read once, independent of pipe information.
        self.lengths=np.linalg.norm(np.diff(backbone(self.model,self.data),axis=0),axis=1)
        self.controller=FixedSectionCompliance(parameters,self.lengths,options)
        self.target=parameters.targets(parameters.initial_phase_rad)

    def step(self,phase=None):
        if self.hold_phase is not None:phase=self.hold_phase
        elif phase is None:phase=self.p.phase(self.data.time)
        if self.data.time<self.p.settle_s:
            self.target=self.p.targets(phase)
        else:
            self.target=self.controller.update(self.data.qpos[self.drive.qadr],phase,
                                                self.data.time,self.model.opt.timestep)
        self.last_phase=phase
        self.drive.apply_targets(self.data,self.target,self.enabled)
        mujoco.mj_step(self.model,self.data)
        return phase,self.target

    def measure(self,phase=None):
        row=super().measure(self.last_phase)
        row['enabled']=self.enabled.tolist()
        row['target_rad']=self.target.tolist()
        row['geometry']=self.controller.snapshot()
        return row
