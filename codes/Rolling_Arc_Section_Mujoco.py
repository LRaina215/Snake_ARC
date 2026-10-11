"""Fixed-window section compliance viewer. Run one Python file; no EXE needed."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,time,queue
from dataclasses import replace
import numpy as np
import mujoco
import mujoco.viewer
from snake_control.rolling_arc import ArcParameters
from snake_control.fixed_section_compliance import SectionOptions
from snake_control.fixed_section_simulation import FixedSectionSimulation
from snake_control.simulation import ROOT,backbone


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--scene',default='straight');a.add_argument('--gain',type=float,default=.02)
    a.add_argument('--duration',type=float,default=110);a.add_argument('--window',default='8:13')
    a.add_argument('--start-x',type=float,default=-.5)
    args=a.parse_args();first,last=map(int,args.window.split(':'))
    p=replace(ArcParameters.load(ROOT/'configs/step1.json'),first_joint=first,last_joint=last)
    sim=FixedSectionSimulation(args.scene,p,args.start_x,options=SectionOptions(gain=args.gain))
    commands=queue.SimpleQueue();paused=False;direction=1;phase=p.initial_phase_rad
    print('固定使能 J%d–J%d；空格暂停/继续相位，E切换几何收紧增益，R反向。窗口不在线改变。'%(first,last),flush=True)
    def key(code):
        if code in (32,69,82):commands.put(code)
    with mujoco.viewer.launch_passive(sim.model,sim.data,key_callback=key) as viewer:
        viewer.cam.distance=1.65;viewer.cam.azimuth=75;viewer.cam.elevation=-15
        while viewer.is_running() and sim.data.time<args.duration:
            start=time.monotonic()
            while not commands.empty():
                code=commands.get()
                if code==32:paused=not paused
                elif code==82:direction*=-1
                elif code==69:
                    gain=0 if sim.controller.options.gain else args.gain
                    sim.controller.options=replace(sim.controller.options,gain=gain)
                    print('gain=',gain,flush=True)
            for _ in range(20):
                if sim.data.time>=p.settle_s and not paused:phase+=direction*p.omega_rad_s*sim.model.opt.timestep
                sim.step(phase)
            with viewer.lock():
                viewer.cam.lookat[:]=backbone(sim.model,sim.data)[first:last+2].mean(axis=0)+[0,0,-.17]
            viewer.sync();time.sleep(max(0,.02-(time.monotonic()-start)))


if __name__=='__main__':main()
