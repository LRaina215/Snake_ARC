"""Step2 局部顺应性与支撑窗口：E 切换控制，空格暂停相位，R 反向。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from dataclasses import replace
import json
import time
import numpy as np
from snake_control.rolling_arc import ArcParameters
from snake_control.adaptive_simulation import AdaptiveArcSimulation
from snake_control.arc_compliance import SlidingArcOptions
from snake_control.simulation import ROOT

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',choices=['straight','bend90','elbow90','taper_straight','taper_bend90'],default='taper_bend90')
    parser.add_argument('--window',default='8:13',help='初始搭管构型的首:尾关节，之后由支撑反馈调整')
    parser.add_argument('--gain',type=float,default=.02)
    parser.add_argument('--takeover',type=float,default=2.)
    parser.add_argument('--duration',type=float,default=90.)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not np.isfinite([args.takeover,args.duration]).all() or args.takeover<0 or args.duration<=0:
        parser.error('duration 必须正数，takeover 必须非负')
    first,last=map(int,args.window.split(':'))
    p=replace(ArcParameters.load(ROOT/'configs/step1.json'),first_joint=first,last_joint=last)
    start={'bend90':-.35,'elbow90':-.25}.get(args.scene,-.5)
    sim=AdaptiveArcSimulation(args.scene,p,start,SlidingArcOptions(gain=args.gain),args.takeover)
    samples=[]
    def key_callback(key):
        if key==32:sim.paused=not sim.paused
        elif key in (69,101):
            sim.compliance_enabled=not sim.compliance_enabled
            if sim.compliance_enabled:sim.controller=None  # 从实测姿态重新接管，不复用旧目标
            print('顺应性:',sim.compliance_enabled,flush=True)
        elif key in (82,114):sim.p=replace(sim.p,direction=-sim.p.direction)
    def run(viewer=None):
        next_sample=0.;wall=time.monotonic()
        while sim.data.time<args.duration and (viewer is None or viewer.is_running()):
            sim.step()
            if sim.data.time>=next_sample:
                samples.append(sim.measure());next_sample+=.1
                if viewer:
                    sim.color_window();viewer.sync()
                    time.sleep(max(0.,sim.data.time-(time.monotonic()-wall)))
            if not np.isfinite(sim.data.qpos).all():raise RuntimeError('Nonfinite simulation state')
    if args.headless:run()
    else:
        import mujoco.viewer
        print('空格：冻结相位（顺应性继续）；E：顺应性开/关；R：反向。绿=管道支撑观测，蓝=使能，紫=失能。')
        with mujoco.viewer.launch_passive(sim.model,sim.data,key_callback=key_callback) as viewer:
            viewer.cam.lookat[:]=[-.1,.2,1.1];viewer.cam.distance=2.6
            viewer.cam.azimuth=135;viewer.cam.elevation=-22
            run(viewer)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(samples,indent=2),encoding='utf-8')
    if samples:print(json.dumps(dict(scene=args.scene,travel_m=samples[-1]['progress_m']-samples[0]['progress_m'],
                                  compliance=samples[-1]['compliance']),ensure_ascii=False))

if __name__=='__main__':main()
