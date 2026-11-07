"""Three ARC amplitudes, two feed directions, identical initial state and feeder."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from snake_control.arc_helix_transition import ArcHelixOptions
from snake_control.gravity_handoff import GravityFrontTransition, GravityCrossing
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import backbone
from validation.active_transfer_experiment import material_truth
from validation.gm_arc_handoff_experiment import place_arc_for_adjustment
from validation.gm_locked_experiment import APPROACH_SETUP
from validation.gm_shift_cycle_experiment import CycleTrial
from validation.gm_wrap_first_experiment import pipe_contacts


def run(output,amplitude,direction):
    trial=CycleTrial(setup=dict(APPROACH_SETUP,scene='straight'))
    sim,m,d=trial.sim,trial.model,trial.data
    p=ArcParameters(joints=18,first_joint=7,last_joint=12)
    sim.p=sim.drive.p=p
    place_arc_for_adjustment(trial,p,.1)
    target,enabled=p.targets(0.),p.enabled
    for _ in range(2500):sim.drive.apply_targets(d,target,enabled);mujoco.mj_step(m,d)
    initial=material_truth(backbone(m,d))
    # All cases start from the same physical 60-degree ARC. Establish the new
    # amplitude before defining the feed-only material-motion baseline.
    wanted=np.where(enabled,np.where(np.arange(18)%2==0,np.radians(amplitude),0.),0.)
    for _ in range(3000):
        target+=np.clip(wanted-target,-np.radians(45)*.001,np.radians(45)*.001)
        sim.drive.apply_targets(d,target,enabled);mujoco.mj_step(m,d)
    baseline=material_truth(backbone(m,d));start=float(d.time)
    q=np.degrees(d.qpos[sim.drive.qadr]);lengths=np.full(19,.0645)
    radius=GravityCrossing._section_radius(np.radians(q),lengths,(7,12))
    core=GravityFrontTransition(lengths,(7,12),q,options=ArcHelixOptions(prepare_front=True))
    core.amplitude=np.radians(amplitude);core.desired_start=7+2*direction;core._prepare_transfer(direction)
    core.running=True
    last_control=last_feedback=last_save=-1.;ft=0.;measured=q;completed=None;failure=None
    rows=[];poses=[];targets=[];masks=[];forces=[];collisions=[]
    while d.time-start<60.:
        if d.time-last_control>=.02-1e-8:
            if d.time-last_feedback>=.2-1e-8:
                measured=np.degrees(d.qpos[sim.drive.qadr]);ft=last_feedback=float(d.time)
            if completed is None:
                core.advance(.02,float(d.time),ft,measured)
                if core.stage=='transfer_done':completed=float(d.time)
            target,enabled=core.targets,core.enabled
            last_control=float(d.time)
        head=m.body('head').id
        for c in d.contact:
            a,b=m.geom_bodyid[[c.geom1,c.geom2]]-head
            if min(a,b)>=0 and abs(a-b)>2 and c.dist<=0:
                collisions.append([float(d.time),int(a),int(b)]);failure='self_contact';break
        sim.drive.apply_targets(d,target,enabled);mujoco.mj_step(m,d)
        if d.time-last_save>=.05-1e-8:
            points=backbone(m,d);truth=material_truth(points);main,_,_=pipe_contacts(sim)
            yz=points[:,1:]-[0,1.2];near=np.linalg.norm(yz,axis=1)<.145
            polar=np.sort(np.mod(np.arctan2(yz[near,1],yz[near,0]),2*np.pi))
            coverage=360.-np.degrees(np.diff(np.r_[polar,polar[0]+2*np.pi]).max()) if len(polar)>2 else 0.
            rows.append(dict(time_s=float(d.time),material=truth,contacts=main,coverage_deg=float(coverage),
                center_x_m=float(points[:,0].mean()),position_joints=core.transfer.position,
                tracking_deg=float(np.max(abs(np.degrees(target[enabled])-np.degrees(d.qpos[sim.drive.qadr][enabled])))),
                passive_force_nm=float(np.max(abs(d.actuator_force[~enabled]),initial=0.))))
            poses.append(d.qpos.copy());targets.append(target.copy());masks.append(enabled.copy());forces.append(d.actuator_force.copy());last_save=float(d.time)
        if failure or completed is not None and d.time-completed>=2.:break
    final=material_truth(backbone(m,d));delta=None
    if baseline['valid'] and final['valid']:
        delta=.5*((final['head_hanging_m']-final['tail_hanging_m'])-(baseline['head_hanging_m']-baseline['tail_hanging_m']))
    duration=None if completed is None else completed-start
    report=dict(amplitude_deg=amplitude,direction=direction,model_arc_radius_m=radius,
        completed_control=completed is not None,feed_duration_s=duration,actual_material_delta_m=delta,
        actual_mean_speed_m_s=None if delta is None or duration is None else abs(delta)/duration,
        max_coverage_deg=max(r['coverage_deg'] for r in rows),
        axial_center_range_m=max(r['center_x_m'] for r in rows)-min(r['center_x_m'] for r in rows),
        no_support_frames=sum(not r['contacts'] for r in rows),
        max_tracking_deg=max(r['tracking_deg'] for r in rows),self_contacts=collisions[:1],
        passive_force_max_nm=max(r['passive_force_nm'] for r in rows),failure=failure,
        initial_material=initial,feed_baseline=baseline,final_material=final,
        controller_options=asdict(core.transfer.options))
    report['passed']=bool(completed is not None and delta is not None and direction*delta>=.9*.129 and
        not failure and report['no_support_frames']==0 and report['max_coverage_deg']<300. and
        report['passive_force_max_nm']<1e-10)
    label=f'A{amplitude}_'+('head' if direction>0 else 'tail')
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/(label+'_trace.npz'),time=[r['time_s'] for r in rows],qpos=poses,targets=targets,masks=masks,actuator_force=forces)
    (output/(label+'.json')).write_text(json.dumps(dict(summary=report,samples=rows),indent=1),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('initial_material','feed_baseline','final_material','controller_options')}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    results=[run(a.output,amp,side) for amp in (50,60,70) for side in (1,-1)]
    (a.output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
