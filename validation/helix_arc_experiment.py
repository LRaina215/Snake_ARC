"""18-joint reverse unwinding; physics truth is recorded for evaluation only."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import shutil
from types import SimpleNamespace

import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from snake_control.rolling_arc import ArcParameters
from snake_control.tee_simulation import TeeSimulation,PIPE_HEIGHT
from snake_control.simulation import backbone
from snake_control.gravity_moment import GravityMomentGait,GravityMomentOptions
from snake_control.helix_arc_transition import HelixArcTransition
from validation.gm_wrap_first_experiment import pipe_contacts


def run(beta,output,duration=150.,hotkeys=False,speed_scale=None):
    output.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(__file__),output/'experiment_source.py')
    shutil.copyfile(Path(__file__).resolve().parents[1]/'snake_control/helix_arc_transition.py',output/'controller_source.py')
    params=ArcParameters(joints=18,first_joint=1,last_joint=18)
    sim=TeeSimulation('straight',params)
    m,d=sim.model,sim.data
    gait=GravityMomentGait(18,GravityMomentOptions(r_h1=.094,pitch=.13))
    target=gait.targets(.4,.094)[0];mask=np.ones(18,bool)
    initial_radius=sim.place_shape(target,19,.2,math.radians(beta))
    for _ in range(2000):
        sim.drive.apply_targets(d,target,mask);mujoco.mj_step(m,d)
    initial_points=backbone(m,d)
    core=HelixArcTransition(d.qpos[sim.drive.qadr].copy(),[.0645]*19)
    start=float(d.time);measured=d.qpos[sim.drive.qadr].copy();next_control=start;next_feedback=start;next_sample=start
    rows=[];states=[];vel=[];targets=[];masks=[];forces=[];statuses=[]
    failure=None;roll_start=None;printed_stage=None
    controller=None;commands=[];center_hold_at=None;roll_commanded=False
    if hotkeys:
        from codes.Rolling_Arc_Debug_Mujoco import DebugArcController,MujocoMotorBus,decode_sim_hotkey,real
        if speed_scale is not None:real.HELIX_ARC_SPEED_SCALE=speed_scale
        session=SimpleNamespace(sim=sim,elapsed=0.,physics_started=True,record_event=lambda *a,**k:None)
        bus=MujocoMotorBus(sim)
        controller=DebugArcController(bus,session)
        controller.enabled=mask.tolist();controller.targets=np.degrees(d.qpos[sim.drive.qadr]).tolist()
        controller.first,controller.last,controller.mode=1,18,'hold'
        bus.prepare_enabled(controller.targets,controller.enabled)
        shutil.copyfile(Path(__file__).resolve().parents[1]/'codes/Rolling_Arc_Real.py',output/'real_entry_source.py')
        schedule=[(0.,'7'),(0.,'J'),(.1,'J'),(1.5,'G'),(4.,'G'),(5.,'J'),(6.5,'G')]
        def command(key):
            nonlocal roll_commanded
            before=controller.helix_arc_transition
            result=controller.command(decode_sim_hotkey(ord(key),True))
            after=controller.helix_arc_transition
            if key=='G' and after is None and controller.basic_arc_reference is not None:
                roll_commanded=True
            commands.append(dict(time_s=session.elapsed,key='Ctrl+'+key,result=result or 'accepted',
                stage_before='basic_arc' if before is None and controller.basic_arc_reference is not None else None if before is None else before.stage,
                stage='basic_arc' if after is None and controller.basic_arc_reference is not None else None if after is None else after.stage,
                mode=controller.mode,phase_rad=controller.phase))
    while d.time-start<duration:
        if d.time+1e-8>=next_control:
            fresh=d.time+1e-8>=next_feedback
            if fresh:measured=d.qpos[sim.drive.qadr].copy();next_feedback+=core.o.feedback_period_s
            if controller is None:
                target,mask=core.step(.02,float(d.time-start),measured,fresh)
            else:
                session.elapsed=float(d.time-start)
                if fresh:controller.refresh_compliance_feedback()
                while schedule and session.elapsed+1e-8>=schedule[0][0]:
                    _,key=schedule.pop(0);command(key)
                if controller.helix_arc_transition is None and controller.helix_arc_last is not None and center_hold_at is None:
                    center_hold_at=session.elapsed
                    schedule.extend([(session.elapsed+1.,'J'),(session.elapsed+1.5,'H'),(session.elapsed+2.,'G')])
                controller.tick(.02)
                if controller.helix_arc_transition is not None:
                    core=controller.helix_arc_transition.core
                target=np.radians(controller.targets);mask=np.asarray(controller.enabled,bool)
            current_status=(core.status() if controller is None else controller.helix_arc_transition.status()
                if controller.helix_arc_transition is not None else dict(controller.helix_arc_last,
                    phase=controller.phase,basic_arc=True,mode=controller.mode,
                    pending_phase_rad=controller.pending_phase,enabled_joints=(np.flatnonzero(mask)+1).tolist()))
            display_stage=current_status['stage']
            if display_stage!=printed_stage:
                print(json.dumps(dict(t=round(float(d.time-start),2),stage=display_stage,
                    enabled=int(mask.sum()),head_m=core.head,center_error_m=core.center_error),ensure_ascii=False),flush=True)
                printed_stage=display_stage
            next_control+=.02
        sim.drive.apply_targets(d,target,mask);mujoco.mj_step(m,d)
        if d.time+1e-8<next_sample:continue
        next_sample+=.05
        pts=backbone(m,d);yz=pts[:,1:]-[0,PIPE_HEIGHT]
        metrics=sim.measure(current_status['phase']);main,branch,_=pipe_contacts(sim)
        active=np.flatnonzero(mask)
        first,last=int(active.min()),int(active.max())
        arcpts=yz[first:last+3]
        angles=np.unwrap(np.arctan2(arcpts[:,1],arcpts[:,0]))
        plane_rms=float(np.std(pts[first:last+3,0]))
        headlength=float(np.linalg.norm(np.diff(pts[:first+2],axis=0),axis=1).sum())
        taillength=float(np.linalg.norm(np.diff(pts[last+2:],axis=0),axis=1).sum())
        collision=[]
        h=m.body('head').id
        for c in d.contact:
            a,b=m.geom_bodyid[[c.geom1,c.geom2]]-h
            if min(a,b)>=0 and abs(a-b)>2 and c.dist<=0:collision.append([int(a),int(b)])
        row=dict(time_s=float(d.time-start),stage=display_stage,reference_roll_rad=current_status['phase'],
            x_min_m=float(pts[:,0].min()),center_x_m=float(pts[:,0].mean()),head_x_m=float(pts[0,0]),
            front_angle_deg=float(np.degrees(np.arctan2(yz[4,1],yz[4,0]))),
            tail_z_m=float(pts[-1,2]),min_height_m=float(pts[:,2].min()),
            main_links=main,branch_links=branch,support_n=metrics['normal_force_n'],
            ground_force_n=metrics['ground_contact_force_n'],enabled_count=int(mask.sum()),
            tracking_max_deg=float(np.degrees(abs(target-d.qpos[sim.drive.qadr]))[mask].max()),
            actual_max_joint_deg=float(np.degrees(abs(d.qpos[sim.drive.qadr])).max()),
            saturated_joints=(np.flatnonzero(abs(d.actuator_force)>=params.torque_limit_nm-1e-6)+1).tolist(),
            passive_force_max_nm=float(np.max(abs(d.actuator_force[~mask]),initial=0.)),
            active_coverage_deg=float(np.degrees(np.ptp(angles))),active_plane_rms_m=plane_rms,
            head_passive_m=headlength,tail_passive_m=taillength,
            active_end_height_m=arcpts[[0,-1],1].tolist(),self_contacts=collision)
        rows.append(row);states.append(d.qpos.copy());vel.append(d.qvel.copy());targets.append(target.copy());masks.append(mask.copy());forces.append(d.actuator_force.copy());statuses.append(current_status)
        if metrics['ground_contact_force_n']>1. or pts[:,2].max()<.9:
            failure='floor contact or whole body lost pipe';break
        if core.stage=='arc_roll' and (controller is None or roll_commanded):
            roll_start=float(d.time) if roll_start is None else roll_start
            if d.time-roll_start>=6.:break
    finished=(core.stage=='arc_roll' and (controller is None or
        (controller.helix_arc_transition is None and controller.helix_arc_last is not None)))
    if not finished and failure is None:failure='time budget reached at '+display_stage
    summary=dict(scene='straight',parameters=asdict(params),joints=18,beta_deg=beta,initial_radius_m=initial_radius,
        initial_tail_azimuth_deg=float(np.degrees(np.arctan2(initial_points[-1,2]-PIPE_HEIGHT,initial_points[-1,1]))),
        feedback_period_s=core.o.feedback_period_s,controller_options=asdict(core.o),binary_enable=True,
        controller_inputs='sampled motor angles, dimensions, time only',
        passed=finished and failure is None,failure=failure,final_stage=display_stage,dynamics_s=float(d.time-start),
        controller_status=current_status,events=core.events,commands=commands,
        hotkey_entry=hotkeys,center_completion_auto_hold_s=None if center_hold_at is None else 1.,
        main_pipe_radius_m=float(m.geom_size[m.geom('pipe_main').id,0]),
        model_link_lengths_m=np.linalg.norm(np.diff(backbone(m,d),axis=0),axis=1).tolist(),
        returned_to_basic=controller is not None and controller.basic_arc_reference is not None)
    (output/'locked_gait.json').write_text(json.dumps(dict(summary=summary,samples=rows),ensure_ascii=False,indent=1),encoding='utf-8')
    (output/'controller.json').write_text(json.dumps(statuses,ensure_ascii=False),encoding='utf-8')
    np.savez_compressed(output/'locked_gait_trace.npz',time=[r['time_s'] for r in rows],qpos=states,qvel=vel,targets=targets,masks=masks,actuator_force=forces)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--beta',type=float,default=0.);p.add_argument('--output',type=Path,required=True);p.add_argument('--duration',type=float,default=150.)
    p.add_argument('--hotkeys',action='store_true',help='Use the actual shared Ctrl+7/J/G debug dispatcher')
    p.add_argument('--speed-scale',type=float,help='Override motion speed in this experiment process; preserve user source parameters')
    a=p.parse_args();run(a.beta,a.output,a.duration,a.hotkeys,a.speed_scale)
