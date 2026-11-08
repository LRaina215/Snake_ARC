"""One physical run; recorded world/contacts are never controller feedback."""
import argparse,json,sys,shutil
from dataclasses import asdict
from pathlib import Path
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from snake_control.vertical_scene import VerticalScene
from snake_control.vertical_arc_transition import VerticalArcTransition,VerticalArcOptions
from snake_control.curve_gait import CurveGait,helix
from snake_control.simulation import backbone


def json_default(x):
    return x.item() if isinstance(x,np.generic) else x.tolist()


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    opt=VerticalArcOptions(phase_rate_rad_s=args.rate,retained_turns=args.turns,finish_after_pilot=args.case=='pilot')
    (out/'hypothesis.json').write_text(json.dumps(dict(hypothesis='full helix and then local wrap can climb under gravity' if args.case=='pilot' else 'a supported local wrap can migrate over a known elbow and become centered open ARC',
        case=args.case,radius_m=args.radius,pitch_m=args.pitch,options=asdict(opt),controller_inputs=['joint_angles','commands','time','configured_lengths'],
        acceptance='local wrap net ascent >=72mm, no floor support, passive actuator zero' if args.case=='pilot' else 'complete transfer, open ARC, balanced ends, >=72mm forward continuation, no floor/self contact; run independent saved-state review'),indent=2))
    sim=VerticalScene('straight' if args.case=='pilot' else 'elbow');m,d=sim.model,sim.data
    shutil.copyfile(sim.model_path,out/'model.xml')
    for file in ['snake_control/vertical_arc_transition.py','snake_control/vertical_scene.py','snake_control/curve_gait.py',
                 'snake_control/bent_helix_curve.py','snake_control/helix_arc_transition.py','tools/build_vertical_arc_scene.py',__file__]:
        shutil.copyfile(Path(file),out/Path(file).name)
    g=CurveGait(18,lengths=sim.lengths);target=g.targets([helix(g.body_length,args.radius,args.pitch)],.4)
    head_height=(1.42 if args.case=='pilot' else 1.) if args.head_height is None else args.head_height
    initial=sim.place_helix(target,head_height,args.beta)
    core=VerticalArcTransition(target,sim.lengths,opt);mask=np.ones(18,bool)
    rows=[];states=[];speeds=[];targets=[];masks=[];forces=[];statuses=[]
    next_control=next_feedback=next_sample=0.;measured=target.copy();fail=None;previous=None;feedback_time=0.;roll_start=None;operator=[]
    while d.time<args.duration:
        if d.time+1e-8>=next_control:
            if d.time+1e-8>=next_feedback:measured=d.qpos[sim.drive.qadr].copy();feedback_time=float(d.time);next_feedback+=.2
            if args.case=='elbow':
                pts=backbone(m,d)
                if core.stage in ('local_climb','vertical_approach') and pts[0,2]>=1.72:
                    core.begin_bend(-.08);operator.append(dict(time_s=float(d.time),command='begin_bend',source='script_emulates_operator_using_evaluation_pose',nominal_offset_m=-.08))
                elif core.stage=='bend_transfer' and core.head_u>.65:
                    end=int(np.searchsorted(core.curve.s,core.helix_length))
                    if pts[:end+1,0].min()>.33:
                        core.confirm_horizontal(measured);operator.append(dict(time_s=float(d.time),command='horizontal_supported',source='script_emulates_operator_using_evaluation_pose'))
            try:target,mask=core.step(.02,measured,feedback_time)
            except Exception as exc:fail='controller_exception:'+type(exc).__name__+':'+str(exc);break
            next_control+=.02
            if core.stage!=previous:
                print(json.dumps(dict(t=round(d.time,2),stage=core.stage,z=round(backbone(m,d)[0,2],3))),flush=True);previous=core.stage
        sim.drive.apply_targets(d,target,mask);mujoco.mj_step(m,d)
        if d.time+1e-8>=next_sample:
            rows.append(dict(time_s=float(d.time),stage=core.stage,**sim.evaluate(target,mask)))
            states.append(d.qpos.copy());speeds.append(d.qvel.copy());targets.append(target.copy());masks.append(mask.copy());forces.append(d.actuator_force.copy());statuses.append(core.status());next_sample+=.05
            if rows[-1]['min_height_m']<.05:fail='floor_clearance_lost_before_contact';break
            baseline=[r for r in rows if r['stage']=='baseline_climb']
            if core.stage=='exit_transition' and baseline and baseline[-1]['head_z_m']-baseline[0]['head_z_m']<.02:
                fail='baseline_did_not_climb';break
            if core.stage=='complete':break
            if core.stage=='arc_roll':
                roll_start=float(d.time) if roll_start is None else roll_start
                if d.time-roll_start>=6.:break
    stage_ranges={}
    for name in sorted(set(r['stage'] for r in rows)):
        x=[r for r in rows if r['stage']==name];stage_ranges[name]=dict(start_s=x[0]['time_s'],end_s=x[-1]['time_s'],head_delta_z_m=x[-1]['head_z_m']-x[0]['head_z_m'],min_head_z_m=min(r['head_z_m'] for r in x))
    local=stage_ranges.get('local_climb',{})
    completed=core.stage=='complete' if args.case=='pilot' else core.stage=='arc_roll' and roll_start is not None and d.time-roll_start>=6.
    summary=dict(case=args.case,initial=initial,failure=fail,stages=stage_ranges,completed=completed,
        passive_max_nm=max(r['passive_actuator_nm'] for r in rows),floor_frames=sum(r['floor_force_n']>1 for r in rows),
        self_contact_frames=sum(bool(r['self_contacts']) for r in rows),max_joint_deg=max(r['max_joint_deg'] for r in rows),
        max_tracking_deg=max(r['tracking_deg'] for r in rows),saturated_frames=sum(bool(r['saturated_joints']) for r in rows),
        passed=bool(completed and local.get('head_delta_z_m',-1)>=.072 and not fail and not any(r['floor_force_n']>1 for r in rows)) if args.case=='pilot' else None)
    np.savez_compressed(out/'trace.npz',qpos=states,qvel=speeds,targets=targets,masks=masks,forces=forces,time=[r['time_s'] for r in rows])
    (out/'run.json').write_text(json.dumps(dict(summary=summary,samples=rows),indent=1,default=json_default));(out/'controller.json').write_text(json.dumps(statuses,default=json_default))
    (out/'events.json').write_text(json.dumps(core.events,indent=1,default=json_default));print(json.dumps(summary,default=json_default),flush=True)
    (out/'operator.json').write_text(json.dumps(operator,indent=1))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['pilot','elbow'],default='pilot')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--radius',type=float,default=.094);p.add_argument('--pitch',type=float,default=.13)
    p.add_argument('--rate',type=float,default=-.5);p.add_argument('--turns',type=float,default=1.5);p.add_argument('--duration',type=float,default=220.)
    p.add_argument('--head-height',type=float);p.add_argument('--beta',type=float,default=0.)
    run(p.parse_args())
