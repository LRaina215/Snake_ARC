"""Two short pre-U trials from the same saved real-command MuJoCo checkpoint."""
import argparse
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validation.mode_handoff_experiment import ExperimentController, real
from codes.Rolling_Arc_Debug_Mujoco import MujocoMotorBus
from validation.gm_locked_experiment import APPROACH_SETUP
from validation.gm_shift_cycle_experiment import CycleTrial
from validation.gm_replay_wrap_experiment import save
from snake_control.simulation import backbone


def run(source,output,adjusted,tighten_steps=None):
    record=json.loads((source/'locked_gait.json').read_text(encoding='utf-8'))
    rows=record['samples'];trace=np.load(source/'locked_gait_trace.npz')
    index=next(i for i,r in enumerate(rows) if r['stage']=='unwind')-1
    trial=CycleTrial(setup=APPROACH_SETUP);sim,m,d=trial.sim,trial.model,trial.data
    d.qpos[:],d.qvel[:]=trace['qpos'][index],trace['qvel'][index]
    mujoco.mj_forward(m,d)
    c=ExperimentController(MujocoMotorBus(sim))
    c.enabled,c.targets=trace['masks'][index].tolist(),np.degrees(trace['targets'][index]).tolist()
    c.bus.prepare_enabled(c.targets,c.enabled)
    c.first,c.last,c.mode=1,18,'hold'
    c.refresh_compliance_feedback()
    c.helix_transition=real.GravityCrossing(real.LINK_LENGTHS_M,c.enabled,c.measured_angles(),
        real.GravityMomentOptions(r_h1=real.HELIX_RADIUS_M,r_h2=real.GRAVITY_OPEN_RADIUS_M,pitch=real.HELIX_PITCH_M),
        from_helix=True)
    c.phase=c.command_phase=c.helix_transition.phase
    actions=[(.2,'O'),(.4,'O'),(.6,'P')] if adjusted else []
    if adjusted and tighten_steps is not None:
        actions=[(.2+.02*i,'O') for i in range(tighten_steps)]
    actions += [(3.,'L'),(3.,'G')]
    commands=[];control_rows=[];collisions=[];last_control=last_feedback=-1.
    def control(_):
        nonlocal last_control,last_feedback
        for contact in d.contact:
            a,b=m.geom_bodyid[[contact.geom1,contact.geom2]]-m.body('head').id
            if min(a,b)>=0 and abs(a-b)>2 and contact.dist<=0:
                collisions.append(dict(time_s=float(d.time),links=[int(a),int(b)]))
        if d.time-last_control>=.02-1e-8:
            if d.time-last_feedback>=.2-1e-8:
                c.refresh_compliance_feedback();last_feedback=float(d.time)
            while actions and d.time>=actions[0][0]-1e-8:
                _,key=actions.pop(0)
                with contextlib.redirect_stdout(io.StringIO()):c.command(real.decode_hotkey(ord(key),True))
                commands.append(dict(time_s=float(d.time),key='Ctrl+'+key,stage=c.helix_transition.stage))
            with contextlib.redirect_stdout(io.StringIO()):c.tick(.02)
            trial.phase=c.command_phase
            control_rows.append(dict(time_s=float(d.time),**{k:v for k,v in c.helix_transition.status().items()
                if k in ('stage','phase_rad','grip_radius_m','grip_radius_target_m','unwind_release_radius_m','return_radius_m')}))
            last_control=float(d.time)
        return np.radians(c.bus.targets_deg)
    trial.phase_run(1,lambda:c.helix_transition.stage,12.2,control,stiffness=lambda:np.asarray(c.enabled,float))
    rows=trial.samples
    # Full actual saved-pose review; the trial also checks self-contact every ms.
    shape=[]
    tail=slice(m.body('head').id+9,m.body('head').id+19)
    for i,state in enumerate(trial.states):
        d.qpos[:]=state;mujoco.mj_forward(m,d)
        points=backbone(m,d);yz=points[:9,1:]-[0,1.2]
        main=rows[i]['main_links']
        tail_y=float(np.average(d.xipos[tail,1],weights=m.body_mass[tail]))
        ready=bool(tail_y<-.04 and points[-1,1]<-.12 and points[-1,2]>=1.175 and
                   any(j<=8 for j in main) and any(j>10 and points[j,2]>1.2 for j in main))
        shape.append(dict(front_center_x_m=float(points[:9,0].mean()),
            front_turn_deg=float(np.degrees(np.ptp(np.unwrap(np.arctan2(yz[:,1],yz[:,0]))))),
            front_max_radius_m=float(np.linalg.norm(yz,axis=1).max()),actual_right_u_ready=ready))
    before=[(r,s) for r,s in zip(rows,shape) if 3.<=r['time_s']<10.]
    original_radii=all(abs(r['unwind_release_radius_m']-.09554165)<1e-12 and abs(r['return_radius_m']-.098)<1e-12 for r in control_rows)
    trim=[r for r in control_rows if r['stage']=='grip_adjust']
    phase_spread=0. if not trim else float(np.ptp([r['phase_rad'] for r in trim]))
    report=dict(adjusted=adjusted,tighten_steps=tighten_steps,source_record=str(source),source_index=index,source_time_s=float(trace['time'][index]),
        dynamics_s=float(trial.data.time),final_stage=c.helix_transition.stage,
        grip_radius_m=c.helix_transition.grip_radius_m,original_later_radii_unchanged=original_radii,
        trim_phase_spread_rad=phase_spread,enabled_changed=any(not np.all(mask) for mask in trial.masks),
        max_saved_target_change_deg=float(np.max(abs(np.diff(np.asarray(trial.targets),axis=0)))*180/np.pi),
        max_tracking_deg=max(r['tracking_max_deg'] for r in rows),
        max_actual_joint_deg=max(r['actual_max_joint_deg'] for r in rows),
        saturated_frames=sum(bool(r['saturated_joints']) for r in rows),
        no_main_contact_frames=sum(not r['main_links'] for r in rows),
        no_front_contact_frames=sum(not any(j<=8 for j in r['main_links']) for r in rows),
        actual_u_ready_frames=sum(s['actual_right_u_ready'] for s in shape),
        actual_u_end_ready=shape[-1]['actual_right_u_ready'],
        floor_frames=sum(r['ground_force_n']>1 for r in rows),self_contact_events=collisions[:1],
        pre_loosen_front_turn_min_deg=min(s['front_turn_deg'] for _,s in before),
        pre_loosen_front_axial_range_m=float(np.ptp([s['front_center_x_m'] for _,s in before])))
    expected_radius=.094 if not adjusted else .094-.001*(1 if tighten_steps is None else tighten_steps)
    report['interface_passed']=bool(original_radii and phase_spread<1e-12 and not report['enabled_changed']
        and abs(report['grip_radius_m']-expected_radius)<1e-10 and report['final_stage']=='right_u_hold')
    report['bounded_contact_check_passed']=bool(report['interface_passed'] and not collisions and not report['floor_frames']
        and not report['no_main_contact_frames'])
    report['dynamics_check_passed']=bool(report['bounded_contact_check_passed'] and report['actual_u_ready_frames']>0)
    clock=np.asarray([r['time_s'] for r in control_rows])
    for row in rows:
        state=control_rows[max(0,int(np.searchsorted(clock,row['time_s'],side='right'))-1)]
        row.update({k:state[k] for k in ('grip_radius_m','grip_radius_target_m','unwind_release_radius_m','return_radius_m')})
        row['reference_roll_rad']=state['phase_rad']
    label='adjusted' if adjusted else 'baseline'
    with contextlib.redirect_stdout(io.StringIO()):
        save(trial,output/label,'locked_gait',report['source_time_s'],
            **{k:v for k,v in report.items() if k not in ('source_time_s','dynamics_s')},
            binary_enable=True,feedback_period_s=.2,parameters=asdict(sim.p),start_x=.2,hardware_run=False,commands=commands,
            controller_status=c.helix_transition.status())
    (output/label/'control_trace.json').write_text(json.dumps(control_rows,indent=1),encoding='utf-8')
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('artifacts/sequential_handoff/round19/buffer6'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    results=[run(a.source,a.output,adjusted) for adjusted in (False,True)]
    (a.output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
