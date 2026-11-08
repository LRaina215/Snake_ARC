"""Review all saved actual poses and exact operator-confirmation states for route demos."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from snake_control.rolling_arc import ArcParameters
from snake_control.tee_simulation import TeeSimulation
from snake_control.simulation import backbone
from validation.active_transfer_experiment import material_truth


def review(folder):
    record=json.loads((folder/'locked_gait.json').read_text(encoding='utf-8'))
    summary,rows=record['summary'],record['samples'];trace=np.load(folder/'locked_gait_trace.npz')
    sim=TeeSimulation(summary['scene'],ArcParameters(**summary['parameters']))
    m,d=sim.model,sim.data;n=sim.p.joints
    # Legacy recordings retained eight; new ones save the exact curve boundary.
    retained=summary.get('controller_status',{}).get('release_retain_joints',8)
    tail=slice(m.body('head').id+retained+1,m.body('head').id+n+1)
    held=[];rolling=[];open_angles=[];material=[];wind=[];drops=[];episodes=[]
    previous=None
    for i,row in enumerate(rows):
        d.qpos[:]=trace['qpos'][i]
        # Positions need kinematics, not a fresh dynamic/contact solve. Contact
        # and collision evidence comes from the actual 1 ms run and saved rows.
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        points=backbone(m,d);yz=points[:,1:]-[0,1.2]
        stage=row['stage']
        if stage!=previous:
            episodes.append(dict(stage=stage,start_s=row['time_s'],end_s=row['time_s']));previous=stage
        episodes[-1]['end_s']=row['time_s']
        if stage=='feeding' or stage=='arc_settle' or (stage=='front' and row.get('front_stage')=='adjust'):
            near=np.linalg.norm(yz,axis=1)<.145
            if near.sum()>=3:
                polar=np.sort(np.mod(np.arctan2(yz[near,1],yz[near,0]),2*np.pi))
                open_angles.append(float(360-np.degrees(np.diff(np.r_[polar,polar[0]+2*np.pi]).max())))
        if stage=='feeding':material.append(material_truth(points))
        if stage=='return_wind':wind.append(dict(time_s=row['time_s'],head_x_m=row['head_x_m'],main_links=row['main_links']))
        if stage in ('complete','roll'):
            shape=dict(time_s=row['time_s'],turns=float(np.ptp(np.unwrap(np.arctan2(yz[:,1],yz[:,0])))/(2*np.pi)),
                max_radius_m=float(np.linalg.norm(yz,axis=1).max()))
            (held if stage=='complete' else rolling).append(shape)
        if stage in ('drop','drop_hold'):
            frame_retained=int(np.asarray(trace['masks'][i],bool).sum())
            released=slice(m.body('head').id+frame_retained+1,m.body('head').id+n+1)
            com=np.average(d.xipos[released],axis=0,weights=m.body_mass[released])
            drops.append(dict(time_s=row['time_s'],tail_com_y_m=float(com[1]),tail_com_z_m=float(com[2]),
                gravity_moment_x_nm=float(-9.81*m.body_mass[released].sum()*com[1]),
                head_x_m=row['head_x_m'],front_angle_deg=row['front_angle_deg']))
    u_confirmations=[]
    for e in summary.get('operator_events',[]):
        if e['stage']!='right_u_hold':continue
        if 'qpos' not in e:
            u_confirmations.append(dict(time_s=e['time_s'],exact_pose_saved=False));continue
        d.qpos[:]=e['qpos'];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);points=backbone(m,d)
        retained=e.get('retained_joints',8)
        tail=slice(m.body('head').id+retained+1,m.body('head').id+n+1)
        com=np.average(d.xipos[tail],axis=0,weights=m.body_mass[tail]);main=e['main_links']
        ready=bool(com[1]<-.04 and points[-1,1]<-.12 and points[-1,2]>=1.175 and
            any(j<=retained for j in main) and any(j>retained+2 and points[j,2]>1.2 for j in main))
        u_confirmations.append(dict(time_s=e['time_s'],exact_pose_saved=True,actual_right_u_ready=ready,
            tail_com_y_m=float(com[1]),tail_com_z_m=float(com[2]),tail_end_m=points[-1].tolist(),main_links=main))
    forces,masks=trace['actuator_force'],trace['masks']
    passive=float(np.max(abs(forces[~masks]),initial=0.))
    final=rolling or held
    shape_ok=bool(final and min(x['turns'] for x in final)>=1.5 and max(x['max_radius_m'] for x in final)<.145)
    material_delta=None
    if material and material[0]['valid'] and material[-1]['valid']:
        material_delta=.5*((material[-1]['head_hanging_m']-material[-1]['tail_hanging_m'])-
                           (material[0]['head_hanging_m']-material[0]['tail_hanging_m']))
    report=dict(case=summary['case'],controller_passed=summary['passed'],failure=summary['failure'],
        final_stage=summary['final_stage'],frames=len(rows),stage_intervals=episodes,
        commands=summary['commands'],front_before_tail=summary['front_before_tail'],
        actual_helix_shape_passed=shape_ok,helix_shape_frames=len(final),
        final_min_turns=min((r['turns'] for r in final),default=None),
        final_max_radius_m=max((r['max_radius_m'] for r in final),default=None),
        preparation_max_near_pipe_coverage_deg=max(open_angles,default=None),
        actual_mode4_material_change_m=material_delta,passive_actuator_force_max_nm=passive,
        self_contact_events=summary['self_contact_events'],floor_frames=sum(r['ground_force_n']>1. for r in rows),
        low_total_force_frames=sum(r['support_n']<=.1 for r in rows),
        maximum_actual_joint_deg=max(r['actual_max_joint_deg'] for r in rows),
        maximum_tracking_deg=max(r['tracking_max_deg'] for r in rows),
        saturated_frames=sum(bool(r['saturated_joints']) for r in rows),
        right_u_confirmations=u_confirmations,recovery=summary['recovery'],
        gravity_drop=None if not drops else dict(start_tail_com_y_m=drops[0]['tail_com_y_m'],
            end_tail_com_y_m=drops[-1]['tail_com_y_m'],end_tail_com_z_m=drops[-1]['tail_com_z_m'],
            start_gravity_moment_x_nm=drops[0]['gravity_moment_x_nm'],
            front_rotation_deg=float(np.degrees(np.unwrap(np.radians([r['front_angle_deg'] for r in drops]))[-1]-
                                                 np.unwrap(np.radians([r['front_angle_deg'] for r in drops]))[0])),
            head_advance_m=drops[-1]['head_x_m']-drops[0]['head_x_m']))
    report['passed']=bool(summary['passed'] and shape_ok and passive<1e-10 and not report['self_contact_events'] and
        not report['floor_frames'] and (not summary['config'].get('tee') or bool(u_confirmations and
        all(e.get('actual_right_u_ready',False) for e in u_confirmations))))
    (folder/'full_review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('commands','stage_intervals')}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folders',type=Path,nargs='+');a=p.parse_args()
    for folder in a.folders:review(folder)
