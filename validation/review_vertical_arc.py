"""Independent full saved-state review; pipe truth is evaluation, never feedback."""
import argparse,json,sys
from pathlib import Path
import mujoco,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from snake_control.vertical_scene import VerticalScene
from snake_control.simulation import backbone


def nominal_axis_coordinates(points):
    """Evaluation route coordinates and azimuth around the real nominal axis."""
    p=np.asarray(points);a=np.clip(np.arctan2(p[:,2]-1.8,.3-p[:,0]),0,np.pi/2)
    vertical=np.column_stack([np.zeros(len(p)),np.zeros(len(p)),np.minimum(p[:,2],1.8)])
    bent=np.column_stack([.3*(1-np.cos(a)),np.zeros(len(p)),1.8+.3*np.sin(a)])
    horizontal=np.column_stack([np.maximum(p[:,0],.3),np.zeros(len(p)),np.full(len(p),2.1)])
    candidates=np.stack([vertical,bent,horizontal]);idx=np.argmin(np.linalg.norm(candidates-p,axis=2),axis=0)
    center=candidates[idx,np.arange(len(p))];tilt=np.where(idx==0,0.,np.where(idx==2,np.pi/2,a))
    normal=np.column_stack([np.cos(tilt),np.zeros(len(p)),-np.sin(tilt)])
    delta=p-center
    return np.unwrap(np.arctan2(delta[:,1],np.sum(delta*normal,axis=1)))


def review(folder):
    data=json.loads((folder/'run.json').read_text());rows=data['samples'];trace=np.load(folder/'trace.npz')
    statuses=json.loads((folder/'controller.json').read_text());sim=VerticalScene('elbow');m,d=sim.model,sim.data
    material=np.r_[0.,np.cumsum(sim.lengths)];actual=[];wrap=[];peak_curvature=0.
    allowed={'horizontal_arc_end_reset','horizontal_arc_settle','horizontal_auto_center','horizontal_center_settle','horizontal_arc_final_end_reset','arc_roll'}
    for i,(r,st) in enumerate(zip(rows,statuses)):
        if r['stage'] in ('local_climb','bend_transfer'):
            d.qpos[:]=trace['qpos'][i];mujoco.mj_kinematics(m,d);pts=backbone(m,d)
            end=st['helix_length_m'];coords=np.r_[material[material<end],end]
            p=np.column_stack([np.interp(coords,material,pts[:,j]) for j in range(3)])
            turn=float(np.degrees(np.ptp(nominal_axis_coordinates(p))))
            wrap.append(dict(time_s=r['time_s'],stage=r['stage'],actual_retained_coverage_deg=turn))
            tangents=np.diff(pts,axis=0)/sim.lengths[:,None]
            k=np.arccos(np.clip((tangents[:-1]*tangents[1:]).sum(1),-1,1))/((sim.lengths[:-1]+sim.lengths[1:])/2)
            peak_curvature=max(peak_curvature,float(k[trace['masks'][i]].max(initial=0.)))
        if r['stage'] not in allowed:continue
        h=st['horizontal'];a=h['arc_head_m'];b=a+h['arc_length_m']
        d.qpos[:]=trace['qpos'][i];d.qvel[:]=trace['qvel'][i];d.ctrl[:]=trace['forces'][i];mujoco.mj_forward(m,d)
        pts=backbone(m,d);coords=np.r_[a,material[(material>a)&(material<b)],b]
        p=np.column_stack([np.interp(coords,material,pts[:,j]) for j in range(3)])
        yz=p[:,1:]-[0.,2.1];polar=np.unwrap(np.arctan2(yz[:,1],yz[:,0]));radius=np.linalg.norm(yz,axis=1)
        turn=float(np.degrees(np.ptp(polar)));gap=float(np.linalg.norm(p[-1]-p[0])/h['arc_length_m']);plane=float(np.std(p[:,0]))
        top=int(np.argmin(abs(material-coords[np.argmax(p[:,2])])));height=pts[:,2]-2.1
        def crossing(ids):
            for j,k in zip(ids[:-1],ids[1:]):
                if height[j]>=0>=height[k]:return float(material[j]+height[j]/max(height[j]-height[k],1e-12)*(material[k]-material[j]))
            return None
        left=crossing(list(range(top,-1,-1)));right=crossing(list(range(top,len(material))))
        balance=None if left is None or right is None else float(material[-1]-right-left)
        shape=150<=turn<=220 and gap>=.45 and plane<=.025 and p[:,2].max()-2.1>=.04 and radius.max()<.17
        vertical_force=0.
        vertical=m.geom('pipe_vertical').id
        for j,c in enumerate(d.contact):
            if vertical in (c.geom1,c.geom2) and max(m.geom_bodyid[[c.geom1,c.geom2]])>=sim.head:
                f=np.zeros(6);mujoco.mj_contactForce(m,d,j,f);vertical_force+=abs(f[0])
        actual.append(dict(time_s=r['time_s'],stage=r['stage'],turn_deg=turn,gap_fraction=gap,plane_rms_m=plane,
            balance_m=balance,shape_passed=bool(shape),arc_min_x_m=float(p[:,0].min()),arc_center_x_m=float(p[:,0].mean()),
            vertical_contact_normal_n=vertical_force))
    final=[a for a in actual if a['stage']=='arc_roll'];ri=[i for i,r in enumerate(rows) if r['stage']=='arc_roll']
    summary=dict(controller_completed=data['summary']['completed'],failure=data['summary']['failure'],
        arc_process_shape_passed=bool(actual and all(a['shape_passed'] for a in actual)),
        final_shape_passed=bool(final and all(a['shape_passed'] for a in final)),
        final_turn_min_max_deg=None if not final else [min(a['turn_deg'] for a in final),max(a['turn_deg'] for a in final)],
        final_balance_max_m=max((abs(a['balance_m']) for a in final if a['balance_m'] is not None),default=None),
        final_plane_rms_max_m=max((a['plane_rms_m'] for a in final),default=None),
        final_arc_min_x_m=min((a['arc_min_x_m'] for a in final),default=None),
        final_vertical_contact_normal_max_n=max((a['vertical_contact_normal_n'] for a in final),default=None),
        forward_m=None if not final else final[-1]['arc_center_x_m']-final[0]['arc_center_x_m'],
        final_ends_disabled=bool(ri and all(not trace['masks'][i,0] and not trace['masks'][i,-1] for i in ri)),
        passive_max_nm=float(np.max(abs(trace['forces'][~trace['masks']]),initial=0.)),
        floor_frames=sum(r['floor_force_n']>1. for r in rows),self_contact_frames=sum(bool(r['self_contacts']) for r in rows),
        maximum_joint_deg=max(r['max_joint_deg'] for r in rows),maximum_tracking_deg=max(r['tracking_deg'] for r in rows),
        saturated_frames=sum(bool(r['saturated_joints']) for r in rows),duration_s=rows[-1]['time_s'])
    summary['peak_discrete_curvature_per_m_during_local_and_bend']=peak_curvature
    summary['actual_wrap_by_stage_deg']={stage:[min(a['actual_retained_coverage_deg'] for a in wrap if a['stage']==stage),
        max(a['actual_retained_coverage_deg'] for a in wrap if a['stage']==stage)] for stage in ('local_climb','bend_transfer') if any(a['stage']==stage for a in wrap)}
    summary['near_zero_resultant_support_samples']=int(sum(np.linalg.norm(r['pipe_force_n'])<.1 for r in rows))
    summary['passed']=bool(summary['controller_completed'] and not summary['failure'] and summary['arc_process_shape_passed'] and
        summary['final_shape_passed'] and summary['final_balance_max_m'] is not None and summary['final_balance_max_m']<=.072 and
        summary['forward_m']>=.072 and summary['final_arc_min_x_m']>.33 and summary['final_vertical_contact_normal_max_n']<1. and
        summary['final_ends_disabled'] and summary['passive_max_nm']<1e-9 and not summary['floor_frames'] and not summary['self_contact_frames'])
    (folder/'review.json').write_text(json.dumps(dict(summary=summary,actual=actual,retained_wrap=wrap),indent=1));print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folders',nargs='+',type=Path);a=p.parse_args()
    for folder in a.folders:review(folder)
