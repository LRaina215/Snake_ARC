"""Review actual saved states; target parameters select material intervals, not shape truth."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from snake_control.rolling_arc import ArcParameters
from snake_control.tee_simulation import TeeSimulation,PIPE_HEIGHT
from snake_control.simulation import backbone


def review(folder):
    r=json.loads((folder/'locked_gait.json').read_text(encoding='utf-8'));summary=r['summary'];rows=r['samples']
    states=json.loads((folder/'controller.json').read_text(encoding='utf-8'))
    trace=np.load(folder/'locked_gait_trace.npz');sim=TeeSimulation('straight',ArcParameters(**summary['parameters']))
    s=np.arange(20)*.0645;actual=[]
    for i,(row,st) in enumerate(zip(rows,states)):
        if row['stage'] not in ('arc_settle','auto_center','center_settle','arc_roll'):continue
        sim.data.qpos[:]=trace['qpos'][i];mujoco.mj_kinematics(sim.model,sim.data)
        pts=backbone(sim.model,sim.data)
        a=st['arc_head_m'];b=a+st['arc_length_m']
        coords=np.r_[a,s[(s>a)&(s<b)],b]
        p=np.column_stack([np.interp(coords,s,pts[:,j]) for j in range(3)])
        yz=p[:,1:]-[0,PIPE_HEIGHT];radial=np.linalg.norm(yz,axis=1)
        turn=float(np.degrees(np.ptp(np.unwrap(np.arctan2(yz[:,1],yz[:,0])))))
        gap=float(np.linalg.norm(p[-1]-p[0])/st['arc_length_m'])
        top=int(np.argmin(abs(s-coords[np.argmax(p[:,2])])))
        h=pts[:,2]-PIPE_HEIGHT
        def crossing(indices):
            for j,k in zip(indices[:-1],indices[1:]):
                if h[j]>=0 and h[k]<=0:
                    return float((j+h[j]/max(h[j]-h[k],1e-12)*(k-j))*.0645)
            return None
        left=crossing(list(range(top,-1,-1)));right=crossing(list(range(top,20)))
        balance=None if left is None or right is None else float(s[-1]-right-left)
        plane=float(np.std(p[:,0]));crown=float(p[:,2].max()-PIPE_HEIGHT)
        shape=150<=turn<=220 and gap>=.45 and plane<=.025 and crown>=.04 and radial.max()<.17
        actual.append(dict(time_s=row['time_s'],stage=row['stage'],coverage_deg=turn,gap_fraction=gap,
            plane_rms_m=plane,crown_height_m=crown,max_radius_m=float(radial.max()),
            balance_difference_m=balance,shape_passed=bool(shape),
            controller_difference_m=st['geometry'].get('difference_m')))
    final=[x for x in actual if x['stage']=='arc_roll']
    shape_passed=bool(final and all(x['shape_passed'] for x in final))
    process_shape_passed=bool(actual and all(x['shape_passed'] for x in actual))
    balanced=bool(final and all(x['balance_difference_m'] is not None and abs(x['balance_difference_m'])<=.065 for x in final))
    final_indices=[i for i,row in enumerate(rows) if row['stage']=='arc_roll']
    ends_disabled=bool(final_indices and all(not trace['masks'][i,0] and not trace['masks'][i,-1] for i in final_indices))
    stages=[]
    for row in rows:
        if not stages or stages[-1]['stage']!=row['stage']:stages.append(dict(stage=row['stage'],start_s=row['time_s'],end_s=row['time_s']))
        stages[-1]['end_s']=row['time_s']
    roll_rows=[x for x in rows if x['stage']=='arc_roll']
    report=dict(beta_deg=summary['beta_deg'],initial_tail_azimuth_deg=summary['initial_tail_azimuth_deg'],
        controller_passed=summary['passed'],failure=summary['failure'],stages=stages,
        actual_final_shape_passed=shape_passed,arc_process_shape_passed=process_shape_passed,
        actual_balanced=balanced,final_ends_disabled=ends_disabled,
        final_coverage_min_max_deg=None if not final else [min(x['coverage_deg'] for x in final),max(x['coverage_deg'] for x in final)],
        final_max_balance_difference_m=max((abs(x['balance_difference_m']) for x in final if x['balance_difference_m'] is not None),default=None),
        final_max_plane_rms_m=max((x['plane_rms_m'] for x in final),default=None),
        roll_center_advance_m=None if not roll_rows else roll_rows[-1]['center_x_m']-roll_rows[0]['center_x_m'],
        total_center_displacement_m=rows[-1]['center_x_m']-rows[0]['center_x_m'],
        passive_force_max_nm=max(x['passive_force_max_nm'] for x in rows),
        floor_frames=sum(x['ground_force_n']>1. for x in rows),self_contact_frames=sum(bool(x['self_contacts']) for x in rows),
        unsupported_frames=sum(x['support_n']<.1 for x in rows),maximum_actual_joint_deg=max(x['actual_max_joint_deg'] for x in rows),
        maximum_tracking_deg=max(x['tracking_max_deg'] for x in rows),saturated_frames=sum(bool(x['saturated_joints']) for x in rows))
    report['passed']=bool(summary['passed'] and shape_passed and process_shape_passed and balanced and ends_disabled and not report['floor_frames'] and
        not report['self_contact_frames'] and report['passive_force_max_nm']<1e-9)
    (folder/'shape_review.json').write_text(json.dumps(dict(summary=report,actual=actual),ensure_ascii=False,indent=1),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folders',nargs='+',type=Path);a=p.parse_args()
    for folder in a.folders:review(folder)
