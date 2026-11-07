"""Full actual-state review of source-return ARC transitions."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import backbone
from snake_control.tee_simulation import TeeSimulation, PIPE_HEIGHT


def review(folder):
    record=json.loads((folder/'locked_gait.json').read_text(encoding='utf-8'))
    summary,rows=record['summary'],record['samples']
    trace=np.load(folder/'locked_gait_trace.npz')
    sim=TeeSimulation(summary['scene'],ArcParameters(**summary['parameters']))
    qpos,targets,masks,forces=(trace[k] for k in ('qpos','targets','masks','actuator_force'))
    front_count=summary['controller_status']['front_count']
    finished=[];intake=[];opening=[];collisions=[]
    settle_indices=[i for i,r in enumerate(rows) if r['stage']=='arc_settle']
    preparation_end=max(settle_indices,default=0)
    for i,row in enumerate(rows):
        sim.data.qpos[:]=qpos[i];mujoco.mj_forward(sim.model,sim.data)
        points=backbone(sim.model,sim.data);yz=points[:,1:]-[0,PIPE_HEIGHT]
        if row['stage']=='return_wind':
            intake.append(dict(time_s=row['time_s'],head_x_m=row['head_x_m'],center_x_m=row['center_x_m'],
                front_turn_deg=float(np.degrees(np.ptp(np.unwrap(np.arctan2(yz[:front_count+2,1],yz[:front_count+2,0]))))),
                targets=targets[i,:front_count].tolist(),contacts=[j for j in row['main_links'] if j<=front_count]))
        if row['stage'] in ('complete','roll'):
            finished.append(dict(time_s=row['time_s'],turns=float(np.ptp(np.unwrap(np.arctan2(yz[:,1],yz[:,0])))/(2*np.pi)),
                max_radius_m=float(np.linalg.norm(yz,axis=1).max()),contacts=row['main_links']))
        # Once the leading helix grows, total pipe coverage is intentionally
        # more than a semicircle. Only the exclusive ARC/transfer phase is an
        # open-ARC test, not every frame before motor 1 becomes enabled.
        if i<=preparation_end:
            near=np.linalg.norm(yz,axis=1)<.145
            if near.sum()>=3:
                polar=np.sort(np.mod(np.arctan2(yz[near,1],yz[near,0]),2*np.pi))
                opening.append(float(360.-np.degrees(np.diff(np.r_[polar,polar[0]+2*np.pi]).max())))
        head=sim.model.body('head').id
        for c in sim.data.contact:
            a,b=sim.model.geom_bodyid[[c.geom1,c.geom2]]-head
            if min(a,b)>=0 and abs(a-b)>2 and c.dist<=0:collisions.append((row['time_s'],int(a),int(b)))
    passive=float(np.max(abs(forces[~masks]),initial=0.))
    shape=bool(finished and min(x['turns'] for x in finished)>=1.7 and max(x['max_radius_m'] for x in finished)<.145)
    fore_motion=0. if not intake else float(np.max(abs(np.array(intake[-1]['targets'])-intake[0]['targets'])))
    advance=None if not intake else intake[-1]['head_x_m']-intake[0]['head_x_m']
    report=dict(case=summary['case'],controller_passed=summary['passed'],actual_final_shape_passed=shape,
        frames=len(rows),final_min_turns=min((x['turns'] for x in finished),default=None),
        final_max_radius_m=max((x['max_radius_m'] for x in finished),default=None),
        preparation_max_near_pipe_coverage_deg=max(opening,default=None),
        source_front_target_change_rad=fore_motion,source_head_advance_m=advance,
        source_intake_supported=bool(intake and all(x['contacts'] for x in intake)),
        passive_force_max_nm=passive,self_contact_frames=len(collisions),first_self_contact=collisions[:1],
        floor_frames=sum(r['ground_force_n']>1. for r in rows),saturated_frames=sum(bool(r['saturated_joints']) for r in rows),
        max_joint_deg=max(r['actual_max_joint_deg'] for r in rows),failure=summary['failure'])
    report['passed']=bool(summary['passed'] and shape and passive<1e-10 and not collisions and
        not report['floor_frames'] and opening and max(opening)<300. and fore_motion>.1 and
        advance is not None and advance>.01 and report['source_intake_supported'])
    (folder/'source_review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);a=p.parse_args()
    print(json.dumps(review(a.folder)))
