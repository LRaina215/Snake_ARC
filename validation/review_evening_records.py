"""Additional actual-state checks of saved radius/reset/T-entry records; no new trial."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import backbone
from snake_control.tee_simulation import TeeSimulation


def review(root):
    radii=[];resets=[]
    for p in sorted((root/'radius_tradeoff').glob('A*.json')):
        record=json.loads(p.read_text(encoding='utf-8'));rows=record['samples'];summary=record['summary']
        trace=np.load(p.with_name(p.stem+'_trace.npz'))
        sim=TeeSimulation('straight',ArcParameters(joints=18,first_joint=7,last_joint=12))
        gaps=[];longest=current=0;z=[];self_frames=0;saturated=0
        for i,row in enumerate(rows):
            sim.data.qpos[:]=trace['qpos'][i];mujoco.mj_forward(sim.model,sim.data)
            z.append(float(backbone(sim.model,sim.data)[:,2].max()))
            current=current+1 if not row['contacts'] else 0;longest=max(longest,current)
            if not row['contacts']:
                # Contact forces below the logging cutoff (.15 N) do not mean
                # geometric separation. Inspect saved-pose contact distances.
                main=sim.model.geom('pipe_main').id
                distances=[float(c.dist) for c in sim.data.contact if main in (c.geom1,c.geom2)]
                if distances:gaps.append(min(distances))
            self_frames+=any(min(sim.model.geom_bodyid[[c.geom1,c.geom2]])>=sim.model.body('head').id
                             and abs(int(sim.model.geom_bodyid[c.geom1])-int(sim.model.geom_bodyid[c.geom2]))>2
                             and c.dist<=0 for c in sim.data.contact)
            saturated+=bool(np.any(abs(trace['actuator_force'][i])>=2.355-1e-6))
        delta=summary['actual_material_delta_m'];requested=summary['direction']*.129
        radii.append(dict(case=p.stem,strict_original_passed=summary['passed'],
            longest_sampled_below_contact_force_cutoff_s=longest*.05,
            below_force_cutoff_max_contact_distance_m=max(gaps,default=None),
            crown_height_range_m=max(z)-min(z),self_contact_frames=self_frames,saturated_frames=saturated,
            actual_material_error_m=None if delta is None else delta-requested))
    for p in sorted((root/'end_reset_round2').glob('*.json')):
        record=json.loads(p.read_text(encoding='utf-8'));s=record['summary'];trace=np.load(p.with_name(p.stem+'_trace.npz'))
        sim=TeeSimulation('straight',ArcParameters(joints=18,first_joint=8,last_joint=13))
        sim.model.body_pos[sim.model.body('pipe_network').id,2]+=s['pipe_axis_height_m']-1.2
        collision=[];saturated=0
        for i,q in enumerate(trace['qpos']):
            sim.data.qpos[:]=q;mujoco.mj_forward(sim.model,sim.data)
            for c in sim.data.contact:
                a,b=sim.model.geom_bodyid[[c.geom1,c.geom2]]-sim.model.body('head').id
                if min(a,b)>=0 and abs(a-b)>2 and c.dist<=0:collision.append((i,int(a),int(b)))
            saturated+=bool(np.any(abs(trace['actuator_force'][i])>=2.355-1e-6))
        resets.append(dict(case=p.stem,self_contact_frames=len(collision),first_self_contact=collision[:1],
                           saturated_frames=saturated,frames=len(trace['qpos'])))
    p=root/'round13/buffer6/locked_gait.json';record=json.loads(p.read_text(encoding='utf-8'))
    rows=record['samples'];s=record['summary'];trace=np.load(p.with_name('locked_gait_trace.npz'))
    wind=[i for i,r in enumerate(rows) if r['stage']=='return_wind']
    sim=TeeSimulation('up60',ArcParameters(**s['parameters']));final=[]
    # Source-return dynamics were checked per saved frame above/in the main
    # review. Here alignment shape is just a bounded sampling diagnostic.
    for i in range(next(i for i,r in enumerate(rows) if r['stage']=='align'),len(rows),20):
        sim.data.qpos[:]=trace['qpos'][i];mujoco.mj_forward(sim.model,sim.data)
        yz=backbone(sim.model,sim.data)[:,1:]-[0,1.2]
        final.append(float(np.ptp(np.unwrap(np.arctan2(yz[:,1],yz[:,0])))/(2*np.pi)))
    tee=dict(source_entry_front_before_tail=s['front_before_tail'],
        wind_front_always_supported=all(any(j<=9 for j in rows[i]['main_links']) for i in wind),
        wind_head_advance_m=rows[wind[-1]]['head_x_m']-rows[wind[0]]['head_x_m'],
        min_sampled_turns_in_align=min(final),self_contact_events=s['self_contact_events'],
        final_stage=s['final_stage'],full_crossing_passed=s['passed'])
    result=dict(radius=radii,end_reset=resets,tee=tee)
    (root/'supplementary_review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();review(a.root)
