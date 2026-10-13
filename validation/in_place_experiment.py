"""One planar-feeding case plus at most one documented targeted repair."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,gzip,json,math
from dataclasses import asdict,replace
from datetime import datetime
import numpy as np
import mujoco
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation
from snake_control.in_place_feeding import InPlaceFeeding,FeedingOptions
from validation.active_transfer_experiment import TransferSimulation,snapshot

REPORT=ROOT/'validation/reports/in_place_feeding'


class FeedingSimulation(TransferSimulation):
    def __init__(self,options=FeedingOptions()):
        ArcSimulation.__init__(self,'straight',ArcParameters(amplitude_deg=options.amplitude_deg),-.5)
        self.lengths=np.full(21,.0645)
        self.controller=InPlaceFeeding(self.lengths,options)
        self.target,self.enabled=self.controller.targets.copy(),self.controller.enabled.copy()
        self.next_feedback=self.next_control=0.
        self.last_feedback=None


def spatial_truth(sim,row):
    """Privileged positions ONLY for audit, never passed to the controller."""
    points=np.asarray(row['backbone_world_m'])
    locations,weights=[],[]
    for j,length in enumerate(np.linalg.norm(np.diff(points,axis=0),axis=1)):
        for u in (np.arange(16)+.5)/16:
            locations.append((1-u)*points[j]+u*points[j+1]);weights.append(length/16)
    locations=np.array(locations);w=np.maximum(0.,locations[:,2]-locations[:,2].max()+.0080625)*weights
    crown=(w[:,None]*locations).sum(0)/w.sum()
    contacts=[]
    for i in range(sim.data.ncon):
        c=sim.data.contact[i];pair={int(c.geom1),int(c.geom2)}
        if not pair&sim.pipe_geoms or not pair&sim.robot_geoms:continue
        robot=next(iter(pair&sim.robot_geoms));f=np.zeros(6);mujoco.mj_contactForce(sim.model,sim.data,i,f)
        world=np.array(c.frame).reshape(3,3).T@f[:3]
        if robot==c.geom1:world*=-1
        contacts.append(dict(position_world_m=c.pos.tolist(),up_n=max(0.,float(world[2]))))
    total=sum(c['up_n'] for c in contacts)
    support=None if total<1e-9 else sum(c['up_n']*np.array(c['position_world_m']) for c in contacts)/total
    return dict(crown_world_m=crown.tolist(),support_point_world_m=None if support is None else support.tolist(),
                contacts=contacts,mean_body_world_m=points.mean(0).tolist())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repair',action='store_true')
    parser.add_argument('--report-root',type=Path,default=REPORT,
                        help='Use a new directory to reproduce without overwriting this round')
    args=parser.parse_args();report=args.report_root.resolve()
    report.mkdir(parents=True,exist_ok=True)
    if len(list(report.glob('*/status.json')))>=2:raise RuntimeError('One main and one repair budget exhausted')
    if args.repair and not (report/'repair_plan.md').exists():raise RuntimeError('Document targeted repair before running')
    if not args.repair and list(report.glob('main_*/status.json')):raise RuntimeError('Main case already ran')
    options=replace(FeedingOptions(),amplitude_deg=200./3.,feed_gain_s=.15,
                    maximum_progress_rate_s=.06,maximum_feed_s=35.) if args.repair else FeedingOptions()
    sim=FeedingSimulation(options)
    folder=report/(('repair_' if args.repair else 'main_')+datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder.mkdir()
    for path in ('snake_control/in_place_feeding.py','validation/in_place_experiment.py'):
        (folder/Path(path).name).write_text((ROOT/path).read_text(encoding='utf8'),encoding='utf8')
    statistics=dict(steps=0,passive_torque_max_nm=0.,active_torque_max_nm=0.,joint_limit_excess_max_rad=0.,
        saturation_joint_steps=0,floor_contact_steps=0,maximum_enabled_count=0,maximum_shared_target_step_deg=0.)
    limits=sim.model.jnt_range[sim.model.actuator_trnid[:,0]];floor=sim.model.geom('floor').id
    reason,error='duration',None
    baseline=None;previous_target,previous_mask=sim.target.copy(),sim.enabled.copy()
    try:
        with gzip.open(folder/'actual_trace.jsonl.gz','wt',encoding='utf8') as stream:
            for frame in range(4250 if args.repair else 3250):
                for _ in range(20):
                    sim.step();statistics['steps']+=1
                    tau=sim.data.actuator_force;q=sim.data.qpos[sim.drive.qadr]
                    statistics['passive_torque_max_nm']=max(statistics['passive_torque_max_nm'],float(np.max(abs(tau[~sim.enabled]),initial=0.)))
                    statistics['active_torque_max_nm']=max(statistics['active_torque_max_nm'],float(np.max(abs(tau))))
                    statistics['joint_limit_excess_max_rad']=max(statistics['joint_limit_excess_max_rad'],float(np.maximum(np.maximum(limits[:,0]-q,q-limits[:,1]),0).max()))
                    statistics['saturation_joint_steps']+=int(np.count_nonzero(abs(tau)>=sim.p.torque_limit_nm-1e-6))
                    statistics['floor_contact_steps']+=int(any(floor in (int(c.geom1),int(c.geom2)) and bool({int(c.geom1),int(c.geom2)}&sim.robot_geoms) for c in sim.data.contact))
                    statistics['maximum_enabled_count']=max(statistics['maximum_enabled_count'],int(sim.enabled.sum()))
                row=snapshot(sim);spatial=spatial_truth(sim,row)
                if sim.data.time>=6. and baseline is None:baseline=spatial.copy()
                if baseline:
                    for key in ('crown_world_m','support_point_world_m','mean_body_world_m'):
                        if spatial[key] is not None and baseline[key] is not None:
                            spatial[key+'_delta']= (np.array(spatial[key])-baseline[key]).tolist()
                row['evaluation']['spatial']=spatial
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                shared=sim.enabled&previous_mask
                statistics['maximum_shared_target_step_deg']=max(statistics['maximum_shared_target_step_deg'],float(np.degrees(abs(sim.target[shared]-previous_target[shared])).max()))
                previous_target,previous_mask=sim.target.copy(),sim.enabled.copy()
                if statistics['floor_contact_steps'] or sim.controller.fault or not np.isfinite(sim.data.qpos).all():
                    reason='floor_or_controller_fault';break
                if sim.controller.completed:reason='task_complete';break
                if frame%250==249:
                    m=row['evaluation']['material'];delta=spatial.get('support_point_world_m_delta',[0,0,0])
                    print(f"t={sim.data.time:.1f}s {sim.controller.stage} u={sim.controller.progress:.3f} crown={m['crown_joint']:.3f} "
                          f"support_shift={np.linalg.norm(delta)*1000:.2f}mm tails={m['head_hanging_m']},{m['tail_hanging_m']}",flush=True)
    except Exception as exc:
        error=repr(exc);raise
    finally:
        status=dict(scene='straight',route='in_place_feeding',parameters=asdict(sim.p),options=asdict(options),
            final_time_s=float(sim.data.time),termination=reason,error=error,statistics=statistics,
            controller=sim.controller.snapshot(),events=sim.controller.events,spatial_baseline=baseline,
            pipe_centerline=sim.model.numeric('pipe_centerline').data.reshape(-1,4).tolist(),
            pipe_friction=sim.model.geom('pipe_straight').friction.tolist(),gravity=sim.model.opt.gravity.tolist(),
            hardware_run=False,control_period_s=.02,feedback_period_s=.2,physics_period_s=.001)
        (folder/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf8')
        print(folder,reason,error,flush=True)


if __name__=='__main__':main()
