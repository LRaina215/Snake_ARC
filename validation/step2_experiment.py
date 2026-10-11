"""Step2 physical trials; fixed body evaluation points, no imposed root motion."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from dataclasses import asdict, replace
import json
import platform
import time
import numpy as np
import mujoco
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation, ROOT, backbone
from snake_control.adaptive_simulation import AdaptiveArcSimulation
from snake_control.arc_compliance import SlidingArcOptions

CASES = [
    ('straight','straight',30.,-.5,{},2.),
    ('bend90','bend90',70.,-.35,{},2.),
    ('elbow90','elbow90',60.,-.25,{},2.),
    ('taper_straight','taper_straight',70.,-.5,{},2.),
    ('taper_bend90','taper_bend90',90.,-.5,{},2.),
    ('front_taper_straight','taper_straight',70.,-.5,dict(first_joint=4,last_joint=9),2.),
    ('front_taper_bend90','taper_bend90',90.,-.5,dict(first_joint=4,last_joint=9),2.),
    ('rear_taper_bend90','taper_bend90',90.,-.5,dict(first_joint=12,last_joint=17),2.),
    ('phase_taper_bend90','taper_bend90',90.,-.5,{},4.),
]

def experiment(case, output, baseline=False, initial_state=None):
    label,scene,duration,start,overrides,takeover=case
    p=replace(ArcParameters.load(ROOT/'configs/step1.json'),**overrides)
    options=SlidingArcOptions()
    sim=(ArcSimulation(scene,p,start) if baseline else
         AdaptiveArcSimulation(scene,p,start,options,takeover))
    if initial_state is not None:
        sim.adopt_supported_pose(initial_state)
        takeover=0.
    records=[]; states=[]; velocities=[]; points=[]
    def measure():
        row=sim.measure()
        center=backbone(sim.model,sim.data).mean(axis=0)
        row['progress_m']=float(sim.route_coordinates(center)[0][0])
        row['enabled']=getattr(sim,'enabled',p.enabled).tolist()
        return row
    initial=measure(); next_sample=0.; work=0.; passive_max=0.; force_max=0.; violation=0.
    warnings_before=np.array([w.number for w in sim.data.warning]); wall=time.monotonic()
    while sim.data.time<duration:
        sim.step()
        mask=getattr(sim,'enabled',p.enabled)
        passive_max=max(passive_max,float(np.max(abs(sim.data.actuator_force[~mask]),initial=0)))
        force_max=max(force_max,float(np.max(abs(sim.data.actuator_force))))
        work+=np.maximum(sim.data.actuator_force*sim.data.qvel[sim.drive.vadr],0).sum()*sim.model.opt.timestep
        violation=max(violation,float(np.max(np.maximum(abs(sim.data.qpos[sim.drive.qadr])-np.pi/2,0))))
        if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all():
            raise RuntimeError('Nonfinite physical state')
        if sim.data.time>=next_sample:
            records.append(measure()); states.append(sim.data.qpos.copy()); velocities.append(sim.data.qvel.copy())
            points.append(backbone(sim.model,sim.data));next_sample+=.1
    final=records[-1]; delta=final['progress_m']-initial['progress_m']; unsupported=streak=0.
    for row in records:
        if row['time_s']<takeover:continue
        streak=streak+.1 if row['normal_force_n']<.1 else 0.
        unsupported=max(unsupported,streak)
    ground=max(r['ground_contact_force_n'] for r in records)
    controller=getattr(sim,'controller',None)
    checks=dict(passive_drive_zero=passive_max<1e-12,torque_bounded=force_max<=p.torque_limit_nm+1e-9,
                no_ground_support=ground<.1,support_retained=unsupported<.5,joint_limits=violation<.02,
                no_new_solver_warnings=bool(np.all(np.array([w.number for w in sim.data.warning])==warnings_before)),
                forward_progress=delta>.3)
    if scene!='straight':checks['whole_body_beyond_feature']=final['min_body_progress_m']>sim.feature[-1]+.05
    if not baseline:checks['controller_entered']=controller is not None
    checks={key:bool(value) for key,value in checks.items()}
    report=dict(label=label,scene=scene,baseline=baseline,passed=all(checks.values()),checks=checks,
                parameters=asdict(p),options=asdict(options),start_x=start,takeover_s=takeover,
                simulation_duration_s=final['time_s'],wall_duration_s=time.monotonic()-wall,
                progress_delta_m=delta,min_body_final_progress_m=final['min_body_progress_m'],
                feature_span_m=sim.feature.tolist(),passive_actuator_force_max_nm=passive_max,
                maximum_torque_nm=force_max,max_ground_contact_force_n=ground,max_unsupported_s=unsupported,
                positive_actuator_work_j=float(work),window_events=[] if controller is None else controller.events,
                final_compliance=None if controller is None else controller.snapshot())
    fields=['time_s','phase_rad','progress_m','min_body_progress_m','normal_force_n','ground_contact_force_n']
    enabled=np.array([r['enabled'] for r in records],bool)
    amplitude=np.array([r.get('compliance',None)['amplitude_deg'] if r.get('compliance') else [p.amplitude_deg]*p.joints for r in records])
    suffix='_baseline' if baseline else ''
    output.mkdir(parents=True,exist_ok=True)
    (output/f'{label}{suffix}.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(output/f'{label}{suffix}_trace.npz',qpos=states,qvel=velocities,backbone=points,
                        metrics=[[r[k] for k in fields] for r in records],metric_fields=fields,enabled=enabled,
                        amplitude_deg=amplitude,targets_rad=[r['target_rad'] for r in records],
                        actuator_force_nm=[r['actuator_force_nm'] for r in records],
                        support_force_n=[r.get('support_force_n',[0.]*p.joints) for r in records],
                        window=[r['compliance']['window'] if r.get('compliance') else [p.first_joint,p.last_joint] for r in records])
    print(json.dumps(dict(case=label,baseline=baseline,passed=report['passed'],delta_m=round(delta,4),
                         events=len(report['window_events']),checks=checks)),flush=True)
    return report

def pose_experiment(output):
    source=output/'front_taper_bend90_trace.npz'
    with np.load(source) as trace:
        index=int(np.argmin(abs(trace['metrics'][:,0]-26.7)))
        q=trace['qpos'][index].copy()
    # Change the hanging tail posture. Neither root nor active arc is imposed
    # again after this single initial placement; old drives all start disabled.
    reference=AdaptiveArcSimulation('taper_bend90',ArcParameters())
    q[reference.drive.qadr[13:]]=np.radians([12,-18,22,-12,18,-8,10])
    report=experiment(('pose_takeover','taper_bend90',65.,-.5,{},0.),output,initial_state=q)
    (output/'pose_initial_state.npz').parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'pose_initial_state.npz',qpos=q,source=str(source.name),source_time_s=26.7)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',choices=[c[0] for c in CASES])
    parser.add_argument('--output',type=Path,default=ROOT/'validation/reports/step2')
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--pose-takeover',action='store_true',help='从已存前部弯管中间姿态接管，改变悬垂段角度')
    args=parser.parse_args()
    if args.pose_takeover:
        result=pose_experiment(args.output)
        if not result['passed']:raise SystemExit(1)
        return
    results=[experiment(c,args.output,args.baseline) for c in CASES if not args.cases or c[0] in args.cases]
    if not args.cases and not args.baseline:results.append(pose_experiment(args.output))
    report=dict(environment=dict(python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__),
                passed=all(r['passed'] for r in results),cases=results)
    name='baseline_summary.json' if args.baseline else 'summary.json'
    (args.output/name).write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
