"""One declared fixed-window experiment; no automatic test matrix."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,time
from dataclasses import replace,asdict
import numpy as np
import mujoco
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation,backbone,ROOT
from snake_control.fixed_section_simulation import FixedSectionSimulation
from snake_control.fixed_section_compliance import SectionOptions
from snake_control.arc_shape import ArcShapeEvaluator

# Frozen after R0: 212.71° maximum + 5° reconstruction/numerical margin, rounded.
# Target section is independently limited to 190°. Lower coverage is a nominal
# half-circle minus 30°. These are engineering acceptance bounds, not paper claims.
ACTUAL_WRAP_MAX=218.
ACTUAL_WRAP_MIN=150.
GAP_MIN=.4


def run(scene,label,output,duration=34,gain=.02,baseline=False,hold_phase=None,start=-.5,
        initial_state=None,initial_index=0,exit_cycles=0):
    p=ArcParameters.load(ROOT/'configs/step1.json')
    if hold_phase is not None:p=replace(p,initial_phase_rad=hold_phase)
    if initial_state is not None:p=replace(p,settle_s=0.)
    options=SectionOptions(gain=gain)
    sim=ArcSimulation(scene,p,start) if baseline else FixedSectionSimulation(scene,p,start,options,hold_phase)
    if initial_state is not None:
        with np.load(initial_state) as saved:
            sim.data.qpos[:]=saved['qpos'][initial_index]
            sim.data.qvel[:]=saved['qvel'][initial_index]
        mujoco.mj_forward(sim.model,sim.data)
    evaluation=ArcShapeEvaluator(sim.route)
    output.mkdir(parents=True,exist_ok=True)
    states=[]; velocities=[]; skeleton=[]; records=[]; curves=[]; projections=[]; axes=[]
    next_sample=0.;work=0.;passive_max=0.;torque_max=0.;wall=time.monotonic()
    warnings=np.array([w.number for w in sim.data.warning]);limit_error=0.
    pipe_radii=sim.model.numeric('pipe_centerline').data.reshape(-1,4)[:,3]
    def record():
        row=sim.measure();pts=backbone(sim.model,sim.data)
        q=sim.data.qpos[sim.drive.qadr]
        shape=evaluation.measure(pts,np.degrees(q),p.first_joint,p.last_joint)
        local=pts[p.first_joint:p.last_joint+2]
        shape['gap_fraction']=float(np.linalg.norm(local[-1]-local[0])/np.linalg.norm(np.diff(local,axis=0),axis=1).sum())
        row['shape']=shape
        row['progress_m']=float(sim.route_coordinates(pts.mean(axis=0))[0][0])
        row['local_pipe_radius_m']=float(np.interp(sim.route_coordinates(local.mean(axis=0))[0][0],sim.route_s,pipe_radii))
        row['enabled']=p.enabled.tolist();row['geometry']=row.get('geometry',{})
        g=row['geometry'];curve=np.asarray(g.get('section_curve_m',[[0.,0.],[0.,0.]]))
        distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
        x=np.linspace(0,max(distance[-1],1e-12),64)
        curves.append(np.column_stack([np.interp(x,distance,curve[:,j]) for j in (0,1)]))
        projections.append(g.get('projected_m',[[0.,0.]]*9));axes.append(g.get('axis',[0.,0.,0.]))
        states.append(sim.data.qpos.copy());velocities.append(sim.data.qvel.copy());skeleton.append(pts)
        records.append(row)
    record();initial_progress=records[0]['progress_m']
    period=2*np.pi/p.omega_rad_s
    exit_time=None
    while sim.data.time<duration:
        sim.step(hold_phase if baseline else None)
        force=sim.data.actuator_force
        passive_max=max(passive_max,float(np.max(abs(force[~p.enabled]),initial=0)))
        torque_max=max(torque_max,float(np.max(abs(force))))
        work+=float(np.maximum(force*sim.data.qvel[sim.drive.vadr],0).sum())*sim.model.opt.timestep
        limit_error=max(limit_error,float(np.max(np.maximum(abs(sim.data.qpos[sim.drive.qadr])-np.pi/2,0))))
        if not np.isfinite(sim.data.qpos).all():raise RuntimeError('Nonfinite state')
        if sim.data.time>=next_sample:
            record();next_sample+=.1
            if scene!='straight' and hold_phase is None:
                if exit_time is None and records[-1]['min_body_progress_m']>sim.feature[-1]+.05:
                    exit_time=float(sim.data.time)
                if exit_cycles and exit_time is not None and sim.data.time>=exit_time+exit_cycles*period:
                    break
    audited=[r for r in records if r['time_s']>=p.settle_s+1.]
    valid=[r for r in audited if r['shape']['wrap_valid']]
    unsupported=streak=0.
    for r in audited:
        streak=streak+.1 if r['normal_force_n']<.1 else 0.;unsupported=max(unsupported,streak)
    final=records[-1];wrap=[r['shape']['wrap_deg'] for r in valid]
    geometry=[r['geometry'] for r in audited if r['geometry'].get('target_section_turn_deg') is not None]
    progressing=final['progress_m']-initial_progress
    checks=dict(passive_zero=passive_max<1e-12,torque_bounded=torque_max<=p.torque_limit_nm+1e-9,
        no_ground_support=max(r['ground_contact_force_n'] for r in audited)<.1,
        support_retained=unsupported<.5,limits=limit_error<.02,
        solver_stable=bool(np.all(np.array([w.number for w in sim.data.warning])==warnings)),
        actual_open=bool(wrap) and max(wrap)<=ACTUAL_WRAP_MAX and min(wrap)>=ACTUAL_WRAP_MIN,
        shape_observable=len(valid)==len(audited),
        opening=min(r['shape']['gap_fraction'] for r in audited)>=GAP_MIN,
        progress=True if hold_phase is not None else progressing>.3)
    if not baseline:
        checks['target_constraint']=bool(geometry) and max(g['target_section_turn_deg'] for g in geometry)<=190.01
        checks['reference_feasible']=bool(geometry) and all(g['status']=='running' for g in geometry)
    if scene!='straight' and hold_phase is None:
        checks['whole_body_beyond_feature']=final['min_body_progress_m']>sim.feature[-1]+.05
        if exit_cycles:
            checks['exit_observation']=exit_time is not None and final['time_s']-exit_time>=exit_cycles*period
    checks={k:bool(v) for k,v in checks.items()}
    worst=max(valid,key=lambda r:r['shape']['wrap_deg']) if valid else None
    cycle_max=[]
    for a in np.arange(p.settle_s,final['time_s']-period+1e-8,period):
        cycle=[r['shape']['wrap_deg'] for r in records if a<=r['time_s']<a+period and r['shape']['wrap_valid']]
        if cycle:cycle_max.append(float(max(cycle)))
    report=dict(label=label,scene=scene,baseline=baseline,passed=all(checks.values()),checks=checks,
        parameters=asdict(p),options=asdict(options),duration_s=final['time_s'],maximum_duration_s=duration,
        exit_time_s=exit_time,post_exit_s=None if exit_time is None else final['time_s']-exit_time,
        requested_exit_cycles=exit_cycles,hold_phase=hold_phase,start_x=start,
        initial_state=None if initial_state is None else str(initial_state),initial_index=initial_index,
        progress_m=progressing,whole_body_final_s=final['min_body_progress_m'],feature_span_m=sim.feature.tolist(),
        maximum_wrap_deg=None if not wrap else max(wrap),minimum_wrap_deg=None if not wrap else min(wrap),
        worst_wrap_time_s=None if worst is None else worst['time_s'],cycle_max_wrap_deg=cycle_max,
        final_shape=final['shape'],maximum_unsupported_s=unsupported,positive_work_j=work,
        passive_max_nm=passive_max,maximum_torque_nm=torque_max,
        final_geometry=final['geometry'],wall_s=time.monotonic()-wall,
        acceptance=dict(actual_wrap_min=ACTUAL_WRAP_MIN,actual_wrap_max=ACTUAL_WRAP_MAX,gap_fraction_min=GAP_MIN,
                        target_section_turn_max_deg=190.))
    (output/f'{label}.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    fields=['time_s','phase_rad','progress_m','min_body_progress_m','normal_force_n','ground_contact_force_n','local_pipe_radius_m']
    gfields=['measured_section_turn_deg','target_section_turn_deg','fit_rms_m','fitted_body_radius_m','axis_confidence']
    sfields=list(final['shape'])
    np.savez_compressed(output/f'{label}_trace.npz',qpos=states,qvel=velocities,backbone=skeleton,
        metrics=[[r[f] for f in fields] for r in records],metric_fields=fields,
        targets_rad=[r['target_rad'] for r in records],enabled=np.tile(p.enabled,(len(records),1)),
        actuator_force_nm=[r['actuator_force_nm'] for r in records],shape_fields=sfields,
        shape=[[r['shape'][f] for f in sfields] for r in records],geometry_fields=gfields,
        geometry=[[r['geometry'].get(f,0.) for f in gfields] for r in records],
        projected_m=projections,section_curve_m=curves,axis=axes)
    print(json.dumps({k:report[k] for k in ['label','passed','checks','progress_m','maximum_wrap_deg','cycle_max_wrap_deg','wall_s']}),flush=True)
    return report


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--scene',default='straight');a.add_argument('--label',required=True)
    a.add_argument('--output',type=Path,default=ROOT/'validation/reports/fixed_section')
    a.add_argument('--duration',type=float,default=34);a.add_argument('--gain',type=float,default=.02)
    a.add_argument('--baseline',action='store_true');a.add_argument('--hold-phase',type=float)
    a.add_argument('--start',type=float,default=-.5)
    a.add_argument('--initial-state',type=Path);a.add_argument('--initial-index',type=int,default=0)
    a.add_argument('--exit-cycles',type=int,default=0,help='Stop after this many cycles beyond the whole-body exit; duration remains the upper limit.')
    args=a.parse_args()
    run(args.scene,args.label,args.output,args.duration,args.gain,args.baseline,args.hold_phase,args.start,
        args.initial_state,args.initial_index,args.exit_cycles)
