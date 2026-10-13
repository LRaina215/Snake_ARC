"""Read-only complete recorded process audit/render; no dynamics reruns."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import gzip,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco,cv2
from snake_control.simulation import ArcSimulation
from snake_control.rolling_arc import ArcParameters

REPORT=ROOT/'validation/reports/active_transfer'


def support_center(row):
    # upward_support index0 is link1; its midpoint is material coordinate1.5.
    force=np.asarray(row['evaluation']['support_by_link_n'])
    return float((np.arange(len(force))+1.5)@force/max(force.sum(),1e-9))


def audit(folder):
    status=json.loads((folder/'status.json').read_text(encoding='utf8'))
    rows=[json.loads(s) for s in gzip.open(folder/'actual_trace.jsonl.gz','rt',encoding='utf8')]
    t=np.array([r['time_s'] for r in rows]);base=min(rows,key=lambda r:abs(r['time_s']-6.))
    baseline=base['evaluation'];summary=dict(baseline=baseline,termination=status['termination'],statistics=status['statistics'])
    for k in ('legacy_enabled_wrap_deg','enabled_gap_fraction','section_turn_deg','plane_rms_m',
              'maximum_curvature_per_m','support_up_n','mean_x_m','mean_y_m','min_z_m'):
        v=np.array([r['evaluation'][k] for r in rows if r['time_s']>=6.])
        summary[k]=dict(min=float(v.min()),max=float(v.max()))
    rr=[r for r in rows if r['time_s']>=6.]
    summary['legacy_wrap_over218_s']=sum(r['evaluation']['legacy_enabled_wrap_deg']>218. for r in rr)*.02
    summary['stage_results']=[]
    groups=[]
    for r in rr:
        key=r['controller']['stage']
        if not groups or groups[-1][0]!=key:groups.append((key,[]))
        groups[-1][1].append(r)
    for stage,group in groups:
        last=group[-1]
        m=last['evaluation']['material'];bm=baseline['material']
        delta={k:m[k]-bm[k] for k in ('crown_joint','head_hanging_m','tail_hanging_m') if m[k] is not None}
        sample_deltas=[r['evaluation']['material']['crown_joint']-bm['crown_joint'] for r in group]
        summary['stage_results'].append(dict(stage=stage,start_s=group[0]['time_s'],end_s=last['time_s'],
            last_delta=delta,crown_delta_range=[min(sample_deltas),max(sample_deltas)],
            support_center_delta_joints=support_center(last)-support_center(base),
            maximum_abs_pipe_displacement_m=max(abs(r['evaluation']['mean_x_m']-baseline['mean_x_m']) for r in group),
            maximum_wrap_deg=max(r['evaluation']['legacy_enabled_wrap_deg'] for r in group)))
    errors=[r['controller']['observation']['crest_joint']-r['evaluation']['material']['crown_joint'] for r in rr
            if (r['controller'].get('observation') or {}).get('valid')]
    summary['observer_error_joints']=dict(rmse=float(np.sqrt(np.mean(np.array(errors)**2))),maximum=float(np.max(abs(np.array(errors)))))
    summary['final_material_delta']=summary['stage_results'][-1]['last_delta']
    summary['final_net_material_transfer_m']=.5*(summary['final_material_delta']['head_hanging_m']-summary['final_material_delta']['tail_hanging_m'])
    summary['support_material_center']=dict(baseline_joint=support_center(base),final_joint=support_center(rows[-1]))
    summary['initial_trace_review']=dict(maximum_wrap_deg=max(r['evaluation']['legacy_enabled_wrap_deg'] for r in rows if r['time_s']<=6.),
        minimum_z_m=min(r['evaluation']['min_z_m'] for r in rows),
        finite=all(np.isfinite(r['actual_qpos']).all() and np.isfinite(r['actual_qvel']).all() for r in rows))
    (folder/'review.json').write_text(json.dumps(summary,indent=2),encoding='utf8')

    fig,axs=plt.subplots(5,1,figsize=(12,12),sharex=True,constrained_layout=True)
    m=[r['evaluation']['material'] for r in rows];ev=[r['evaluation'] for r in rows]
    axs[0].plot(t,[r['legacy_enabled_wrap_deg'] for r in ev],label='wrap over enabled interval (boundary-dependent)')
    axs[0].plot(t,[r['section_turn_deg'] for r in ev],label='actual smooth section turn')
    axs[0].axhline(218,ls='--',color='r');axs[0].axhline(180,ls='--',color='gray');axs[0].set_ylabel('deg')
    for k in ('head_hanging_m','tail_hanging_m'):axs[1].plot(t,[(r[k]-baseline['material'][k])*1000 if r[k] is not None else np.nan for r in m],label=k+' change')
    axs[1].set_ylabel('mm')
    axs[2].plot(t,[r['crown_joint'] for r in m],label='world crown material coordinate')
    axs[2].plot(t,[(r['controller'].get('observation') or {}).get('crest_joint',np.nan) for r in rows],label='angle-only conditional crown')
    axs[2].set_ylabel('joint coordinate')
    axs[3].plot(t,[(r['mean_x_m']-baseline['mean_x_m'])*1000 for r in ev],label='pipe-axis displacement mm')
    axs[3].plot(t,[r['controller']['phase_rad'] for r in rows],label='commanded rolling phase rad')
    axs[4].plot(t,[r['support_up_n'] for r in ev],label='true upward support N')
    axs[4].step(t,[sum(r['enabled']) for r in rows],label='enabled count');axs[4].set_xlabel('physical time s')
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(folder/'timeline.png',dpi=120);plt.close(fig)

    sim=ArcSimulation('straight',ArcParameters(**status['parameters']),-.5)
    for geom in sim.pipe_geoms:sim.model.geom_rgba[geom,3]=.38
    cam=mujoco.MjvCamera();cam.distance=1.6
    selected_times=[.02,2.,4.]+list(np.arange(6.,t[-1]+.1,4.))+[t[-1]]
    selected_times += [e['time_s'] for e in status['events'] if e['kind'] in ('begin_transfer','transfer_complete','observed_drift')]
    selected_times += [rows[max(range(len(rows)),key=lambda i:rows[i]['evaluation']['legacy_enabled_wrap_deg'])]['time_s']]
    indices=sorted(set(int(np.argmin(abs(t-v)))//5*5 for v in selected_times))
    strip=[]
    with mujoco.Renderer(sim.model,height=300,width=500) as renderer:
        writer=cv2.VideoWriter(str(folder/'actual_full.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1000,300))
        if not writer.isOpened():raise RuntimeError('Video writer unavailable')
        try:
            for i in range(0,len(rows),5):
                r=rows[i];sim.data.qpos[:]=r['actual_qpos'];sim.data.qvel[:]=r['actual_qvel'];mujoco.mj_forward(sim.model,sim.data)
                for geom in sim.robot_geoms:
                    link=int(sim.model.geom_bodyid[geom])-sim.model.body('head').id;j=min(max(link-1,0),19)
                    sim.model.geom_rgba[geom]=[.12,.5,.9,1] if r['enabled'][j] else [.55,.3,.7,1]
                cam.lookat[:]=[r['evaluation']['mean_x_m'],0.,.98]
                images=[]
                for az,el in ((75,-15),(0,0)):
                    cam.azimuth=az;cam.elevation=el;renderer.update_scene(sim.data,camera=cam);images.append(renderer.render().copy())
                a=np.concatenate(images,axis=1);e=r['evaluation'];mm=e['material']
                label=f"{status['route']} t={r['time_s']:.1f}s {r['controller']['stage']} crown={mm['crown_joint']:.2f} active={sum(r['enabled'])}"
                cv2.putText(a,label,(10,24),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1)
                if mm['valid']:cv2.putText(a,f"hanging head/tail {mm['head_hanging_m']*1000:.1f}/{mm['tail_hanging_m']*1000:.1f} mm",(10,45),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                writer.write(cv2.cvtColor(a,cv2.COLOR_RGB2BGR))
                if i in indices:strip.append(a)
        finally:writer.release()
    for page in range(math.ceil(len(strip)/5)):
        ims=strip[page*5:(page+1)*5];fig,axs=plt.subplots(len(ims),1,figsize=(12,3.5*len(ims)),squeeze=False,constrained_layout=True)
        for ax,im in zip(axs[:,0],ims):ax.imshow(im);ax.axis('off')
        fig.savefig(folder/f'filmstrip_{page+1}.png',dpi=95);plt.close(fig)
    print(folder.name,summary['final_material_delta'],flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--feedback-return',action='store_true');args=parser.parse_args()
    report=REPORT/'feedback_return' if args.feedback_return else REPORT
    for p in report.glob('*/status.json'):audit(p.parent)
