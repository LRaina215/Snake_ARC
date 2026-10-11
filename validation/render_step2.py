"""Render archived actual Step2 states, plus support/window/amplitude evidence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.rolling_arc import ArcParameters
from snake_control.adaptive_simulation import AdaptiveArcSimulation
from snake_control.simulation import ROOT

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'validation/reports/step2')
    parser.add_argument('--video',action='store_true')
    args=parser.parse_args();path=args.input
    # Legacy scene ID taper_bend90 reuses a 45-degree source bend.
    labels={'taper_straight':'Tapered straight pipe','taper_bend90':'Taper + 45 degree bend',
            'front_taper_bend90':'Front placement + sliding window'}
    fig,axes=plt.subplots(3,3,figsize=(14,9),constrained_layout=True)
    for row,(name,label) in enumerate(labels.items()):
        report=json.loads((path/f'{name}.json').read_text(encoding='utf-8'))
        with np.load(path/f'{name}_trace.npz') as trace:
            m={key:trace['metrics'][:,i] for i,key in enumerate(trace['metric_fields'])}
            sim=AdaptiveArcSimulation(report['scene'],ArcParameters(**report['parameters']),report['start_x'])
            sim.model.vis.global_.offwidth=1000;sim.model.vis.global_.offheight=650
            floor=sim.model.geom('floor').id
            sim.model.geom_matid[floor]=-1;sim.model.geom_rgba[floor]=[.9,.92,.95,1]
            cam=mujoco.MjvCamera();cam.lookat[:]=[-.05,.35,1.0];cam.distance=3.0
            cam.azimuth,cam.elevation=135,-22
            t=m['time_s'];events=report['window_events']
            middle=events[0]['time_s']+report['takeover_s']+1.3 if events else t[np.argmin(abs(m['progress_m']-report['feature_span_m'][0]))]
            indices=[np.argmin(abs(t-2)),np.argmin(abs(t-middle)),len(t)-1]
            with mujoco.Renderer(sim.model,height=650,width=1000) as renderer:
                def frame(index):
                    sim.data.qpos[:]=trace['qpos'][index];sim.data.qvel[:]=trace['qvel'][index]
                    sim.enabled=trace['enabled'][index];sim.support_force=trace['support_force_n'][index]
                    sim.color_window();mujoco.mj_forward(sim.model,sim.data)
                    renderer.update_scene(sim.data,camera=cam)
                    return renderer.render().copy()
                for col,index in enumerate(indices):
                    rendered=frame(index);axes[row,col].imshow(rendered);axes[row,col].axis('off')
                    first,last=trace['window'][index]
                    axes[row,col].set_title(f'{label}\nt={t[index]:.1f}s | window J{first}-{last}',fontsize=10)
                    plt.imsave(path/f'{name}_stage{col+1}.png',rendered)
                if args.video and name=='front_taper_bend90':
                    import cv2
                    video=ROOT/'validation/results/videos/step2_front_taper_bend90_4x.mp4'
                    video.parent.mkdir(parents=True,exist_ok=True)
                    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(1000,650))
                    if not writer.isOpened():raise RuntimeError('Video writer failed')
                    try:
                        for index in range(0,len(t),2):
                            img=cv2.cvtColor(frame(index),cv2.COLOR_RGB2BGR);first,last=trace['window'][index]
                            cv2.putText(img,f'Step2 | t={t[index]:.1f}s | J{first}-{last} | 4x replay',
                                        (20,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(35,35,35),2)
                            writer.write(img)
                    finally:writer.release()
        print('Rendered actual states:',name,flush=True)
    fig.suptitle('Step2 actual dynamics | Blue: enabled | Green: observed support | Purple: passive',fontsize=13)
    fig.savefig(path/'actual_motion.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(3,1,figsize=(12,9),constrained_layout=True)
    with np.load(path/'front_taper_bend90_trace.npz') as trace:
        t=trace['metrics'][:,0]
        axes[0].imshow(trace['enabled'].T,origin='lower',aspect='auto',extent=[t[0],t[-1],.5,20.5],
                       cmap='Blues',vmin=0,vmax=1,interpolation='nearest')
        for j in range(20):
            observed=trace['support_force_n'][:,j]>.5
            axes[0].scatter(t[observed],np.full(observed.sum(),j+1),s=2,color='green')
        axes[0].set(ylabel='Joint number',title='Front placement: enabled window (blue) and observed support (green)',yticks=range(1,21))
        for j in (3,5,7,9):axes[1].plot(t,trace['amplitude_deg'][:,j],label=f'J{j+1}')
        axes[1].set(ylabel='Local amplitude (deg)',title='Measured-angle yielding and window widening');axes[1].legend()
    for name in ('straight','bend90','elbow90','taper_straight','taper_bend90'):
        for suffix,style in (('', '-'),('_baseline','--')):
            with np.load(path/f'{name}{suffix}_trace.npz') as trace:
                metrics=trace['metrics'];axes[2].plot(metrics[:,0],metrics[:,2]-metrics[0,2],style,
                                                   label=name+(' Step1' if suffix else ' Step2'))
    axes[2].set(xlabel='Simulation time (s)',ylabel='Whole-backbone route travel (m)',title='Same scene / same evaluation points; solid Step2, dashed Step1')
    axes[2].legend(ncol=2,fontsize=8)
    for ax in axes[1:]:ax.grid(alpha=.2)
    fig.savefig(path/'metrics.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
