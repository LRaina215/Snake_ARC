"""Review full recorded dynamics, side/end filmstrips and optional full video."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
from snake_control.simulation import ArcSimulation,ROOT
from snake_control.rolling_arc import ArcParameters


def render(label,path,video=False):
    report=json.loads((path/f'{label}.json').read_text(encoding='utf-8'))
    with np.load(path/f'{label}_trace.npz') as z:
        t=z['metrics'][:,0];s=z['shape'];sf=list(z['shape_fields']);g=z['geometry'];gf=list(z['geometry_fields'])
        g=g.copy();g[g[:,gf.index('target_section_turn_deg')]==0]=np.nan
        wrap=s[:,sf.index('wrap_deg')]; gap=s[:,sf.index('gap_fraction')]
        observable=s[:,sf.index('wrap_valid')].astype(bool)
        fig,axes=plt.subplots(4,1,figsize=(12,10),constrained_layout=True)
        axes[0].plot(t,np.where(observable,wrap,np.nan),label='actual pipe wrap (observable)')
        if not observable.all():
            axes[0].plot(t,np.where(observable,np.nan,wrap),color='gray',ls=':',label='invalid axis projection; not an accepted wrap')
        axes[0].axhline(218,color='r',ls='--');axes[0].axhline(150,color='r',ls='--')
        axes[0].plot(t,g[:,gf.index('measured_section_turn_deg')],label='estimated section turn')
        axes[0].plot(t,g[:,gf.index('target_section_turn_deg')],label='target section turn');axes[0].legend(fontsize=8)
        axes[0].set(ylabel='degrees',title=label+' | entire recorded process')
        axes[1].plot(t,gap,label='end opening / active length');axes[1].axhline(.4,color='r',ls='--');axes[1].legend()
        if z['metrics'].shape[1]>6:axes[2].plot(t,z['metrics'][:,6]*1000,label='true pipe radius (evaluation)')
        axes[2].plot(t,g[:,gf.index('fitted_body_radius_m')]*1000,label='fitted body radius')
        axes[2].set(ylabel='mm');axes[2].legend(fontsize=8)
        axes[3].plot(t,z['metrics'][:,2]-z['metrics'][0,2],label='whole body progress')
        axes[3].plot(t,z['metrics'][:,4]/100,label='normal force / 100');axes[3].set(xlabel='physical time (s)');axes[3].legend()
        for ax in axes:ax.grid(alpha=.2)
        fig.savefig(path/f'{label}_review.png',dpi=120);plt.close(fig)
        p=ArcParameters(**report['parameters']);sim=ArcSimulation(report['scene'],p,report['start_x'])
        # A long axial view otherwise hides the active arc behind the pipe.
        # Transparency is rendering-only; replayed collision/state is unchanged.
        for geom in sim.pipe_geoms:sim.model.geom_rgba[geom,3]=.45
        for geom in sim.robot_geoms:
            link=int(sim.model.geom_bodyid[geom])-sim.model.body('head').id
            j=min(max(link-1,0),19)
            sim.model.geom_rgba[geom]=[.12,.5,.9,1] if p.enabled[j] else [.55,.3,.7,1]
        camera=mujoco.MjvCamera();camera.distance=1.65
        def frame(i,end=False):
            sim.data.qpos[:]=z['qpos'][i];sim.data.qvel[:]=z['qvel'][i];mujoco.mj_forward(sim.model,sim.data)
            active=z['backbone'][i,p.first_joint:p.last_joint+2];center=active.mean(axis=0)
            route=sim.route;delta=np.diff(route,axis=0);idx=np.argmin(np.linalg.norm((route[1:]+route[:-1])/2-center,axis=1))
            heading=np.degrees(np.arctan2(delta[idx,1],delta[idx,0]))
            camera.lookat[:]=center+[0,0,-.17]
            camera.azimuth=heading if end else heading+75;camera.elevation=0 if end else -15
            renderer.update_scene(sim.data,camera=camera);return renderer.render().copy()
        indices=sorted(set([int(np.argmin(abs(t-a))) for a in np.arange(2,t[-1]+.01,2.5)]+[len(t)-1,int(np.argmax(wrap)),int(np.argmin(wrap[30:])+30)]))
        with mujoco.Renderer(sim.model,height=300,width=500) as renderer:
            for page in range(math.ceil(len(indices)/5)):
                ix=indices[page*5:(page+1)*5]
                fig,axs=plt.subplots(2,len(ix),figsize=(4*len(ix),5),squeeze=False,constrained_layout=True)
                for col,i in enumerate(ix):
                    for row in (0,1):
                        axs[row,col].imshow(frame(i,row==1));axs[row,col].axis('off')
                        shape_label=f'wrap={wrap[i]:.1f}' if observable[i] else 'wrap unobservable'
                        axs[row,col].set_title(f't={t[i]:.1f}s {shape_label} '+('end' if row else 'side'),fontsize=9)
                fig.savefig(path/f'{label}_filmstrip{page+1}.png',dpi=100);plt.close(fig)
            if video:
                import cv2
                writer=cv2.VideoWriter(str(path/f'{label}_full.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1000,300))
                if not writer.isOpened():raise RuntimeError('Video writer unavailable')
                try:
                    for i in range(len(t)):
                        a=np.concatenate([frame(i),frame(i,True)],axis=1)
                        a=cv2.cvtColor(a,cv2.COLOR_RGB2BGR)
                        shape_label=f'wrap={wrap[i]:.1f}deg' if observable[i] else 'wrap unobservable (sharp axis change)'
                        cv2.putText(a,f'{label} t={t[i]:.1f}s {shape_label}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,30,30),1)
                        writer.write(a)
                finally:writer.release()
    print('Reviewed rendering:',label,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--labels',nargs='+',required=True)
    a.add_argument('--path',type=Path,default=ROOT/'validation/reports/fixed_section');a.add_argument('--video',action='store_true')
    args=a.parse_args()
    for label in args.labels:render(label,args.path,args.video)
