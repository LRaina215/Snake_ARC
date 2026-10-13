from pathlib import Path
import sys,json,gzip,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from snake_control.simulation import ArcSimulation,backbone
from validation.half_circle_experiment import MeasuredParameters

def review(folder):
 s=json.loads((folder/'status.json').read_text());ev=json.loads((folder/'evaluation.json').read_text());t=np.array([e['time_s'] for e in ev])
 fig,axs=plt.subplots(4,1,figsize=(12,10),sharex=True,constrained_layout=True)
 for key in ('true_pipe_wrap_deg','coverage_deg','actual_turn_deg'):axs[0].plot(t,[e[key] for e in ev],label=key)
 for value in (150,180,218):axs[0].axhline(value,ls='--',alpha=.4)
 axs[0].legend();axs[0].set_ylabel('deg')
 axs[1].plot(t,[e['radius_m']*1000 for e in ev],label='fitted BODY radius mm');axs[1].legend()
 axs[2].step(t,[e['window'][1]-e['window'][0]+1 for e in ev],label='active count');axs[2].legend()
 axs[3].plot(t,[e['support_up_n'] for e in ev],label='support up N');axs[3].plot(t,[e['body_mean_x_m']*100 for e in ev],label='mean x cm');axs[3].legend();axs[3].set_xlabel('physical time s')
 for ax in axs:ax.grid(alpha=.2)
 fig.savefig(folder/'timeline.png',dpi=120);plt.close(fig)
 rows=[]
 with gzip.open(folder/'actual_trace.jsonl.gz','rt',encoding='utf8') as f:
  for i,line in enumerate(f):
   if i%5==0 or i==len(ev)-1:rows.append(json.loads(line))
 p=MeasuredParameters(**s['parameters']);sim=ArcSimulation(s['scene'],p,s['start_x_m'])
 for geom in sim.pipe_geoms:sim.model.geom_rgba[geom,3]=.4
 cam=mujoco.MjvCamera();cam.distance=1.65
 selected=sorted(set([min(range(len(rows)),key=lambda i:abs(rows[i]['time_s']-tt)) for tt in np.arange(2,t[-1],5)]+[len(rows)-1,max(range(len(rows)),key=lambda i:rows[i]['evaluation']['true_pipe_wrap_deg'])]))
 images=[]
 with mujoco.Renderer(sim.model,height=300,width=500) as renderer:
  writer=cv2.VideoWriter(str(folder/'actual_full.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1000,300))
  if not writer.isOpened():raise RuntimeError('writer unavailable')
  for i,r in enumerate(rows):
   sim.data.qpos[:]=r['actual_qpos'];sim.data.qvel[:]=r['actual_qvel'];mujoco.mj_forward(sim.model,sim.data)
   for geom in sim.robot_geoms:
    link=int(sim.model.geom_bodyid[geom])-sim.model.body('head').id;j=min(max(link-1,0),19)
    sim.model.geom_rgba[geom]=[.12,.5,.9,1] if r['enabled'][j] else [.55,.3,.7,1]
   frames=[];points=backbone(sim.model,sim.data);a,b=r['evaluation']['window'];cam.lookat[:]=points[a:b+2].mean(axis=0)+[0,0,-.17]
   for az,el in ((75,-15),(0,0)):
    cam.azimuth=az;cam.elevation=el;renderer.update_scene(sim.data,camera=cam);frames.append(renderer.render().copy())
   combined=np.concatenate(frames,axis=1);label=f"t={r['time_s']:.1f}s J{a}-{b} wrap={r['evaluation']['true_pipe_wrap_deg']:.1f} deg"
   cv2.putText(combined,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
   writer.write(cv2.cvtColor(combined,cv2.COLOR_RGB2BGR))
   if i in selected:images.append(combined)
  writer.release()
 for page in range(math.ceil(len(images)/6)):
  fig,axs=plt.subplots(6,1,figsize=(12,21),constrained_layout=True)
  for ax,im in zip(axs,images[page*6:page*6+6]):ax.imshow(im)
  for ax in axs:ax.axis('off')
  fig.savefig(folder/f'filmstrip_{page+1}.png',dpi=90);plt.close(fig)
 print('reviewed',folder,flush=True)
if __name__=='__main__':
 for path in (ROOT/'validation/reports/half_circle_window').glob('*/status.json'):review(path.parent)
