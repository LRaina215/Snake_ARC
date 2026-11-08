"""Replay complete actual states with commands/masks, or an inspection sheet."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import cv2,mujoco
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from snake_control.vertical_scene import VerticalScene
from snake_control.simulation import backbone


def render(folder,sheet=False,speed=4.):
    trace=np.load(folder/'trace.npz');hyp=json.loads((folder/'hypothesis.json').read_text())
    sim=VerticalScene('straight' if hyp['case']=='pilot' else 'elbow');m,d=sim.model,sim.data
    rows=json.loads((folder/'run.json').read_text())['samples'] if (folder/'run.json').exists() else None
    statuses=json.loads((folder/'controller.json').read_text()) if (folder/'controller.json').exists() else None
    commands=json.loads((folder/'operator.json').read_text()) if (folder/'operator.json').exists() else []
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    width,height=600,430
    pipe_ids=sorted(sim.pipe_geoms);pipe_pos=m.geom_pos.copy();pipe_size=m.geom_size.copy();pipe_quat=m.geom_quat.copy();pipe_rgba=m.geom_rgba.copy()
    ids=np.linspace(0,len(trace['time'])-1,12).astype(int) if sheet else np.searchsorted(trace['time'],np.arange(trace['time'][0],trace['time'][-1],speed/15.))
    board=Image.new('RGB',(1200,6*330),'white') if sheet else None
    path=folder/('inspection.jpg' if sheet else f'process_{speed:g}x.mp4')
    writer=None if sheet else cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),15,(1200,650))
    with mujoco.Renderer(m,height=height,width=width) as rr:
        for k,i in enumerate(ids):
            d.qpos[:]=trace['qpos'][i];mask=trace['masks'][i];target=np.degrees(trace['targets'][i])
            m.geom_rgba[m.geom_bodyid>=sim.head]=[.55,.26,.7,1.]
            for j in np.flatnonzero(mask)+1:m.geom_rgba[m.geom_bodyid==m.jnt_bodyid[m.joint(f'joint{j}').id]]=[.02,.55,.85,1]
            mujoco.mj_forward(m,d);pts=backbone(m,d);images=[]
            for view in range(2 if not sheet else 1):
                m.geom_pos[pipe_ids]=pipe_pos[pipe_ids];m.geom_size[pipe_ids]=pipe_size[pipe_ids]
                m.geom_quat[pipe_ids]=pipe_quat[pipe_ids];m.geom_rgba[pipe_ids]=pipe_rgba[pipe_ids]
                cam=mujoco.MjvCamera();cam.lookat[:]=pts.mean(0)
                if view==0:
                    cam.azimuth=-75;cam.elevation=-15;cam.distance=2.1
                else:
                    active=np.flatnonzero(mask);cam.lookat[:]=pts[active[0]+1:active[-1]+3].mean(0)
                    u=None if statuses is None else statuses[i].get('nominal_head_u_m')
                    tilt=0 if u is None else float(np.clip((u-.08)/.3,0,np.pi/2))
                    cam.azimuth=0;cam.elevation=min(80.,90.-np.degrees(tilt));cam.distance=1.25
                    # Replay-only local pipe slice prevents the long foreground
                    # cylinder hiding the robot in the axial view.
                    center=cam.lookat.copy()
                    if hyp['case']=='pilot':c=np.array([0.,0.,center[2]]);tilt=0.
                    else:
                        a=np.clip(np.arctan2(center[2]-1.8,.3-center[0]),0,np.pi/2)
                        candidates=np.array([[0.,0.,min(center[2],1.8)],
                            [.3*(1-np.cos(a)),0.,1.8+.3*np.sin(a)],[max(center[0],.3),0.,2.1]])
                        choice=int(np.argmin(np.linalg.norm(candidates-center,axis=1)));c=candidates[choice]
                        tilt=0. if choice==0 else np.pi/2 if choice==2 else a
                    normal=np.array([np.cos(tilt),0.,-np.sin(tilt)]);tangent=np.array([np.sin(tilt),0.,np.cos(tilt)])
                    m.geom_rgba[pipe_ids,3]=0.
                    gid=m.geom('pipe_vertical').id;m.geom_rgba[gid]=pipe_rgba[gid];m.geom_pos[gid]=c;m.geom_size[gid,:2]=[.055,.14]
                    q=Rotation.from_matrix(np.column_stack([normal,[0.,1.,0.],tangent])).as_quat();m.geom_quat[gid]=q[[3,0,1,2]]
                    cam.elevation=min(80.,90.-np.degrees(tilt))
                mujoco.mj_forward(m,d)
                rr.update_scene(d,camera=cam);images.append(Image.fromarray(rr.render()))
            stage='' if rows is None else rows[i]['stage']
            canvas=Image.new('RGB',(1200,650),'#f4f7fa');draw=ImageDraw.Draw(canvas)
            draw.text((12,6),f'18关节｜竖直Helix→水平ARC｜t={trace["time"][i]:.1f}s｜{stage}',font=font,fill='#17354b')
            recent=[c for c in commands if c['time_s']<=trace['time'][i]]
            cmd='自动上行/逐段退尾' if not recent else ('人工事件：进入弯头' if recent[-1]['command']=='begin_bend' else '人工事件：水平已搭稳，接原逆向解缠/成弧/归中')
            phase='' if statuses is None else f'｜相位{statuses[i]["phase_rad"]:.2f}rad'
            draw.text((12,33),f'回放{speed:g}倍｜蓝=使能 紫=失能｜{cmd}{phase}',font=small,fill='#385264')
            for v,im in enumerate(images):canvas.paste(im,(v*600,60))
            draw.text((610,70),'沿局部管轴观察（管道仅作显示切片）',font=small,fill='white')
            text=f'使能 {mask.sum()}/18｜头端Z={pts[0,2]:.3f} m｜最下点Z={pts[:,2].min():.3f} m'
            if rows is not None:text+=f'｜管道支撑Fz={rows[i]["pipe_force_n"][2]:.1f} N（评价）'
            draw.text((12,498),text,font=small,fill='#17354b')
            draw.text((2,565),'目标',font=small,fill='#385264');draw.text((2,595),'实测',font=small,fill='#385264')
            for j,on in enumerate(mask):
                x=50+j*63;color='#0089b2' if on else '#9259b5';draw.rectangle((x,532,x+58,559),fill=color)
                draw.text((x+3,534),f'J{j+1}',font=small,fill='white');draw.text((x+3,565),f'{target[j]:.0f}' if on else '—',font=small,fill=color)
                draw.text((x+3,595),f'{np.degrees(d.qpos[sim.drive.qadr[j]]):.0f}',font=small,fill='#17354b')
            if sheet:
                crop=canvas.crop((0,0,600,650));crop.thumbnail((600,330));board.paste(crop,((k%2)*600,(k//2)*330))
            else:writer.write(cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR))
    if sheet:board.save(path)
    else:writer.release()
    print(path,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--sheet',action='store_true');p.add_argument('--speed',type=float,default=4.);a=p.parse_args();render(a.folder,a.sheet,a.speed)
