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
    acute=hyp['case']=='acute'
    straight=hyp['case']=='straight'
    recorded=next((path for path in (folder/'model.xml',folder.parent/'model.xml') if path.exists()),None) if acute or straight else None
    scene=hyp.get('scene','acute' if acute else 'straight' if hyp['case']=='pilot' else 'elbow')
    sim=VerticalScene(scene,recorded);m,d=sim.model,sim.data
    run=json.loads((folder/'run.json').read_text()) if (folder/'run.json').exists() else None
    rows=run['samples'] if run else None
    if acute and (folder/'review.json').exists():
        review=json.loads((folder/'review.json').read_text(encoding='utf-8'))
        run['summary']=review['summary'];hyp['exit_x_m']=review['summary']['exit_x_m']
        for row, evaluated in zip(rows,review['evaluation']):row.update(evaluated)
    statuses=json.loads((folder/'controller.json').read_text()) if (folder/'controller.json').exists() else None
    commands=json.loads((folder/'operator.json').read_text()) if (folder/'operator.json').exists() else []
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    width,height=600,430
    pipe_ids=sorted(sim.pipe_geoms);pipe_pos=m.geom_pos.copy();pipe_size=m.geom_size.copy();pipe_quat=m.geom_quat.copy();pipe_rgba=m.geom_rgba.copy()
    ids=np.linspace(0,len(trace['time'])-1,12).astype(int) if sheet else np.unique(np.r_[np.searchsorted(trace['time'],np.arange(trace['time'][0],trace['time'][-1],speed/15.)),len(trace['time'])-1])
    board=Image.new('RGB',(600,6*325),'white') if sheet else None
    path=folder/('inspection.jpg' if sheet else f'process_{speed:g}x.mp4')
    writer=None if sheet else cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),15,(1200,650))
    with mujoco.Renderer(m,height=height,width=width) as rr:
        for k,i in enumerate(ids):
            d.qpos[:]=trace['qpos'][i];mask=trace['masks'][i];target=np.degrees(trace['targets'][i])
            m.geom_rgba[m.geom_bodyid>=sim.head]=[.55,.26,.7,1.]
            for j in np.flatnonzero(mask)+1:m.geom_rgba[m.geom_bodyid==m.jnt_bodyid[m.joint(f'joint{j}').id]]=[.02,.55,.85,1]
            if (acute or straight) and mask[0]:m.geom_rgba[m.geom_bodyid==sim.head]=[.02,.55,.85,1]
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
                    if acute:
                        axis=np.array(rows[i]['fitted_normal']);tilt=float(np.clip(np.arctan2(axis[0],axis[2]),0.,np.pi/2))
                        if rows[i]['clear_supported']:
                            tilt=np.pi/2
                        corner=hyp['corner_z_m']
                        entry=hyp.get('entry_z_m',corner-hyp['exit_x_m'])
                        c=np.array([0.,0.,min(corner,center[2])]) if center[2]<entry else np.array([max(0.,center[0]),0.,corner])
                    elif straight or hyp['case']=='pilot':c=np.array([0.,0.,center[2]]);tilt=0.
                    else:
                        a=np.clip(np.arctan2(center[2]-1.8,.3-center[0]),0,np.pi/2)
                        candidates=np.array([[0.,0.,min(center[2],1.8)],
                            [.3*(1-np.cos(a)),0.,1.8+.3*np.sin(a)],[max(center[0],.3),0.,2.1]])
                        choice=int(np.argmin(np.linalg.norm(candidates-center,axis=1)));c=candidates[choice]
                        tilt=0. if choice==0 else np.pi/2 if choice==2 else a
                    normal=np.array([np.cos(tilt),0.,-np.sin(tilt)]);tangent=np.array([np.sin(tilt),0.,np.cos(tilt)])
                    m.geom_rgba[pipe_ids,3]=0.
                    radius=.055
                    if straight:
                        route=m.numeric('pipe_centerline').data.reshape(-1,4)
                        radius=float(np.interp(c[2],route[:,2],route[:,3]))
                    gid=m.geom('pipe_vertical').id
                    if scene=='round_to_square' and c[2]>=hyp['feature_exit_z_m']:
                        gid=m.geom('pipe_upper').id
                    m.geom_rgba[gid]=pipe_rgba[gid];m.geom_pos[gid]=c
                    if m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_BOX:
                        m.geom_size[gid,2]=.14
                    else:
                        m.geom_size[gid,:2]=[radius,.14]
                    q=Rotation.from_matrix(np.column_stack([normal,[0.,1.,0.],tangent])).as_quat();m.geom_quat[gid]=q[[3,0,1,2]]
                    cam.elevation=min(80.,90.-np.degrees(tilt))
                mujoco.mj_forward(m,d)
                rr.update_scene(d,camera=cam)
                if acute and view==0:
                    # Transparent plane at the actual transition-to-straight seam.
                    geom=rr.scene.geoms[rr.scene.ngeom]
                    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_BOX,np.array([.0015,.19,.24]),
                                        np.array([hyp['exit_x_m'],0.,hyp['corner_z_m']]),np.eye(3).ravel(),
                                        np.array([.10,.75,.9,.18],dtype=np.float32))
                    rr.scene.ngeom+=1
                images.append(Image.fromarray(rr.render()))
            stage='' if rows is None else rows[i]['stage']
            canvas=Image.new('RGB',(1200,650),'#f4f7fa');draw=ImageDraw.Draw(canvas)
            if straight:
                title={'variable':'竖直渐扩','square':'竖直方管外侧','round_to_square':'竖直圆管转方管'}.get(scene,'竖直等径')
                title+='｜2023 TRO Helix' if folder.name=='tro' else '｜整圈 ARC'
            else:
                title='急弯 A组·Helix' if acute and folder.name=='helix' else '急弯 B组·整圈ARC' if acute else '竖直Helix→水平ARC'
            draw.text((12,6),f'18关节｜{title}｜t={trace["time"][i]:.1f}s｜{stage}',font=font,fill='#17354b')
            recent=[c for c in commands if c['time_s']<=trace['time'][i]]
            cmd='自动上行/逐段退尾'
            if recent:
                cmd={'begin_bend':'模拟操作：进入弯头','confirm_outlet':'模拟操作：确认水平管接管',
                     'takeover_and_continuous':'仿真接管：连续执行'}.get(
                    recent[-1]['command'],'人工事件：水平已搭稳，接原逆向解缠/成弧/归中')
            phase='' if statuses is None else f'｜相位{statuses[i]["phase_rad"]:.2f}rad'
            if acute and statuses is not None and statuses[i].get('options',{}).get('section_shell_diameter_m') is not None:
                phase+=f'｜单截面交接{100*statuses[i].get("section_weight",0.):.0f}%'
            if acute and statuses is not None and statuses[i].get('options',{}).get('progressive_tail_release'):
                phase+=f'｜尾端已逐节释放{statuses[i]["released_tail_count"]}关节'
            if acute and statuses is not None and statuses[i].get('outlet_weight',0.)>0.:
                phase+=f'｜水平端部拼接{100*statuses[i]["outlet_weight"]:.0f}%'
            if straight and statuses is not None and statuses[i].get('section_feedback'):
                phase+=f'｜实测截面顺应｜承载弧长 {1000*statuses[i]["wrap_length_m"]:.0f} mm'
            draw.text((12,33),f'回放{speed:g}倍｜蓝=使能 紫=失能｜{cmd}{phase}',font=small,fill='#385264')
            for v,im in enumerate(images):canvas.paste(im,(v*600,60))
            draw.text((610,70),'沿局部管轴观察（管道仅作显示切片）',font=small,fill='white')
            text=f'使能 {mask.sum()}/18｜头端Z={pts[0,2]:.3f} m｜最下点Z={pts[:,2].min():.3f} m'
            if rows is not None:text+=f'｜管道支撑Fz={rows[i]["pipe_force_n"][2]:.1f} N（评价）'
            if acute:
                row=rows[i];clear_time=run['summary']['clear_time_s']
                state='已完整出弯' if row.get('exit_attached',row['clear_supported']) else '尚未完整出弯'
                last=1000*(row['last_envelope_x_m']-hyp['exit_x_m'])
                text=(f'使能 {mask.sum()}/18｜实际包角 {row["wrap_deg"]:.0f}°｜轴向跨度 {1000*row["span_m"]:.0f} mm'
                      f'｜尾侧离出口 {last:+.0f} mm｜{state}')
                if clear_time is None or row['time_s']<clear_time:text+='｜续行未开始计程'
                if row.get('physical_progress_m') is not None:text+=f'｜出弯净位移 {1000*row["physical_progress_m"]:+.0f} mm'
                shape=('整圈形态合格' if row['ring_shape'] else '整圈形态超限') if folder.name=='arc' else 'Helix 对照'
                draw.text((12,491),f'青色截面 = 弯头实体出口｜右图为局部观察｜{shape}',font=small,fill='#385264')
            if straight:
                row=rows[i]
                text+=f'｜包角 {row["wrap_deg"]:.0f}°｜轴向跨度 {1000*row["span_m"]:.0f} mm'
                exit_z=hyp.get('feature_exit_z_m')
                if exit_z is not None:
                    draw.text((12,491),f'最后一个外壳距变径段上缘 {1000*(row["last_envelope_z_m"]-exit_z):+.0f} mm',
                              font=small,fill='#385264')
            draw.text((12,511),text,font=small,fill='#17354b')
            draw.text((2,565),'目标',font=small,fill='#385264');draw.text((2,595),'实测',font=small,fill='#385264')
            for j,on in enumerate(mask):
                x=50+j*63;color='#0089b2' if on else '#9259b5';draw.rectangle((x,532,x+58,559),fill=color)
                draw.text((x+3,534),f'J{j+1}',font=small,fill='white');draw.text((x+3,565),f'{target[j]:.0f}' if on else '—',font=small,fill=color)
                draw.text((x+3,595),f'{np.degrees(d.qpos[sim.drive.qadr[j]]):.0f}',font=small,fill='#17354b')
            if sheet:
                crop=canvas.crop((0,0,600,650));crop.thumbnail((300,325));board.paste(crop,((k%2)*300,(k//2)*325))
            else:writer.write(cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR))
    if sheet:board.save(path)
    else:writer.release()
    print(path,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--sheet',action='store_true');p.add_argument('--speed',type=float,default=4.);a=p.parse_args();render(a.folder,a.sheet,a.speed)
