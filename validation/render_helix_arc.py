"""Render saved physical states, enable masks and motor commands; never render target poses."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from snake_control.rolling_arc import ArcParameters
from snake_control.tee_simulation import TeeSimulation

STAGES=dict(exit_transition='尾端建立出管过渡',reverse_unwrap='逆向Shift，逐段解缠与失能',
    crown_align='选择顶部承载弧',flatten='导出头端，消除螺距形成半圆',arc_release='逐节释放已导出的端部',
    arc_end_reset='承载半圆保持，两端短时归零再失能',
    arc_final_end_reset='归中结束，最终两端归零再失能',
    arc_settle='开放半圆保持',auto_center='角反馈自动送链归中',center_settle='归中后保持',arc_roll='开放半圆ARC持续滚动')


def render(folder,speed=4.,fps=15):
    record=json.loads((folder/'locked_gait.json').read_text(encoding='utf-8'))
    summary,rows=record['summary'],record['samples'];status=json.loads((folder/'controller.json').read_text(encoding='utf-8'))
    trace=np.load(folder/'locked_gait_trace.npz');sim=TeeSimulation('straight',ArcParameters(**summary['parameters']))
    m,d=sim.model,sim.data;head=m.body('head').id;pipe=m.geom('pipe_main').id;floor=m.geom('floor').id
    m.geom_matid[floor]=-1;m.geom_rgba[floor]=[.92,.94,.96,1]
    pipe_pos=m.geom_pos[pipe].copy();pipe_size=m.geom_size[pipe].copy()
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    width,height=512,400
    path=folder/f'helix_arc_{speed:g}x.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width*2,height+240))
    if not writer.isOpened():raise RuntimeError('video writer unavailable')
    sample_times=np.asarray(trace['time'])
    indices=np.searchsorted(sample_times,np.arange(sample_times[0],sample_times[-1],speed/fps)).tolist()
    if indices[-1]!=len(rows)-1:indices.append(len(rows)-1)
    with mujoco.Renderer(m,height=height,width=width) as renderer:
        for i in indices:
            d.qpos[:]=trace['qpos'][i];mask=trace['masks'][i]
            m.geom_rgba[m.geom_bodyid>=head]=[.55,.26,.70,1]
            for j in np.flatnonzero(mask)+1:m.geom_rgba[m.geom_bodyid==m.jnt_bodyid[m.joint(f'joint{j}').id]]=[.05,.50,.78,1]
            mujoco.mj_kinematics(m,d);center=d.xpos[head:].mean(0)
            images=[]
            for end in (False,True):
                m.geom_pos[pipe]=pipe_pos;m.geom_size[pipe]=pipe_size
                if end:m.geom_pos[pipe][0]=center[0];m.geom_size[pipe][1]=.24
                mujoco.mj_forward(m,d)
                cam=mujoco.MjvCamera();cam.lookat[:]=[center[0],0.,.90]
                cam.azimuth=0 if end else -60;cam.elevation=-3 if end else -22;cam.distance=1.65
                renderer.update_scene(d,camera=cam);images.append(renderer.render())
            image=Image.new('RGB',(2*width,height+240),'#f4f7fa');image.paste(Image.fromarray(np.hstack(images)),(0,65));draw=ImageDraw.Draw(image)
            r=rows[i];st=status[i];g=st['geometry']
            draw.text((12,5),f"18关节 | Helix→ARC→自动归中 | 周向起态 {summary['beta_deg']:g}° | t={r['time_s']:.1f}s",font=font,fill='#17354b')
            commands=[c for c in summary.get('commands',[]) if c['time_s']<=r['time_s']]
            recent=' → '.join(c['key'] for c in commands[-4:]) if commands else '自动仿真控制'
            draw.text((12,33),f"{STAGES.get(r['stage'],r['stage'])} | {speed:g}倍速 | 蓝=使能 紫=失能 | {recent}",font=small,fill='#385264')
            draw.text((12,75),'斜视',font=small,fill='#253b49');draw.text((width+12,75),'端视（沿主管轴）',font=small,fill='white')
            y=height+72
            diff=g.get('difference_m');label='尚未形成两侧悬垂参考' if diff is None else f'角反馈长度差 尾−头={diff*1000:+.1f} mm'
            draw.text((12,y),f"使能 {int(mask.sum())}/18 | {label} | 相位={st['phase']:.2f} rad",font=small,fill='#17354b')
            draw.text((12,y+25),f"解缠直尾参数 {st['straight_m']*1000:.0f} mm | 半圆目标长 {st['arc_length_m']*1000:.0f} mm | 跟踪误差 {r['tracking_max_deg']:.1f}°",font=small,fill='#385264')
            cell=(2*width-115)/18
            for label,yy in [('关节/使能',y+54),('目标角°',y+82),('实测角°',y+109)]:draw.text((7,yy),label,font=small,fill='#385264')
            for j,on in enumerate(mask):
                x=110+j*cell;color='#0089b2' if on else '#9552ad'
                draw.rectangle((x,y+52,x+cell-3,y+76),fill=color);draw.text((x+3,y+53),str(j+1),font=small,fill='white')
                draw.text((x+1,y+81),f'{np.degrees(trace["targets"][i,j]):.0f}' if on else '—',font=small,fill=color)
                draw.text((x+1,y+108),f'{np.degrees(d.qpos[sim.drive.qadr[j]]):.0f}',font=small,fill='#385264')
            writer.write(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR))
    writer.release();print(str(path),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folders',nargs='+',type=Path);p.add_argument('--speed',type=float,default=4.);a=p.parse_args()
    for folder in a.folders:render(folder,a.speed)
