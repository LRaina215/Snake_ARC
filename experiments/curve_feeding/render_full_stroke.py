"""Replay full-stroke ACTUAL states with commands and both length measures."""
import argparse
import gzip
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from snake_control.rolling_arc import ArcParameters
from snake_control.tee_simulation import TeeSimulation
from validation.render_tee_video import FONT, SMALL, W, H
from .full_stroke import DEFAULT, LABEL


def millimetres(value):
    return '--' if value is None or not np.isfinite(value) else f'{value*1000:.0f}'


def render(folder, output, speed=4.):
    data = json.loads((folder/(LABEL+'.json')).read_text(encoding='utf-8'))
    report, rows = data['summary'], data['samples']
    trace = np.load(folder/(LABEL+'_trace.npz'))
    with gzip.open(folder/(LABEL+'_control.jsonl.gz'), 'rt', encoding='utf-8') as stream:
        controls = [json.loads(line) for line in stream]
    control_times = np.array([r['time_s'] for r in controls])
    sim = TeeSimulation('straight', ArcParameters(**report['parameters']))
    m, d = sim.model, sim.data
    head = m.body('head').id
    main = m.geom('pipe_main').id
    floor = m.geom('floor').id
    m.geom_matid[floor] = -1
    m.geom_rgba[floor] = [.92, .94, .96, 1.]
    main_pos, main_size = m.geom_pos[main].copy(), m.geom_size[main].copy()
    fixed_x = np.median(trace['backbone_world'][:, :, 0])
    fps = 20
    # The 20ms control loop saves alternating 40/60ms intervals. Select by
    # simulation time; using their median silently changes replay speed.
    frame_times = np.arange(trace['time'][0], trace['time'][-1], speed/fps)
    indexes = np.minimum(np.searchsorted(trace['time'], frame_times), len(rows)-1).tolist()
    if indexes[-1] != len(rows)-1:
        indexes.append(len(rows)-1)
    output.parent.mkdir(parents=True, exist_ok=True)
    height = H+70+290
    z_min = float(trace['backbone_world'][:, :, 2].min())
    z_max = float(trace['backbone_world'][:, :, 2].max())
    look_z = .5*(z_min+z_max)
    frame_distance = max(1.4, 1.5*(z_max-z_min))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (2*W, height))
    if not writer.isOpened():
        raise RuntimeError('Video writer did not open')
    with mujoco.Renderer(m, height=H, width=W) as renderer:
        for index in indexes:
            row, time = rows[index], trace['time'][index]
            d.qpos[:] = trace['qpos'][index]
            mask, targets = trace['masks'][index], trace['targets'][index]
            m.geom_rgba[m.geom_bodyid >= head] = [.50, .26, .70, 1.]
            for j in np.flatnonzero(mask):
                body = m.jnt_bodyid[m.actuator_trnid[j, 0]]
                m.geom_rgba[m.geom_bodyid == body] = [.10, .48, .85, 1.]
            views = []
            for name, (azimuth, elevation, distance) in (
                    ('oblique', (-60, -25, frame_distance+.1)), ('end', (0, -5, frame_distance))):
                m.geom_pos[main], m.geom_size[main] = main_pos, main_size
                if name == 'end':
                    m.geom_pos[main][0] = fixed_x
                    m.geom_size[main][1] = .25
                mujoco.mj_forward(m, d)
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [fixed_x, 0, look_z]
                camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
                renderer.update_scene(d, camera=camera)
                views.append(renderer.render())
            canvas = Image.new('RGB', (2*W, height), 'white')
            for k in range(2):
                canvas.paste(Image.fromarray(views[k]), (k*W, 70))
            draw = ImageDraw.Draw(canvas)
            stage = row['stage'] if not report['failure'] or index < len(rows)-1 else '未完成：'+report['failure']
            draw.text((12, 5), f'18关节 | 曲线积分：中间→左极限→右极限→中间    t={time:.1f}s    {stage}',
                      font=FONT, fill=(23, 53, 74))
            draw.text((12, 38), f'实际动力学连续记录，{speed:g}倍速 | 仅J/H，无强制L、无复位 | 蓝=使能 紫=失能 | 使能{mask.sum()}关节',
                      font=SMALL, fill=(62, 86, 102))
            draw.text((12, 76), '斜视', font=SMALL, fill=(70, 70, 70))
            draw.text((W+12, 76), '尾→头端视：左=尾侧(+Y)，右=头侧(-Y)', font=SMALL, fill=(230, 230, 230))
            current = controls[max(0, int(np.searchsorted(control_times, time, side='right')-1))]
            f = current['feeding']
            observation = f.get('observation') or {}
            sent = [e for e in report['commands'] if e['time_s'] <= time][-8:]
            y = H+75
            draw.text((12, y), '按键记录（最近8次，仿真时间）：', font=SMALL, fill=(23, 53, 74))
            for line in range(2):
                text = '   |   '.join(f'{e["time_s"]:.2f}s {e["key"]}' for e in sent[line*4:(line+1)*4])
                draw.text((12, y+26+line*24), text, font=SMALL, fill=(23, 53, 74))
            bounds = f['profile_position_bounds_joints']
            draw.text((12, y+78), f'曲线分配 q={f["position_joints"]:.3f} / [{bounds[0]:g}, {bounds[1]:g}] '
                f'（中间 q={report["centre_position_joints"]:.3f}） | {f["stage"]} | '
                f'{"保持" if current["paused"] else "运行"} | 控制仅用角度/时间', font=SMALL, fill=(23, 53, 74))
            draw.text((12, y+105), '角度重建悬垂长：左 '+millimetres(observation.get('tail_hanging_m'))+
                ' mm，右 '+millimetres(observation.get('head_hanging_m'))+
                ' mm  |  仿真评价真值：左 '+millimetres(row['actual_tail_hanging_m'])+
                ' mm，右 '+millimetres(row['actual_head_hanging_m'])+' mm',
                font=SMALL, fill=(23, 53, 74))
            draw.text((12, y+132), '真值只作评价，不送入控制器；边界为现有控制器可分配范围。', font=SMALL, fill=(80, 80, 80))
            measured = np.degrees(d.qpos[sim.drive.qadr])
            cell = (2*W-165)/18
            for j in range(18):
                x = 160+j*cell
                color = (20, 124, 179) if mask[j] else (144, 79, 172)
                draw.rounded_rectangle((x, y+163, x+cell-4, y+189), radius=4, fill=color)
                draw.text((x+4, y+163), f'J{j+1}', fill='white', font=SMALL)
                draw.text((x+4, y+196), f'{np.degrees(targets[j]):.0f}' if mask[j] else '—', fill=color, font=SMALL)
                draw.text((x+4, y+225), f'{measured[j]:.0f}', fill=(62, 86, 102), font=SMALL)
            draw.text((12, y+163), '关节/使能', font=SMALL, fill=(62, 86, 102))
            draw.text((12, y+196), '已发目标(°)', font=SMALL, fill=(62, 86, 102))
            draw.text((12, y+225), '实际角度(°)', font=SMALL, fill=(62, 86, 102))
            writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
    writer.release()
    capture = cv2.VideoCapture(str(output))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    print(output, frames, 'frames', frames/fps, 'seconds', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=Path('artifacts/curve_feeding_full_stroke_20261108/full_stroke_4x.mp4'))
    parser.add_argument('--speed', type=float, default=4.)
    args = parser.parse_args()
    render(args.input, args.output, args.speed)
