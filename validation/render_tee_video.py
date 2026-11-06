"""Replay video of saved ACTUAL tee-scene states (oblique + end view); replay never counts as a trial."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import re
import json
import cv2
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from snake_control.rolling_arc import ArcParameters
from snake_control.tee_simulation import TeeSimulation

FONT = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
SMALL = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
STAGES = {
          'u': 'U形成', 'release': '释放（尾部失能、前部放松）', 'drop': '甩落', 'rewind': '缠回（带滚动）', 'hold': '保持', 'form': '形成螺旋', 'arc': 'ARC滚动', 'settle': '静置', 'roll': 'ARC滚动', 'form': '形成进入姿态', 'hold': '保持', 'loosen': '放松（r_h1→r_h2）',
          'curl': '卷尾（论文第4阶段）', 'observe': '观察', 'return': '返回ARC', 'resume': '恢复ARC滚动'}
GM_STAGES = {'settle': '静置', 'approach': '阶段1 螺旋滚动接近', 'arc': '阶段2 θ1 0→π（小U）', 'line': '阶段2 增长直线',
             'loosen': '阶段2 半径放大', 'drop': '阶段2 尾部失能下落', 'extend': '阶段3 尾部伸展', 'extend_hold': '阶段3 保持',
             'rewind': '阶段4 卷回螺旋（带滚动）', 'straighten': '回到阶段1'}
LOCKED_STAGES = {**GM_STAGES, 'drop': '尾部失能，自然摆至轴下',
                 'coupled_rewrap': '尾段沿曲线回卷，前部同时滚动', 'recover_roll': '恢复螺旋滚动',
                 'unwind_arc': '尾部再次解旋成小 U', 'unwind_line': '小 U 搭上主管右侧',
                 'progressive_rewrap': '前部滚动，尾段逐节接入',
                 'source_seat': '版2：按顺序恢复切线圆弧与直尾',
                 'source_active_straight': '版6：下发低速伸直目标',
                 'source_rewrap': '原曲线缩短直尾，补偿螺旋相位',
                 'source_roll_open': '保留短直尾并滚动',
                 'source_tighten': '收紧螺旋半径',
                 'source_finish_tail': '将余下直尾送入螺旋',
                 'arc_approach': 'ARC 滚动接近', 'arc_adjust': 'ARC 主动调节两侧长度',
                 'arc_capture_head': '保留 ARC 承载，头侧开始搭管',
                 'arc_form_helix': '前部形成承载螺旋', 'arc_finish_helix': '尾侧接入螺旋',
                 'arc_intake': '尾部沿主管逐节包绕',
                 'arc_curve_shift': '曲线移位，头侧螺旋，尾侧接入',
                 'arc_align_helix': '支管前校准螺旋方位', 'arc_form_right_u': '尝试形成释放小 U',
                 'direct_shift': '整身采样区间向前移，尾部逐节接入',
                 'direct_shift_cruise': '接入全螺旋后继续推进'}
W, H = 640, 480


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--speed', type=float, default=4.)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--fixed-camera', action='store_true', help='Keep the oblique view fixed along the pipe')
    args = parser.parse_args()
    data = json.loads((args.input / f'{args.label}.json').read_text(encoding='utf-8'))
    report, samples = data['summary'], data['samples']
    trace = np.load(args.input / f'{args.label}_trace.npz')
    if 'options' in report and 'r_h1' in report['options']:
        STAGES.update(GM_STAGES)
    if args.label == 'locked_gait':
        STAGES.update(LOCKED_STAGES)
    params = ArcParameters(**report['parameters']) if 'parameters' in report else ArcParameters(joints=report.get('joints', 18), first_joint=1, last_joint=report.get('joints', 18))
    sim = TeeSimulation(report.get('scene', 'straight'), params, report.get('start_x', 0.))
    model, d = sim.model, sim.data
    floor = model.geom('floor').id
    model.geom_matid[floor] = -1
    model.geom_rgba[floor] = [.92, .94, .96, 1]
    head = model.body('head').id
    main = model.geom('pipe_main').id
    full_pos, full_size = model.geom_pos[main].copy(), model.geom_size[main].copy()
    times = trace['time']
    fixed_x = float(np.median(trace['qpos'][:, 0]))
    shift_samples = [row for row in report.get('shift', {}).get('curve_samples', []) if 'window_shift_m' in row]
    shift_times = [row['time_s'] for row in shift_samples]
    step = max(1, int(round(args.speed / args.fps / np.median(np.diff(times)))))
    indices = list(range(0, len(times), step))
    if indices[-1] != len(times) - 1:
        indices.append(len(times) - 1)
    out = args.input / f'{args.label}_{args.speed:g}x.mp4'
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (2 * W, H + 70))
    if not writer.isOpened():
        raise RuntimeError('could not open video writer')
    with mujoco.Renderer(model, height=H, width=W) as renderer:
        for index in indices:
            d.qpos[:] = trace['qpos'][index]
            mask = trace['masks'][index] if 'masks' in trace.files else sim.p.enabled
            stiffness = trace['stiffness_scale'][index] if 'stiffness_scale' in trace.files else mask.astype(float)
            model.geom_rgba[model.geom_bodyid >= head] = [.50, .26, .70, 1]
            for j in np.flatnonzero(mask) + 1:
                color = [.10, .48, .85, 1] if stiffness[j-1] >= .99 else [.10, .72, .65, 1]
                model.geom_rgba[model.geom_bodyid == model.jnt_bodyid[model.joint(f'joint{j}').id]] = color
            mujoco.mj_forward(model, d)
            center = d.xpos[head:].mean(axis=0)
            views = []
            for name, (az, el, dist) in (('oblique', (-60, -25, 1.3)), ('end', (0, -5, 1.1))):
                model.geom_pos[main], model.geom_size[main] = full_pos, full_size
                if name == 'end':          # display only: keep the camera outside the main pipe
                    model.geom_pos[main][0] = center[0]
                    model.geom_size[main][1] = .25
                mujoco.mj_forward(model, d)
                cam = mujoco.MjvCamera()
                view_x = fixed_x if args.fixed_camera and name == 'oblique' else center[0]
                cam.lookat[:] = [view_x, 0, 1.05]
                cam.azimuth, cam.elevation, cam.distance = az, el, dist
                renderer.update_scene(d, camera=cam)
                views.append(renderer.render())
            sample = samples[min(index, len(samples) - 1)]
            raw = sample.get('stage', 'roll' if times[index] >= 2 else 'settle')
            key = raw.split('_', 1)[1] if re.match(r'c\d+_', raw) else raw
            stage = (raw.split('_')[0] + ' ' if re.match(r'c\d+_', raw) else '') + STAGES.get(key, key)
            if report.get('right_intake') and raw == 'arc_curve_shift':
                stage = '前部移位，右侧逐节上管'
            if args.label == 'locked_gait' and 'cycle' in sample:
                stage = f"第 {sample['cycle']} 轮  {stage}"
            fell = sample.get('ground_force_n', 0.) > 1.
            if fell:
                stage = '失去主管支撑，落地'
            canvas = Image.new('RGB', (2 * W, H + 70), 'white')
            canvas.paste(Image.fromarray(views[0]), (0, 70))
            canvas.paste(Image.fromarray(views[1]), (W, 70))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 6), f'{args.title}    t = {times[index]:5.1f} s    {stage}', fill=(23, 53, 74), font=FONT)
            contact = sample.get('branch_links') or []
            if raw in {'progressive_rewrap', 'arc_intake', 'arc_curve_shift'}:
                control = f'全刚度 {int(np.sum(stiffness >= .99))}  接收 {int(np.sum(mask & (stiffness < .99)))}'
                colors = '蓝=全刚度 绿=接收 紫=失能'
            else:
                control = f'使能 {int(np.sum(mask))} 关节'
                colors = '蓝=使能 紫=失能'
            draw.text((12, 40), f'全身平均位置 x = {center[0] * 1000:6.0f} mm（支管轴 x = 350 mm）    {control}    '
                                f'支管接触: {"连杆" + ",".join(map(str, contact)) if contact else "无"}    '
                                f'{args.speed:g}倍速回放（{colors}）', fill=(62, 86, 102), font=SMALL)
            draw.text((12, 76), '斜视', fill=(80, 80, 80), font=SMALL)
            if args.fixed_camera and 'head_x_m' in sample:
                draw.text((12, 100), f'头端 x = {sample["head_x_m"] * 1000:.0f} mm', fill=(80, 80, 80), font=SMALL)
            if raw in {'direct_shift', 'direct_shift_cruise'}:
                confirmed = max((event['joint'] for event in report['events'] if event['time_s'] <= times[index]),
                                default=report['initial_accepted'])
                draw.text((12, 124), f'已接入 J{confirmed}  |  全身最小 x = {sample["x_min_m"]*1000:.0f} mm',
                          fill=(80, 80, 80), font=SMALL)
            if shift_times and times[index] >= shift_times[0]:
                shifted = shift_samples[np.searchsorted(shift_times, times[index], side='right')-1]['window_shift_m']
                draw.text((12, 124), f'采样区间前移 {shifted * 1000:.0f} mm', fill=(80, 80, 80), font=SMALL)
            draw.text((W + 12, 76), '端视（沿管轴）', fill=(230, 230, 230), font=SMALL)
            frame = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
            for _ in range(args.fps if fell and index == indices[-1] else 1):
                writer.write(frame)
    writer.release()
    check = cv2.VideoCapture(str(out))
    frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()
    print(out, f'{frames} frames, {frames / args.fps:.1f} s')


if __name__ == '__main__':
    main()
