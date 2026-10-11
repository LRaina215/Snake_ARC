"""Wide 3D views and worst states from saved sharp-taper traces; no dynamics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import gzip
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import mujoco
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation, ROOT


def render(label, path, times, worst=False):
    report = json.loads((path/f'{label}.json').read_text(encoding='utf-8'))
    if report['scene'] != 'elbow90_taper110_210':
        raise ValueError('Annotations and camera are for the 110→210 mm sharp elbow')
    with gzip.open(path/f'{label}_records.json.gz', 'rt', encoding='utf-8') as stream:
        rows = json.load(stream)
    sim = ArcSimulation(report['scene'], ArcParameters(**report['parameters']), report['start_x'])
    for geom in sim.pipe_geoms:
        sim.model.geom_rgba[geom, 3] = .6
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.45, 135, -30
    camera.lookat[:] = [-.25, .3, 1.05]
    with np.load(path/f'{label}_trace.npz') as trace:
        t = trace['time_s']
        if worst:
            valid = [i for i, row in enumerate(rows) if t[i] >= 3 and row['shape']['wrap_valid']]
            physical = [i for i in range(len(rows)) if t[i] >= 3]
            indices = [min(valid, key=lambda i: rows[i]['shape']['wrap_deg']),
                       max(valid, key=lambda i: rows[i]['shape']['wrap_deg']),
                       min(physical, key=lambda i: rows[i]['body_geometry']['turn_deg']),
                       max(physical, key=lambda i: rows[i]['body_geometry']['turn_deg'])]
            captions = ['可观测管轴包绕最小', '可观测管轴包绕最大', '身体截面转角最小', '身体截面转角最大']
        else:
            indices = [int(np.argmin(abs(t-x))) for x in times]
            captions = ['入弯与交接前', '跨越拐角与渐扩区域', '进入210 mm出口管', '全身离开急弯变径段']
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
        with mujoco.Renderer(sim.model, height=600, width=800) as renderer:
            for ax, i, caption in zip(axes.flat, indices, captions):
                sim.data.qpos[:] = trace['qpos'][i]
                sim.data.qvel[:] = trace['qvel'][i]
                mujoco.mj_forward(sim.model, sim.data)
                for geom in sim.robot_geoms:
                    link = int(sim.model.geom_bodyid[geom])-sim.model.body('head').id
                    weight = trace['weights'][i, min(max(link-1, 0), 19)]
                    sim.model.geom_rgba[geom] = [.5-.4*weight, .26+.22*weight, .7+.2*weight, 1]
                renderer.update_scene(sim.data, camera=camera)
                ax.imshow(renderer.render().copy())
                ax.axis('off')
                row = rows[i]
                wrap = f"{row['shape']['wrap_deg']:.1f}°" if row['shape']['wrap_valid'] else '不可观测'
                ax.set_title(f"{caption}  t={t[i]:.1f} s  J{row['window'][0]}–{row['window'][1]}\n"
                             f"身体转角{row['body_geometry']['turn_deg']:.1f}°；管轴包绕{wrap}",
                             fontproperties=font, fontsize=11)
        fig.suptitle('110→210 mm连续90°急弯变径：真实动力学保存态\n'
                     f'{label}；拐角前后各150 mm渐扩；蓝色使能、紫色失能',
                     fontproperties=font, fontsize=14)
        filename = path/f'{label}_{"worst" if worst else "overview"}.png'
        fig.savefig(filename, dpi=140)
        plt.close(fig)
    print(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='s4_sliding_cyclemean')
    parser.add_argument('--path', type=Path, default=ROOT/'validation/reports/sliding_section')
    parser.add_argument('--times', nargs=4, type=float, default=[5, 26, 30, 40])
    parser.add_argument('--worst', action='store_true')
    args = parser.parse_args()
    render(**vars(args))
