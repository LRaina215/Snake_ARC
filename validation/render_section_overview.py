"""Wide views of the saved 90-degree gradual-taper trial; no new dynamics."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import mujoco
import numpy as np

from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation, ROOT


def render(label, path, times):
    report = json.loads((path / f'{label}.json').read_text(encoding='utf-8'))
    if report['scene'] != 'taper_true90':
        raise ValueError('This camera and its annotations are for taper_true90 only')
    parameters = ArcParameters(**report['parameters'])
    sim = ArcSimulation(report['scene'], parameters, report['start_x'])
    # Render-only transparency helps expose the arc; original states/collisions
    # are retained. One fixed camera makes the bend and its progress visible.
    for geom in sim.pipe_geoms:
        sim.model.geom_rgba[geom, 3] = .6
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-.25, .3, 1.05]
    camera.distance = 2.45
    camera.azimuth = 135
    camera.elevation = -30
    with np.load(path / f'{label}_trace.npz') as saved:
        recorded_times = saved['metrics'][:, 0]
        figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
        captions = ['入弯前', '转弯与渐扩同时发生', '弯段后部', '全身出弯后']
        with mujoco.Renderer(sim.model, height=600, width=800) as renderer:
            for axis, requested, caption in zip(axes.flat, times, captions):
                index = int(np.argmin(abs(recorded_times - requested)))
                sim.data.qpos[:] = saved['qpos'][index]
                sim.data.qvel[:] = saved['qvel'][index]
                mujoco.mj_forward(sim.model, sim.data)
                renderer.update_scene(sim.data, camera=camera)
                axis.imshow(renderer.render().copy())
                axis.axis('off')
                axis.set_title(f'{caption}  t={recorded_times[index]:.1f} s', fontproperties=font, fontsize=13)
        figure.suptitle('已完成的90°柔弯内渐扩测试：入口直径113 mm → 出口163 mm\n'
                         '弯曲中心线半径300 mm；蓝色为局部使能圆弧，紫色为被动段；保存态回放',
                         fontproperties=font, fontsize=15)
        output = path / f'{label}_overview.png'
        figure.savefig(output, dpi=140)
        plt.close(figure)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='r5_taper90')
    parser.add_argument('--path', type=Path, default=ROOT / 'validation/reports/fixed_section')
    parser.add_argument('--times', nargs=4, type=float, default=[5, 16, 27, 42])
    args = parser.parse_args()
    render(args.label, args.path, args.times)
