"""Render saved ACTUAL simulated states and metrics; replay never counts as a trial."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation, ROOT

LABELS = {'straight':'Straight pipe', 'bend90':'Smooth 90-degree bend', 'elbow90':'Continuous sharp 90-degree elbow'}


def load_case(path, name):
    trace = np.load(path/f'{name}_trace.npz')
    report = json.loads((path/f'{name}.json').read_text(encoding='utf-8'))
    fields = list(trace['metric_fields'])
    metrics = {field:trace['metrics'][:,i] for i,field in enumerate(fields)}
    return trace, report, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'validation/reports/step1')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    path = args.input
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.2), constrained_layout=True)
    for row, name in enumerate(LABELS):
        trace, report, metrics = load_case(path, name)
        sim = ArcSimulation(name, ArcParameters(**report['parameters']))
        sim.model.vis.global_.offwidth = 1000
        sim.model.vis.global_.offheight = 650
        # Presentation only: hide the checker texture; dynamics results are already stored.
        floor = sim.model.geom('floor').id
        sim.model.geom_matid[floor] = -1
        sim.model.geom_rgba[floor] = [.9,.92,.95,1]
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [-.2,.25,1.0] if name != 'straight' else [.0,0,1.0]
        cam.distance = 2.5
        cam.azimuth, cam.elevation = 135, -22
        times, progress = metrics['time_s'], metrics['progress_m']
        if name == 'straight':
            indices = [np.argmin(abs(times-t)) for t in (2,10,20,30)]
        else:
            feature = np.array(report['feature_span_m'])
            indices = [np.argmin(abs(times-2)), np.argmin(abs(progress-feature[0])),
                       np.argmin(abs(progress-feature.mean())), len(times)-1]
        with mujoco.Renderer(sim.model, height=650, width=1000) as renderer:
            def frame(index):
                sim.data.qpos[:] = trace['qpos'][index]
                sim.data.qvel[:] = trace['qvel'][index]
                mujoco.mj_forward(sim.model, sim.data)
                renderer.update_scene(sim.data, camera=cam)
                return renderer.render().copy()
            for column, index in enumerate(indices):
                rendered = frame(index)
                axes[row,column].imshow(rendered)
                axes[row,column].axis('off')
                axes[row,column].set_title(f't = {times[index]:.1f} s   |   travel = {progress[index]-progress[0]:.2f} m', fontsize=10)
                if column == 0:
                    axes[row,column].text(0,.98,LABELS[name],transform=axes[row,column].transAxes,
                                          color='white',va='top',fontsize=10)
                plt.imsave(path/f'{name}_stage{column+1}.png', rendered)
            if args.video:
                import cv2
                video_dir = ROOT/'validation/results/videos'
                video_dir.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(video_dir/f'{name}_4x.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (1000,650))
                if not writer.isOpened():
                    raise RuntimeError('Could not create video writer')
                try:
                    for index in range(0,len(times),2):
                        rendered = cv2.cvtColor(frame(index),cv2.COLOR_RGB2BGR)
                        cv2.putText(rendered,f'{LABELS[name]} | t={times[index]:.1f}s | 4x playback',
                                    (20,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                        writer.write(rendered)
                finally:
                    writer.release()
        trace.close()
        print('Rendered actual trajectory:',name,flush=True)
    fig.suptitle('Step1 actual MuJoCo dynamics | Blue: enabled arc | Purple: passive hanging joints', fontsize=14)
    fig.savefig(path/'actual_motion.png',dpi=150)
    plt.close(fig)
    fig,axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for name in ['straight','bend90','elbow90','hold','reverse']:
        trace,report,m = load_case(path,name)
        axes[0,0].plot(m['time_s'],m['progress_m']-m['progress_m'][0],label=name)
        if name in LABELS:
            parameters = report['parameters']
            centers = trace['backbone'][:,parameters['first_joint']:parameters['last_joint']+2].mean(axis=1)
            axes[0,1].plot(centers[:,0],centers[:,1],label=name)
            axes[1,0].plot(m['time_s'],m['normal_force_n'],label=name)
            axes[1,1].plot(m['time_s'],m['passive_actuator_force_max_nm'],label=name)
        trace.close()
    for ax in axes.flat:
        ax.grid(alpha=.25);ax.legend()
    axes[0,0].set(xlabel='Simulation time (s)',ylabel='Actual route travel (m)',title='Forward / hold / reverse controls')
    axes[0,1].set(xlabel='World x (m)',ylabel='World y (m)',title='Actual enabled-arc center trajectory')
    axes[0,1].axis('equal')
    axes[1,0].set(xlabel='Simulation time (s)',ylabel='Total pipe normal force (N)',title='Pipe support retained')
    axes[1,1].set(xlabel='Simulation time (s)',ylabel='Maximum passive actuator force (Nm)',title='Disabled actuator output = 0')
    fig.savefig(path/'metrics.png',dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
