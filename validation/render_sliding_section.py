"""Replay every saved actual state; charts, chronological side/end sheets/video."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import gzip
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
from snake_control.simulation import ArcSimulation, ROOT
from snake_control.rolling_arc import ArcParameters


def render(label, path, video=False):
    report = json.loads((path/f'{label}.json').read_text(encoding='utf-8'))
    saved_records = path/f'{label}_records.json.gz'
    if saved_records.exists():
        with gzip.open(saved_records, 'rt', encoding='utf-8') as stream:
            rows = json.load(stream)
    else:
        rows = json.loads((path/f'{label}_records.json').read_text(encoding='utf-8'))
    with np.load(path/f'{label}_trace.npz') as trace:
        t = trace['time_s']
        turn = np.array([r['body_geometry']['turn_deg'] for r in rows])
        gap = np.array([r['body_geometry']['gap_fraction'] for r in rows])
        wrap = np.array([r['shape']['wrap_deg'] for r in rows])
        observable = np.array([r['shape']['wrap_valid'] for r in rows])
        fig, axs = plt.subplots(6, 1, figsize=(13, 14), constrained_layout=True)
        axs[0].plot(t, turn, label='actual body section turn')
        axs[0].plot(t, [r['sent_geometry']['turn_deg'] for r in rows], label='sent command turn')
        axs[0].plot(t, np.where(observable, wrap, np.nan), alpha=.6, label='pipe wrap (observable)')
        if not observable.all():
            axs[0].plot(t, np.where(observable, np.nan, wrap), ls=':', color='gray', label='pipe projection unobservable')
        axs[0].axhline(218, c='r', ls='--'); axs[0].axhline(150, c='r', ls='--')
        axs[0].set(ylabel='degrees', title=label+' | all saved physical states')
        axs[1].plot(t, gap, label='actual endpoint gap / active length')
        axs[1].axhline(.4, c='r', ls='--')
        axs[1].plot(t, np.array([r['body_geometry']['fit_rms_m'] for r in rows])*10, label='plane RMS (m) x10')
        axs[2].imshow(trace['weights'].T, origin='lower', aspect='auto', extent=[t[0], t[-1], .5, 20.5],
                      vmin=0, vmax=1, cmap='Blues')
        axs[2].set(ylabel='joint enable weights', yticks=np.arange(2, 21, 2))
        axs[3].imshow(trace['support_up_n'].T, origin='lower', aspect='auto', extent=[t[0], t[-1], .5, 20.5], cmap='Greens')
        axs[3].set(ylabel='observed upward load (N)', yticks=np.arange(2, 21, 2))
        axs[4].plot(t, [r['progress_m']-rows[0]['progress_m'] for r in rows], label='fixed whole-body-point progress (m)')
        axs[4].plot(t, np.sum(trace['support_up_n'], axis=1)/100, label='upward support /100 N')
        axs[4].plot(t, np.array([r['payload_n'] for r in rows])/10, label='external payload /10 N')
        axs[5].plot(t, np.max(abs(trace['actuator_force_nm']), axis=1), label='max active torque (Nm)')
        axs[5].plot(t, [r['local_pipe_radius_m']*10 for r in rows], label='pipe radius (evaluation only) x10 m')
        axs[5].set(xlabel='physical time (s)')
        for ax in axs:
            ax.grid(alpha=.2)
            handles, labels = ax.get_legend_handles_labels()
            if handles: ax.legend(fontsize=8, loc='upper right')
        fig.savefig(path/f'{label}_review.png', dpi=110); plt.close(fig)
        sim = ArcSimulation(report['scene'], ArcParameters(**report['parameters']), report['start_x'])
        for geom in sim.pipe_geoms:
            sim.model.geom_rgba[geom, 3] = .5
        camera = mujoco.MjvCamera()
        camera.distance = 1.8

        def frame(i, end=False):
            sim.data.qpos[:] = trace['qpos'][i]
            sim.data.qvel[:] = trace['qvel'][i]
            mujoco.mj_forward(sim.model, sim.data)
            for geom in sim.robot_geoms:
                link = int(sim.model.geom_bodyid[geom])-sim.model.body('head').id
                joint = min(max(link-1, 0), 19)
                weight = trace['weights'][i, joint]
                sim.model.geom_rgba[geom] = [ .5-.4*weight, .26+.22*weight, .7+.2*weight, 1]
            center = trace['backbone'][i].mean(axis=0)
            route = sim.route
            idx = np.argmin(np.linalg.norm((route[1:]+route[:-1])/2-center, axis=1))
            heading = np.degrees(np.arctan2(*(route[idx+1]-route[idx])[1::-1]))
            camera.lookat[:] = center+[0, 0, -.1]
            camera.azimuth = heading if end else 135
            camera.elevation = 0 if end else -23
            renderer.update_scene(sim.data, camera=camera)
            return renderer.render().copy()

        event_times = [e['time_s'] for e in report['events']]
        indices = sorted(set([int(np.argmin(abs(t-a))) for a in np.r_[np.arange(2, t[-1]+.001, 2), event_times]]+
                             [len(t)-1, int(np.argmax(turn)), int(np.argmin(gap)), int(np.argmin(turn[30:])+30)]))
        with mujoco.Renderer(sim.model, height=250, width=420) as renderer:
            for page in range(math.ceil(len(indices)/16)):
                selected = indices[page*16:(page+1)*16]
                fig, axs = plt.subplots(math.ceil(len(selected)/4), 4,
                                        figsize=(16, 3.7*math.ceil(len(selected)/4)), squeeze=False,
                                        constrained_layout=True)
                for ax in axs.flat: ax.axis('off')
                for ax, i in zip(axs.flat, selected):
                    ax.imshow(np.concatenate([frame(i), frame(i, True)], axis=0))
                    first, last = rows[i]['window']
                    status = f'pipe {wrap[i]:.0f}' if observable[i] else 'pipe unobservable'
                    ax.set_title(f"t={t[i]:.1f}s J{first}-{last} | body {turn[i]:.0f} | {status}", fontsize=8)
                fig.savefig(path/f'{label}_filmstrip{page+1}.png', dpi=90); plt.close(fig)
            if video:
                import cv2
                writer = cv2.VideoWriter(str(path/f'{label}_full.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (840, 250))
                if not writer.isOpened(): raise RuntimeError('Cannot create replay video')
                try:
                    for i in range(len(t)):
                        pixels = cv2.cvtColor(np.concatenate([frame(i), frame(i, True)], axis=1), cv2.COLOR_RGB2BGR)
                        cv2.putText(pixels, f"{label} t={t[i]:.1f}s J{rows[i]['window']} body={turn[i]:.1f} gap={gap[i]:.2f}",
                                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (25, 25, 25), 1)
                        writer.write(pixels)
                finally: writer.release()
    print('Full saved-state replay rendered:', label, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--path', type=Path, default=ROOT/'validation/reports/sliding_section')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    for label in args.labels:
        render(label, args.path, args.video)
