"""Step1 viewer/headless entry; shared fixed-window gait, no gait transitions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import time
import numpy as np
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=['straight', 'bend90', 'elbow90'], default='straight')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/step1.json')
    parser.add_argument('--duration', type=float, default=70.)
    parser.add_argument('--start-x', type=float, default=-.5)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not np.isfinite([args.duration, args.start_x]).all() or args.duration <= 0:
        parser.error('duration must be positive; start-x and duration must be finite')
    sim = ArcSimulation(args.scene, ArcParameters.load(args.config), args.start_x)
    samples = []
    paused = [False]
    def key_callback(key):
        if key == 32:
            paused[0] = not paused[0]
            print('Hold phase' if paused[0] else 'Resume rolling', flush=True)
    def run(viewer=None):
        next_sample = 0.
        start = time.monotonic()
        phase = sim.p.initial_phase_rad
        while sim.data.time < args.duration and (viewer is None or viewer.is_running()):
            previous_time = sim.data.time
            sim.step(phase)
            if sim.data.time >= next_sample:
                samples.append(sim.measure(phase)); next_sample += .05
                if viewer:
                    viewer.sync()
                    delay = sim.data.time - (time.monotonic() - start)
                    if delay > 0:
                        time.sleep(delay)
            if not np.isfinite(sim.data.qpos).all():
                raise RuntimeError('Nonfinite dynamic state')
            if not paused[0]:
                rolling_dt = max(0., sim.data.time - sim.p.settle_s) - max(0., previous_time - sim.p.settle_s)
                phase += sim.p.omega_rad_s * rolling_dt
    if args.headless:
        run()
    else:
        import mujoco.viewer
        print('Space: hold/resume rolling phase; close viewer to stop. Blue=enabled, purple=passive.')
        with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=key_callback) as viewer:
            viewer.cam.lookat[:] = [-.2, 0, 1.]
            viewer.cam.distance = 2.3
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            run(viewer)
    if not samples:
        print('Viewer closed before the first simulation step.')
        return
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(samples, indent=2), encoding='utf-8')
    print(json.dumps(dict(scene=args.scene, duration_s=samples[-1]['time_s'],
                         progress_delta_m=samples[-1]['progress_m']-samples[0]['progress_m'],
                         final_center_m=samples[-1]['center_m'],
                         passive_force_max_nm=max(s['passive_actuator_force_max_nm'] for s in samples),
                         final_contact_links=samples[-1]['contact_links']), ensure_ascii=False))


if __name__ == '__main__':
    main()
