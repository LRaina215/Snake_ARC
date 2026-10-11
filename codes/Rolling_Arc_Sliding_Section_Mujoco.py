"""Simulation viewer for measured-section compliance and load-based windows.

Run this file directly. Space: hold/resume phase; A/D: request adjacent
head/tail window; U: automatic selection on/off; R: reverse rolling phase.
Physics/compliance continue while the phase is held. No F keys or hardware IO.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import queue
import time
import numpy as np
import mujoco.viewer
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ROOT, backbone
from snake_control.sliding_section_simulation import SlidingSectionSimulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='elbow90_taper110_210')
    parser.add_argument('--start-x', type=float, default=-.25)
    parser.add_argument('--duration', type=float, default=150.)
    args = parser.parse_args()
    p = ArcParameters.load(ROOT/'configs/step1.json')
    sim = SlidingSectionSimulation(args.scene, p, args.start_x)
    commands = queue.SimpleQueue()
    direction = 1
    event_count = 0

    def key(code):
        if code in (32, 65, 68, 85, 82):
            commands.put(code)

    print('空格：保持/恢复相位；A/D：向头/尾交接；U：自动选择；R：反向。仅MuJoCo仿真。', flush=True)
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=key) as viewer:
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -23
        while viewer.is_running() and sim.data.time < args.duration:
            wall = time.monotonic()
            while not commands.empty():
                command = commands.get()
                if command == 32:
                    sim.paused = not sim.paused
                elif command == 82:
                    direction *= -1
                elif command == 85:
                    sim.controller.automatic = not sim.controller.automatic
                    print('自动选择=', sim.controller.automatic, flush=True)
                else:
                    print('交接请求=', sim.request_neighbour(-1 if command == 65 else 1), flush=True)
            for _ in range(20):
                phase = sim.phase
                sim.step(phase)
                if sim.data.time >= p.settle_s and not sim.paused and not sim.controller.phase_hold:
                    sim.phase += direction*p.omega_rad_s*sim.model.opt.timestep
            with viewer.lock():
                viewer.cam.lookat[:] = backbone(sim.model, sim.data).mean(axis=0)+[0, 0, -.1]
                for geom in sim.robot_geoms:
                    link = int(sim.model.geom_bodyid[geom])-sim.model.body('head').id
                    j = min(max(link-1, 0), p.joints-1)
                    w = sim.weights[j]
                    sim.model.geom_rgba[geom] = [.5-.4*w, .26+.22*w, .7+.2*w, 1]
            viewer.sync()
            while event_count < len(sim.controller.events):
                print(sim.controller.events[event_count], flush=True)
                event_count += 1
            time.sleep(max(0., .02-(time.monotonic()-wall)))


if __name__ == '__main__':
    main()
