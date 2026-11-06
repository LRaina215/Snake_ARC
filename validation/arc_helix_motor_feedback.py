"""Validate three ARC starts on a straight pipe using motor samples only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.arc_helix_transition import ArcHelixTransition
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation, backbone


WINDOWS = {'balanced': (8, 13), 'right_short': (2, 7), 'left_short': (14, 19)}


def run(case, output):
    params = ArcParameters(first_joint=WINDOWS[case][0], last_joint=WINDOWS[case][1])
    sim = ArcSimulation('straight', params, start_x=-.5)
    m, d, drive = sim.model, sim.data, sim.drive
    pipe = m.geom('pipe_straight')
    pipe.friction[0] = .5
    step = m.opt.timestep
    seed = params.targets(0.)
    for _ in range(round(3./step)):
        drive.apply_targets(d, seed, params.enabled)
        mujoco.mj_step(m, d)

    prepared = backbone(m, d)
    pipe_center_yz = sim.route[0, 1:]
    z = prepared[:, 2]-pipe_center_yz[1]
    crossings = np.flatnonzero((z[:-1] >= 0.) != (z[1:] >= 0.))
    a, b = crossings+z[crossings]/(z[crossings]-z[crossings+1])
    feedback = np.degrees(d.qpos[drive.qadr])
    core = ArcHelixTransition(np.full(params.joints+1, .0645), WINDOWS[case], feedback)
    core.running = True
    rows, states, forces, targets, masks = [], [], [], [], []
    finish = None
    control_steps = round(.02/step)
    feedback_steps = round(.20/step)
    feedback_time = d.time
    for i in range(round(130./step)):
        if i % feedback_steps == 0:
            feedback = np.degrees(d.qpos[drive.qadr])
            feedback_time = d.time
        if i % control_steps == 0:
            core.advance(.02, d.time, feedback_time, feedback)
        drive.apply_targets(d, core.targets, core.enabled)
        mujoco.mj_step(m, d)
        if i % round(.05/step) != 0:
            continue
        truth = sim.measure()
        points = backbone(m, d)
        yz = points[:, 1:]-pipe_center_yz
        angles = np.unwrap(np.arctan2(yz[:, 1], yz[:, 0]))
        rows.append(dict(time_s=d.time, stage=core.stage, head_x_m=float(points[0,0]),
                         center_x_m=float(points[:,0].mean()), shift_m=getattr(core,'shift',0.),
                         turns=float(abs(angles[-1]-angles[0])/(2.*np.pi)),
                         max_radius_m=float(max(np.linalg.norm(yz,axis=1))),
                         main_links=truth['contact_links'], support_n=truth['normal_force_n'],
                         ground_force_n=truth['ground_contact_force_n'], feedback_time_s=feedback_time,
                         feedback_deg=feedback.tolist(), actual_max_deg=float(max(abs(np.degrees(d.qpos[drive.qadr]))))))
        states.append(d.qpos.copy())
        forces.append(d.actuator_force.copy())
        targets.append(core.targets.copy())
        masks.append(core.enabled.copy())
        if core.stage == 'complete' and finish is None:
            finish = d.time
        if truth['ground_contact_force_n'] > 1. or core.stage == 'blocked' or (finish and d.time-finish >= 2.):
            break
    held = [r for r in rows if finish is not None and r['time_s'] >= finish]
    passed = bool(len(held)>=40 and all(r['turns'] >= 1.6 and r['max_radius_m'] <= .145 and
        any(k <= 6 for k in r['main_links']) and any(k >= 14 for k in r['main_links']) for r in held)
        and not any(r['ground_force_n']>1. for r in rows))
    summary = dict(case=case, initial_window=WINDOWS[case], initial_head_links=float(a),
                   initial_tail_links=float(params.joints+1-b), initial_arc_deg=core.arc_turn_deg,
                   initial_end_y_m=prepared[[0,-1], 1].tolist(),
                   input='joint angles and feedback timestamp', finish_s=finish, passed=passed,
                   events=core.events, pipe_radius_m=float(pipe.size[0]), friction=float(pipe.friction[0]),
                   initial_settle_s=3., start_x_m=-.5, scene='straight',
                   control_period_s=.02, feedback_period_s=.20, trace_period_s=.05,
                   controller_options=asdict(core.options), arc_parameters=asdict(params))
    output.mkdir(parents=True, exist_ok=True)
    samples_json = ',\n'.join('  '+json.dumps(row, separators=(',', ':')) for row in rows)
    result_json = '{"summary": '+json.dumps(summary, indent=2)+',\n "samples": [\n'+samples_json+'\n ]}\n'
    (output/'result.json').write_text(result_json, encoding='utf-8')
    np.savez_compressed(output/'trace.npz',qpos=states,actuator_force=forces,targets=targets,masks=masks)
    print(json.dumps(summary))
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=WINDOWS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(0 if run(args.case,args.output) else 1)
