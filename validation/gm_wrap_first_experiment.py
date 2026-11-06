"""One MuJoCo checkpoint experiment: passive drop, then rewrap before rolling."""
from pathlib import Path
import argparse
import json
import math
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.gravity_moment import GravityMomentGait, GravityMomentOptions, smooth
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import backbone
from snake_control.tee_simulation import PIPE_HEIGHT, TeeSimulation
from validation.gm_experiment import top_crossings


def pipe_contacts(sim):
    """Measure the robot-side force for each pipe contact, in world coordinates."""
    model, data = sim.model, sim.data
    main = model.geom('pipe_main').id
    head = model.body('head').id
    main_links, branch_links = set(), set()
    branch_force = np.zeros(3)
    for i in range(data.ncon):
        contact = data.contact[i]
        a, b = int(contact.geom1), int(contact.geom2)
        pipe, robot = (b, a) if a in sim.robot_geoms else (a, b)
        if robot not in sim.robot_geoms or pipe not in (main, sim.branch_geom):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        if force[0] < .15:
            continue
        link = int(model.geom_bodyid[robot] - head)
        if pipe == main:
            main_links.add(link)
        else:
            branch_links.add(link)
            branch_force += (1 if robot == b else -1) * contact.frame.reshape(3, 3).T @ force[:3]
    return sorted(main_links), sorted(branch_links), branch_force


def run(output):
    joints = 26
    params = ArcParameters(joints=joints, first_joint=1, last_joint=joints)
    sim = TeeSimulation('up60', params, friction=.5)
    m, d = sim.model, sim.data
    gait = GravityMomentGait(joints, GravityMomentOptions(r_h1=.094, r_h2=.098))
    ls = gait.turn_length(.094) - math.pi * gait.o.r_c + .5 * gait.l
    u_pose, helix_length = gait.targets(0., .094, ls=ls, th1=math.pi)
    loosened, _ = gait.targets(0., .098, ls=ls, th1=math.pi)
    rewrapped, _ = gait.targets(0., .098)
    front = gait.helix_joints(helix_length)
    front_points = int(helix_length // gait.l)
    # Manually aligned starting checkpoint. No state writes after dynamics starts.
    sim.place_shape(u_pose, front_points, .42, math.radians(-135.))
    crossing = top_crossings(backbone(m, d)[:front_points+1])[-1]
    d.qpos[0] += .425 - crossing
    mujoco.mj_forward(m, d)

    times, states, speeds, targets, masks, forces, points, samples = ([] for _ in range(8))
    dt = m.opt.timestep
    next_sample = 0.
    release_s, rewrap_s = .8, 4.
    # Engage pairs from the already-supported helix toward the tip. Each new
    # pair starts at the measured angle. The remaining tail hangs passively.
    tail_ids = np.flatnonzero(~front)
    start_times = rewrap_s + (np.arange(tail_ids.size) // 2) * .85
    pickup_angles = np.zeros(joints)
    previous_mask = front.copy()
    tail_contact_since = None
    wrap_confirmed_s = None
    stage = 'checkpoint'
    while d.time < 20.:
        t = d.time
        if t < release_s:
            stage, q, mask = 'checkpoint', u_pose, np.ones(joints, bool)
        elif t < rewrap_s:
            stage, mask = 'drop', front
            u = smooth((t-release_s) / .7)
            q = u_pose + u * (loosened-u_pose)
        else:
            stage = 'rewrap'
            mask = front.copy()
            mask[tail_ids] = t >= start_times
            entering = mask & ~previous_mask
            pickup_angles[entering] = d.qpos[sim.drive.qadr[entering]]
            q = rewrapped.copy()
            for j, start in zip(tail_ids, start_times):
                u = smooth((t-start) / 2.0)
                q[j] = pickup_angles[j] + u * (rewrapped[j]-pickup_angles[j])
        previous_mask = mask.copy()
        sim.drive.apply_targets(d, q, mask)
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            raise RuntimeError('nonfinite simulation state')
        if d.time + 1e-9 < next_sample:
            continue
        next_sample += .05
        pts = backbone(m, d).copy()
        main_links, branch_links, branch_force = pipe_contacts(sim)
        angles = np.unwrap(np.arctan2(pts[:, 2]-PIPE_HEIGHT, pts[:, 1]))
        distal_contact = [j for j in main_links if j >= 21]
        tail_span = float(np.rad2deg(np.ptp(angles[-7:])))
        distal_r = np.linalg.norm(pts[-7:, 1:]-[0., PIPE_HEIGHT], axis=1)
        error = np.abs(np.rad2deg(q-d.qpos[sim.drive.qadr]))
        # This is an evaluator for the operator's visual check, not a claim
        # that these contact forces are available to the robot controller.
        on_pipe = len(distal_contact) >= 2 and tail_span >= 75. and max(distal_r) < .14
        if stage == 'rewrap' and on_pipe and mask.all() and max(error[~front]) < 15.:
            tail_contact_since = d.time if tail_contact_since is None else tail_contact_since
            if d.time-tail_contact_since >= .5 and wrap_confirmed_s is None:
                wrap_confirmed_s = float(d.time)
        else:
            tail_contact_since = None
        support = sim.measure(0.)
        sample = dict(time_s=float(d.time), stage=stage, head_x_m=float(pts[0, 0]),
                      x_min_m=float(pts[:, 0].min()), center_x_m=float(pts[:, 0].mean()),
                      front_angle_deg=float(np.rad2deg(angles[4])), tail_z_m=float(pts[-1, 2]),
                      tail_span_deg=tail_span, tail_max_radius_m=float(max(distal_r)),
                      tail_main_links=distal_contact, main_links=main_links, branch_links=branch_links,
                      branch_force_world_n=branch_force.tolist(), joint_error_max_deg=float(max(error[mask])),
                      tail_error_max_deg=float(max(error[~front])),
                      max_actual_joint_deg=float(max(abs(np.rad2deg(d.qpos[sim.drive.qadr])))),
                      saturated_joints=(np.flatnonzero(abs(d.actuator_force) >= params.torque_limit_nm-1e-6)+1).tolist(),
                      ground_force_n=support['ground_contact_force_n'], support_n=support['normal_force_n'])
        samples.append(sample)
        times.append(d.time)
        states.append(d.qpos.copy())
        speeds.append(d.qvel.copy())
        targets.append(q.copy())
        masks.append(mask.copy())
        forces.append(d.actuator_force.copy())
        points.append(pts)
        if sample['ground_force_n'] > 1. or (wrap_confirmed_s is not None and d.time-wrap_confirmed_s >= 1.):
            break

    output.mkdir(parents=True, exist_ok=True)
    summary = dict(scene='up60', joints=joints, roll_command_rad=0., friction=.5, ls_m=ls,
                   fixed_front_joints=np.flatnonzero(front).size, wrap_confirmed_s=wrap_confirmed_s,
                   ever_clear=any(s['x_min_m'] > .409 for s in samples),
                   end_clear=samples[-1]['x_min_m'] > .409, duration_s=float(d.time),
                   ground_force_max_n=max(s['ground_force_n'] for s in samples))
    (output/'wrap_first.json').write_text(json.dumps(dict(summary=summary, samples=samples), indent=1), encoding='utf-8')
    np.savez_compressed(output/'wrap_first_trace.npz', time=times, qpos=states, qvel=speeds,
                        targets=targets, masks=masks, actuator_force=forces, points=points)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
