"""Matched continuation with and without an upright branch during ARC→Helix."""
import argparse
import copy
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.arc_helix_transition import ArcHelixOptions, ArcHelixTransition
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation, ROOT, backbone


BRANCH_RADIUS_M = .030
BRANCH_HEIGHT_M = .60
START_X_M = -.50
PIPE_Z_M = 1.20
SETTLE_S = 3.
SPLIT_AFTER_SETTLE_S = 46.60
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


def make_sim(parameters, branch_x=None):
    model = None
    if branch_x is not None:
        spec = mujoco.MjSpec.from_file(str(ROOT/'mujoco_model/Snake_Robot_o20_Rolling_Arc_straight.xml'))
        # pipe_network maps local z to world x and local y to world z.
        spec.find_body('pipe_network').add_geom(
            name='pipe_branch', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            fromto=[0., 0., branch_x, 0., BRANCH_HEIGHT_M, branch_x],
            size=[BRANCH_RADIUS_M, 0., 0.], friction=[.5, .005, .002],
            condim=6, rgba=[.62, .66, .69, 1.], group=1)
        model = spec.compile()
    sim = ArcSimulation('straight', parameters, start_x=START_X_M, model=model)
    sim.model.geom('pipe_straight').friction[0] = .5
    if branch_x is not None:
        sim.pipe_geoms.remove(sim.model.geom('pipe_branch').id)
    return sim


def step(sim, core, tick, feedback, feedback_time):
    model, data, drive = sim.model, sim.data, sim.drive
    if tick % 200 == 0:
        feedback = np.degrees(data.qpos[drive.qadr])
        feedback_time = float(data.time)
    if tick % 20 == 0:
        core.advance(.02, data.time, feedback_time, feedback)
    drive.apply_targets(data, core.targets, core.enabled)
    mujoco.mj_step(model, data)
    return feedback, feedback_time


def contacts_on_branch(sim, branch, x):
    """Robot-side contact forces and their lever-arm moment about the main axis."""
    links, force, moment = set(), np.zeros(3), 0.
    if branch is not None:
        model, data = sim.model, sim.data
        for i, contact in enumerate(data.contact):
            if branch not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == branch else contact.geom1
            if other not in sim.robot_geoms:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, wrench)
            reaction = contact.frame.reshape(3,3).T@wrench[:3]
            if contact.geom2 == branch:
                reaction = -reaction
            links.add(int(model.geom_bodyid[other])-model.body('head').id)
            force += reaction
            moment += float(np.cross(contact.pos-[x,0.,PIPE_Z_M], reaction)[0])
    return sorted(links), force.tolist(), moment


def measure(sim, branch, branch_x, core, feedback, feedback_time):
    truth = sim.measure()
    points = backbone(sim.model, sim.data)
    yz = points[:,1:]-[0.,PIPE_Z_M]
    angles = np.unwrap(np.arctan2(yz[:,1],yz[:,0]))
    links, force, moment = contacts_on_branch(sim, branch, branch_x)
    end_x = points[:,0]
    clear = end_x > branch_x+BRANCH_RADIUS_M+.020
    prefix = np.flatnonzero(~(clear[:-1]&clear[1:]))
    return dict(time_s=float(sim.data.time)-SETTLE_S, stage=core.stage, shift_m=core.shift,
                accepted=core.accepted, head_x_m=float(end_x[0]), front_x_m=float(end_x[:5].mean()),
                min_x_m=float(end_x.min()), distal_height_m=float(points[-3:,2].mean()),
                cleared_prefix_links=int(prefix[0]) if len(prefix) else len(points)-1,
                turns=float(abs(angles[-1]-angles[0])/(2.*np.pi)),
                max_radius_m=float(np.linalg.norm(yz,axis=1).max()),
                main_links=truth['contact_links'], support_n=truth['normal_force_n'],
                ground_force_n=truth['ground_contact_force_n'],
                branch_links=links, branch_force_n=force, branch_moment_x_nm=moment,
                feedback_time_s=feedback_time-SETTLE_S, feedback_deg=feedback.tolist())


def continuation(sim, core, split, feedback, feedback_time, branch_x, has_branch, duration, folder):
    m, d = sim.model, sim.data
    branch = m.geom('pipe_branch').id if has_branch else None
    if contacts_on_branch(sim, branch, branch_x)[0]:
        raise ValueError('Branch overlaps the initial shared pose')
    trace, rows, finish = [], [], None
    for tick in range(split, split+round(duration/.001)):
        feedback, feedback_time = step(sim, core, tick, feedback, feedback_time)
        if tick % 50:
            continue
        row = measure(sim, branch, branch_x, core, feedback, feedback_time)
        rows.append(row)
        trace.append((d.qpos.copy(), d.qvel.copy(), d.actuator_force.copy(), core.targets, core.enabled))
        if core.stage == 'complete' and finish is None:
            finish = row['time_s']
        if row['ground_force_n']>1. or (finish is not None and row['time_s']-finish >= 2.):
            break
    qpos, qvel, torque, targets, masks = map(np.asarray, zip(*trace))
    held = [r for r in rows if finish is not None and r['time_s']>=finish]
    contact = [r for r in rows if r['branch_links']]
    front_x = np.array([r['front_x_m'] for r in rows])
    summary = dict(finish_s=finish, samples=len(rows),
                   sustained_helix=(len(held)>=40 and all(r['turns']>=1.6 and r['max_radius_m']<=.145
                       and any(j<=6 for j in r['main_links']) and any(j>=14 for j in r['main_links']) for r in held)),
                   highest_accepted=max(r['accepted'] for r in rows),
                   saturation_fraction=float(np.mean(np.any(abs(torque)>=sim.p.torque_limit_nm-1.e-6,axis=1))),
                   max_tracking_error_deg=float(np.degrees(np.max(abs(qpos[:,sim.drive.qadr]-targets)[masks]))),
                   backward_excursion_m=float(np.max(np.maximum.accumulate(front_x)-front_x)),
                   max_cleared_prefix_links=max(r['cleared_prefix_links'] for r in rows),
                   final_cleared_prefix_links=rows[-1]['cleared_prefix_links'],
                   ground_hit=any(r['ground_force_n']>1. for r in rows),
                   first_branch_contact=None if not contact else contact[0],
                   branch_impulse_ns=(.05*np.sum([r['branch_force_n'] for r in rows],axis=0)).tolist(),
                   branch_angular_impulse_x_nms=.05*sum(r['branch_moment_x_nm'] for r in rows))
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'result.json').write_text(json.dumps(dict(summary=summary,samples=rows),separators=(',',':')),encoding='utf8')
    np.savez_compressed(folder/'trace.npz',qpos=qpos,qvel=qvel,torques=torque,targets=targets,masks=masks)
    return summary


def prepare_split(params):
    base = make_sim(params)
    m, d = base.model, base.data
    if m.opt.timestep != .001:
        raise ValueError('This experiment uses 1 ms physics, 20 ms commands and 200 ms samples')
    seed = params.targets(0.)
    for _ in range(round(SETTLE_S/.001)):
        base.drive.apply_targets(d, seed, params.enabled)
        mujoco.mj_step(m,d)
    feedback = np.degrees(d.qpos[base.drive.qadr])
    feedback_time = float(d.time)
    core = ArcHelixTransition(np.full(params.joints+1,.0645), (params.first_joint,params.last_joint),
                              feedback, options=ArcHelixOptions(prepare_front=True))
    core.running = True
    split = round(SPLIT_AFTER_SETTLE_S/.001)
    for tick in range(split):
        feedback, feedback_time = step(base, core, tick, feedback, feedback_time)
        if tick % 50 == 0:
            base.measure()
    return base, core, split, feedback, feedback_time


def run(branch_x, duration, output, *, tee_only=False):
    if duration < .05:
        raise ValueError('Continuation must include at least one trace sample')
    params = ArcParameters()
    base, core, split, feedback, feedback_time = prepare_split(params)
    m, d = base.model, base.data
    checkpoint = np.empty(mujoco.mj_stateSize(m,STATE))
    mujoco.mj_getState(m,d,checkpoint,STATE)
    common = dict(branch_x_m=branch_x, branch_radius_m=BRANCH_RADIUS_M,
                  branch_height_m=BRANCH_HEIGHT_M, split_s=SPLIT_AFTER_SETTLE_S,
                  clear_plane_x_m=branch_x+BRANCH_RADIUS_M+.020,
                  split_stage=core.stage, split_enabled=core.enabled.tolist(), split_accepted=core.accepted)
    results = {}
    cases = [('tee',True)] if tee_only else [('straight',False),('tee',True)]
    for name, with_branch in cases:
        sim = make_sim(params,branch_x if with_branch else None)
        mujoco.mj_setState(sim.model,sim.data,checkpoint,STATE)
        mujoco.mj_forward(sim.model,sim.data)
        results[name] = continuation(sim, copy.deepcopy(core), split, feedback.copy(), feedback_time,
                                     branch_x, with_branch, duration, output/name)
    comparison = dict(common=common, **results)
    (output/'comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf8')
    np.savez_compressed(output/'checkpoint.npz', integration_state=checkpoint, feedback_deg=feedback,
                        feedback_time_s=feedback_time, mask=core.enabled, targets=core.targets)
    print(json.dumps(comparison,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch-x', type=float, default=-.370)
    parser.add_argument('--duration', type=float, default=115.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tee-only', action='store_true', help='Reuse the straight dynamics from the matched run')
    args = parser.parse_args()
    run(args.branch_x,args.duration,args.output,tee_only=args.tee_only)
