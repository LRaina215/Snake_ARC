"""Test a brief downstream brace after the head rounds the tee branch."""
import argparse
import copy
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.active_arc_transfer import ease
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import backbone
from validation.arc_helix_tee_brace import STATE, make_sim, measure, prepare_split, step


BRANCH_X = -.320
FORK_S = 123.70
FRONT_JOINTS = 5
APPROACH_S = 3.
HOLD_S = 6.
RELEASE_S = 2.
DURATION_S = 50.


def fit_front_guide(sim, goal):
    """Fit the first five angles with the rest of the body fixed in scratch FK."""
    m, d = sim.model, sim.data
    scratch = mujoco.MjData(m)
    scratch.qpos[:] = d.qpos
    joints = sim.drive.qadr[:FRONT_JOINTS]
    anchor = m.body(f'link{FRONT_JOINTS}').id
    base = d.qpos[joints].copy()
    anchor_pos = d.xpos[anchor].copy()
    anchor_rotation = d.xmat[anchor].reshape(3,3).copy()

    def points(angles):
        scratch.qpos[joints] = angles
        mujoco.mj_kinematics(m, scratch)
        rotation = anchor_rotation@scratch.xmat[anchor].reshape(3,3).T
        return (backbone(m,scratch)-scratch.xpos[anchor])@rotation.T+anchor_pos

    nominal = points(base)

    def residual(angles):
        pose = points(angles)
        return np.r_[pose[:2].mean(axis=0)-goal, .027*(angles-base),
                     .25*(pose[2:FRONT_JOINTS]-nominal[2:FRONT_JOINTS]).ravel()]

    # J4 was saturated when asked to reach the branch by itself. Keep its
    # loaded bend and distribute the front correction to the other hinges.
    travel = np.radians([40.,40.,40.,4.,40.])
    low = np.maximum(-np.radians(88.),base-travel)
    high = np.minimum(np.radians(88.),base+travel)
    fit = least_squares(residual,np.clip(base,low,high),bounds=(low,high),max_nfev=55)
    fitted_head = points(fit.x)[:2].mean(axis=0)
    # Verify the proposed guide against both pipe surfaces in scratch only.
    rotation = anchor_rotation@scratch.xmat[anchor].reshape(3,3).T
    scratch.qpos[:3] = rotation@(d.qpos[:3]-scratch.xpos[anchor])+anchor_pos
    attitude = Rotation.from_matrix(rotation)*Rotation.from_quat(d.qpos[[4,5,6,3]])
    scratch.qpos[3:7] = attitude.as_quat()[[3,0,1,2]]
    mujoco.mj_kinematics(m,scratch)
    mujoco.mj_comPos(m,scratch)
    mujoco.mj_collision(m,scratch)
    main = m.geom('pipe_straight').id
    branch = m.geom('pipe_branch').id
    head = m.body('head').id
    overlap, head_normal = 0., None
    for contact in scratch.contact:
        pair = (contact.geom1,contact.geom2)
        if main in pair:
            body = m.geom_bodyid[pair[1] if pair[0]==main else pair[0]]
            if head <= body < anchor:
                overlap = max(overlap,-contact.dist)
        if branch in pair:
            other = pair[1] if pair[0]==branch else pair[0]
            if m.geom_bodyid[other] == head:
                head_normal = float(contact.frame[0] if pair[0]==branch else -contact.frame[0])
    diagnostics = dict(main_penetration_m=float(overlap), head_branch_normal_x=head_normal)
    return fit.x, fitted_head, diagnostics


def axial_by_link(sim, branch):
    """Robot-side axial component, separately for each link contacting the branch."""
    model, data = sim.model, sim.data
    head = model.body('head').id
    forces = {}
    for index, c in enumerate(data.contact):
        if branch not in (c.geom1,c.geom2):
            continue
        side = 1. if c.geom1 == branch else -1.
        robot = c.geom2 if c.geom1 == branch else c.geom1
        if robot not in sim.robot_geoms:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model,data,index,force)
        link = int(model.geom_bodyid[robot])-head
        forces[link] = forces.get(link,0.)+float(side*(c.frame.reshape(3,3).T@force[:3])[0])
    return forces


def trial(sim, core, checkpoint, fork, feedback, feedback_time, guide, output):
    m, d, drive = sim.model, sim.data, sim.drive
    mujoco.mj_setState(m,d,checkpoint,STATE)
    mujoco.mj_forward(m,d)
    branch = m.geom('pipe_branch').id
    rows, states, velocities, torques, targets, masks = [], [], [], [], [], []
    sent = core.targets.copy()
    finish = None
    start = float(d.time)
    for tick in range(fork,fork+round(DURATION_S/.001)):
        elapsed = float(d.time)-start
        if tick % 200 == 0:
            feedback = np.degrees(d.qpos[drive.qadr])
            feedback_time = float(d.time)
        if tick % 20 == 0:
            if elapsed >= APPROACH_S:
                core.advance(.02,d.time,feedback_time,feedback)
            goal = core.targets
            if guide is not None:
                weight = ease(elapsed/APPROACH_S)*(1.-ease((elapsed-APPROACH_S-HOLD_S)/RELEASE_S))
                goal[:FRONT_JOINTS] += weight*(guide-goal[:FRONT_JOINTS])
            sent += np.clip(goal-sent,-np.radians(.46),np.radians(.46))
        drive.apply_targets(d,sent,core.enabled)
        mujoco.mj_step(m,d)
        if tick % 50:
            continue
        row = measure(sim,branch,BRANCH_X,core,feedback,feedback_time)
        row.update(elapsed_s=float(d.time)-start, axial_by_link_n=axial_by_link(sim,branch))
        rows.append(row)
        states.append(d.qpos.copy())
        velocities.append(d.qvel.copy())
        torques.append(d.actuator_force.copy())
        targets.append(sent.copy())
        masks.append(core.enabled)
        if core.stage == 'complete' and finish is None:
            finish = row['elapsed_s']
        if row['ground_force_n']>1. or (finish is not None and row['elapsed_s']-finish>=2.):
            break
    actual, torque, mask = np.asarray(states), np.asarray(torques), np.asarray(masks)
    support = [r for r in rows if r['axial_by_link_n'].get(0,0.) > .5]
    lift = [r for r in rows if APPROACH_S <= r['elapsed_s'] <= APPROACH_S+HOLD_S]
    front_x = np.array([r['front_x_m'] for r in rows])
    held = [r for r in rows if finish is not None and r['elapsed_s']>=finish]
    summary = dict(finish_elapsed_s=finish, highest_accepted=max(r['accepted'] for r in rows),
                   sustained_helix=(len(held)>=40 and all(r['turns']>=1.6 and r['max_radius_m']<=.145
                       and any(j<=6 for j in r['main_links']) and any(j>=14 for j in r['main_links']) for r in held)),
                   ground_hit=any(r['ground_force_n']>1. for r in rows),
                   first_downstream_brace=None if not support else support[0]['elapsed_s'],
                   downstream_brace_samples=len(support),
                   peak_head_forward_force_n=max(r['axial_by_link_n'].get(0,0.) for r in rows),
                   lift_window_tail_height_change_m=None if not lift else lift[-1]['distal_height_m']-lift[0]['distal_height_m'],
                   lift_window_tail_height_max_m=max((r['distal_height_m'] for r in lift),default=None),
                   lift_window_accepted=max((r['accepted'] for r in lift),default=None),
                   backward_excursion_m=float(np.max(np.maximum.accumulate(front_x)-front_x)),
                   saturation_fraction=float(np.mean(np.any(abs(torque)>=sim.p.torque_limit_nm-1.e-6,axis=1))),
                   max_cleared_prefix_links=max(r['cleared_prefix_links'] for r in rows))
    output.mkdir(parents=True,exist_ok=True)
    (output/'result.json').write_text(json.dumps(dict(summary=summary,samples=rows),separators=(',',':')),encoding='utf8')
    np.savez_compressed(output/'trace.npz',qpos=actual,qvel=velocities,torques=torque,targets=targets,masks=mask)
    return summary


def run(output):
    params = ArcParameters()
    base,core,split,feedback,feedback_time = prepare_split(params)
    initial = np.empty(mujoco.mj_stateSize(base.model,STATE))
    mujoco.mj_getState(base.model,base.data,initial,STATE)
    sim = make_sim(params,BRANCH_X)
    m,d = sim.model,sim.data
    mujoco.mj_setState(m,d,initial,STATE)
    mujoco.mj_forward(m,d)
    fork = round(FORK_S/.001)
    for tick in range(split,fork):
        feedback,feedback_time = step(sim,core,tick,feedback,feedback_time)
        if tick % 50 == 0:
            sim.measure()
    checkpoint = np.empty_like(initial)
    mujoco.mj_getState(m,d,checkpoint,STATE)
    mujoco.mj_forward(m,d)
    branch = m.geom('pipe_branch').id
    shared = measure(sim,branch,BRANCH_X,core,feedback,feedback_time)
    if shared['cleared_prefix_links']<2 or core.accepted>=params.joints:
        raise ValueError('Require two leading links past the branch and an unfinished tail')
    guide,fit,diagnostics = fit_front_guide(sim,np.array([BRANCH_X+.052,.020,1.275]))
    common = dict(fork_s=FORK_S, branch_x_m=BRANCH_X, shared=shared,
                  approach_s=APPROACH_S, hold_s=HOLD_S, release_s=RELEASE_S,
                  guide_deg=np.degrees(guide).tolist(), fitted_head_m=fit.tolist(),
                  fitted_geometry=diagnostics)
    results = {}
    for name,target in [('nominal',None),('brace',guide)]:
        results[name] = trial(sim,copy.deepcopy(core),checkpoint,fork,feedback.copy(),feedback_time,target,output/name)
    report = dict(common=common,**results)
    (output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    np.savez_compressed(output/'checkpoint.npz',integration_state=checkpoint, feedback_deg=feedback,
                        feedback_time_s=feedback_time, targets=core.targets, mask=core.enabled)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
