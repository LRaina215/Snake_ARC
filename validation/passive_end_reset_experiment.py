"""Bent-end reset from the same initial states, including deliberate floor contact."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from snake_control.passive_end_reset import PassiveEndReset
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import backbone
from snake_control.tee_simulation import TeeSimulation, PIPE_HEIGHT
from validation.gm_arc_handoff_experiment import place_arc_for_adjustment
from validation.gm_locked_experiment import APPROACH_SETUP
from validation.gm_shift_cycle_experiment import CycleTrial
from validation.gm_wrap_first_experiment import pipe_contacts


def run(output, floor, reset):
    trial = CycleTrial(setup=dict(APPROACH_SETUP, scene='straight'))
    sim, m, d = trial.sim, trial.model, trial.data
    p = ArcParameters(joints=18, first_joint=8, last_joint=13)
    sim.p = sim.drive.p = p
    place_arc_for_adjustment(trial, p, .1)
    ref = backbone(m, d)[7:15].copy()
    d.qpos[sim.drive.qadr[[1, 16]]] = np.radians([25.,-45.])
    mujoco.mj_forward(m, d)
    actual = backbone(m, d)[7:15]
    u, _, vt = np.linalg.svd((actual-actual.mean(0)).T@(ref-ref.mean(0)))
    rotation = vt.T@np.diag([1,1,np.linalg.det(vt.T@u.T)])@u.T
    translation = ref.mean(0)-rotation@actual.mean(0)
    d.qpos[:3] = rotation@d.qpos[:3]+translation
    root = Rotation.from_quat(d.qpos[[4,5,6,3]]).as_matrix()
    d.qpos[3:7] = Rotation.from_matrix(rotation@root).as_quat()[[3,0,1,2]]
    mujoco.mj_forward(m,d)
    height = PIPE_HEIGHT
    if floor:
        floor_id=m.geom('floor').id
        distance=min(mujoco.mj_geomDistance(m,d,floor_id,int(g),3.,None) for g in sim.robot_geoms)
        shift = float(distance-.0005)
        m.body_pos[m.body('pipe_network').id,2] -= shift
        d.qpos[2] -= shift
        height -= shift
        mujoco.mj_forward(m,d)
    target, enabled = p.targets(0.), p.enabled
    controller = PassiveEndReset(np.degrees(d.qpos[sim.drive.qadr]), np.degrees(target), enabled) if reset else None
    initial = backbone(m,d).copy()
    rows, poses, targets, masks, forces = [],[],[],[],[]
    last_control = last_sample = -1.
    passive = ~p.enabled
    final_reset = None
    while d.time < 5.:
        if d.time-last_control >= .02-1e-8:
            if controller is not None:
                controller.advance(.02,np.degrees(d.qpos[sim.drive.qadr]))
                target, enabled = np.radians(controller.targets), controller.enabled
                if controller.done and final_reset is None:
                    final_reset = dict(time_s=float(d.time), **controller.status())
            last_control = float(d.time)
        sim.drive.apply_targets(d,target,enabled)
        mujoco.mj_step(m,d)
        if d.time-last_sample >= .02-1e-8:
            points = backbone(m,d); main,_,_=pipe_contacts(sim)
            ground=0.
            for ci,contact in enumerate(d.contact):
                if m.geom('floor').id in (contact.geom1,contact.geom2):
                    cf=np.zeros(6);mujoco.mj_contactForce(m,d,ci,cf)
                    ground+=abs(float(cf[0]))
            arc=points[7:15,1:]-[0,height]
            rows.append(dict(time_s=float(d.time), passive_max_deg=float(max(abs(np.degrees(d.qpos[sim.drive.qadr][passive])))),
                arc_center_z_m=float(np.mean(arc[:,1])), arc_radius_max_m=float(np.linalg.norm(arc,axis=1).max()),
                arc_sweep_deg=float(np.degrees(np.ptp(np.unwrap(np.arctan2(arc[:,1],arc[:,0]))))),
                arc_displacement_m=float(np.max(np.linalg.norm(points[7:15]-initial[7:15],axis=1))),
                contacts=main, active=int(enabled.sum()),
                ground_force_n=ground,
                passive_force_nm=float(np.max(abs(d.actuator_force[~enabled]),initial=0.))))
            poses.append(d.qpos.copy()); targets.append(target.copy()); masks.append(enabled.copy()); forces.append(d.actuator_force.copy())
            last_sample=float(d.time)
    report=dict(floor=floor, reset=reset, pipe_axis_height_m=height, result=final_reset,
        final_passive_max_deg=rows[-1]['passive_max_deg'],
        max_arc_displacement_m=max(r['arc_displacement_m'] for r in rows),
        min_arc_center_above_axis_m=min(r['arc_center_z_m'] for r in rows),
        min_arc_sweep_deg=min(r['arc_sweep_deg'] for r in rows), max_arc_sweep_deg=max(r['arc_sweep_deg'] for r in rows),
        no_contact_frames=sum(not r['contacts'] for r in rows),
        floor_contact_frames=sum(r['ground_force_n']>.1 for r in rows),
        passive_force_after_release_nm=max((r['passive_force_nm'] for r in rows if r['active']==6),default=0.))
    report['passed']=bool((not reset or final_reset['reason']=='zero_then_disabled') and
         report['max_arc_displacement_m']<.04 and report['min_arc_center_above_axis_m']>-.02 and
         report['no_contact_frames']==0 and report['max_arc_sweep_deg']<270. and report['passive_force_after_release_nm']<1e-10)
    label=('floor' if floor else 'hanging')+('_reset' if reset else '_passive')
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/(label+'_trace.npz'),time=[r['time_s'] for r in rows],qpos=poses,targets=targets,masks=masks,actuator_force=forces)
    (output/(label+'.json')).write_text(json.dumps(dict(summary=report,samples=rows),ensure_ascii=False,indent=1),encoding='utf-8')
    print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    for floor in (False,True):
        for reset in (False,True):run(args.output,floor,reset)


if __name__=='__main__':main()
