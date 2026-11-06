"""Form the front coil from ARC, then use gravity-gait shift rolling for the tail."""
import argparse
import math
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.gravity_global_shift import ArcHelixShift
from snake_control.gravity_moment import smooth
from snake_control.simulation import backbone
from snake_control.tee_simulation import PIPE_HEIGHT
from validation.gm_arc_approach_handoff import START_PHASE, adaptive_short_head, approach_roll
from validation.gm_arc_handoff_experiment import place_arc_for_adjustment
from validation.gm_experiment import BRANCH_X, BRANCH_R, LINK_R
from validation.gm_locked_experiment import APPROACH_SETUP, progressive_rewrap, tail_engaged
from validation.gm_replay_wrap_experiment import save
from validation.gm_shift_cycle_experiment import CycleTrial
from validation.gm_wrap_first_experiment import pipe_contacts


def front_coil(trial, transfer, parameters, head_clearance, physical_front_seat):
    """Grow only the first helix, keeping the ARC support and rear torque off."""
    g,m,d,sim = trial.gait,trial.model,trial.data,trial.sim
    first,last = transfer.window
    curve = ArcHelixShift(g.gait,transfer.window,math.radians(parameters.amplitude_deg),
                          g.o.r_h1,g.o.pitch,g.o.handed)
    q = d.qpos[sim.drive.qadr].copy()
    reference = curve.targets(0.,0.,curve.arc_length,transfer.phase)[0]
    residual = np.where(transfer.enabled,q-reference,0.)
    stiffness = transfer.enabled.astype(float)
    j = np.arange(1,m.nu+1)
    distance = 0.
    roll = 0.
    next_update = 0.
    goal = q.copy()
    ready_at = None
    geometry = {}
    tail_bodies = slice(m.body('head').id+last+1,m.body('head').id+m.nu+1)

    def lead(t):
        nonlocal distance,roll,next_update,goal
        observed = d.qpos[sim.drive.qadr]
        if t>=next_update:
            pace = float(np.clip((math.radians(25.)-max(abs(q[stiffness>.75]-observed[stiffness>.75])))/
                                 math.radians(18.),.15,1.))
            distance = min(curve.head_length,distance+.035*pace*.02)
            roll += .23*smooth(t/1.5)*pace*.02
            goal,trial.phase = curve.targets(distance,0.,curve.arc_length,transfer.phase+roll)
            goal += (1.-smooth(t/2.))*residual
            np.clip(goal,-math.radians(88.),math.radians(88.),out=goal)
            next_update = t+.02
        coverage = np.where(j<=last,np.clip(distance/g.l+j-first+1.,0.,1.),0.)
        dt = m.opt.timestep
        stiffness[:] += np.clip(coverage-stiffness,-dt/.5,dt/.5)
        q[:] += np.clip(goal-q,-math.radians(23.)*dt,math.radians(23.)*dt)
        q[stiffness<=0.] = observed[stiffness<=0.]
        return q

    def seated(_):
        nonlocal ready_at, geometry
        points = backbone(m,d)
        yz = points[:last+1,1:]-[0.,PIPE_HEIGHT]
        phi = np.unwrap(np.arctan2(yz[:,1],yz[:,0]))
        turns = float(abs(phi[-1]-phi[0])/(2.*math.pi))
        main,branch,_ = pipe_contacts(sim)
        head_supported = (0 in main and np.all(stiffness[1:last]>=.80) if physical_front_seat
                          else np.all(stiffness[:last]>=.999))
        ready = (head_supported and turns>=.65 and
                 max(np.linalg.norm(yz,axis=1))<=.148 and any(k<=last for k in main)
                 and any(k<=last for k in branch) and points[0,0]>head_clearance)
        ready_at = (d.time if ready_at is None else ready_at) if ready else None
        geometry = dict(time_s=float(d.time), front_turns=turns, front_max_radius_m=float(max(np.linalg.norm(yz,axis=1))),
                        main_links=main,branch_links=branch,head_x_m=float(points[0,0]),
                        tail_com_m=np.average(d.xipos[tail_bodies],axis=0,weights=m.body_mass[tail_bodies]).tolist(),
                        mask=(stiffness>0.).tolist())
        return ready_at is not None and d.time-ready_at>=.20

    formed = trial.phase_run(1,'front_only_coil',24.,lead,stiffness=lambda:stiffness,stop_when=seated)
    return bool(formed and not trial.stopped),geometry,stiffness


def run(output,start_x,roll_rate,head_clearance,physical_front_seat):
    trial = CycleTrial(setup=APPROACH_SETUP)
    sim,d = trial.sim,trial.data
    p = replace(sim.p,first_joint=7,last_joint=12,initial_phase_rad=START_PHASE)
    sim.p = sim.drive.p = p
    trial.phase = START_PHASE
    place_arc_for_adjustment(trial,p,start_x)
    trial.phase_run(1,'settle',p.settle_s,lambda _: p.targets(START_PHASE),p.enabled)
    approach_roll(trial,p)
    if trial.stopped:
        save(trial,output,'locked_gait',None,start_x_m=start_x,front_ready=False,crossing=False)
        return
    transfer,lengths,ready = adaptive_short_head(trial,p)
    report = dict(start_x_m=start_x, length_observations=lengths, short_head_ready=ready,
                  roll_rate_rad_s=roll_rate,head_clearance_m=head_clearance,
                  physical_front_seat=physical_front_seat,front_ready=False,crossing=False)
    if ready and not trial.stopped:
        ready,geometry,stiffness = front_coil(trial,transfer,p,head_clearance,physical_front_seat)
        report.update(front_ready=ready,front_at_handoff=geometry)
        if ready:
            first_row = len(trial.samples)
            report['shift_rolling'] = progressive_rewrap(trial,transfer.window[1],roll_rate,
                duration=105.,fit_local=False,tighten_before_speed=True,roll_when_blocked=.055,
                initial_stiffness=stiffness)
            rows = trial.samples[first_row:]
            clearance = BRANCH_X+BRANCH_R+LINK_R
            report['crossing'] = (not trial.stopped and len(rows)>=40 and all(
                row['x_min_m']>clearance and not row['branch_links'] and tail_engaged(row) for row in rows[-40:]))
    save(trial,output,'locked_gait',None,**report)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start-x',type=float,default=.080)
    parser.add_argument('--roll-rate',type=float,default=.35)
    parser.add_argument('--head-clearance',type=float,default=BRANCH_X+BRANCH_R)
    parser.add_argument('--physical-front-seat',action='store_true',
                        help='Use physical head contact even while J1 is still coming on')
    args=parser.parse_args()
    run(args.output,args.start_x,args.roll_rate,args.head_clearance,args.physical_front_seat)
