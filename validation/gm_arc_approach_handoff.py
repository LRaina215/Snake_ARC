"""Start in ARC, roll to the tee, and stop at the gravity-gait entry."""
import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.arc_helix_transition import ArcHelixTransition
from snake_control.section_geometry import local_section
from snake_control.simulation import backbone
from validation.gm_arc_handoff_experiment import (
    adjust_arc, align_and_unwind, arc_entry_pose, entry_geometry, place_arc_for_adjustment,
    shift_arc_to_helix,
)
from validation.gm_locked_experiment import APPROACH_SETUP, right_u_ready
from validation.gm_replay_wrap_experiment import save
from validation.gm_shift_cycle_experiment import CycleTrial


START_PHASE = 2.09
ARC_SPEED = .60
PHASE_RAMP_S = .65


def approach_roll(trial, p):
    """One ARC cycle with a smooth start and stop at the short-head phase."""
    duration = math.tau/ARC_SPEED+PHASE_RAMP_S

    def ramp(t):
        u = np.clip(t/PHASE_RAMP_S,0.,1.)
        return ARC_SPEED*PHASE_RAMP_S*(u**3-.5*u**4)

    def rolling(t):
        if t < PHASE_RAMP_S:
            advance = ramp(t)
        elif t > duration-PHASE_RAMP_S:
            advance = math.tau-ramp(duration-t)
        else:
            advance = ARC_SPEED*(t-.5*PHASE_RAMP_S)
        trial.phase = START_PHASE+advance
        return p.targets(trial.phase)

    trial.phase_run(1,'arc_approach',duration,rolling,p.enabled)
    trial.phase = START_PHASE+math.tau
    trial.phase_run(1,'arc_approach_settle',.6,lambda _: p.targets(trial.phase),p.enabled)


def adaptive_short_head(trial, parameters):
    """Move the window one hinge at a time using measured material lengths."""
    p = parameters
    lengths = np.full(p.joints+1,trial.gait.l)
    observations = []
    transfer, _ = adjust_arc(trial,p,0,-1)
    for attempt in range(4):
        window = transfer.window
        measured = trial.data.qpos[trial.sim.drive.qadr]
        section = local_section(measured,lengths,*window)
        head,tail,_,_ = ArcHelixTransition.estimate_end_geometry(measured,lengths,section,window)
        observations.append(dict(time_s=float(trial.data.time),window=list(window),
                                 head_length_m=head,tail_length_m=tail))
        if head+trial.gait.l*.5 <= tail:
            return transfer, observations, True
        if attempt == 3 or window[0] <= 2 or trial.stopped:
            break
        p = replace(p,first_joint=window[0],last_joint=window[1])
        transfer, result = adjust_arc(trial,p,1,-1)
        if not result['completed']:
            break
    return transfer, observations, False


def run(output, start_x, branch_enabled, alignment_min_roll_s=.16):
    trial = CycleTrial(setup=APPROACH_SETUP)
    sim,m,d = trial.sim,trial.model,trial.data
    p = replace(sim.p,first_joint=7,last_joint=12,initial_phase_rad=START_PHASE)
    sim.p = sim.drive.p = p
    if not branch_enabled:
        # Retain exactly the same main pipe and dynamics; disable only branch contact.
        m.geom_contype[sim.branch_geom] = 0
        m.geom_conaffinity[sim.branch_geom] = 0
        m.geom_rgba[sim.branch_geom,3] = 0.
    trial.phase = START_PHASE
    place_arc_for_adjustment(trial,p,start_x)
    trial.phase_run(1,'settle',p.settle_s,lambda _: p.targets(START_PHASE),p.enabled)
    approach_roll(trial,p)
    points = backbone(m,d)
    arc_x = float(np.mean(points[p.first_joint:p.last_joint+1,0]))
    branch_x = float(d.geom_xpos[sim.branch_geom,0])
    pose = arc_entry_pose(trial,(p.first_joint,p.last_joint))
    approach_ready = .045 <= branch_x-arc_x <= .110 and 150. <= pose['sweep_deg'] <= 210.
    report = dict(branch_enabled=branch_enabled, branch_x_m=branch_x, start_x_m=start_x,
                  approached_arc_x_m=arc_x, approach_ready=approach_ready,
                  arc_approach_pose=pose, alignment_min_roll_s=alignment_min_roll_s,
                  feedback='joint angles for arm lengths; simulation world/contact for intake and handoff gates')
    if approach_ready and not trial.stopped:
        transfer, observations, ready = adaptive_short_head(trial,p)
        report.update(length_observations=observations, short_head_ready=ready,
                      arc_entry_pose=arc_entry_pose(trial,transfer.window))
        if ready and not trial.stopped:
            report['shift'] = shift_arc_to_helix(trial,transfer,p,0.)
            if report['shift']['completed'] and branch_enabled and not trial.stopped:
                keep = np.arange(p.joints)<8
                target, alignment = align_and_unwind(trial,keep,.09554165,.32201388,
                    {'formed':entry_geometry(trial,keep)}, head_alignment=(90.,105.),
                    min_release_moment=.63, alignment_min_roll_s=alignment_min_roll_s)
                reached = target is not None and right_u_ready(trial,keep) and not trial.stopped
                report.update(helix=alignment, gravity_entry_reached=reached,
                              entry=entry_geometry(trial,keep), retained_joint_ids=(np.flatnonzero(keep)+1).tolist())
                if reached:
                    output.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(output/'handoff_state.npz',qpos=d.qpos,qvel=d.qvel,
                                        targets=target, phase_rad=trial.phase, enabled_before_release=np.ones(p.joints,bool),
                                        enabled_after_release=keep)
    save(trial,output,'locked_gait',None,**report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start-x',type=float,default=.080)
    parser.add_argument('--straight-control',action='store_true')
    parser.add_argument('--alignment-min-roll',type=float,default=.16)
    args = parser.parse_args()
    run(args.output,args.start_x,not args.straight_control,args.alignment_min_roll)
