"""Continue one body-curve shift from the live tee contact before tail intake ends."""
import argparse
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.gravity_global_shift import MeasuredGravityShift
from snake_control.gravity_moment import smooth
from snake_control.simulation import backbone
from validation.gm_experiment import BRANCH_X, BRANCH_R, LINK_R
from validation.gm_locked_experiment import APPROACH_SETUP, intake_connections, tail_engaged
from validation.gm_replay_wrap_experiment import save
from validation.gm_shift_cycle_experiment import CycleTrial


def phase_from_span(gait, radius, measured, selected):
    references = np.column_stack([gait.targets(a,radius)[0][selected] for a in (0.,math.pi/2)])
    coeff = np.linalg.lstsq(references,measured[selected],rcond=None)[0]
    return math.atan2(coeff[1],coeff[0])


def run(record, output, at, duration, roll_rate, fixed_curve, cruise_after_intake):
    trial = CycleTrial(setup=APPROACH_SETUP)
    m,d,drive,g = trial.model,trial.data,trial.sim.drive,trial.gait
    data = json.loads(record.read_text(encoding='utf8'))
    rows = data['samples']
    with np.load(record.with_name(record.stem+'_trace.npz')) as trace:
        i = int(np.argmin(abs(trace['time']-at)))
        initial = rows[i]
        if initial['stage'] != 'arc_curve_shift' or not initial['branch_links']:
            raise ValueError('Select the actual branch-contact phase before tail wrap completes')
        d.qpos[:] = trace['qpos'][i]
        d.qvel[:] = trace['qvel'][i]
        stiffness = trace['stiffness_scale'][i].copy()
        command = trace['targets'][i].copy()
    mujoco.mj_forward(m,d)
    shift_record = data['summary']['shift']
    initial_accepted = max((event['joint'] for event in shift_record['events']
                            if event['time_s'] <= initial['time_s']),
                           default=shift_record['initial_window'][1])
    accepted = initial_accepted
    if accepted >= m.nu:
        raise ValueError('The selected frame already finished intake')
    measured = d.qpos[drive.qadr].copy()
    radius = g.o.r_h1
    selected = np.flatnonzero(stiffness>.75)
    selected = selected[selected<initial_accepted]
    phase = phase_from_span(g,radius,measured,selected)
    curve = MeasuredGravityShift(g,radius,phase,measured,accepted,upstream_m=1.)
    initial_stiffness = stiffness.copy()
    first = int(np.flatnonzero(stiffness>0.)[0])
    offset = roll = 0.
    confirmed_at = cleared_at = None
    events = []
    last_control = -.02
    cruise = None
    clearance = BRANCH_X+BRANCH_R+LINK_R

    def advance(t):
        nonlocal accepted, confirmed_at, last_control, offset, roll, cruise
        if t-last_control < .02-1.e-9:
            return command
        dt = t-last_control
        last_control = t
        q = d.qpos[drive.qadr]
        if accepted < m.nu:
            joins = intake_connections(backbone(m,d),radius,g.o.pitch,winding=-g.o.handed)
            tracking = abs(command[accepted]-q[accepted]) < math.radians(18.)
            ready = joins[accepted-1] and joins[accepted] and tracking
            confirmed_at = (t if confirmed_at is None else confirmed_at) if ready else None
            if confirmed_at is not None and t-confirmed_at>=.24:
                accepted += 1
                confirmed_at = None
                events.append(dict(joint=accepted,time_s=t,shift_m=offset,
                                   head_x_m=float(backbone(m,d)[0,0])))
        if (cruise_after_intake and cruise is None and accepted==m.nu and len(trial.samples)>=8 and
                all(tail_engaged(row) for row in trial.samples[-8:])):
            # Once the hanging part has entered, a full uniform helix no
            # longer needs the fitted hanging-curve correction.
            helix_phase = phase_from_span(g,radius,q,np.arange(m.nu))
            cruise = dict(time_s=t, phase=helix_phase,
                          residual=command-g.targets(helix_phase,radius)[0])
        desired = np.zeros(m.nu)
        desired[:accepted] = 1.
        if accepted < m.nu:
            desired[accepted] = .50
        # The already-powered front stays on. Restore the two free head hinges in sequence.
        for j in range(first):
            desired[j] *= smooth((t-(first-1-j)*.65)/.85)
        stiffness[:] = np.maximum(initial_stiffness,stiffness+np.clip(desired-stiffness,-dt/.65,dt/.65))
        relevant = stiffness>.5
        error = np.max(abs(command[relevant]-q[relevant]))
        pace = float(np.clip((math.radians(26.)-error)/math.radians(19.),.12,1.))
        if cruise is None:
            limit = curve.shift_limit_m if accepted==m.nu else (accepted-initial_accepted+1.8)*g.l
            offset = min(limit,offset+.025*pace*dt)
            roll += roll_rate*smooth(t/1.25)*pace*dt
            target = curve.targets(offset,roll)
            if not fixed_curve:
                ideal = curve.helical_targets(offset,roll)
                target[:accepted] += smooth(offset/.45)*(ideal[:accepted]-target[:accepted])
            trial.phase = phase-curve.torsion*offset+roll
        else:
            age = t-cruise['time_s']
            ramp = min(age,2.)
            advance = g.o.roll_rate*(max(age-2.,0.)+ramp**2/4.)
            trial.phase = cruise['phase']+advance
            target = g.targets(trial.phase,radius)[0]+(1.-smooth(age/3.))*cruise['residual']
        np.clip(target,-math.radians(88.),math.radians(88.),out=target)
        command[:] += np.clip(target-command,-math.radians(19.)*dt,math.radians(19.)*dt)
        command[stiffness==0.] = q[stiffness==0.]
        return command

    def clear(_):
        nonlocal cleared_at
        if not trial.samples:
            return False
        sample = trial.samples[-1]
        ready = (accepted==m.nu and sample['x_min_m']>clearance and
                 not sample['branch_links'] and tail_engaged(sample))
        cleared_at = (d.time if cleared_at is None else cleared_at) if ready else None
        return cleared_at is not None and d.time-cleared_at>=2.

    completed = trial.phase_run(1,lambda: 'direct_shift' if cruise is None else 'direct_shift_cruise',
                                duration,advance,stiffness=lambda:stiffness,stop_when=clear)
    save(trial,output,'locked_gait',initial['time_s'],initial=initial,initial_accepted=initial_accepted,
         accepted=accepted,events=events,shift_m=offset,roll_rad=roll,roll_rate_rad_s=roll_rate,
         fixed_curve=fixed_curve,cruise_s=None if cruise is None else cruise['time_s'],
         clearance_m=clearance,direct_crossing=completed and not trial.stopped)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--at',type=float,default=22.7)
    parser.add_argument('--duration',type=float,default=80.)
    parser.add_argument('--roll-rate',type=float,default=.23)
    parser.add_argument('--fixed-curve',action='store_true')
    parser.add_argument('--cruise-after-intake',action='store_true')
    args=parser.parse_args()
    run(args.record,args.output,args.at,args.duration,args.roll_rate,args.fixed_curve,args.cruise_after_intake)
