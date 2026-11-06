"""Run the supplied gravity-moment curve in MuJoCo without hardware I/O."""
import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.gravity_source_gait import SourceCurveGait, SourceShape
from snake_control.gravity_moment import smooth
from snake_control.simulation import backbone
from snake_control.tee_simulation import PIPE_HEIGHT
from validation.gm_experiment import BRANCH_R, BRANCH_X, LINK_R, angle, top_crossings
from validation.gm_locked_experiment import drop_metrics, recovery_metrics
from validation.gm_replay_wrap_experiment import save
from validation.gm_shift_cycle_experiment import CycleTrial
from validation.gm_wrap_first_experiment import pipe_contacts


def run(output, pitch, radius, align_deg, align_x, cycles, advance_rate,
        sequential_seat=False, roll_during_return=0., source_auto=False, prewrapped=False):
    setup = dict(scene='up60', joints=26, friction=.5, options=dict(r_h1=radius, r_h2=radius+.0028, pitch=pitch))
    trial = CycleTrial(setup=setup)
    g, sim, m, d = trial.gait, trial.sim, trial.model, trial.data
    source = SourceCurveGait(26, g.l, pitch)
    step = .4*g.l
    clearance = BRANCH_X+BRANCH_R+LINK_R
    shape = SourceShape(radius, 3.*g.l, source.lead(radius))
    if prewrapped:
        shape = SourceShape(radius, 0., 0., -(source.joints+1)*g.l*source.torsion(radius))
    q0 = source.targets(shape)
    helix_links = 26 if prewrapped else 21
    sim.place_shape(q0, helix_links, .2)
    beta = math.radians(align_deg-angle(backbone(m, d)[0]))
    sim.place_shape(q0, helix_links, .2, beta)
    top = top_crossings(backbone(m, d))[-1]
    sim.place_shape(q0, helix_links, .2+align_x-top, beta)
    q = q0.copy()
    records = []

    def phase_run(cycle, stage, duration, shape_at, *, mask=None, stiffness=None, stop=None, speed=math.inf):
        def targets(t):
            nonlocal q
            target = shape_at(t)
            trial.phase = target.phase
            desired = np.clip(source.targets(target), -math.radians(89.9), math.radians(89.9))
            delta = speed*m.opt.timestep
            q += np.clip(desired-q, -delta, delta)
            return q.copy()
        return trial.phase_run(cycle, stage, duration, targets, mask=mask, stiffness=stiffness, stop_when=stop)

    phase_run(1, 'settle', 2., lambda _: shape)
    for cycle in range(1, cycles+1):
        if trial.stopped:
            break
        state = dict(cycle=cycle)
        records.append(state)
        last_shape = shape
        approach_at = float(d.time)

        def approach(t):
            nonlocal shape
            phase_delta = .6*t if prewrapped else .2*int(t/.10)
            shape = replace(last_shape, phase=last_shape.phase+phase_delta)
            return shape

        def meets_branch(_):
            wait = 1. if prewrapped and cycle == 1 else .15
            return d.time-approach_at >= wait and bool(pipe_contacts(sim)[1])

        if not phase_run(cycle, 'approach', 9., approach, stop=meets_branch, speed=4.6):
            state['failure'] = 'branch not detected'
            break
        if not prewrapped:
            shape = replace(shape, phase=shape.phase-.1)
        shape_before_u = shape
        # In the source, changes in |theta| exchange arc and line lengths.
        # Apply the commanded increments with a finite actuator speed.
        theta_steps = list(np.arange(shape.bend, -3.16, -.2))
        u_shapes = []
        last_theta = shape.bend
        arc_straight = shape.straight
        for bend in theta_steps:
            if source_auto:
                arc_straight += .2*source.arc_radius if abs(last_theta) > abs(bend) else -.2*source.arc_radius
                if arc_straight < 0.:
                    arc_straight += step
                last_theta = bend
            else:
                arc_straight = max(0., shape_before_u.straight+
                                   (abs(shape_before_u.bend)-abs(bend))*source.arc_radius)
            phase = shape_before_u.phase
            if prewrapped:
                delta = arc_straight-shape_before_u.straight+source.arc_radius*(abs(bend)-abs(shape_before_u.bend))
                phase += delta*source.torsion(radius)
            u_shapes.append(replace(shape_before_u, straight=arc_straight, bend=float(bend), phase=phase))
        tick = .08 if source_auto else .18 if prewrapped else .15
        phase_run(cycle, 'unwind_arc', len(u_shapes)*tick,
                  lambda t: u_shapes[min(len(u_shapes)-1, int(t/tick))], speed=3.0)
        shape = u_shapes[-1]
        nominal_released = int(source.turn_length(radius+.0028)/g.l)+1
        keep = sim.p.joints-nominal_released
        mask = np.arange(sim.p.joints) < keep
        last_shape = shape
        max_straight = source.turn_length(shape.radius)-abs(shape.bend)*source.arc_radius+step
        pickup = None
        line_at = float(d.time)

        def unwind(t):
            nonlocal shape
            straight = min(max_straight, last_shape.straight+step*int(t/tick))
            shape = source.change_straight(last_shape, straight)
            return shape

        def right_u(_):
            nonlocal pickup
            p = backbone(m, d)
            main, branch, _ = pipe_contacts(sim)
            y, torque = trial.release_moment(mask)
            # The original release is operator timed. Confirm the same physical
            # posture from the live MuJoCo state and record the decision.
            if (d.time-line_at < .2 or p[-1, 1] >= -.12 or p[-1, 2] < PIPE_HEIGHT-.025 or y >= -.025 or
                    not any(link <= keep for link in main) or
                    not any(link > keep+2 and p[link, 2] > PIPE_HEIGHT for link in main)):
                return False
            pickup = dict(time_s=float(d.time), main_links=main, branch_links=branch, tail_com_y_m=y,
                          torque_about_pipe_nm=torque, keep=keep)
            return True

        line_s = max(.3, (max_straight-shape.straight)/step*tick)+(.08 if source_auto else .65)
        phase_run(cycle, 'unwind_line', line_s, unwind, stop=None if source_auto else right_u, speed=3.0)
        if trial.stopped:
            break
        if pickup is None and not source_auto:
            state['failure'] = 'right U not observed'
            state['end_tip_m'] = backbone(m, d)[-1].tolist()
            break
        if source_auto:
            before_open = shape

            def opening(t):
                return replace(before_open, radius=radius+.0004*min(7, 1+int(t/.08)))

            phase_run(cycle, 'loosen', .56, opening, speed=4.6)
            shape = opening(.56)
            # Flag_Space uses the decremented theta cursor in the original.
            phase_shape = replace(shape, bend=shape.bend-.2)
            shape = source.prepare_return(phase_shape, nominal_released)
            phase_run(cycle, 'source_active_straight', 1., lambda _: shape, speed=1.53)
            state['active_straight'] = True
        else:
            state['release'] = pickup
            start_drop = len(trial.samples)
            shape_before_drop = shape

            def drop(t):
                # Source opens by 0.4 mm seven times. Hold rolling phase fixed.
                return replace(shape_before_drop, radius=radius+.0004*min(7, int(t/.08)))

            def hangs(_):
                rows = trial.samples[max(start_drop, len(trial.samples)-3):]
                return (len(rows) == 3 and rows[-1]['time_s']-pickup['time_s'] >= 1. and
                        all(abs(r['released_tail_y_m']) <= .018 and
                            r['released_tail_com_m'][2] < PIPE_HEIGHT-.1 for r in rows))

            hung = phase_run(cycle, 'drop', 6., drop, mask=mask, stop=hangs, speed=3.0)
            state['drop'] = drop_metrics(trial.samples[start_drop:])
            if not hung or trial.stopped:
                state['failure'] = 'passive tail did not reach axis'
                break
            shape = source.prepare_return(drop(d.time-pickup['time_s']), nominal_released)
        if trial.stopped:
            break
        ready_shape = shape
        if not source_auto:
            q[~mask] = d.qpos[sim.drive.qadr][~mask]
        scale = np.ones_like(q) if source_auto else mask.astype(float)
        received = []
        if sequential_seat:
            # v2 writes the return shape from the junction toward the tip,
            # one servo per second. Simulate the speed and order of the writes.
            seated_goal = source.targets(ready_shape)
            seat_start = q.copy()
            first = max(0, keep-4)

            def seat(t):
                trial.phase = ready_shape.phase
                end = min(len(q), first+int(t)+1)
                desired = seat_start.copy()
                desired[first:end] = seated_goal[first:end]
                q[:] += np.clip(desired-q, -m.opt.timestep*.45, m.opt.timestep*.45)
                scale[:end] = 1.
                return q.copy()

            trial.phase_run(cycle, 'source_seat', len(q)-first, seat, stiffness=lambda: scale)
            if trial.stopped:
                break
        # With no seating stage, the curve boundary itself enables the next
        # joint. In either case the commanded curvature advances along links.
        return_shift = (int((nominal_released-2)/.4)*step if source_auto else (nominal_released-2)*g.l)
        wind_rate = step/.08 if source_auto else advance_rate
        return_duration = return_shift/wind_rate
        rewrap_at = float(d.time)

        def wind(t):
            nonlocal shape
            offset = min(return_shift, step*int(t/(step/wind_rate)))
            shape = source.change_straight(ready_shape, ready_shape.straight-offset)
            shape = replace(shape, phase=shape.phase+roll_during_return*offset/g.l)
            target = source.targets(shape)
            next_joint = keep+len(received)
            if not (sequential_seat or source_auto) and next_joint < len(mask) and abs(target[next_joint]) >= math.radians(2.):
                if not received or d.time-received[-1]['time_s'] >= .35:
                    received.append(dict(joint=next_joint+1, time_s=float(d.time)))
            if not (sequential_seat or source_auto):
                target_mask = np.arange(len(mask)) < keep+len(received)
                scale[:] += np.clip(target_mask.astype(float)-scale, -m.opt.timestep/.35, m.opt.timestep/.35)
            q[scale == 0.] = d.qpos[sim.drive.qadr][scale == 0.]
            return shape

        phase_run(cycle, 'source_rewrap', return_duration+step/wind_rate+(.0 if source_auto else .5), wind,
                  stiffness=lambda: scale, speed=4.6 if source_auto else math.radians(45.))
        state['rewrap'] = dict(start_s=rewrap_at, received=received, straight_end_m=shape.straight,
                               seated_sequentially=sequential_seat)
        if trial.stopped:
            break
        last_shape = shape
        phase_run(cycle, 'source_roll_open', 1.5,
                  lambda t: replace(last_shape, phase=last_shape.phase+.2*min(10, int(t/.15))),
                  stiffness=lambda: scale, speed=math.radians(60.))
        shape = replace(last_shape, radius=radius, phase=last_shape.phase+2.)
        phase_run(cycle, 'source_tighten', 1., lambda _: shape, stiffness=lambda: scale, speed=math.radians(45.))
        state['end_x_min_m'] = trial.samples[-1]['x_min_m']
        if all(row['x_min_m'] > clearance for row in trial.samples[-10:]):
            # Finish the last three straight links once the branch is behind.
            last_shape = shape
            def finish_tail(t):
                offset = last_shape.straight*smooth(t/5.)
                tail_shape = source.change_straight(last_shape, last_shape.straight-offset)
                desired = source.targets(tail_shape)
                active = np.flatnonzero(scale > 0.)
                next_joint = int(active[-1])+1
                if next_joint < len(mask) and abs(desired[next_joint]) >= math.radians(2.):
                    scale[next_joint] = 1.
                q[scale == 0.] = d.qpos[sim.drive.qadr][scale == 0.]
                return tail_shape

            phase_run(cycle, 'source_finish_tail', 5.2, finish_tail, stiffness=lambda: scale, speed=math.radians(35.))
            shape = source.change_straight(last_shape, 0.)
            last_shape = shape
            phase_run(cycle, 'recover_roll', 12., lambda t: replace(last_shape, phase=last_shape.phase+.6*t),
                      speed=math.radians(60.))
            break
    recovery = recovery_metrics([a for a in trial.samples if a['stage']=='recover_roll'])
    save(trial, output, 'locked_gait', None, source_version='supplied v6 auto' if source_auto else 'supplied v2 passive / v6 curve',
         setup=setup, align_deg=align_deg, align_x=align_x, advance_rate=advance_rate, cycles=records,
         sequential_seat=sequential_seat, roll_during_return_rad_per_link=roll_during_return, prewrapped=prewrapped,
         release_events=trial.release_events, recovery=recovery)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pitch', type=float, default=.260)
    parser.add_argument('--radius', type=float, default=.0956)
    parser.add_argument('--align-deg', type=float, default=130.)
    parser.add_argument('--align-x', type=float, default=.414)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--advance-rate', type=float, default=.045)
    parser.add_argument('--sequential-seat', action='store_true', help='Use the v2 sequential post-drop position writes')
    parser.add_argument('--roll-during-return', type=float, default=0.,
                        help='Additional rolling phase per link during shift; the source rolls afterwards')
    parser.add_argument('--source-auto', action='store_true',
                        help='Contrast the supplied v6 active-straightening sequence in simulation')
    parser.add_argument('--prewrapped', action='store_true',
                        help='Start from a continuous helix rather than the three-link straight tail')
    args = parser.parse_args()
    if args.pitch <= 0. or args.radius <= 0. or args.advance_rate <= 0. or args.cycles < 1:
        parser.error('pitch, radius, advance rate and cycles must be positive')
    if args.source_auto and (args.sequential_seat or args.roll_during_return or args.prewrapped):
        parser.error('--source-auto uses the original straightening and rolling sequence')
    run(args.output, args.pitch, args.radius, args.align_deg, args.align_x, args.cycles, args.advance_rate,
        args.sequential_seat, args.roll_during_return, args.source_auto, args.prewrapped)
