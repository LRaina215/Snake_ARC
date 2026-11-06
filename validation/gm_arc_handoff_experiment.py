"""18-joint ARC approach and transition to the recorded gravity-moment gait."""
import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.active_arc_transfer import ActiveArcTransfer, TransferOptions
from snake_control.gravity_moment import smooth
from snake_control.simulation import backbone
from snake_control.tee_simulation import PIPE_HEIGHT, PIPE_RADIUS
from validation.gm_experiment import BRANCH_X, angle, top_crossings
from validation.gm_locked_experiment import (
    APPROACH_SETUP, attached_prefix, drop_and_rewrap, fit_intake_window,
    intake_connections, release_snapshot, rewrap, right_u_ready, tail_engaged,
)
from validation.gm_replay_wrap_experiment import save
from validation.gm_shift_cycle_experiment import CycleTrial
from validation.gm_wrap_first_experiment import pipe_contacts


def support_material(trial):
    """Material coordinates of the actual crossings of the pipe's midplane."""
    p = backbone(trial.model, trial.data)
    z = p[:, 2]-PIPE_HEIGHT
    indices = np.flatnonzero((z[:-1] >= 0.) != (z[1:] >= 0.))
    if len(indices) != 2:
        return None
    a, b = indices+z[indices]/(z[indices]-z[indices+1])
    return dict(head_links=float(a), tail_links=float(len(p)-1-b),
                arc_mid_joint=float(.5*(a+b)))


def adjust_arc(trial, parameters, steps, direction):
    """Reuse the overlapping arc handoff, ending before its demonstration return."""
    d, sim = trial.data, trial.sim
    options = TransferOptions(first_joint=parameters.first_joint, last_joint=parameters.last_joint,
                              hold_phase_rad=trial.phase, desired_direction=direction)
    transfer = ActiveArcTransfer(np.full(sim.p.joints+1, trial.gait.l), 'transfer', options)
    transfer.phase = trial.phase
    transfer.targets = parameters.targets(trial.phase)
    before = support_material(trial)
    next_update = float(d.time)

    def command(_):
        nonlocal next_update
        if d.time >= next_update-1e-9:
            t = float(d.time)
            transfer.advance(np.degrees(d.qpos[sim.drive.qadr]), t, t)
            next_update = t+.02
        return transfer.targets

    completed = 0
    for _ in range(steps):
        if trial.stopped:
            break
        transfer.begin_transfer(direction, np.degrees(d.qpos[sim.drive.qadr]), float(d.time),
                                'set material position for the approach helix')
        moved = trial.phase_run(1, 'arc_adjust', 9., command,
            stiffness=lambda: transfer.enabled.astype(float),
            stop_when=lambda _: transfer.stage == 'review' or transfer.fault is not None)
        if not moved or transfer.stage != 'review':
            break
        completed += 1
    after = support_material(trial)
    return transfer, dict(completed=completed == steps, steps_completed=completed,
                          before=before, after=after, window=list(transfer.window),
                          fault=transfer.fault, events=transfer.events)


def entry_geometry(trial, mask):
    info = release_snapshot(trial, mask)
    p = backbone(trial.model, trial.data)
    count = int(mask.sum())
    yz = p[:count+1, 1:]-[0., PIPE_HEIGHT]
    angles = np.unwrap(np.arctan2(yz[:, 1], yz[:, 0]))
    info.update(front_turns=float(abs(angles[-1]-angles[0])/(2.*math.pi)),
                head_x_m=float(p[0, 0]), front_x_min_m=float(np.min(p[:count+1, 0])),
                tail_tip_y_m=float(p[-1, 1]), tail_tip_height_m=float(p[-1, 2]-PIPE_HEIGHT))
    return info


def fit_tail(trial, scratch, joints, nodes, guide, command):
    """Fit only the tail hinges in private FK, with the actual head pose fixed."""
    m, d, drive = trial.model, trial.data, trial.sim.drive
    scratch.qpos[:] = d.qpos
    angles = command[joints].copy()
    measured = d.qpos[drive.qadr[joints]]
    jacobian = np.zeros((3, m.nv))
    head = m.body('head').id
    axes = drive.vadr[joints]
    for _ in range(4):
        scratch.qpos[drive.qadr[joints]] = angles
        mujoco.mj_kinematics(m, scratch)
        mujoco.mj_comPos(m, scratch)
        current = backbone(m, scratch)[nodes]
        matrix = np.empty((3*len(nodes), len(joints)))
        for i, (node, point) in enumerate(zip(nodes, current)):
            mujoco.mj_jac(m, scratch, jacobian, None, point, head+min(int(node), m.nu))
            matrix[3*i:3*i+3] = jacobian[:, axes]
        residual = (guide-current).ravel()
        delta = np.linalg.solve(matrix.T@matrix+.03**2*np.eye(len(joints)), matrix.T@residual)
        angles += np.clip(delta, -.12, .12)
        angles = np.clip(angles, np.maximum(-math.radians(88.), measured-.4),
                         np.minimum(math.radians(88.), measured+.4))
    return angles


def guide_right_u(trial, stiffness, keep, reference, front_command):
    """Lift the tail over the pipe in world coordinates while holding the front."""
    m, d, sim = trial.model, trial.data, trial.sim
    scratch = mujoco.MjData(m)
    start = backbone(m, d)
    desired = reference.copy()
    desired[:, 0] += start[8, 0]-desired[8, 0]
    yz_axis = np.array([0., PIPE_HEIGHT])
    source_yz, final_yz = start[:, 1:]-yz_axis, desired[:, 1:]-yz_axis
    source_theta = np.arctan2(source_yz[:, 1], source_yz[:, 0])
    final_theta = np.arctan2(final_yz[:, 1], final_yz[:, 0])
    turn = np.mod(final_theta-source_theta, 2.*np.pi)
    turn[turn > 1.25*np.pi] -= 2.*np.pi
    source_radius = np.linalg.norm(source_yz, axis=1)
    final_radius = np.linalg.norm(final_yz, axis=1)
    command = d.qpos[sim.drive.qadr].copy()
    last_fit = command.copy()
    begin = d.time
    next_fit = 0.
    ready_since = None
    target = None
    existing = stiffness.copy()

    def shape(t):
        nonlocal next_fit, last_fit
        delays = .34*np.arange(len(command)-8)
        gains = np.array([smooth((t-delay)/1.4) for delay in delays])
        stiffness[8:] = np.maximum(existing[8:], gains)
        if t >= next_fit:
            joints = np.flatnonzero(stiffness[8:] > 0.)+8
            if len(joints):
                nodes = np.arange(9, joints[-1]+3)
                mu = smooth(t/8.)
                theta = source_theta[nodes]+mu*turn[nodes]
                radius = source_radius[nodes]+mu*(final_radius[nodes]-source_radius[nodes])
                x = start[nodes, 0]+mu*(desired[nodes, 0]-start[nodes, 0])
                guide = np.column_stack([x, radius*np.cos(theta), PIPE_HEIGHT+radius*np.sin(theta)])
                last_fit[joints] = fit_tail(trial, scratch, joints, nodes, guide, command)
            next_fit = t+.04
        last_fit[:8] = front_command[:8]
        step = math.radians(35.)*m.opt.timestep
        command[:] += np.clip(last_fit-command, -step, step)
        return command

    def seated(q):
        nonlocal ready_since, target
        x = backbone(m, d)[:9, 0]
        valid = d.time-begin >= 3. and x[0] > BRANCH_X+.04 and min(x) < BRANCH_X and right_u_ready(trial, keep)
        ready_since = (d.time if ready_since is None else ready_since) if valid else None
        if ready_since is not None and d.time-ready_since >= .15:
            target = q.copy()
            return True
        return False

    trial.phase_run(1, 'arc_form_right_u', 13., shape, stiffness=lambda: stiffness, stop_when=seated)
    return target


def receive_tail(trial, stiffness, phase, radius):
    """Advance a small intake region only after the next real link wraps."""
    m, d, g, sim = trial.model, trial.data, trial.gait, trial.sim
    scratch = mujoco.MjData(m)
    measured = d.qpos[sim.drive.qadr].copy()
    command = measured.copy()
    n = len(measured)
    incoming = np.flatnonzero(stiffness > 0.)
    held = int(incoming[-1])+1
    limit = min(held, n-1)
    initial = attached_prefix(intake_connections(backbone(m, d), radius, g.o.pitch), limit)
    if initial < 4:
        return dict(initial_prefix=initial, attached_prefix=initial, completed=False,
                    events=[], reason='head helix has insufficient connected support')
    accepted = initial
    eligible_at = None
    events = []
    next_fit = 0.
    started = d.time
    nominal = g.targets(phase, radius)[0]
    goal = measured.copy()
    # Already powered ARC joints still hold their measured shape during entry;
    # torque on the suspended suffix stays off until the intake reaches it.
    retained = stiffness.copy()

    def command_next(t):
        nonlocal next_fit, eligible_at, accepted, goal
        angles = d.qpos[sim.drive.qadr]
        connections = intake_connections(backbone(m, d), radius, g.o.pitch)
        attached = attached_prefix(connections, n)
        previous_ready = attached >= accepted
        next_connected = accepted < n and previous_ready and connections[accepted]
        if next_connected:
            eligible_at = t if eligible_at is None else eligible_at
            if t-eligible_at >= .30:
                accepted += 1
                eligible_at = None
                next_fit = t
                events.append(dict(time_s=float(d.time), joint=accepted))
        else:
            eligible_at = None

        desired_scale = np.zeros(n)
        desired_scale[:accepted] = 1.
        end = min(n, accepted+2)
        desired_scale[accepted:end] = [.60, .35][:end-accepted]
        # Retain the original ARC support initially, then let the receiving
        # envelope decide which of those joints can still apply motor torque.
        desired_scale = np.maximum(desired_scale, retained*(1.-smooth(t/1.25)))
        stiffness[:] += np.clip(desired_scale-stiffness, -m.opt.timestep/.7, m.opt.timestep/.7)
        if t >= next_fit:
            goal = measured+smooth(t/2.)*(nominal-measured)
            first = max(1, min(accepted-1, attached+1)) if attached else accepted-1
            if accepted < n:
                local = fit_intake_window(trial, scratch, first, accepted, end, nominal, radius)
                goal[first-1:end] = local[first-1:end]
            next_fit = t+.04
        speed = math.radians(20.)*m.opt.timestep
        command[:] += np.clip(goal-command, -speed, speed)
        command[stiffness == 0.] = angles[stiffness == 0.]
        return command

    def wrapped(_):
        return (accepted == n and len(trial.samples) >= 8 and
                all(tail_engaged(row) for row in trial.samples[-8:]))

    formed = trial.phase_run(1, 'arc_intake', 34., command_next, stiffness=lambda: stiffness,
                             stop_when=wrapped)
    return dict(initial_prefix=initial, attached_prefix=accepted, completed=bool(formed and not trial.stopped),
                events=events, duration_s=float(d.time-started))


def enter_helix_then_unwind(trial, stiffness, mask, phase, release_radius, release_line, progressive=False):
    """Join the verified unwind at its pre-unwind helix, using real contacts."""
    d, g, sim = trial.data, trial.gait, trial.sim
    start_q = d.qpos[sim.drive.qadr].copy()
    existing = stiffness > 0.
    delays = np.zeros(sim.p.joints)
    delays[~existing] = np.arange(np.count_nonzero(~existing))*.70
    command = start_q.copy()
    radius = g.o.r_h1

    def form(t):
        trial.phase = phase
        q, _ = g.targets(trial.phase, radius)
        gains = np.array([smooth((t-delay)/1.4) for delay in delays])
        stiffness[:] = np.maximum(existing, gains)
        mix = np.array([smooth((t-delay)/3.) for delay in delays])
        command[:] = start_q+mix*(q-start_q)
        return command

    intake = None
    if progressive:
        intake = receive_tail(trial, stiffness, phase, radius)
    else:
        trial.phase_run(1, 'arc_finish_helix', float(max(delays))+4., form, stiffness=lambda: stiffness)
    outcome = {'formed': entry_geometry(trial, mask)}
    if intake is not None:
        outcome['intake'] = intake
    if trial.stopped or (intake is not None and not intake['completed']):
        return None, outcome
    roll_start = d.time

    def roll(t):
        trial.phase = phase+.6*t
        return g.targets(trial.phase, radius)[0]

    def aligned(_):
        points = backbone(trial.model, d)
        main, branch, _ = pipe_contacts(sim)
        crossings = top_crossings(points)
        # In the verified 18-joint approach the branch sat between
        # crossings at about x=0.286 and 0.422 m, with the head above.
        return (d.time-roll_start > .2 and any(link >= 13 for link in main) and
                any(8 <= link <= 12 for link in branch) and
                any(.25 < x < .32 for x in crossings) and any(.39 < x < .45 for x in crossings) and
                97. < angle(points[0]) < 151.)

    located = trial.phase_run(1, 'arc_align_helix', 14., roll, stop_when=aligned)
    outcome['alignment'] = entry_geometry(trial, mask)
    outcome['alignment']['phase_aligned'] = bool(located)
    if trial.stopped or not located:
        return None, outcome
    fixed = trial.phase
    trial.phase_run(1, 'unwind_arc', 3., lambda t: g.targets(fixed, radius, th1=math.pi*smooth(t/3.))[0])
    trial.phase_run(1, 'unwind_line', 4., lambda t: g.targets(fixed, radius, ls=release_line*smooth(t/4.),
                                                            th1=math.pi)[0])
    target = None

    def opening(t):
        r = radius+(release_radius-radius)*smooth(t/.85)
        return g.targets(fixed, r, ls=release_line, th1=math.pi)[0]

    def seated(q):
        nonlocal target
        if right_u_ready(trial, mask):
            target = q.copy()
            return True
        return False

    trial.phase_run(1, 'arc_form_right_u', 2., opening, stop_when=seated)
    return target, outcome


def capture(trial, transfer, phase_offset, align=False, full_helix=False, keep_arc=False,
            fit_support=False, spatial_reference=None, progressive=False):
    """Keep the arc loaded while adding curvature ahead and then shaping the U."""
    g, d, sim = trial.gait, trial.data, trial.sim
    mask = np.arange(sim.p.joints) < 8
    first, last = transfer.window
    anchor = (first+last)*.5*g.l
    release_radius = .09554165
    release_line = .32201388
    torsion = g.segments(release_radius)[0][0].tau
    trial.phase = transfer.phase-torsion*anchor+phase_offset
    if fit_support:
        # The existing ARC can overlap the U rather than the helix. Fit the
        # phase from the actual angles on the retained support instead of
        # extrapolating helical torsion across that boundary.
        bases = [g.targets(phase, release_radius, ls=release_line, th1=math.pi)[0]
                 for phase in (0., math.pi/2.)]
        basis = np.column_stack(bases)[transfer.enabled]
        observed = d.qpos[sim.drive.qadr][transfer.enabled]
        weights, *_ = np.linalg.lstsq(basis, observed, rcond=None)
        trial.phase = math.atan2(weights[1], weights[0])+phase_offset
    if progressive:
        # Form only the head-side helix here. The direct U template used to
        # introduce tail curvature while the tail was still suspended.
        final, _ = g.targets(trial.phase, release_radius)
    else:
        final, _ = g.targets(trial.phase, release_radius, ls=release_line, th1=math.pi)
    q = d.qpos[sim.drive.qadr].copy()
    start_q = q.copy()
    stiffness = transfer.enabled.astype(float)
    head_stage = mask | transfer.enabled
    if keep_arc:
        entering_head = mask & ~transfer.enabled

        def wrap_head(t):
            stiffness[entering_head] = smooth(t/1.5)
            return start_q+smooth(t/4.)*(final-start_q)*entering_head

        trial.phase_run(1, 'arc_capture_head', 4.5, wrap_head, stiffness=lambda: stiffness)
        if trial.stopped:
            return None, mask, entry_geometry(trial, mask)
        start_q = d.qpos[sim.drive.qadr].copy()

    def head(t):
        ratio = smooth(t/6.)
        if not keep_arc:
            stiffness[mask] = smooth(t/2.)
        stiffness[transfer.enabled] = 1.
        q[:] = start_q+ratio*(final-start_q)*head_stage
        return q

    trial.phase_run(1, 'arc_form_helix', 6.5, head, stiffness=lambda: stiffness)
    after_head = entry_geometry(trial, mask)
    if trial.stopped:
        return None, mask, after_head
    if spatial_reference is not None:
        target = guide_right_u(trial, stiffness, mask, spatial_reference, final)
        return target, mask, after_head
    if full_helix or progressive:
        target, outcome = enter_helix_then_unwind(
            trial, stiffness, mask, trial.phase, release_radius, release_line, progressive)
        after_head['full_helix'] = outcome
        return target, mask, after_head
    if align:
        start_q = d.qpos[sim.drive.qadr].copy()
        phase_at_start = trial.phase
        begin_align = d.time

        def align_front(t):
            trial.phase = phase_at_start+.48*max(0., t-1.5)
            helix_target, _ = g.targets(trial.phase, release_radius)
            return start_q+smooth(t/1.5)*(helix_target-start_q)

        def phase_aligned(_):
            p = backbone(trial.model, d)
            y, z = p[8, 1:]-[0., PIPE_HEIGHT]
            # The real junction faces the right shoulder of the pipe before
            # the tail is lifted, leaving space to place the U from above.
            azimuth = math.atan2(z, y)
            return d.time-begin_align > 2. and abs(math.atan2(
                math.sin(azimuth+2.7), math.cos(azimuth+2.7))) < .25

        aligned = trial.phase_run(1, 'arc_align_helix', 14., align_front, stiffness=lambda: stiffness,
                                  stop_when=phase_aligned)
        after_head['alignment'] = entry_geometry(trial, mask)
        after_head['alignment']['phase_aligned'] = bool(aligned)
        if trial.stopped or not aligned:
            return None, mask, after_head
        final, _ = g.targets(trial.phase, release_radius, ls=release_line, th1=math.pi)
    start_q = d.qpos[sim.drive.qadr].copy()
    entering = np.flatnonzero(~head_stage)
    enable_time = np.zeros(sim.p.joints)
    enable_time[entering] = np.arange(len(entering))*.42

    def tail(t):
        gain = np.array([smooth((t-a)/1.4) for a in enable_time])
        stiffness[:] = np.maximum(head_stage, gain)
        mix = np.array([smooth((t-a)/5.) for a in enable_time])
        return start_q+mix*(final-start_q)

    ready_since = None
    target = None

    def seat(command):
        nonlocal ready_since, target
        x = backbone(trial.model, d)[:9, 0]
        ready = (d.time-tail_start >= 3. and x[0] >= BRANCH_X+PIPE_RADIUS and min(x) < BRANCH_X
                 and right_u_ready(trial, mask))
        ready_since = (d.time if ready_since is None else ready_since) if ready else None
        if ready_since is not None and d.time-ready_since >= .15:
            target = command.copy()
            return True
        return False

    tail_start = d.time
    trial.phase_run(1, 'arc_form_right_u', 10., tail, stiffness=lambda: stiffness, stop_when=seat)
    return target, mask, after_head


def run(output, trigger_x, phase_offset, align, mirrored, feed_steps, full_helix, keep_arc, feed_direction,
        fit_support, guide_tail, progressive):
    trial = CycleTrial(setup=APPROACH_SETUP)
    sim, d = trial.sim, trial.data
    p = replace(sim.p, first_joint=7, last_joint=12, initial_phase_rad=math.pi if mirrored else 0.)
    sim.p = sim.drive.p = p
    reference = None
    if guide_tail:
        trace_path = Path(__file__).resolve().parent/'reports/gm_locked_process/continuous18_final/locked_gait_trace.npz'
        scratch = mujoco.MjData(trial.model)
        with np.load(trace_path) as trace:
            index = np.argmin(abs(trace['time']-10.85))
            scratch.qpos[:] = trace['qpos'][index]
        mujoco.mj_forward(trial.model, scratch)
        reference = backbone(trial.model, scratch)
    sim.place(.12)  # only physical placement
    trial.phase_run(1, 'settle', p.settle_s, lambda _: p.targets(p.initial_phase_rad), p.enabled)

    def rolling(t):
        trial.phase = p.initial_phase_rad+(-1. if mirrored else 1.)*p.omega_rad_s*t
        return p.targets(trial.phase)

    def reach(_):
        p_arc = backbone(trial.model, d)[p.first_joint:p.last_joint+1]
        return np.mean(p_arc[:, 0]) >= trigger_x

    reached = trial.phase_run(1, 'arc_approach', 15., rolling, p.enabled, stop_when=reach)
    summary = dict(arc_trigger_x_m=trigger_x, phase_offset_rad=phase_offset, align_before_u=align,
                   mirrored_arc=mirrored, feed_steps=feed_steps, feed_direction=feed_direction,
                   full_helix=full_helix, keep_arc=keep_arc, fit_support=fit_support, guide_tail=guide_tail,
                   progressive_intake=progressive,
                   reached_approach=bool(reached), start_x_m=.12,
                   setup={k: APPROACH_SETUP[k] for k in ('scene', 'joints', 'friction', 'options')})
    if reached and not trial.stopped:
        transfer, feed = adjust_arc(trial, p, feed_steps, feed_direction)
        summary['feed'] = feed
        if feed['completed'] and not trial.stopped:
            target, mask, after_head = capture(trial, transfer, phase_offset, align, full_helix, keep_arc,
                                               fit_support, reference, progressive)
            summary['after_head'] = after_head
            summary['entry'] = entry_geometry(trial, mask)
            if target is not None and not trial.stopped:
                summary['crossing'] = drop_and_rewrap(
                    trial, 1, target, mask, mask, .098, .40, .35, False, rewrap)
    save(trial, output, 'locked_gait', None, **summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trigger-x', type=float)
    parser.add_argument('--phase-offset', type=float, default=0.)
    parser.add_argument('--align-before-u', action='store_true')
    parser.add_argument('--mirrored-arc', action='store_true')
    parser.add_argument('--feed-steps', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--feed-direction', choices=('head', 'tail'), default='head')
    parser.add_argument('--unwind-from-helix', action='store_true')
    parser.add_argument('--keep-arc-during-head-capture', action='store_true')
    parser.add_argument('--fit-retained-support', action='store_true')
    parser.add_argument('--guide-tail', action='store_true')
    parser.add_argument('--legacy-entry', action='store_true',
                        help='Reproduce the earlier direct-lift and comparison runs')
    args = parser.parse_args()
    if not args.legacy_entry and (args.guide_tail or args.fit_retained_support or
                                  args.align_before_u or args.unwind_from_helix):
        parser.error('the old entry variants require --legacy-entry')
    trigger_x = args.trigger_x if args.trigger_x is not None else (.270 if args.legacy_entry else .195)
    run(args.output, trigger_x, args.phase_offset, args.align_before_u, args.mirrored_arc, args.feed_steps,
        args.unwind_from_helix, args.keep_arc_during_head_capture, -1 if args.feed_direction == 'head' else 1,
        args.fit_retained_support, args.guide_tail, not args.legacy_entry)
