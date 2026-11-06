"""Continue a saved crossing attempt for two unwind and whole-body shift cycles."""
import argparse
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.gravity_global_shift import GravityGlobalShift, MeasuredGravityShift
from snake_control.gravity_moment import GravityMomentGait, GravityMomentOptions, smooth
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import backbone
from snake_control.tee_simulation import PIPE_HEIGHT, TeeSimulation
from validation.check_gm_global_shift import load_checkpoint
from validation.gm_experiment import top_crossings
from validation.gm_wrap_first_experiment import pipe_contacts


class CycleTrial:
    def __init__(self, source=None, reference=None, passive_tail_damping=None, *, setup=None):
        if source is None:
            summary = setup
            n = summary['joints']
            self.gait = GravityMomentGait(n, GravityMomentOptions(**summary['options']))
            self.phase = 0.
        else:
            record = json.loads(source.read_text(encoding='utf-8'))
            summary = record['summary']
            trace_path = source.with_name(source.stem+'_trace.npz')
            with np.load(trace_path) as trace:
                n = trace['targets'].shape[1]
                initial_qpos = trace['qpos'][-1]
                initial_qvel = trace['qvel'][-1]
            if reference is None:
                self.gait = GravityMomentGait(n, GravityMomentOptions(**summary['options']))
                self.phase = record['samples'][-1]['roll_rad']
            else:
                _, _, _, self.gait, shift, _, _, _ = load_checkpoint(reference, summary['source_time_s'])
                self.phase = shift.phase_at_curve_start
        self.sim = TeeSimulation(summary['scene'], ArcParameters(joints=n, first_joint=1, last_joint=n),
                                 friction=summary['friction'])
        self.model, self.data = self.sim.model, self.sim.data
        if source is not None:
            self.data.qpos[:] = initial_qpos
            self.data.qvel[:] = initial_qvel
            mujoco.mj_forward(self.model, self.data)
        self.full_mask = np.ones(n, bool)
        self.powered_damping = self.model.dof_damping[self.sim.drive.vadr].copy()
        self.passive_tail_damping = passive_tail_damping
        self.next_sample = 0.
        self.samples, self.states, self.velocities, self.targets, self.masks, self.forces = ([] for _ in range(6))
        self.stiffness_scales = []
        self.choices = []
        self.release_checks = []
        self.release_events = []
        self.rewrap_checks = []
        self.released_support_count = None
        self.stopped = False
        self.grip_since = None

    def choose_release(self):
        """Operator preview in the current root pose. Only the scratch data is moved."""
        g, m, d = self.gait, self.model, self.data
        preview = mujoco.MjData(m)
        preview.qpos[:] = d.qpos
        standard = g.turn_length(g.o.r_h1)-math.pi*g.o.r_c
        candidates = []
        for extra in np.arange(-1.5, 1.51, .25):
            length = standard+extra*g.l
            target, lh = g.targets(self.phase, g.o.r_h1, ls=length, th1=math.pi)
            preview.qpos[self.sim.drive.qadr] = target
            mujoco.mj_forward(m, preview)
            points = backbone(m, preview)
            first = int(math.ceil(lh/g.l))
            last = int(math.floor((lh+math.pi*g.o.r_c)/g.l))
            head = m.body('head').id
            tail_bodies = slice(head+first, head+g.gait.joints+1)
            height = float(np.min(points[first:last+1, 2]-PIPE_HEIGHT))
            moment = float(-9.81*np.dot(m.body_mass[tail_bodies], preview.xipos[tail_bodies, 1]))
            # Keep the U on the upper side and bias the released tail toward
            # positive rotation. Prefer a release near one helix turn.
            score = (height >= .025 and moment > 0, height + .018*moment-.012*abs(extra))
            candidates.append((score, length, lh, height, moment))
        _, length, lh, height, moment = max(candidates, key=lambda row: row[0])
        self.choices.append(dict(time_s=self.data.time, straight_m=length, helix_m=lh,
                                 planned_u_min_above_axis_m=height, planned_tail_moment_nm=moment))
        return length, lh

    def phase_run(self, cycle, name, duration, targets, mask=None, stop_on_wrap=False, stop_when=None,
                  stiffness=None, torque_limit=None):
        d, m, sim = self.data, self.model, self.sim
        saturation_limit = sim.p.torque_limit_nm if torque_limit is None else torque_limit
        mask = self.full_mask if mask is None else mask
        damping = self.powered_damping.copy()
        if self.passive_tail_damping is not None:
            damping[~mask] = self.passive_tail_damping
        m.dof_damping[sim.drive.vadr] = damping
        start = d.time
        self.grip_since = None
        tail_loose_since = None
        while d.time-start < duration and not self.stopped:
            elapsed = d.time-start
            q = targets(elapsed)
            scale = None if stiffness is None else stiffness()
            if scale is not None:
                mask = scale > 0.
                if self.passive_tail_damping is not None:
                    m.dof_damping[sim.drive.vadr] = np.where(mask, self.powered_damping, self.passive_tail_damping)
            sim.drive.apply_targets(d, q, mask, stiffness=scale, torque_limit=torque_limit)
            mujoco.mj_step(m, d)
            if name == 'drop' and elapsed == 0.:
                self.release_events.append(dict(command_time_s=start, first_step_time_s=float(d.time),
                    disabled_joints=(np.flatnonzero(~mask)+1).tolist(),
                    disabled_damping=damping[~mask].tolist(),
                    disabled_actuator_force_nm=d.actuator_force[~mask].tolist()))
            if stop_when is not None and stop_when(q):
                return True
            if not np.isfinite(d.qpos).all():
                raise RuntimeError('nonfinite simulation state')
            if d.time+1e-9 < self.next_sample:
                continue
            self.next_sample += .05
            stage = name() if callable(name) else name
            points = backbone(m, d)
            yz = points[-7:, 1:]-[0., PIPE_HEIGHT]
            angles = np.unwrap(np.arctan2(yz[:, 1], yz[:, 0]))
            tail_span = float(np.degrees(np.ptp(angles)))
            tail_radius = float(max(np.linalg.norm(yz, axis=1)))
            main, branch, branch_force = pipe_contacts(sim)
            tail = [j for j in main if j >= len(mask)-5]
            wrapped = len(tail) >= 2 and tail_span >= 75. and tail_radius < .14
            self.grip_since = (d.time if self.grip_since is None else self.grip_since) if wrapped else None
            metrics = sim.measure(self.phase)
            error = np.degrees(abs(q-d.qpos[sim.drive.qadr]))
            self.samples.append(dict(time_s=float(d.time), cycle=cycle, stage=stage,
                x_min_m=float(points[:, 0].min()), head_x_m=float(points[0, 0]), center_x_m=float(points[:, 0].mean()),
                reference_roll_rad=float(self.phase),
                tail_z_m=float(points[-1, 2]), min_height_m=float(points[:, 2].min()),
                front_angle_deg=float(np.degrees(np.arctan2(points[4, 2]-PIPE_HEIGHT, points[4, 1]))),
                tail_span_deg=tail_span, tail_max_radius_m=tail_radius,
                main_links=main, distal_contacts=tail, tail_wrapped=wrapped, branch_links=branch,
                branch_force_world_n=branch_force.tolist(), support_n=metrics['normal_force_n'],
                ground_force_n=metrics['ground_contact_force_n'], enabled_count=int(mask.sum()),
                tracking_max_deg=float(max(error[mask])), actual_max_joint_deg=float(max(abs(np.degrees(d.qpos[sim.drive.qadr])))),
                saturated_joints=(np.flatnonzero(abs(d.actuator_force) >= saturation_limit-1e-6)+1).tolist()))
            if name == 'drop':
                tail_y, torque = self.release_moment(mask)
                head = m.body('head').id
                tail_bodies = slice(head+int(mask.sum())+1, head+len(mask)+1)
                tail_com = np.average(d.xipos[tail_bodies], axis=0, weights=m.body_mass[tail_bodies])
                self.samples[-1].update(released_tail_y_m=tail_y, released_tail_gravity_torque_nm=torque,
                                        released_tail_com_m=tail_com.tolist(),
                                        passive_force_max_nm=float(max(abs(d.actuator_force[~mask]))))
            self.states.append(d.qpos.copy())
            self.velocities.append(d.qvel.copy())
            self.targets.append(q.copy())
            self.masks.append(mask.copy())
            self.forces.append(d.actuator_force.copy())
            self.stiffness_scales.append(mask.astype(float) if scale is None else scale.copy())
            if self.samples[-1]['ground_force_n'] > 1.:
                self.stopped = True
            if name == 'aux_roll':
                loose = tail_radius > .17 or tail_span < 75.
                tail_loose_since = (d.time if tail_loose_since is None else tail_loose_since) if loose else None
                if tail_loose_since is not None and d.time-tail_loose_since >= .3:
                    break
            if stop_on_wrap and self.grip_since is not None and d.time-self.grip_since >= .5:
                return True
        return stop_when is None and self.grip_since is not None and d.time-self.grip_since >= .5

    def tighten(self, cycle):
        g = self.gait
        self.phase_run(cycle, 'tighten', 1.5,
            lambda t: g.targets(self.phase, g.o.r_h2+(g.o.r_h1-g.o.r_h2)*smooth(t/1.5))[0])

    def release_moment(self, mask):
        head = self.model.body('head').id
        first_passive = int(np.flatnonzero(~mask)[0]) + 1
        tail = slice(head+first_passive, head+len(mask)+1)
        mass = self.model.body_mass[tail]
        center_y = float(np.average(self.data.xipos[tail, 1], weights=mass))
        return center_y, -9.81*float(mass.sum())*center_y

    def drop_right(self, index, ls, lh):
        """Open the front with the tail powered, then switch the tail off at once."""
        g, o = self.gait, self.gait.o
        mask = g.helix_joints(lh)
        before = g.targets(self.phase, o.r_h1, ls=ls, th1=math.pi)[0]
        opened = g.targets(self.phase, o.r_h2, ls=ls, th1=math.pi)[0]
        held_tail = self.data.qpos[self.sim.drive.qadr].copy()
        u_first = math.ceil(lh/g.l)
        u_last = math.floor((lh+math.pi*o.r_c)/g.l)
        front_count = int(mask.sum())+1
        yz_axis = np.array([0., PIPE_HEIGHT])
        min_opening = .0005

        def front_radius(points):
            return float(np.mean(np.linalg.norm(points[:front_count, 1:]-yz_axis, axis=1)))

        initial_radius = front_radius(backbone(self.model, self.data))
        release_target = before.copy()

        def loosen(t):
            q = before+smooth(t/.12)*(opened-before)
            q[~mask] = held_tail[~mask]
            return q

        def opened_while_u_ready(q):
            points = backbone(self.model, self.data)
            center_y, _ = self.release_moment(mask)
            u_high = min(points[u_first:u_last+1, 2]) > PIPE_HEIGHT
            if front_radius(points)-initial_radius < min_opening or center_y >= -.02 or not u_high:
                return False
            release_target[:] = q
            return True

        ready = self.phase_run(index, 'loosen', .12, loosen, stop_when=opened_while_u_ready)
        tail_y, torque = self.release_moment(mask)
        main, branch, _ = pipe_contacts(self.sim)
        points = backbone(self.model, self.data)
        min_u_height = float(min(points[u_first:u_last+1, 2])-PIPE_HEIGHT)
        ready = ready and not self.stopped
        self.release_checks.append(dict(time_s=float(self.data.time), cycle=index, permitted=ready,
            tail_center_y_m=tail_y, gravity_torque_nm=torque, min_u_height_m=min_u_height,
            front_opening_m=front_radius(points)-initial_radius,
            main_links=main, branch_links=branch))
        if ready:
            self.released_support_count = int(mask.sum())
            self.phase_run(index, 'drop', 3.0, lambda t: release_target, mask)
        return ready

    def cycle(self, index):
        g, o = self.gait, self.gait.o
        ls, lh = self.choose_release()
        self.phase_run(index, 'unwind_arc', 2.5,
            lambda t: g.targets(self.phase, o.r_h1, th1=math.pi*smooth(t/2.5))[0])
        self.phase_run(index, 'unwind_line', 3.5,
            lambda t: g.targets(self.phase, o.r_h1, ls=ls*smooth(t/3.5), th1=math.pi)[0])
        if self.drop_right(index, ls, lh):
            self.rewrap(index)
        else:
            self.stopped = True

    def rewrap(self, index):
        g, o = self.gait, self.gait.o
        if self.stopped:
            return
        straight = (g.n_ext(o.r_h2)+1)*g.l
        shift = GravityGlobalShift(g, o.r_h2, self.phase, straight)
        start, goal = self.data.qpos[self.sim.drive.qadr].copy(), shift.targets(0.)
        tail_dz = self.reference_tail_dz(goal)
        if tail_dz > 0.:
            result = self.rewrap_measured(index, self.released_support_count)
            self.rewrap_checks.append(dict(cycle=index, method='measured_curve', reference_tail_dz_m=tail_dz, **result))
            # Continue the next cycle only after a supported full rewrap.
            self.stopped |= not (result['complete_shift'] and result['stable_wrap'])
            return
        self.phase_run(index, 'fit', 3., lambda t: start+smooth(t/2.0)*(goal-start))
        fit_error = float(max(abs(np.degrees(goal-self.data.qpos[self.sim.drive.qadr]))))
        self.rewrap_checks.append(dict(cycle=index, method='nominal_curve', reference_tail_dz_m=tail_dz,
                                       fit_error_deg=fit_error, follow_fit=fit_error <= 8.))
        if fit_error > 8.:
            self.stopped = True
            return
        self.phase_run(index, 'shift', shift.shift_limit_m/.03+.05,
                       lambda t: shift.targets(min(.03*t, shift.shift_limit_m)))
        self.phase = shift.phase_at_curve_start
        self.tighten(index)
        # Simulated contact is an operator-style check before adding active rolling.
        ready = self.phase_run(index, 'check_grip', 2., lambda t: g.targets(self.phase, o.r_h1)[0], stop_on_wrap=True)
        if ready and not self.stopped:
            initial_phase = self.phase
            roll_start = self.data.time
            self.phase_run(index, 'aux_roll', 8., lambda t: g.targets(initial_phase+.35*t, o.r_h1)[0])
            self.phase = initial_phase+.35*(self.data.time-roll_start)
            self.phase_run(index, 'hold', 1., lambda t: g.targets(self.phase, o.r_h1)[0])

    def reference_tail_dz(self, angles):
        preview = mujoco.MjData(self.model)
        preview.qpos[:] = self.data.qpos
        preview.qpos[self.sim.drive.qadr] = angles
        mujoco.mj_forward(self.model, preview)
        points = backbone(self.model, preview)
        return float(points[-1, 2]-points[-5, 2])

    def rewrap_measured(self, index, support_count):
        angles = self.data.qpos[self.sim.drive.qadr].copy()
        plan = MeasuredGravityShift(self.gait, self.gait.o.r_h2, self.phase, angles, support_count)
        self.phase_run(index, 'fit_measured', .25, lambda t: angles)
        position, last_target = 0., angles.copy()
        stalled_since = reached_s = None

        def command(t):
            nonlocal position, last_target, stalled_since, reached_s
            measured = self.data.qpos[self.sim.drive.qadr]
            if max(abs(np.degrees(last_target-measured))) > 8.:
                stalled_since = t if stalled_since is None else stalled_since
            else:
                stalled_since = None
                position = min(position+.03*self.model.opt.timestep, plan.shift_limit_m)
            last_target = np.clip(plan.targets(position), -np.pi/2, np.pi/2)
            if position >= plan.shift_limit_m and reached_s is None:
                reached_s = t
            return last_target

        shift_start = self.data.time

        def finished(q):
            elapsed = self.data.time-shift_start
            return (stalled_since is not None and elapsed-stalled_since >= 2.) or (
                    reached_s is not None and elapsed-reached_s >= 1.)

        self.phase_run(index, 'measured_shift', plan.shift_limit_m/.03+6., command, stop_when=finished)
        stable_wrap = self.phase_run(index, 'hold', 1., lambda t: last_target)
        if reached_s is not None:
            self.phase -= plan.torsion*position
        return dict(shift_m=position, shift_limit_m=plan.shift_limit_m, tracking_wait_deg=8.,
                    stopped_on_tracking=stalled_since is not None, complete_shift=reached_s is not None,
                    stable_wrap=stable_wrap)


def run(source, reference, output, right_checkpoint=False, aligned_trace=None,
        passive_tail_damping=None, continue_after_drop=False):
    trial = CycleTrial(source, reference, passive_tail_damping)
    if right_checkpoint:
        # Initial operator placement from a kinematic preview. The physical
        # simulation then runs continuously through drop and rewrap.
        g, o, d = trial.gait, trial.gait.o, trial.data
        trial.phase = 0.
        ls = g.turn_length(o.r_h1)-math.pi*o.r_c+.25*g.l
        target, lh = g.targets(trial.phase, o.r_h1, ls=ls, th1=math.pi)
        if aligned_trace is None:
            points_on_helix = int(lh//g.l)
            trial.sim.place_shape(target, points_on_helix, .42, math.radians(-135.))
            top = top_crossings(backbone(trial.model, d)[:points_on_helix+1])[-1]
            d.qpos[0] += .425-top
        else:
            # Reuse the already settled right-U pose. Move it axially once,
            # so a small positive roll approaches the far side of the branch.
            with np.load(aligned_trace) as trace:
                frame = int(np.argmin(abs(trace['time']-.1)))
                d.qpos[:] = trace['qpos'][frame]
                d.qvel[:] = trace['qvel'][frame]
            d.qpos[0] -= .0143
        mujoco.mj_forward(trial.model, d)
        trial.phase_run(1, 'right_checkpoint', .12 if aligned_trace is None else .001, lambda t: target)
        if trial.drop_right(1, ls, lh) and (aligned_trace is None or continue_after_drop):
            trial.rewrap(1)
    else:
        trial.tighten(0)
        for i in range(1, 3):
            if trial.stopped:
                break
            trial.cycle(i)
            if trial.samples[-1]['x_min_m'] > .409:
                break
    rows = trial.samples
    summary = dict(scene='up60', joints=26, source='manual_right_u' if right_checkpoint else source.stem,
                   release_choices=trial.choices, release_checks=trial.release_checks,
                   release_events=trial.release_events, passive_tail_damping=passive_tail_damping,
                   rewrap_checks=trial.rewrap_checks,
                   initial_alignment_axial_m=-.0143 if aligned_trace is not None else None,
                   dynamics_s=float(trial.data.time), clear_samples=sum(row['x_min_m'] > .409 for row in rows),
                   aux_roll_started=any(row['stage']=='aux_roll' for row in rows),
                   end_x_min_m=rows[-1]['x_min_m'], ground_force_max_n=max(row['ground_force_n'] for row in rows))
    output.mkdir(parents=True, exist_ok=True)
    (output/'shift_cycle.json').write_text(json.dumps(dict(summary=summary, samples=rows), indent=1), encoding='utf-8')
    np.savez_compressed(output/'shift_cycle_trace.npz', time=[row['time_s'] for row in rows],
        qpos=trial.states, qvel=trial.velocities, targets=trial.targets, masks=trial.masks, actuator_force=trial.forces)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--right-checkpoint', action='store_true')
    parser.add_argument('--aligned-trace', type=Path, help='One short drop from an axially aligned saved right U')
    parser.add_argument('--passive-tail-damping', type=float, help='Damping sensitivity for disabled tail joints only')
    parser.add_argument('--continue-after-drop', action='store_true', help='Rewrap after the aligned drop')
    args = parser.parse_args()
    if args.aligned_trace is not None and not args.right_checkpoint:
        parser.error('--aligned-trace requires --right-checkpoint')
    if args.passive_tail_damping is not None and (
            not math.isfinite(args.passive_tail_damping) or args.passive_tail_damping < 0.):
        parser.error('--passive-tail-damping must be finite and non-negative')
    run(args.source, args.reference, args.output, args.right_checkpoint, args.aligned_trace,
        args.passive_tail_damping, args.continue_after_drop)
