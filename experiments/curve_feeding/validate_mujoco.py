"""Paired actual-physics trials, with the existing real hotkey dispatcher.

No serial connection. World/contact information is saved for review only.
The only controller substitution is real.ContinuousFeeding in this process.
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from codes import Rolling_Arc_Real as real
from codes.Rolling_Arc_Debug_Mujoco import MujocoMotorBus
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import backbone
from snake_control.tee_simulation import TeeSimulation
from validation.gm_arc_handoff_experiment import place_arc_for_adjustment
from validation.mode_handoff_experiment import ExperimentController
from .backend import curve_feeding_class

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'validation/reports/curve_feeding_20261108'
# Keep the original comparator after the curve backend is released to real.
BASE = getattr(real, 'LegacyContinuousFeeding', real.ContinuousFeeding)
CURVE = curve_feeding_class(BASE)
CASES = dict(roundtrip=(7, 12), boundary=(2, 7))


class RecordingBus(MujocoMotorBus):
    def __init__(self, sim):
        self.handovers = []
        super().__init__(sim)

    def prepare_enabled(self, angles, enabled, configure_mask=None):
        entering = np.asarray(enabled) & ~self.enabled
        self.handovers.append(dict(time_s=float(self.sim.data.time),
            entering=(np.flatnonzero(entering)+1).tolist(),
            leaving=(np.flatnonzero(self.enabled & ~np.asarray(enabled))+1).tolist(),
            preload_deg=np.asarray(angles)[entering].tolist(),
            actual_deg=np.degrees(self.sim.data.qpos[self.sim.drive.qadr])[entering].tolist()))
        super().prepare_enabled(angles, enabled, configure_mask)


def make_initial(case, output):
    p = ArcParameters(joints=18, first_joint=CASES[case][0], last_joint=CASES[case][1])
    sim = TeeSimulation('straight', p, friction=.5)
    trial = SimpleNamespace(model=sim.model, data=sim.data, sim=sim)
    place_arc_for_adjustment(trial, p, .1)
    for _ in range(2500):
        sim.drive.apply_targets(sim.data, p.targets(0.), p.enabled)
        mujoco.mj_step(sim.model, sim.data)
    np.savez_compressed(output/(case+'_initial.npz'), qpos=sim.data.qpos,
        qvel=sim.data.qvel, warmstart=sim.data.qacc_warmstart)
    return p


def run(case, backend, output, parameters):
    real.ContinuousFeeding = CURVE if backend == 'curve' else BASE
    p = parameters
    sim = TeeSimulation('straight', p, friction=.5)
    m, d = sim.model, sim.data
    initial = np.load(output/(case+'_initial.npz'))
    d.qpos[:], d.qvel[:], d.qacc_warmstart[:] = initial['qpos'], initial['qvel'], initial['warmstart']
    mujoco.mj_forward(m, d)
    real.FIRST_JOINT, real.LAST_JOINT = CASES[case]
    c = ExperimentController(RecordingBus(sim))
    c.targets, c.enabled = np.degrees(p.targets(0.)).tolist(), p.enabled.tolist()
    c.first, c.last = CASES[case]
    c.configured_window = c.feeding_origin = CASES[case]
    c.phase = c.command_phase = 0.
    c.mode, c.touched_hardware = 'hold', True
    c.bus.prepare_enabled(c.targets, c.enabled)
    commands, control_rows, rows, curves = [], [], [], []
    states, velocities, targets, masks, forces, points_record = ([] for _ in range(6))
    stage_events, finished_actions = [], []
    statistics = dict(passive_force_max_nm=0., saturated_joint_steps=0,
        joint_steps=0, limit_excess_max_rad=0., self_contact_steps=0,
        floor_contact_steps=0, max_shared_target_increment_deg=0.)
    limits = m.jnt_range[m.actuator_trnid[:, 0]]
    floor = m.geom('floor').id
    previous_targets, previous_mask = c.bus.targets_deg.copy(), c.bus.enabled.copy()
    action = 0
    paused_at = resumed_at = forced_at = terminal = None
    last_stage = None
    failure = None
    progress = 10.
    next_feedback = 0.
    next_sample = 0.

    def command(key):
        event = dict(time_s=float(d.time), key='Ctrl+'+key,
            stage_before=None if c.feeding is None else c.feeding.stage)
        try:
            with redirect_stdout(io.StringIO()):
                c.command(real.decode_hotkey(ord(key), True))
        except Exception as exc:
            event.update(result='rejected', reason=str(exc))
            commands.append(event)
            raise
        event.update(result='accepted', stage=None if c.feeding is None else c.feeding.stage)
        commands.append(event)

    command('4')
    command('J' if case == 'roundtrip' else 'H')
    for frame in range(5000):  # 100s ceiling; completed cases stop after 3s hold.
        t = float(d.time)
        if t+1e-9 >= next_feedback:
            c.refresh_compliance_feedback()
            next_feedback += .2
        core = c.feeding
        stage = core.stage
        if stage != last_stage:
            stage_events.append(dict(time_s=t, action=action, stage=stage,
                position_joints=core.position, fault=core.fault))
            last_stage = stage
        if case == 'roundtrip':
            if action == 0 and stage == 'feed_out' and paused_at is None:
                command('G'); paused_at = t
            elif paused_at is not None and resumed_at is None and t-paused_at >= .6:
                command('G'); resumed_at = t
            elif resumed_at is not None and forced_at is None and t-resumed_at >= 1.:
                command('L'); forced_at = t
            elif forced_at is not None and action == 0 and c.feeding_paused and core.forced_step is None:
                command('G')
        else:
            # At material-coordinate boundary, L explicitly attempts one
            # unavailable H frame; then J must still start a reverse task.
            if action == 0 and stage == 'feed_out' and forced_at is None:
                command('L'); forced_at = t
            elif action == 0 and forced_at is not None and t-forced_at >= .6:
                finished_actions.append(dict(kind='boundary_attempt', time_s=t,
                    position_joints=core.position, snapshot=core.snapshot()))
                command('J'); action = 1; last_stage = None
        if core.completed and terminal is None:
            finished_actions.append(dict(kind='relative_goal', action=action, time_s=t, snapshot=core.snapshot()))
            if case == 'roundtrip' and action == 0:
                command('H'); action = 1; last_stage = None
            else:
                terminal = t
        if core.fault:
            failure = core.fault
        with redirect_stdout(io.StringIO()):
            c.tick(.02)
        current_targets, current_mask = c.bus.targets_deg.copy(), c.bus.enabled.copy()
        shared = current_mask & previous_mask
        statistics['max_shared_target_increment_deg'] = max(statistics['max_shared_target_increment_deg'],
            float(np.max(abs(current_targets[shared]-previous_targets[shared]), initial=0.)))
        previous_targets, previous_mask = current_targets, current_mask
        control_rows.append(dict(time_s=t, action=action, mode=c.mode,
            paused=c.feeding_paused, feeding=core.snapshot(), motor_input=c.feeding_inputs,
            target_deg=c.targets.copy(), enabled=c.enabled.copy()))
        for _ in range(20):
            c.bus.apply_drive()
            mujoco.mj_step(m, d)
            statistics['joint_steps'] += p.joints
            statistics['passive_force_max_nm'] = max(statistics['passive_force_max_nm'],
                float(np.max(abs(d.actuator_force[~c.bus.enabled]), initial=0.)))
            statistics['saturated_joint_steps'] += int(np.count_nonzero(abs(d.actuator_force) >= p.torque_limit_nm-1e-6))
            q = d.qpos[sim.drive.qadr]
            statistics['limit_excess_max_rad'] = max(statistics['limit_excess_max_rad'],
                float(np.maximum(np.maximum(limits[:, 0]-q, q-limits[:, 1]), 0.).max()))
            for contact in d.contact:
                if floor in (contact.geom1, contact.geom2) and {int(contact.geom1), int(contact.geom2)} & sim.robot_geoms:
                    statistics['floor_contact_steps'] += 1
                a, b = m.geom_bodyid[[contact.geom1, contact.geom2]]-m.body('head').id
                if min(a, b) >= 0 and abs(a-b) > 2 and contact.dist <= 0.:
                    statistics['self_contact_steps'] += 1
            if not np.isfinite(d.qpos).all():
                failure = 'nonfinite dynamics'
        if d.time+1e-9 >= next_sample:
            next_sample += .05
            pts = backbone(m, d)
            metrics = sim.measure(0.)
            rows.append(dict(time_s=float(d.time), stage='feeding', action=action,
                feeding_stage=core.stage, x_min_m=float(pts[:, 0].min()),
                center_x_m=float(pts[:, 0].mean()), min_height_m=float(pts[:, 2].min()),
                support_n=metrics['normal_force_n'], ground_force_n=metrics['ground_contact_force_n'],
                tracking_max_deg=float(np.max(abs(c.bus.targets_deg[c.bus.enabled]-np.degrees(d.qpos[sim.drive.qadr])[c.bus.enabled]), initial=0.)),
                position_joints=core.position))
            states.append(d.qpos.copy()); velocities.append(d.qvel.copy())
            targets.append(np.radians(c.bus.targets_deg)); masks.append(c.bus.enabled.copy())
            forces.append(d.actuator_force.copy()); points_record.append(pts.copy())
            if backend == 'curve':
                curves.append(dict(time_s=float(d.time), **core.last_curve.as_dict()))
        if t >= progress:
            print(f'{case}/{backend} t={t:.1f} {core.stage} p={core.position:.3f} action={action} fault={core.fault}', flush=True)
            progress += 10.
        if statistics['floor_contact_steps']:
            failure = failure or 'floor contact'
        if failure or terminal is not None and t-terminal >= 3.:
            break
    label = case+'_'+backend
    np.savez_compressed(output/(label+'_trace.npz'), time=[r['time_s'] for r in rows],
        qpos=states, qvel=velocities, targets=targets, masks=masks, actuator_force=forces,
        backbone_world=points_record)
    import gzip
    with gzip.open(output/(label+'_control.jsonl.gz'), 'wt', encoding='utf-8') as stream:
        for row in control_rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    summary = dict(case=case, backend=backend, scene='straight', joints=p.joints,
        parameters=asdict(p), start_x=.1, binary_enable=True, feedback_period_s=.2,
        physics_period_s=.001, control_period_s=.02, hardware_run=False,
        commands=commands, stage_events=stage_events, completed_actions=finished_actions,
        handovers=c.bus.handovers, statistics=statistics,
        final_time_s=float(d.time), final=core.snapshot(), failure=failure,
        completed=terminal is not None, controller_inputs=['angles_deg', 'feedback_time_s', 'control_time_s'],
        geometry=dict(pipe_geom_size=m.geom('pipe_main').size.tolist(),
            pipe_geom_pos=m.geom('pipe_main').pos.tolist(),
            centerline=m.numeric('pipe_centerline').data.reshape(-1, 4).tolist(),
            friction=m.geom('pipe_main').friction.tolist(), gravity=m.opt.gravity.tolist(),
            link_lengths_m=real.LINK_LENGTHS_M))
    (output/(label+'.json')).write_text(json.dumps(dict(summary=summary, samples=rows),
        ensure_ascii=False, indent=1), encoding='utf-8')
    if curves:
        (output/(label+'_curves.json')).write_text(json.dumps(curves), encoding='utf-8')
    print(label, summary['completed'], failure, float(d.time), flush=True)
    real.ContinuousFeeding = BASE
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [args.case] if args.case else CASES
    for case in cases:
        if any((args.output/(case+'_'+backend+'.json')).exists() for backend in ('legacy', 'curve')):
            raise FileExistsError('Keep prior physics evidence; choose a new --output directory')
    for case in cases:
        parameters = make_initial(case, args.output)
        for backend in ('legacy', 'curve'):
            run(case, backend, args.output, parameters)


if __name__ == '__main__':
    main()
