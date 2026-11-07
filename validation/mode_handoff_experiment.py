"""Exercise real hotkey routes with sampled angles and binary simulated motors.

World geometry in this file represents the operator's Ctrl+L confirmations and
evaluation. It is never passed to the real controller.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codes import Rolling_Arc_Real as real
from codes.Rolling_Arc_Debug_Mujoco import MujocoMotorBus
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import backbone
from validation.gm_arc_handoff_experiment import helix_alignment_ready, place_arc_for_adjustment
from validation.gm_locked_experiment import APPROACH_SETUP, recovery_metrics, right_u_ready
from validation.gm_replay_wrap_experiment import save
from validation.gm_shift_cycle_experiment import CycleTrial
from snake_control.tee_simulation import PIPE_HEIGHT
from validation.gm_wrap_first_experiment import pipe_contacts


CASES = {
    'basic5': dict(window=(7, 12)),
    'feed_j5': dict(window=(7, 12), feed=1),
    'feed_h5': dict(window=(7, 12), feed=-1),
    'buffer6': dict(tee=True, buffered=True),
    'five_then6': dict(tee=True, sequential=True),
    'phase6': dict(tee=True, phase_delta=.15),
    'head_short5': dict(window=(2, 7)),
    'tail_short5': dict(window=(11, 16)),
    'feed_h6': dict(tee=True, feed=-1),
}


class ExperimentController(real.RollingArcController):
    def refresh_compliance_feedback(self):
        super().refresh_compliance_feedback()
        self.feedback_motor_clock_s = self.bus.sim.data.time

    def motor_sample_clock(self):
        return self.bus.sim.data.time, self.feedback_motor_clock_s

    def show_status(self, include_feedback=False):
        pass


def run(case, output):
    config = CASES[case]
    setup = dict(APPROACH_SETUP, scene='up60' if config.get('tee') else 'straight')
    trial = CycleTrial(setup=setup)
    sim, model, data = trial.sim, trial.model, trial.data
    n = sim.p.joints
    real.NUM_JOINTS = n
    real.LINK_LENGTHS_M = [.0645]*(n+1)
    real.MIN_ANGLE_DEG, real.MAX_ANGLE_DEG = [-90.]*n, [90.]*n
    real.SERVO_IDS, real.ZERO_RAW = real.SERVO_IDS[:n], real.ZERO_RAW[:n]
    real.MODEL_TO_SERVO_SIGN, real.SERVO_DIRECTION = real.MODEL_TO_SERVO_SIGN[:n], real.SERVO_DIRECTION[:n]
    if config.get('tee'):
        source = Path('artifacts/gm_arc_shift_rolling/gravity_continuous')
        previous = json.loads((source/'locked_gait.json').read_text(encoding='utf-8'))
        trace = np.load(source/'locked_gait_trace.npz')
        index = next(i for i, r in enumerate(previous['samples']) if r['stage'] == 'front_only_coil')
        data.qpos[:], data.qvel[:] = trace['qpos'][index], trace['qvel'][index]
        window = (6, 11)
        mask = trace['masks'][index].copy() if config.get('buffered') else trace['stiffness_scale'][index] >= .999
        initial = trace['targets'][index].copy()
        phase = previous['samples'][index]['reference_roll_rad']
        mujoco.mj_forward(model, data)
    else:
        window, phase = config['window'], 0.
        parameters = replace(sim.p, first_joint=window[0], last_joint=window[1])
        sim.p = sim.drive.p = parameters
        trial.phase = phase
        place_arc_for_adjustment(trial, parameters, .1)
        initial, mask = parameters.targets(phase), parameters.enabled
        for _ in range(2500):
            sim.drive.apply_targets(data, initial, mask)
            mujoco.mj_step(model, data)
    real.FIRST_JOINT, real.LAST_JOINT = window
    controller = ExperimentController(MujocoMotorBus(sim))
    controller.enabled, controller.targets = mask.tolist(), np.degrees(initial).tolist()
    controller.first, controller.last = window
    controller.configured_window = controller.feeding_origin = window
    controller.phase = controller.command_phase = phase
    controller.mode = 'hold'
    controller.bus.prepare_enabled(controller.targets, controller.enabled)
    controller.touched_hardware = True
    if config.get('phase_delta'):
        target = real.SectionReference(*window, 60.).targets(phase+config['phase_delta'])
        for _ in range(1500):
            sim.drive.apply_targets(data, target, mask)
            mujoco.mj_step(model, data)
        controller.targets = np.degrees(target).tolist()
        controller.bus.send_targets(controller.targets, controller.enabled)
    commands, entries, confirmations = [], [], []
    support_since = None
    last_control = last_feedback = -1.
    first_tail_on = None
    front_confirmed_at = None
    front_ready_roll = None
    terminal_at = None
    failure = None

    def command(key):
        controller.command(real.decode_hotkey(ord(key), True))
        core = controller.helix_transition
        commands.append(dict(time_s=float(data.time), key='Ctrl+'+key,
                             stage=None if core is None else core.stage))
        if key in ('5', '6'):
            entries.append(core.status())

    def confirm(stage):
        nonlocal front_confirmed_at
        command('N')
        command('L')
        command('G')
        confirmations.append(dict(stage=stage, time_s=float(data.time)))
        if stage == 'front':
            front_confirmed_at = float(data.time)

    if config.get('feed'):
        command('4')
        command('J' if config['feed'] > 0 else 'H')
    else:
        command('5' if config.get('sequential') or not config.get('tee') else '6')
        command('G')

    def actual_front_ready(core):
        nonlocal support_since
        last = core.front.window[1]
        points = backbone(model, data)
        yz = points[:last+1, 1:]-[0., PIPE_HEIGHT]
        turn = np.ptp(np.unwrap(np.arctan2(yz[:, 1], yz[:, 0])))
        main, _, _ = pipe_contacts(sim)
        attached = (turn >= .65*2*np.pi and max(np.linalg.norm(yz, axis=1)) <= .148 and
                    any(j <= 1 for j in main) and any(3 <= j <= last for j in main))
        support_since = (data.time if support_since is None else support_since) if attached else None
        return support_since is not None and data.time-support_since >= .4

    def control(_):
        nonlocal last_control, last_feedback, first_tail_on, terminal_at, failure, front_ready_roll
        if data.time-last_control < .02-1e-8:
            return np.radians(controller.bus.targets_deg)
        if data.time-last_feedback >= .2-1e-8:
            controller.refresh_compliance_feedback()
            last_feedback = data.time
        core = controller.helix_transition
        if controller.feeding_selected:
            if controller.feeding.completed:
                command('6' if config.get('tee') else '5')
                command('G')
                core = controller.helix_transition
            elif controller.feeding.fault:
                failure = controller.feeding.fault
        if core is not None:
            front_attached = (actual_front_ready(core) if core.stage == 'front' and core.front.stage == 'lead' else False)
            if core.stage == 'front' and core.front_extended:
                front_ready_roll = core.front.roll if front_ready_roll is None else front_ready_roll
                if core.front.roll-front_ready_roll > 2.*np.pi:
                    failure = 'front support not established within one complete roll'
            if core.stage == 'front' and core.front_extended and front_attached:
                confirm('front')
            elif core.stage == 'align' and helix_alignment_ready(trial, (90., 130.), 1.035, .10):
                confirm('align')
            elif (core.stage == 'right_u_hold' or core.stage == 'unwind' and core.elapsed >= 3.) and right_u_ready(trial, np.arange(n)<8):
                confirm('right_u_hold')
            elif core.stage == 'drop_hold':
                tail = slice(model.body('head').id+9, model.body('head').id+n+1)
                com = np.average(data.xipos[tail], axis=0, weights=model.body_mass[tail])
                if abs(com[1]) <= .015 and com[2] <= PIPE_HEIGHT-.10:
                    confirm('drop_hold')
            elif core.stage == 'complete':
                if config.get('sequential'):
                    command('6')
                    command('G')
                    core = controller.helix_transition
                elif terminal_at is None:
                    terminal_at = float(data.time)
            elif core.stage == 'blocked':
                failure = core.status()
            elif controller.mode == 'hold' and core.stage == 'front' and core.front_extended:
                command('G')
            elif controller.mode == 'hold' and core.stage == 'align':
                command('G')
        controller.tick(.02)
        core = controller.helix_transition
        trial.phase = controller.command_phase
        if (core is not None and core.front is not None and core.stage == 'front' and core.front.stage == 'lead'
                and any(controller.enabled[core.front.window[1]:])):
            failure = 'tail enabled before the front confirmation'
        if (core is not None and core.front is not None and core.stage == 'first_shift' and
                any(controller.enabled[core.front.window[1]:]) and first_tail_on is None):
            first_tail_on = float(data.time)
        last_control = data.time
        return np.radians(controller.bus.targets_deg)

    def stopped(_):
        return (failure is not None or terminal_at is not None and data.time-terminal_at >= 3. or
                sum(r['stage'] == 'roll' for r in trial.samples[-212:]) >= 211)

    def current_stage():
        return controller.helix_transition.stage if controller.helix_transition is not None else 'feeding'

    try:
        trial.phase_run(1, current_stage, 480. if config.get('tee') else 350., control,
                        stiffness=lambda: np.asarray(controller.enabled, float),
                        stop_when=stopped)
    except (ValueError, RuntimeError) as exc:
        failure = str(exc)
    rolling = [r for r in trial.samples if r['stage'] == 'roll']
    held = [r for r in trial.samples if r['stage'] == 'complete']
    recovery = recovery_metrics(rolling)
    helix = (len(held) >= 50 and all(r['tail_max_radius_m'] < .145 and r['tail_span_deg'] >= 75. and
             bool(r['distal_contacts']) and any(j <= n//2 for j in r['main_links']) for r in held))
    ordered = first_tail_on is not None and front_confirmed_at is not None and first_tail_on >= front_confirmed_at
    passed = bool(failure is None and not trial.stopped and ordered and (recovery['passed'] if config.get('tee') else helix))
    save(trial, output, 'locked_gait', None, case=case, passed=passed, failure=failure, config=config,
         binary_enable=True, feedback_period_s=.2, commands=commands, operator_events=confirmations,
         entries=entries, recovery=recovery, helix_held=helix, first_tail_on_s=first_tail_on,
         front_confirmed_s=front_confirmed_at, front_before_tail=ordered,
         final_stage=current_stage(), hardware_run=False,
         controller_status=None if controller.helix_transition is None else controller.helix_transition.status())
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(0 if run(args.case, args.output) else 1)
