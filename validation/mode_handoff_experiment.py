"""Exercise real hotkey routes with sampled angles and binary simulated motors.

World geometry in this file represents the operator's Ctrl+L confirmations and
evaluation. It is never passed to the real controller.
"""
import argparse
import contextlib
from dataclasses import asdict, replace
import json
import io
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
    'feed_j6': dict(tee=True, feed=1),
}

MIN_FRONT_ATTACHED_TURNS = .8
MAX_FRONT_ATTACHED_RADIUS_M = .120
FRONT_SUPPORT_SETTLE_S = .6
FRONT_MIN_CONTACT_SPAN = 0
FRONT_UNSUPPORTED_TIMEOUT_S = 5.
INTAKE_STAGES = ('return_seat', 'return_wind', 'return_finish')
ASSERT_FRONT_FROZEN = False


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
        fixture = json.loads((Path(__file__).parent/'fixtures/arc_tee_initial18.json').read_text(encoding='utf-8'))
        data.qpos[:], data.qvel[:] = fixture['qpos'], fixture['qvel']
        window = tuple(fixture['window'])
        mask = np.asarray(fixture['buffered_mask'] if config.get('buffered') else fixture['arc_mask'], bool)
        initial, phase = np.asarray(fixture['targets']), fixture['phase']
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
    terminal_at = None
    failure = None
    self_contact_events = []
    locked_front = None
    held_front_count = None
    front_target_change = 0.
    front_hold_started = None
    next_progress = float(data.time)+30.
    clear_since = None

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
        if stage == 'front_hold':
            front_confirmed_at = float(data.time)

    if config.get('feed'):
        command('4')
        command('J' if config['feed'] > 0 else 'H')
    else:
        command('5' if config.get('sequential') or not config.get('tee') else '6')
        command('G')

    def actual_front_ready(core):
        nonlocal support_since
        last = core.front_count
        points = backbone(model, data)
        yz = points[:last+2, 1:]-[0., PIPE_HEIGHT]
        turn = np.ptp(np.unwrap(np.arctan2(yz[:, 1], yz[:, 0])))
        main, _, _ = pipe_contacts(sim)
        supported = [j for j in main if j <= last]
        attached = (turn >= MIN_FRONT_ATTACHED_TURNS*2*np.pi and max(np.linalg.norm(yz, axis=1)) <= MAX_FRONT_ATTACHED_RADIUS_M and
                    len(supported) >= 2 and max(supported)-min(supported) >= FRONT_MIN_CONTACT_SPAN)
        support_since = (data.time if support_since is None else support_since) if attached else None
        return support_since is not None and data.time-support_since >= FRONT_SUPPORT_SETTLE_S

    def control(_):
        nonlocal last_control, last_feedback, first_tail_on, terminal_at, failure
        nonlocal locked_front, held_front_count, front_target_change, front_hold_started
        nonlocal support_since
        nonlocal next_progress
        for contact in data.contact:
            a, b = model.geom_bodyid[[contact.geom1, contact.geom2]]-model.body('head').id
            if min(a, b) >= 0 and abs(a-b) > 2 and contact.dist <= 0.:
                if not self_contact_events:
                    self_contact_events.append(dict(time_s=float(data.time), links=[int(a), int(b)],
                        stage=current_stage(), penetration_m=float(-contact.dist), qpos=data.qpos.tolist(),
                        targets=controller.targets.copy(), enabled=controller.enabled.copy()))
                failure = 'nonadjacent robot links touched'
        if data.time-last_control < .02-1e-8:
            return np.radians(controller.bus.targets_deg)
        if data.time-last_feedback >= .2-1e-8:
            controller.refresh_compliance_feedback()
            last_feedback = data.time
        core = controller.helix_transition
        if data.time >= next_progress:
            print(json.dumps(dict(time_s=round(float(data.time), 2), stage=current_stage(),
                accepted=None if core is None else core.accepted,
                operator_mode=controller.mode)), flush=True)
            next_progress += 30.
        if controller.feeding_selected:
            if controller.feeding.completed:
                command('6' if config.get('tee') else '5')
                command('G')
                core = controller.helix_transition
            elif controller.feeding.fault:
                failure = controller.feeding.fault
        if core is not None:
            front_attached = actual_front_ready(core) if core.stage == 'front_hold' and controller.mode == 'hold' else False
            if core.stage == 'front_hold' and controller.mode != 'hold':
                support_since = None
            if core.stage == 'front_hold':
                front_hold_started = data.time if front_hold_started is None else front_hold_started
            if core.stage == 'front_hold' and front_attached and controller.mode == 'hold':
                confirm('front_hold')
                held_front_count = core.front_count
                locked_front = np.radians(controller.targets[:held_front_count])
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
            elif controller.mode == 'hold' and core.stage == 'front_hold' and data.time-front_hold_started >= FRONT_UNSUPPORTED_TIMEOUT_S:
                failure = 'front hold did not establish the required actual support'
            elif controller.mode == 'hold' and core.stage == 'align':
                command('G')
        previous_mode = controller.mode
        controller.tick(.02)
        core = controller.helix_transition
        if core is not None and core.stage == 'front_hold' and controller.mode == 'hold' and previous_mode != 'hold':
            front_hold_started = data.time
        trial.phase = controller.command_phase
        if (core is not None and core.front is not None and core.stage == 'front' and core.front.stage == 'lead'
                and any(controller.enabled[core.front.window[1]:])):
            failure = 'tail enabled before the front confirmation'
        if (core is not None and core.front is not None and core.stage in INTAKE_STAGES and
                any(controller.enabled[core.front_count:]) and first_tail_on is None):
            first_tail_on = float(data.time)
        if core is not None and core.stage in INTAKE_STAGES and locked_front is not None:
            front_target_change = max(front_target_change,
                float(np.max(abs(np.radians(controller.targets[:held_front_count])-locked_front))))
        last_control = data.time
        return np.radians(controller.bus.targets_deg)

    def stopped(_):
        nonlocal clear_since
        row = trial.samples[-1] if trial.samples else None
        clear = bool(row and row['stage'] == 'roll' and row['x_min_m'] > .409 and not row['branch_links'])
        clear_since = (data.time if clear_since is None else clear_since) if clear else None
        return (failure is not None or terminal_at is not None and data.time-terminal_at >= 3. or
                clear_since is not None and data.time-clear_since >= 10.5)

    def current_stage():
        return controller.helix_transition.stage if controller.helix_transition is not None else 'feeding'

    try:
        trial.phase_run(1, current_stage, 600. if config.get('tee') else 350., control,
                        stiffness=lambda: np.asarray(controller.enabled, float),
                        stop_when=stopped)
    except (ValueError, RuntimeError) as exc:
        failure = str(exc)
    rolling = [r for r in trial.samples if r['stage'] == 'roll']
    if failure is None and trial.stopped:
        failure = 'ground contact exceeded 1 N'
    held = [r for r in trial.samples if r['stage'] == 'complete']
    recovered = [row for row in rolling if clear_since is not None and row['time_s'] >= clear_since]
    recovery = recovery_metrics(recovered)
    helix = (len(held) >= 50 and all(r['tail_max_radius_m'] < .145 and r['tail_span_deg'] >= 75. and
             bool(r['distal_contacts']) and any(j <= n//2 for j in r['main_links']) for r in held))
    ordered = first_tail_on is not None and front_confirmed_at is not None and first_tail_on >= front_confirmed_at
    passed = bool(failure is None and not trial.stopped and ordered and (recovery['passed'] if config.get('tee') else helix))
    with contextlib.redirect_stdout(io.StringIO()):
        save(trial, output, 'locked_gait', None, case=case, passed=passed, failure=failure, config=config,
         binary_enable=True, feedback_period_s=.2, commands=commands, operator_events=confirmations,
         entries=entries, recovery=recovery, helix_held=helix, first_tail_on_s=first_tail_on,
         front_confirmed_s=front_confirmed_at, front_before_tail=ordered,
         final_stage=current_stage(), hardware_run=False,
         controller_status=None if controller.helix_transition is None else controller.helix_transition.status(),
         self_contact_events=self_contact_events, front_target_change_rad=front_target_change,
         front_freeze_count=held_front_count,
         clear_since_s=clear_since, total_rolling_s=0. if not rolling else rolling[-1]['time_s']-rolling[0]['time_s'],
         controller_options=None if controller.helix_transition is None else asdict(controller.helix_transition.options),
         gait_options=None if controller.helix_transition is None else asdict(controller.helix_transition.gait.o),
         parameters=asdict(sim.p), start_x=.1, assert_front_frozen=ASSERT_FRONT_FROZEN,
         front_support_criteria=dict(min_turns=MIN_FRONT_ATTACHED_TURNS, max_radius_m=MAX_FRONT_ATTACHED_RADIUS_M,
                                     settle_s=FRONT_SUPPORT_SETTLE_S, minimum_contact_index_span=FRONT_MIN_CONTACT_SPAN))
    print(json.dumps(dict(output=str(output), passed=passed, failure=failure, final_stage=current_stage(),
                          dynamics_s=float(data.time), helix_held=helix, front_before_tail=ordered)), flush=True)
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(0 if run(args.case, args.output) else 1)
