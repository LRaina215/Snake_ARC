"""Replay recorded front poses through the real hotkey dispatcher, without hardware."""
import contextlib
import io
import json
from pathlib import Path
import queue
import unittest
from unittest.mock import patch

import numpy as np

from validation.test_real_controller import real, RecordingSerial
from snake_control import gravity_handoff as library


DATA = json.loads((Path(__file__).parent/'fixtures/front_confirmation_20261107.json').read_text(encoding='utf-8'))


def recorded_core(module, stage='front_hold'):
    row = DATA['states'][stage]
    options = module.GravityMomentOptions(r_h1=.094, r_h2=.098)
    gait = module.GravityMomentGait(18, options, .0645)
    ideal = np.degrees(gait.targets(0., .094)[0])
    core = module.GravityCrossing(DATA['link_lengths_m'], [True]*18, ideal,
                                  options, from_helix=True, helix_only=True)
    core.stage, core.started = stage, True
    core.front_count = row['core']['front_count']
    core.phase = row['core']['phase_rad']
    core._targets = np.radians(row['target_controller_deg'])
    core._enabled = np.asarray(row['enabled'], bool)
    core.measured = np.radians([row['measured_controller_deg'][f'J{i}'] for i in range(1, 19)])
    return core


def recorded_controller(stage='front_hold'):
    c = real.RollingArcController(real.FeetechBus(RecordingSerial()))
    c.helix_transition = core = recorded_core(real, stage)
    c.enabled, c.targets = core.enabled.tolist(), np.degrees(core.targets).tolist()
    c.first, c.last = 1, core.front_count
    c.mode, c.freeze_targets = 'hold', True
    c.states = {i: {'angle_deg': angle} for i, angle in enumerate(np.degrees(core.measured))}
    return c


class FrontConfirmationTests(unittest.TestCase):
    def test_recorded_pose_accepts_l_during_step_and_repeated_l_does_not_reset(self):
        for stage in ('front_finish', 'front_seat', 'front_hold'):
            for module in (library, real):
                with self.subTest(stage=stage, module=module.__name__):
                    core = recorded_core(module, 'front_finish' if stage == 'front_seat' else stage)
                    core.stage = stage
                    core.step_once()  # L must not depend on whether a one-second step just ended.
                    before = core.targets.copy()
                    self.assertTrue(core.confirm(1., np.degrees(core.measured)))
                    self.assertEqual(core.stage, 'first_shift')
                    self.assertFalse(core.running)
                    self.assertEqual(core.step_left, 0.)
                    np.testing.assert_array_equal(core.targets, before)
                    curve = core.curve
                    core.offset = .012
                    self.assertFalse(core.confirm(1.2, np.degrees(core.measured)))
                    self.assertIs(core.curve, curve)
                    self.assertEqual(core.offset, .012)

    def test_j_cannot_keep_rotating_the_front_while_confirmation_is_missing(self):
        c = recorded_controller()
        before, phase = c.helix_transition.targets, c.helix_transition.phase
        for key in ('J', 'G'):
            with self.assertRaisesRegex(ValueError, 'Ctrl\\+L'):
                c.command(real.decode_hotkey(ord(key), True))
            c.tick(.02)
        np.testing.assert_array_equal(c.helix_transition.targets, before)
        self.assertEqual(c.helix_transition.phase, phase)
        self.assertEqual(c.mode, 'hold')

    def test_l_then_j_enables_one_tail_joint_and_holds_the_recorded_front(self):
        c = recorded_controller('front_finish')
        core = c.helix_transition
        # Confirm a supported measured pose which differs from the fixed nominal helix.
        core._targets[0] += np.radians(9.)
        core.measured[0] = core._targets[0]
        c.targets = np.degrees(core.targets).tolist()
        c.states[0]['angle_deg'] = c.targets[0]
        front = core.targets[:core.front_count].copy()
        c.mode = 'step'
        core.step_once()
        with contextlib.redirect_stdout(io.StringIO()), patch.object(c, 'refresh_compliance_feedback'), \
                patch.object(c, 'motor_sample_clock', return_value=(1., 1.)):
            c.command(real.decode_hotkey(ord('L'), True))
        self.assertEqual(c.mode, 'hold')
        self.assertEqual(core.stage, 'first_shift')
        self.assertEqual(sum(c.enabled), 9)  # Confirmation alone never moves or enables the tail.
        c.command(real.decode_hotkey(ord('J'), True))
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(1, 26):
                q = np.degrees(core.targets).tolist()  # An ideal follower only tests command handoff.
                c.states = {j: {'angle_deg': angle} for j, angle in enumerate(q)}
                with patch.object(c, 'motor_sample_clock', return_value=(1.+i*.02, 1.+i*.02)):
                    c.tick(.02)
                np.testing.assert_array_equal(core.targets[:core.front_count], front)
        self.assertTrue(c.enabled[9])
        self.assertGreater(core.offset, 0.)
        self.assertGreaterEqual(core.accepted, 10)  # Never demand that the locked front fit another helix.

    def test_dispatch_records_the_actual_command_stage_and_rejection_reason(self):
        c = recorded_controller()
        commands = queue.Queue()
        for key in ('J', 'L', 'L', 'J'):
            commands.put(real.decode_hotkey(ord(key), True))
        with contextlib.redirect_stdout(io.StringIO()), patch.object(c, 'refresh_compliance_feedback'), \
                patch.object(c, 'motor_sample_clock', return_value=(1., 1.)):
            real.process_control_commands(c, commands, 1.)
        events = c.snapshot()['control_commands']
        self.assertEqual([e['command'] for e in events], ['step', 'feed_tick', 'feed_tick', 'step'])
        self.assertEqual(events[0]['result'], 'rejected')
        self.assertIn('Ctrl+L', events[0]['reason'])
        self.assertEqual(events[1]['stage_after'], 'first_shift')
        self.assertEqual(events[-1]['mode_after'], 'step')

    def test_other_confirmations_keep_the_gravity_sequence_and_name_unavailable_stage(self):
        for stage, after in (('align', 'unwind'), ('right_u_hold', 'drop'),
                             ('drop_hold', 'return_seat')):
            with self.subTest(stage=stage):
                core = recorded_core(real)
                core.stage = stage
                core.step_once()
                self.assertTrue(core.confirm(1., np.degrees(core.measured)))
                self.assertEqual(core.stage, after)
                self.assertEqual(core.step_left, 0.)
        core = recorded_core(real)
        core.stage = 'arc_settle'
        with self.assertRaisesRegex(ValueError, 'arc_settle'):
            core.confirm(1., np.degrees(core.measured))
        self.assertEqual(core.stage, 'arc_settle')

    def test_manual_mujoco_entry_uses_the_same_l_then_j_handoff(self):
        from codes import Rolling_Arc_Debug_Mujoco as debug
        with patch.object(debug.real, 'open_robot_serial', side_effect=AssertionError('no hardware')):
            session = debug.DebugSession()
            try:
                c = session.controller
                c.helix_transition = core = recorded_core(debug.real)
                c.enabled, c.targets = core.enabled.tolist(), np.degrees(core.targets).tolist()
                c.first, c.last, c.mode = 1, core.front_count, 'hold'
                measured = np.degrees(core.measured).tolist()
                with contextlib.redirect_stdout(io.StringIO()), \
                        patch.object(c, 'refresh_compliance_feedback'), \
                        patch.object(c, 'motor_sample_clock', return_value=(1., 1.)), \
                        patch.object(c, 'measured_angles', return_value=measured):
                    c.command(debug.real.decode_hotkey(ord('L'), True))
                    c.command(debug.real.decode_hotkey(ord('J'), True))
                    c.tick(.02)
                self.assertEqual(core.stage, 'first_shift')
                self.assertEqual(sum(c.enabled), 10)
                self.assertGreater(core.offset, 0.)
                np.testing.assert_array_equal(core.targets[:9], np.radians(DATA['states']['front_hold']['target_controller_deg'][:9]))
            finally:
                session.close()


if __name__ == '__main__':
    unittest.main()
