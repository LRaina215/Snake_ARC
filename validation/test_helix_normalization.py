"""Reproduce a real 'complete' pose with a frozen, non-helical front; no hardware."""
import contextlib
import io
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from validation.test_front_confirmation import recorded_core, real
from validation.test_real_controller import RecordingSerial
from snake_control import gravity_handoff as library


DATA = json.loads((Path(__file__).parent/'fixtures/incomplete_helix_20261107.json').read_text(encoding='utf-8'))


def logged_core(module):
    core = recorded_core(module)
    core.stage = 'complete'
    core._enabled = np.asarray(DATA['enabled'])
    core._targets = np.radians(DATA['targets_deg'])
    core.measured = np.radians(DATA['measured_deg'])
    core.phase = DATA['phase_rad']
    return core


def mathematical_helix(phase):
    # Independent constant-curvature/torsion integral for the configured helix.
    radius, rise, link = .094, .13/(2*math.pi), .0645
    curvature = radius/(radius*radius+rise*rise)
    torsion = -rise/(radius*radius+rise*rise)
    coordinate = (np.arange(18)+1)*link
    amplitude = 2*curvature*link*np.sinc(torsion*link/math.pi)
    psi = phase+torsion*coordinate
    result = amplitude*np.cos(psi)
    result[1::2] = amplitude*np.sin(psi[1::2])
    return result


class HelixNormalizationTests(unittest.TestCase):
    def test_recorded_incomplete_pose_converges_to_one_global_helix(self):
        for module in (library, real):
            with self.subTest(module=module.__name__):
                core = logged_core(module)
                self.assertGreater(np.max(abs(core.targets[:9]-mathematical_helix(core.phase)[:9])), math.radians(40.))
                before, enabled = core.targets, core.enabled
                core.restore_helix(0., DATA['measured_deg'])
                self.assertEqual(core.stage, 'helix_normalize')
                np.testing.assert_array_equal(core.targets, before)
                np.testing.assert_array_equal(core.enabled, enabled)
                core.running = True
                for tick in range(1, 501):
                    old = core.targets.copy()
                    core.advance(.02, tick*.02, tick*.02, np.degrees(old))
                    self.assertLessEqual(float(np.max(abs(core.targets-old))), math.radians(.36)+1e-10)
                    if core.stage == 'complete':
                        break
                self.assertEqual(core.stage, 'complete')
                self.assertFalse(core.running)
                np.testing.assert_allclose(core.targets, mathematical_helix(core.phase), atol=1e-12, rtol=0.)

    def test_tail_completion_enters_normalization_instead_of_accepting_frozen_front(self):
        core = logged_core(real)
        core.stage, core.accepted = 'first_shift', 18
        core.locked_front = core.targets[:9].copy()
        core.curve = real.MeasuredGravityShift(core.gait, .094, core.phase, core.measured, 9, upstream_m=10*.0645)
        core.offset = core.curve.shift_limit_m
        core.measured[9:] = mathematical_helix(core.phase)[9:]
        core.elapsed, core.tracking_since, core.running = 1., 0., True
        core._advance_tail_intake(.02, True)
        self.assertEqual(core.stage, 'helix_normalize')
        self.assertTrue(core.running)  # One G continues through normalization.

    def test_ctrl5_restore_is_a_hold_then_j_or_g_and_partial_tail_is_not_enabled(self):
        c = real.RollingArcController(real.FeetechBus(RecordingSerial()))
        c.helix_transition = core = logged_core(real)
        c.enabled, c.targets = core.enabled.tolist(), np.degrees(core.targets).tolist()
        c.mode = 'hold'
        c.states = {i:{'angle_deg':v} for i,v in enumerate(DATA['measured_deg'])}
        with contextlib.redirect_stdout(io.StringIO()), patch.object(c, 'refresh_compliance_feedback'), \
                patch.object(c, 'motor_sample_clock', return_value=(1.,1.)):
            c.command(real.decode_hotkey(ord('5'), True))
            self.assertEqual(core.stage, 'helix_normalize')
            self.assertEqual(c.mode, 'hold')
            c.command(real.decode_hotkey(ord('G'), True))
            self.assertEqual(c.mode, 'run')
            c.command('pause')
            core._enabled[-1] = c.enabled[-1] = False
            core.helix_only = False
            core.stage = 'drop_hold'
            with self.assertRaisesRegex(ValueError, '失能尾段'):
                c.command('gait_helix')
            self.assertFalse(core.enabled[-1])

    def test_restart_can_preload_an_existing_global_helix_pose_before_enabling(self):
        port = RecordingSerial()
        c = real.RollingArcController(real.FeetechBus(port))
        c.states = {i:{'angle_deg':v} for i,v in enumerate(DATA['measured_deg'])}
        with contextlib.redirect_stdout(io.StringIO()), patch.object(c, 'refresh_compliance_feedback'), \
                patch.object(c, 'motor_sample_clock', return_value=(1.,1.)):
            c.command('gait_helix')
        self.assertEqual(c.mode, 'hold')
        self.assertEqual(c.helix_transition.stage, 'helix_normalize')
        self.assertTrue(all(c.enabled))
        writes = [packet for packet in port.packets if packet[4] == 0x83]
        self.assertEqual([packet[5] for packet in writes], [0x2A,0x30,0x29,0x28])
        np.testing.assert_allclose(c.targets, DATA['measured_deg'])

    def test_manual_mujoco_entry_uses_the_same_ctrl5_normalization(self):
        from codes import Rolling_Arc_Debug_Mujoco as debug
        with patch.object(debug.real, 'open_robot_serial', side_effect=AssertionError('no hardware')):
            session = debug.DebugSession()
            try:
                c = session.controller
                c.helix_transition = core = logged_core(debug.real)
                c.enabled, c.targets = core.enabled.tolist(), np.degrees(core.targets).tolist()
                c.first, c.last, c.mode = 1, 18, 'hold'
                c.states = {i:{'angle_deg':v} for i,v in enumerate(DATA['measured_deg'])}
                before = core.targets.copy()
                with contextlib.redirect_stdout(io.StringIO()), patch.object(c, 'refresh_compliance_feedback'), \
                        patch.object(c, 'motor_sample_clock', return_value=(1.,1.)):
                    c.command(debug.real.decode_hotkey(ord('5'), True))
                    self.assertEqual(core.stage, 'helix_normalize')
                    np.testing.assert_array_equal(core.targets, before)
                    c.command(debug.real.decode_hotkey(ord('J'), True))
                    c.tick(.02)
                self.assertTrue(all(c.enabled))
                self.assertLessEqual(float(np.max(abs(core.targets-before))), math.radians(.36)+1e-10)
            finally:
                session.close()


if __name__ == '__main__':
    unittest.main()
