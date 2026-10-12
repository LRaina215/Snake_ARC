"""Three targeted checks for simulated modes and force keys; no physical serial."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import io
import unittest
from collections import namedtuple
from contextlib import redirect_stdout
from types import SimpleNamespace
import numpy as np
from codes import Rolling_Arc_Real as real
from codes.Rolling_Arc_Debug_Mujoco import DebugSession, decode_sim_hotkey, decode_sim_release


def prepared():
    s = DebugSession()
    s.commands.put('apply')
    for _ in range(110):
        s.advance()
    return s


class DebugModeTests(unittest.TestCase):
    def test_three_modes_preserve_pose_phase_targets_and_enabled_mask_at_switch(self):
        with redirect_stdout(io.StringIO()):
            s = prepared()
            try:
                for command, expected in (
                    ('gait_fixed', 'fixed_section_compliance'),
                    ('gait_sliding', 'sliding_section_compliance'),
                    ('gait_fixed', 'fixed_section_compliance'),
                    ('gait_basic', 'basic_arc'),
                ):
                    qpos, targets = s.sim.data.qpos.copy(), s.controller.targets.copy()
                    phase, mask = s.controller.phase, s.controller.enabled.copy()
                    s.controller.command(command)
                    np.testing.assert_array_equal(s.sim.data.qpos, qpos)
                    np.testing.assert_array_equal(s.controller.targets, targets)
                    self.assertEqual(s.controller.phase, phase)
                    self.assertEqual(s.controller.enabled, mask)
                    self.assertEqual(s.controller.snapshot()['gait'], expected)
                self.assertEqual(decode_sim_hotkey(49, True), 'gait_basic')
                self.assertEqual(decode_sim_hotkey(50, True), 'gait_fixed')
                self.assertEqual(decode_sim_hotkey(51, True), 'gait_sliding')
                self.assertIsNone(decode_sim_hotkey(51, False))
            finally:
                s.close()

    def test_hold_force_moves_actual_body_without_changing_window_and_keyup_stops(self):
        class Listener:
            def __init__(self, on_press, on_release):
                self.press, self.release = on_press, on_release
            def start(self):
                pass
        keyboard = SimpleNamespace(Key=SimpleNamespace(ctrl='ctrl', ctrl_l='left', ctrl_r='right'),
                                   Listener=Listener)
        key = namedtuple('Key', ['vk'])(65)
        with redirect_stdout(io.StringIO()):
            s = prepared()
            listener = real.global_key_listener(s.commands, keyboard, decode_sim_hotkey, decode_sim_release)
            try:
                before = s.snapshot()
                listener.press('left')
                listener.press(key)
                listener.press(key)  # 操作系统重复键不增加力。
                for _ in range(10):
                    s.advance()
                after = s.snapshot()
                self.assertAlmostEqual(np.linalg.norm(after['perturb_force_world_n']), s.push_force)
                self.assertLess(after['perturb_force_world_n'][1], 0.)
                self.assertLess(after['perturb_force_world_n'][2], 0.)
                self.assertEqual(after['window'], before['window'])
                self.assertGreater(np.linalg.norm(np.asarray(after['body_center_m'])-before['body_center_m']), 1e-7)
                listener.release('left')  # 字母尚未松开，释放Ctrl也立即结束扰动。
                s.advance()
                self.assertFalse(np.any(s.sim.data.xfrc_applied))
                listener.release(key)
                s.advance()
                listener.press('left')
                listener.press(key)
                opposite = namedtuple('Opposite', ['vk'])(68)
                listener.press(opposite)
                s.advance()
                self.assertFalse(np.any(s.sim.data.xfrc_applied))  # 两个相反方向不叠加向下力。
                listener.release('left')
                listener.release(key)
                listener.release(opposite)
                s.advance()
                self.assertEqual(s.controller.phase, before['phase_rad'])
            finally:
                s.close()
                self.assertFalse(np.any(s.sim.data.xfrc_applied))

    def test_reset_starts_explicit_new_episode_and_deferred_mode_waits_for_formation(self):
        with redirect_stdout(io.StringIO()):
            s = prepared()
            identity = s.controller
            try:
                s.commands.put('push_right')
                s.advance()
                previous = s.sim.data.qpos.copy()
                s.commands.put('reset_preview')
                s.commands.put('gait_sliding')
                s.advance()
                self.assertIs(s.controller, identity)
                self.assertEqual(s.episode_index, 1)
                self.assertEqual(s.sim.data.time, 0.)
                self.assertFalse(s.physics_started)
                self.assertFalse(np.any(s.sim.data.xfrc_applied))
                self.assertFalse(s.bus.enabled.any())
                self.assertEqual(s.controller.pending_gait, 'sliding')
                np.testing.assert_array_equal(s.events[-1]['previous_state']['actual_qpos'], previous)
                s.commands.put('apply')
                for _ in range(110):
                    s.advance()
                self.assertIsNotNone(s.controller.sliding)
                self.assertEqual(s.controller.snapshot()['gait'], 'sliding_section_compliance')
                self.assertTrue(s.controller.sliding.automatic)
            finally:
                s.close()


if __name__ == '__main__':
    unittest.main()
