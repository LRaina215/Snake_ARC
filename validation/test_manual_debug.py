"""Two focused manual-debug checks. Memory/physics backends, never real serial."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import io
import json
import unittest
from collections import namedtuple
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from codes import Rolling_Arc_Real as real
from codes.Rolling_Arc_Debug_Mujoco import DebugSession

ROOT = Path(__file__).resolve().parents[1]


class ManualDebugTests(unittest.TestCase):
    def test_backend_reads_actual_angles_and_preview_does_not_actuate(self):
        with patch.object(real, 'open_robot_serial', side_effect=AssertionError('no physical IO')):
            s = DebugSession()
            initial = s.sim.data.qpos.copy()
            for _ in range(3):
                s.advance()
            np.testing.assert_array_equal(s.sim.data.qpos, initial)
            self.assertEqual(s.sim.data.time, 0.)
            self.assertFalse(s.physics_started)
            self.assertFalse(s.bus.enabled.any())
            angles = s.bus.read_angles(range(20))
            changed = list(angles.values())
            changed[7] += 10.
            s.bus.send_targets(changed, [True]*20)
            self.assertEqual(s.bus.read_angles(range(20)), angles)  # 写目标不是改实际qpos。
            states = s.bus.read_states()
            self.assertEqual(set(states), set(range(20)))
            self.assertTrue(all(v['load_raw'] is None and v['current_raw'] is None for v in states.values()))
            self.assertFalse(np.any(s.sim.data.ctrl))
            s.close()

    def test_real_ctrl_hotkeys_drive_one_complete_manual_physics_workflow(self):
        class FakeListener:
            def __init__(self, on_press, on_release):
                self.press, self.release = on_press, on_release
            def start(self):
                pass
        keyboard = SimpleNamespace(Key=SimpleNamespace(ctrl='ctrl', ctrl_l='left_ctrl', ctrl_r='right_ctrl'),
                                   Listener=FakeListener)
        key_type = namedtuple('Key', ['vk'])
        trace = []
        with patch.object(real, 'open_robot_serial', side_effect=AssertionError('no physical IO')), redirect_stdout(io.StringIO()):
            s = DebugSession()
            listener = real.global_key_listener(s.commands, keyboard)
            def frames(count):
                for _ in range(count):
                    s.advance()
                    row = s.snapshot()
                    trace.append(row)
                    self.assertTrue(np.isfinite(s.sim.data.qpos).all())
                    self.assertTrue(np.isfinite(s.sim.data.qvel).all())
                    self.assertLessEqual(sum(row['enabled']), 6)
                    self.assertEqual(max(abs(s.sim.data.actuator_force[~s.bus.enabled]), default=0.), 0.)
            def tap(vk, control=True, repeat=False):
                key = key_type(vk)
                if control:
                    listener.press('left_ctrl')
                listener.press(key)
                if repeat:
                    listener.press(key)
                listener.release(key)
                if control:
                    listener.release('left_ctrl')
                frames(1)
            try:
                for vk in (ord('U'), ord('J'), ord('E'), 32, 116, 117):
                    tap(vk, control=False)
                self.assertEqual(s.controller.mode, 'released')
                self.assertEqual(s.sim.data.time, 0.)
                tap(ord('U'))
                frames(105)
                self.assertFalse(s.controller.forming)
                self.assertEqual(s.controller.settle_remaining, 0.)
                self.assertEqual(np.flatnonzero(s.bus.enabled).tolist(), list(range(7, 13)))
                phase = s.controller.phase
                tap(ord('J'), repeat=True)
                frames(10)
                self.assertAlmostEqual(s.controller.phase, phase+real.PHASE_STEP_RAD)
                tap(ord('H'))
                frames(10)
                self.assertAlmostEqual(s.controller.phase, phase)
                tap(ord('E'))
                self.assertIsNotNone(s.controller.compliance)
                frames(5)
                tap(ord('G'))
                frames(20)
                tap(32)  # Ctrl+Space暂停相位，物理与形状反馈继续。
                phase, sim_time = s.controller.phase, s.sim.data.time
                frames(15)
                self.assertAlmostEqual(s.controller.phase, phase)
                self.assertGreater(s.sim.data.time, sim_time)
                tap(ord('N'))
                targets = s.controller.targets.copy()
                frames(10)
                np.testing.assert_array_equal(s.controller.targets, targets)
                tap(ord('E'))
                self.assertIsNone(s.controller.compliance)
                frames(45)
                tap(ord('T'))
                tap(ord('G'))
                frames(12)
                self.assertLess(s.controller.phase, phase)
                tap(ord('X'))
                self.assertFalse(s.bus.enabled.any())
                self.assertFalse(np.any(s.sim.data.ctrl))
                released_time = s.sim.data.time
                frames(3)
                self.assertGreater(s.sim.data.time, released_time)
                tap(ord('Q'))
                self.assertTrue(s.controller.quit)
                self.assertFalse(any(row['compliance'] and row['compliance']['block_reason'] for row in trace))
            finally:
                s.close()
                trace.append(s.snapshot())
                output = ROOT/'validation/results/manual_debug'
                output.mkdir(parents=True, exist_ok=True)
                (output/'interaction_trace.jsonl').write_text(
                    ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in trace), encoding='utf-8')


if __name__ == '__main__':
    unittest.main()
