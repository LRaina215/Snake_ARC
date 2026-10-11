"""Consequential real-control checks using an in-memory serial bus, never COM ports."""
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('real_controller', ROOT / 'codes/Rolling_Arc_Real.py')
real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real)
sys.path.insert(0, str(ROOT))
from snake_control.rolling_arc import ArcParameters


class RecordingSerial(real.DryRunSerial):
    def __init__(self):
        super().__init__()
        self.packets = []
        self.closed = False

    def write(self, packet):
        self.packets.append(packet)
        return super().write(packet)

    def close(self):
        self.closed = True


def prepared_controller():
    port = RecordingSerial()
    controller = real.RollingArcController(real.FeetechBus(port))
    controller.command('apply')
    for _ in range(320):  # 4 s formation + 2 s settling, using actual controller ticks
        controller.tick(real.CONTROL_PERIOD_S)
    return controller, port


class RealControllerTests(unittest.TestCase):
    def test_calibration_and_wave_agree_with_validated_simulation(self):
        profile = json.loads((ROOT / 'configs/hardware_20j_reference.json').read_text(encoding='utf-8'))
        joints = sorted(profile['Joints'], key=lambda j: j['Index'])
        self.assertEqual(real.ZERO_RAW, [j['Zero'] for j in joints])
        self.assertEqual(real.SERVO_IDS, [j['ServoId'] for j in joints])
        self.assertEqual(real.SERVO_DIRECTION, [j['Direction'] for j in joints])
        for phase in (0., .7, math.pi / 2, math.pi):
            expected = [math.degrees(x) for x in ArcParameters().targets(phase)]
            actual = real.joint_angles_deg(phase, 60., 8, 13)
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b)
        self.assertEqual(real.angle_to_raw(7, 60), real.ZERO_RAW[7] - round(60 * 4096 / 360))

    def test_preload_then_enable_and_no_target_writes_to_passive_joints(self):
        port = RecordingSerial()
        for regs in port.registers.values():
            regs[0x28] = 1  # Simulate previous code leaving every servo enabled.
        controller = real.RollingArcController(real.FeetechBus(port))
        controller.command('apply')
        self.assertEqual([port.registers[sid][0x28] for sid in real.SERVO_IDS],
                         [int(8 <= j <= 13) for j in range(1, 21)])
        writes = [p for p in port.packets if p[4] == 0x83]
        self.assertEqual([p[5] for p in writes], [0x2A, 0x30, 0x29, 0x28])
        first_target = writes[0]
        for offset in range(7, len(first_target) - 1, 7):
            sid = first_target[offset]
            self.assertEqual(int.from_bytes(first_target[offset + 1:offset + 3], 'little'), real.ZERO_RAW[sid - 1])
        for _ in range(4):
            controller.tick(.02)
        for p in port.packets:
            if p[4] == 0x83 and p[5] == 0x2A:
                self.assertEqual(list(p[7:-1:7]), [8, 9, 10, 11, 12, 13])
        controller.command('release')
        self.assertTrue(all(port.registers[sid][0x28] == 0 for sid in real.SERVO_IDS))

    def test_smooth_formation_single_step_hold_and_continuous_reverse(self):
        controller, _ = prepared_controller()
        self.assertFalse(controller.forming)
        self.assertEqual(controller.phase, 0.)
        controller.command('step')
        for _ in range(20):
            controller.tick(.02)
        self.assertAlmostEqual(controller.phase, .1)
        self.assertEqual(controller.mode, 'hold')
        controller.command('back')
        for _ in range(20):
            controller.tick(.02)
        self.assertAlmostEqual(controller.phase, 0.)
        controller.command('run')
        for _ in range(10):
            controller.tick(.02)
        old_phase = controller.phase
        old_target = controller.targets.copy()
        controller.command('reverse')
        self.assertEqual(controller.targets, old_target)  # Direction change itself sends no jump.
        controller.tick(.02)
        self.assertLess(controller.phase, old_phase)
        controller.command('pause')
        phase, target = controller.phase, controller.targets.copy()
        for _ in range(5):
            controller.tick(.02)
        self.assertEqual(controller.phase, phase)
        self.assertEqual(controller.targets, target)
        controller.command('amplitude=58 omega=0.5')
        previous = controller.targets.copy()
        controller.tick(10.)  # A delayed loop must never catch up with a 10-second angle jump.
        self.assertLessEqual(max(abs(a - b) for a, b in zip(controller.targets, previous)),
                             real.FORM_RATE_DEG_S * real.CONTROL_PERIOD_S + 1e-9)

    def test_invalid_assignment_is_atomic_and_window_requires_release(self):
        controller, _ = prepared_controller()
        previous = controller.snapshot()
        for command in ('amplitude=58 omega=nan', 'amplitude=58 window=7:14',
                        'delta=0', 'window=0:13', 'omega=5'):
            with self.subTest(command=command), self.assertRaises(ValueError):
                controller.command(command)
            self.assertEqual(controller.snapshot(), previous)
        controller.command('release')
        controller.command('window=7:14')
        controller.command('apply')
        self.assertEqual(sum(controller.enabled), 8)

    def test_mask_readback_mismatch_is_detected(self):
        class IgnoredMaskSerial(RecordingSerial):
            def write(self, packet):
                if packet[4] == 0x83 and packet[5] == 0x28:
                    return len(packet)
                return super().write(packet)
        bus = real.FeetechBus(IgnoredMaskSerial())
        with self.assertRaisesRegex(RuntimeError, '读回不一致'):
            real.RollingArcController(bus).command('apply')

    def test_protocol_error_echo_and_timeout(self):
        class ResponseSerial(RecordingSerial):
            response_mode = 'echo'

            def write(self, packet):
                count = super().write(packet)
                if packet[4] == 0x02:
                    response = bytes(self.received)
                    if self.response_mode == 'echo':
                        self.received[:] = b'noise' + packet + response
                    elif self.response_mode == 'corrupt':
                        self.received[-1] ^= 1
                    elif self.response_mode == 'error':
                        self.received[:] = real.instruction_packet(packet[2], 4, response[5:-1])
                    elif self.response_mode == 'timeout':
                        self.received.clear()
                return count
        port = ResponseSerial()
        bus = real.FeetechBus(port)
        self.assertEqual(bus.read_register(1, 0x38, 2), real.ZERO_RAW[0].to_bytes(2, 'little'))
        for mode, error in (('corrupt', RuntimeError), ('error', RuntimeError), ('timeout', TimeoutError)):
            port.response_mode = mode
            with self.subTest(mode=mode), self.assertRaises(error):
                bus.read_register(1, 0x38, 2)

    def test_ctrl_key_mapping_and_ignore_unmodified_keys(self):
        self.assertEqual(real.decode_hotkey(ord('J'), True), 'step')
        self.assertEqual(real.decode_hotkey(ord('H'), True), 'back')
        self.assertEqual(real.decode_hotkey(ord('U'), True), 'apply')
        self.assertEqual(real.decode_hotkey(32, True), 'pause')
        for vk in (ord('J'), 32, 116, 117):
            self.assertIsNone(real.decode_hotkey(vk, False))
        self.assertIsNone(real.decode_hotkey(116, True))
        class FakeListener:
            def __init__(self, on_press, on_release):
                self.press, self.release = on_press, on_release
            def start(self):
                pass
        keyboard = SimpleNamespace(Key=SimpleNamespace(ctrl='ctrl', ctrl_l='left_ctrl', ctrl_r='right_ctrl'),
                                   Listener=FakeListener)
        commands = real.queue.Queue()
        listener = real.global_key_listener(commands, keyboard)
        key = namedtuple('KeyCode', ['vk'])(ord('J'))
        listener.press('left_ctrl')
        listener.press(key)
        listener.press(key)
        self.assertEqual(commands.get_nowait(), 'step')
        self.assertTrue(commands.empty())
        listener.release(key)
        listener.press(key)
        self.assertEqual(commands.get_nowait(), 'step')

    def test_disconnect_stops_main_and_cleanup_releases_and_closes(self):
        class FaultSerial(RecordingSerial):
            def write(self, packet):
                if packet[4] == 0x02 and packet[5] == 0x38 and self.registers[8][0x28] == 1:
                    raise OSError('simulated cable failure during feedback')
                return super().write(packet)
        port = FaultSerial()
        argv = ['Rolling_Arc_Real.py', '--dry-run', '--no-hotkeys', '--no-console',
                '--auto', '--fast', '--duration', '1']
        with patch.object(sys, 'argv', argv), patch.object(real, 'DryRunSerial', return_value=port), redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                real.main()
        self.assertTrue(port.closed)
        self.assertTrue(all(port.registers[sid][0x28] == 0 for sid in real.SERVO_IDS))


if __name__ == '__main__':
    unittest.main()
