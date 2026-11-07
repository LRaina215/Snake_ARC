"""Consequential real-control checks using an in-memory serial bus, never COM ports."""
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
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
    def test_default_eighteen_joint_setup_and_table_diagnostics(self):
        real.validate_setup()
        self.assertEqual(real.NUM_JOINTS, 18)
        self.assertEqual(real.SERVO_IDS, list(range(1, 19)))
        self.assertEqual(len(real.LINK_LENGTHS_M), 19)
        with patch.object(real, 'ZERO_RAW', real.ZERO_RAW+[2057, 2070]):
            with self.assertRaisesRegex(ValueError, 'NUM_JOINTS=18.*ZERO_RAW=20'):
                real.validate_setup()
        with patch.object(real, 'SERVO_IDS', [1]+real.SERVO_IDS[:-1]):
            with self.assertRaisesRegex(ValueError, 'SERVO_IDS.*重复ID'):
                real.validate_setup()

    def test_eighteen_joint_mode_entries_keep_motor_dimensions(self):
        for mode in ('gait_sliding', 'gait_feeding', 'gait_helix', 'gait_gravity'):
            with self.subTest(mode=mode), redirect_stdout(io.StringIO()):
                controller, port = prepared_controller()
                controller.command(mode)
                controller.refresh_compliance_feedback()
                self.assertEqual(len(controller.targets), 18)
                self.assertEqual(len(controller.enabled), 18)
                self.assertEqual(len(controller.measured_angles()), 18)
                self.assertEqual(set(port.registers), set(range(1, 19)))
                controller.command('step')
                for _ in range(5):
                    controller.refresh_compliance_feedback()
                    controller.tick(real.CONTROL_PERIOD_S)
                self.assertEqual(len(controller.targets), 18)
                self.assertEqual(len(controller.enabled), 18)

    def test_calibration_and_wave_agree_with_validated_simulation(self):
        profile = json.loads((ROOT / 'configs/hardware_18j_reference.json').read_text(encoding='utf-8'))
        joints = sorted(profile['Joints'], key=lambda j: j['Index'])[:real.NUM_JOINTS]
        self.assertEqual(real.ZERO_RAW, [j['Zero'] for j in joints])
        self.assertEqual(real.SERVO_IDS, [j['ServoId'] for j in joints])
        self.assertEqual(real.SERVO_DIRECTION, [j['Direction'] for j in joints])
        for phase in (0., .7, math.pi / 2, math.pi):
            expected = [math.degrees(x) for x in ArcParameters(joints=real.NUM_JOINTS).targets(phase)]
            actual = real.joint_angles_deg(phase, 60., 8, 13)
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b)
        self.assertEqual(real.angle_to_raw(7, 60), real.ZERO_RAW[7] - round(60 * 4096 / 360))

    def test_confirmed_straight_pose_has_zero_angles_and_straight_reconstruction(self):
        fixture = json.loads((ROOT/'validation/fixtures/straight_pose_hardware18_20261107.json').read_text(encoding='utf-8'))
        raw = fixture['reconstructed_raw_positions']
        old = [(value-zero)*360/4096*real.MODEL_TO_SERVO_SIGN[i]
               for i, (value, zero) in enumerate(zip(raw, fixture['old_zero_raw']))]
        self.assertAlmostEqual(old[6], -86.220703125)
        self.assertAlmostEqual(old[9], -89.82421875)
        angles = [real.raw_to_angle(i, value) for i, value in enumerate(raw)]
        self.assertLess(max(abs(a) for a in angles), 1e-9)
        points = real.forward(real.np.radians(angles), real.LINK_LENGTHS_M)
        self.assertLess(float(real.np.max(real.np.abs(points[:, 1:]))), 1e-12)
        self.assertEqual([real.angle_to_raw(i, 0.) for i in range(real.NUM_JOINTS)], raw)

    def test_preload_then_enable_and_no_target_writes_to_passive_joints(self):
        port = RecordingSerial()
        for regs in port.registers.values():
            regs[0x28] = 1  # Simulate previous code leaving every servo enabled.
        controller = real.RollingArcController(real.FeetechBus(port))
        controller.command('apply')
        self.assertEqual([port.registers[sid][0x28] for sid in real.SERVO_IDS],
                         [int(8 <= j <= 13) for j in range(1, real.NUM_JOINTS+1)])
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
                        'delta=0', 'window=0:13'):
            with self.subTest(command=command), self.assertRaises(ValueError):
                controller.command(command)
            self.assertEqual(controller.snapshot(), previous)
        with patch.object(real, 'DIY_PARAMETERS', False):
            with self.assertRaises(ValueError):
                controller.command('omega=5')
        with patch.object(real, 'DIY_PARAMETERS', True):
            controller.command('omega=5')
            self.assertEqual(controller.omega, 5.)
            controller.command('omega=0.6')
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
        with patch.object(real, 'DryRunSerial', return_value=port), redirect_stdout(io.StringIO()):
            with self.assertRaises(OSError):
                real.run_controller(dry_run=True, hotkeys=False, console=False,
                                    auto=True, fast=True, duration_s=1)
        self.assertTrue(port.closed)
        self.assertTrue(all(port.registers[sid][0x28] == 0 for sid in real.SERVO_IDS))

    def test_auto_ch340_and_manual_port_never_select_an_ambiguous_device(self):
        import serial
        import serial.tools.list_ports
        robot = SimpleNamespace(device='COM9', description='USB-SERIAL ch340')
        other = SimpleNamespace(device='COM3', description='Bluetooth')
        duplicate = SimpleNamespace(device='COM12', description='USB-SERIAL CH340')
        with patch.object(real, 'PORT', None), \
             patch.object(serial.tools.list_ports, 'comports') as scan, \
             patch.object(serial, 'Serial') as constructor:
            for available in ([], [other], [robot, duplicate]):
                scan.return_value = available
                with self.subTest(available=available), self.assertRaises(RuntimeError):
                    real.open_robot_serial()
            constructor.assert_not_called()
            scan.return_value = [other, robot]
            real.open_robot_serial()
            constructor.assert_called_once_with(
                'COM9', real.BAUD_RATE, timeout=.01, write_timeout=real.SERIAL_TIMEOUT_S)
            scan.reset_mock()
            constructor.reset_mock()
            with patch.object(real, 'PORT', 'COM12'):
                real.open_robot_serial()
            scan.assert_not_called()
            constructor.assert_called_once_with(
                'COM12', real.BAUD_RATE, timeout=.01, write_timeout=real.SERIAL_TIMEOUT_S)

    def test_plain_main_reads_top_settings_keeps_motors_idle_and_creates_new_logs(self):
        import serial
        import serial.tools.list_ports
        opened = []
        def open_fake(name, *args, **kwargs):
            port = RecordingSerial()
            port.port = name
            opened.append(port)
            return port
        def idle_listener(commands, keyboard):
            commands.put('quit')
            return SimpleNamespace(stop=lambda: None, join=lambda timeout: None)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(real, 'PORT', None), patch.object(real, 'DRY_RUN', False), \
                 patch.object(real, 'SHOW_PREVIEW', False), \
                 patch.object(real, 'SAVE_LOG', True), patch.object(real, 'LOG_DIRECTORY', Path(folder)), \
                 patch.object(serial.tools.list_ports, 'comports', return_value=[
                     SimpleNamespace(device='COM9', description='USB-SERIAL CH340')]), \
                 patch.object(serial, 'Serial', side_effect=open_fake), \
                 patch.object(real, 'global_key_listener', side_effect=idle_listener), \
                 patch.object(real, 'console_listener', return_value=None), \
                 patch.object(real, 'datetime') as clock, \
                 patch.object(sys, 'argv', ['Rolling_Arc_Real.py']), redirect_stdout(io.StringIO()):
                clock.now.return_value.strftime.return_value = '20261012_120000_000000'
                real.main()
                real.main()
            self.assertEqual(len(list(Path(folder).glob('rolling_arc_*.jsonl'))), 2)
        self.assertEqual(len(opened), 2)
        for port in opened:
            self.assertEqual(port.port, 'COM9')
            self.assertTrue(port.closed)
            self.assertEqual(port.packets, [])  # 直接启动与退出没有自行形成圆弧/改变使能。


if __name__ == '__main__':
    unittest.main()
