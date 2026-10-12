"""Fixed-section hardware port: saved physics poses + memory serial only.

Replaces obsolete load-estimated sliding-window hardware checks. No COM port,
new dynamics run or claim of physical success.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import unittest
from unittest.mock import patch
import numpy as np
from codes import Rolling_Arc_Real as real
from snake_control.fixed_section_compliance import FixedSectionCompliance, SectionOptions
from snake_control.rolling_arc import ArcParameters

ROOT = Path(__file__).resolve().parents[1]


class RecordingSerial(real.DryRunSerial):
    def __init__(self):
        super().__init__()
        self.packets = []

    def write(self, packet):
        self.packets.append(packet)
        return super().write(packet)


def prepared(section=True):
    port = RecordingSerial()
    c = real.RollingArcController(real.FeetechBus(port))
    c.command('apply')
    for _ in range(320):
        c.tick(.02)
    if section:
        c.command('compliance')
    return c, port


def fixed_bits(c):
    return (c.amplitude, c.omega, c.phase_step, c.first, c.last, c.options,
            tuple(c.enabled), tuple(c.targets), c.phase)


class RealFixedSectionTests(unittest.TestCase):
    def test_port_matches_verified_core_on_eight_saved_actual_poses_and_units(self):
        with np.load(ROOT/'validation/reports/fixed_section/r5_taper90_trace.npz') as trace:
            valid = np.flatnonzero(trace['metrics'][:, 0] >= 3)
            for i in valid[np.linspace(0, len(valid)-1, 8).astype(int)]:
                measured = trace['qpos'][i, 7:]
                phase = float(trace['metrics'][i, 1])
                expected = FixedSectionCompliance(ArcParameters(), real.LINK_LENGTHS_M, SectionOptions())
                actual = real.FixedSectionCompliance(real.SectionReference(8, 13, 60.),
                                                      real.LINK_LENGTHS_M, real.make_section_options())
                target = actual.update(measured, phase, 0., .02)
                np.testing.assert_allclose(target, expected.update(measured, phase, 0., .02), atol=1e-12)
                self.assertAlmostEqual(actual.geometry['total_turn'], expected.geometry['total_turn'], places=12)
                for j in range(7, 13):
                    degrees = math.degrees(target[j])
                    decoded = real.raw_to_angle(j, real.angle_to_raw(j, degrees))
                    self.assertLessEqual(abs(decoded-degrees), 360/4096/2+1e-10)

    def test_ctrl_e_preserves_run_phase_targets_and_fixed_mask_both_directions(self):
        c, port = prepared(section=False)
        self.assertIsNone(c.compliance)
        self.assertEqual(real.decode_hotkey(69, True), 'compliance')
        c.command('run')
        for _ in range(5):
            c.tick(.02)
        before = (c.phase, c.mode, c.targets.copy(), c.enabled.copy())
        writes = len([p for p in port.packets if p[4] == 0x83])
        c.command('compliance')
        self.assertIsInstance(c.compliance, real.FixedSectionCompliance)
        self.assertEqual(before, (c.phase, c.mode, c.targets, c.enabled))
        self.assertEqual(writes, len([p for p in port.packets if p[4] == 0x83]))
        for _ in range(30):
            c.refresh_compliance_feedback()
            old = c.targets.copy()
            c.tick(.02)
            self.assertLessEqual(max(abs(c.targets[i]-old[i]) for i in range(7, 13)), .9+1e-9)
        before = (c.phase, c.mode, c.targets.copy(), c.enabled.copy())
        c.command('compliance')
        self.assertIsNone(c.compliance)
        self.assertEqual(before, (c.phase, c.mode, c.targets, c.enabled))
        old = c.targets.copy()
        c.tick(.02)
        self.assertLessEqual(max(abs(c.targets[i]-old[i]) for i in range(7, 13)), .9+1e-9)
        for packet in port.packets:
            if packet[4] == 0x83 and packet[5] == 0x2A:
                self.assertEqual(list(packet[7:-1:7]), [8, 9, 10, 11, 12, 13])
        self.assertEqual([port.registers[s][0x28] for s in real.SERVO_IDS],
                         [int(8 <= j <= 13) for j in range(1, 21)])
        # Turning it off while paused must also return to the basic arc,
        # instead of leaving the compliant shape frozen indefinitely.
        c.command('pause')
        while c.forming:
            c.tick(.02)
        c.command('compliance')
        for _ in range(10):
            c.refresh_compliance_feedback()
            c.tick(.02)
        phase = c.phase
        c.command('compliance')
        self.assertTrue(c.forming)
        for _ in range(80):
            c.tick(.02)
        self.assertEqual(c.phase, phase)
        nominal = real.joint_angles_deg(phase, c.amplitude, c.first, c.last)
        np.testing.assert_allclose(c.targets[7:13], nominal[7:13], atol=1e-8)

    def test_pause_keeps_feedback_hold_freezes_targets_and_single_steps_finish(self):
        c, _ = prepared()
        before = c.targets.copy()
        for _ in range(12):
            c.refresh_compliance_feedback()
            c.tick(.02)
        self.assertEqual(c.phase, 0.)
        self.assertNotEqual(before, c.targets)
        c.command('step')
        for _ in range(40):
            c.refresh_compliance_feedback()
            c.tick(.02)
        self.assertAlmostEqual(c.phase, .1)
        self.assertEqual(c.mode, 'hold')
        c.command('back')
        for _ in range(40):
            c.refresh_compliance_feedback()
            c.tick(.02)
        self.assertAlmostEqual(c.phase, 0.)
        c.command('hold')
        phase, targets = c.phase, c.targets.copy()
        for _ in range(8):
            c.refresh_compliance_feedback()
            c.tick(.02)
        self.assertEqual(c.phase, phase)
        self.assertEqual(c.targets, targets)

    def test_unarmed_forming_or_invalid_actual_shape_rejects_enable(self):
        port = RecordingSerial()
        c = real.RollingArcController(real.FeetechBus(port))
        with self.assertRaisesRegex(ValueError, '先Ctrl'):
            c.command('compliance')
        self.assertFalse(c.touched_hardware)
        c.command('apply')
        with self.assertRaisesRegex(ValueError, '形成'):
            c.command('compliance')
        for _ in range(320):
            c.tick(.02)
        for sid, zero in zip(real.SERVO_IDS, real.ZERO_RAW):
            port.registers[sid][0x38:0x3A] = zero.to_bytes(2, 'little')
        before = fixed_bits(c)
        with self.assertRaisesRegex(ValueError, '不是有效开放圆弧'):
            c.command('compliance')
        self.assertIsNone(c.compliance)
        self.assertEqual(before, fixed_bits(c))

    def test_stale_feedback_fails_and_invalid_geometry_holds_without_mask_change(self):
        c, port = prepared()
        with patch.object(real.time, 'monotonic', return_value=c.feedback_wall_s+2):
            with self.assertRaises(TimeoutError):
                c.tick(.02)
        with patch.object(c.bus, 'read_states', return_value=c.states), \
                patch.object(real.time, 'monotonic', side_effect=[0., 1.]):
            with self.assertRaisesRegex(TimeoutError, '读取过慢'):
                c.refresh_compliance_feedback()
        c.command('run')
        phase, targets, mask = c.phase, c.targets.copy(), c.enabled.copy()
        for sid, zero in zip(real.SERVO_IDS, real.ZERO_RAW):
            port.registers[sid][0x38:0x3A] = zero.to_bytes(2, 'little')
        c.refresh_compliance_feedback()
        c.tick(.02)
        self.assertEqual(c.mode, 'hold')
        self.assertIsNotNone(c.section_block_reason)
        self.assertEqual((phase, targets, mask), (c.phase, c.targets, c.enabled))
        c.command('release')
        self.assertTrue(all(port.registers[sid][0x28] == 0 for sid in real.SERVO_IDS))

    def test_parameter_errors_atomic_no_support_api_and_raw_load_not_control_input(self):
        c, port = prepared()
        before = fixed_bits(c)
        for command in ('gain=.03 support=9:11', 'load=40', 'gain=.3', 'yielding=1',
                        'filter=0', 'amplitude=70', 'gain=.03 window=7:12'):
            with self.subTest(command=command), self.assertRaises(ValueError):
                c.command(command)
            self.assertEqual(before, fixed_bits(c))
        c.command('gain=.03 yielding=.8 filter=.2')
        self.assertEqual(c.options.gain, .03)
        self.assertEqual(c.options.yielding, .8)
        self.assertEqual(c.options.target_filter_s, .2)
        other, _ = prepared()
        other.command('gain=.03 yielding=.8 filter=.2')
        for regs in port.registers.values():
            regs[0x3C:0x3E] = (1024+1000).to_bytes(2, 'little')
            regs[0x45:0x47] = (32768+1000).to_bytes(2, 'little')
        for controller in (c, other):
            controller.command('run')
            for _ in range(20):
                controller.refresh_compliance_feedback()
                controller.tick(.02)
        self.assertEqual(c.targets, other.targets)
        self.assertEqual(c.enabled, other.enabled)
        self.assertEqual(c.phase, other.phase)

    def test_open_commands_and_calculations_do_not_invent_new_feedback_samples(self):
        c, _ = prepared()
        c.command('run')
        sequence = c.feedback_sequence
        for _ in range(8):
            c.tick(.02)
            self.assertTrue(c.compliance._check_goal(np.radians(c.targets))[0])
        self.assertEqual(c.feedback_sequence, sequence)
        self.assertGreater(c.compliance.updates, 1)
        for _ in range(100):
            c.refresh_compliance_feedback()
            c.tick(.02)
            self.assertEqual(c.enabled, [8 <= j <= 13 for j in range(1, 21)])
            self.assertTrue(c.compliance._check_goal(np.radians(c.targets))[0])


if __name__ == '__main__':
    unittest.main()
