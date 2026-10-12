"""Replay saved motor inputs through the standalone port; no physics or COM.

Compare targets, masks, phase and events over every archived control frame.
Separate memory-serial checks cover preload/torque order and Ctrl semantics.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ast
from contextlib import redirect_stdout
from dataclasses import asdict
import gzip
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from codes import Rolling_Arc_Real as real
from codes import Rolling_Arc_Debug_Mujoco as debug
from snake_control.rolling_arc import ArcParameters
from snake_control.motor_feedback_sliding import MotorFeedbackSliding
from snake_control.motor_shape_observer import PhaseShapeReference

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'validation/reports/repeated_motor_sliding'
CHECKS = []


class AngleBus:
    def __init__(self):
        self.angles = np.zeros(20)
        self.enabled = np.zeros(20, bool)
        self.preloads = []
        self.reads = 0

    def read_states(self):
        self.reads += 1
        return {i: dict(angle_deg=float(q), load_raw=None, current_raw=None)
                for i, q in enumerate(self.angles)}

    def read_angles(self, indices):
        return {i: float(self.angles[i]) for i in indices}

    def send_targets(self, target, enabled):
        self.target = np.asarray(target).copy()
        self.sent_ids = (np.flatnonzero(enabled)+1).tolist()

    def prepare_enabled(self, angles, enabled, configure_mask=None):
        entering = np.asarray(enabled) & ~self.enabled
        if not np.array_equal(entering, configure_mask):
            raise AssertionError('Must configure only the entering joint')
        self.preloads.append(dict(entering=(np.flatnonzero(entering)+1).tolist(),
            preload=np.asarray(angles)[entering].tolist(), old=(np.flatnonzero(self.enabled)+1).tolist(),
            new=(np.flatnonzero(enabled)+1).tolist()))
        self.enabled = np.asarray(enabled).copy()

    def set_enabled(self, enabled, verify=True):
        self.enabled = np.asarray(enabled).copy()


class RecordingSerial(real.DryRunSerial):
    def __init__(self):
        super().__init__()
        self.packets = []

    def write(self, packet):
        self.packets.append(packet)
        return super().write(packet)


class RepeatedPortTests(unittest.TestCase):
    def test_complete_saved_input_replay_matches_experimental_core_and_two_adapters(self):
        reference = PhaseShapeReference.from_dict(json.loads((ROOT/'configs/motor_shape_reference.json').read_text()))
        for path in sorted(p for p in REPORT.iterdir() if p.is_dir()):
            status = json.loads((path/'status.json').read_text())
            p = ArcParameters(**status['parameters'])
            expected = MotorFeedbackSliding(p, real.LINK_LENGTHS_M, reference, feedback_period_s=.2)
            port = real.MotorFeedbackSliding(real.ArcParameters(**asdict(p)), real.LINK_LENGTHS_M,
                real.PhaseShapeReference.from_dict(real.MOTOR_PHASE_REFERENCE), feedback_period_s=.2)
            bus = AngleBus()
            adapter = real.RollingArcController(bus)
            adapter.first, adapter.last = p.first_joint, p.last_joint
            adapter.sliding, adapter.compliance = port, port.core
            adapter.enabled = p.enabled.tolist()
            adapter.targets = np.degrees(p.targets(0)).tolist()
            adapter.mode = 'hold'  # Use archived known commanded phase below.
            bus.enabled = p.enabled.copy()
            manual = None
            if status['case'] == 'natural':
                session = SimpleNamespace(sim=SimpleNamespace(p=p), elapsed=0., record_event=lambda *a, **k: None)
                mbus = AngleBus()
                manual = debug.DebugArcController(mbus, session)
                manual.sliding = real.MotorFeedbackSliding(real.ArcParameters(**asdict(p)), real.LINK_LENGTHS_M,
                    real.PhaseShapeReference.from_dict(real.MOTOR_PHASE_REFERENCE), feedback_period_s=.2)
                manual.compliance = manual.sliding.core
                manual.enabled, manual.targets = p.enabled.tolist(), adapter.targets.copy()
                manual.mode = 'hold'
                mbus.enabled = p.enabled.copy()
            frames, target_error, phase_error, mask_error, manual_error = 0, 0., 0., 0, 0.
            with gzip.open(path/'actual_trace.jsonl.gz', 'rt', encoding='utf8') as stream, redirect_stdout(io.StringIO()):
                for row in map(json.loads, stream):
                    sample = row['motor_input']
                    if sample is None:
                        continue
                    t = row['controller']['control_time_s']
                    q, ft, phi = sample['angles_deg'], sample['feedback_time_s'], sample['phase_rad']
                    target, mask = expected.advance(q, ft, t, phase_at_feedback=phi)
                    bus.angles = np.asarray(q)
                    adapter.states = bus.read_states()
                    adapter.feedback_phase_rad = phi
                    adapter.motor_sample_clock = lambda: (t, ft)
                    adapter.phase = row['controller']['phase_rad']
                    adapter.tick_motor_sliding(.02)
                    target_error = max(target_error, float(abs(np.radians(adapter.targets)-target).max()),
                                       float(abs(target-np.asarray(row['target_rad'])).max()))
                    phase_error = max(phase_error, abs(adapter.phase-expected.phase))
                    mask_error += int(adapter.enabled != mask.tolist() or adapter.enabled != row['enabled'])
                    self.assertEqual(bus.sent_ids, (np.flatnonzero(mask)+1).tolist())
                    if manual is not None:
                        session.elapsed = t
                        mbus.angles = np.asarray(q)
                        manual.command_phase = phi
                        manual.phase = row['controller']['phase_rad']
                        manual.tick_motor_sliding(.02)
                        self.assertEqual(manual.enabled, mask.tolist())
                        manual_error = max(manual_error, float(abs(np.radians(manual.targets)-target).max()))
                    frames += 1
            expected_events = [e['to'] for e in expected.events if e['kind'] == 'complete']
            self.assertEqual([e['to'] for e in port.events if e['kind'] == 'complete'], expected_events)
            self.assertEqual(expected_events, [e['to'] for e in status['events'] if e['kind'] == 'complete'])
            self.assertLess(target_error, 1e-9)
            self.assertLess(phase_error, 1e-9)
            self.assertEqual(mask_error, 0)
            self.assertEqual(len(bus.preloads), 2)
            if manual is not None:
                self.assertLess(manual_error, 1e-9)
                self.assertEqual(mbus.reads, int(round(row['controller']['feedback_time_s']/.2))+1)
            CHECKS.append(dict(source=path.name, frames=frames, target_error_rad=target_error,
                phase_error_rad=phase_error, mask_mismatches=mask_error, completed=expected_events,
                real_adapter_preloads=bus.preloads, manual_target_error_rad=manual_error,
                manual_feedback_reads=None if manual is None else mbus.reads))

    def test_serial_preloads_entering_joint_before_six_joint_torque_exchange(self):
        path = next(REPORT.glob('natural_*/actual_trace.jsonl.gz'))
        with gzip.open(path, 'rt', encoding='utf8') as stream:
            row = next(r for r in map(json.loads, stream) if r['controller']['window'] == [5, 10])
        serial = RecordingSerial()
        bus = real.FeetechBus(serial)
        old, new = np.zeros(20, bool), np.asarray(row['enabled'], bool)
        old[5:11] = True
        bus.set_enabled(old)
        serial.packets.clear()
        entering = new & ~old
        q = row['motor_input']['angles_deg']
        bus.prepare_enabled(q, new, configure_mask=entering)
        bus.send_targets(np.degrees(row['target_rad']), new)
        writes = [p for p in serial.packets if p[4] == 0x83]
        self.assertEqual([p[5] for p in writes], [0x2A, 0x30, 0x29, 0x28, 0x2A])
        self.assertEqual(list(writes[0][7:-1:7]), [5])
        raw = int.from_bytes(writes[0][8:10], 'little')
        self.assertEqual(raw, real.angle_to_raw(4, q[4]))
        self.assertEqual(list(writes[-1][7:-1:7]), list(range(5, 11)))
        self.assertEqual([serial.registers[sid][0x28] for sid in real.SERVO_IDS], new.astype(int).tolist())
        bus.set_enabled([False]*20)
        self.assertFalse(any(serial.registers[sid][0x28] for sid in real.SERVO_IDS))

    def test_shared_hotkeys_pause_freeze_cancel_and_guarded_mode_switches(self):
        path = next(REPORT.glob('natural_*/actual_trace.jsonl.gz'))
        with gzip.open(path, 'rt', encoding='utf8') as stream:
            row = next(r for r in map(json.loads, stream) if r['time_s'] >= 2.2)
        bus = AngleBus()
        bus.angles = np.asarray(row['motor_input']['angles_deg'])
        c = real.RollingArcController(bus)
        c.first, c.last = 6, 11
        c.enabled, c.targets = row['enabled'], np.degrees(row['target_rad']).tolist()
        bus.enabled = np.asarray(c.enabled)
        c.phase = c.command_phase = row['controller']['phase_rad']
        c.mode, c.forming, c.settle_remaining = 'hold', False, 0.
        now = [2.2]
        c.motor_sample_clock = lambda: (now[0], now[0])
        with patch.object(real, 'open_robot_serial', side_effect=AssertionError('No hardware')), redirect_stdout(io.StringIO()):
            for vk, command in ((49, 'gait_basic'), (50, 'gait_fixed'), (51, 'gait_sliding'), (87, 'window_auto')):
                self.assertEqual(real.decode_hotkey(vk, True), command)
                self.assertEqual(debug.decode_sim_hotkey(vk, True), command)
                self.assertIsNone(real.decode_hotkey(vk, False))
            c.command('gait_sliding')
            c.tick(.02)
            old_phase = c.phase
            c.command('run'); now[0] += .02; c.tick(.02)
            self.assertAlmostEqual(c.phase, old_phase+.012)
            c.command('pause'); now[0] += .02; c.tick(.02)
            self.assertAlmostEqual(c.phase, old_phase+.012)
            c.command('hold')
            targets, mask = c.targets.copy(), c.enabled.copy()
            now[0] += .02; c.tick(.02)
            self.assertEqual((c.targets, c.enabled), (targets, mask))
            c.sliding.pending = dict(to=[5, 10])
            c.sliding.phase_hold = True
            c.command('window_auto')
            self.assertIsNone(c.sliding.pending)
            self.assertFalse(c.sliding.phase_hold)
            self.assertFalse(c.sliding.automatic)
            for command in ('reverse', 'back', 'amplitude=62', 'omega=.7', 'window=7:12'):
                with self.assertRaises(ValueError):
                    c.command(command)
            # A new invalid actual shape keeps the old mask/targets and latches hold.
            bus.angles[:] = 0.
            c.refresh_compliance_feedback()
            now[0] += .2
            c.tick(.02)
            self.assertTrue(c.sliding_fault)
            self.assertEqual((c.targets, c.enabled), (targets, mask))
            c.motor_sample_clock = lambda: (now[0]+.81, now[0])
            with self.assertRaises(TimeoutError):
                c.tick(.02)
            c.motor_sample_clock = lambda: (now[0], now[0])
            bus.angles = np.asarray(row['motor_input']['angles_deg'])
            c.command('gait_fixed')
            self.assertIsNone(c.sliding)
            c.command('gait_basic')
            self.assertIsNone(c.compliance)
            c.command('release')
            self.assertFalse(any(c.enabled))
        tree = ast.parse((ROOT/'codes/Rolling_Arc_Real.py').read_text(encoding='utf8'))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any(value and value.startswith(('codes', 'snake_control', 'mujoco')) for value in imports))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RepeatedPortTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (REPORT/'port_checks.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),
        tests=result.testsRun, dynamics_runs=0, physical_serial_connections=0, replays=CHECKS),
        ensure_ascii=False, indent=2), encoding='utf8')
    sys.exit(0 if result.wasSuccessful() else 1)
