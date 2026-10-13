"""Authorized v3 port: recorded inputs, memory serial and preview only; no new physics/COM."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ast
import gzip
import io
import json
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from codes import Rolling_Arc_Real as real
from codes import Rolling_Arc_Debug_Mujoco as debug
from validation.test_repeated_motor_port import AngleBus, RecordingSerial

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'validation/reports/phase_balanced_window'
REPORT = ROOT/'validation/reports/posture_port'
CHECKS = []


def rows(case):
    path = next(SOURCE.glob(case+'/actual_trace.jsonl.gz'))
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        yield from map(json.loads, stream)


def sample_at(t=71.3):
    return next(r for r in rows('return_repair_*') if r['time_s'] >= t-1e-8)


def references():
    return {6: real.PhaseShapeReference.from_dict(real.MOTOR_PHASE_REFERENCE),
            8: real.PhaseShapeReference.from_dict(real.MOTOR_PHASE_REFERENCE8)}


class PosturePortTests(unittest.TestCase):
    def test_full_v3_records_through_real_and_manual_adapters(self):
        for pattern in ('return_repair_*', 'acute_*'):
            path = next(p for p in SOURCE.glob(pattern) if p.is_dir())
            status = json.loads((path/'status.json').read_text(encoding='utf8'))
            p = real.HalfArcParameters(**status['parameters'])
            session = SimpleNamespace(sim=SimpleNamespace(p=p), elapsed=0., record_event=lambda *a, **k: None)
            buses = [AngleBus(), AngleBus()]
            controllers = [real.RollingArcController(buses[0]), debug.DebugArcController(buses[1], session)]
            for c, bus in zip(controllers, buses):
                c.sliding = real.PostureWindow(p, real.LINK_LENGTHS_M, references(), feedback_period_s=.2)
                c.compliance = c.sliding.core
                c.first, c.last = p.first_joint, p.last_joint
                c.enabled = p.enabled.tolist()
                c.targets = np.degrees(p.targets(p.initial_phase_rad)).tolist()
                c.mode, c.length_normalized = 'hold', True
                bus.enabled = p.enabled.copy()
            errors, frames = [0., 0.], 0
            with redirect_stdout(io.StringIO()):
                for row in rows(path.name):
                    f = row['motor_input']
                    if f is None:
                        continue
                    t, ft = row['controller']['control_time_s'], f['feedback_time_s']
                    session.elapsed = t
                    for i, (c, bus) in enumerate(zip(controllers, buses)):
                        bus.angles = np.array(f['angles_deg'])
                        if i == 0:
                            c.states = bus.read_states()
                            c.feedback_phase_rad = f['phase_rad']
                            c.motor_sample_clock = lambda: (t, ft)
                        c.command_phase = f['phase_rad']
                        c.phase = row['controller']['phase_rad']
                        c.tick_motor_sliding(.02)
                        self.assertFalse(c.sliding_fault, (path.name, t, c.section_block_reason))
                        self.assertEqual(c.enabled, row['enabled'], (path.name, t))
                        self.assertEqual(bus.sent_ids, (np.flatnonzero(row['enabled'])+1).tolist())
                        errors[i] = max(errors[i], float(abs(np.radians(c.targets)-row['target_rad']).max()))
                    frames += 1
            expected = [(e['kind'], e.get('operation'), e.get('to')) for e in status['events']]
            for c in controllers:
                actual = [(e['kind'], e.get('operation'), e.get('to')) for e in c.sliding.events]
                self.assertEqual(actual, expected)
            self.assertLess(max(errors), 1e-9)
            self.assertEqual(len(buses[0].preloads), len(buses[1].preloads))
            CHECKS.append(dict(case=path.name, frames=frames, target_errors_rad=errors,
                               mask_and_event_match=True, transfers=buses[0].preloads))
            print(f'{path.name}: {frames} frames, max error {max(errors):.3g}', flush=True)

    def test_expand_contract_serial_order(self):
        serial = RecordingSerial()
        bus = real.FeetechBus(serial)
        old = np.zeros(20, bool); old[7:13] = True
        bus.set_enabled(old)
        q = sample_at()['motor_input']['angles_deg']
        for window, entering_ids in (((8, 14), [14]), ((8, 13), [])):
            new = np.zeros(20, bool); new[window[0]-1:window[1]] = True
            serial.packets.clear()
            bus.prepare_enabled(q, new, configure_mask=new & ~old)
            bus.send_targets(q, new)
            writes = [p for p in serial.packets if p[4] == 0x83]
            self.assertEqual([p[5] for p in writes], [0x2A, 0x30, 0x29, 0x28, 0x2A] if entering_ids else [0x28, 0x2A])
            if entering_ids:
                self.assertEqual(list(writes[0][7:-1:7]), entering_ids)
            self.assertEqual(list(writes[-1][7:-1:7]), list(range(window[0], window[1]+1)))
            self.assertEqual([serial.registers[s][0x28] for s in real.SERVO_IDS], new.astype(int).tolist())
            old = new
        bus.set_enabled([False]*20)
        self.assertFalse(any(serial.registers[s][0x28] for s in real.SERVO_IDS))

    def test_seven_joint_mode_switches_and_hold(self):
        row = sample_at(40.)
        bus = AngleBus(); bus.angles = np.array(row['motor_input']['angles_deg'])
        c = real.RollingArcController(bus)
        c.first, c.last = 8, 14
        c.enabled = row['enabled']; bus.enabled = np.array(c.enabled)
        c.targets = np.degrees(row['target_rad']).tolist()
        c.length_normalized, c.mode = True, 'hold'
        c.phase = c.command_phase = row['controller']['phase_rad']
        now = [40.]
        c.motor_sample_clock = lambda: (now[0], now[0])
        before = c.targets.copy()
        with redirect_stdout(io.StringIO()):
            c.command('gait_sliding')
            self.assertIsInstance(c.sliding, real.PostureWindow)
            self.assertEqual(c.targets, before)
            c.command('run'); now[0] += .02; c.tick(.02)
            phase = c.phase
            c.command('pause'); now[0] += .02; c.tick(.02)
            self.assertEqual(c.phase, phase)
            c.command('hold'); target, mask = c.targets.copy(), c.enabled.copy()
            now[0] += .02; c.tick(.02)
            self.assertEqual((c.targets, c.enabled), (target, mask))
            c.sliding.pending = dict(to=[7, 14])
            c.command('window_auto')
            self.assertIsNone(c.sliding.pending)
            self.assertFalse(c.sliding.automatic)
            for command in ('back', 'reverse', 'amplitude=62', 'omega=.7'):
                with self.assertRaises(ValueError): c.command(command)
            c.command('gait_fixed')
            self.assertIsNone(c.sliding)
            self.assertEqual(c.enabled, mask)
            np.testing.assert_allclose(c.compliance.p.targets(phase), real.nominal_angles(real.LINK_LENGTHS_M, (8, 14), phase))
            c.command('gait_basic')
            self.assertIsNone(c.compliance)
            np.testing.assert_allclose(np.radians(c.basic_targets()), real.nominal_angles(real.LINK_LENGTHS_M, (8, 14), phase))
            c.command('release')
            self.assertFalse(any(bus.enabled))
        for vk in (49, 50, 51, 87, 85, 88):
            self.assertEqual(real.decode_hotkey(vk, True), debug.decode_sim_hotkey(vk, True))
            self.assertIsNone(real.decode_hotkey(vk, False))

    def test_preview_uses_measured_passive_angles_and_idle_feedback_without_writes(self):
        row = sample_at(40.)
        bus = AngleBus(); bus.angles = np.array(row['motor_input']['angles_deg'])
        c = real.RollingArcController(bus)
        c.preview_feedback = True
        real.poll_control_feedback(c, 0., 0.)
        self.assertEqual(bus.reads, 1)
        self.assertFalse(hasattr(bus, 'target'))
        self.assertFalse(real.preview_payload(c)['enable_known'])
        c.enabled = row['enabled']; c.targets = np.degrees(row['target_rad']).tolist()
        c.first, c.last = row['controller']['window']
        c.touched_hardware = True
        c.mode = 'run'
        c.phase = row['controller']['phase_rad']
        c.sliding = SimpleNamespace(automatic=True, selection=row['controller']['selection'],
                                    pending=row['controller']['pending'], observation=row['controller']['observation'])
        payload = real.preview_payload(c)
        passive = ~np.asarray(c.enabled)
        np.testing.assert_array_equal(np.asarray(payload['target_deg'])[passive], bus.angles[passive])
        np.testing.assert_array_equal(payload['angles_deg'], bus.angles)
        import matplotlib
        matplotlib.use('Agg')
        figure = real.JointPreviewFigure()
        figure.update(payload)
        REPORT.mkdir(parents=True, exist_ok=True)
        figure.figure.savefig(REPORT/'preview_actual.png', dpi=110)
        figure.plt.close(figure.figure)
        tree = ast.parse((ROOT/'codes/Rolling_Arc_Real.py').read_text(encoding='utf8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or '').startswith(('codes', 'snake_control', 'mujoco')))


if __name__ == '__main__':
    REPORT.mkdir(parents=True, exist_ok=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PosturePortTests))
    (REPORT/'checks.json').write_text(json.dumps(dict(passed=result.wasSuccessful(), tests=result.testsRun,
        dynamics_runs=0, hardware_connections=0, replays=CHECKS), ensure_ascii=False, indent=2), encoding='utf8')
    sys.exit(0 if result.wasSuccessful() else 1)
