"""Mode-4 repeated-key regression; recorded inputs and memory serial only."""
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
import queue
import subprocess
import sys
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codes import Rolling_Arc_Real as real

REPORT = ROOT/'validation/reports/feeding_key_response_20261108'
BEFORE = 'ee22e39'
EVIDENCE = {}


def old_module():
    module = types.ModuleType('_before_feeding_key_response')
    module.__file__ = str(ROOT/'codes/Rolling_Arc_Real.py')
    sys.modules[module.__name__] = module
    source = subprocess.check_output(['git', 'show', BEFORE+':codes/Rolling_Arc_Real.py']).decode('utf-8')
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def software_controller(module):
    class Port(module.DryRunSerial):
        def __init__(self):
            super().__init__(); self.packets = []
        def write(self, packet):
            self.packets.append(packet)
            return super().write(packet)

    class Clock(module.RollingArcController):
        def __init__(self):
            self.t = 0.; self.port = Port()
            super().__init__(module.FeetechBus(self.port))
        def motor_sample_clock(self):
            return self.t, self.feedback_motor_clock_s
        def refresh_compliance_feedback(self):
            super().refresh_compliance_feedback()
            self.feedback_motor_clock_s = self.feedback_time_s = self.t
        def cycle(self):
            module.poll_control_feedback(self, self.t, self.t+100.)
            self.tick(.02); self.t += .02
        def show_status(self, include_feedback=False):
            pass

    c = Clock(); c.command('gait_feeding'); c.command('apply')
    for _ in range(450): c.cycle()
    assert not c.forming and c.settle_remaining <= 0
    c.command('step')
    for _ in range(450):
        c.cycle()
        if c.feeding.stage == 'acquire': break
    assert c.feeding.stage == 'acquire', (c.feeding.stage, c.feeding.fault)
    return c


class FeedingKeyResponseTests(unittest.TestCase):
    def test_same_key_preserves_acquisition_baseline_target_and_mask(self):
        old = old_module()
        results = []
        with redirect_stdout(io.StringIO()):
            for module in (old, real):
                c = software_controller(module)
                core = c.feeding
                core.acquisition_samples = [(1., np.zeros(module.NUM_JOINTS), .1, 'fixture')]
                baseline = core.baseline = .123
                goal = core.goal_m = .1875
                targets, mask, action = c.targets.copy(), c.enabled.copy(), c.feeding_action_id
                packets = len(c.port.packets)
                result = c.command('step')
                reads = sum(p[4] == 2 for p in c.port.packets[packets:])
                results.append(dict(version='old' if module is old else 'fixed',
                    samples=len(core.acquisition_samples), baseline=core.baseline,
                    goal=core.goal_m, action_id_delta=c.feeding_action_id-action, read_packets=reads))
                if module is real:
                    self.assertEqual(result, 'feeding_continues')
                    self.assertEqual(len(core.acquisition_samples), 1)
                    self.assertEqual(core.baseline, baseline); self.assertEqual(core.goal_m, goal)
                    self.assertEqual(c.targets, targets); self.assertEqual(c.enabled, mask)
                    self.assertEqual(reads, 0)
                else:
                    self.assertEqual(len(core.acquisition_samples), 0)
                    self.assertIsNone(core.baseline)
                    self.assertEqual(reads, 2*module.NUM_JOINTS)
        EVIDENCE['repeat_key_comparison'] = results

    def test_repeated_keys_no_longer_prevent_stable_sample_collection(self):
        results = []
        with redirect_stdout(io.StringIO()):
            for module in (old_module(), real):
                c = software_controller(module)
                before = c.feeding_action_id
                for frame in range(150):
                    if frame % 10 == 0: c.command('step')
                    c.cycle()
                results.append(dict(version='fixed' if module is real else 'old',
                    stage=c.feeding.stage, position_joints=c.feeding.position,
                    action_id_delta=c.feeding_action_id-before, fault=c.feeding.fault))
                if module is real:
                    self.assertEqual(c.feeding.stage, 'feed_out')
                    self.assertGreater(c.feeding.position, 0.)
                    self.assertEqual(c.feeding_action_id, before)
                else:
                    self.assertEqual(c.feeding.stage, 'acquire')
                    self.assertEqual(c.feeding.position, 0.)
                    self.assertEqual(c.feeding_action_id-before, 15)
        EVIDENCE['continuous_key_comparison'] = results

    def test_reverse_pause_next_segment_and_return_remain_available(self):
        with redirect_stdout(io.StringIO()):
            c = software_controller(real)
            core = c.feeding
            count = c.feeding_action_id; p = core.position
            packets = len(c.port.packets)
            c.command('back')
            self.assertLess(core.options.requested_transfer_m, 0.)
            self.assertEqual(c.feeding_action_id, count+1); self.assertEqual(core.position, p)
            self.assertEqual(sum(p[4] == 2 for p in c.port.packets[packets:]), real.NUM_JOINTS)
            c.pause_feeding(); count = c.feeding_action_id
            c.command('back')
            self.assertEqual(c.feeding_action_id, count+1)
            core.completed = True; core.stage = 'complete'; c.pause_feeding()
            count = c.feeding_action_id
            c.command('back')
            self.assertEqual(c.feeding_action_id, count+1)
            core.stage = 'feed_return'
            count = c.feeding_action_id
            c.command('back')
            self.assertEqual(c.feeding_action_id, count+1)
            self.assertEqual(core.stage, 'acquire')
            c.command('release')
            self.assertTrue(all(c.port.registers[sid][0x28] == 0 for sid in real.SERVO_IDS))

    def test_dispatch_logs_received_key_and_feeding_stage(self):
        with redirect_stdout(io.StringIO()):
            c = software_controller(real)
            q = queue.Queue(); q.put('step')
            real.process_control_commands(c, q, c.t)
            event = c.control_commands[-1]
            self.assertEqual(event['result'], 'continued')
            self.assertEqual(event['stage_before'], 'acquire')
            self.assertEqual(event['stage_after'], 'acquire')
            self.assertIn('未重置', event['reason'])


def archive_real_evidence():
    path = ROOT/'codes/logs/rolling_arc_20261108_182119_098682.jsonl'
    rows, events = [], {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            if not row.get('feeding'): continue
            rows.append(row)
            for event in row.get('control_commands', []):
                if event['time_s'] >= rows[0]['time_s']:
                    events[event['sequence']] = event
            if row['feeding'].get('completed'): break
    REPORT.mkdir(parents=True, exist_ok=True)
    with gzip.open(REPORT/'actual_mode4_slice.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    EVIDENCE['real_log'] = dict(source=str(path.relative_to(ROOT)), rows=len(rows),
        start_s=rows[0]['time_s'], end_s=rows[-1]['time_s'],
        received_step_commands=sum(e['command']=='step' for e in events.values()),
        maximum_action_id=max(r['feeding_action_id'] for r in rows),
        completed_relative_goals=1, final_position_joints=rows[-1]['feeding']['position_joints'],
        meaning='Archived real input evidence, not a new real test of this fix')


if __name__ == '__main__':
    archive_real_evidence()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FeedingKeyResponseTests))
    (REPORT/'checks.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),
        tests=result.testsRun, baseline_commit=BEFORE, hardware_run=False, new_physics_runs=0,
        failures=[str(t) for t,_ in result.failures+result.errors], **EVIDENCE),
        ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
