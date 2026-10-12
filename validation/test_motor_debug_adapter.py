"""Check the existing manual adapter on saved states; no dynamics run."""
import gzip
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from codes import Rolling_Arc_Debug_Mujoco as debug


class MotorDebugAdapterTests(unittest.TestCase):
    def test_motor_mode_uses_no_contact_inputs_and_keeps_manual_phase_controls(self):
        root = Path(__file__).resolve().parents[1]
        trace = next((root/'validation/reports/motor_feedback_sliding').glob('closed_*/actual_trace.jsonl.gz'))
        with gzip.open(trace, 'rt', encoding='utf8') as stream:
            row = next(value for value in map(json.loads, stream) if value['time_s'] >= 2.2)
        session = debug.DebugSession()
        try:
            c, sim = session.controller, session.sim
            self.assertEqual(c.amplitude, 60.)  # Real-file tuning must not invalidate this simulation baseline.
            sim.data.qpos[:], sim.data.qvel[:], sim.data.time = row['actual_qpos'], row['actual_qvel'], row['time_s']
            session.elapsed = row['time_s']
            c.first, c.last = 5, 10
            c.enabled = row['enabled']
            c.targets = np.degrees(row['target_rad']).tolist()
            session.bus.enabled[:] = row['enabled']
            session.bus.targets_deg[:] = c.targets
            c.phase = row['controller']['phase_rad']
            c.mode, c.forming, c.settle_remaining = 'hold', False, 0.
            with patch.object(debug, 'upward_support', side_effect=AssertionError('Privileged contact feedback used')):
                c.select_gait('sliding')
                self.assertIsInstance(c.sliding, debug.MotorFeedbackSliding)
                phase = c.phase
                c.tick(.02)
                self.assertAlmostEqual(c.phase, phase)
                sim.data.time += .02; session.elapsed = sim.data.time
                c.mode = 'run'
                c.tick(.02)
                self.assertAlmostEqual(c.phase, phase+.012)
                self.assertEqual(sum(c.enabled), 6)
                c.command('window_auto')
                self.assertFalse(c.sliding.automatic)
                with self.assertRaises(ValueError):
                    c.command('reverse')
                with self.assertRaises(ValueError):
                    c.change_parameters(amplitude=62.)
                c.snapshot()
                c.select_gait('fixed')
                c.select_gait('basic')
                self.assertEqual((c.first, c.last), (5, 10))
        finally:
            session.close()


if __name__ == '__main__':
    unittest.main()
