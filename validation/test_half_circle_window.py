"""Focused geometry/regression checks; no hardware or new dynamics runs."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import copy
import json
import math
import unittest
from dataclasses import replace
import numpy as np
from codes import Rolling_Arc_Real as real
from snake_control.phase_balanced_count import PhaseBalancedCount
from snake_control.posture_window import PostureWindow
from snake_control.fixed_section_compliance import FixedSectionCompliance
from validation.half_circle_experiment import MeasuredParameters
from validation.posture_window_experiment import references


class HalfCircleTests(unittest.TestCase):
    def test_reverse_phase_accumulates_and_hold_does_not(self):
        results = []
        for sign in (-1, 1):
            c = PhaseBalancedCount()
            for i, phase in enumerate(np.linspace(0, 3.13, 40)):
                r = c.update(5., 210., sign*phase, .2*i)
            self.assertTrue(r['ready'])
            self.assertAlmostEqual(r['represented_phase_span_rad'], 3.13)
            results.append(r['balanced_count_joints'])
            # Reversing direction resets once; holding cannot add coverage.
            r = c.update(5., 210., sign*3.0, 8.)
            self.assertFalse(r['ready'])
            for i in range(10):
                r = c.update(5., 210., sign*3.0, 8.2+i*.2)
            self.assertEqual(r['new_samples'], 1)
        self.assertEqual(results[0], results[1])

    def test_count_expands_contracts_independent_of_initial_amplitude(self):
        for actual_amplitude, operation in ((45., 'expand'), (75., 'contract')):
            decisions = []
            for initial_amplitude in (35., 60., 85.):
                p = MeasuredParameters(amplitude_deg=initial_amplitude)
                c = PostureWindow(p, p.mechanical_lengths_m, references())
                for i, phase in enumerate(np.linspace(0, -3.13, 40)):
                    angles = np.degrees(replace(p, amplitude_deg=actual_amplitude).targets(phase))
                    c.feedback_phase, c.last_time = phase, i*.2
                    c.observation = c.observer.update(angles, phase, c.enabled, i*.2)
                    c._observe_count(angles, i*.2)
                    c._select(angles, i*.2)
                self.assertIsNotNone(c.pending)
                self.assertEqual(c.pending['operation'], operation)
                decisions.append((c.filtered_count, c.pending['to']))
            self.assertEqual(decisions[0], decisions[1])
            self.assertEqual(decisions[1], decisions[2])

    def test_discrete_inverse_preserves_actual_curve_contraction(self):
        p = MeasuredParameters()
        for phase in (0., .7, 1.57):
            q = p.targets(phase)
            c = FixedSectionCompliance(p, p.mechanical_lengths_m)
            c.update(q, phase, 0., .02)
            d = c.inverse_diagnostic
            self.assertFalse(d['nominal_used'])
            self.assertLess(d['residual_deg'], .2)
            other = FixedSectionCompliance(replace(p, amplitude_deg=85.), p.mechanical_lengths_m)
            other.update(q, phase, 0., .02)
            np.testing.assert_allclose(c.last_goal, other.last_goal)

    def test_real_record_intrinsic_recommendation_can_complete_exact_slide(self):
        r = json.loads((ROOT/'validation/reports/half_circle_window/recommendation_fixture.json').read_text(encoding='utf8'))
        for controller_type in (PostureWindow, real.PostureWindow):
            p = MeasuredParameters()
            c = controller_type(p, p.mechanical_lengths_m, references())
            c.observation = copy.deepcopy(r['observation'])
            angles = np.array([r['measured_controller_deg'][f'J{i}'] for i in range(1, 21)])
            self.assertNotIn('down_head', c.observation)
            self.assertEqual(c.observation['suggested_window'], [7, 12])
            self.assertIsNone(c._position_evidence((8, 14)))
            c._select(angles, 0.)
            c._select(angles, 1.01)
            self.assertEqual(c.pending['to'], [7, 12])
            c.targets = np.radians(angles)
            c.phase = r['phase_rad']
            c.fresh_feedback = c.angle_velocity_available = True
            c.angle_velocity_deg_s = np.zeros(20)
            c._handover(angles, 1.02)
            self.assertEqual(c.window, (7, 12), c.last_gate)
            np.testing.assert_array_equal(np.flatnonzero(c.enabled)+1, np.arange(7, 13))
            self.assertFalse(c.last_gate['unloading_observed'])
            self.assertEqual(c.last_gate['waiting_reasons'], [])
            # Removing corroborated recommendation must not authorise a blind slide.
            c.observation = dict(valid=True, position_valid=True, state='retain')
            self.assertIsNone(c._position_evidence())


if __name__ == '__main__':
    unittest.main(verbosity=2)
