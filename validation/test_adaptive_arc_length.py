"""Focused budget/transaction tests, no physics or hardware."""
import math
import unittest
from dataclasses import replace
import numpy as np
from snake_control.adaptive_arc_length import HalfArcParameters, AdaptiveArcLength
from snake_control.motor_shape_observer import PhaseShapeReference
from snake_control.fixed_section_compliance import FixedSectionCompliance
from validation.audit_arc_length import exit_review


class ActiveLengthTests(unittest.TestCase):
    def test_exit_counter_uses_last_continuous_suffix_and_last_change(self):
        rows = [dict(time_s=i, whole_body_min_route_m=value,
                     controller=dict(phase_rad=i*math.pi, pending=None), enabled=[True]*6)
                for i, value in enumerate((0., 2., 0., 2., 2., 2.))]
        status = dict(scene_geometry=dict(feature_span_m=[0., 1.]),
                      events=[dict(kind='resize_complete', time_s=4.)])
        result = exit_review(rows, status)
        self.assertEqual(result['continuous_exit_start_s'], 3.)
        self.assertEqual(result['continuous_exit_cycles'], 1.)
        self.assertEqual(result['post_last_change_cycles'], .5)

    def test_each_family_keeps_pi_budget_for_six_seven_eight(self):
        for first, last in ((8, 13), (7, 13), (8, 14), (7, 14)):
            p = HalfArcParameters(first_joint=first, last_joint=last)
            for phase in np.linspace(0., 2*np.pi, 24, endpoint=False):
                target = p.targets(phase)
                self.assertAlmostEqual(target[::2].sum(), np.pi*np.cos(phase))
                self.assertAlmostEqual(target[1::2].sum(), np.pi*np.sin(phase))
                self.assertTrue(FixedSectionCompliance(p, [.0645]*21)._check_goal(target)[0])

    def test_resize_requires_two_fresh_samples_and_changes_one_boundary(self):
        p = HalfArcParameters(settle_s=0.)
        ref = PhaseShapeReference([[0., 0., 0.]]*24, {})
        c = AdaptiveArcLength(p, [.0645]*21, {6: ref, 8: ref}, resize=False)
        def valid_observer():
            c.observer.update = lambda *args: dict(valid=True, direction=0, suggested_window=None)
        valid_observer()
        actual = np.degrees(p.targets(0.))
        c.length_observation = dict(state='unit-test executor request')
        c._request_resize(8, 0., actual)
        for tick in range(20):
            t = tick*.02
            c.advance(actual, 0. if tick < 10 else .2, t, commanded_phase_rad=0., phase_at_feedback=0.)
            self.assertEqual(c.window, (8, 13))
        c.advance(actual, .4, .4, commanded_phase_rad=0., phase_at_feedback=0.)
        self.assertEqual(c.window, (8, 14))
        self.assertEqual(c.enabled.sum(), 7)
        self.assertEqual(c.pending['to'], [7, 14])
        self.assertEqual([e['kind'] for e in c.events].count('resize_step'), 1)
        valid_observer()
        phase = np.pi/2
        current = replace(p, first_joint=8, last_joint=14)
        actual = np.degrees(current.targets(phase))
        c.targets = c.core.targets = np.radians(actual)
        for t in (.6, .8, 1., 1.2):
            c.advance(actual, t, t, commanded_phase_rad=phase, phase_at_feedback=phase)
        self.assertEqual(c.window, (7, 14))
        self.assertEqual(c.enabled.sum(), 8)
        self.assertIsNone(c.pending)
        self.assertEqual([e['kind'] for e in c.events].count('resize_complete'), 1)
        self.assertFalse(c.last_gate['unloading_observed'])


if __name__ == '__main__':
    unittest.main()
