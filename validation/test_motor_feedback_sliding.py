"""Small physics-independent checks of signal meaning and six-joint handover."""
import unittest
from dataclasses import replace
import numpy as np
from snake_control.rolling_arc import ArcParameters
from snake_control.motor_shape_observer import PhaseShapeReference, bending_features, local_phase
from snake_control.motor_feedback_sliding import MotorFeedbackSliding


class MotorSlidingTests(unittest.TestCase):
    def test_reverse_passive_fold_is_not_counted_as_main_arc(self):
        angles = np.zeros(20)
        angles[[8, 10, 12]] = 60.
        initial = bending_features(angles, [.0645]*21, (8, 13))
        angles[14] = -30.
        folded = bending_features(angles, [.0645]*21, (8, 13))
        self.assertAlmostEqual(initial['bending_center_joint'], folded['bending_center_joint'])
        self.assertAlmostEqual(folded['reverse_boundary_turn_deg'], 30.)

    def test_local_phase_parity_maps_the_same_local_angle_sequence(self):
        for phase in np.linspace(-5., 8., 17):
            p = local_phase(phase, 8)
            np.testing.assert_allclose([np.cos(p), np.sin(p)], [np.sin(phase), np.cos(phase)], atol=1e-14)
        reference = PhaseShapeReference([[0., 0., 0.]]*24, {})
        np.testing.assert_array_equal(reference.at(-.1, 8), reference.at(2*np.pi-.1, 8))

    def test_handover_keeps_six_and_preserves_the_five_shared_targets(self):
        p = ArcParameters(initial_phase_rad=np.pi/2, settle_s=0., omega_rad_s=0.)
        reference = PhaseShapeReference([[0., 0., 0.]]*24, {})
        controller = MotorFeedbackSliding(p, [.0645]*21, reference)
        angles = np.degrees(p.targets(np.pi/2))
        controller.pending = dict(to=[7, 12], time_s=0., ready_since=None, hold_since=None)
        before = controller.targets.copy()
        for t in np.arange(0., .24, .02):
            target, enabled = controller.update(angles, float(t))
            self.assertEqual(enabled.sum(), 6)
        self.assertEqual(controller.window, (7, 12))
        self.assertTrue(controller.enabled[6])
        self.assertFalse(controller.enabled[12])
        self.assertEqual([r['kind'] for r in controller.events].count('complete'), 1)
        np.testing.assert_allclose(target[7:12], before[7:12], atol=.02)
        self.assertFalse(controller.events[-1]['gate']['unloading_observed'])
        with self.assertRaises(ValueError):
            controller.update(angles, float(t))

    def test_feedback_gap_cannot_count_as_continuous_handover_readiness(self):
        p = ArcParameters(initial_phase_rad=np.pi/2, settle_s=0., omega_rad_s=0.)
        controller = MotorFeedbackSliding(p, [.0645]*21, PhaseShapeReference([[0., 0., 0.]]*24, {}))
        angles = np.degrees(p.targets(np.pi/2))
        controller.pending = dict(to=[7, 12], time_s=0., ready_since=None, hold_since=None)
        for t in np.arange(0., .16, .02):
            controller.update(angles, float(t))
        self.assertEqual(controller.window, (8, 13))
        controller.update(angles, 1.14)
        self.assertFalse(controller.last_gate['velocity_from_continuous_fresh_feedback'])
        self.assertEqual(controller.window, (8, 13))
        for t in (1.16, 1.18, 1.20):
            controller.update(angles, t)
            self.assertEqual(controller.window, (8, 13))
        controller.update(angles, 1.22)
        self.assertEqual(controller.window, (7, 12))


if __name__ == '__main__':
    unittest.main()
