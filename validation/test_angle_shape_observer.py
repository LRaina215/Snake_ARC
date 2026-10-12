"""Two meaningful software checks; no simulated or real actuator execution."""
import unittest
from unittest.mock import patch
import numpy as np
from snake_control.angle_shape_observer import AngleShapeObserver
from snake_control.section_geometry import Segment


class AngleObserverTests(unittest.TestCase):
    def test_same_geometric_curve_has_same_crest_when_a_short_piece_is_split_off(self):
        observer = AngleShapeObserver([.0645] * 21)
        points = np.column_stack([np.arange(22) * .0645, np.zeros(22), np.zeros(22)])
        section = {'axis': {'point': np.zeros(3), 'basis': np.eye(3)}}
        whole = [Segment(np.zeros(2), np.array([1., 0.]), .0645, 0., 0., 1, 'C')]
        split = [Segment(np.zeros(2), np.array([1., 0.]), .0001, 0., 0., 1, 'C'),
                 Segment(np.array([.0001, 0.]), np.array([1., 0.]), .0644, 0., 0., 1, 'C')]
        with patch('snake_control.angle_shape_observer.continuous_section', return_value=whole):
            first = observer.crest(points, (8, 13), section, np.array([0., -1., 0.]))
        with patch('snake_control.angle_shape_observer.continuous_section', return_value=split):
            second = observer.crest(points, (8, 13), section, np.array([0., -1., 0.]))
        self.assertAlmostEqual(first['joint_coordinate'], second['joint_coordinate'], places=8)
        self.assertAlmostEqual(first['joint_coordinate'], 7., places=8)

    def test_persistent_measured_arc_offset_only_recommends_and_never_moves_input_mask(self):
        observer = AngleShapeObserver([.0645] * 21)
        enabled = np.zeros(20, bool)
        enabled[7:13] = True
        before = enabled.copy()
        q = np.zeros(20)
        q[[8, 10, 12]] = 60.
        normal = observer.update(q, 0., enabled, 0.)
        self.assertTrue(normal['valid'])
        self.assertEqual(normal['direction'], 0)
        # A different actual open U: first bend reaches 90 degrees earlier.
        q[[8, 10, 12]] = [90., 70., 20.]
        rows = [observer.update(q, 0., enabled, .02 * k) for k in range(1, 71)]
        self.assertTrue(any(r['suggested_window'] == [7, 12] for r in rows))
        self.assertFalse(any(r['executes_window_update'] for r in rows))
        self.assertTrue(all(r['current_window'] == [8, 13] for r in rows))
        np.testing.assert_array_equal(enabled, before)

    def test_unreliable_passive_shape_and_stale_feedback_cannot_confirm_a_shift(self):
        observer = AngleShapeObserver([.0645] * 21)
        enabled = np.zeros(20, bool)
        enabled[7:13] = True
        q = np.zeros(20)
        q[[8, 10, 12]] = [90., 70., 20.]
        observer.update(q, 0., enabled, 0.)
        q[0] = 60.  # Passive end folds, invalidating the natural-hanging assumption.
        bad = observer.update(q, 0., enabled, .02)
        self.assertFalse(bad['valid'])
        self.assertEqual(bad['direction'], 0)
        self.assertIsNone(bad['suggested_window'])
        self.assertTrue(bad['reasons'])
        with self.assertRaises(ValueError):
            observer.update(q, 0., enabled, .02)


if __name__ == '__main__':
    unittest.main()
