"""Focused travelling-profile and encoder-feedback contracts, not dynamics."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import unittest
import numpy as np
from snake_control.continuous_feeding import ContinuousFeeding
from snake_control.feeding_length_observer import FeedingLengthObserver


class ContinuousFeedingTests(unittest.TestCase):
    def test_many_cell_boundaries_preserve_plane_and_turn(self):
        c = ContinuousFeeding(np.full(21, .0645))
        for bias in (-.18, 0., .18):
            previous = None
            for position in np.linspace(-5.99, 5.99, 1201):
                q, focus, _, _ = c.profile(position, bias)
                self.assertAlmostEqual(float(q[::2].sum()), np.pi)
                self.assertTrue(np.all(q[1::2] == 0.))
                self.assertLess(float(abs(q).max()), np.pi/2)
                self.assertEqual(int(c.helper.mask(focus).sum()), 7)
                if previous is not None:
                    self.assertLess(float(np.degrees(abs(q-previous)).max()), .36)
                previous = q

    def test_length_difference_is_not_a_window_count(self):
        c = ContinuousFeeding(np.full(21, .0645))
        q = np.degrees(c.targets)
        observer = FeedingLengthObserver(c.lengths)
        values = [observer.observe(q, window) for window in ((8, 13), (7, 13), (8, 14))]
        self.assertTrue(all(v['valid'] for v in values))
        coordinates = [v['transfer_coordinate_m'] for v in values]
        self.assertLess(max(coordinates)-min(coordinates), 1e-10)
        self.assertGreater(max(v['above_cut_length_m'] for v in values)-min(v['above_cut_length_m'] for v in values), .01)

    def test_encoder_timestamp_contract(self):
        c = ContinuousFeeding(np.full(21, .0645))
        q = np.degrees(c.targets)
        c.advance(q, 0., 0.); c.advance(q, 0., .02)
        with self.assertRaisesRegex(ValueError, 'same feedback'):
            c.advance(q+1., 0., .04)
        with self.assertRaisesRegex(ValueError, 'Current'):
            c.advance(q, 0., .6)
        with self.assertRaisesRegex(ValueError, 'Increasing'):
            c.advance(q, 0., .02)

    def test_untrusted_length_finishes_shape_and_has_bounded_wait(self):
        c = ContinuousFeeding(np.full(21, .0645))
        q = np.degrees(c.targets)
        c.advance(q, 6., 6.)
        c.stage = 'feed_return'; c.stage_since = 6.
        c.position = -1.97
        c.enabled = c.helper.mask((7, 13)); c.window = (7, 13)
        c.targets[12] = np.radians(-39.)
        q[12] = -39.
        c.length_observer.observe = lambda angles, window: dict(valid=False, reason='temporarily_untrusted')
        c.helper.position = lambda angles, window: dict(valid=False)
        target, mask = c.advance(q, 6.2, 6.2)
        self.assertEqual(c.position, -1.97)
        self.assertGreater(target[12], np.radians(-39.))
        self.assertTrue(mask[12])
        c.advance(q, 9.22, 9.22)
        self.assertIn('did not recover', c.fault)


if __name__ == '__main__':
    unittest.main(verbosity=2)
