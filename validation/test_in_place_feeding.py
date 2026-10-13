"""Focused software contracts; ideal encoder feedback is not a physics test."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import unittest
import numpy as np
from snake_control.in_place_feeding import InPlaceFeeding, FeedingOptions


class FeedingTests(unittest.TestCase):
    def test_both_directions_exchange_real_bend_axes_without_rolling(self):
        for direction in (-1, 1):
            c = InPlaceFeeding(np.full(21, .0645), FeedingOptions(direction=direction))
            # Check that either direction moves an actual bending hinge,
            # rather than merely sliding the mask over a zero-angle axis.
            old = c.helper.template(c.old, 0., 60.)
            new = c.helper.template(c.new, 0., 60.)
            self.assertEqual(np.count_nonzero(abs(new-old)>1e-8), 2)
            self.assertEqual(int(c.wave_mask.sum()), 7)
            c.helper.position = lambda angles, focus: dict(valid=True, crest_joint=11.)
            c.advance(np.degrees(old), 6., 6.)
            last = c.targets.copy()
            for i in range(1, 61):
                target, mask = c.advance(np.degrees(last), 6.+i*.02, 6.+i*.02)
                self.assertEqual(c.phase, 0.)
                self.assertLessEqual(float(np.degrees(abs(target[mask]-last[mask])).max()), .9+1e-8)
                self.assertAlmostEqual(float(target[mask & (np.arange(20)%2 == 0)].sum()), np.pi)
                self.assertTrue(np.allclose(target[mask & (np.arange(20)%2 == 1)], 0.))
                last = target
            self.assertGreater(c.progress, 0.)

    def test_review_keeps_feedback_active(self):
        c = InPlaceFeeding(np.full(21, .0645))
        q = np.degrees(c.targets)
        c.helper.position = lambda angles, focus: dict(valid=True, crest_joint=11.)
        c.advance(q, 6., 6.)
        c.stage = 'review_out'; c.stage_since = 6.; c.progress = .5
        before = c.progress
        c.advance(q, 6.02, 6.02)
        self.assertGreater(c.progress, before)
        self.assertEqual(c.stage, 'review_out')

    def test_encoder_clock_contract(self):
        c = InPlaceFeeding(np.full(21, .0645)); q = np.degrees(c.targets)
        c.advance(q, 0., 0.); c.advance(q, 0., .02)
        with self.assertRaisesRegex(ValueError, 'same feedback'):
            c.advance(q+1., 0., .04)
        with self.assertRaisesRegex(ValueError, 'current'):
            c.advance(q, 0., .6)
        with self.assertRaisesRegex(ValueError, 'Increasing'):
            c.advance(q, 0., .02)


if __name__ == '__main__':
    unittest.main(verbosity=2)
