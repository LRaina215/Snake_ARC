"""Focused travelling-profile and encoder-feedback contracts, not dynamics."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import unittest
import math
from dataclasses import replace
import numpy as np
from snake_control.continuous_feeding import ContinuousFeeding, ContinuousFeedingOptions
from snake_control.feeding_length_observer import FeedingLengthObserver


class ContinuousFeedingTests(unittest.TestCase):
    def test_many_cell_boundaries_preserve_plane_and_turn(self):
        c = ContinuousFeeding(np.full(21, .0645))
        for bias in (-.18, 0., .18):
            previous = None
            positions = np.linspace(-5.99, 5.99, 1201)
            # Bound the interpolation plus sin-squared allocation plus
            # smoothstep ramp derivatives, rather than an empirical constant
            # that omitted the new continuously varying allocation gain.
            opt = c.options
            slope_bound = opt.amplitude_deg*(1+abs(bias))/2 + opt.fractional_middle_boost_deg*(
                math.pi/2+1.5/opt.middle_boost_ramp_joints)
            step_bound = slope_bound*(positions[1]-positions[0])+1e-8
            for position in positions:
                q, focus, _, _ = c.profile(position, bias)
                self.assertAlmostEqual(float(q[::2].sum()), np.pi)
                self.assertTrue(np.all(q[1::2] == 0.))
                self.assertLess(float(abs(q).max()), np.pi/2)
                self.assertEqual(int(c.helper.mask(focus).sum()), 7)
                if previous is not None:
                    self.assertLess(float(np.degrees(abs(q-previous)).max()), step_bound)
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

    def test_release_requires_persistent_low_aligned_geometry(self):
        c = ContinuousFeeding(np.full(21, .0645))
        q = np.degrees(c.targets)
        c.observation = c.length_observer.observe(q, c.origin)
        self.assertTrue(c.observation['valid'])
        c.release_below_peak_m = c.observation['cut_depth_m']+.03225
        required = c.helper.mask(c.origin)
        c.enabled[2] = True  # Straight distal J3, below the upper arc.
        self.assertFalse(c.release_ready(q, required, 0., True)[2])
        self.assertFalse(c.release_ready(q, required, .2, True)[2])
        self.assertTrue(c.release_ready(q, required, .4, True)[2])
        tilted = q.copy(); tilted[0] = 30.
        self.assertFalse(c.release_ready(tilted, required, .6, True)[2])
        self.assertTrue(np.isnan(c.release_since[2]))
        c.enabled[7] = True  # J8 is zero angle, but still at the upper arc.
        self.assertFalse(c.release_ready(q, np.zeros(c.n, bool) | c.helper.mask((9, 13)), .8, True)[7])

    def test_fold_retention_cannot_consume_both_passive_references(self):
        for direction in (-1., 1.):
            c = ContinuousFeeding(np.full(21, .0645), replace(
                ContinuousFeedingOptions(), requested_transfer_m=direction*.258))
            q = np.degrees(c.targets)
            c.advance(q, 6., 6.)
            c.fold_capture_position[:] = 0.
            _, mask = c.advance(q, 6.2, 6.2)
            first, last = c.window
            self.assertGreaterEqual(first-1 if direction>0 else c.n-last, 3)
            self.assertTrue(np.any(~mask))
            self.assertTrue(c.length_observer.observe(q, c.window)['valid'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
