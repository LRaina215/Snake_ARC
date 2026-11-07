"""Curve joins and motor-only intake; these checks do not establish physical support."""
import unittest

import numpy as np

from validation.test_front_confirmation import recorded_core, real
from validation.progressive_tail_candidate import ProgressiveHelixTail
from validation import progressive_tail_candidate as experimental
from snake_control.gravity_moment import GravityMomentGait, GravityMomentOptions


class ProgressiveTailTests(unittest.TestCase):
    def test_curve_preserves_measured_tail_at_entry_and_finishes_as_helix(self):
        g = GravityMomentGait(18, GravityMomentOptions(r_h1=.094), .0645)
        actual = recorded_core(experimental).measured
        curve = ProgressiveHelixTail(g, .094, .4, 9, actual)
        lh, la, ls = curve.geometry(0.)
        np.testing.assert_allclose(curve.targets(0.)[9:], actual[9:], atol=1e-12)
        np.testing.assert_allclose(curve.targets(curve.shift_limit_m), g.targets(.4, .094)[0])
        self.assertAlmostEqual(lh+la+ls, g.L)
        self.assertEqual(int(curve.enabled(0.).sum()), 11)
        old = 11
        for shift in np.linspace(0., curve.shift_limit_m, 1001):
            lh, la, ls = curve.geometry(shift)
            self.assertAlmostEqual(lh+la+ls, g.L)
            mask = curve.enabled(shift)
            active = int(mask.sum())
            self.assertGreaterEqual(active, old)
            self.assertLessEqual(active-old, 1)
            # Two-link integration plus the short tangent arc spans at most
            # three hinges beyond the fully helical prefix.
            self.assertLessEqual(active-curve.helix_prefix(shift), 3)
            self.assertTrue(np.all(mask[:active]))
            self.assertFalse(np.any(mask[active:]))
            old = active

    def test_slow_entering_joint_holds_the_curve_join_and_leaves_distal_tail_passive(self):
        core = recorded_core(experimental)
        core.confirm(0., np.degrees(core.measured))
        core.running = True
        core.measured[9:11] = core.curve.targets(0.)[9:11]+np.radians(30.)
        core.advance(.02, .02, .02, np.degrees(core.measured))
        self.assertEqual(core.offset, 0.)
        self.assertEqual(core.tail_wait_reason, 'tail_curve_tracking')
        self.assertEqual(int(core.enabled.sum()), 11)
        self.assertFalse(core.enabled[11:].any())
        self.assertLessEqual(float(np.max(abs(core.targets[9:11]-core.measured[9:11]))), np.radians(.9)+1e-10)

    def test_continuous_intake_reaches_completion_without_more_keys_and_stops(self):
        core = recorded_core(experimental)
        core.confirm(0., np.degrees(core.measured))
        locked = core.targets[:9].copy()
        core.running = True  # Ctrl+G once, not repeated Ctrl+J.
        previous_active = 9
        for step in range(1, 2001):
            measured = np.degrees(core.targets)
            old = core.targets.copy()
            core.advance(.02, step*.02, step*.02, measured)
            active = int(core.enabled.sum())
            self.assertLessEqual(active-previous_active, 2 if previous_active == 9 else 1)
            self.assertLessEqual(float(np.max(abs(core.targets-old))), np.radians(.9)+1e-10)
            np.testing.assert_array_equal(core.targets[:9], locked)
            previous_active = active
            if core.stage == 'complete':
                break
        self.assertEqual(core.stage, 'complete')
        self.assertFalse(core.running)
        self.assertLess(step*.02, 25.)


if __name__ == '__main__':
    unittest.main()
