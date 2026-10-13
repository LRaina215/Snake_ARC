"""Check phase coverage and rejection of held/stale samples, without physics."""
import math
import unittest
import numpy as np
from snake_control.phase_balanced_count import PhaseBalancedCount


class PhaseCountTests(unittest.TestCase):
    def test_periodic_count_peak_is_not_a_persistent_extra_joint(self):
        history = PhaseBalancedCount()
        decisions = []
        for i in range(80):
            phase = i*.12
            stats = history.update(7.2+.7*math.cos(2*phase), 174.-10*math.cos(2*phase), phase, i*.2)
            if stats['ready']: decisions.append(stats['balanced_count_joints'])
        self.assertTrue(decisions)
        self.assertLess(max(decisions), 7.5)
        self.assertLess(np.ptp(decisions), .2)

    def test_hold_and_feedback_gap_cannot_manufacture_phase_coverage(self):
        history = PhaseBalancedCount()
        for i in range(30):
            stats = history.update(7.8, 165., .2, i*.2)
        self.assertFalse(stats['ready'])
        self.assertEqual(stats['new_samples'], 1)
        stats = history.update(7.8, 165., .3, 7.)
        self.assertFalse(stats['ready'])
        self.assertEqual(stats['new_samples'], 1)


if __name__ == '__main__':
    unittest.main()
