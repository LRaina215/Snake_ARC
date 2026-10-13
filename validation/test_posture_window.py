"""Mechanical and measured-input regressions; no new physics or hardware."""
import gzip
import json
import math
import unittest
import numpy as np
from snake_control.adaptive_arc_length import HalfArcParameters
from snake_control.posture_window import PostureWindow, mechanically_feasible, nominal_angles
from snake_control.simulation import ROOT
from validation.posture_window_experiment import references


class PostureWindowTests(unittest.TestCase):
    def test_uncalibrated_odd_widths_have_full_cycle_open_mechanically_feasible_references(self):
        for width in (7, 9):
            p = HalfArcParameters(first_joint=6, last_joint=5+width)
            c = PostureWindow(p, [.0645]*21, references())
            self.assertTrue(mechanically_feasible(c.lengths, c.window))
            self.assertIsNone(c.snapshot()['stable_width_whitelist'])
            for phase in np.linspace(0, 2*math.pi, 24, endpoint=False):
                q = nominal_angles(c.lengths, c.window, phase)
                good, _, _ = c.observer.geometry_observer.geometry(q, c.window)
                self.assertTrue(good)
                self.assertTrue(np.isfinite(c.observer.reference.at(phase, c.window[0])).all())
        self.assertFalse(mechanically_feasible([.0645]*21, (6, 10)))

    def test_measured_acute_boundary_can_become_stable_seven_without_hold_or_shared_jump(self):
        path = ROOT/'validation/reports/rolling_handover/combined_repair_20261013_120622_382379/actual_trace.jsonl.gz'
        wanted = {}
        with gzip.open(path, 'rt', encoding='utf8') as stream:
            for line in stream:
                row = json.loads(line)
                if round(row['time_s'], 2) in (25.2, 25.4):
                    wanted[round(row['time_s'], 2)] = row
                if row['time_s'] > 25.41:
                    break
        before, switch = wanted[25.2], wanted[25.4]
        p = HalfArcParameters(first_joint=7, last_joint=12, settle_s=0.)
        c = PostureWindow(p, [.0645]*21, references())
        c.advance(np.degrees(before['actual_qpos'][7:]), 25.2, 25.2,
                  commanded_phase_rad=before['controller']['phase_rad']+.012)
        c.targets = np.array(switch['target_rad']); c.core.targets = c.targets.copy()
        c.pending = dict(kind='resize', operation='expand', to=[7, 13], time_s=25.2, evidence={})
        c.advance(np.degrees(switch['actual_qpos'][7:]), 25.4, 25.4,
                  commanded_phase_rad=switch['controller']['phase_rad']+.012)
        self.assertEqual(c.window, (7, 13))
        self.assertEqual(int(c.enabled.sum()), 7)
        self.assertFalse(c.phase_hold)
        self.assertEqual(c.last_gate['shared_preload_jump_rad'], 0.)
        self.assertFalse(c.last_gate['unloading_observed'])


if __name__ == '__main__':
    unittest.main()
