"""Recorded mode-4 release lifecycle; no hardware or new dynamics."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from snake_control.continuous_feeding import ContinuousFeeding, ContinuousFeedingOptions
from validation.test_active_feeding_port import real


ROWS = {r['line']: r for r in json.loads((Path(__file__).parent /
    'reports/feeding_release_memory_20261021/fixture.json').read_text(encoding='utf8'))['rows']}


class ReleasedFoldTests(unittest.TestCase):
    def make_core(self, embedded=False):
        before = ROWS[4414]
        f = before['feeding']
        cls = real.ContinuousFeeding if embedded else ContinuousFeeding
        opts = real.ContinuousFeedingOptions if embedded else ContinuousFeedingOptions
        c = cls(np.full(21, .0645), opts(**f['options']))
        c.stage = 'feed_out'
        c.stage_since = before['motor_input']['control_time_s']
        c.position, c.bias = f['position_joints'], f['bearing_bias']
        c.enabled = np.array(before['enabled'], bool)
        c.targets = np.radians(before['target_deg'])
        c.window = tuple(f['window'])
        c.reference_direction = 1  # Recorded passive reference is head.
        c.baseline, c.goal_m = f['baseline_m'], f['goal_m']
        c.velocity_m_s = f['velocity_m_s']
        c.release_below_peak_m = f['release_below_peak_m']
        c.last_time = before['motor_input']['control_time_s']
        # Reconstruct the last J17 capture and a release-ready timer. The
        # log does not serialize these arrays, so they are explicit setup.
        c.fold_capture_position[16] = ROWS[3007]['feeding']['position_joints']
        c.release_since[15:17] = ROWS[4415]['motor_input']['feedback_time_s'] - .5
        return c

    def release(self, c):
        row = ROWS[4415]
        inp = row['motor_input']
        # Freeze the recorded observer output to isolate enable/release
        # bookkeeping from observer warm-up, not to fake a dynamics run.
        with patch.object(c.length_observer, 'observe', return_value=row['feeding']['observation']):
            return c.advance(inp['angles_deg'], inp['feedback_time_s'], inp['control_time_s'])

    def test_real_release_clears_capture_and_does_not_revive_on_recorded_positions(self):
        research, embedded = self.make_core(), self.make_core(True)
        a, am = self.release(research)
        b, bm = self.release(embedded)
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(am, bm)
        for c in (research, embedded):
            self.assertFalse(c.enabled[15:17].any())
            self.assertTrue(np.isnan(c.fold_capture_position[16]))
            self.assertTrue(np.isnan(c.release_since[15:17]).all())
            for line in (4433, 4453, 4467, 4493, 4509, 4532):
                row = ROWS[line]
                f = row['feeding']
                c.observation = f['observation']
                _, focus, _, _ = c.profile(f['position_joints'], f['bearing_bias'])
                required = c.required_mask(focus, np.array(row['motor_input']['angles_deg']), f['position_joints'])
                self.assertFalse(required[15:17].any(), line)

    def test_new_nearby_bend_can_still_request_takeover_after_release(self):
        c = self.make_core()
        self.release(c)
        # Earlier recorded shape has J17 inside the current arc's bend
        # neighbourhood. Clearing history must not become a permanent ban.
        row = ROWS[3007]
        f = row['feeding']
        c.observation = f['observation']
        _, focus, _, _ = c.profile(f['position_joints'], f['bearing_bias'])
        required = c.required_mask(focus, np.array(row['motor_input']['angles_deg']), f['position_joints'])
        self.assertTrue(required[16])
        self.assertEqual(c.fold_capture_position[16], f['position_joints'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
