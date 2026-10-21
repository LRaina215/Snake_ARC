"""Explicit Ctrl+L commands against the recorded 8.122 degree stall."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from snake_control.continuous_feeding import ContinuousFeeding, ContinuousFeedingOptions
from validation.test_repeated_motor_port import AngleBus, real, debug

DATA = json.loads((Path(__file__).parent / 'reports/feeding_forced_step_20261021/fixture.json').read_text(encoding='utf8'))


def restore(core):
    f = DATA['feeding']
    core.position, core.bias = f['position_joints'], f['bearing_bias']
    core.targets = np.radians(DATA['target_deg'])
    core.enabled = np.array(DATA['enabled'], bool)
    core.window = tuple(f['window'])
    core.stage, core.stage_since = 'acquire', 0.
    core.observation = f['observation']
    core.tracking_error_deg = f['tracking_error_deg']


class ForcedStepTests(unittest.TestCase):
    def test_stalled_real_and_debug_each_press_moves_once_then_holds(self):
        for cls in (real.RollingArcController, debug.DebugArcController):
            session = SimpleNamespace(elapsed=0., sim=SimpleNamespace(p=real.ManualArcReference()),
                                      stop_push=lambda: None)
            bus = AngleBus()
            bus.angles = np.array(DATA['angles_deg'])  # Deliberately never follows commands.
            bus.enabled = np.array(DATA['enabled'])
            c = cls(bus) if cls is real.RollingArcController else cls(bus, session)
            c.motor_sample_clock = lambda: (session.elapsed, session.elapsed)
            core = real.ContinuousFeeding(real.LINK_LENGTHS_M, real.ContinuousFeedingOptions(**DATA['feeding']['options']))
            restore(core)
            core.fault = 'takeover shape/length did not settle; no transfer baseline recorded'
            c.feeding, c.feeding_selected = core, True
            c.feeding_clock_origin = 0.
            c.targets, c.enabled = list(DATA['target_deg']), list(DATA['enabled'])
            c.first, c.last = core.window
            c.mode, c.feeding_paused = 'hold', True
            start = core.position
            with contextlib.redirect_stdout(io.StringIO()):
                for count in range(1, 4):
                    c.command('feed_tick')
                    self.assertAlmostEqual(core.position, start+count*real.FEED_FORCE_STEP_JOINTS)
                    for _ in range(150):
                        session.elapsed += .02
                        c.refresh_compliance_feedback()
                        c.tick(.02)
                        if c.feeding_paused:break
                    self.assertTrue(c.feeding_paused)
                    self.assertIsNone(core.forced_step)
                    self.assertIsNone(core.baseline)
                    self.assertIsNone(core.fault)
                    held = c.targets.copy()
                    for _ in range(5):
                        session.elapsed += .02;c.tick(.02)
                    self.assertEqual(c.targets, held)
                self.assertFalse(np.allclose(c.targets, DATA['target_deg']))
                self.assertEqual(sum(e['kind']=='forced_step_targets_sent' for e in core.events), 3)
                c.command('back')  # J/H still start acquire; H sets negative direction.
                self.assertIs(c.feeding, core)
                self.assertEqual(core.stage, 'acquire')
                self.assertLess(core.options.requested_transfer_m, 0.)
                p = core.position
                c.command('feed_tick')
                self.assertAlmostEqual(core.position, p-real.FEED_FORCE_STEP_JOINTS)
                c.command('release')
                c.tick(.02)
                self.assertFalse(bus.enabled.any())

    def test_force_bypasses_unobservable_shape_and_matches_embedded_targets(self):
        pair = [ContinuousFeeding(np.full(21,.0645), ContinuousFeedingOptions(**DATA['feeding']['options'])),
                real.ContinuousFeeding(real.LINK_LENGTHS_M, real.ContinuousFeedingOptions(**DATA['feeding']['options']))]
        for c in pair:
            restore(c)
            c.start_forced_step(DATA['angles_deg'], .05, 0.)
        old = pair[0].targets.copy()
        with patch.object(pair[0].length_observer, 'observe', return_value={'valid':False,'reason':'no_stable_passive_segment'}), \
             patch.object(pair[1].length_observer, 'observe', return_value={'valid':False,'reason':'no_stable_passive_segment'}):
            for i in range(100):
                a, am = pair[0].advance(DATA['angles_deg'], i*.02, i*.02)
                b, bm = pair[1].advance(DATA['angles_deg'], i*.02, i*.02)
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
                np.testing.assert_array_equal(am, bm)
                self.assertLessEqual(float(np.max(abs(np.degrees(a-old)))), .900001)
                old = a.copy()
                if pair[0].forced_step is None:break
        self.assertIsNone(pair[0].forced_step)
        self.assertIsNone(pair[0].baseline)

    def test_return_direction_and_template_boundary(self):
        c = ContinuousFeeding(np.full(21,.0645), ContinuousFeedingOptions())
        c.stage = 'feed_return'  # Negative outbound request means positive return.
        self.assertTrue(c.start_forced_step(np.degrees(c.targets), .05, 0.))
        self.assertAlmostEqual(c.position, .05)
        c.forced_step = None
        c.position = c.position_bounds[1]
        before = c.targets.copy()
        self.assertFalse(c.start_forced_step(np.degrees(c.targets), .05, .1))
        np.testing.assert_array_equal(c.targets, before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
