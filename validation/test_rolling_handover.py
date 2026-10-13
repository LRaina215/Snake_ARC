"""Two specific regressions; no physics or hardware connection."""
import json
import math
import unittest
import numpy as np
from snake_control.simulation import ROOT
from snake_control.adaptive_arc_length import HalfArcParameters
from snake_control.rolling_handover import RollingHandover
from snake_control.rolling_shape_observer import RollingShapeObserver
from snake_control.motor_shape_observer import PhaseShapeReference


def reference():
    return PhaseShapeReference.from_dict(json.loads((ROOT/'configs/motor_shape_reference.json').read_text()))


class MovingExchangeTests(unittest.TestCase):
    def test_moving_exit_above_old_velocity_limit_can_exchange_without_phase_hold(self):
        p = HalfArcParameters(settle_s=0., initial_phase_rad=math.pi/2-.12)
        c = RollingHandover(p, [.0645]*21, {6: reference()}, resize=False, sliding=False)
        c.observer.update = lambda *a: dict(valid=True, direction=-1, suggested_window=None)
        c.advance(np.degrees(p.targets(p.initial_phase_rad)), 0., 0., commanded_phase_rad=p.initial_phase_rad)
        old = c.targets.copy()
        c.pending = dict(kind='slide', to=[7, 12], time_s=0., ready_since=None, hold_since=None)
        c.advance(np.degrees(p.targets(math.pi/2)), .2, .2, commanded_phase_rad=math.pi/2)
        self.assertEqual(c.window, (7, 12))
        self.assertFalse(c.phase_hold)
        self.assertEqual(c.enabled.sum(), 6)
        self.assertGreater(abs(c.last_gate['exiting_velocity_deg_s'][0]), 8.)
        self.assertEqual(c.last_gate['shared_preload_jump_rad'], 0.)
        self.assertLessEqual(np.degrees(abs(c.targets[7:12]-old[7:12])).max(), .9+1e-8)
        self.assertFalse(c.last_gate['unloading_observed'])

    def test_single_short_end_can_use_the_qualified_long_end(self):
        p = HalfArcParameters(first_joint=2, last_joint=7)
        o = RollingShapeObserver([.0645]*21, reference())
        r = o.update(np.degrees(p.targets(0.)), 0., p.enabled, 0.)
        self.assertTrue(r['valid'])
        self.assertFalse(r['observes_absolute_down'])
        self.assertEqual(r['end_quality']['source'], 'single_tail')
        self.assertTrue(r['end_quality']['does_not_prove_gravity'])


if __name__ == '__main__':
    unittest.main()
