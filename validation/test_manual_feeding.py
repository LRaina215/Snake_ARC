"""Focused log replay and manual-action checks. No hardware or dynamics."""
import contextlib
import io
import gzip
import json
import unittest
from dataclasses import replace
import numpy as np
from validation.test_active_feeding_port import ready, ROOT, real
from snake_control.continuous_feeding import ContinuousFeeding, ContinuousFeedingOptions


class ManualFeedingTests(unittest.TestCase):
    def test_task19_straightening_is_before_baseline(self):
        selected = {}
        path = ROOT/'validation/reports/passive_segment_selection_20261021/actual_log.jsonl.gz'
        with gzip.open(path,'rt',encoding='utf8') as stream:
            for line, text in enumerate(stream, 1):
                if line in (9384, 9386, 9390, 9394, 9399):
                    selected[line] = json.loads(text)
                if line >= 9399:
                    break
        c = ContinuousFeeding(np.full(21,.0645), replace(ContinuousFeedingOptions(),
            preparation_s=.4, requested_transfer_m=.0645, automatic_return=False))
        embedded = real.ContinuousFeeding(c.lengths,
            real.ContinuousFeedingOptions(**c.options.__dict__))
        def advance(angles, t):
            targets, mask = c.advance(angles,t,t)
            actual, enabled = embedded.advance(angles,t,t)
            np.testing.assert_allclose(actual,targets,rtol=0,atol=1e-12)
            np.testing.assert_array_equal(enabled,mask)
            self.assertEqual(embedded.stage,c.stage)
            self.assertEqual(embedded.baseline,c.baseline)
        q0 = selected[9384]['feeding_motor_input']['angles_deg']
        for t in (0.,.2,.4):
            advance(q0,t)
        self.assertEqual(c.stage,'acquire')
        for i, row in enumerate(selected.values()):
            t=.6+i*.2
            advance(row['feeding_motor_input']['angles_deg'],t)
            self.assertEqual(c.position,0.)
            self.assertIsNone(c.baseline)
        # A synthetic constant tail checks settling logic only, not execution.
        q=selected[9399]['feeding_motor_input']['angles_deg']
        for t in np.arange(1.6,3.,.2):
            advance(q,float(t))
            if c.baseline is not None:
                break
        self.assertIsNotNone(c.baseline)
        self.assertEqual(c.position,0.)
        self.assertAlmostEqual(c.goal_m-c.baseline,.0645)
        self.assertAlmostEqual(c.observation['transfer_coordinate_m']-c.baseline,0.)
        self.assertIsNone(c.fault)

    def test_relative_actions_keep_profile_and_hold_without_return(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c=ready();c.command('step')
            positions=[]
            for command in ('step','step','back'):
                if positions:
                    core=c.feeding;p=core.position;targets=c.targets.copy();mask=c.enabled.copy()
                    c.command(command)
                    self.assertIs(c.feeding,core)
                    self.assertEqual(core.position,p)
                    self.assertEqual(c.targets,targets);self.assertEqual(c.enabled,mask)
                    self.assertEqual(core.stage,'acquire')
                for _ in range(3000):
                    c.cycle()
                    if c.feeding_paused:
                        break
                self.assertIsNone(c.feeding.fault)
                self.assertTrue(c.feeding.completed)
                self.assertEqual(c.feeding.stage,'complete')
                self.assertFalse(any(e['kind']=='start_return' for e in c.feeding.events))
                positions.append(c.feeding.position)
                held=c.targets.copy();p=c.feeding.position
                for _ in range(120):c.cycle()
                c.command('toggle');c.cycle()
                self.assertTrue(c.feeding_paused)
                self.assertEqual(c.targets,held);self.assertEqual(c.feeding.position,p)
            self.assertGreater(positions[1],positions[0])
            self.assertLess(positions[2],positions[1])

    def test_unsettled_measurement_never_creates_progress(self):
        c=ContinuousFeeding(np.full(21,.0645))
        q=np.degrees(c.targets)
        for t in (5.4,5.6,5.8,6.):c.advance(q,t,t)
        self.assertEqual(c.stage,'acquire')
        for i in range(51):
            t=6.2+i*.2
            moving=q.copy();moving[4]=12. if i%2 else 6.
            c.advance(moving,t,t)
        self.assertIsNone(c.baseline)
        self.assertEqual(c.position,0.)
        self.assertIn('did not settle',c.fault)


if __name__=='__main__':unittest.main()
