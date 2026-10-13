"""One focused software batch, separate from the three dynamics cases."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import math
import unittest
import numpy as np
from snake_control.active_arc_transfer import ActiveArcTransfer


class ActiveTransferTests(unittest.TestCase):
    def test_position_bias_preserves_axis_turn_budgets(self):
        c=ActiveArcTransfer(np.full(21,.0645),'transfer')
        for window in ((8,13),(7,12)):
            base=c.template(window,.7,60.)
            for bias in (-.4,.4):
                changed=c.template(window,.7,60.,bias)
                self.assertAlmostEqual(changed[::2].sum(),base[::2].sum())
                self.assertAlmostEqual(changed[1::2].sum(),base[1::2].sum())
                self.assertLess(float(np.max(abs(changed))),math.pi/2)

    def test_frozen_phase_adjacent_overlap_and_continuity(self):
        c=ActiveArcTransfer(np.full(21,.0645),'transfer')
        actual=np.degrees(c.targets)
        mask=c.enabled.copy();last=c.targets.copy()
        clock=0.;counts=set();feedback_time=0.;sample=actual.copy()
        for i in range(1300):
            clock=i*.02
            if i%10==0:sample=actual.copy();feedback_time=clock
            target,newmask=c.advance(sample,feedback_time,clock)
            shared=mask&newmask
            self.assertLessEqual(float(np.degrees(abs(target[shared]-last[shared])).max()),.9+1e-8)
            counts.add(int(newmask.sum()))
            actual=np.degrees(target)  # Ideal servo only tests transaction logic, not physical support.
            if clock>=6.:self.assertAlmostEqual(c.phase,.7)
            mask,last=newmask.copy(),target.copy()
            self.assertIsNone(c.fault,c.fault)
            if c.completed:break
        self.assertTrue(c.completed)
        self.assertEqual(counts,{6,7})
        self.assertEqual(c.window,(8,13))
        complete=[e for e in c.events if e['kind']=='transfer_complete']
        self.assertEqual([e['new'] for e in complete],[[7,12],[8,13]])
        self.assertTrue(all(e['gate']['exit_angle_ready'] for e in complete))

    def test_feedback_boundary_rejects_relabelled_samples(self):
        c=ActiveArcTransfer(np.full(21,.0645),'transfer')
        q=np.degrees(c.targets)
        c.advance(q,0.,0.)
        c.advance(q,0.,.02)
        with self.assertRaisesRegex(ValueError,'unchanged feedback'):
            c.advance(q+1.,0.,.04)
        with self.assertRaisesRegex(ValueError,'backwards'):
            c.advance(q,-.01,.04)
        with self.assertRaisesRegex(ValueError,'stale'):
            c.advance(q,0.,.6)


if __name__=='__main__':
    unittest.main(verbosity=2)
