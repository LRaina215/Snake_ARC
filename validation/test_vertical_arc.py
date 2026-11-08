"""New reference integrator, near-vertical frames, shared adapter handover."""
import unittest,math
import numpy as np
from snake_control.curve_gait import CurveGait,helix
from snake_control.bent_helix_curve import BentHelixCurve
from snake_control.vertical_arc_transition import VerticalArcTransition,VerticalArcOptions
from snake_control.helix_arc_transition import HelixArcTransition


class VerticalArcTests(unittest.TestCase):
    def setUp(self):
        self.lengths=[.025]+[.072]*17+[.091];self.g=CurveGait(18,lengths=self.lengths)

    def test_bent_reference_straight_limit_matches_existing_helix_integral(self):
        b=BentHelixCurve(.094,.13);s,k,_=b.field(-.1,self.g.body_length)
        for roll in (0.,.4,1.57):
            a=self.g.sampled_targets(s,k,roll);q=self.g.targets([helix(self.g.body_length,.094,.13)],roll)
            self.assertLess(np.max(abs(np.degrees(a-q))),.3)

    def test_frame_finite_and_continuous_across_vertical_bend_and_horizontal(self):
        b=BentHelixCurve(.094,.13);gammas=[]
        for u in np.linspace(-.05,.75,41):
            s,k,_=b.field(u,.907);gammas.append(b.gamma)
            self.assertTrue(np.isfinite(k).all());self.assertAlmostEqual(s[-1],.907)
        self.assertLess(np.max(abs(np.diff(gammas))),.2)

    def test_shared_reference_application_and_horizontal_entry_after_localisation(self):
        q=self.g.targets([helix(self.g.body_length,.094,.13)],.4)
        c=VerticalArcTransition(q,self.lengths,VerticalArcOptions(phase_rate_rad_s=-.5,retained_turns=1.5,finish_after_pilot=False))
        self.assertIs(type(c).apply_reference,HelixArcTransition.apply_reference)
        for i in range(2600):q,mask=c.step(.02,q)
        c.begin_bend()
        for i in range(10):q,mask=c.step(.02,q)
        c.confirm_horizontal(q)
        self.assertEqual(c.horizontal.stage,'reverse_unwrap')
        self.assertLessEqual(np.max(abs(np.degrees(c.targets))),88.01)


if __name__=='__main__':unittest.main()
