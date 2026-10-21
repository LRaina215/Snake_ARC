"""Small encoder-only boundary checks, no hardware or dynamics."""
import unittest
import numpy as np
from snake_control.continuous_feeding import ContinuousFeeding
from snake_control.whole_feeding_observer import WholeFeedingObserver


class PassiveSegmentTests(unittest.TestCase):
    def setUp(self):
        self.core = ContinuousFeeding(np.full(21,.0645))
        self.q = np.degrees(self.core.targets)

    def warm(self, observer, q, window=(8,13), start=0.):
        for t in (start,start+.2,start+.4):
            result = observer.observe(q,window,t)
        return result

    def test_bent_tips_can_use_inner_passive_spans(self):
        q=self.q.copy();q[0]=40.;q[-1]=60.
        result=self.warm(self.core.length_observer,q)
        self.assertTrue(result['valid'],result)
        ends=result['end_quality']['ends']
        self.assertGreater(ends['head']['selected']['links'][0],1)
        self.assertLess(ends['tail']['selected']['links'][1],21)
        for side in ends.values():
            self.assertGreaterEqual(side['selected']['straightness'],.98)

    def test_repeated_feedback_and_gap_cannot_certify_stability(self):
        observer=self.core.length_observer
        for _ in range(10):
            self.assertFalse(observer.observe(self.q,(8,13),0.)['valid'])
        self.assertFalse(observer.observe(self.q,(8,13),.2)['valid'])
        self.assertTrue(observer.observe(self.q,(8,13),.4)['valid'])
        self.assertFalse(observer.observe(self.q,(8,13),2.)['valid'])
        self.assertTrue(self.warm(observer,self.q,start=2.)['valid'])
        self.assertEqual(observer.observe(self.q+1.,(8,13),2.4)['reason'],'changed_angles_with_same_feedback_time')

    def test_shaking_near_straight_spans_need_to_settle(self):
        observer=self.core.length_observer
        passive=np.r_[np.arange(7),np.arange(13,20)]
        for i in range(5):
            q=self.q.copy();q[passive]=2. if i%2 else -2.
            result=observer.observe(q,(8,13),.2*i)
        self.assertFalse(result['valid'])
        self.assertTrue(any(e['rejection_counts'].get('shape_unstable') for e in result['end_quality']['ends'].values()))
        self.assertTrue(self.warm(observer,q,start=1.)['valid'])

    def test_enable_boundary_invalidates_selected_spans(self):
        observer=self.core.length_observer
        self.assertTrue(self.warm(observer,self.q)['valid'])
        result=observer.observe(self.q,(3,18),.6)
        self.assertFalse(result['valid'])
        self.assertTrue(all(e['reason']=='insufficient_passive_links' for e in result['end_quality']['ends'].values()))
        self.assertFalse(observer.observe(self.q,(8,13),.8)['valid'])

    def test_two_consistent_ends_fuse_and_one_bent_end_can_drop_out(self):
        result=self.warm(self.core.length_observer,self.q)
        self.assertEqual(result['end_quality']['source'],'global_two_consistent_segments')
        q=self.q.copy();q[13:]=40.
        result=self.warm(WholeFeedingObserver(self.core.lengths,(8,13)),q)
        self.assertEqual(result['end_quality']['source'],'global_single_head')

    def test_prepare_boundary_waits_for_stability_but_is_bounded(self):
        c=self.core
        c.advance(self.q,5.8,5.8)
        c.advance(self.q,6.,6.)
        self.assertIsNone(c.fault);self.assertIsNone(c.baseline)
        c.advance(self.q,6.2,6.2)
        self.assertIsNone(c.baseline);self.assertEqual(c.stage,'acquire')
        for t in (6.4,6.6,6.8,7.,7.2,7.4):
            c.advance(np.degrees(c.targets),t,t)
        self.assertIsNotNone(c.baseline);self.assertEqual(c.stage,'feed_out')
        bad=ContinuousFeeding(c.lengths)
        q=np.full(20,40.)
        bad.advance(q,6.,6.);bad.advance(q,9.2,9.2)
        self.assertEqual(bad.fault,'initial differential length unavailable')


if __name__=='__main__':
    unittest.main(verbosity=2)
