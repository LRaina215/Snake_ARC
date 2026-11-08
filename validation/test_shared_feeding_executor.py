"""The three hotkeys share factory/settings/advance; curve handoff retains its seed."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from validation.test_real_controller import real,prepared_controller
from snake_control import feeding_controller as library
from snake_control.curve_gait import CurveGait,arc,line


class SharedFeedingExecutorTests(unittest.TestCase):
    def test_seeded_curve_retains_every_enabled_joint_at_entry(self):
        lengths=[.025]+[.072]*17+[.091]
        gait=CurveGait(18,lengths=lengths);start=.26;span=.30
        for phase in (0.,.7,1.57,-3.78):
            q=gait.targets([line(start),arc(span,np.pi/span),line(gait.body_length-start-span)],phase)
            mask=(gait.s[2:]>start)&(gait.s[:-2]<start+span)
            active=np.flatnonzero(mask)+1;window=(int(active[0]),int(active[-1]))
            for module in (library,real):
                with self.subTest(phase=phase,module=module.__name__):
                    f=module.create_feeding_controller(lengths,window,phase_rad=phase,
                        seed_curve=[line(start),arc(span,np.pi/span),line(gait.body_length-start-span)])
                    np.testing.assert_allclose(f.profile(0.,0.)[0],q,atol=1e-12)

    def test_four_five_seven_inherit_the_same_settings_and_execution(self):
        overrides=dict(tracking_tolerance_deg=11.,exit_angle_deg=13.,target_rate_deg_s=51.,
                       maximum_feed_s=99.,automatic_return=True)
        with contextlib.redirect_stdout(io.StringIO()),\
             patch.object(real,'FEED_OPTIONS_OVERRIDE',overrides),\
             patch.object(real,'FEED_DIRECTION_OVERRIDE',dict(stability_s=.2)),\
             patch.object(real,'HELIX_TRANSITION_SPEED_SCALE',1.),\
             patch.object(real,'HELIX_ARC_SPEED_SCALE',1.):
            c,_=prepared_controller();c.command('gait_feeding');c.command('step');four=c.feeding
            c2,_=prepared_controller();five=c2.read_gravity_entry(False,helix_only=True).front.transfer
            self.assertIsNotNone(five)
            q=real.GravityMomentGait(18,lengths=real.LINK_LENGTHS_M).targets(.4,.12)[0]
            seven=real.HelixArcTransition(q,real.LINK_LENGTHS_M,feeding_settings=c.make_feeding_settings())
            seven.head=.26;seven.stage='auto_center'
            seven.measured=q.copy()
            seven.arc_targets(.02)
            seven.step(.02,0.,seven.targets.copy(),fresh=True)
            feeders=(four,five,seven.center_feeding)
            self.assertTrue(all(type(f) is real.ContinuousFeeding for f in feeders))
            self.assertTrue(all(type(f).advance is real.LegacyContinuousFeeding.advance for f in feeders))
            for f in feeders:
                for k,v in overrides.items():
                    if k!='automatic_return':self.assertEqual(getattr(f.options,k),v)
                self.assertEqual(f.length_observer.options.stability_s,.2)
                self.assertEqual(f.trajectory_backend,'stitched_curve_integral')
            self.assertEqual(five.goal_kind,'position');self.assertEqual(feeders[2].goal_kind,'balance')
            self.assertTrue(four.options.automatic_return)
            self.assertFalse(five.options.automatic_return)
            self.assertFalse(feeders[2].options.automatic_return)

    def test_balancing_calls_the_common_encoder_observer_and_profile_not_a_second_centroid_formula(self):
        for module in (library,real):
            lengths=[.0645]*19;f=module.create_feeding_controller(lengths,(6,11))
            f.request_center()
            calls=[]
            def observed(q,window,t):
                calls.append(t)
                return dict(valid=True,transfer_coordinate_m=(f.position-1.)*.0645,
                    down_head=[1.,0.,0.],actual_arc_window=list(window))
            f.length_observer.observe=observed
            for i in range(3000):
                t=i*.02
                f.advance(np.degrees(f.targets),t,t)
                if f.completed:break
            self.assertTrue(f.completed);self.assertGreater(len(calls),10)
            self.assertLess(abs(f.position-1.),.07)
            self.assertEqual(f.goal_kind,'balance')


if __name__=='__main__':unittest.main()
