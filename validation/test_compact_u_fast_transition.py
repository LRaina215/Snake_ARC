"""Compact U allocation, manual early release and initial-only acceleration; no COM port."""
import math
import unittest

import numpy as np

from validation.test_real_controller import real
from validation.test_gravity_grip_radius import controller
from snake_control import gravity_handoff as library


def core(module=real):
    gait=module.GravityMomentOptions(r_h1=.094,r_h2=.098,pitch=.13)
    q=np.degrees(module.GravityMomentGait(18,gait,.0645).targets(.4,.094)[0])
    options=module.GravityCrossingOptions(retain_joints=None,release_line_m=.129,
        release_arc_rad=math.radians(150),transition_speed_scale=1.5)
    return module.GravityCrossing([.0645]*19,[True]*18,q,gait,options,from_helix=True)


class CompactUFastTransitionTests(unittest.TestCase):
    def test_candidate_preserves_twelve_complete_helical_joint_neighbourhoods(self):
        for module in (library,real):
            with self.subTest(module=module.__name__):
                c=core(module);s=c.status()
                self.assertAlmostEqual(s['planned_u_length_m'],.09*math.radians(150))
                self.assertAlmostEqual(s['planned_front_length_m'],19*.0645-.129-.09*math.radians(150))
                self.assertEqual(s['planned_retain_joints'],12)
                c.confirm(0.,np.degrees(c.targets))
                c.elapsed=7.85;c._advance_unwind(.02)
                self.assertAlmostEqual(c.unwind_angle_rad,math.radians(150))
                self.assertAlmostEqual(c.unwind_straight_m,.129)
                c.confirm(8.,np.degrees(c.targets))
                self.assertEqual(c.release_retain_joints,12)
                c.running=True;c.advance(.02,8.02,8.02,np.degrees(c.targets))
                np.testing.assert_array_equal(c.enabled,np.arange(18)<12)
                c._start_source_return()
                self.assertEqual(c.return_retain,12)
                self.assertAlmostEqual(c.return_straight,7*.0645)

    def test_early_l_keeps_more_front_support_and_does_not_wait_for_three_seconds(self):
        c=core();c.confirm(0.,np.degrees(c.targets));c.elapsed=1.5;c._advance_unwind(.02)
        expected=int(c.gait.helix_joints(c.unwind_helix_m).sum());before=c.targets
        self.assertGreater(expected,12)
        c.confirm(1.5,np.degrees(c.targets))
        self.assertEqual(c.stage,'drop');self.assertFalse(c.running)
        self.assertEqual(c.release_retain_joints,expected)
        np.testing.assert_array_equal(c.targets,before)

    def test_acceleration_applies_only_to_initial_arc_intake_not_gravity_return(self):
        for initial,speed in ((True,1.5),(False,1.)):
            c=core();c._start_source_return(retain_joints=12,initial=initial)
            c.stage='return_wind';c.elapsed=1.;phase=c.phase
            c._advance_source_return(.02)
            self.assertAlmostEqual(c.offset,.012*speed)
            self.assertAlmostEqual(c.phase-phase,.084*speed*.02)
            self.assertEqual(c.gait.o.r_h2,.098)

    def test_real_factory_uses_candidate_and_exposes_speed_without_changing_manual_mode_four(self):
        c=controller();new=c.read_gravity_entry(True)
        self.assertIsNone(new.options.retain_joints)
        self.assertAlmostEqual(new.options.release_line_m,.129)
        self.assertAlmostEqual(new.options.release_arc_rad,math.radians(150))
        self.assertAlmostEqual(new.options.transition_speed_scale,real.HELIX_TRANSITION_SPEED_SCALE)
        self.assertEqual(new.unwind_retain_count(),12)
        self.assertAlmostEqual(real.ContinuousFeedingOptions().maximum_position_rate_joints_s,.1)


if __name__=='__main__':unittest.main()
