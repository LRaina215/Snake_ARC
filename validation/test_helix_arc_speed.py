"""A single speed multiplier preserves curve paths and the real-time J/feedback contract."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from validation.test_real_controller import real
from validation.test_gravity_grip_radius import controller
from snake_control import helix_arc_transition as library


def make_core(module,speed):
    q=real.GravityMomentGait(18,real.GravityMomentOptions(r_h1=.094,pitch=.13),.0645).targets(.4,.094)[0]
    return module.HelixArcTransition(q,[.0645]*19,module.HelixArcOptions(speed_scale=speed)),q


class HelixArcSpeedTests(unittest.TestCase):
    def test_reverse_and_flatten_follow_same_curve_at_scaled_motion_time(self):
        for module in (library,real):
            for stage in ('exit_transition','flatten'):
                with self.subTest(module=module.__name__,stage=stage):
                    fast,q=make_core(module,2.);reference,_=make_core(module,1.)
                    if stage=='flatten':
                        for core in (fast,reference):
                            core.stage='flatten';core.crown=.425;core.start_phase=core.phase
                            core.helix_length=.605;core.flatten_mask=np.ones(18,bool);core.clearance_sign=1.
                    count=180 if stage=='exit_transition' else 100
                    for i in range(count):
                        fast.step(.02,i*.02,q)
                        reference.step(.04,i*.04,q)
                        np.testing.assert_allclose(fast.targets,reference.targets,atol=1e-12)
                        np.testing.assert_array_equal(fast.enabled,reference.enabled)

    def test_center_and_roll_scale_while_settling_and_j_keep_real_time(self):
        for module in (library,real):
            fast,q=make_core(module,2.);reference,_=make_core(module,1.)
            for core in (fast,reference):
                core.stage='auto_center';core.head=.27
                core.targets=core.curve.targets([module.line(.27),module.arc(core.arc_length,1/core.radius),
                    module.line(core.L-.27-core.arc_length)],core.phase)
                core.enabled=(core.curve.s[2:]>.27)&(core.curve.s[:-2]<.27+core.arc_length)
                core.step(.02,0.,core.targets.copy())
                core.center_feeding.length_observer.observe=lambda *a:dict(valid=True,
                    transfer_coordinate_m=-.25,down_head=[1.,0.,0.])
                core.step(.02,.02,core.targets.copy(),fresh=True)
            self.assertAlmostEqual(fast.head-.27,2*(reference.head-.27))
            self.assertAlmostEqual(fast.center_feeding.options.maximum_position_rate_joints_s,
                                   2*reference.center_feeding.options.maximum_position_rate_joints_s)
            for core in (fast,reference):
                core.stage='arc_roll';before=core.phase;core.step(.02,.02,q)
                core.roll_change=core.phase-before
            self.assertAlmostEqual(fast.roll_change,2*reference.roll_change)
            fast.stage='arc_settle';fast.elapsed=0.
            for i in range(55):fast.step(.02,i*.02,q)
            self.assertEqual(fast.stage,'arc_settle')  # Still less than 2 physical seconds.
            manual=module.HelixArcManual(np.degrees(q),[.0645]*19,module.HelixArcOptions(speed_scale=2.))
            manual.step_once()
            for i in range(50):manual.advance(.02,i*.02,(i//10)*.2,np.degrees(q))
            self.assertEqual(manual.step_left,0.)
            self.assertAlmostEqual(manual.control_time,1.)

    def test_one_real_knob_is_used_without_changing_mode_five(self):
        with patch.object(real,'HELIX_ARC_SPEED_SCALE',1.5),contextlib.redirect_stdout(io.StringIO()):
            c=controller();c.command('gait_helix_arc')
            options=c.helix_arc_transition.core.o
            self.assertAlmostEqual(options.speed_scale,1.5)
            self.assertAlmostEqual(options.reverse_feed_m_s,.016)
            self.assertAlmostEqual(options.flatten_s,6.)
            self.assertAlmostEqual(c.helix_arc_transition.status()['speed_scale'],1.5)
            self.assertAlmostEqual(c.make_gravity_options().transition_speed_scale,real.HELIX_TRANSITION_SPEED_SCALE)


if __name__=='__main__':unittest.main()
