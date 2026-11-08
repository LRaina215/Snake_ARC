"""Manual pre-U grip control, including the original later loosening; no serial device."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from validation.test_real_controller import real, RecordingSerial
from snake_control import gravity_handoff as library
from codes.Rolling_Arc_Debug_Mujoco import decode_sim_hotkey


def aligned_core(module=real):
    options=module.GravityMomentOptions(r_h1=.094,r_h2=.098,pitch=.13)
    gait=module.GravityMomentGait(18,options,.0645)
    q=np.degrees(gait.targets(.4,.094)[0])
    return module.GravityCrossing([.0645]*19,[True]*18,q,options,from_helix=True)


def controller():
    c=real.RollingArcController(real.FeetechBus(RecordingSerial()))
    c.helix_transition=core=aligned_core()
    c.enabled=core.enabled.tolist();c.targets=np.degrees(core.targets).tolist()
    c.bus.prepare_enabled(c.targets,c.enabled)
    c.first,c.last,c.mode=1,18,'hold'
    return c


class GripRadiusTests(unittest.TestCase):
    def test_hotkeys_match_manual_mujoco_and_need_control(self):
        for decode in (real.decode_hotkey,decode_sim_hotkey):
            self.assertEqual(decode(ord('O'),True),'grip_tighten')
            self.assertEqual(decode(ord('P'),True),'grip_loosen')
            self.assertIsNone(decode(ord('O'),False))
            self.assertIsNone(decode(ord('P'),False))

    def test_repeated_requests_are_smooth_keep_phase_and_mask_then_hold(self):
        for module in (library,real):
            with self.subTest(module=module.__name__):
                c=aligned_core(module);phase=c.phase;mask=c.enabled;previous=c.targets
                c.request_grip_radius_change(-.001);c.request_grip_radius_change(-.001)
                c.request_grip_radius_change(.001)
                self.assertAlmostEqual(c.grip_radius_target_m,.093)
                np.testing.assert_array_equal(c.targets,previous)
                for i in range(1,100):
                    old_radius=c.grip_radius_m
                    c.advance(.02,i*.02,i*.02,np.degrees(previous))
                    self.assertLessEqual(float(np.max(abs(c.targets-previous))),np.radians(.36)+1e-10)
                    self.assertLessEqual(abs(c.grip_radius_m-old_radius),.002*.02+1e-12)
                    self.assertEqual(c.phase,phase);np.testing.assert_array_equal(c.enabled,mask)
                    previous=c.targets
                    if c.stage=='align':break
                self.assertAlmostEqual(c.grip_radius_m,.093)
                self.assertFalse(c.running)
                self.assertEqual(c.stage,'align')

    def test_loosen_release_return_and_recovery_keep_the_original_radii(self):
        c=aligned_core();original_gait=c.gait.o;release=c.options.release_radius_m
        c.request_grip_radius_change(-.002)
        for i in range(1,100):
            c.advance(.02,i*.02,i*.02,np.degrees(c.targets))
            if c.stage=='align':break
        self.assertEqual(c.gait.o,original_gait)
        self.assertEqual(c.options.release_radius_m,release)
        c.confirm(3.,np.degrees(c.targets))
        for elapsed,expected in ((0.,.092),(3.,.092),(7.,.092),(7.85,release)):
            c.elapsed=elapsed
            with patch.object(c.gait,'targets',wraps=c.gait.targets) as target:
                c._advance_unwind(.02)
                self.assertAlmostEqual(target.call_args_list[0].args[1],expected)
        # Releasing still uses the old .098 m, and recovery returns to .094 m.
        c.stage='drop';c.running=True;c.elapsed=0.
        with patch.object(c.gait,'targets',wraps=c.gait.targets) as target:
            c.advance(.02,4.,4.,np.degrees(c.targets))
            self.assertAlmostEqual(target.call_args_list[0].args[1],.098)
        c._start_source_return()
        expected=c.gait.targets(c.phase,.098,ls=c.return_straight,th2=c.return_lead)[0]
        np.testing.assert_allclose(c.return_seating,expected)
        c.recover_phase=c.phase;c.anchor=c.targets.copy();c.elapsed=3.
        with patch.object(c.gait,'targets',wraps=c.gait.targets) as target:
            c._advance_recover(.02)
            self.assertAlmostEqual(target.call_args_list[-1].args[1],.094)

    def test_early_l_uses_current_radius_without_finishing_a_hidden_pending_request(self):
        c=aligned_core();c.request_grip_radius_change(-.003)
        c.advance(.02,.02,.02,np.degrees(c.targets))
        before=c.targets;current=c.grip_radius_m
        c.confirm(.02,np.degrees(before))
        self.assertEqual(c.stage,'unwind');self.assertFalse(c.running)
        self.assertAlmostEqual(c.grip_radius_target_m,current)
        np.testing.assert_array_equal(c.targets,before)
        self.assertEqual(c.options.release_radius_m,real.GravityCrossingOptions().release_radius_m)

    def test_pause_resume_never_reconstructs_and_loses_the_manual_setting(self):
        c=controller();core=c.helix_transition
        with contextlib.redirect_stdout(io.StringIO()),patch.object(c,'motor_sample_clock',return_value=(0.,0.)):
            c.command('grip_tighten');c.tick(.02);c.command('pause')
        before=core.targets;radius=core.grip_radius_m
        for _ in range(5):c.tick(.02)
        np.testing.assert_array_equal(core.targets,before);self.assertEqual(core.grip_radius_m,radius)
        with contextlib.redirect_stdout(io.StringIO()),patch.object(c,'read_gravity_entry',side_effect=AssertionError('lost setting')), \
                patch.object(c,'motor_sample_clock',return_value=(.02,.02)):
            c.command('run');c.tick(.02)
        self.assertIs(c.helix_transition,core)
        self.assertLess(core.grip_radius_m,radius)

    def test_only_pre_u_gravity_stage_changes_and_invalid_request_is_atomic(self):
        for stage,helix_only in (('unwind',False),('drop_hold',False),('align',True)):
            c=aligned_core();c.stage=stage;c.helix_only=helix_only;before=c.status()
            with self.assertRaisesRegex(ValueError,'小U之前'):c.request_grip_radius_change(-.001)
            self.assertEqual(c.status(),before)
        c=aligned_core();before=c.status()
        with self.assertRaises(ValueError):c.request_grip_radius_change(-1.)
        self.assertEqual(c.status(),before)


if __name__=='__main__':unittest.main()
