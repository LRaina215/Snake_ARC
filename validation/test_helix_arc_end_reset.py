"""End-zero lifecycle: bearing held, explicit torque release, bounded and resumable manual steps."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from validation.test_real_controller import real
from validation.test_gravity_grip_radius import controller
from snake_control import helix_arc_transition as library


def prepared(module):
    q=real.GravityMomentGait(18,real.GravityMomentOptions(r_h1=.094,pitch=.13),.0645).targets(.4,.094)[0]
    manual=module.HelixArcManual(np.degrees(q),[.0645]*19,
        end_reset_options=dict(rate_deg_s=60.,hold_s=.25,max_s=2.))
    core=manual.core
    core.stage='arc_settle';core.head=(core.L-core.arc_length)/2;core.phase=.7
    core.targets=core.curve.targets([module.line(core.head),module.arc(core.arc_length,1/core.radius),
        module.line(core.L-core.head-core.arc_length)],core.phase)
    centers=(np.arange(18)+1)*core.l
    core.enabled=(centers+core.l>core.head)&(centers-core.l<core.head+core.arc_length)
    angles=np.degrees(core.targets);angles[~core.enabled]=30.
    return manual,angles,core.enabled.copy(),core.targets.copy()


class HelixArcEndResetTests(unittest.TestCase):
    def test_bearing_held_then_ends_disabled_once_before_centering(self):
        for module in (library,real):
            m,q,bearing,held=prepared(module);m.running=True
            m.advance(.02,0.,0.,q)
            self.assertEqual(m.stage,'arc_end_reset');self.assertTrue(m.enabled.all())
            np.testing.assert_allclose(m.targets[bearing],held[bearing],atol=1e-12)
            for i in range(1,101):
                q=np.degrees(m.targets)
                m.advance(.02,i*.02,i*.02,q)
                np.testing.assert_allclose(m.targets[bearing],held[bearing],atol=1e-12)
                if m.end_reset_done:break
            self.assertTrue(m.end_reset_done)
            np.testing.assert_array_equal(m.enabled,bearing)
            self.assertEqual(m.end_reset_last['reason'],'zero_then_disabled')
            self.assertEqual(m.stage,'arc_settle')
            last=m.end_reset_last
            for j in range(110):
                now=(i+j+1)*.02;m.advance(.02,now,now,np.degrees(m.targets))
            self.assertIsNone(m.end_reset);self.assertIs(m.end_reset_last,last)

    def test_step_budget_releases_and_timeout_cannot_block_centering(self):
        for module in (library,real):
            m,q,bearing,held=prepared(module)
            m.step_once()
            for i in range(50):m.advance(.02,i*.02,i*.02,q)
            self.assertTrue(m.reset_suspended);self.assertEqual(m.stage,'arc_end_reset')
            np.testing.assert_array_equal(m.enabled,bearing)
            elapsed=m.end_reset.elapsed
            # The observer is held during the pause, and the next step must
            # rebase targets to the new measured pose, not the old end target.
            q[~bearing]=12.;m.step_once();m.advance(.02,10.,10.,q)
            np.testing.assert_allclose(np.degrees(m.targets[~bearing]),10.8,atol=1e-10)
            self.assertAlmostEqual(m.end_reset.elapsed,elapsed+.02)
            for j in range(1,60):
                if m.step_left<=0.:m.step_once()
                m.advance(.02,10.+j*.02,10.+j*.02,q)
                if m.end_reset_done:break
            self.assertTrue(m.end_reset_done)
            self.assertEqual(m.end_reset_last['reason'],'timeout_then_disabled')
            np.testing.assert_array_equal(m.enabled,bearing)

    def test_real_pause_disables_temporary_torque_and_g_preloads_current_angles(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c=controller();c.command('gait_helix_arc')
            m,q,bearing,held=prepared(real)
            c.helix_arc_transition=m;m.started=m.running=True
            c.enabled=bearing.tolist();c.targets=np.degrees(held).tolist();c.mode='run'
            c.bus.prepare_enabled(c.targets,c.enabled)
            c.states={i:{'angle_deg':float(a)} for i,a in enumerate(q)}
            with patch.object(c,'motor_sample_clock',return_value=(0.,0.)):
                c.tick(.02)
                self.assertTrue(all(c.enabled))
                c.command('toggle')
                self.assertEqual(c.mode,'hold')
                self.assertEqual(c.enabled,bearing.tolist())
                self.assertEqual([c.bus.port.registers[s][0x28] for s in real.SERVO_IDS],bearing.astype(int).tolist())
                q[~bearing]=12.;c.states={i:{'angle_deg':float(a)} for i,a in enumerate(q)}
                c.command('toggle');c.tick(.02)
            self.assertTrue(all(c.enabled))
            np.testing.assert_allclose(np.asarray(c.targets)[~bearing],10.8,atol=1e-10)


if __name__=='__main__':unittest.main()
