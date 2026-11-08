"""Real hotkey/serial scheduling, standalone math parity and existing-mode dispatch; no COM."""
import contextlib
import io
import unittest
from types import SimpleNamespace

import numpy as np

from validation.test_real_controller import real, prepared_controller
from validation.test_gravity_grip_radius import controller
from codes.Rolling_Arc_Debug_Mujoco import decode_sim_hotkey,DebugArcController,DebugSession
from snake_control.rolling_arc import ArcParameters
from snake_control.helix_arc_transition import HelixArcManual,HelixArcTransition


class HelixArcManualTests(unittest.TestCase):
    def setUp(self):
        self.output=contextlib.redirect_stdout(io.StringIO());self.output.__enter__()

    def tearDown(self):self.output.__exit__(None,None,None)

    def test_hotkeys_shared_and_existing_four_five_six_preserved(self):
        for decode in (real.decode_hotkey,decode_sim_hotkey):
            for vk in (55,103):
                self.assertEqual(decode(vk,True),'gait_helix_arc')
                self.assertIsNone(decode(vk,False))
            for vk,name in [(52,'gait_feeding'),(53,'gait_helix'),(54,'gait_gravity')]:
                self.assertEqual(decode(vk,True),name)

    def test_selection_holds_and_j_budget_g_pause_preserve_progress(self):
        c=controller();clock=[0.,0.];c.motor_sample_clock=lambda:tuple(clock)
        before=c.targets.copy();mask=c.enabled.copy()
        writes=lambda:sum(p[4] in (3,0x83) for p in c.bus.port.packets)
        before_writes=writes()
        c.command('gait_helix_arc')
        self.assertEqual(c.mode,'hold');self.assertEqual(c.targets,before);self.assertEqual(c.enabled,mask)
        self.assertEqual(writes(),before_writes)
        self.assertIsNone(c.helix_transition)
        c.command('step');core=c.helix_arc_transition
        c.tick(.02);self.assertEqual(c.command('step'),'in_progress')
        for i in range(1,50):
            clock[0]=i*.02;clock[1]=clock[0];c.refresh_compliance_feedback();c.tick(.02)
        self.assertEqual(c.mode,'hold');self.assertAlmostEqual(core.control_time,1.)
        held=c.targets.copy()
        clock[:]=[20.,20.];c.tick(.02)
        self.assertEqual(c.targets,held);self.assertAlmostEqual(core.control_time,1.)
        c.command('toggle');c.refresh_compliance_feedback();c.tick(.02)
        self.assertAlmostEqual(core.control_time,1.02)
        c.command('toggle');self.assertEqual(c.mode,'hold')
        self.assertEqual(c.snapshot()['gait'],'helix_to_arc')
        self.assertEqual(real.preview_payload(c)['gait'],'Helix→ARC→自动归中')
        c.command('release');self.assertIsNone(c.helix_arc_transition)
        self.assertFalse(any(c.enabled))

    def test_wrapper_matches_validated_core_and_standalone_does_not_depend_on_repo_imports(self):
        gait=real.GravityMomentGait(18,real.GravityMomentOptions(r_h1=.094,pitch=.13),.0645)
        angles=np.degrees(gait.targets(.4,.094)[0]);lengths=[.0645]*19
        plain=HelixArcTransition(np.radians(angles),lengths)
        wrapped=HelixArcManual(angles,lengths);embedded=real.HelixArcManual(angles,lengths)
        wrapped.running=embedded.running=True
        # Enough time to reach the first curved/straight material handover.
        for i in range(600):
            now=i*.02;sample=(i//10)*.2
            if i%10==0:feedback=angles.copy()
            q,m=plain.step(.02,now,np.radians(feedback),fresh=i%10==0)
            for core in (wrapped,embedded):
                self.assertTrue(core.advance(.02,now,sample,feedback))
                np.testing.assert_allclose(core.targets,q,atol=1e-12)
                np.testing.assert_array_equal(core.enabled,m)

    def test_debug_dispatch_preserves_in_progress_result(self):
        original=controller()
        session=SimpleNamespace(sim=SimpleNamespace(p=ArcParameters(joints=18,first_joint=1,last_joint=18)),elapsed=0.)
        c=DebugArcController(original.bus,session)
        c.targets=original.targets.copy();c.enabled=original.enabled.copy();c.mode='hold'
        c.command('gait_helix_arc');c.command('step');c.tick(.02)
        self.assertEqual(c.command('step'),'in_progress')

    def test_debug_can_start_seven_directly_from_unarmed_helix_preview(self):
        session=DebugSession(scene='tee_up60',start_x=.2)
        gait=real.GravityMomentGait(18,real.GravityMomentOptions(r_h1=.094,pitch=.13),.0645)
        session.sim.place_shape(gait.targets(.4,.094)[0],19,.2)
        session.commands.put('gait_helix_arc');session.advance()
        self.assertFalse(session.physics_started)
        self.assertFalse(session.controller.touched_hardware)
        session.commands.put('step');session.advance()
        self.assertTrue(session.physics_started)
        self.assertTrue(session.controller.touched_hardware)
        self.assertGreater(session.controller.helix_arc_transition.control_time,0.)

    def test_stale_feedback_freezes_and_centering_completion_pauses_once(self):
        c=controller();c.command('gait_helix_arc');core=c.helix_arc_transition
        core.running=True
        self.assertFalse(core.advance(.02,5.,0.,c.targets));self.assertEqual(core.control_time,0.)
        core.core.stage='center_settle';core.core.elapsed=2.99;core.started=True
        core.core.head=(core.core.L-core.core.arc_length)/2
        self.assertTrue(core.advance(.02,5.,5.,c.targets))
        self.assertTrue(core.completed);self.assertFalse(core.running);self.assertEqual(core.step_left,0.)
        phase=core.phase;core.running=True
        core.advance(.02,5.02,5.02,c.targets)
        self.assertGreater(core.phase,phase);self.assertTrue(core.running)

    def test_leaving_seven_routes_back_to_four_five_six(self):
        for mode in ('gait_feeding','gait_helix','gait_gravity'):
            with self.subTest(mode=mode):
                c,_=prepared_controller()
                # Realistic final ARC from an existing controller; the new
                # scheduler must not capture or reject another gait command.
                c.helix_arc_transition=real.HelixArcManual(np.degrees(
                    real.GravityMomentGait(18).targets(.4,.094)[0]),[.0645]*19)
                c.command(mode)
                self.assertIsNone(c.helix_arc_transition)
                self.assertTrue(c.feeding_selected if mode=='gait_feeding' else c.helix_transition is not None)


if __name__=='__main__':unittest.main()
