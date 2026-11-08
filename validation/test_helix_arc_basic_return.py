"""Replay the reported final mask and check real/debug handoff to ordinary ARC J/H/G."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from validation.test_real_controller import real, RecordingSerial
from snake_control import helix_arc_transition as library
from codes import Rolling_Arc_Debug_Mujoco as debug


DATA=json.loads((Path(__file__).parent/'fixtures/helix_arc_center_return_20261108.json').read_text(encoding='utf8'))


def centered(module):
    row=DATA['states']['centered_before_completion'];status=row['helix_arc_transition']
    q=real.GravityMomentGait(18,lengths=DATA['link_lengths_m']).targets(.4,.12)[0]
    m=module.HelixArcManual(np.degrees(q),DATA['link_lengths_m'],
        end_reset_options=dict(rate_deg_s=60.,hold_s=.25,max_s=2.))
    core=m.core
    core.stage='center_settle';core.elapsed=2.99
    core.head=status['arc_head_m'];core.arc_length=status['arc_length_m'];core.radius=status['fit']['radius']
    core.phase=status['phase'];core.enabled=np.array(row['enabled'],bool)
    core.targets=np.radians(row['target_controller_deg'])
    m.end_reset_done=True;m.end_reset_last=status['end_reset_last'];m.end_reset_history=[m.end_reset_last]
    m.started=m.running=True
    return m,np.array([row['measured_controller_deg'][f'J{i}'] for i in range(1,19)])


class HelixArcBasicReturnTests(unittest.TestCase):
    def setUp(self):
        self.output=contextlib.redirect_stdout(io.StringIO());self.output.__enter__()

    def tearDown(self):self.output.__exit__(None,None,None)

    def test_recorded_failure_was_reset_before_center_and_h_rejection(self):
        before=DATA['states']['before_center_reset_finished'];after=DATA['states']['completed_arc']
        self.assertEqual(before['window'],[4,9]);self.assertEqual(after['window'],[7,13])
        self.assertEqual(before['helix_arc_transition']['end_reset_last']['reason'],'zero_then_disabled')
        self.assertLess(before['helix_arc_transition']['end_reset_last']['actual_max_deg'],1.1)
        self.assertTrue(DATA['rejected_back'])
        self.assertTrue(all(c['result']=='rejected' and c['stage_before']=='arc_roll' for c in DATA['rejected_back']))

    def test_final_reset_uses_new_mask_and_releases_before_completion(self):
        for module in (library,real):
            with self.subTest(module=module.__name__):
                m,q=centered(module);mask=m.enabled.copy();held=m.targets.copy();phase=m.phase
                self.assertFalse(mask[4])  # J5 was previously bearing and has now been released.
                m.advance(.02,0.,0.,q)
                self.assertEqual(m.stage,'arc_final_end_reset');self.assertFalse(m.completed)
                self.assertTrue(m.enabled.all())
                np.testing.assert_allclose(m.targets[mask],held[mask],atol=1e-10)
                for i in range(1,101):
                    q=np.degrees(m.targets);m.advance(.02,i*.02,i*.02,q)
                    if m.completed:break
                self.assertTrue(m.completed);self.assertTrue(m.final_end_reset_done)
                self.assertFalse(m.running);np.testing.assert_array_equal(m.enabled,mask)
                self.assertEqual(m.phase,phase)
                self.assertEqual(m.end_reset_last['when'],'after_center')
                self.assertEqual(m.end_reset_last['bearing_joints'],list(range(7,14)))
                self.assertIn(5,m.end_reset_last['passive_joints'])
                self.assertLessEqual(m.end_reset_last['actual_max_deg'],5.)
                self.assertEqual(len(m.end_reset_history),2)

    def check_normal_arc_controls(self,c,module):
        m,q=centered(module);c.helix_arc_transition=m;c.helix_transition=None
        c.mode='run';c.targets=np.degrees(m.targets).tolist();c.enabled=m.enabled.tolist()
        c.first,c.last=7,13;c.bus.prepare_enabled(c.targets,c.enabled)
        clock=[0.,0.];c.motor_sample_clock=lambda:tuple(clock)
        with patch.object(c,'measured_angles',side_effect=lambda:c.targets.copy()):
            c.tick(.02)
            for i in range(1,110):
                clock[:]=[i*.02,i*.02];c.tick(.02)
                if c.helix_arc_transition is None:break
        self.assertIsNone(c.helix_arc_transition);self.assertEqual(c.snapshot()['gait'],'basic_arc')
        self.assertEqual(c.mode,'hold');self.assertIsNotNone(c.basic_arc_reference)
        c.states={i:{'angle_deg':float(a)} for i,a in enumerate(c.targets)}
        c.show_status()
        self.assertEqual(c.enabled,list(DATA['states']['completed_arc']['enabled']))
        np.testing.assert_allclose(c.basic_targets(),c.targets,atol=1e-9)
        self.assertEqual([c.bus.port.registers[s][0x28] for s in module.SERVO_IDS],list(map(int,c.enabled)))
        # Exact ordinary J/H contract: phase increments, not another one-second transition.
        phase=c.phase;mask=c.enabled.copy();held=c.targets.copy()
        for name,sign in [('step',1),('back',-1)]:
            start=c.phase;c.command(name)
            for _ in range(100):
                c.tick(.02)
                if c.mode=='hold':break
            self.assertEqual(c.mode,'hold');self.assertAlmostEqual(c.phase-start,sign*c.phase_step)
            self.assertEqual(c.enabled,mask)
        self.assertAlmostEqual(c.phase,phase)
        np.testing.assert_allclose(c.targets,held,atol=1e-8)
        c.command('toggle');c.tick(.02)
        self.assertAlmostEqual(c.phase-phase,c.direction*c.omega*.02)
        c.command('pause');held=c.targets.copy();c.tick(.02)
        self.assertEqual(c.targets,held);self.assertIsNone(c.helix_arc_transition)
        c.command('gait_feeding');self.assertIsNone(c.basic_arc_reference)

    def test_real_returns_to_ordinary_arc_without_shape_phase_or_enable_jump(self):
        c=real.RollingArcController(real.FeetechBus(RecordingSerial()))
        self.check_normal_arc_controls(c,real)

    def test_final_reset_timeout_is_recorded_as_timeout_and_not_zero_success(self):
        for module in (library,real):
            with self.subTest(module=module.__name__):
                m,q=centered(module);mask=m.enabled.copy();q[~mask]=15.
                m.advance(.02,0.,0.,q)
                for i in range(1,102):
                    m.advance(.02,i*.02,i*.02,q)
                    if m.completed:break
                self.assertTrue(m.completed);np.testing.assert_array_equal(m.enabled,mask)
                self.assertEqual(m.end_reset_last['reason'],'timeout_then_disabled')
                self.assertEqual(m.end_reset_last['actual_max_deg'],15.)

    def test_manual_mujoco_routes_both_directions_through_same_arc_scheduler(self):
        from snake_control.rolling_arc import ArcParameters
        session=SimpleNamespace(sim=SimpleNamespace(p=ArcParameters(joints=18)),elapsed=0.,stop_push=lambda:None)
        c=debug.DebugArcController(debug.real.FeetechBus(debug.real.DryRunSerial()),session)
        self.check_normal_arc_controls(c,debug.real)


if __name__=='__main__':unittest.main()
