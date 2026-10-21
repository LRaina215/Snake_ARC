"""Real-log failure fixtures plus a software servo; no contact/world input."""
import contextlib
import io
import json
import unittest
from dataclasses import replace
import numpy as np
from validation.test_active_feeding_port import real, ready, ROOT, ClockController
from snake_control.continuous_feeding import ContinuousFeeding, ContinuousFeedingOptions


class BoundaryTests(unittest.TestCase):
    @unittest.skip("Historical mode-4 change withdrawn: user requested 84b28cb rollback")
    def test_bounds_cover_every_required_bend_with_both_reference_sides(self):
        for sign,bounds in ((1,(-4,6)),(-1,(-6,4))):
            c=ContinuousFeeding(np.full(21,.0645),replace(ContinuousFeedingOptions(),requested_transfer_m=sign*.0645))
            self.assertEqual(c.position_bounds,bounds)
            for p in np.linspace(*bounds,101):
                q,_,_,_=c.profile(p,-c.options.travelling_bias_amplitude*np.sin(np.pi*p))
                self.assertLess(abs(q[c.reference_mask()]).max(),1e-10)
                self.assertAlmostEqual(q.sum(),np.pi)
        for window in ((2,7),(14,19)):
            for sign in (-1,1):
                c=ContinuousFeeding(np.full(21,.0645),replace(ContinuousFeedingOptions(),
                    first_joint=window[0],last_joint=window[1],requested_transfer_m=sign*.0645))
                self.assertFalse(np.any(abs(c.targets[c.reference_mask()])>1e-10))
                self.assertLessEqual(c.position_bounds[0],0.)
                self.assertGreaterEqual(c.position_bounds[1],0.)

    @unittest.skip("Historical mode-4 change withdrawn: user requested 84b28cb rollback")
    def test_repeated_j_boundary_and_h_returns_without_reset(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c=ready();c.command('back')  # Same reference side as the real failure.
            for action in range(10):
                for _ in range(3500):
                    c.cycle()
                    if c.feeding_paused:break
                self.assertIsNone(c.feeding.fault)
                if c.feeding.stage=='boundary':break
                self.assertTrue(c.feeding.completed)
                c.command('step')
            self.assertEqual(c.feeding.stage,'boundary')
            self.assertEqual(c.feeding.position,4.)
            before=c.targets.copy();c.command('step');c.cycle()
            self.assertEqual(c.feeding.stage,'boundary');self.assertEqual(c.targets,before)
            self.assertTrue(c.feeding_paused);self.assertEqual(c.mode,'hold')
            c.command('back')
            for _ in range(3500):
                c.cycle()
                if c.feeding_paused:break
            self.assertIsNone(c.feeding.fault)
            self.assertTrue(c.feeding.completed)
            self.assertLess(c.feeding.position,3.5)
            self.assertEqual(c.feeding.options.tracking_tolerance_deg,5.)

    @unittest.skip("Historical mode-4 change withdrawn: user requested 84b28cb rollback")
    def test_recorded_j15_stall_has_an_inward_bridge_and_matches_embedded_core(self):
        evidence=json.loads((ROOT/'validation/reports/feeding_boundary_20261021/evidence.json').read_text(encoding='utf8'))
        row=evidence['selected_original_rows']['4812'];f=row['feeding']
        q=np.array(row['feeding_motor_input']['angles_deg'])
        options=ContinuousFeedingOptions(**f['options'])
        c=ContinuousFeeding(np.full(21,.0645),options)
        e=real.ContinuousFeeding(c.lengths,real.ContinuousFeedingOptions(**f['options']))
        for core in (c,e):
            core.position=f['position_joints'];core.bias=f['bearing_bias']
            core.enabled=np.array(row['enabled'],bool);core.window=tuple(row['window'])
            core.targets=np.radians(row['target_controller_deg'])
            core.request_step(-.0645,0.,q)
        self.assertEqual(c.stage,'reverse_recover')
        self.assertGreater(c.recovery['initial_error_deg'],5.)
        self.assertLessEqual(c.recovery['destination'],4.)
        first=c.targets.copy()
        for i in range(1500):
            t=i*.02
            if i==100:
                c.request_step(-.0645,t,q);e.request_step(-.0645,t,q)
                self.assertEqual(c.stage,'reverse_recover')
            previous=c.targets.copy()
            target,mask=c.advance(q,t,t)
            other,om=e.advance(q,t,t)
            np.testing.assert_allclose(target,other,rtol=0,atol=1e-12)
            np.testing.assert_array_equal(mask,om)
            self.assertLessEqual(np.degrees(abs(target[mask]-previous[mask])).max(),.900001)
            self.assertIsNone(c.fault)
            # Software plant follows commanded enabled joints only; torque-off
            # joints retain the recorded values. Not real recovery evidence.
            q[mask]=np.degrees(target[mask])
            if c.completed:break
        self.assertTrue(c.completed)
        self.assertEqual(c.stage,'recovered')
        self.assertEqual(c.position,4.)
        self.assertIsNone(c.baseline)  # Not passed off as a completed 64.5 mm step.
        self.assertLess(np.degrees(c.targets[14]),np.degrees(first[14]))

    def test_release_arm_restores_configuration_after_sliding_and_manual_edit(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c=ready();seed=c.configured_window
            c.feeding_selected=False;c.first,c.last=6,16
            c.enabled=[6<=j<=16 for j in range(1,21)]
            c.command('release');c.command('apply')
            self.assertEqual((c.first,c.last),seed);self.assertEqual(sum(c.enabled),6)
            c.command('release');c.command('window=5:12');c.command('apply')
            self.assertEqual((c.first,c.last),(5,12));self.assertEqual(sum(c.enabled),8)

    def test_recorded_slide_does_not_require_a_radius_fit(self):
        evidence=json.loads((ROOT/'validation/reports/feeding_boundary_20261021/evidence.json').read_text(encoding='utf8'))
        row=evidence['selected_original_rows']['8044']
        p=real.ManualArcReference(joints=20,first_joint=6,last_joint=16,
            amplitude_deg=70.,omega_rad_s=.6,settle_s=0.)
        slider=real.PostureWindow(p,real.LINK_LENGTHS_M,
            {6:real.PhaseShapeReference.from_dict(real.MOTOR_PHASE_REFERENCE),
             8:real.PhaseShapeReference.from_dict(real.MOTOR_PHASE_REFERENCE8)},
            section_options=real.make_section_options())
        slider.observation=row['sliding']['observation']
        q=np.array([row['measured_controller_deg'][f'J{i}'] for i in range(1,21)])
        result=slider._candidate(q,(5,15))
        self.assertGreater(result['actual_body_circle']['circle_rms_m'],.012)
        self.assertTrue(result['eligible']);self.assertFalse(result['circle_fit_required'])
        # Changing size still requires a radius/coverage estimate.
        result=slider._candidate(q,(5,16))
        self.assertTrue(result['circle_fit_required'])
        self.assertFalse(result['eligible'])

    def test_ctrl3_arms_measured_offset_arc_and_ctrl_u_keeps_configured_seed(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c=ClockController()
            q=np.zeros(20);q[[12,14,16]]=60.
            for i,sid in enumerate(real.SERVO_IDS):
                raw=real.angle_to_raw(i,q[i])
                c.port.registers[sid][0x38:0x3A]=raw.to_bytes(2,'little')
            c.command('gait_sliding')
            self.assertTrue(c.touched_hardware);self.assertTrue(c.forming)
            self.assertEqual(c.pending_gait,'sliding')
            self.assertGreaterEqual(c.first,11)
            self.assertTrue(all(c.enabled[j] for j in (12,14,16)))
            self.assertEqual(c.configured_window,(real.FIRST_JOINT,real.LAST_JOINT))
            selection=c.initial_window_selection
            self.assertEqual(selection['chosen'],[c.first,c.last])
            # Complete the normal formation lifecycle with a software servo.
            for _ in range(400):c.cycle()
            self.assertIsNotNone(c.sliding)
            c.command('release');c.command('apply')
            self.assertEqual((c.first,c.last),c.configured_window)

    def test_ctrl3_rejects_unobservable_straight_pose_without_enabling(self):
        c=ClockController()
        with self.assertRaisesRegex(ValueError,'未使能'):
            c.command('gait_sliding')
        self.assertFalse(c.touched_hardware)
        self.assertEqual(c.mode,'released');self.assertFalse(any(c.enabled))

    def test_debug_ctrl3_uses_same_selector_and_configured_reset(self):
        from codes import Rolling_Arc_Debug_Mujoco as debug
        from unittest.mock import patch
        with contextlib.redirect_stdout(io.StringIO()),patch.object(debug.real,'open_robot_serial',side_effect=AssertionError('No hardware')):
            session=debug.DebugSession();c=session.controller
            configured=c.configured_window
            c.command('gait_sliding')
            self.assertIsNotNone(c.initial_window_selection)
            self.assertEqual(c.pending_gait,'sliding')
            c.command('release');c.command('apply')
            self.assertEqual((c.first,c.last),configured)
            self.assertEqual(session.sim.data.time,0.)


if __name__=='__main__':unittest.main()
