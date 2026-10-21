"""Existing real poses and software clocks only; no serial device or dynamics."""
import contextlib
import gzip
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from validation.test_active_feeding_port import ROOT, real, ClockController
from validation.test_repeated_motor_port import AngleBus
from snake_control.continuous_feeding import ContinuousFeeding
from snake_control.whole_feeding_observer import WholeFeedingObserver


def recorded_switch():
    return json.loads((ROOT/'validation/reports/feeding_boundary_20261021/evidence.json').read_text(encoding='utf8'))['selected_original_rows']['6681']


class ModeSwitchTests(unittest.TestCase):
    def test_reselection_writes_exact_torque_mask_and_only_sends_selected_ids(self):
        row=recorded_switch();c=ClockController()
        q=np.array([row['measured_controller_deg'][f'J{i}'] for i in range(1,21)])
        c.first,c.last=row['window'];c.enabled=list(row['enabled'])
        c.bus.set_enabled(c.enabled)
        for i,sid in enumerate(real.SERVO_IDS):
            raw=real.angle_to_raw(i,q[i])
            c.port.registers[sid][0x38:0x3A]=raw.to_bytes(2,'little')
        c.targets=q.tolist();c.mode='hold';c.feeding_selected=True
        with contextlib.redirect_stdout(io.StringIO()):c.command('gait_sliding')
        self.assertEqual((c.first,c.last),(12,17))
        self.assertEqual([c.port.registers[s][0x28] for s in real.SERVO_IDS],list(map(int,c.enabled)))
        c.cycle()
        self.assertFalse(c.sliding_fault,c.section_block_reason)
        packet=c.port.packets[-1]
        self.assertEqual(list(packet[7:-1:7]),[real.SERVO_IDS[i] for i in np.flatnonzero(c.enabled)])

    def test_mode4_to3_can_automatically_commit_a_position_window(self):
        bus=AngleBus();c=real.RollingArcController(bus)
        q=np.degrees(ContinuousFeeding(np.full(21,.0645)).targets)
        bus.angles=q;c.first,c.last=6,16;c.enabled=[6<=j<=16 for j in range(1,21)]
        bus.enabled=np.array(c.enabled);c.targets=q.tolist();c.mode='hold';c.feeding_selected=True
        clock=[0.];c.motor_sample_clock=lambda:(clock[0],clock[0])
        with contextlib.redirect_stdout(io.StringIO()):
            c.command('gait_sliding');initial=c.sliding.window
            for i in range(20):
                clock[0]=i*.2;c.states=bus.read_states();c.feedback_phase_rad=0.
                c.tick_motor_sliding(.02)
                self.assertFalse(c.sliding_fault,c.section_block_reason)
                if c.sliding.window!=initial:break
        self.assertNotEqual(c.sliding.window,initial)
        self.assertEqual(sum(c.enabled),6)
        self.assertTrue(any(e['kind']=='complete' and e.get('operation')=='slide' for e in c.sliding.events),c.sliding.events)
        np.testing.assert_array_equal(bus.enabled,c.enabled)

    def test_real_and_debug_reselect_recorded_feeding_pose_and_run_observers(self):
        from codes import Rolling_Arc_Debug_Mujoco as debug
        row=recorded_switch()
        q=np.array([row['measured_controller_deg'][f'J{i}'] for i in range(1,21)])
        session=SimpleNamespace(elapsed=400.,sim=SimpleNamespace(p=real.ManualArcReference()),
            record_event=lambda *a,**k:None)
        result=[]
        with contextlib.redirect_stdout(io.StringIO()):
            for cls in (real.RollingArcController,debug.DebugArcController):
                bus=AngleBus();bus.angles=q.copy()
                c=cls(bus) if cls is real.RollingArcController else cls(bus,session)
                session.elapsed=400.
                c.motor_sample_clock=lambda:(session.elapsed,session.elapsed)
                c.first,c.last=row['window'];c.enabled=list(row['enabled']);bus.enabled=np.array(c.enabled)
                c.targets=list(q);c.mode='run';c.phase=0.
                c.feeding_selected=True;c.feeding=object()
                c.command('gait_sliding')
                self.assertEqual((c.first,c.last),(12,17))
                self.assertEqual(sum(c.enabled),6)
                self.assertFalse(c.feeding_selected);self.assertIsNone(c.feeding)
                self.assertIsNone(c.feeding_inputs);self.assertFalse(c.forming)
                self.assertEqual(c.mode,'hold');self.assertFalse(c.freeze_targets)
                self.assertTrue(c.sliding.automatic)
                self.assertEqual(c.sliding.events[-1]['reason'],'remeasured_after_feeding')
                np.testing.assert_array_equal(c.sliding.targets,np.radians(q))
                np.testing.assert_array_equal(bus.enabled,c.enabled)
                # Verify fresh full-angle feedback reaches position/count
                # selection after the switch, not just its mode label.
                for t in (400.,400.2,400.4,400.6):
                    session.elapsed=t
                    c.motor_sample_clock=lambda t=t:(t,t)
                    c.states=bus.read_states();c.feedback_phase_rad=0.
                    c.tick_motor_sliding(.02)
                    self.assertFalse(c.sliding_fault,c.section_block_reason)
                self.assertIsNotNone(c.sliding.observation)
                self.assertIsNotNone(c.sliding.selection)
                self.assertIsNotNone(c.sliding.count_observation)
                self.assertEqual(c.sliding.last_time,400.6)
                result.append((c.enabled,c.sliding.selection))
        self.assertEqual(result[0],result[1])

    def test_failed_reselection_holds_old_mask_without_claiming_mode3(self):
        c=ClockController();c.mode='run';c.feeding_selected=True;c.feeding=object()
        c.first,c.last=6,16;c.enabled=[6<=j<=16 for j in range(1,21)]
        old=list(c.enabled)
        with self.assertRaisesRegex(ValueError,'模式4→3重选未完成'):
            c.command('gait_sliding')
        self.assertEqual(c.enabled,old);self.assertTrue(c.feeding_selected)
        self.assertIsNone(c.sliding);self.assertTrue(c.feeding_paused)
        self.assertEqual(c.mode,'hold')


class ReferenceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.core=ContinuousFeeding(np.full(21,.0645))
        self.q=np.degrees(self.core.targets)

    def test_serial_cadence_keeps_three_samples_without_loosening_thresholds(self):
        observer=self.core.length_observer
        for t in (0.,.215,.430,.645,.860,1.075):
            o=observer.observe(self.q,(8,13),t)
            if t>=.430:
                self.assertTrue(o['valid'],o)
                self.assertGreaterEqual(o['end_quality']['ends']['head']['selected']['samples'],3)
        self.assertEqual(observer.options.minimum_straightness,.98)
        self.assertEqual(observer.options.stability_s,.4)

    def test_failed_section_fit_cannot_refresh_the_last_valid_reference(self):
        observer=self.core.length_observer
        for t in (0.,.2,.4):observer.observe(self.q,(8,13),t)
        with patch('snake_control.whole_feeding_observer.local_section',side_effect=ValueError('section unavailable')):
            for t in (.6,.8,1.,1.2):
                self.assertFalse(observer.observe(self.q,(8,13),t)['valid'])
        self.assertEqual(observer._last_valid_time,.4)
        o=observer.observe(self.q,(8,13),1.4)
        self.assertEqual(o['end_quality']['reference_event'],'reacquire_after_gap')

    def test_review_does_not_complete_or_return_while_reference_is_missing(self):
        c=self.core
        for t in np.arange(5.4,7.5,.2):c.advance(np.degrees(c.targets),float(t),float(t))
        c.stage='review_out';c.stage_since=5.5
        with patch.object(c.length_observer,'observe',return_value={'valid':False,'reason':'no_stable_passive_segment'}):
            c.advance(np.degrees(c.targets),7.6,7.6)
        self.assertEqual(c.stage,'review_out');self.assertFalse(c.completed)

    def test_enabled_old_reference_is_never_used_for_continuity(self):
        observer=self.core.length_observer
        for t in (0.,.2,.4):o=observer.observe(self.q,(8,13),t)
        epoch=o['end_quality']['reference_epoch']
        # Retire the old 5-link head reference, but a warmed 3-link
        # passive subspan remains. Tail reference is still passive too.
        o=observer.observe(self.q,(4,13),.6)
        self.assertTrue(o['valid'],o)
        # No prior enabled span can be used to bridge a direction.
        self.assertEqual(o['end_quality']['reference_event'],'reacquire_no_overlap')
        self.assertGreater(o['end_quality']['reference_epoch'],epoch)

    def test_qualified_backup_joins_without_an_artificial_length_step(self):
        observer=self.core.length_observer
        bent=self.q.copy();bent[13:]=40.
        for t in (0.,.2,.4):o=observer.observe(bent,(8,13),t)
        self.assertEqual(o['end_quality']['source'],'global_single_head')
        straight=self.q.copy();straight[13:]=1.
        previous=None;handovers=[]
        for t in (.6,.8,1.,1.2,1.4):
            o=observer.observe(straight,(8,13),t)
            self.assertTrue(o['valid'],o)
            if o['end_quality']['reference_event']=='overlap_handover':
                self.assertAlmostEqual(o['transfer_coordinate_m'],previous['transfer_coordinate_m'])
                handovers.append(o)
            previous=o
        self.assertTrue(handovers)

    def test_short_missing_observation_recovers_without_velocity_spike(self):
        c=self.core
        for t in np.arange(5.4,7.5,.2):
            c.advance(np.degrees(c.targets),float(t),float(t))
        self.assertEqual(c.stage,'feed_out')
        q=np.degrees(c.targets);p=c.position
        with patch.object(c.length_observer,'observe',return_value={'valid':False,'reason':'no_stable_passive_segment'}):
            c.advance(q,7.6,7.6)
        self.assertEqual(c.position,p)
        c.advance(q,7.8,7.8)
        self.assertIsNone(c.fault)
        self.assertEqual(c.velocity_m_s,0.)

    def test_long_gap_requires_stable_redatum_and_never_credits_unknown_motion(self):
        c=self.core
        e=real.ContinuousFeeding(c.lengths,real.ContinuousFeedingOptions(**c.options.__dict__))
        def advance(q,t):
            a,mask=c.advance(q,t,t);b,em=e.advance(q,t,t)
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
            np.testing.assert_array_equal(mask,em)
            self.assertEqual(c.baseline,e.baseline)
            self.assertEqual(c.observation_recovery is None,e.observation_recovery is None)
        for t in np.arange(5.4,7.5,.2):advance(np.degrees(c.targets),float(t))
        q=np.degrees(c.targets);p=c.position
        old_progress=c.previous_coordinate-c.baseline
        advance(q,8.2)
        self.assertFalse(c.observation['valid']);self.assertEqual(c.position,p)
        for t in (8.4,8.6,8.8,9.):
            advance(q,t)
            self.assertEqual(c.position,p)
        self.assertIsNotNone(c.observation_recovery)
        advance(q,9.2)
        self.assertIsNone(c.observation_recovery);self.assertIsNone(c.fault)
        self.assertAlmostEqual(c.observation['transfer_coordinate_m']-c.baseline,old_progress)
        self.assertTrue(any(e['kind']=='length_reference_reacquired' and not e['unknown_gap_motion_counted'] for e in c.events))

    def test_actual_tracking_stall_is_identified_and_not_accepted_as_stable(self):
        path=ROOT/'validation/reports/feeding_reference_recovery_20261021/rolling_arc_20261021_133912_480257.fresh.jsonl.gz'
        with gzip.open(path,'rt',encoding='utf8') as stream:
            row=next(r['row'] for r in map(json.loads,stream) if r['row']['feeding_action_id']==117)
        f=row['feeding'];q=np.array(row['feeding_motor_input']['angles_deg'])
        from snake_control.continuous_feeding import ContinuousFeedingOptions
        c=ContinuousFeeding(np.full(21,.0645),ContinuousFeedingOptions(**f['options']))
        e=real.ContinuousFeeding(c.lengths,real.ContinuousFeedingOptions(**f['options']))
        for core in (c,e):
            core.position=f['position_joints'];core.bias=f['bearing_bias']
            core.enabled=np.array(row['enabled'],bool);core.window=tuple(f['window'])
            core.targets=np.radians(row['target_controller_deg'])
            core.stage='acquire';core.stage_since=0.
        for t in np.arange(0.,10.5,.2):
            a,mask=c.advance(q,float(t),float(t));b,em=e.advance(q,float(t),float(t))
            np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
            np.testing.assert_array_equal(mask,em)
        self.assertIsNone(c.baseline);self.assertEqual(c.position,f['position_joints'])
        self.assertIn('waiting_for_takeover_tracking',c.fault)
        self.assertEqual(c.acquisition_metrics['tracking_joint'],13)
        self.assertAlmostEqual(c.tracking_error_deg,5.10878842513867)
        self.assertEqual(c.options.tracking_tolerance_deg,5.)


if __name__=='__main__':unittest.main(verbosity=2)
