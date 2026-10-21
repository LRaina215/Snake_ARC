"""Recorded failure inputs and bounded software handovers; no COM or physics."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from snake_control.continuous_feeding import ContinuousFeeding,ContinuousFeedingOptions
from snake_control.posture_window import PostureWindow
from validation.half_circle_experiment import MeasuredParameters
from validation.posture_window_experiment import references
from validation.test_repeated_motor_port import AngleBus,real,debug
from snake_control.feature_arc import feature_arc_candidates
from snake_control.section_geometry import local_section
from snake_control.rolling_shape_observer import RollingShapeObserver
from snake_control.posture_window import GeneralPhaseReference
from dataclasses import replace

REPORT=Path(__file__).parent/'reports/runtime_relocalization_20261021'
DATA=json.loads((REPORT/'fixtures.json').read_text(encoding='utf8'))


class RuntimeRecoveryTests(unittest.TestCase):
    def test_mode1_or3_to4_preserves_measured_radius_and_nonzero_rolling_phase(self):
        p=MeasuredParameters(amplitude_deg=45.)
        q=np.degrees(p.targets(.8))
        for cls in (real.RollingArcController,debug.DebugArcController):
            for previous_mode in ('basic','sliding'):
                bus=AngleBus();bus.angles=q.copy();bus.enabled=p.enabled.copy()
                session=SimpleNamespace(elapsed=0.,sim=SimpleNamespace(p=real.ManualArcReference()),record_event=lambda *a,**k:None,stop_push=lambda:None)
                c=cls(bus) if cls is real.RollingArcController else cls(bus,session)
                c.motor_sample_clock=lambda:(session.elapsed,session.elapsed)
                c.first,c.last=p.first_joint,p.last_joint;c.enabled=p.enabled.tolist()
                c.targets=q.tolist();c.mode='hold';c.phase=.8;c.amplitude=45.
                if previous_mode=='sliding':c.sliding=object();c.compliance=object()
                with contextlib.redirect_stdout(io.StringIO()):
                    c.command('gait_feeding')
                    np.testing.assert_array_equal(c.targets,q)
                    c.command('step')
                    np.testing.assert_array_equal(c.targets,q)
                    core=c.feeding
                    self.assertEqual(core.snapshot()['seed_source'],'measured_feature_arc')
                    research=ContinuousFeeding(real.LINK_LENGTHS_M,ContinuousFeedingOptions(**core.options.__dict__),seed_angles_deg=q)
                    for position in (-1.,0.,1.):
                        np.testing.assert_allclose(core.profile(position,.05)[0],research.profile(position,.05)[0],atol=1e-12)
                    initial,_,_,_=core.profile(0.,0.)
                    seedmask=core.helper.mask(core.origin)
                    np.testing.assert_allclose(np.degrees(initial[seedmask]),q[seedmask],atol=1e-12)
                    self.assertGreater(abs(initial[1::2]).sum(),.1)
                    for i in range(5):
                        session.elapsed=i*.2;c.states=bus.read_states();c.tick_feeding(.02)
                        self.assertIsNone(core.fault)
                        self.assertAlmostEqual(core.position,0.)
                        np.testing.assert_allclose(np.array(c.targets)[seedmask],q[seedmask],atol=1e-10)

    def test_full_normal_cycle_does_not_recommend_false_migration(self):
        p=MeasuredParameters();lengths=np.full(21,.0645)
        o=RollingShapeObserver(lengths,GeneralPhaseReference(lengths,6,references()))
        o.options=replace(o.options,maximum_feedback_gap_s=.6)
        for i,phase in enumerate(np.linspace(0.,2*np.pi,33)):
            q=np.degrees(p.targets(phase))
            observation=o.update(q,float(phase),p.enabled,i*.2)
            self.assertNotIn(observation['state'],('recommend_head','recommend_tail','relocalize'))
            self.assertIsNone(observation['suggested_window'])

    def test_shared_feature_turns_match_control_smooth_section(self):
        for name in ('feeding','sliding'):
            r=DATA[name];q=np.array([r['measured_controller_deg'][f'J{i}'] for i in range(1,21)])
            result=feature_arc_candidates(q,np.full(21,.0645))
            self.assertIsNotNone(result['chosen'])
            for candidate in result['candidates']:
                section=local_section(np.radians(q),np.full(21,.0645),*candidate['window'])
                self.assertAlmostEqual(candidate['turn_deg'],np.degrees(section['total_turn']),places=7)

    def test_global_relocation_reaches_real_torque_mask_adapter(self):
        r=DATA['sliding'];q=np.array([r['measured_controller_deg'][f'J{i}'] for i in range(1,21)])
        bus=AngleBus();bus.angles=q.copy();bus.enabled=np.array(r['enabled'])
        c=real.RollingArcController(bus);clock=[0.,0.]
        c.motor_sample_clock=lambda:tuple(clock)
        c.first,c.last=r['window'];c.enabled=list(r['enabled']);c.targets=q.tolist()
        c.mode='hold';c.phase=r['phase_rad']
        with contextlib.redirect_stdout(io.StringIO()):
            c.command('gait_sliding')
            for i in range(500):
                clock[0]=i*.02
                if i%10==0:
                    clock[1]=clock[0];c.states=bus.read_states();c.feedback_phase_rad=c.phase
                c.tick_motor_sliding(.02)
                self.assertFalse(c.sliding_fault,c.section_block_reason)
                if c.sliding.relocalization is not None:
                    bus.angles[np.array(c.enabled)]=np.array(c.targets)[np.array(c.enabled)]
                if any(e['kind']=='relocalization_complete' for e in c.sliding.events):break
        self.assertEqual((c.first,c.last),(8,12));self.assertEqual(bus.sent_ids,[8,9,10,11,12])
        np.testing.assert_array_equal(bus.enabled,c.enabled)

    def slider(self,embedded=False):
        r=DATA['sliding'];q=np.array([r['measured_controller_deg'][f'J{i}'] for i in range(1,21)])
        p=MeasuredParameters(first_joint=2,last_joint=6,settle_s=0.)
        c=(real.PostureWindow if embedded else PostureWindow)(p,np.full(21,.0645),references())
        c.targets=np.radians(r['target_controller_deg']);c.core.targets=c.targets.copy()
        c.phase=r['phase_rad']
        return c,q

    def test_recorded_wrong_arc_relocates_with_bounded_commands_and_same_embedded_result(self):
        c,q=self.slider();e,_=self.slider(True);original=q.copy();peak=0.;trace=[]
        for i in range(500):
            t=i*.02
            # Only after actual acquisition do ideal software motors follow.
            # This verifies the transaction; it is not hardware motion evidence.
            if i%10==0:
                sample=q.copy();stamp=t
            old=c.targets.copy();oldmask=c.enabled.copy()
            a,mask=c.advance(sample,stamp,t,commanded_phase_rad=c.phase,phase_at_feedback=c.phase)
            b,em=e.advance(sample,stamp,t,commanded_phase_rad=e.phase,phase_at_feedback=e.phase)
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-12);np.testing.assert_array_equal(mask,em)
            common=mask & oldmask
            peak=max(peak,float(np.degrees(abs(a[common]-old[common])).max()))
            if c.relocalization is not None:q[mask]=np.degrees(a[mask])
            trace.append(dict(t=t,window=list(c.window),enabled=mask.tolist(),actual_deg=sample.tolist(),
                              target_deg=np.degrees(a).tolist(),state=(c.selection or {}).get('state')))
            if any(x['kind']=='relocalization_complete' for x in c.events):break
        self.assertEqual(c.window,(8,12));self.assertIsNone(c.fault)
        self.assertLessEqual(peak,.900001);self.assertFalse(c.phase_hold)
        np.testing.assert_allclose(np.degrees(c.targets[7:12]),original[7:12],atol=1e-12)
        self.assertEqual(int(mask.sum()),5)
        (REPORT/'relocation_software_trace.json').write_text(json.dumps(dict(
            input='recorded initial pose; ideal target following after acquire',max_step_deg=peak,
            events=c.events,trace=trace),ensure_ascii=False),encoding='utf8')

    def test_motion_not_reported_complete_when_old_arc_does_not_follow(self):
        c,q=self.slider()
        for i in range(650):
            t=i*.02;c.advance(q,t,t,commanded_phase_rad=c.phase,phase_at_feedback=c.phase)
            if c.fault:break
        self.assertIsNotNone(c.fault)
        self.assertFalse(any(e['kind']=='relocalization_complete' for e in c.events))

    def test_recorded_feeding_buffers_release_and_static_feedback_can_acquire(self):
        r=DATA['feeding'];f=r['feeding'];q=np.array(r['feeding_motor_input']['angles_deg'])
        c=ContinuousFeeding(np.full(21,.0645),ContinuousFeedingOptions(**f['options']))
        e=real.ContinuousFeeding(c.lengths,real.ContinuousFeedingOptions(**f['options']))
        for obj in (c,e):
            obj.position=f['position_joints'];obj.bias=f['bearing_bias']
            obj.targets=np.radians(r['target_controller_deg']);obj.enabled=np.array(r['enabled'])
            obj.window=tuple(r['window']);obj.stage='acquire';obj.stage_since=0.
        for i in range(51):
            t=i*.2;a,mask=c.advance(q,t,t);b,em=e.advance(q,t,t)
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-12);np.testing.assert_array_equal(mask,em)
            if c.baseline is not None:break
        self.assertIsNotNone(c.baseline,c.acquisition_metrics)
        self.assertTrue(c.observation['valid']);self.assertIsNone(c.fault)
        self.assertLess(c.window[1],19)
        self.assertAlmostEqual(c.position,f['position_joints'])
        self.assertAlmostEqual(c.goal_m-c.baseline,.0645)
        self.assertTrue(any(x['kind']=='retire_straight_buffers' for x in c.events))

    def test_mode4_to3_needs_no_passive_direction_and_matches_debug(self):
        r=DATA['feeding'];q=np.array(r['feeding_motor_input']['angles_deg']);f=r['feeding']
        for cls in (real.RollingArcController,debug.DebugArcController):
            bus=AngleBus();bus.angles=q.copy();bus.enabled=np.array(r['enabled'])
            session=SimpleNamespace(elapsed=0.,sim=SimpleNamespace(p=real.ManualArcReference()),record_event=lambda *a,**k:None)
            c=cls(bus) if cls is real.RollingArcController else cls(bus,session)
            core=real.ContinuousFeeding(real.LINK_LENGTHS_M,real.ContinuousFeedingOptions(**f['options']))
            core.position=f['position_joints'];core.bias=f['bearing_bias']
            c.feeding=core;c.feeding_selected=True;c.mode='hold'
            c.first,c.last=r['window'];c.enabled=list(r['enabled']);c.targets=list(q)
            with contextlib.redirect_stdout(io.StringIO()):c.command('gait_sliding')
            self.assertEqual((c.first,c.last),(9,15))
            self.assertFalse(c.initial_window_selection['requires_down_before_switch'])
            self.assertFalse(c.feeding_selected);self.assertIsNotNone(c.sliding)
            np.testing.assert_allclose(c.targets,q)
            np.testing.assert_array_equal(bus.enabled,[9<=j<=15 for j in range(1,21)])


if __name__=='__main__':unittest.main(verbosity=2)
