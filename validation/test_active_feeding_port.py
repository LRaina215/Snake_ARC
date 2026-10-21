"""Recorded inputs + in-memory bus only. No COM port or new dynamics."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import contextlib
from dataclasses import replace
import gzip
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
import numpy as np
from unittest.mock import patch
from snake_control.continuous_feeding import ContinuousFeeding as ResearchFeeding, ContinuousFeedingOptions as ResearchOptions
from snake_control.whole_feeding_observer import WholeFeedingObserver

spec = importlib.util.spec_from_file_location('feeding_real', ROOT/'codes/Rolling_Arc_Real.py')
real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real)
REPORT = ROOT/'validation/reports/passive_segment_selection_20261021'


class RecordingSerial(real.DryRunSerial):
    def __init__(self):
        super().__init__()
        self.packets = []

    def write(self, packet):
        self.packets.append(packet)
        return super().write(packet)


class ClockController(real.RollingArcController):
    """Deterministic software clock with actual protocol quantization."""
    def __init__(self):
        self.t = 0.
        self.port = RecordingSerial()
        super().__init__(real.FeetechBus(self.port))

    def motor_sample_clock(self):
        return self.t, self.feedback_motor_clock_s

    def refresh_compliance_feedback(self):
        super().refresh_compliance_feedback()
        self.feedback_motor_clock_s = self.feedback_time_s = self.t

    def cycle(self):
        real.poll_control_feedback(self, self.t, self.t+100.)
        self.tick(.02)
        self.t += .02


def ready():
    c = ClockController()
    c.command('gait_feeding')
    c.command('apply')
    for _ in range(330):
        c.cycle()
    assert not c.forming and c.settle_remaining==0
    return c


class FeedingPortTests(unittest.TestCase):
    def test_full_record_current_core_matches_port_not_archived_commands(self):
        files = list((ROOT/'validation/reports/active_feeding_continuation/round_09').glob('*/status.json'))
        files += [ROOT/'validation/reports/active_feeding_continuation/round_07/small_20261014_140308/status.json']
        results = []
        for file in sorted(files):
            status = json.loads(file.read_text(encoding='utf8'))
            options = ResearchOptions(**status['options'])
            if status['case']=='small':
                options = replace(options, fractional_middle_boost_deg=15., middle_boost_activation_joints=1.5, middle_boost_ramp_joints=1.)
            source = ResearchFeeding(np.full(21,.0645), options)
            embedded = real.ContinuousFeeding(np.full(21,.0645), real.ContinuousFeedingOptions(**options.__dict__))
            observer = WholeFeedingObserver(np.full(21,.0645), (options.first_joint,options.last_joint))
            last_feedback = None
            actual_observations = invalid_after_prepare = 0
            invalid_since = None
            longest_invalid_s = archived_target_difference = 0.
            archived_mask_difference = 0
            port = RecordingSerial(); bus = real.FeetechBus(port)
            maximum = 0.; mismatch = count = handovers = 0
            previous_mask = np.zeros(20,bool)
            for line in gzip.open(file.parent/'actual_trace.jsonl.gz','rt',encoding='utf8'):
                row = json.loads(line); feed = row['motor_input']; t = row['time_s']-.02
                args = (feed['angles_deg'],feed['feedback_time_s'],t)
                a, am = source.advance(*args); b,bm = embedded.advance(*args)
                maximum = max(maximum,float(np.degrees(abs(a-b)).max()))
                mismatch += int(not np.array_equal(am,bm))
                # The changed observer need not reproduce old commands. The
                # saved states are not a new closed-loop dynamics experiment.
                archived_target_difference=max(archived_target_difference,float(np.degrees(abs(b-np.array(row['target_rad']))).max()))
                archived_mask_difference += int(not np.array_equal(bm,row['enabled']))
                if feed['feedback_time_s'] != last_feedback:
                    active=np.flatnonzero(row['enabled'])+1
                    observation=observer.observe(feed['angles_deg'],(int(active[0]),int(active[-1])),feed['feedback_time_s'])
                    if t>=options.preparation_s:
                        actual_observations += 1
                        invalid_after_prepare += int(not observation['valid'])
                        if not observation['valid']:
                            invalid_since=feed['feedback_time_s'] if invalid_since is None else invalid_since
                        elif invalid_since is not None:
                            longest_invalid_s=max(longest_invalid_s,feed['feedback_time_s']-invalid_since)
                            invalid_since=None
                    last_feedback=feed['feedback_time_s']
                if not np.array_equal(bm,previous_mask):
                    bus.prepare_enabled(feed['angles_deg'],bm,configure_mask=bm & ~previous_mask)
                    self.assertEqual([port.registers[sid][0x28] for sid in real.SERVO_IDS],bm.astype(int).tolist())
                    handovers += 1
                bus.send_targets(np.degrees(b),bm)
                packet = port.packets[-1]
                self.assertEqual(list(packet[7:-1:7]),[real.SERVO_IDS[i] for i in np.flatnonzero(bm)])
                previous_mask = bm.copy(); count += 1
            results.append(dict(record=file.parent.name,samples=count,target_maximum_error_deg=maximum,
                mask_mismatch_frames=mismatch,serial_mask_updates=handovers,
                archived_target_difference_deg=archived_target_difference,archived_mask_difference_frames=archived_mask_difference,
                actual_feedback_samples_after_prepare=actual_observations, actual_invalid_samples_after_prepare=invalid_after_prepare,
                longest_invalid_interval_s=longest_invalid_s, counterfactual_fault=source.fault,
                closed_loop_validation=False))
            self.assertEqual(mismatch,0);self.assertLess(maximum,1e-8)
            if invalid_since is not None:
                longest_invalid_s=max(longest_invalid_s,last_feedback-invalid_since)
                results[-1]['longest_invalid_interval_s']=longest_invalid_s
            self.assertLessEqual(longest_invalid_s,options.maximum_untrusted_s)
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT/'simulation_record_replay.json').write_text(json.dumps(dict(hardware_run=False,new_physics=False,records=results),indent=2),encoding='utf8')

    def test_hotkeys_and_idle_selection_do_not_enable(self):
        c = ClockController();c.command('gait_feeding')
        self.assertTrue(c.feeding_selected);self.assertFalse(c.touched_hardware)
        self.assertEqual(real.decode_hotkey(52,True),'gait_feeding')
        self.assertEqual(real.decode_hotkey(ord('Y'),True),'feed_return')
        self.assertEqual(real.decode_hotkey(ord('L'),True),'feed_tick')
        self.assertIsNone(real.decode_hotkey(52,False))
        self.assertEqual(len(c.port.packets),0)

    def test_feedback_required_with_preview_closed(self):
        c = ready();self.assertFalse(c.preview_feedback)
        self.assertEqual(set(c.states),set(range(20)))
        c.command('feed_mm=258');c.command('back')
        self.assertEqual(c.feeding.options.requested_transfer_m,-.258)
        self.assertAlmostEqual(c.feeding.goal_speed_tolerance_m_s,.004)
        c.cycle();self.assertEqual(set(c.states),set(range(20)))

    def test_pause_and_forced_step_do_not_consume_pause(self):
        c = ready();c.command('step')
        for _ in range(400):c.cycle()
        self.assertIsNotNone(c.feeding.baseline)
        c.command('pause');p=c.feeding.position;targets=c.targets.copy();stage=c.feeding.stage
        c.t += 10.;c.refresh_compliance_feedback();c.tick(.02)
        self.assertEqual(c.feeding.position,p);self.assertEqual(c.targets,targets)
        c.command('feed_tick');c.tick(.02)
        self.assertAlmostEqual(c.feeding.position-p,real.FEED_FORCE_STEP_JOINTS)
        self.assertLessEqual(float(np.max(abs(np.array(c.targets)-targets))),.90001)
        for _ in range(100):
            if c.feeding_paused:break
            c.cycle()
        self.assertTrue(c.feeding_paused);self.assertIsNone(c.feeding.fault)
        self.assertEqual(c.feeding.stage,stage)

    def test_return_repeated_task_and_emergency_release(self):
        c=ready();c.command('back')
        for _ in range(400):c.cycle()
        c.command('feed_return');self.assertEqual(c.feeding.stage,'feed_return')
        self.assertEqual(c.feeding.goal_m,c.feeding.baseline)
        c.cycle();old_id=c.feeding_action_id;c.command('step')
        self.assertGreater(c.feeding_action_id,old_id);self.assertEqual(c.feeding.stage,'acquire')
        self.assertEqual(c.feeding.targets.tolist(),np.radians(c.targets).tolist())
        c.command('release')
        self.assertTrue(all(c.port.registers[sid][0x28]==0 for sid in real.SERVO_IDS))
        self.assertIsNone(c.feeding);self.assertFalse(c.feeding_selected)

    def test_paused_feedback_keeps_segments_ready_for_human_paced_single_steps(self):
        c=ready();c.command('step')
        for _ in range(400):c.cycle()
        c.command('pause')
        before=(c.feeding.position,c.feeding.baseline,c.targets.copy(),c.enabled.copy())
        # Multiple real encoder samples arrive while targets remain frozen.
        for _ in range(12):
            c.t += .2;c.refresh_compliance_feedback();c.tick(.02)
        self.assertTrue(c.feeding.observation['valid'])
        self.assertEqual(before,(c.feeding.position,c.feeding.baseline,c.targets,c.enabled))
        quality=c.feeding.observation['end_quality']
        self.assertGreaterEqual(quality['candidates'][0]['samples'],3)
        lines=real.feeding_direction_lines(c.feeding.observation)
        self.assertTrue(any('直线度=' in line and '稳定=' in line for line in lines))
        self.assertTrue(any('两侧夹角=' in line for line in lines))
        c.command('feed_tick');c.tick(.02)
        for _ in range(100):
            if c.feeding_paused:break
            c.cycle()
        self.assertTrue(c.feeding.observation['valid'])
        self.assertTrue(c.feeding_paused);self.assertIsNone(c.feeding.fault)

    def test_switch_to_basic_preserves_sent_targets(self):
        c=ready();c.command('step');c.cycle()
        before=c.targets.copy();mask=c.enabled.copy()
        c.command('gait_basic');c.cycle()
        self.assertFalse(c.feeding_selected)
        self.assertEqual(c.targets,before);self.assertEqual(c.enabled,mask)

    def test_complete_roundtrip_and_optional_manual_return(self):
        for automatic in (True,False):
            with patch.object(real,'FEED_AUTOMATIC_RETURN',automatic):
                c=ready();c.command('back')
                for _ in range(2500):
                    c.cycle()
                    if c.feeding_paused:break
                self.assertIsNone(c.feeding.fault)
                if not automatic:
                    self.assertEqual(c.feeding.stage,'complete')
                    self.assertAlmostEqual(c.feeding.goal_m-c.feeding.baseline,-real.FEED_DISTANCE_M)
                    c.command('feed_return')
                    for _ in range(2500):
                        c.cycle()
                        if c.feeding_paused:break
                self.assertTrue(c.feeding.completed)
                self.assertTrue(c.feeding_paused)

    def test_manual_retry_preserves_fault_evidence_and_current_targets(self):
        c=ready();c.command('step');c.cycle()
        c.feeding.fault='scripted observation failure';c.pause_feeding()
        targets=c.targets.copy();c.t += 4.;c.command('toggle')
        self.assertIsNone(c.feeding.fault);self.assertEqual(c.targets,targets)
        self.assertEqual(c.feeding.events[-2]['previous_fault'],'scripted observation failure')
        self.assertAlmostEqual(c.feeding.stage_since,c.t-c.feeding_clock_origin)

    def test_basic_parameter_edit_does_not_stall_feeding(self):
        c=ready();c.command('step');c.cycle()
        c.command('amplitude=70 omega=0.8')
        self.assertFalse(c.forming)
        self.assertEqual(c.feeding.options.amplitude_deg,real.FEED_AMPLITUDE_DEG)
        c.command('back');self.assertEqual(c.feeding.stage,'acquire')

    def test_preview_and_standalone_copy(self):
        c=ready();c.command('back');c.cycle()
        row=real.preview_payload(c)
        self.assertEqual(row['gait'],'主动送链');self.assertIsNotNone(row['feeding'])
        self.assertIsNone(row['smooth_section'])  # Feeding uses polygon length; curves are display reconstruction.
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'controller.py';p.write_text((ROOT/'codes/Rolling_Arc_Real.py').read_text(encoding='utf8'),encoding='utf8')
            result=subprocess.run([sys.executable,'-I','-c',
                "import runpy; m=runpy.run_path('controller.py',run_name='offline_import'); assert 'ContinuousFeeding' in m"],
                cwd=folder,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_debug_entry_shares_action_without_advancing_physics(self):
        from codes import Rolling_Arc_Debug_Mujoco as debug
        with patch.object(debug.real, 'open_robot_serial',side_effect=AssertionError('No COM port')):
            session=debug.DebugSession()
            c=session.controller
            c.command('gait_feeding');c.command('apply')
            # Software dispatch check at the model's initialized arc pose;
            # do not claim formation/settling physics from this fixture.
            c.forming=False;c.settle_remaining=0.;c.pending_gait=None
            c.command('back');c.tick(.02)
            self.assertTrue(c.feeding_selected)
            self.assertEqual(c.feeding.options.requested_transfer_m,-real.FEED_DISTANCE_M)
            self.assertEqual(debug.decode_sim_hotkey(52,True),'gait_feeding')
            self.assertEqual(c.phase,0.);self.assertEqual(session.sim.data.time,0.)
            self.assertEqual(c.tick_feeding.__func__,debug.real.RollingArcController.tick_feeding)
            row=c.snapshot();json.dumps(row)
            c.command('release')
            self.assertFalse(session.bus.enabled.any())


if __name__=='__main__':
    with contextlib.redirect_stdout(io.StringIO()):
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FeedingPortTests))
    raise SystemExit(0 if result.wasSuccessful() else 1)
