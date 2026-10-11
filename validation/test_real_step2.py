"""Offline Step2 protocol/control checks; no physical ports or invented force calibration."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import math
import time
import unittest
from unittest.mock import patch
from codes import Rolling_Arc_Real as real
from snake_control.arc_compliance import SlidingArcCompliance

class RecordingSerial(real.DryRunSerial):
    def __init__(self):super().__init__();self.packets=[]
    def write(self,packet):self.packets.append(packet);return super().write(packet)

def prepared():
    port=RecordingSerial();c=real.RollingArcController(real.FeetechBus(port));c.command('apply')
    for _ in range(320):c.tick(.02)
    c.command('support=9:11');c.command('compliance')
    return c,port

class RealStep2Tests(unittest.TestCase):
    def test_core_is_shared_and_signed_feedback_has_no_force_units(self):
        self.assertIs(SlidingArcCompliance,real.SlidingArcCompliance)
        port=RecordingSerial();regs=port.registers[9]
        regs[0x3C:0x3E]=(1024+73).to_bytes(2,'little')
        regs[0x45:0x47]=(32768+25).to_bytes(2,'little')
        states=real.FeetechBus(port).read_states()
        self.assertEqual(states[8]['load_raw'],-73);self.assertEqual(states[8]['current_raw'],-25)
        self.assertNotIn('force_n',states[8]);self.assertNotIn('torque_nm',states[8])

    def test_no_support_rejects_takeover_without_enabling_hardware(self):
        port=RecordingSerial();c=real.RollingArcController(real.FeetechBus(port))
        with self.assertRaisesRegex(ValueError,'支撑观测'):c.command('compliance')
        self.assertTrue(all(not port.registers[s][0x28] for s in real.SERVO_IDS))
        self.assertFalse(c.touched_hardware)

    def test_adopt_offcenter_posture_without_resetting_measured_angles(self):
        port=RecordingSerial()
        measured=real.joint_angles_deg(.7,50.,4,9)
        for i,angle in enumerate(measured):port.registers[i+1][0x38:0x3A]=real.angle_to_raw(i,angle).to_bytes(2,'little')
        c=real.RollingArcController(real.FeetechBus(port));c.command('support=5:7');c.command('compliance')
        self.assertAlmostEqual(c.phase,.7,delta=.01)
        self.assertLess(sum(c.enabled),20)
        for i,e in enumerate(c.enabled):
            if e:self.assertAlmostEqual(c.targets[i],measured[i],delta=.1)
        self.assertEqual(c.mode,'hold')

    def test_mask_transfer_preloads_new_joints_and_disables_exited_joints(self):
        c,port=prepared();phase=c.phase
        c.command('support=4:6')
        for _ in range(150):
            c.refresh_compliance_feedback()
            before=c.targets.copy();old_mask=c.enabled.copy();c.tick(.02)
            for i,e in enumerate(c.enabled):
                if e and old_mask[i]:self.assertLessEqual(abs(c.targets[i]-before[i]),.9+1e-9)
            self.assertEqual([port.registers[s][0x28] for s in real.SERVO_IDS],[int(e) for e in c.enabled])
        self.assertEqual(c.phase,phase)
        self.assertLess(sum(c.enabled),20)
        self.assertTrue(c.first<=4<=6<=c.last)
        self.assertGreater(len(c.compliance.events),0)
        # For each newly enabled ID, its previous position is preloaded before enable.
        enabled_previous=[False]*20;targeted=set()
        for packet in port.packets:
            if packet[4]!=0x83:continue
            if packet[5]==0x2A:targeted.update(packet[7:-1:7])
            if packet[5]==0x28:
                mask=[bool(x) for x in packet[8:-1:2]]
                for i,e in enumerate(mask):
                    if e and not enabled_previous[i]:self.assertIn(i+1,targeted)
                enabled_previous=mask;targeted=set()

    def test_estimator_raw_threshold_hold_and_stale_feedback(self):
        c,port=prepared();c.command('support=auto');c.command('run')
        for _ in range(40):c.refresh_compliance_feedback();c.tick(.02)
        self.assertTrue(c.compliance.phase_paused)  # Mock has no load/contact evidence.
        phase=c.phase
        for _ in range(5):c.tick(.02)
        self.assertEqual(c.phase,phase)
        with patch.object(real.time,'monotonic',return_value=c.feedback_wall_s+2.):
            with self.assertRaises(TimeoutError):c.tick(.02)
        with patch.object(c.bus,'read_states',return_value=c.states),patch.object(real.time,'monotonic',side_effect=[0.,1.]):
            with self.assertRaisesRegex(TimeoutError,'读取过慢'):c.refresh_compliance_feedback()
        c.command('release')
        self.assertTrue(all(not port.registers[s][0x28] for s in real.SERVO_IDS))

    def test_ctrl_e_switch_preserves_phase_and_parameter_errors_are_atomic(self):
        c,port=prepared();self.assertEqual(real.decode_hotkey(69,True),'compliance')
        c.command('run');c.refresh_compliance_feedback()
        for _ in range(5):c.tick(.02)
        phase=c.phase;targets=c.targets.copy();c.command('compliance')
        self.assertEqual(c.phase,phase);self.assertEqual(c.targets,targets)
        before=c.snapshot()
        with self.assertRaises(ValueError):c.command('gain=0.03 support=0:9')
        self.assertEqual(c.snapshot(),before)

if __name__=='__main__':unittest.main()
