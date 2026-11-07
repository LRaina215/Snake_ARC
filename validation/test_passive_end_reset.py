"""End-zero serial ordering, torque release and operator interruption, no COM port."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from validation.test_real_controller import real, RecordingSerial


class PassiveEndResetTests(unittest.TestCase):
    def controller(self):
        port=RecordingSerial()
        c=real.RollingArcController(real.FeetechBus(port))
        c.command('apply')
        c.forming=False;c.settle_remaining=0.
        c.targets=real.joint_angles_deg(0.,60.,c.first,c.last)
        c.bus.send_targets(c.targets,c.enabled)
        # A passive bent end must be preloaded at its actual angle, never at zero.
        raw=real.angle_to_raw(1,25.)
        port.registers[2][0x38]=raw&255;port.registers[2][0x39]=raw>>8
        port.packets.clear()
        return c,port

    def test_reset_preserves_bearing_and_preloads_actual_before_torque(self):
        c,port=self.controller()
        bearing=c.enabled.copy();targets=np.asarray(c.targets).copy()
        with contextlib.redirect_stdout(io.StringIO()):c.tick(.02)
        self.assertTrue(all(c.enabled))
        np.testing.assert_array_equal(np.asarray(c.targets)[bearing],targets[bearing])
        sync=[p for p in port.packets if p[4]==0x83]
        self.assertEqual([p[5] for p in sync[:4]],[0x2A,0x30,0x29,0x28])
        preload={sync[0][i]:int.from_bytes(sync[0][i+1:i+3],'little') for i in range(7,len(sync[0])-1,7)}
        self.assertAlmostEqual(real.raw_to_angle(1,preload[2]),25.,delta=.1)
        self.assertTrue(all(not bearing[sid-1] for sid in preload))
        # The fake bus follows commands. Refresh at the same 200 ms cadence.
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(1,110):
                now=i*.02
                with patch.object(c,'motor_sample_clock',return_value=(now,c.feedback_motor_clock_s)):
                    c.tick(.02)
        self.assertIsNone(c.end_reset)
        self.assertEqual(c.enabled,bearing)
        self.assertEqual(c.end_reset_last['reason'],'zero_then_disabled')
        self.assertTrue(all(port.registers[sid][0x28]==int(bearing[sid-1]) for sid in real.SERVO_IDS))

    def test_pause_or_release_never_leaves_passive_ends_enabled(self):
        for command in ('pause','release','gait_basic'):
            with self.subTest(command=command),contextlib.redirect_stdout(io.StringIO()):
                c,_=self.controller();bearing=c.enabled.copy();c.tick(.02)
                c.command(command)
                self.assertIsNone(c.end_reset)
                self.assertFalse(c.end_reset_pending)
                self.assertEqual(c.enabled,[False]*18 if command=='release' else bearing)

    def test_unmoving_end_times_out_and_releases_without_reporting_zero_success(self):
        c,_=self.controller()
        with contextlib.redirect_stdout(io.StringIO()):
            c.tick(.02)
            measured=np.zeros(18);measured[1]=25.
            with patch.object(c,'measured_angles',return_value=measured):
                for _ in range(105):c.tick(.02)
        self.assertEqual(c.end_reset_last['reason'],'timeout_then_disabled')
        self.assertAlmostEqual(c.end_reset_last['actual_max_deg'],25.)
        self.assertEqual(sum(c.enabled),6)


if __name__=='__main__':unittest.main()
