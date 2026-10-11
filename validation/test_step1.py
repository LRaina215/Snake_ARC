"""Tests for consequential gait, actuator-disable and calibration boundaries."""
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import unittest
import numpy as np
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation, ROOT
from snake_control.hardware_preview import preview_frame


class Step1Tests(unittest.TestCase):
    def test_invalid_parameters(self):
        for override in ({'first_joint':0}, {'last_joint':21}, {'joints':20.5}, {'direction':0},
                         {'omega_rad_s':float('nan')}, {'amplitude_deg':91}, {'torque_limit_nm':3.}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                replace(ArcParameters(), **override)

    def test_original_rolling_wave_and_direction(self):
        p = ArcParameters()
        np.testing.assert_allclose(p.targets(0)[[8,10,12]], math.pi/3)
        self.assertTrue(np.all(p.targets(0)[~p.enabled] == 0))
        np.testing.assert_allclose(p.targets(math.pi/2)[[7,9,11]], math.pi/3)
        np.testing.assert_allclose(replace(p,direction=-1).targets(math.pi/2)[[7,9,11]], -math.pi/3)

    def test_disable_has_no_restoring_torque(self):
        p = ArcParameters()
        sim = ArcSimulation('straight', p)
        sim.data.qpos[sim.drive.qadr[~p.enabled]] = .4
        sim.data.qvel[sim.drive.vadr[~p.enabled]] = .2
        sim.step()
        np.testing.assert_array_equal(sim.data.ctrl[~p.enabled], 0)
        np.testing.assert_array_equal(sim.data.actuator_force[~p.enabled], 0)
        self.assertGreater(np.max(abs(sim.data.qfrc_passive[sim.drive.vadr[~p.enabled]])), 0)

    def test_hold_and_resume_phase(self):
        sim = ArcSimulation('straight', ArcParameters())
        for _ in range(5):
            sim.step(.7)
        self.assertEqual(sim.measure()['phase_rad'], .7)
        np.testing.assert_array_equal(sim.measure()['target_rad'], sim.p.targets(.7))
        sim.step(.71)
        self.assertEqual(sim.measure()['phase_rad'], .71)

    def test_hardware_mapping_and_passive_mask(self):
        p = ArcParameters()
        f = preview_frame(p, math.pi/2, ROOT/'configs/hardware_20j_reference.json')
        self.assertEqual(sum(f['enabled']), 6)
        self.assertAlmostEqual(f['target_protocol_deg'][7], -60)
        packet = bytes.fromhex(f['torque_mask_packet_hex'])
        self.assertEqual(packet[:3], b'\xff\xff\xfe')
        self.assertEqual(packet[3], len(packet)-4)
        self.assertEqual(sum(packet[2:]) & 255, 255)
        self.assertEqual(list(packet[8:-1:2]), [int(e) for e in p.enabled])


if __name__ == '__main__':
    unittest.main()
