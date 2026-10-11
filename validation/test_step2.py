"""Meaningful Step2 boundaries: support switching, measured yielding, motor disable."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import math
import unittest
import numpy as np
from snake_control.arc_compliance import SlidingArcCompliance, SlidingArcOptions, support_seed
from snake_control.rolling_arc import ArcParameters
from snake_control.adaptive_simulation import AdaptiveArcSimulation

class Step2Tests(unittest.TestCase):
    def test_seed_uses_one_connected_support_region(self):
        self.assertIsNone(support_seed([0.]*20))
        scores=[0.]*20;scores[2]=3;scores[3]=4;scores[17]=1
        first,last=support_seed(scores)
        self.assertTrue(first<=3<=4<=last)
        self.assertLess(last,18)

    def test_measured_envelope_yields_and_does_not_recursively_tighten(self):
        q=[0.]*20;q[7:13]=[0.,40.,0.,40.,0.,40.]
        c=SlidingArcCompliance(q,0.,(8,13))
        support=[False]*20;support[8]=support[10]=True
        for _ in range(150):target,enabled=c.update(q,support,.02,0.)
        # Distant observations do not alter the amplitude law; actual orthogonal
        # angle envelope 40 degrees, gain .02 => bounded equilibrium 40.8 degrees.
        self.assertAlmostEqual(c.amplitudes[9],40.8,delta=.1)
        self.assertLessEqual(max(c.amplitudes),60.)
        self.assertLessEqual(max(abs(a-b) for a,b in zip(target,c.targets)),1e-12)

    def test_hysteresis_transfer_and_contact_loss_hold(self):
        q=[0.]*20;c=SlidingArcCompliance(q,0.,(8,13))
        support=[False]*20;support[7]=support[8]=True
        for _ in range(60):c.update(q,support,.02,0.)
        self.assertEqual(c.window,(7,12))  # Keep a sufficient window.
        shifted=[False]*20;shifted[5]=shifted[6]=True
        for _ in range(10):c.update(q,shifted,.02,0.)
        self.assertEqual(c.window,(7,12))  # No instantaneous mask churn.
        old=c.targets.copy()
        for _ in range(100):
            target,enabled=c.update(q,shifted,.02,0.)
            self.assertLessEqual(max(abs(a-b) for a,b in zip(target,old)),.9+1e-9)
            old=target
        self.assertGreater(len(c.events),0)
        self.assertTrue(c.window[0]<=5<=6<=c.window[1])
        self.assertLess(sum(enabled),20)
        for _ in range(30):c.update(q,[False]*20,.02,0.)
        self.assertTrue(c.phase_paused)

    def test_dynamic_passive_mask_has_zero_motor_torque(self):
        sim=AdaptiveArcSimulation('straight',ArcParameters())
        sim.data.qpos[sim.drive.qadr]=.4;sim.data.qvel[sim.drive.vadr]=.2
        mask=np.zeros(20,bool);mask[2:8]=True
        sim.drive.apply_targets(sim.data,np.ones(20)*.8,mask)
        import mujoco
        mujoco.mj_step(sim.model,sim.data)
        np.testing.assert_array_equal(sim.data.actuator_force[~mask],0.)
        self.assertGreater(np.max(abs(sim.data.actuator_force[mask])),0.)

    def test_reject_invalid_feedback(self):
        c=SlidingArcCompliance([0.]*20,0.,(8,13))
        for q,dt in (([0.]*19,.02),([math.nan]*20,.02),([0.]*20,0.)):
            with self.assertRaises(ValueError):c.update(q,[False]*20,dt,0.)

if __name__=='__main__':unittest.main()
