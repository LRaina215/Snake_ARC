"""Measured 18-servo dimensions: nonuniform curve integration and shared mode entries.

No real serial port. These checks validate geometry, not hardware support/friction.
"""
import contextlib
import io
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from validation.test_real_controller import real, prepared_controller, ROOT
from snake_control import curve_gait, gravity_global_shift, gravity_moment, helix_arc_transition
from codes import Rolling_Arc_Debug_Mujoco as debug


LENGTHS=[.025]+[.072]*17+[.091]


class MeasuredDimensionsTests(unittest.TestCase):
    def test_real_dimensions_and_reconstruction_use_axis_distances(self):
        np.testing.assert_allclose(real.LINK_LENGTHS_M,LENGTHS,atol=1e-12)
        self.assertAlmostEqual(sum(LENGTHS),1.340)
        self.assertAlmostEqual(real.FEED_DISTANCE_M,.072)
        points=real.forward(np.zeros(18),real.LINK_LENGTHS_M)
        np.testing.assert_allclose(np.linalg.norm(np.diff(points,axis=0),axis=1),LENGTHS,atol=1e-12)
        # Geometry must not silently remove the head servo or change its mapping.
        self.assertEqual(real.SERVO_IDS,list(range(1,19)))

    def test_nonuniform_helix_integrals_and_inverse_fit(self):
        for module,segment in ((curve_gait,curve_gait.helix),(real,real.helix)):
            with self.subTest(module=module.__name__):
                gait=module.CurveGait(18,lengths=LENGTHS)
                h=segment(gait.body_length,.120,.130,-1)
                phase=.43
                q=gait.targets([h],phase)
                expected=[]
                for j in range(18):
                    a,b=gait.s[j],gait.s[j+2]
                    if j%2==0:
                        value=h.kappa*(math.sin(phase+h.tau*b)-math.sin(phase+h.tau*a))/h.tau
                    else:
                        value=h.kappa*(math.cos(phase+h.tau*a)-math.cos(phase+h.tau*b))/h.tau
                    expected.append(value)
                np.testing.assert_allclose(q,expected,atol=1e-12)
                fit=(real.fit_helix if module is real else helix_arc_transition.fit_helix)(q,.072,LENGTHS)
                self.assertAlmostEqual(fit['radius'],.120,places=8)
                self.assertAlmostEqual(fit['pitch'],.130,places=8)
                self.assertAlmostEqual(fit['phase'],phase,places=8)

    def test_measured_shift_preserves_initial_pose_and_uses_true_material_extent(self):
        for moment,shift in ((gravity_moment,gravity_global_shift),(real,real)):
            with self.subTest(module=moment.__name__):
                gait=moment.GravityMomentGait(18,lengths=LENGTHS)
                q=np.linspace(-.3,.4,18)
                curve=shift.MeasuredGravityShift(gait,.120,.4,q,8)
                np.testing.assert_allclose(curve.targets(0.),q,atol=1e-12)
                self.assertAlmostEqual(curve.support_end_m,.025+8*.072)
                self.assertAlmostEqual(curve.shift_limit_m,1.340-curve.support_end_m)
                self.assertTrue(np.isfinite(curve.targets(min(.05,curve.shift_limit_m))).all())

    def test_mode_five_six_seven_use_nonuniform_dimensions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c,_=prepared_controller()
            for helix_only in (True,False):
                core=c.read_gravity_entry(False,helix_only=helix_only)
                np.testing.assert_allclose(core.gait.gait.lengths,LENGTHS,atol=1e-12)
                self.assertAlmostEqual(core.gait.L,1.340)
            gait=real.GravityMomentGait(18,lengths=LENGTHS)
            q=np.degrees(gait.targets(.4,.120)[0])
            c.states={i:{'angle_deg':float(a)} for i,a in enumerate(q)}
            with patch.object(c,'refresh_compliance_feedback',return_value=None):
                reverse=c.read_helix_arc_entry()
            np.testing.assert_allclose(reverse.core.curve.lengths,LENGTHS,atol=1e-12)
            np.testing.assert_allclose(reverse.core.curve.targets(
                reverse.core.reverse_segments(0.),reverse.core.phase),np.radians(q),atol=1e-7)

    def test_manual_sim_uses_loaded_model_without_rewriting_real_configuration(self):
        model=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o18_Tee_straight.xml'))
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        with patch.object(debug.real,'LINK_LENGTHS_M',list(LENGTHS)),\
             patch.object(debug.real,'FEED_DISTANCE_M',.072):
            debug.configure_sim_geometry(SimpleNamespace(model=model,data=data))
            np.testing.assert_allclose(debug.real.LINK_LENGTHS_M,[.0645]*19,atol=1e-10)
            self.assertAlmostEqual(debug.real.FEED_DISTANCE_M,.0645)
            debug.real.FEED_DISTANCE_M=.150  # Explicit custom distances survive model setup.
            debug.configure_sim_geometry(SimpleNamespace(model=model,data=data))
            self.assertAlmostEqual(debug.real.FEED_DISTANCE_M,.150)
        # Separate real-entry import retains the confirmed physical dimensions.
        np.testing.assert_allclose(real.LINK_LENGTHS_M,LENGTHS,atol=1e-12)


if __name__=='__main__':unittest.main()
