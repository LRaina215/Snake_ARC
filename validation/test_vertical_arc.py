"""New reference integrator, near-vertical frames, shared adapter handover."""
import unittest,math
from dataclasses import replace
import numpy as np
from snake_control.curve_gait import CurveGait,helix
from snake_control.bent_helix_curve import BentHelixCurve
from snake_control.acute_arc_transition import AcuteArcTransition, AcuteOptions
from snake_control.vertical_arc_transition import VerticalArcTransition,VerticalArcOptions
from snake_control.helix_arc_transition import HelixArcTransition
from snake_control.vertical_ring_manual import VerticalRingManual
from snake_control.vertical_ring_manual import vertical_short_options
from validation.acute_arc_experiment import summarize_straight


class VerticalArcTests(unittest.TestCase):
    def setUp(self):
        self.lengths=[.025]+[.072]*17+[.091];self.g=CurveGait(18,lengths=self.lengths)

    def test_bent_reference_straight_limit_matches_existing_helix_integral(self):
        for handedness in (-1, 1):
            b=BentHelixCurve(.094,.13,handedness=handedness);s,k,_=b.field(-.1,self.g.body_length)
            for roll in (0.,.4,1.57):
                a=self.g.sampled_targets(s,k,roll)
                q=self.g.targets([helix(self.g.body_length,.094,.13,handed=handedness)],roll)
                self.assertLess(np.max(abs(np.degrees(a-q))),.3)

    def test_manual_takeover_freezes_clock_during_hold_or_stale_read(self):
        q = np.degrees(self.g.targets([helix(self.g.body_length, .094, .13)], .4))
        manual = VerticalRingManual(q, self.lengths, helix_targets_deg=q, step_s=.04)
        np.testing.assert_allclose(manual.targets, q)
        self.assertFalse(manual.advance(.02, 3., 3., q))
        manual.step_once()
        self.assertFalse(manual.advance(.02, 3., 2., q))
        self.assertTrue(manual.feedback_wait)
        self.assertTrue(manual.advance(.02, 3., 3., q))
        self.assertTrue(manual.advance(.02, 3.02, 3., q))
        self.assertFalse(manual.advance(.02, 3.04, 3.04, q))
        self.assertAlmostEqual(manual.core.time, .04)
        manual.toggle_continuous()
        self.assertFalse(manual.advance(.02, 3.04, 2.9, q))
        manual.pause()
        self.assertFalse(manual.running)

    def test_standalone_vertical_controller_matches_research_at_outlet(self):
        from codes import Rolling_Arc_Real as real

        angles = np.degrees(self.g.targets([helix(self.g.body_length, .094, .13)], .4))
        shared = VerticalRingManual(angles, self.lengths, vertical_short_options(), helix_targets_deg=angles)
        standalone = real.VerticalRingManual(angles, self.lengths, real.vertical_short_options(), helix_targets_deg=angles)
        for manual in (shared, standalone):
            core = manual.core
            core.wrap_radius -= .003
            core.pitch = .075
            core.section_ring = core.make_section()
            core.vertical_section = core.section_ring.section_field(-.055)[:2]
            core.helix_length = core.vertical_section[0][-1]
            core.enabled = core.ring_mask()
            core.enter('vertical_ring')
            self.assertTrue(manual.begin_bend(.24))
            self.assertTrue(manual.confirm_outlet())
            manual.toggle_continuous()
        for i in range(100):
            now = i*.02
            self.assertTrue(shared.advance(.02, now, now, angles))
            self.assertTrue(standalone.advance(.02, now, now, angles))
            np.testing.assert_allclose(shared.targets, standalone.targets, rtol=0., atol=1e-9)
            np.testing.assert_array_equal(shared.enabled, standalone.enabled)

    def test_frame_finite_and_continuous_across_vertical_bend_and_horizontal(self):
        b=BentHelixCurve(.094,.13);gammas=[]
        for u in np.linspace(-.05,.75,41):
            s,k,_=b.field(u,.907);gammas.append(b.gamma)
            self.assertTrue(np.isfinite(k).all());self.assertAlmostEqual(s[-1],.907)
        self.assertLess(np.max(abs(np.diff(gammas))),.2)

    def test_acute_helix_keeps_full_curve_and_drives_tail(self):
        q = self.g.targets([helix(self.g.body_length, .094, .13)], .4)
        controller = AcuteArcTransition(q, self.lengths)
        controller.branch('helix')
        for _ in range(1000):
            q, enabled = controller.step(.02, q)
            self.assertTrue(enabled.all())
        controller.begin_bend(-.055)
        q, enabled = controller.step(.02, q)
        self.assertTrue(enabled.all())
        self.assertAlmostEqual(controller.field[0][-1], self.g.body_length)
        self.assertGreater(np.max(abs(q[-3:])), .3)

    def test_progressive_tail_releases_only_one_joint_per_fresh_read(self):
        q = self.g.targets([helix(self.g.body_length, .094, .13)], .4)
        controller = AcuteArcTransition(q, self.lengths, AcuteOptions(progressive_tail_release=True))
        controller.branch('arc')
        controller.enter('release_tail')
        # Let the desired boundary advance while feedback is stale, then check
        # that it cannot turn off several pending joints on the next read.
        controller.helix_length = .75
        _, first = controller.step(.02, q, 1.)
        _, repeat = controller.step(.02, q, 1.)
        _, second = controller.step(.02, q, 1.2)
        np.testing.assert_array_equal(np.flatnonzero(~first), [17])
        np.testing.assert_array_equal(repeat, first)
        np.testing.assert_array_equal(np.flatnonzero(~second), [16, 17])

    def test_section_feedback_updates_once_per_encoder_frame(self):
        q = self.g.targets([helix(self.g.body_length, .094, .13)], .4)
        c = AcuteArcTransition(q, self.lengths, replace(vertical_short_options(), section_feedback_weight=.35))
        c.branch('arc')
        c.wrap_radius -= .003
        c.pitch = .075
        c.section_ring = c.make_section()
        c.vertical_section = c.section_ring.section_field(-.055)[:2]
        c.helix_length = c.vertical_section[0][-1]
        c.enabled = c.ring_mask()
        c.enter('vertical_ring')
        _, mask = c.step(.02, q, 1.)
        positions, correction = c.section_feedback
        changed = q * .9
        target, repeated_mask = c.step(.02, changed, 1.)
        np.testing.assert_array_equal(c.section_feedback[1], correction)
        np.testing.assert_array_equal(repeated_mask, mask)
        np.testing.assert_array_equal(target[~mask], changed[~mask])
        c.step(.02, changed, 1.2)
        np.testing.assert_array_equal(c.section_feedback[0], positions)
        self.assertFalse(np.array_equal(c.section_feedback[1], correction))

    def test_straight_review_distinguishes_transient_and_sustained_clearance(self):
        rows = [dict(stage='vertical_ring', time_s=i*.1, ring_center_m=[0.,0.,1.+i*.01],
                     last_envelope_z_m=.99 if i in (0, 4) else 1.01,
                     ring_shape=True, supporting_links=[2, 5], floor_force_n=0., self_contacts=[],
                     normal_n={'vertical': 20.}) for i in range(20)]
        result = summarize_straight(rows, 2.*math.pi, 1., 'arc')
        self.assertAlmostEqual(result['first_clear_time_s'], .1)
        self.assertFalse(result['continuation']['above_feature_all'])
        self.assertAlmostEqual(result['sustained_continuation']['time_s'][0], .5)
        self.assertTrue(result['sustained_continuation']['above_feature_all'])

    def test_shared_reference_application_and_horizontal_entry_after_localisation(self):
        q=self.g.targets([helix(self.g.body_length,.094,.13)],.4)
        c=VerticalArcTransition(q,self.lengths,VerticalArcOptions(phase_rate_rad_s=-.5,retained_turns=1.5,finish_after_pilot=False))
        self.assertIs(type(c).apply_reference,HelixArcTransition.apply_reference)
        for i in range(2600):q,mask=c.step(.02,q)
        c.begin_bend()
        for i in range(10):q,mask=c.step(.02,q)
        c.confirm_horizontal(q)
        self.assertEqual(c.horizontal.stage,'reverse_unwrap')
        self.assertLessEqual(np.max(abs(np.degrees(c.targets))),88.01)


if __name__=='__main__':unittest.main()
