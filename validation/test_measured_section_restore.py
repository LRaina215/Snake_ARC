"""Focused offline checks: reference law, duplicate clocks and motor-mask packets.

No serial device is opened and no new dynamics experiment is run. Synthetic
angle feedback proves command generation, not physical contact or support.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import copy
from dataclasses import replace
import importlib.util
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from codes import Rolling_Arc_Real as real
from codes import Rolling_Arc_Debug_Mujoco as debug
from snake_control import fixed_section_compliance as research
from snake_control.motor_feedback_sliding import MotorFeedbackSliding
from snake_control.motor_shape_observer import PhaseShapeReference
from validation.test_repeated_motor_port import AngleBus, RecordingSerial


def reference_module():
    path = ROOT.parent/'Snake_Robot_Gait/snake_control/pole_compliance.py'
    spec = importlib.util.spec_from_file_location('reference_pole_compliance', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reference_angles(reference, geometry, first, last, phase, gain):
    """Feed our local section to the ORIGINAL repository's independent integrator.

    alpha=0 is the planar limit. Convert metres to its mm protocol, then map
    its local odd/even convention to ARC's global cos/sin convention.
    """
    active_count = last-first+1
    links = np.asarray(real.LINK_LENGTHS_M[first-1:last+2])*1000
    count = len(links)-1  # Includes the passive boundary joint, never actuated.
    winding = 1 if sum(s.turn for s in geometry['curve'] if 1 <= s.joint <= active_count) >= 0 else -1
    segments = [dict(kind='arc' if s.radius else 'line', length_mm=s.length*1000,
                     radius_mm=s.radius*1000, turn_rad=s.turn*winding,
                     joint_index=s.joint if 1 <= s.joint <= count else None,
                     start_mm=s.start*1000, tangent=s.tangent)
                for s in geometry['curve']]
    data = dict(config=dict(joint_count=count, head_length_mm=links[0],
                joints=[dict(joint_index=i+1, axis='Yaw' if i % 2 == 0 else 'Pitch',
                             link_length_mm=length) for i, length in enumerate(links[1:])]),
                curve=dict(valid=True, segments=segments),
                points_mm=geometry['points']*1000,
                projection_axis=dict(point_mm=np.zeros(3), horizontal=[1., 0., 0.],
                                     vertical=[0., 1., 0.], direction=[0., 0., 1.]))
    gains = [gain*.5 if i in (0, active_count-1) else gain for i in range(active_count)]+[0.]*(count-active_count)
    backbone = reference.build_backbone(data, reference.ComplianceParameters(gains, 0.))
    integrals = -backbone.joint_angles(math.pi/4)/math.sqrt(.5)
    factors = [math.cos(phase) if j % 2 else math.sin(phase) for j in range(first, last+1)]
    return integrals[:active_count]*np.asarray(factors)


class MeasuredSectionRestoreTests(unittest.TestCase):
    def test_matches_original_reference_and_ignores_nominal_amplitude(self):
        reference = reference_module()
        for first, last in ((8, 13), (5, 9)):
            for phase in (0., .7, 1.6):
                p = real.SectionReference(first, last, 50., manual_tuning=True)
                measured = p.targets(phase)
                geometry = real.local_section(measured, real.LINK_LENGTHS_M, first, last)
                expected = reference_angles(reference, geometry, first, last, phase, .02)
                outputs = []
                for module in (real, research):
                    for amplitude, legacy_yield in ((30., 0.), (85., .9)):
                        core = module.FixedSectionCompliance(replace(p, amplitude_deg=amplitude),
                            real.LINK_LENGTHS_M, module.SectionOptions(yielding=legacy_yield, correct_discrete_turn=False))
                        # Fail if the nominal generator is consulted during compliance.
                        with patch.object(real.SectionReference, 'targets', side_effect=AssertionError('nominal used')):
                            core.update(measured, phase, 0., .02, reference_targets=np.full(20, 9.))
                        np.testing.assert_allclose(core.last_goal[p.enabled], expected, atol=2e-14)
                        np.testing.assert_array_equal(core.last_goal[~p.enabled], measured[~p.enabled])
                        outputs.append(core.targets)
                for actual in outputs[1:]:
                    np.testing.assert_allclose(actual, outputs[0], atol=2e-14)

    def test_larger_measured_section_reduces_bending_without_recursive_tightening(self):
        p = real.SectionReference(8, 13, 70., manual_tuning=True)
        radii, bending = [], []
        for measured_amplitude in (65., 40.):
            q = replace(p, amplitude_deg=measured_amplitude).targets(.7)
            core = real.FixedSectionCompliance(p, real.LINK_LENGTHS_M)
            core.update(q, .7, 0., .02)
            candidate = core.last_goal.copy()
            radii.append(core.snapshot()['fitted_body_radius_m'])
            bending.append(float(np.linalg.norm(candidate[p.enabled])))
            for t in (.2, .4):
                core.update(q, .7, t, .02)
                np.testing.assert_allclose(core.last_goal, candidate, atol=2e-14)
        self.assertGreater(radii[1], radii[0])
        self.assertLess(bending[1], bending[0])

    def test_duplicate_clock_is_idempotent_and_next_tick_resumes(self):
        for cls, reference in ((real.MotorFeedbackSliding, real.PhaseShapeReference.from_dict(real.MOTOR_PHASE_REFERENCE)),
                               (MotorFeedbackSliding, PhaseShapeReference.from_dict(real.MOTOR_PHASE_REFERENCE))):
            p = real.ManualArcReference(first_joint=8, last_joint=13, settle_s=0.)
            controller = cls(p, real.LINK_LENGTHS_M, reference, execute=False)
            q = np.degrees(p.targets(.7))
            target, mask = controller.advance(q, 1., 1., commanded_phase_rad=.7)
            updates = controller.core.updates
            repeated, repeated_mask = controller.advance(q, 1., 1., commanded_phase_rad=.8)
            np.testing.assert_array_equal(repeated, target)
            np.testing.assert_array_equal(repeated_mask, mask)
            self.assertEqual(controller.phase, .7)
            self.assertEqual(controller.core.updates, updates)
            self.assertFalse(controller.fresh_feedback)
            controller.advance(q, 1., 1.02, commanded_phase_rad=.8)
            self.assertEqual(controller.phase, .8)
            with self.assertRaisesRegex(ValueError, 'backwards'):
                controller.advance(q, 1., 1.01)

    def test_real_and_manual_adapters_share_targets_and_do_not_consume_duplicate_step(self):
        controllers = []
        clock = [10.]
        for adapter in ('real', 'manual'):
            bus = AngleBus()
            p = real.ManualArcReference(first_joint=8, last_joint=13, amplitude_deg=72., settle_s=0.)
            session = SimpleNamespace(sim=SimpleNamespace(p=p), elapsed=10.,
                                      record_event=lambda *a, **k: None, stop_push=lambda: None)
            c = real.RollingArcController(bus) if adapter == 'real' else debug.DebugArcController(bus, session)
            c.amplitude = 72.
            c.mode, c.phase, c.command_phase = 'hold', .7, .7
            c.targets = np.degrees(p.targets(.7)).tolist()
            c.enabled = p.enabled.tolist()
            bus.enabled, bus.angles = p.enabled.copy(), np.asarray(c.targets)
            c.motor_sample_clock = lambda: (clock[0], clock[0])
            c.select_gait('fixed')
            c.tick(.02)
            self.assertIsNone(c.section_block_reason)
            self.assertEqual(bus.sent_ids, list(range(8, 14)))
            c.select_gait('sliding')
            c.sliding.automatic = False
            c.command('step')
            c.tick(.02)
            state = copy.deepcopy((c.targets, c.enabled, c.phase, c.pending_phase))
            c.tick(.02)  # Deliberately same motor clock.
            self.assertEqual((c.targets, c.enabled, c.phase, c.pending_phase), state)
            self.assertFalse(c.sliding_fault)
            controllers.append(c)
        for i in range(1, 6):
            clock[0] = 10.+.02*i
            for c in controllers:
                c.refresh_compliance_feedback()
                c.tick(.02)
                self.assertFalse(c.sliding_fault, c.section_block_reason)
                self.assertEqual(c.bus.sent_ids, list(range(8, 14)))
            np.testing.assert_allclose(controllers[0].targets, controllers[1].targets, atol=1e-12)
        self.assertLess(controllers[0].pending_phase, state[-1])

    def test_high_resolution_clock_and_torque_off_packets(self):
        with patch.object(real.time, 'monotonic', return_value=100.), \
             patch.object(real.time, 'perf_counter', side_effect=[500., 500.01, 500.02, 500.020001]):
            c = real.RollingArcController(AngleBus())
            c.refresh_compliance_feedback()
            first, sample = c.motor_sample_clock()
            second, repeated_sample = c.motor_sample_clock()
            self.assertGreater(second, first)
            self.assertLess(sample, first)
            self.assertEqual(sample, repeated_sample)
        serial = RecordingSerial()
        bus = real.FeetechBus(serial)
        enabled = real.SectionReference(8, 13, 60.).enabled
        bus.prepare_enabled([0.]*20, enabled)
        bus.send_targets([10.]*20, enabled)
        self.assertEqual([serial.registers[sid][0x28] for sid in real.SERVO_IDS], enabled.astype(int).tolist())
        target_packet = [p for p in serial.packets if p[4] == 0x83 and p[5] == 0x2A][-1]
        self.assertEqual(list(target_packet[7:-1:7]), real.SERVO_IDS[7:13])
        bus.set_enabled([False]*20)
        self.assertFalse(any(serial.registers[sid][0x28] for sid in real.SERVO_IDS))


if __name__ == '__main__':
    unittest.main(verbosity=2)
