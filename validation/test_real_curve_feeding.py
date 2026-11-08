"""Standalone real-port checks. Memory serial only; NEVER opens a COM port."""
from contextlib import redirect_stdout
from dataclasses import asdict
import gzip
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codes import Rolling_Arc_Real as real
from experiments.curve_feeding.backend import CurveContinuousFeeding
from snake_control.continuous_feeding import ContinuousFeedingOptions
from tools.build_real_curve_feeding import embed
from validation.test_active_feeding_port import ready, real as memory_real

REPORT = ROOT/'validation/reports/real_curve_feeding_20261108'
EVIDENCE = {}


class RealCurveFeedingTests(unittest.TestCase):
    def test_real_defaults_and_state_machine_are_preserved(self):
        real.validate_setup()
        self.assertFalse(real.DRY_RUN)
        self.assertEqual(real.NUM_JOINTS, 18)
        self.assertEqual(len(real.SERVO_IDS), 18)
        self.assertEqual(len(real.LINK_LENGTHS_M), 19)
        self.assertEqual(real.ContinuousFeeding.trajectory_backend, 'stitched_curve_integral')
        for name in ('advance', 'request_step', 'start_forced_step', 'acquire_start',
                     'required_mask', 'required_with_buffer', 'release_ready'):
            self.assertIs(getattr(real.ContinuousFeeding, name), getattr(real.LegacyContinuousFeeding, name))
        for key, expected in [('J', 'step'), ('H', 'back'), ('G', 'toggle'),
                              ('L', 'feed_tick'), ('Y', 'feed_return'), ('X', 'release')]:
            self.assertEqual(real.decode_hotkey(ord(key), True), expected)

    def test_embedded_curve_and_old_reference_match(self):
        maximum = 0.
        count = 0
        for n in (18, 20):
            lengths = np.full(n+1, .0645)
            for window in ((2, 7), (7, 12), (11, 16)):
                for amplitude in (60., 70.):
                    values = dict(first_joint=window[0], last_joint=window[1], amplitude_deg=amplitude)
                    old = real.LegacyContinuousFeeding(lengths, real.ContinuousFeedingOptions(**values))
                    new = real.ContinuousFeeding(lengths, real.ContinuousFeedingOptions(**values))
                    source = CurveContinuousFeeding(lengths, ContinuousFeedingOptions(**values))
                    for phase in (0., .4, -1.2, math.pi/2):
                        old.phase = new.phase = source.phase = phase
                        for position in np.linspace(*old.position_bounds, 25):
                            bias = -.12*math.sin(math.pi*position)
                            a, b, c = [core.profile(position, bias) for core in (old, new, source)]
                            maximum = max(maximum, float(np.max(abs(a[0]-b[0]))), float(np.max(abs(b[0]-c[0]))))
                            self.assertLess(float(np.max(abs(a[0]-b[0]))), 2e-12)
                            np.testing.assert_allclose(b[0], c[0], atol=2e-12, rtol=0)
                            self.assertEqual(a[1:], b[1:])
                            count += 1
                    with patch.object(new.helper.original, 'template', side_effect=AssertionError('legacy path')):
                        new.profile(.4, .1)
                        new.helper.template(window, .2, amplitude)
                    json.dumps(new.snapshot())
        EVIDENCE.update(parameter_profiles=count, max_target_difference_rad=maximum)

    def test_complete_saved_stroke_target_replay(self):
        path = ROOT/'validation/reports/curve_feeding_full_stroke_20261108/normal/full_stroke_curve_control.jsonl.gz'
        core = source = None
        signature = None
        count = 0
        maximum = 0.
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line); f = row['feeding']
                values = f['options']
                current = tuple(values.items())
                if current != signature:
                    signature = current
                    core = real.ContinuousFeeding(real.LINK_LENGTHS_M, real.ContinuousFeedingOptions(**values))
                    source = CurveContinuousFeeding(real.LINK_LENGTHS_M, ContinuousFeedingOptions(**values))
                core.phase = source.phase = f['phase_rad']
                a = core.profile(f['position_joints'], f['bearing_bias'])
                b = source.profile(f['position_joints'], f['bearing_bias'])
                maximum = max(maximum, float(np.max(abs(a[0]-b[0]))))
                np.testing.assert_allclose(a[0], b[0], atol=2e-12, rtol=0)
                self.assertEqual(a[1:], b[1:])
                count += 1
        EVIDENCE.update(saved_stroke_control_frames=count, saved_stroke_max_difference_rad=maximum)

    def test_memory_serial_j_h_g_l_y_x(self):
        with patch.object(memory_real, 'open_robot_serial', side_effect=AssertionError('No hardware IO')), redirect_stdout(io.StringIO()):
            c = ready()
            def press(key):
                c.command(memory_real.decode_hotkey(ord(key), True))
            press('J')
            for _ in range(800):
                c.cycle()
                if c.feeding.baseline is not None:
                    break
            self.assertIsNotNone(c.feeding.baseline)
            self.assertIsNone(c.feeding.fault)
            self.assertEqual(c.feeding.trajectory_backend, 'stitched_curve_integral')
            press('G')
            p, targets, mask = c.feeding.position, c.targets.copy(), c.enabled.copy()
            for _ in range(30):
                c.cycle()
            self.assertEqual(c.feeding.position, p)
            self.assertEqual(c.targets, targets)
            self.assertEqual(c.enabled, mask)
            press('G')
            for _ in range(15): c.cycle()
            p = c.feeding.position
            press('L')
            self.assertAlmostEqual(c.feeding.position, p+memory_real.FEED_FORCE_STEP_JOINTS)
            for _ in range(150):
                c.cycle()
                if c.feeding_paused: break
            self.assertTrue(c.feeding_paused)
            core, p = c.feeding, c.feeding.position
            press('H')
            self.assertIs(c.feeding, core)
            self.assertEqual(core.position, p)
            self.assertLess(core.options.requested_transfer_m, 0.)
            for _ in range(800):
                c.cycle()
                if core.baseline is not None: break
            self.assertIsNotNone(core.baseline)
            press('Y')
            self.assertEqual(core.stage, 'feed_return')
            self.assertEqual(core.goal_m, core.baseline)
            json.dumps(c.snapshot())
            self.assertEqual(core.snapshot()['trajectory_backend'], 'stitched_curve_integral')
            press('X')
            self.assertFalse(any(c.enabled))
            self.assertTrue(all(c.port.registers[sid][0x28] == 0 for sid in memory_real.SERVO_IDS))
        EVIDENCE['memory_serial_hotkeys'] = ['J', 'H', 'G', 'L', 'Y', 'X']

    def test_standalone_copy_without_repository_or_mujoco(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'Rolling_Arc_Real.py'
            path.write_text((ROOT/'codes/Rolling_Arc_Real.py').read_text(encoding='utf-8'), encoding='utf-8')
            code = (
                "import builtins,runpy; original=builtins.__import__\n"
                "def isolated(name,*args,**kwargs):\n"
                " if name.split('.')[0] in ('mujoco','snake_control','experiments','serial'): raise AssertionError(name)\n"
                " return original(name,*args,**kwargs)\n"
                "builtins.__import__=isolated\n"
                "m=runpy.run_path('Rolling_Arc_Real.py',run_name='offline_import'); m['validate_setup'](); "
                "c=m['ContinuousFeeding'](m['LINK_LENGTHS_M']); "
                "assert c.trajectory_backend=='stitched_curve_integral'; "
                "assert m['DRY_RUN'] is False; print('standalone_real_ready')"
            )
            result = subprocess.run([sys.executable, '-I', '-c', code], cwd=directory,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('standalone_real_ready', result.stdout)
        EVIDENCE['standalone_no_repository_mujoco_or_serial_import'] = True

    def test_embed_is_idempotent_and_preserves_manual_settings(self):
        text = (ROOT/'codes/Rolling_Arc_Real.py').read_text(encoding='utf-8')
        self.assertEqual(embed(text), text)
        modified = text.replace('PORT = None', "PORT = 'COM_TEST_NOT_OPEN'", 1)
        modified = modified.replace('AMPLITUDE_DEG = 60.0', 'AMPLITUDE_DEG = 68.0', 1)
        self.assertEqual(embed(modified), modified)
        self.assertEqual(text.count('# BEGIN CURVE_FEEDING_CORE'), 1)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RealCurveFeedingTests))
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/'checks.json').write_text(json.dumps(dict(passed=result.wasSuccessful(),
        checks=result.testsRun, hardware_run=False, new_physics_runs=0,
        failures=[str(t) for t,_ in result.failures+result.errors], **EVIDENCE),
        ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
