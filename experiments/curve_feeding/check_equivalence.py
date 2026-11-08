"""Numerical compatibility, real-log replay and a recorded forced-step stall."""
from copy import deepcopy
from dataclasses import replace
import gzip
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

from snake_control.continuous_feeding import ContinuousFeeding, ContinuousFeedingOptions
from .backend import CurveContinuousFeeding

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'validation/reports/curve_feeding_20261108'


def check():
    maximum = 0.
    count = 0
    for nonuniform in (False, True):
        lengths = np.full(19, .0645)
        if nonuniform:
            lengths[1::2] = .071
        for window in ((7, 12), (2, 7), (11, 16)):
            for amplitude in (45., 60., 70.):
                opt = replace(ContinuousFeedingOptions(), first_joint=window[0],
                              last_joint=window[1], amplitude_deg=amplitude)
                old, new = ContinuousFeeding(lengths, opt), CurveContinuousFeeding(lengths, opt)
                for phase in (0., .4, math.pi/2, -1.2, math.pi):
                    old.phase = new.phase = phase
                    for position in np.unique(np.r_[np.linspace(*old.position_bounds, 49), -2., 0., 2.]):
                        bias = -.12*math.sin(math.pi*position)
                        a, af, ac, ax = old.profile(position, bias)
                        b, bf, bc, bx = new.profile(position, bias)
                        error = float(np.max(abs(a-b)))
                        maximum = max(maximum, error)
                        assert error < 2e-12, (window, phase, position, error)
                        assert (af, ac, ax) == (bf, bc, bx)
                        np.testing.assert_allclose(old.helper.template(window, phase, amplitude, bias),
                            new.helper.template(window, phase, amplitude, bias), atol=2e-12, rtol=0)
                        count += 1
                for method in ('advance', 'request_step', 'start_forced_step', 'acquire_start',
                               'required_mask', 'required_with_buffer', 'release_ready'):
                    assert getattr(type(new), method) is getattr(type(old), method)
                # All future paths must survive disabling the legacy generator.
                with patch.object(new.helper.original, 'template', side_effect=AssertionError('old generator used')):
                    new.profile(.7, .1)
                    new.helper.template(window, .3, amplitude)
                    new.start_forced_step(np.degrees(new.targets), .05, 0.)

    # Keep original recorded angles; do not pass simulator/world information.
    replays = []
    for path in sorted((ROOT/'validation/reports/feeding_reference_recovery_20261021').glob('*.fresh.jsonl.gz')):
        cores = None
        previous = None
        rows = resets = 0
        errors = []
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)['row']
                f, m = row['feeding'], row['feeding_motor_input']
                angles = np.asarray(m['angles_deg'])
                t, now = m['feedback_time_s'], m['control_time_s']
                key = (t, now, row.get('feeding_action_id'))
                if cores is None or t <= previous[0] or now <= previous[1] or now-t > .5 or t-previous[0] > .5 or key[2] != previous[2]:
                    opt = ContinuousFeedingOptions(**f['options'])
                    cores = [cls(np.full(len(angles)+1, .0645), opt)
                             for cls in (ContinuousFeeding, CurveContinuousFeeding)]
                    resets += 1
                previous = key
                if now-t > .5:
                    continue
                outputs = []
                for core in cores:
                    core.phase = f.get('phase_rad', 0.)
                    outputs.append(core.advance(angles, t, now))
                a, b = outputs
                err = float(np.max(abs(a[0]-b[0])))
                assert err < 2e-11
                np.testing.assert_array_equal(a[1], b[1])
                assert cores[0].stage == cores[1].stage
                assert cores[0].fault == cores[1].fault
                assert cores[0].observation == cores[1].observation
                errors.append(err)
                rows += 1
        replays.append(dict(source=str(path.relative_to(ROOT)), frames=rows,
            replay_blocks=resets, max_target_error_rad=max(errors, default=0.),
            scope='same recorded feedback replay; resets at actions/gaps; not a hardware experiment'))

    fixture = json.loads((ROOT/'validation/reports/feeding_forced_step_20261021/fixture.json').read_text(encoding='utf-8'))
    f = fixture['feeding']
    old = ContinuousFeeding(np.full(len(fixture['angles_deg'])+1, .0645), ContinuousFeedingOptions(**f['options']))
    old.position, old.bias = f['position_joints'], f['bearing_bias']
    old.targets, old.enabled = np.radians(fixture['target_deg']), np.array(fixture['enabled'], bool)
    old.window, old.stage = tuple(f['window']), 'acquire'
    new = CurveContinuousFeeding(old.lengths, old.options)
    # This is the same stored state; retain the experimental trajectory adapter.
    for name, value in old.__dict__.items():
        if name not in ('helper',):
            setattr(new, name, deepcopy(value))
    for core in (old, new):
        core.start_forced_step(fixture['angles_deg'], .05, 0.)
    frames = 0
    for i in range(100):
        outputs = [core.advance(fixture['angles_deg'], i*.02, i*.02) for core in (old, new)]
        np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=2e-12, rtol=0)
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
        frames += 1
        if old.forced_step is None:
            assert new.forced_step is None
            break
    result = dict(passed=True, parameter_profiles=count, max_profile_error_rad=maximum,
        real_log_replay=replays, forced_stall_frames=frames,
        inherited_state_machine=True, legacy_generator_disabled_check=True,
        hardware_run=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'equivalence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    check()
