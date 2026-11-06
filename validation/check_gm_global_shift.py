"""Check the fixed-curve shift against an archived state. No dynamics steps."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.gravity_global_shift import GravityGlobalShift
from snake_control.gravity_moment import GravityMomentGait, GravityMomentOptions


def load_checkpoint(record, checkpoint_s):
    saved = json.loads(record.read_text(encoding='utf-8'))
    summary = saved['summary']
    with np.load(record.with_name(record.stem+'_trace.npz')) as trace:
        index = int(np.argmin(abs(trace['time']-checkpoint_s)))
        frame = saved['samples'][index]
        n = summary['joints']
        qpos = trace['qpos'][index]
        qvel = trace['qvel'][index]
        issued = trace['targets'][index]
    if frame['stage'] != 'extend_hold' or not frame['branch_links']:
        raise ValueError('Select a saved tail-extension frame with branch contact')

    options = GravityMomentOptions(**summary['options'])
    gait = GravityMomentGait(n, options)
    r = options.r_h2
    phase = frame['roll_rad']
    straight = (gait.n_ext(r)+1)*gait.l
    shift = GravityGlobalShift(gait, r, phase, straight)
    return summary, frame, index, gait, shift, qpos, qvel, issued


def check(record, checkpoint_s):
    summary, frame, index, gait, shift, qpos, _, issued = load_checkpoint(record, checkpoint_s)
    options = gait.o
    r = options.r_h2
    phase = frame['roll_rad']
    straight = (gait.n_ext(r)+1)*gait.l
    measured = qpos[-summary['joints']:]
    start = shift.targets(0.)
    end = shift.targets(shift.shift_limit_m)
    reference_start, _ = gait.targets(phase, r, ls=straight, th2=gait.lead_angle(r))
    reference_end, _ = gait.targets(shift.phase_at_curve_start, r)
    np.testing.assert_allclose(start, reference_start, atol=1e-12, rtol=0)
    np.testing.assert_allclose(end, reference_end, atol=1e-12, rtol=0)
    # Two link lengths retain hinge parity. The corresponding joints must
    # see the exact same curvature on the overlapping part of the curve.
    delta = 2*gait.l
    np.testing.assert_allclose(shift.targets(delta)[2:], start[:-2], atol=1e-12, rtol=0)
    positions = np.linspace(0., shift.shift_limit_m, 121)
    targets = np.stack([shift.targets(s) for s in positions])
    error = np.degrees(start-measured)
    return dict(record=record.stem, frame=index, time_s=frame['time_s'],
                branch_links=frame['branch_links'], branch_normal_n=frame['branch_force_n'],
                start_measured_rms_error_deg=float(np.sqrt(np.mean(error**2))),
                start_measured_max_error_deg=float(max(abs(error))),
                start_issued_max_difference_deg=float(max(abs(np.degrees(start-issued)))),
                virtual_shift_limit_m=shift.shift_limit_m,
                reference_head_axial_advance_m=shift.shift_limit_m*options.pitch/gait.turn_length(r),
                max_target_deg=float(np.max(abs(np.degrees(targets)))),
                joints_changed_after_two_link_shift=int(np.sum(abs(shift.targets(delta)-start) > np.deg2rad(.1))),
                overlapping_targets_match=True, dynamics_steps=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('--time', type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.record, args.time), indent=2))
