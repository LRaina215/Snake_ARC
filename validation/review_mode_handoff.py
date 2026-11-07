"""Review all recorded states from the real-entry mode handoff experiments."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import backbone
from snake_control.tee_simulation import TeeSimulation, PIPE_HEIGHT


def review(folder):
    record = json.loads((folder/'locked_gait.json').read_text(encoding='utf-8'))
    summary, rows = record['summary'], record['samples']
    trace = np.load(folder/'locked_gait_trace.npz')
    n = trace['targets'].shape[1]
    sim = TeeSimulation(summary['scene'], ArcParameters(joints=n, first_joint=1, last_joint=n))
    held, rolling = [], []
    for index, row in enumerate(rows):
        if row['stage'] not in ('complete', 'roll'):
            continue
        sim.data.qpos[:] = trace['qpos'][index]
        mujoco.mj_forward(sim.model, sim.data)
        points = backbone(sim.model, sim.data)
        yz = points[:, 1:]-[0., PIPE_HEIGHT]
        angle = np.unwrap(np.arctan2(yz[:, 1], yz[:, 0]))
        actual = dict(time_s=row['time_s'], turns=float(np.ptp(angle)/(2.*np.pi)),
                      max_radius_m=float(np.max(np.linalg.norm(yz, axis=1))))
        (held if row['stage'] == 'complete' else rolling).append(actual)
    final = rolling or held
    shape_passed = bool(final and min(r['turns'] for r in final) >= 1.5 and
                        max(r['max_radius_m'] for r in final) <= .145)
    mask, force = trace['masks'], trace['actuator_force']
    passive = float(np.max(abs(force[~mask]), initial=0.))
    shared = mask[1:] & mask[:-1]
    target_steps = np.degrees(abs(np.diff(trace['targets'], axis=0)))
    drop_indices = [i for i, row in enumerate(rows) if row['stage'] in ('drop', 'drop_hold')]
    drop_review = None
    if drop_indices:
        lateral = []
        tail = slice(sim.model.body('head').id+9, sim.model.body('head').id+n+1)
        for index in (drop_indices[0], drop_indices[-1]):
            sim.data.qpos[:] = trace['qpos'][index]
            mujoco.mj_forward(sim.model, sim.data)
            com = np.average(sim.data.xipos[tail], axis=0, weights=sim.model.body_mass[tail])
            lateral.append(float(com[1]))
        front = np.unwrap(np.radians([rows[i]['front_angle_deg'] for i in drop_indices]))
        drop_review = dict(tail_com_y_start_m=lateral[0], tail_com_y_end_m=lateral[1],
                           front_rotation_deg=float(np.degrees(front[-1]-front[0])),
                           head_advance_m=rows[drop_indices[-1]]['head_x_m']-rows[drop_indices[0]]['head_x_m'])
    report = dict(case=summary['case'], controller_passed=summary['passed'],
                  shape_passed=shape_passed, passed=bool(summary['passed'] and shape_passed and passive < 1e-10),
                  actual_review_frames=len(rows), final_shape_frames=len(final),
                  final_min_turns=None if not final else min(r['turns'] for r in final),
                  final_max_radius_m=None if not final else max(r['max_radius_m'] for r in final),
                  passive_torque_max_nm=passive,
                  maximum_actual_joint_deg=max(r['actual_max_joint_deg'] for r in rows),
                  minimum_support_n=min(r['support_n'] for r in rows),
                  low_support_frames=sum(r['support_n'] <= .1 for r in rows),
                  floor_frames=sum(r['ground_force_n'] > 1. for r in rows),
                  saturated_frames=sum(bool(r['saturated_joints']) for r in rows),
                  shared_target_step_max_deg=float(np.max(target_steps[shared], initial=0.)),
                  front_confirmed_s=summary['front_confirmed_s'], first_tail_on_s=summary['first_tail_on_s'],
                  operator_events=summary['operator_events'], failure=summary['failure'])
    report['gravity_release'] = drop_review
    (folder/'review.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    for record in sorted(args.root.glob('*/locked_gait.json')):
        result = review(record.parent)
        print(record.parent.name, json.dumps(result))
