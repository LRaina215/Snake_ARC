"""Distinguish missing reverse motion from observer rejection on saved data."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import gzip
import json
import math
from collections import Counter
import numpy as np
import mujoco
from snake_control.simulation import ROOT, backbone
from snake_control.motor_shape_observer import PhaseShapeReference, bending_features
from validation.angle_observer_replay import CANONICAL_TO_MODEL


def load(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    raw, ev = load(args.path/'actual_trace.jsonl.gz'), load(args.path/'evaluation.jsonl.gz')
    states = {round(r['time_s'], 8): r for r in raw}
    reference = PhaseShapeReference.from_dict(json.loads((ROOT/'configs/motor_shape_reference.json').read_text(encoding='utf8')))
    model = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Rolling_Arc_straight.xml'))
    data, records = mujoco.MjData(model), []
    for r, assessment in zip(raw, ev):
        if not r['controller'].get('new_feedback_this_tick') or r['controller']['observation'] is None:
            continue
        o, f = r['controller']['observation'], r['motor_input']
        w = tuple(o['current_window'])
        base = reference.at(f['phase_rad'], w[0])
        features = bending_features(f['angles_deg'], [.0645]*21, w)
        origin = states[round(f['feedback_time_s'], 8)]
        data.qpos[:], data.qvel[:], data.time = origin['actual_qpos'], origin['actual_qvel'], origin['time_s']
        mujoco.mj_forward(model, data)
        rot = data.body('head').xmat.reshape(3, 3)@CANONICAL_TO_MODEL
        down = rot.T@np.array([0., 0., -1.])
        quality = o.get('end_quality', {})
        errors = {end: math.degrees(math.acos(float(np.clip(down@quality[end+'_direction'], -1., 1.))))
                  for end in ('head', 'tail') if end+'_direction' in quality}
        head_id = model.body('head').id
        tail_bank = model.jnt_bodyid[model.joint(f'joint{w[1]+1}').id]
        head_lever, tail_lever = data.xipos[head_id].copy(), data.xipos[tail_bank].copy()
        head_lever[0], head_lever[2] = 0., head_lever[2]-1.2
        tail_lever[0], tail_lever[2] = 0., tail_lever[2]-1.2
        forces = np.asarray(origin['external_force_world'])[:, :3]
        levers = data.xipos.copy()
        levers[:, 0] = 0.
        levers[:, 2] -= 1.2
        actual_moment = np.cross(levers, forces)[:, 0].sum()
        down_force, up_force = (20.+20./np.sqrt(2))/2, (20.-20./np.sqrt(2))/2
        vertical_couple_moment = np.cross(head_lever, [0., 0., -down_force])[0]+np.cross(tail_lever, [0., 0., up_force])[0]
        bone = backbone(model, data)[w[0]:w[1]+2]
        theta = np.unwrap(np.arctan2(bone[:, 2]-1.2, bone[:, 1]))
        signed_span = theta[-1]-theta[0]
        records.append(dict(time_s=f['feedback_time_s'], valid=o['valid'], state=o['state'],
            true_crest_offset=assessment['true_gravity_crest_joint']-sum(w)/2-base[0],
            bending_offset=features['bending_center_joint']-sum(w)/2-base[1],
            boundary_offset_deg=features['tail_boundary_turn_deg']-features['head_boundary_turn_deg']-base[2],
            head_error_deg=errors.get('head'), tail_error_deg=errors.get('tail'), quality=quality,
            actual_external_moment_x_nm=float(actual_moment), hypothetical_vertical_moment_x_nm=float(vertical_couple_moment),
            rigid_crown_rotation_actual_sign=float(-np.sign(actual_moment*signed_span)),
            rigid_crown_rotation_vertical_sign=float(-np.sign(vertical_couple_moment*signed_span)),
            support_center=float(np.array(origin['support_up_n'])@np.arange(1, 21)/sum(origin['support_up_n']))+.5))
    result = {}
    for label, begin, end in [('before', 29.6, 36.), ('pulse', 36., 39.), ('recovery', 39., 50.), ('late', 50., 83.)]:
        subset = [r for r in records if begin <= r['time_s'] < end]
        result[label] = dict(samples=len(subset), states=dict(Counter(r['state'] for r in subset)))
        for key in ('true_crest_offset', 'bending_offset', 'boundary_offset_deg', 'head_error_deg', 'tail_error_deg', 'support_center',
                    'actual_external_moment_x_nm', 'hypothetical_vertical_moment_x_nm',
                    'rigid_crown_rotation_actual_sign', 'rigid_crown_rotation_vertical_sign'):
            values = [r[key] for r in subset if r[key] is not None]
            result[label][key] = None if not values else dict(minimum=min(values), maximum=max(values), median=float(np.median(values)))
        result[label]['tail_crest_and_bending_samples'] = int(sum(r['true_crest_offset'] >= .5 and
            (r['bending_offset'] >= .2 or r['boundary_offset_deg'] >= 8) for r in subset))
    output = dict(stages=result, all_new_feedbacks=records)
    (args.path/'cause_audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
