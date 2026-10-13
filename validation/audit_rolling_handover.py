"""Full saved-motion audit and independent position/transfer review, no dynamics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import gzip
import json
import math
import numpy as np
import mujoco
from snake_control.simulation import ROOT
from snake_control.motor_shape_observer import PhaseShapeReference, local_phase
from snake_control.angle_shape_observer import AngleShapeObserver
from snake_control.section_geometry import forward
from validation.angle_observer_replay import CANONICAL_TO_MODEL
import validation.audit_arc_length as archive_audit

REPORT = ROOT/'validation/reports/rolling_handover'


def load(path):
    with gzip.open(path, 'rt', encoding='utf8') as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('path', type=Path)
    path = parser.parse_args().path
    archive_audit.REPORT = REPORT
    archive_audit.audit(path)
    review = json.loads((path/'review.json').read_text())
    status = json.loads((path/'status.json').read_text())
    rows = load(path/'actual_trace.jsonl.gz')
    states = {round(r['time_s'], 8): r for r in rows}
    model = mujoco.MjModel.from_xml_path(str(ROOT/f"mujoco_model/Snake_Robot_o20_Rolling_Arc_{status['scene']}.xml"))
    data, helper = mujoco.MjData(model), AngleShapeObserver([.0645]*21)
    refs = {w: PhaseShapeReference.from_dict(json.loads((ROOT/f'configs/motor_shape_reference{suffix}.json').read_text()))
            for w, suffix in ((6, ''), (8, '8'))}
    bins = [dict(samples=0, position_valid=0, recommendations=0, direction_agrees=0,
                 opposite_direction=0, migration_error_abs=[]) for _ in range(24)]
    observations, handovers = [], []

    def independent(f, o):
        w = tuple(o['current_window']); width = w[1]-w[0]+1
        origin = states[round(f['feedback_time_s'], 8)]
        data.qpos[:], data.qvel[:], data.time = origin['actual_qpos'], origin['actual_qvel'], origin['time_s']
        mujoco.mj_forward(model, data)
        q = np.radians(f['angles_deg'])
        if np.max(abs(q-data.qpos[7:])) > 1e-8:
            raise ValueError('Feedback must match independently evaluated state')
        down = (data.body('head').xmat.reshape(3, 3)@CANONICAL_TO_MODEL).T@np.array([0., 0., -1.])
        _, _, section = helper.geometry(q, w)
        offset = None
        if width in refs:
            try:
                crest = helper.crest(forward(q, [.0645]*21), w, section, down)
                offset = crest['joint_coordinate']-sum(w)/2-refs[width].at(f['phase_rad'], w[0])[0]
            except ValueError:
                pass
        candidate = o.get('suggested_window')
        support = np.asarray(origin['support_up_n'])
        floor = model.geom('floor').id
        robot = set(np.flatnonzero(model.geom_bodyid >= model.body('head').id))
        floor_contact = any(floor in (int(c.geom1), int(c.geom2)) and
                            bool({int(c.geom1), int(c.geom2)} & robot) for c in data.contact)
        coverage = None if not candidate else float(support[candidate[0]-1:candidate[1]+1].sum()/max(support.sum(), 1e-12))
        return dict(time_s=f['feedback_time_s'], phase_rad=f['phase_rad'], window=w,
            observation=o, true_gravity_crest_offset=offset, candidate_support_coverage=coverage,
            support_n=float(support.sum()), floor_contact=bool(floor_contact),
            supported_on_pipe=bool(support.sum() >= 2. and not floor_contact),
            crest_proxy_available='down_head' in o,
            direction_agrees=bool(o['direction'] and offset is not None and o['direction']*offset >= .5),
            opposite_direction=bool(o['direction'] and offset is not None and o['direction']*offset < 0),
            migration_error_abs=None if offset is None or 'filtered_error_joints' not in o else abs(o['filtered_error_joints']-offset))

    last_feedback = None
    exchange_observations = {round(e['time_s'], 8): e['observation'] for e in status['events']
                             if e['kind'] in ('complete', 'resize_step', 'resize_complete')}
    for row in rows:
        f, o = row.get('motor_input'), row['controller']['observation']
        if f and o is None:
            o = exchange_observations.get(round(f['feedback_time_s'], 8))
        if not f or not o or f['feedback_time_s'] < 2. or f['feedback_time_s'] == last_feedback:
            continue
        # After exchange o may be reset; the event retains its preceding intent.
        if len(o['current_window']) != 2 or o['current_window'][1]-o['current_window'][0]+1 not in (6, 8): continue
        last_feedback = f['feedback_time_s']
        item = independent(f, o); observations.append(item)
        b = bins[int(local_phase(f['phase_rad'], o['current_window'][0])/(2*math.pi)*24) % 24]
        b['samples'] += 1; b['position_valid'] += bool(o.get('position_valid', False)); b['recommendations'] += bool(o['direction'])
        b['direction_agrees'] += item['direction_agrees']; b['opposite_direction'] += item['opposite_direction']
        if item['migration_error_abs'] is not None: b['migration_error_abs'].append(item['migration_error_abs'])
    for event in status['events']:
        if event['kind'] not in ('complete', 'resize_step', 'resize_complete'): continue
        row = min(rows, key=lambda r: abs(r['time_s']-(event['time_s']+.02)))
        item = independent(row['motor_input'], event['observation'])
        origin = states[round(row['motor_input']['feedback_time_s'], 8)]
        item.update(kind=event['kind'], old=event['old'], to=event['to'], gate=event['gate'],
            entering_passive_torque_nm=[origin['actuator_force_nm'][j-1] for j in event['gate']['entering_joint']],
            exiting_active_torque_nm=[origin['actuator_force_nm'][j-1] for j in event['gate']['exiting_joint']])
        handovers.append(item)
    for b in bins:
        errors = b.pop('migration_error_abs')
        b['migration_error_p90_joints'] = None if not errors else float(np.percentile(errors, 90))
        b['migration_error_max_joints'] = None if not errors else max(errors)
    phase_errors = []
    for r in rows:
        c = r['controller']
        t = c.get('control_time_s')
        if t is None: continue
        expected = status['parameters']['initial_phase_rad']+.6*max(0., t-2.)
        phase_errors.append(abs(c['phase_rad']-expected))
    slides = [e for e in handovers if e['kind'] == 'complete']
    visited = [tuple((status['parameters']['first_joint'], status['parameters']['last_joint']))]+[tuple(e['to']) for e in slides]
    directions = [e['to'][0]-e['old'][0] for e in slides]
    returned = len(visited) >= 3 and any(w in visited[:i] for i, w in enumerate(visited[2:], 2)) and -1 in directions and 1 in directions
    pipe_slides = [e for e in slides if e['supported_on_pipe'] and e['direction_agrees']]
    pipe_visited = [visited[0]]+[tuple(e['to']) for e in pipe_slides]
    pipe_directions = [e['to'][0]-e['old'][0] for e in pipe_slides]
    pipe_return = (len(pipe_visited) >= 3 and any(w in pipe_visited[:i] for i, w in enumerate(pipe_visited[2:], 2)) and
                   -1 in pipe_directions and 1 in pipe_directions)
    supported = [o for o in observations if o['supported_on_pipe']]
    sources = {}
    for o in supported:
        label = o['observation'].get('end_quality', {}).get('source', 'no_position')
        record = sources.setdefault(label, dict(samples=0, recommendations=0, maximum_migration_error_joints=0.))
        record['samples'] += 1; record['recommendations'] += bool(o['observation']['direction'])
        if o['migration_error_abs'] is not None:
            record['maximum_migration_error_joints'] = max(record['maximum_migration_error_joints'], o['migration_error_abs'])
    location = dict(new_observations=len(observations), phase_bins=bins,
        all_phase_bins_observed=all(b['samples'] > 0 for b in bins),
        all_phase_bins_have_valid_position=all(b['position_valid'] > 0 for b in bins),
        position_valid_samples=sum(o['observation'].get('position_valid', False) for o in observations),
        supported_samples=len(supported), unsupported_samples=len(observations)-len(supported), supported_sources=sources,
        supported_recommendations=sum(bool(o['observation']['direction']) for o in supported),
        supported_direction_agrees=sum(o['direction_agrees'] for o in supported),
        recommendations=sum(bool(o['observation']['direction']) for o in observations),
        direction_agrees=sum(o['direction_agrees'] for o in observations),
        opposite_direction=sum(o['opposite_direction'] for o in observations),
        true_tail_displacement_samples=int(sum(o['true_gravity_crest_offset'] is not None and o['true_gravity_crest_offset'] >= .5 for o in observations)),
        true_head_displacement_samples=int(sum(o['true_gravity_crest_offset'] is not None and o['true_gravity_crest_offset'] <= -.5 for o in observations)))
    s = review['summary']
    changes = [e for e in status['events'] if e['kind'] in ('complete', 'resize_complete')]
    if changes:
        # The saved state at event.time precedes that control tick. Use the
        # event's issued phase, rather than counting an extra 20ms before it.
        s['post_last_change_cycles'] = (rows[-1]['controller']['phase_rad']-changes[-1]['phase_rad'])/(2*math.pi)
        s['post_last_change_phase_basis'] = 'recorded_event_phase'
    normal_phase = bool(max(phase_errors, default=0.) < 1e-7 and not any(e['kind'] == 'phase_hold' for e in status['events']))
    correct_transfers = bool(slides and all(e['direction_agrees'] for e in slides))
    mechanism = bool(len(slides) >= 2 and normal_phase and correct_transfers and review['shape_pass'] and review['mechanics_pass'])
    short_states = [r for r in rows if min(r['controller']['window'][0]-1, 20-r['controller']['window'][1]) < 3 and r['time_s'] >= 2.]
    new = dict(moving_exchange_pass=mechanism, phase_continuity_pass=normal_phase,
        maximum_phase_error_rad=max(phase_errors, default=0.), transfers=len(slides), visited_windows=visited,
        window_index_return_observed=returned, bidirectional_return_on_pipe_observed=bool(pipe_return), short_passive_states=len(short_states),
        minimum_passive_joint_count=min(min(r['controller']['window'][0]-1, 20-r['controller']['window'][1]) for r in rows),
        full_phase_sampling_coverage=location['all_phase_bins_observed'],
        full_phase_reliable_position_pass=bool(location['all_phase_bins_observed'] and supported and
            all(o['crest_proxy_available'] and o['migration_error_abs'] is not None and o['migration_error_abs'] <= .5 for o in supported)),
        location=location, handovers=handovers)
    post_ok = s['post_last_change_cycles'] >= 1.
    if status['case'] == 'natural':
        passed = mechanism and review['strict_pipe_pass'] and post_ok
    elif status['case'] == 'short_return':
        passed = mechanism and pipe_return and bool(short_states) and review['strict_pipe_pass'] and post_ok
    else:
        passed = mechanism and review['resize_mechanism_pass'] and post_ok and (s['continuous_exit_cycles'] or 0) >= 1.
    review.update(new); review['passed'] = bool(passed)
    review['complete_single_pipe_axis_validation'] = review['strict_pipe_pass']
    (path/'review.json').write_text(json.dumps(review, indent=2), encoding='utf8')
    (REPORT/(status['case']+'_review.json')).write_text(json.dumps(review, indent=2), encoding='utf8')
    with gzip.open(path/'position_review.jsonl.gz', 'wt', encoding='utf8') as f:
        for o in observations: f.write(json.dumps(o)+'\n')
    print(json.dumps({k: v for k, v in new.items() if k not in ('location', 'handovers')})+' passed='+str(passed))
    print(json.dumps(location))


if __name__ == '__main__':
    main()
