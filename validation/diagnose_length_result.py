"""Targeted saved-state diagnosis, including sharp-corner axis ambiguity."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import mujoco
import cv2
from snake_control.simulation import ROOT, backbone
from validation.audit_arc_length import load
from snake_control.angle_shape_observer import AngleShapeObserver
from snake_control.section_geometry import forward
from validation.angle_observer_replay import CANONICAL_TO_MODEL

REPORT = ROOT/'validation/reports/adaptive_arc_length'


def main():
    folder = next(REPORT.glob('combined_*'))
    rows = load(folder/'actual_trace.jsonl.gz')
    ev = load(folder/'evaluation.jsonl.gz')
    model = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model/Snake_Robot_o20_Rolling_Arc_elbow90_taper110_210.xml'))
    data = mujoco.MjData(model)
    pipe_body = model.body('pipe_network').id
    head = model.body('head').id
    route = model.numeric('pipe_centerline').data.reshape(-1, 4)[:, :3]
    tangent = np.diff(route, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    details = []
    for r, e in zip(rows, ev):
        if not e['pipe_shape'].get('nearest_wrap_valid', e['pipe_shape']['wrap_valid']) or e['pipe_shape']['wrap_deg'] >= 150 or r['time_s'] < 2.:
            continue
        data.qpos[:], data.qvel[:], data.time = r['actual_qpos'], r['actual_qvel'], r['time_s']
        mujoco.mj_forward(model, data)
        first, last = r['controller']['window']
        points = backbone(model, data)[first:last+2]
        delta = np.diff(route, axis=0)
        frac = np.clip(np.sum((points[:, None]-route[:-1])*delta, axis=2)/np.sum(delta*delta, axis=1), 0, 1)
        near = route[:-1]+frac[:, :, None]*delta
        index = np.argmin(np.linalg.norm(points[:, None]-near, axis=2), axis=1)
        nearest_axes = sorted(set(tuple(np.round(tangent[k], 6)) for k in index))
        axes, all_axes = [], []
        for c in data.contact:
            pair = [int(c.geom1), int(c.geom2)]
            pipes = [g for g in pair if model.geom_bodyid[g] == pipe_body]
            robots = [g for g in pair if model.geom_bodyid[g] >= head]
            if not pipes or not robots:
                continue
            segment = int(model.geom(pipes[0]).name.rsplit('_', 1)[1])
            axis = tuple(np.round(tangent[segment], 6))
            all_axes.append(axis)
            index = model.geom_bodyid[robots[0]]-head-1
            if first-2 <= index <= last:
                axes.append(axis)
        details.append(dict(time_s=r['time_s'], window=[first, last], nearest_axis_span_deg=e['pipe_shape']['wrap_deg'],
            nearest_axis_spread_deg=e['pipe_shape']['axis_spread_deg'], active_contact_axes=[list(a) for a in sorted(set(axes))],
            nearest_axes=[list(a) for a in nearest_axes],
            contact_axis_not_in_nearest=set(axes).isdisjoint(nearest_axes) if axes else False,
            all_robot_contact_axes=[list(a) for a in sorted(set(all_axes))]))
    ambiguous = sum(len(r['active_contact_axes']) > 1 for r in details)
    mismatch = sum(r['contact_axis_not_in_nearest'] for r in details)
    diagnosis = dict(source=folder.name, low_nearest_axis_span_frames=len(details),
        active_contacts_on_both_axes_frames=ambiguous, contact_nearest_axis_mismatch_frames=mismatch, frames=details,
        inference='Nearest-centerline wrap can be ambiguous even before its 90deg axis-spread flag; '
                  'these alerts alone cannot prove inadequate physical coverage. Full sharp-corner pipe verification remains incomplete.')
    (folder/'corner_axis_diagnosis.json').write_text(json.dumps(diagnosis, indent=2), encoding='utf8')
    status = json.loads((folder/'status.json').read_text())
    states = {round(r['time_s'], 8): r for r in rows}
    helper, handovers = AngleShapeObserver([.0645]*21), []
    for event in status['events']:
        if event['kind'] != 'complete':
            continue
        r = next(r for r in rows if r['time_s'] > event['time_s']+1e-8)
        o, f = r['controller']['observation'], r['motor_input']
        source = states[round(f['feedback_time_s'], 8)]
        data.qpos[:], data.qvel[:], data.time = source['actual_qpos'], source['actual_qvel'], source['time_s']
        mujoco.mj_forward(model, data)
        down = (data.body('head').xmat.reshape(3, 3)@CANONICAL_TO_MODEL).T@np.array([0., 0., -1.])
        q = np.radians(f['angles_deg'])
        old = tuple(o['current_window'])
        _, _, section = helper.geometry(q, old)
        crown = helper.crest(forward(q, [.0645]*21), old, section, down)['joint_coordinate']
        new = event['to']
        torque = source['actuator_force_nm']
        exiting = [j for j in range(event['old'][0], event['old'][1]+1) if not new[0] <= j <= new[1]]
        handovers.append(dict(time_s=event['time_s'], old=event['old'], new=new,
            true_gravity_crown_offset=float(crown-sum(old)/2-o['phase_reference'][0]),
            down_error_deg=float(np.degrees(np.arccos(np.clip(down@o['down_head'], -1, 1)))),
            new_support_coverage=float(sum(source['support_up_n'][new[0]-1:new[1]+1])/sum(source['support_up_n'])),
            exiting_torque_before_nm=[torque[j-1] for j in exiting]))
    (folder/'handover_truth.json').write_text(json.dumps(handovers, indent=2), encoding='utf8')
    # Inspect the movie around the lowest computed span and first ambiguous
    # frame, preserving the movie's own actual timestamp in its HUD.
    movie = ROOT/'validation/results/adaptive_arc_length'/(folder.name+'.mp4')
    cap = cv2.VideoCapture(str(movie))
    wanted = [('low_span', min(details, key=lambda r: r['nearest_axis_span_deg'])['time_s'])]
    if ambiguous:
        wanted.append(('both_contact_axes', next(r['time_s'] for r in details if len(r['active_contact_axes']) > 1)))
    for name, time in wanted:
        cap.set(cv2.CAP_PROP_POS_MSEC, time*1000)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(folder/(name+'_video_frame.png')), frame)
    cap.release()
    print(json.dumps({k: v for k, v in diagnosis.items() if k != 'frames'}))
    print(json.dumps(handovers))


if __name__ == '__main__':
    main()
