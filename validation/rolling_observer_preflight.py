"""Read saved states: geometry localisation versus independent gravity crest.

No mj_step. All privileged quantities are evaluated AFTER the angle observer.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import math
import numpy as np
import mujoco
from snake_control.simulation import ROOT
from snake_control.motor_shape_observer import PhaseShapeReference, MotorObserverOptions, local_phase
from snake_control.rolling_shape_observer import RollingShapeObserver
from snake_control.section_geometry import forward
from validation.angle_observer_replay import CANONICAL_TO_MODEL

REPORT = ROOT/'validation/reports/rolling_handover'
SOURCES = [
    ROOT/'validation/reports/repeated_motor_sliding/natural_20261013_010202_444234',
    ROOT/'validation/reports/repeated_motor_sliding/return_repair_20261013_012403_696880',
    ROOT/'validation/reports/adaptive_arc_length/baseline_20261013_081646_022244',
    ROOT/'validation/reports/adaptive_arc_length/combined_20261013_083240_926004',
    ROOT/'validation/reports/rolling_handover/natural_20261013_112105_172339',
]


def replay(path):
    status = json.loads((path/'status.json').read_text(encoding='utf8'))
    with gzip.open(path/'actual_trace.jsonl.gz', 'rt', encoding='utf8') as f:
        rows = [json.loads(line) for line in f]
    states = {round(r['time_s'], 8): r for r in rows}
    references = {w: PhaseShapeReference.from_dict(json.loads((ROOT/f'configs/motor_shape_reference{suffix}.json').read_text()))
                  for w, suffix in ((6, ''), (8, '8'))}
    model = mujoco.MjModel.from_xml_path(str(ROOT/f"mujoco_model/Snake_Robot_o20_Rolling_Arc_{status['scene']}.xml"))
    data, observer, previous_width, previous_time = mujoco.MjData(model), None, None, None
    records, bins = [], [dict(samples=0, valid=0, recommendations=0, crest_direction_agrees=0) for _ in range(24)]
    for r in rows:
        f = r.get('motor_input')
        if f is None or f['feedback_time_s'] < 2. or f['feedback_time_s'] == previous_time:
            continue
        previous_time = f['feedback_time_s']
        width = sum(f['enabled'])
        if width not in references:
            continue
        if previous_width != width:
            observer = RollingShapeObserver([.0645]*21, references[width], width=width,
                options=MotorObserverOptions(maximum_feedback_gap_s=.5))
            previous_width = width
        o = observer.update(f['angles_deg'], f['phase_rad'], f['enabled'], f['feedback_time_s'])
        w = tuple(o['current_window'])
        origin = states[round(f['feedback_time_s'], 8)]
        data.qpos[:], data.qvel[:], data.time = origin['actual_qpos'], origin['actual_qvel'], origin['time_s']
        mujoco.mj_forward(model, data)
        if np.max(abs(data.qpos[7:]-np.radians(f['angles_deg']))) > 1e-8:
            raise ValueError('Matched saved motor state required')
        down = (data.body('head').xmat.reshape(3, 3)@CANONICAL_TO_MODEL).T@np.array([0., 0., -1.])
        q = np.radians(f['angles_deg'])
        _, _, section = observer.geometry_observer.geometry(q, w)
        try:
            crest = observer.geometry_observer.crest(forward(q, [.0645]*21), w, section, down)
            offset = crest['joint_coordinate']-sum(w)/2-references[width].at(f['phase_rad'], w[0])[0]
        except ValueError:
            offset = None
        candidate = o.get('suggested_window')
        support = np.asarray(origin['support_up_n'])
        coverage = None if candidate is None else float(support[candidate[0]-1:candidate[1]+1].sum()/max(support.sum(), 1e-12))
        agrees = bool(o['direction'] and offset is not None and o['direction']*offset >= .5)
        record = dict(time_s=f['feedback_time_s'], window=w, phase_rad=f['phase_rad'], observation=o,
                      true_gravity_crest_offset=offset, direction_agrees=agrees, candidate_support_coverage=coverage,
                      legacy_valid=(r['controller'].get('observation') or {}).get('valid'))
        records.append(record)
        b = bins[int(local_phase(f['phase_rad'], w[0])/(2*math.pi)*24) % 24]
        b['samples'] += 1; b['valid'] += int(o['valid']); b['recommendations'] += bool(o['direction']); b['crest_direction_agrees'] += agrees
    recommend = [r for r in records if r['observation']['direction']]
    result = dict(source=path.name, samples=len(records), valid=sum(r['observation']['valid'] for r in records),
                  legacy_valid=sum(bool(r['legacy_valid']) for r in records), recommendations=len(recommend),
                  direction_agrees=sum(r['direction_agrees'] for r in recommend),
                  opposite_crest_direction=int(sum(r['true_gravity_crest_offset'] is not None and
                      r['observation']['direction']*r['true_gravity_crest_offset'] < 0 for r in recommend)),
                  weak_crest_displacement=int(sum(r['true_gravity_crest_offset'] is not None and
                      0 <= r['observation']['direction']*r['true_gravity_crest_offset'] < .5 for r in recommend)),
                  minimum_candidate_support=None if not recommend else min(r['candidate_support_coverage'] for r in recommend),
                  phase_bins=bins)
    with gzip.open(REPORT/(path.name+'_observer.jsonl.gz'), 'wt', encoding='utf8') as f:
        for r in records:
            f.write(json.dumps(r)+'\n')
    return result


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    results = [replay(p) for p in SOURCES]
    (REPORT/'observer_preflight.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    for r in results:
        print(json.dumps({k: v for k, v in r.items() if k != 'phase_bins'}), flush=True)


if __name__ == '__main__':
    main()
