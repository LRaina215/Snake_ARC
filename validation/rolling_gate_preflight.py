"""Check gate opportunities on the known missed-migration archive, no mj_step."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
from collections import Counter
import numpy as np
from snake_control.simulation import ROOT
from snake_control.adaptive_arc_length import HalfArcParameters
from snake_control.rolling_handover import RollingHandover
from snake_control.rolling_shape_observer import RollingShapeObserver
from snake_control.motor_shape_observer import PhaseShapeReference, MotorObserverOptions
from snake_control.fixed_section_compliance import FixedSectionCompliance

REPORT = ROOT/'validation/reports/rolling_handover'


def main():
    source = REPORT/'natural_20261013_112105_172339'
    with gzip.open(source/'actual_trace.jsonl.gz', 'rt') as f: rows = [json.loads(line) for line in f]
    states = {round(r['time_s'], 8): r for r in rows}
    ref = PhaseShapeReference.from_dict(json.loads((ROOT/'configs/motor_shape_reference.json').read_text()))
    p = HalfArcParameters(first_joint=6, last_joint=11)
    observer = RollingShapeObserver([.0645]*21, ref, options=MotorObserverOptions(maximum_feedback_gap_s=.5))
    c = RollingHandover(p, [.0645]*21, {6: ref}, resize=False, sliding=False)
    previous, last_time, results = None, None, []
    for row in rows:
        f = row['motor_input']
        if not f or f['feedback_time_s'] < 2. or f['feedback_time_s'] == last_time: continue
        q = np.asarray(f['angles_deg'])
        velocity = np.zeros(20) if previous is None else (q-previous)/(f['feedback_time_s']-last_time)
        previous, last_time = q, f['feedback_time_s']
        o = observer.update(q, f['phase_rad'], f['enabled'], last_time)
        if not o['direction']: continue
        origin = states[round(last_time, 8)]
        c.window, c.enabled = (6, 11), p.enabled.copy()
        c.targets, c.phase = np.asarray(origin['target_rad']), f['phase_rad']+.012
        c.core = FixedSectionCompliance(p, [.0645]*21)
        c.core.targets = c.targets.copy()
        c.observation, c.fresh_feedback, c.last_time = o, True, last_time
        c.angle_velocity_available, c.angle_velocity_deg_s = True, velocity
        c.pending = dict(kind='slide', to=o['suggested_window'], time_s=last_time, ready_since=None, hold_since=None)
        c._handover(q, last_time)
        results.append(dict(time_s=last_time, gate=c.last_gate, simulated_exchange_available=c.pending is None))
    result = dict(source=source.name, suggestions=len(results),
        available=sum(r['simulated_exchange_available'] for r in results),
        gates={k: dict(Counter(r['gate'][k] for r in results)) for k in ('command_open', 'next_command_open', 'phase_compatible')},
        opportunities=results)
    (REPORT/'moving_gate_preflight.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k != 'opportunities'}))
    print('First available:', next((r for r in results if r['simulated_exchange_available']), None))


if __name__ == '__main__': main()
