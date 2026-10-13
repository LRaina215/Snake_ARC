"""Phase geometry and existing failure replay; zero new dynamics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
from dataclasses import replace
import numpy as np
from snake_control.adaptive_arc_length import HalfArcParameters
from snake_control.fixed_section_compliance import FixedSectionCompliance
from snake_control.section_geometry import local_section
from snake_control.simulation import ROOT
from snake_control.arc_shape import ArcShapeEvaluator

REPORT = ROOT/'validation/reports/adaptive_arc_length'


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    result = {}
    for bounds in [(8, 13), (7, 13), (8, 14), (7, 14)]:
        p = HalfArcParameters(first_joint=bounds[0], last_joint=bounds[1])
        core = FixedSectionCompliance(p, [.0645]*21)
        turns, gaps, rms, valid = [], [], [], []
        for phase in np.linspace(0., 2*np.pi, 48, endpoint=False):
            good, g = core._check_goal(p.targets(phase))
            points = g['points'][1:-1]
            turns.append(np.degrees(g['total_turn']))
            gaps.append(np.linalg.norm(points[-1]-points[0])/np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            rms.append(g['fit_rms'])
            valid.append(good)
        result[str(bounds)] = dict(phases=48, command_open=bool(all(valid)), turn_range_deg=[min(turns), max(turns)],
                                   minimum_gap=min(gaps), maximum_plane_rms_m=max(rms))
    failure = np.load(ROOT/'validation/reports/sliding_section/s1_head_trace.npz')
    evaluator = ArcShapeEvaluator(np.array([[-1.2, 0., 1.2], [1.8, 0., 1.2]]))
    wraps, commands = [], []
    for q, bone, target, bounds in zip(failure['qpos'], failure['backbone'], failure['targets_rad'], failure['windows']):
        wraps.append(evaluator.measure(bone, np.degrees(q[7:]), *bounds)['wrap_deg'])
        commands.append(np.degrees(local_section(target, [.0645]*21, *bounds)['total_turn']))
    result['historical_overlap_failure'] = dict(source='s1_head_trace.npz', frames=len(wraps),
        actual_pipe_wrap_max_deg=max(wraps),
        command_body_turn_max_deg=max(commands),
        rule='A body-open gate alone did not prevent excessive actual pipe wrapping; check both in dynamics')
    # Guard remains capable of rejecting the old uniform-60 expanded formula.
    p = HalfArcParameters(first_joint=7, last_joint=14)
    wrong = np.where(p.enabled, np.pi/3*np.where(np.arange(20)%2 == 0, 1., 0.), 0.)
    result['wrong_expanded_60_rejected'] = not FixedSectionCompliance(p, [.0645]*21)._check_goal(wrong)[0]
    result['passed'] = all(r['command_open'] for r in result.values() if isinstance(r, dict) and 'command_open' in r) and result['wrong_expanded_60_rejected']
    (REPORT/'preflight.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))
    if not result['passed']:
        raise SystemExit('Reference shape is not feasible; do not run dynamics')


if __name__ == '__main__':
    main()
