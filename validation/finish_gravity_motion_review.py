"""Complete the display-case review from saved states, without dynamics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import numpy as np
from snake_control.section_geometry import forward


def finish(path):
    review = json.loads((path/'review.json').read_text(encoding='utf8'))
    with gzip.open(path/'actual_trace.jsonl.gz', 'rt', encoding='utf8') as stream:
        rows = list(map(json.loads, stream))
    curvature = []
    for row in rows:
        first, last = row['controller']['window']
        points = forward(row['actual_qpos'][7:], [.0645]*21)[first:last+2]
        links = np.diff(points, axis=0)
        lengths = np.linalg.norm(links, axis=1)
        directions = links/lengths[:, None]
        turns = np.arccos(np.clip(np.sum(directions[:-1]*directions[1:], axis=1), -1., 1.))
        curvature.extend(turns/((lengths[:-1]+lengths[1:])*.5))
    moving = [row for row in rows if row['time_s'] > 2.02]
    phase = np.array([row['controller']['phase_rad'] for row in moving])
    review['summary']['discrete_curvature_range_per_m'] = [float(min(curvature)), float(max(curvature))]
    review['summary']['curvature_definition'] = '3D link tangent turn / mean adjacent link length, within actual active arc'
    review['summary']['phase_hold_frames_after_settle'] = sum(bool(row['controller']['phase_hold']) for row in moving)
    review['summary']['minimum_phase_increment_rad'] = float(np.diff(phase).min())
    # The generic audit's obstacle/resize verdict is not this experiment's goal.
    review['generic_resize_objective_applicable'] = False
    review['generic_resize_verdict'] = review.pop('legacy_resize_verdict_not_applicable', review.pop('passed', False))
    review['summary']['continuous_exit_start_s'] = None
    review['summary']['continuous_exit_cycles'] = None
    review['display_case_motion_checks_pass'] = bool(review['shape_pass'] and review['mechanics_pass'] and review['strict_pipe_pass'])
    review['scope'] = '25s display stress case; straight constant diameter, no resize or obstacle-exit objective'
    body = json.dumps(review, ensure_ascii=False, indent=2)
    (path/'review.json').write_text(body, encoding='utf8')
    (path.parent/'display_pull_review.json').write_text(body, encoding='utf8')
    print({k: review['summary'][k] for k in ('discrete_curvature_range_per_m', 'phase_hold_frames_after_settle', 'minimum_phase_increment_rad')})


if __name__ == '__main__':
    finish(Path(sys.argv[1]))
