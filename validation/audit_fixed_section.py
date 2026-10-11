"""Audit saved commands and sharp-corner geometry; no new dynamics or trials."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np

from snake_control.fixed_section_simulation import FixedSectionSimulation
from snake_control.rolling_arc import ArcParameters
from snake_control.section_geometry import local_section
from snake_control.simulation import ROOT


CONTINUOUS_LABELS = [
    'r2_roll_k0', 'r2_roll_k2', 'r3_small_k0', 'r3_small_k2',
    'r4_large_k2', 'r5_bend90', 'r5_elbow90', 'r5_taper45', 'r5_taper90',
]


def audit_sent_targets(path):
    """Check filtered commands as well as the pre-filter candidate check.

    Mechanical dimensions come from the robot model. No step() is called.
    The geometry-bound result is independent of scene contact/pipe metadata.
    """
    first_report = json.loads((path / f'{CONTINUOUS_LABELS[0]}.json').read_text(encoding='utf-8'))
    parameters = ArcParameters(**first_report['parameters'])
    sim = FixedSectionSimulation('straight', parameters)
    rows = []
    for label in CONTINUOUS_LABELS:
        report = json.loads((path / f'{label}.json').read_text(encoding='utf-8'))
        parameters = ArcParameters(**report['parameters'])
        with np.load(path / f'{label}_trace.npz') as saved:
            turns, gaps = [], []
            for time_s, angles in zip(saved['metrics'][:, 0], saved['targets_rad']):
                if time_s < parameters.settle_s + 1:
                    continue
                geometry = local_section(angles, sim.lengths, parameters.first_joint, parameters.last_joint)
                active = geometry['points'][1:-1]
                turns.append(np.degrees(geometry['total_turn']))
                gaps.append(np.linalg.norm(active[-1] - active[0]) /
                            np.linalg.norm(np.diff(active, axis=0), axis=1).sum())
            rows.append(dict(
                label=label,
                maximum_sent_target_turn_deg=float(max(turns)),
                minimum_sent_target_gap_fraction=float(min(gaps)),
                passed=bool(max(turns) <= 190.01 and min(gaps) >= .4),
            ))
    (path / 'sent_target_geometry_audit.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    return rows


def audit_sharp_corner(path):
    """Keep the failed single-axis check, supplement it with open body geometry."""
    report = json.loads((path / 'r5_elbow90.json').read_text(encoding='utf-8'))
    parameters = ArcParameters(**report['parameters'])
    sim = FixedSectionSimulation('elbow90_section_exit', parameters, -.25)
    with np.load(path / 'r5_elbow90_trace.npz') as saved:
        fields = list(saved['shape_fields'])
        times = saved['metrics'][:, 0]
        measured = saved['qpos'][:, sim.drive.qadr]
        invalid = (times > parameters.settle_s + 1) & ~saved['shape'][:, fields.index('wrap_valid')].astype(bool)
        geometries = [local_section(q, sim.lengths, parameters.first_joint, parameters.last_joint)
                      for q in measured[invalid]]
        turns = np.degrees([g['total_turn'] for g in geometries])
        fits = np.array([g['fit_rms'] for g in geometries])
        gaps = saved['shape'][invalid, fields.index('gap_fraction')]
        transition = []
        for i in np.flatnonzero((times > 19.9) & (times < 20.8)):
            geometry = local_section(measured[i], sim.lengths, parameters.first_joint, parameters.last_joint)
            transition.append(dict(time_s=float(times[i]),
                                   raw_wrap_deg=float(saved['shape'][i, fields.index('wrap_deg')]),
                                   axis_span_deg=float(saved['shape'][i, fields.index('axis_spread_deg')]),
                                   rms_m=float(geometry['fit_rms'])))
    audit = dict(
        note='Supplementary open-shape audit only. Original single-pipe-axis observability check remains failed; thresholds unchanged.',
        invalid_samples=int(invalid.sum()),
        invalid_interval_s=[float(times[invalid].min()), float(times[invalid].max())],
        measured_section_turn_range_deg=[float(turns.min()), float(turns.max())],
        max_plane_fit_rms_mm=float(fits.max() * 1000),
        min_gap_fraction=float(gaps.min()),
        exit_cycle_wrap_peaks=report['cycle_max_wrap_deg'][-3:],
        axis_switch_frames=transition,
    )
    (path / 'r5_elbow90_geometry_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, default=ROOT / 'validation/reports/fixed_section')
    args = parser.parse_args()
    commands = audit_sent_targets(args.path)
    sharp = audit_sharp_corner(args.path)
    print('Saved filtered commands:', len(commands), 'passed:', sum(r['passed'] for r in commands))
    print('Sharp corner invalid single-axis samples:', sharp['invalid_samples'], '(original verdict preserved)')
