"""Inspect archived angles only; no dynamics or controller parameter search."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def bending(angles, window):
    first, last = window
    indices = np.arange(max(1, first-2), min(20, last+2)+1)
    turns = np.abs(np.asarray(angles)[indices-1])
    return dict(center=float(indices @ turns / max(turns.sum(), 1e-12)),
                head=float(turns[indices < first].sum()), tail=float(turns[indices > last].sum()))


def main():
    for label in ('whole_body', 'endpoint'):
        path = ROOT / 'validation/reports/angle_observer' / (label + '_observer_trace.jsonl.gz')
        with gzip.open(path, 'rt', encoding='utf8') as stream:
            rows = [json.loads(line) for line in stream]
        for stage in ('no_force', 'right_force', 'recovery'):
            selected = [r for r in rows if r['stage'] == stage and r['input']['enabled'][7:13] == [True]*6]
            values = [bending(r['input']['angles_deg'], (8, 13)) for r in selected]
            print(label, stage, {key: np.percentile([r[key] for r in values], [0, 50, 100]).round(4).tolist()
                                for key in ('center', 'head', 'tail')})


if __name__ == '__main__':
    main()
