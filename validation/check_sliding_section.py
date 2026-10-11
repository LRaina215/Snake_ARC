"""No dynamics: independent FK checks on the complete handover union."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
import numpy as np
from snake_control.rolling_arc import ArcParameters
from snake_control.sliding_section_compliance import SlidingSectionCompliance, open_geometry
from snake_control.section_geometry import forward
from snake_control.simulation import ROOT


def main():
    p = ArcParameters.load(ROOT/'configs/step1.json')
    lengths = np.full(p.joints+1, .0645)
    c = SlidingSectionCompliance(p, lengths)
    maximum = 0.
    minimum_gap = 1.
    for first in (3, 8, 13):
        for direction in (-1, 1):
            old, new = (first, first+5), (first+direction, first+5+direction)
            union = (min(old[0], new[0]), max(old[1], new[1]))
            for fraction in np.linspace(0, 1, 11):
                weights = (1-fraction)*c._mask(old)+fraction*c._mask(new)
                for phase in np.linspace(0, 2*math.pi, 33):
                    q = c._nominal(phase, weights)
                    g = open_geometry(q, lengths, union)
                    maximum = max(maximum, g['turn_deg'])
                    minimum_gap = min(minimum_gap, g['gap_fraction'])
                    assert g['turn_deg'] <= 190.01, (old, direction, fraction, phase, g)
                    assert g['gap_fraction'] >= .4, g
                    # Global parity must give the same spatial section at
                    # phase+2π; verify in FK rather than repeating its formula.
                    assert np.max(abs(forward(q, lengths)-forward(c._nominal(phase+2*math.pi, weights), lengths))) < 1e-12
    assert np.max(abs(c._nominal(.7, p.enabled.astype(float))-p.targets(.7))) < 1e-12
    c.budget_scale = 5/6
    for old, middle, new in (((8, 13), (8, 12), (7, 12)), ((8, 13), (9, 13), (9, 14))):
        for a, b in ((old, middle), (middle, new)):
            union = (min(a[0], b[0]), max(a[1], b[1]))
            for fraction in np.linspace(0, 1, 11):
                weights = (1-fraction)*c._mask(a)+fraction*c._mask(b)
                for phase in np.linspace(0, 2*math.pi, 33):
                    q = c._nominal(phase, weights)
                    g = open_geometry(q, lengths, union)
                    assert g['turn_deg'] <= 190.01 and g['gap_fraction'] >= .4, g
                    assert np.max(abs(q)) < math.pi/2, q
    print(f'All union FK samples: maximum turn {maximum:.6f} deg, minimum gap {minimum_gap:.6f}; phase continuous.')


if __name__ == '__main__':
    main()
