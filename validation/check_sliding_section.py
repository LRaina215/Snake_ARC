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
    # A real saved supported shape at the odd-family zero phase: high exiting
    # torque must keep the old window; unloading allows one atomic mask switch.
    saved = np.load(ROOT/'validation/reports/fixed_section/r1_hold90_state_k2_trace.npz')
    measured = saved['qpos'][-1, 7:]
    phase = float(saved['metrics'][-1, 1])
    gate = SlidingSectionCompliance(p, lengths, automatic=False)
    gate.window = (8, 13)
    gate.weights = p.enabled.astype(float)
    gate._make_core(gate.window, measured)
    support = np.zeros(p.joints); support[9] = 30.
    gate.support[:] = support
    assert gate.request((7, 12), measured, phase, 0.)
    loaded = np.zeros(p.joints); loaded[12] = 2.
    gate.update(measured, phase, support, .001, .001, loaded)
    assert gate.window == (8, 13) and gate.phase_hold
    before = gate.core.targets.copy()
    target, weights = gate.update(measured, phase, support, .002, .001, np.zeros(p.joints))
    assert gate.window == (7, 12) and weights.sum() == 6 and weights[12] == 0
    overlap = gate._mask((8, 12))
    assert np.max(abs(target[overlap]-before[overlap])) <= math.radians(45)*.001+1e-10
    assert abs(target[6]-measured[6]) <= math.radians(45)*.001+1e-10
    # Alternating contacts and a long phase hold must not manufacture a
    # persistent load displacement. Cover trimming beyond the first cycle.
    observer = SlidingSectionCompliance(p, lengths)
    for phase in np.linspace(0., 4*math.pi, 1601):
        observer.raw_support[:] = 0.
        observer.raw_support[7] = 20*(1+math.sin(phase))
        observer.raw_support[13] = 20*(1-math.sin(phase))
        observer._observe_phase_load(phase)
    center = observer.phase_load_sum @ np.arange(1, 21)/observer.phase_load_sum.sum()
    assert abs(center-11) < 1e-8 and abs(observer.load_phase_span-2*math.pi) < 1e-8
    before = observer.phase_load_sum.copy()
    observer.raw_support[:] = 1000.
    for _ in range(100):
        observer._observe_phase_load(4*math.pi)
    assert np.array_equal(before, observer.phase_load_sum)
    observer._reset_phase_load(4*math.pi)
    assert observer.load_phase_span == 0 and not observer.load_history
    print(f'All union FK samples: maximum turn {maximum:.6f} deg, minimum gap {minimum_gap:.6f}; phase continuous.')


if __name__ == '__main__':
    main()
