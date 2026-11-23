"""Distribute bearing turn over material intervals before joint sampling."""
import math

import numpy as np


def fit_section_turn(gait, bounds, field, extra_length, angle_limit):
    """Keep the requested bend as far as the alternating axes can realize it.

    The two Bishop components of each material interval are scaled together.
    Quadrature joint integrals describe the same curve at every roll phase.
    """
    stop = bounds[-2]
    stretch = 1. + extra_length / stop
    stretched = np.r_[bounds[:-1] * stretch, bounds[-1]]
    field = field / stretch
    spacing = gait.l
    mid = .5 * (stretched[1:] + stretched[:-1])
    bins = np.minimum((mid / spacing).astype(int), int(stretched[-2] / spacing))
    count = int(bins.max()) + 1
    masks = bins == np.arange(count)[:, None]
    pieces = masks[:, :, None] * field
    turn = masks @ (np.diff(stretched) * np.linalg.norm(field, axis=1))
    a = np.array([gait.sampled_targets(stretched, part, 0.) for part in pieces]).T
    b = np.array([gait.sampled_targets(stretched, part, math.pi / 2.) for part in pieces]).T
    phase = np.linspace(0., math.pi, 12, endpoint=False)
    matrix = (np.cos(phase)[:, None, None] * a + np.sin(phase)[:, None, None] * b).reshape(-1, count)
    # The nearest phase sample is at most pi/24 away. Scale the inequalities
    # accordingly so that a peak between two sampled phases fits as well.
    limit = angle_limit * math.cos(math.pi / 24.)
    eye = np.eye(count)
    zero = np.zeros_like(matrix)
    constraints = np.block([[matrix, zero], [-matrix, zero], [eye, -eye], [-eye, -eye]])
    rhs = np.r_[np.full(2 * len(matrix), limit), np.ones(count), -np.ones(count)]
    effort = np.r_[-4. * turn, turn]
    net_turn = np.r_[turn, np.zeros(count)][None, :]
    from scipy.optimize import linprog

    fit = linprog(effort, A_ub=np.vstack((constraints, net_turn)), b_ub=np.r_[rhs, turn.sum()],
                  bounds=[(0., 1.30)] * count + [(0., None)] * count, method='highs')
    weights = fit.x[:count]
    return stretched, field * weights[bins, None], float((turn @ weights) / turn.sum())
