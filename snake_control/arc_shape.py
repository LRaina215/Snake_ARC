"""Evaluation-only actual open-arc geometry, never imported by a controller."""
import numpy as np

# Legacy amplitude/window experiment bounds; the new fixed-section runner
# freezes its own stricter 150..218 degree band. Do not use these constants to
# accept the new controller. Sharp-corner projection is explicitly flagged.
WRAP_LIMIT_DEG = 230.
PAIR_TURN_TOLERANCE = 1.10
MAX_AXIS_SPREAD_DEG = 60.


class ArcShapeEvaluator:
    def __init__(self, route):
        route = np.asarray(route, float)
        self.a = route[:-1]
        self.delta = np.diff(route, axis=0)
        self.length2 = np.sum(self.delta * self.delta, axis=1)
        self.tangent = self.delta / np.sqrt(self.length2)[:, None]
        if np.max(np.abs(self.tangent[:, 2])) > 1e-6:
            raise ValueError('Evaluator currently requires a horizontal centerline')
        self.up = np.array([0., 0., 1.])
        self.lateral = np.cross(self.tangent, self.up)

    def measure(self, points, joint_deg, first, last):
        pts = np.asarray(points)[first:last + 2]
        frac = np.clip(np.sum((pts[:, None] - self.a) * self.delta, axis=2) / self.length2, 0, 1)
        nearest = self.a + frac[:, :, None] * self.delta
        index = np.argmin(np.linalg.norm(pts[:, None] - nearest, axis=2), axis=1)
        rel = pts - nearest[np.arange(len(pts)), index]
        x = np.sum(rel * self.lateral[index], axis=1)
        y = rel @ self.up
        theta = np.unwrap(np.arctan2(x, y))
        tangent = self.tangent[index]
        spread = np.degrees(np.arccos(np.clip((tangent @ tangent.T).min(), -1, 1)))
        radial = np.hypot(x, y)
        angles = np.asarray(joint_deg)[first - 1:last]
        # Pair-angle proxy is distinct from actual pipe wrapping.
        pair_turn = sum(np.hypot(angles[i], angles[i + 1]) for i in range(0, len(angles) - 1, 2))
        if len(angles) % 2:
            pair_turn += abs(angles[-1])
        links = np.diff(pts, axis=0)
        unit = links / np.maximum(np.linalg.norm(links, axis=1)[:, None], 1e-12)
        body_turn = np.degrees(np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1))).sum()
        return dict(wrap_deg=float(np.degrees(np.ptp(theta))),
                    angular_travel_deg=float(np.degrees(np.abs(np.diff(theta)).sum())),
                    axis_spread_deg=float(spread),
                    wrap_valid=bool(spread <= MAX_AXIS_SPREAD_DEG and radial.min() > .02),
                    pair_turn_deg=float(pair_turn), body_turn_deg=float(body_turn),
                    endpoint_gap_m=float(np.linalg.norm(pts[-1] - pts[0])),
                    min_radial_m=float(radial.min()), max_radial_m=float(radial.max()))
