"""Load-bearing window selection and one-budget open-arc handover.

No MuJoCo or pipe geometry here. `support_up` is an explicit observation,
which the simulation adapter obtains from contacts; it is not servo load.
The geometric feedback is delegated to FixedSectionCompliance unchanged.
Joint numbers in bounds are one-based; arrays use zero-based indices.
"""
from dataclasses import dataclass, replace, asdict
import math
import numpy as np
from .fixed_section_compliance import FixedSectionCompliance, SectionOptions
from .section_geometry import local_section


@dataclass(frozen=True)
class WindowOptions:
    width: int = 6
    support_filter_s: float = .4
    boundary_offset_joints: float = 1.25
    persistence_s: float = 1.2
    cooldown_s: float = 5.
    handover_s: float = 3.
    handover_timeout_s: float = 8.
    tracking_tolerance_deg: float = 12.
    minimum_support_n: float = 2.
    minimum_support_coverage: float = .8
    contract_for_handover: bool = True


def open_geometry(q, lengths, window):
    """Actual body plane/curve, independent of route and contact observations."""
    g = local_section(q, lengths, *window)
    points = g['points'][1:-1]
    length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    signed = sum(s.turn for s in g['curve'] if 1 <= s.joint <= window[1]-window[0]+1)
    return dict(turn_deg=math.degrees(g['total_turn']), fit_rms_m=g['fit_rms'],
                gap_fraction=float(np.linalg.norm(points[-1]-points[0])/length),
                winding_coherence=abs(signed)/max(g['total_turn'], 1e-12))


class SlidingSectionCompliance:
    def __init__(self, parameters, lengths, section_options=SectionOptions(),
                 window_options=WindowOptions(), automatic=True):
        self.p = parameters
        self.lengths = np.asarray(lengths)
        self.section_options = section_options
        self.options = window_options
        self.automatic = automatic
        self.window = None
        self.core = None
        self.weights = np.zeros(parameters.joints)
        self.support = np.zeros(parameters.joints)
        self.raw_support = self.support.copy()
        self.transfer = None
        self.preparation = None
        self.budget_scale = 1.
        self.restore_start_s = None
        self.handover_destination = None
        self.handover_origin = None
        self.events = []
        self.pending = None
        self.pending_since = 0.
        self.cooldown_until = 0.
        self.next_selection_s = 0.
        self.selection_reason = 'waiting for load-bearing supported shape'
        self.target = None
        self.sent_geometry = None
        self.sent_constraint_violations = 0
        self.next_command_audit_s = 0.

    def _mask(self, window):
        indices = np.arange(1, self.p.joints+1)
        return (indices >= window[0]) & (indices <= window[1])

    def _coverage(self, window):
        # Include the downstream boundary link, which is shaped by the last
        # active hinge. Observation bins refer to downstream physical links.
        return float(self.support[window[0]-1:window[1]+1].sum()/max(self.support.sum(), 1e-12))

    def _nominal(self, phase, weights=None):
        """One π budget on each orthogonal hinge family, at every transition.

        For six uniform joints this is exactly Step1's 60° cos/sin. For a
        shifted/expanded arc each family's sum remains π, instead of adding
        another 60° for every new hinge. Physical lengths weight the budget.
        """
        weights = self.weights if weights is None else weights
        reference = np.zeros(self.p.joints)
        modules = (self.lengths[:-1]+self.lengths[1:])/2
        weighted = weights*modules
        for parity, factor in ((0, math.cos(phase)), (1, self.p.direction*math.sin(phase))):
            indices = np.arange(parity, self.p.joints, 2)
            reference[indices] = self.budget_scale*math.pi*factor*weighted[indices]/max(weighted[indices].sum(), 1e-12)
        return reference

    def _make_core(self, bounds, measured):
        old = self.core
        self.core = FixedSectionCompliance(replace(self.p, first_joint=bounds[0], last_joint=bounds[1]),
                                           self.lengths, self.section_options)
        self.core.targets = measured.copy() if old is None else old.targets.copy()
        if old is not None:
            self.core.previous_axis = old.previous_axis

    def _acquire(self, measured, time_s):
        if self.support.sum() < self.options.minimum_support_n:
            return False
        center = float(self.support @ np.arange(1, self.p.joints+1)/self.support.sum())
        candidates = []
        for first in range(2, self.p.joints-self.options.width+1):
            bounds = (first, first+self.options.width-1)
            g = open_geometry(measured, self.lengths, bounds)
            coverage = self._coverage(bounds)
            if coverage < self.options.minimum_support_coverage or not (
                    150 <= g['turn_deg'] <= 218 and g['gap_fraction'] >= .4 and g['fit_rms_m'] <= .015):
                continue
            score = abs(g['turn_deg']-180)/30 + abs(center-np.mean(bounds))/3 + 4*(1-coverage)
            candidates.append((score, bounds))
        if not candidates:
            self.selection_reason = 'no six-joint supported open candidate'
            return False
        self.window = min(candidates)[1]
        self.weights = self._mask(self.window).astype(float)
        self._make_core(self.window, measured)
        self.selection_reason = 'acquired from actual shape and upward contact load'
        self.events.append(dict(time_s=time_s, kind='acquire', to=list(self.window),
                                reason=self.selection_reason, support_center_joint=center))
        self.cooldown_until = time_s+self.options.cooldown_s
        return True

    def request(self, bounds, measured, phase, time_s, reason='manual neighbour request'):
        bounds = tuple(map(int, bounds))
        if self.window is None or self.transfer is not None or self.preparation is not None:
            return False
        if bounds == self.window:
            return True
        if bounds[0] < 2 or bounds[1] >= self.p.joints:
            return False
        # Fixed-width migration is adjacent. Width changes are exposed as a
        # separate operation for a later, evidence-triggered extension.
        if bounds[1]-bounds[0] != self.window[1]-self.window[0] or abs(bounds[0]-self.window[0]) != 1:
            return False
        union = (min(bounds[0], self.window[0]), max(bounds[1], self.window[1]))
        weights = self._mask(bounds).astype(float)
        reference = measured.copy()
        reference[self._mask(union)] = self._nominal(phase, weights)[self._mask(union)]
        candidate = open_geometry(reference, self.lengths, union)
        if candidate['turn_deg'] > 190.01 or candidate['gap_fraction'] < .4:
            self.selection_reason = 'candidate reference violates union opening'
            self.events.append(dict(time_s=time_s, kind='reject', to=list(bounds), reason=self.selection_reason))
            return False
        if self.options.contract_for_handover:
            self.handover_destination, self.handover_origin = bounds, self.window
            bounds = (max(bounds[0], self.window[0]), min(bounds[1], self.window[1]))
            union = self.window
        self.preparation = dict(new=bounds, union=union, start_s=time_s, reason=reason,
                                from_scale=self.budget_scale)
        self.restore_start_s = None
        self.events.append(dict(time_s=time_s, kind='prepare', old=list(self.window), to=list(bounds), reason=reason))
        self.pending = None
        return True

    def _begin_prepared(self, measured, time_s):
        preparation = self.preparation
        bounds, union, reason = preparation['new'], preparation['union'], preparation['reason']
        weights = self._mask(bounds).astype(float)
        self.transfer = dict(old=self.window, new=bounds, union=union, start_s=time_s,
                             from_weights=self.weights.copy(), to_weights=weights,
                             fraction=0., rollback_s=None, reason=reason)
        self._make_core(union, measured)
        self.events.append(dict(time_s=time_s, kind='begin', old=list(self.window), to=list(bounds), reason=reason))
        self.selection_reason = reason
        self.pending = None
        self.preparation = None
        return True

    def _select(self, measured, phase, time_s):
        if not self.automatic or self.transfer is not None or self.preparation is not None or time_s < self.cooldown_until:
            return
        if self.support.sum() < self.options.minimum_support_n:
            self.pending = None
            return
        center = float(self.support @ np.arange(1, self.p.joints+1)/self.support.sum())
        offset = center-np.mean(self.window)
        if abs(offset) <= self.options.boundary_offset_joints:
            self.pending = None
            self.selection_reason = 'retain: support remains inside window deadband'
            return
        direction = 1 if offset > 0 else -1
        candidate = (self.window[0]+direction, self.window[1]+direction)
        if candidate[0] < 2 or candidate[1] >= self.p.joints or self._coverage(candidate) < self.options.minimum_support_coverage:
            self.pending = None
            self.selection_reason = 'retain: neighbour does not cover load or reaches boundary'
            return
        if self.pending != candidate:
            self.pending, self.pending_since = candidate, time_s
            self.selection_reason = 'waiting for persistent boundary load displacement'
        elif time_s-self.pending_since >= self.options.persistence_s:
            self.request(candidate, measured, phase, time_s, 'persistent observed load displacement')

    def _advance_transfer(self, measured, phase, time_s):
        t = self.transfer
        if t is None:
            return
        if t['rollback_s'] is not None:
            u = np.clip((time_s-t['rollback_s'])/self.options.handover_s, 0., 1.)
            fraction = t['rollback_fraction']*(1-u*u*(3-2*u))
        else:
            u = np.clip((time_s-t['start_s'])/self.options.handover_s, 0., 1.)
            fraction = min(.95, float(u*u*(3-2*u)))
        t['fraction'] = fraction
        self.weights = (1-fraction)*t['from_weights']+fraction*t['to_weights']
        reference = self._nominal(phase)
        new = self._mask(t['new'])
        exiting = self._mask(t['old']) & ~new
        error = float(np.max(abs(measured[new]-reference[new]), initial=0))
        exit_error = float(np.max(abs(measured[exiting]), initial=0))
        tolerance = math.radians(self.options.tracking_tolerance_deg)
        ready = (u >= 1 and error < tolerance and exit_error < tolerance
                 and self.support.sum() >= self.options.minimum_support_n
                 and self._coverage(t['new']) >= self.options.minimum_support_coverage)
        if t['rollback_s'] is None and ready:
            self.window = t['new']
            self.weights = t['to_weights'].copy()
            self._make_core(self.window, measured)
            final_stage = self.handover_destination is None or self.window == self.handover_destination
            self.events.append(dict(time_s=time_s, kind='complete' if final_stage else 'contract',
                                    old=list(t['old'] if self.handover_origin is None else self.handover_origin), to=list(self.window),
                                    tracking_error_deg=math.degrees(error), exit_angle_deg=math.degrees(exit_error),
                                    support_coverage=self._coverage(self.window)))
            self.transfer = None
            if final_stage:
                self.restore_start_s = time_s
                self.handover_destination = self.handover_origin = None
            else:
                destination = self.handover_destination
                self.preparation = dict(new=destination, union=destination, start_s=time_s,
                                        reason=t['reason'], from_scale=self.budget_scale)
            self.cooldown_until = time_s+self.options.cooldown_s
        elif t['rollback_s'] is None and time_s-t['start_s'] >= self.options.handover_timeout_s:
            t['rollback_s'], t['rollback_fraction'] = time_s, fraction
            self.events.append(dict(time_s=time_s, kind='rollback', to=list(t['old']),
                                    reason='new-window load/tracking not ready', tracking_error_deg=math.degrees(error),
                                    exit_angle_deg=math.degrees(exit_error), support_coverage=self._coverage(t['new'])))
        elif t['rollback_s'] is not None and u >= 1:
            self.window, self.weights = t['old'], t['from_weights'].copy()
            self._make_core(self.window, measured)
            self.events.append(dict(time_s=time_s, kind='abort', to=list(self.window)))
            self.transfer = None
            self.restore_start_s = time_s
            self.cooldown_until = time_s+self.options.cooldown_s

    def update(self, measured, phase, support_up, time_s, dt):
        measured, support_up = np.asarray(measured), np.asarray(support_up)
        if measured.shape != (self.p.joints,) or support_up.shape != measured.shape:
            raise ValueError('Expected one angle and upward support observation per joint')
        if not np.isfinite(measured).all() or not np.isfinite(support_up).all() or np.any(support_up < 0):
            raise ValueError('Nonfinite angles or invalid support observation')
        self.raw_support = support_up.copy()
        self.support += (1-math.exp(-dt/self.options.support_filter_s))*(support_up-self.support)
        if self.window is None:
            if not self._acquire(measured, time_s):
                self.target = measured.copy()
                return self.target.copy(), self.weights.copy()
        if time_s+1e-10 >= self.next_selection_s:
            self.next_selection_s = time_s+self.section_options.geometry_period_s
            self._select(measured, phase, time_s)
        if self.preparation is not None:
            preparation = self.preparation
            u = np.clip((time_s-preparation['start_s'])/2., 0., 1.)
            self.budget_scale = preparation['from_scale']-(preparation['from_scale']-5/6)*u*u*(3-2*u)
            union_actual = open_geometry(measured, self.lengths, preparation['union'])
            union_sent = open_geometry(self.core.targets, self.lengths, preparation['union'])
            if u >= 1 and union_actual['turn_deg'] <= 165 and union_sent['turn_deg'] <= 165 and union_actual['gap_fraction'] >= .4:
                self._begin_prepared(measured, time_s)
            elif time_s-preparation['start_s'] >= 12:
                self.events.append(dict(time_s=time_s, kind='reject', to=list(preparation['new']),
                                        reason='cannot open the complete handover union before enabling'))
                self.preparation = None
                self.handover_destination = self.handover_origin = None
                self.restore_start_s = time_s
                self.cooldown_until = time_s+self.options.cooldown_s
        elif self.transfer is None and self.restore_start_s is not None:
            u = np.clip((time_s-self.restore_start_s)/2., 0., 1.)
            self.budget_scale = 5/6+(1-5/6)*u*u*(3-2*u)
            if u >= 1:
                self.restore_start_s = None
        self._advance_transfer(measured, phase, time_s)
        self.target = self.core.update(measured, phase, time_s, dt, self._nominal(phase))
        # Audit the actual filtered command, not just the held geometric goal.
        if time_s+1e-10 >= self.next_command_audit_s:
            self.next_command_audit_s = time_s+.02
            self.sent_geometry = open_geometry(self.target, self.lengths, (self.core.first, self.core.last))
            if self.sent_geometry['turn_deg'] > 190.01 or self.sent_geometry['gap_fraction'] < .4:
                self.sent_constraint_violations += 1
        return self.target.copy(), self.weights.copy()

    def snapshot(self):
        result = {} if self.core is None else self.core.snapshot()
        result.update(selected_window=None if self.window is None else list(self.window),
                      weights=self.weights.tolist(), support_up_n=self.raw_support.tolist(),
                      filtered_support_n=self.support.tolist(), selection_reason=self.selection_reason,
                      sent_geometry=self.sent_geometry, sent_constraint_violations=self.sent_constraint_violations,
                      window_options=asdict(self.options))
        result['preparation'] = self.preparation
        result['curvature_budget_deg'] = 180*self.budget_scale
        result['transfer'] = None if self.transfer is None else {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in self.transfer.items()
            if k not in ('from_weights', 'to_weights')}
        return result
