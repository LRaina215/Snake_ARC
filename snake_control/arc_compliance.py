"""Local measured-curvature yielding with a support-driven moving window.

Pure Python, angles in degrees. This core knows no pipe radius, route, root pose
or simulator. Support observations are supplied by the execution adapter.
"""
from dataclasses import dataclass, asdict
import math
import statistics


@dataclass(frozen=True)
class SlidingArcOptions:
    gain: float = .02
    curvature_filter_s: float = .5
    minimum_amplitude_deg: float = 25.
    maximum_amplitude_deg: float = 60.
    guard_joints: int = 2
    minimum_width: int = 6
    maximum_width: int = 12
    contact_on_s: float = .12
    contact_off_s: float = .8
    window_dwell_s: float = .6
    transfer_s: float = .8
    target_rate_deg_s: float = 45.
    contact_loss_hold_s: float = .5

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Compliance options must be finite')
        if not 0 <= self.gain < .5 or not 0 < self.minimum_amplitude_deg <= self.maximum_amplitude_deg <= 90:
            raise ValueError('Invalid compliance gain or amplitude bounds')
        if not 2 <= self.minimum_width <= self.maximum_width or self.guard_joints < 0:
            raise ValueError('Invalid window bounds')
        if any(type(v) is not int for v in (self.minimum_width,self.maximum_width,self.guard_joints)) or self.minimum_width%2 or self.maximum_width%2:
            raise ValueError('Window widths must be even integers; guard must be integer')
        if min(self.curvature_filter_s, self.contact_on_s, self.contact_off_s, self.window_dwell_s,
               self.transfer_s, self.target_rate_deg_s, self.contact_loss_hold_s) <= 0:
            raise ValueError('Time constants and target speed must be positive')


def support_seed(scores, minimum_width=6, maximum_width=12, guard=2):
    """Pick the strongest connected support region; returns one-based bounds or None."""
    contacts = [i for i,value in enumerate(scores) if value>0]
    if not contacts:
        return None
    groups = [[contacts[0]]]
    for i in contacts[1:]:
        if i-groups[-1][-1]>2:
            groups.append([])
        groups[-1].append(i)
    region = max(groups,key=lambda group:sum(scores[i] for i in group))
    width = max(minimum_width,region[-1]-region[0]+1+2*guard)
    width = min(len(scores),maximum_width,width+width%2)
    lo = max(0,min(len(scores)-width,round(statistics.mean(region)-(width-1)/2)))
    return lo+1,lo+width


class SlidingArcCompliance:
    def __init__(self, measured_deg, phase, seed_window, options=SlidingArcOptions(), initial_enabled=None):
        self.options = options
        self.n = len(measured_deg)
        first, last = seed_window
        if not 1 <= first <= last <= self.n or self.n < options.minimum_width:
            raise ValueError('Invalid compliance seed window')
        if not math.isfinite(phase) or not all(math.isfinite(q) for q in measured_deg):
            raise ValueError('Invalid measured angles or phase')
        self.phase = phase
        self.window = (first - 1, last - 1)
        self.weights = ([float(e) for e in initial_enabled] if initial_enabled is not None else
                        [float(first <= j <= last) for j in range(1,self.n+1)])
        if len(self.weights)!=self.n:
            raise ValueError('Initial enable mask size mismatch')
        self.targets = list(measured_deg)
        self.amplitudes = [options.maximum_amplitude_deg] * self.n
        self.contact_age = [0.] * self.n
        self.absent_age = [0.] * self.n
        self.support = [False] * self.n
        self.proposal, self.proposal_age = None, 0.
        self.unsupported_s = 0.
        self.events = []
        self.time_s = 0.
        self.phase_paused = False
        self.transfer_remaining_s = 0.

    def update(self, measured_deg, support, dt, phase):
        if len(measured_deg) != self.n or len(support) != self.n or not all(math.isfinite(q) for q in measured_deg):
            raise ValueError('Incomplete or nonfinite compliance observations')
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(phase):
            raise ValueError('Invalid compliance time or phase')
        p = self.options
        self.time_s += dt
        self.phase = phase
        for i, observed in enumerate(support):
            self.contact_age[i] = self.contact_age[i] + dt if observed else 0.
            self.absent_age[i] = 0. if observed else self.absent_age[i] + dt
            if self.contact_age[i] >= p.contact_on_s:
                self.support[i] = True
            elif self.absent_age[i] >= p.contact_off_s:
                self.support[i] = False
        self.unsupported_s = 0. if any(support) else self.unsupported_s + dt
        self.transfer_remaining_s = max(0.,self.transfer_remaining_s-dt)
        self.phase_paused = self.unsupported_s >= p.contact_loss_hold_s or self.transfer_remaining_s>0

        # Orthogonal adjacent joints form a measured curvature envelope.
        # Rebuild from ACTUAL angles each update, never recursively deform a target.
        previous = [i for i, w in enumerate(self.weights) if w > .95]
        alpha = 1 - math.exp(-dt / p.curvature_filter_s)
        for i in previous:
            others = [j for j in previous if (j - i) % 2]
            if not others:
                continue
            opposite = min(others, key=lambda j: abs(j - i))
            observed = math.hypot(measured_deg[i], measured_deg[opposite])
            desired = min(p.maximum_amplitude_deg, max(p.minimum_amplitude_deg, observed * (1 + p.gain)))
            self.amplitudes[i] += alpha * (desired - self.amplitudes[i])

        contacts = [i for i, active in enumerate(self.support) if active]
        if contacts:
            # Preserve one support-connected region; isolated distant collisions
            # must not activate all intervening hanging joints.
            groups = [[contacts[0]]]
            for i in contacts[1:]:
                if i - groups[-1][-1] > 2:
                    groups.append([])
                groups[-1].append(i)
            center = sum(self.window) / 2
            contacts = min(groups, key=lambda g: abs(statistics.mean(g) - center))
            amplitude = statistics.median(self.amplitudes[i] for i in previous) if previous else p.maximum_amplitude_deg
            width = max(p.minimum_width, contacts[-1] - contacts[0] + 1 + 2 * p.guard_joints)
            width = min(self.n, p.maximum_width, width + width % 2)
            lo = max(0, min(self.n - width, round(statistics.mean(contacts) - (width - 1) / 2)))
            candidate = (lo, lo + width - 1)
            if self.window[0] <= contacts[0] and self.window[1] >= contacts[-1]:
                candidate = self.window  # Do not recenter an already sufficient support window.
            if candidate == self.window:
                self.proposal, self.proposal_age = None, 0.
            elif candidate != self.proposal:
                self.proposal, self.proposal_age = candidate, dt
            else:
                self.proposal_age += dt
                if self.proposal_age >= p.window_dwell_s:
                    old = self.window
                    self.window = candidate
                    self.transfer_remaining_s = p.transfer_s + .4
                    self.phase_paused = True
                    self.events.append(dict(time_s=self.time_s, old=[old[0]+1,old[1]+1],
                                            new=[lo+1,lo+width], support=[i+1 for i in contacts]))
                    self.proposal, self.proposal_age = None, 0.
                    old_width = old[1]-old[0]+1
                    scale = min(1.,old_width/width)
                    for i in range(self.n):
                        if self.weights[i]>0:
                            self.amplitudes[i] = max(p.minimum_amplitude_deg,self.amplitudes[i]*scale)
                    for i in range(lo, lo + width):
                        if self.weights[i] == 0:
                            self.targets[i] = measured_deg[i]
                            self.amplitudes[i] = max(p.minimum_amplitude_deg,amplitude*scale)

        for i in range(self.n):
            desired_weight = float(self.window[0] <= i <= self.window[1])
            self.weights[i] += max(-dt/p.transfer_s, min(dt/p.transfer_s, desired_weight-self.weights[i]))
            if self.weights[i] < 1e-9:
                self.weights[i] = 0.
            wave = math.cos(phase) if i%2==0 else math.sin(phase)
            goal = self.amplitudes[i] * wave
            # During handover follow measured pose -> rolling shape; exiting
            # joints yield to measurement before their actuator is switched off.
            goal = measured_deg[i] + self.weights[i] * (goal - measured_deg[i])
            limit = p.target_rate_deg_s * dt
            self.targets[i] += max(-limit, min(limit, goal-self.targets[i]))
        return self.targets.copy(), [w > 0 for w in self.weights]

    def snapshot(self):
        return dict(window=[self.window[0]+1,self.window[1]+1], weights=self.weights.copy(),
                    support=[i+1 for i,flag in enumerate(self.support) if flag],
                    amplitude_deg=self.amplitudes.copy(), phase_paused=self.phase_paused,
                    unsupported_s=self.unsupported_s,transfer_remaining_s=self.transfer_remaining_s)
