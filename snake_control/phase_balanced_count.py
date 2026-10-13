"""Recent angle-derived shape statistics, balanced by rolling phase.

This object has no model, contacts or actuator interface. Holding phase cannot
create phase coverage. The pi-period grouping is a geometric approximation,
not a guarantee for arbitrary external loads or gravity-dependent shapes.
"""
import math
import numpy as np


class PhaseBalancedCount:
    def __init__(self, groups=4, minimum_samples_per_group=2,
                 phase_span_rad=math.pi, maximum_feedback_gap_s=.5):
        self.groups = groups
        self.minimum_samples_per_group = minimum_samples_per_group
        self.phase_span_rad = phase_span_rad
        self.maximum_feedback_gap_s = maximum_feedback_gap_s
        self.reset()

    def reset(self):
        self.samples = []
        self.last_time = self.last_phase = None

    def update(self, count, turn_deg, phase_rad, time_s, nominal_turn_deg=180.):
        if not all(math.isfinite(v) for v in (count, turn_deg, phase_rad, time_s, nominal_turn_deg)):
            raise ValueError('Finite angle-derived count, turn, phase and time required')
        if self.last_time is not None:
            if time_s <= self.last_time:
                raise ValueError('Only new feedback may update phase statistics')
            if time_s-self.last_time > self.maximum_feedback_gap_s or phase_rad < self.last_phase-1e-8:
                self.reset()
        sample = dict(count=float(count), turn_deg=float(turn_deg), nominal_turn_deg=float(nominal_turn_deg),
                      phase_rad=float(phase_rad), time_s=float(time_s))
        if self.samples and abs(phase_rad-self.samples[-1]['phase_rad']) <= 1e-8:
            self.samples[-1] = sample
        else:
            self.samples.append(sample)
        self.last_time, self.last_phase = time_s, phase_rad
        self.samples = [s for s in self.samples if s['phase_rad'] >= phase_rad-self.phase_span_rad-1e-8]
        return self.summary()

    def summary(self):
        buckets = [[] for _ in range(self.groups)]
        for sample in self.samples:
            index = int((sample['phase_rad'] % self.phase_span_rad)/self.phase_span_rad*self.groups) % self.groups
            buckets[index].append(sample)
        count_medians = [None if not b else float(np.median([s['count'] for s in b])) for b in buckets]
        turn_medians = [None if not b else float(np.median([s['turn_deg'] for s in b])) for b in buckets]
        turn_error_medians = [None if not b else float(np.median([s['turn_deg']-s['nominal_turn_deg'] for s in b])) for b in buckets]
        ready = all(len(b) >= self.minimum_samples_per_group for b in buckets)
        return dict(ready=ready, phase_period_rad=self.phase_span_rad, group_sample_counts=[len(b) for b in buckets],
                    group_count_medians=count_medians, group_turn_medians_deg=turn_medians,
                    balanced_count_joints=None if not ready else float(np.mean(count_medians)),
                    balanced_turn_deg=None if not ready else float(np.mean(turn_medians)),
                    group_turn_error_medians_deg=turn_error_medians,
                    balanced_turn_error_deg=None if not ready else float(np.mean(turn_error_medians)),
                    maximum_group_count_joints=None if not ready else max(count_medians),
                    represented_phase_span_rad=0. if not self.samples else self.samples[-1]['phase_rad']-self.samples[0]['phase_rad'],
                    new_samples=len(self.samples), observes_contact=False)
