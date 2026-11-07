"""Resume the rejected tail-entry pose and hold the established front while fitting the rear."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.gravity_global_shift import MeasuredGravityShift
from validation.gm_locked_experiment import APPROACH_SETUP
from validation.gm_replay_wrap_experiment import phase_from_front, save
from validation.gm_shift_cycle_experiment import CycleTrial


def run(output, pitch, source, after_drop, source_seat, return_roll):
    record = json.loads((source/'locked_gait.json').read_text(encoding='utf-8'))
    trace = np.load(source/'locked_gait_trace.npz')
    start = max(i for i, row in enumerate(record['samples']) if row['stage'] == ('drop_hold' if after_drop else 'front'))
    trial = CycleTrial(setup=dict(APPROACH_SETUP, scene=record['summary']['scene'],
                                 options=dict(APPROACH_SETUP['options'], pitch=pitch)))
    m, d, sim = trial.model, trial.data, trial.sim
    d.qpos[:], d.qvel[:] = trace['qpos'][start], trace['qvel'][start]
    mujoco.mj_forward(m, d)
    g = trial.gait
    measured = d.qpos[sim.drive.qadr].copy()
    radius = g.o.r_h2 if after_drop else g.o.r_h1
    support = 8 if after_drop else 5
    phase = phase_from_front(g, radius, measured, support)
    curve = MeasuredGravityShift(g, radius, phase, measured, support, upstream_m=(support+1)*g.l)
    target = measured.copy()
    enabled = np.arange(m.nu) < (8 if after_drop else 11)
    locked = measured[:8 if after_drop else 0].copy()
    offset = 0.
    accepted = 8 if after_drop else 11
    ready_since = None
    front_ready_at = 0. if after_drop else None
    events = []
    last_tick = -1.
    state = 'anchored_tail' if after_drop else 'front_finish'
    self_contacts = []

    if source_seat:
        phase = record['samples'][start]['reference_roll_rad']
        target = trace['targets'][start].copy()
        target[~enabled] = measured[~enabled]
        straight = (m.nu-support+1)*g.l
        lead = g.lead_angle(radius)
        seating = g.targets(phase, radius, ls=straight, th2=lead)[0]

        def seat(t):
            end = min(m.nu, 4+int(t)+1)
            enabled[:end] = True
            desired = target.copy()
            desired[4:end] = seating[4:end]
            target[:] += np.clip(desired-target, -np.radians(18.)*m.opt.timestep, np.radians(18.)*m.opt.timestep)
            return target.copy()

        trial.phase = phase
        trial.phase_run(1, 'source_seat', m.nu-4, seat, stiffness=lambda: enabled.astype(float))

        def wind(t):
            trial.phase = phase+return_roll*t
            desired = g.targets(trial.phase, radius, ls=max(0., straight-.012*t), th2=lead)[0]
            target[:] += np.clip(desired-target, -np.radians(18.)*m.opt.timestep, np.radians(18.)*m.opt.timestep)
            return target.copy()

        trial.phase_run(1, 'source_wind', straight/.012+2., wind, enabled)
        phase += return_roll*(straight/.012+2.)
        trial.phase_run(1, 'source_finish', 3., lambda t:g.targets(phase, radius, th2=lead*(1.-min(1.,t/3.)))[0], enabled)
        trial.phase_run(1, 'recover_roll', 18., lambda t:g.targets(phase+.6*t, radius+(g.o.r_h1-radius)*min(1.,t/3.))[0], enabled)
        save(trial, output, 'locked_gait', record['samples'][start]['time_s'], method='source_seating_then_compensated_wind', return_roll_rad_s=return_roll)
        return

    def control(_):
        nonlocal target, offset, accepted, ready_since, front_ready_at, last_tick, locked, state
        observed = d.qpos[sim.drive.qadr]
        for contact in d.contact:
            a, b = m.geom_bodyid[[contact.geom1, contact.geom2]]-m.body('head').id
            if min(a, b) >= 0 and abs(a-b) > 2 and contact.dist <= 0.:
                if not self_contacts or d.time-self_contacts[-1]['time_s'] >= .05:
                    self_contacts.append(dict(time_s=float(d.time), stage=state, links=[int(a), int(b)],
                                              penetration_m=float(-contact.dist)))
        if d.time-last_tick < .02-1e-8:
            return target
        error = np.max(abs(target[enabled]-observed[enabled]))
        pace = np.clip((np.radians(23.)-error)/np.radians(17.), .1, 1.)
        offset = min(curve.shift_limit_m, offset+.012*pace*.02)
        goal = curve.targets(offset, curve.torsion*offset)
        nominal = curve.helical_targets(offset, curve.torsion*offset)
        goal[:len(locked)] = locked
        target += np.clip(goal-target, -np.radians(18.)*.02, np.radians(18.)*.02)
        target[~enabled] = observed[~enabled]
        ready = np.max(abs(nominal[:accepted]-observed[:accepted])) <= np.radians(3. if state == 'front_finish' else 8.)
        ready_since = (d.time if ready_since is None else ready_since) if ready else None
        if ready_since is not None and d.time-ready_since >= .4:
            if state == 'front_finish':
                front_ready_at = float(d.time)
                locked = target[:accepted].copy()
                state = 'anchored_tail'
            if accepted < m.nu:
                target[accepted] = observed[accepted]
                enabled[accepted] = True
                accepted += 1
                events.append(dict(time_s=float(d.time), joint=accepted))
            ready_since = None
        trial.phase, last_tick = phase, d.time
        return target

    trial.phase_run(1, lambda: state, 150., control, stiffness=lambda: enabled.astype(float))
    save(trial, output, 'locked_gait', record['samples'][start]['time_s'],
         method='phase_compensated_anchored_curve', accepted=accepted, front_ready_s=front_ready_at,
         events=events, locked_count=len(locked), pitch_m=pitch, self_contacts=self_contacts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pitch', type=float, default=.13)
    parser.add_argument('--source', type=Path, default=Path('artifacts/mode_handoff/round6/feed_h5'))
    parser.add_argument('--after-drop', action='store_true')
    parser.add_argument('--source-seat', action='store_true')
    parser.add_argument('--return-roll', type=float, default=0.)
    args = parser.parse_args()
    run(args.output, args.pitch, args.source, args.after_drop, args.source_seat, args.return_roll)
