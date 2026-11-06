"""Execute the supplied v2 gait with a simulated servo transport."""
import argparse
import ast
import bisect
from copy import copy
from enum import Enum
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import warnings
from zipfile import ZipFile

import mujoco
import numpy as np
from scipy import integrate, optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snake_control.simulation import backbone
from snake_control.gravity_source_gait import SourceCurveGait, SourceShape
from snake_control.tee_simulation import PIPE_HEIGHT, PIPE_RADIUS
from validation.gm_experiment import BRANCH_R, BRANCH_X, LINK_R, angle, top_crossings
from validation.gm_locked_experiment import drop_metrics, recovery_metrics, tail_engaged
from validation.gm_replay_wrap_experiment import save
from validation.gm_shift_cycle_experiment import CycleTrial
from validation.gm_wrap_first_experiment import pipe_contacts


SOURCE_CLASSES = {
    'CurveType', 'CurveSegment', 'LineSegment', 'ArcSegment', 'HelixSegment', 'CurveSegments',
}
SOURCE_ACTIONS = {
    'Gait', 'Control', 'single_step', 'reverse_single_step', 'increase_Psi_0',
    'decrease_Psi_0', 'decrease_theta', 'increase_r', 'decrease_r', 'straighten',
}


def load_gait(archive, link, transport, pitch_mm):
    """Load only reviewed gait definitions. The serial startup never runs."""
    with ZipFile(archive) as z:
        entry, = [f for f in z.infolist() if
                  (f.filename if f.flag_bits & 2048 else f.filename.encode('cp437').decode('gb18030'))
                  .endswith('/重力矩避障-版2.py')]
        tree = ast.parse(z.read(entry).decode('utf-8-sig'), filename=entry.filename)
    names = dict(math=math, np=np, integrate=integrate, bisect=bisect, Enum=Enum, Any=Any,
                 Dict=Dict, List=List, Optional=Optional, Tuple=Tuple, ser=None,
                 FT_Goal_Position=transport.goal, FT_Torsion=transport.torque,
                 time=SimpleNamespace(sleep=transport.advance), input=transport.prompt, print=lambda *args: None)
    selected = [node for node in tree.body if
                (isinstance(node, ast.ClassDef) and node.name in SOURCE_CLASSES) or
                (isinstance(node, ast.FunctionDef) and node.name in SOURCE_ACTIONS)]
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<supplied v2 gait definitions>', 'exec'), names)
    radius, pitch, arc_radius = 94., pitch_mm, 90.
    slope = pitch/(2*math.pi)
    lead = math.atan2(slope, radius)
    torsion = slope/(radius**2+slope**2)
    arc_base = 2*math.pi*arc_radius
    curve = names['CurveSegments']()
    curve.add_curve(names['ArcSegment'](arc_radius, 2*math.pi), psi=math.pi)
    curve.add_curve(names['LineSegment'](0.), psi=0.)
    curve.add_curve(names['ArcSegment'](arc_radius, 0.), psi=-math.pi)
    curve.add_curve(names['ArcSegment'](arc_radius, 0.), psi=math.pi/2)
    curve.add_curve(names['HelixSegment'](radius, pitch, 46*link), psi=0.)
    head = arc_base+26*link
    # Match the existing MuJoCo phase gauge on the initial whole helix.
    phase = -math.pi-torsion*(head+link-arc_base)
    curve.set_initial_psi(phase)
    names.update(curve=curve, NUM_JOINTS=26, l=link, r_h=radius, p_h=pitch, r_c=arc_radius,
                 slope=slope, tau_helix=torsion, alpha_h=lead, theta=0., theta_last=0.,
                 s_h=head, straight_line=0., delta_s=.2*link, Base_variation=0.,
                 Arc_len=.08*arc_radius, Psi_0=phase, Psi_Tai_Direction=math.pi,
                 step=0, time_sleep=.1, vel=1000, Flag_straight_line=False, Flag_Rolling=False,
                 Flag_Helix=False, Flag_theta=False, Flag_Tail_Direction=False, Flag_Space=False)
    return names


class SimServos:
    def __init__(self, trial, return_cap=None):
        self.trial = trial
        self.enabled = trial.full_mask.copy()
        self.q = np.zeros(len(self.enabled))
        self.requested = self.q.copy()
        self.speed = np.zeros_like(self.q)
        self.cycle = 1
        self.stage = 'settle'
        self.release_choice = None
        self.source = None
        self.dispatch = []
        self.drop_report = None
        self.return_cap = return_cap

    def goal(self, _, joint, degrees, speed):
        j = joint-1
        if not self.enabled[j]:
            self.q[j] = self.trial.data.qpos[self.trial.sim.drive.qadr[j]]
        self.enabled[j] = True  # Original v2 transport enables on position write.
        self.requested[j] = math.radians(degrees)
        self.speed[j] = speed*math.pi/2048
        if self.stage == 'source_seat':
            self.dispatch.append(dict(cycle=self.cycle, time_s=float(self.trial.data.time),
                                      joint=joint, degrees=degrees, speed=speed))

    def torque(self, joint, enabled):
        self.enabled[joint-1] = bool(enabled)

    def advance(self, seconds, stop=None):
        def target(_):
            limit = self.speed*self.trial.model.opt.timestep
            self.q[:] += np.clip(self.requested-self.q, -limit, limit)
            return self.q.copy()
        self.trial.phase = self.source['Psi_0']
        limit = self.return_cap if self.stage.startswith('source_') or self.stage == 'recover_roll' else None
        model = self.trial.model
        if limit is not None:
            model.actuator_ctrlrange[:] = [-limit, limit]
            model.actuator_forcerange[:] = [-limit, limit]
        return self.trial.phase_run(self.cycle, self.stage, seconds, target, mask=self.enabled.copy(), stop_when=stop,
                                    torque_limit=limit)

    def action(self, name, stage):
        self.stage = stage
        self.source[name]()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', integrate.IntegrationWarning)
            self.source['Gait']()

    def prompt(self, message):
        if '直线段关节数' in message:
            return str(self.release_choice)
        # This pause corresponds to the original program awaiting the
        # operator's decision after issuing the tail torque-off writes.
        self.stage = 'drop'
        trial = self.trial
        start = len(trial.samples)

        def hangs(_):
            rows = trial.samples[max(start, len(trial.samples)-3):]
            return len(rows) == 3 and all(abs(r['released_tail_y_m']) <= .015 and
                r['released_tail_com_m'][2] <= PIPE_HEIGHT-.10 for r in rows) and drop_metrics(trial.samples[start:])['passed']

        hung = self.advance(5., stop=hangs)
        self.drop_report = drop_metrics(trial.samples[start:])
        if not hung or trial.stopped:
            raise DropIncomplete
        self.stage = 'source_seat'
        return '0'


class DropIncomplete(Exception):
    """The operator did not confirm a forward drop at the release prompt."""


def release_candidate(trial, transport, count):
    """Preview only the passive drop in scratch physics before issuing torque-off."""
    d, m, sim = trial.data, trial.model, trial.sim
    scratch = copy(d)
    mask = np.arange(sim.p.joints) < sim.p.joints-count
    head = m.body('head').id
    tail = slice(head+int(mask.sum())+1, head+sim.p.joints+1)
    mass = m.body_mass[tail]
    q = transport.q.copy()
    velocity_step = transport.speed*m.opt.timestep
    start_points = backbone(m, d)
    initial_angle = angle(start_points[4])
    centered = 0
    for i in range(5000):
        q += np.clip(transport.requested-q, -velocity_step, velocity_step)
        sim.drive.apply_targets(scratch, q, mask)
        mujoco.mj_step(m, scratch)
        if i % 50:
            continue
        com = np.average(scratch.xipos[tail], axis=0, weights=mass)
        p = backbone(m, scratch)
        advance = p[0, 0]-start_points[0, 0]
        rotate = (angle(p[4])-initial_angle+180.)%360.-180.
        centered = centered+1 if abs(com[1]) <= .015 and com[2] < PIPE_HEIGHT-.10 else 0
        if centered >= 3 and advance >= .011 and rotate >= 16.:
            return dict(advance_m=float(advance), rotation_deg=rotate, duration_s=(i+1)*m.opt.timestep)
    return None


def observed_u(trial, count):
    main, _, _ = pipe_contacts(trial.sim)
    p = backbone(trial.model, trial.data)
    n = trial.sim.p.joints
    mask = np.arange(n) < n-count
    y, _ = trial.release_moment(mask)
    return (p[-1, 1] < -PIPE_RADIUS and p[-1, 2] >= PIPE_HEIGHT-.025 and y < 0. and
            any(link <= n-count for link in main) and
            any(link > n-count+1 and p[link, 2] > PIPE_HEIGHT for link in main))


def choose_release(trial, servos, state, reseat_steps):
    """Operate the original roll key to move the front contact before release."""
    for i in range(reseat_steps+1):
        candidates = [count for count in (10, 9, 11, 8) if observed_u(trial, count)]
        if not candidates or trial.stopped:
            return None
        for count in candidates:
            state['preview_attempts'] += 1
            preview = release_candidate(trial, servos, count)
            if preview is not None:
                state['release_preview'] = preview
                return count
        if i < reseat_steps:
            servos.action('increase_Psi_0', 'unwind_line')
    return None


def execute_crossing(trial, servos, count, state, roll_every):
    servos.release_choice = count
    state['released'] = count
    try:
        servos.action('straighten', 'drop')
    except DropIncomplete:
        state['failure'] = 'passive drop not confirmed'
    state['drop'] = servos.drop_report
    if 'failure' in state or trial.stopped:
        return
    src = servos.source
    i = 0
    # Version 2 leaves the number of shift keys to the operator. Continue
    # feeding the same curve until only the source's minimum straight remains.
    while src['straight_line'] >= abs(src['delta_s']) and not trial.stopped:
        servos.action('single_step', 'source_rewrap')
        i += 1
        if roll_every and i % roll_every == 0:
            servos.action('increase_Psi_0', 'source_rewrap')
    state['shift_keys'] = i
    # Let the operator keep rolling until the returned tail takes load.
    for _ in range(100):
        recent = trial.samples[-10:]
        if trial.stopped or (recent and all(tail_engaged(row) for row in recent)):
            break
        servos.action('increase_Psi_0', 'source_roll_open')
    state['end_x_min_m'] = trial.samples[-1]['x_min_m']


def finish_if_clear(trial, servos, final_radius_mm=94.):
    src = servos.source
    clearance = BRANCH_X+BRANCH_R+LINK_R
    if trial.stopped or not all(r['x_min_m'] > clearance for r in trial.samples[-8:]):
        return False
    while src['straight_line'] >= src['delta_s'] and not trial.stopped:
        servos.action('single_step', 'source_finish_tail')
    while src['r_h'] > final_radius_mm+.2 and not trial.stopped:
        servos.action('decrease_r', 'source_tighten')
        servos.action('increase_Psi_0', 'source_tighten')
    for _ in range(125):
        if trial.stopped:
            break
        servos.action('increase_Psi_0', 'recover_roll')
    return True


def checkpoint_run(archive, output, checkpoint, time, roll_every, final_radius_mm, return_cap):
    """Run the original release/return starting from an actually supported U."""
    record = json.loads(checkpoint.read_text(encoding='utf-8'))
    setup = record['summary']['setup']
    trial = CycleTrial(setup=setup)
    servos = SimServos(trial, return_cap)
    pitch = trial.gait.o.pitch
    src = load_gait(archive, trial.gait.l*1000, servos, pitch*1000)
    servos.source = src
    with np.load(checkpoint.with_name(checkpoint.stem+'_trace.npz')) as trace:
        index = int(np.argmin(abs(trace['time']-time)))
        frame = record['samples'][index]
        if frame['stage'] not in ('unwind_line', 'unwind_arc'):
            raise ValueError('Select a recorded U before torque-off')
        trial.data.qpos[:] = trace['qpos'][index]
        trial.data.qvel[:] = trace['qvel'][index]
        servos.requested[:] = servos.q[:] = trace['targets'][index]
    mujoco.mj_forward(trial.model, trial.data)
    servos.speed[:] = 1000*math.pi/2048
    servos.cycle = frame['cycle']
    # Recover the source's manual geometry parameters from the archived issued
    # target. The physical state comes from qpos/qvel, never from the fit.
    gait = SourceCurveGait(26, trial.gait.l, pitch)
    radius = setup['options']['r_h2']
    phase = frame['reference_roll_rad']-gait.torsion(radius)*(.9)
    fit = optimize.least_squares(
        lambda v: gait.targets(SourceShape(v[0], v[1], v[2], v[3]))-servos.requested,
        [radius, .32, -math.pi, phase],
        bounds=([.08, .05, -3.4, phase-5.], [.12, .65, -2.6, phase+5.]))
    r, straight, bend, psi = map(float, fit.x)
    src.update(r_h=r*1000, straight_line=straight*1000, theta=bend, theta_last=bend,
               alpha_h=gait.lead(r), tau_helix=gait.torsion(r)/1000, Psi_0=psi-math.pi)
    curve = src['curve']
    curve.update_segment_params(1, {'length': straight*1000})
    curve.update_segment_params(3, {'radius': 90., 'angle': -bend})
    curve.update_segment_params(4, {'radius': r*1000, 'pitch': pitch*1000, 'length': 46*src['l']})
    curve.set_initial_psi(src['Psi_0'])
    state = dict(cycle=servos.cycle, fit_max_error_deg=math.degrees(float(max(abs(fit.fun)))),
                 source_radius_mm=r*1000, source_straight_mm=straight*1000, source_bend_rad=bend)
    released = False
    servos.stage = 'unwind_line'
    for _ in range(5):
        servos.advance(.02)
        for count in (10, 9, 11, 8):
            if not observed_u(trial, count):
                continue
            preview = release_candidate(trial, servos, count)
            if preview is not None:
                state['release_preview'] = preview
                execute_crossing(trial, servos, count, state, roll_every)
                released = True
                break
        if released:
            break
    if not released:
        state['failure'] = 'no supported right U with verified forward passive drop'
    finish_if_clear(trial, servos, final_radius_mm)
    result = recovery_metrics([row for row in trial.samples if row['stage'] == 'recover_roll'])
    save(trial, output, 'locked_gait', frame['time_s'], source_version='native v2', setup=setup,
         source_checkpoint=str(checkpoint), cycles=[state], recovery=result,
         final_radius_mm=src['r_h'], rewrap_torque_limit_nm=return_cap or trial.sim.p.torque_limit_nm,
         release_events=trial.release_events, sequential_writes=servos.dispatch)


def run(archive, output, align_deg, align_x, pitch_mm, max_cycles, roll_every, reseat_steps):
    setup = dict(scene='up60', joints=26, friction=.5, options=dict(r_h1=.094, r_h2=.0968, pitch=pitch_mm/1000))
    trial = CycleTrial(setup=setup)
    sim, m, d = trial.sim, trial.model, trial.data
    servos = SimServos(trial)
    src = load_gait(archive, trial.gait.l*1000, servos, pitch_mm)
    servos.source = src
    q0 = np.asarray(src['curve'].get_joint_angle(26, src['s_h'], math.pi, 1.))
    sim.place_shape(q0, 26, .2)
    beta = math.radians(align_deg-angle(backbone(m, d)[0]))
    sim.place_shape(q0, 26, .2, beta)
    top = top_crossings(backbone(m, d))[-1]
    sim.place_shape(q0, 26, .2+align_x-top, beta)
    servos.requested[:] = servos.q[:] = q0
    servos.speed[:] = 1000*math.pi/2048
    servos.advance(2.)
    rounds = []
    for cycle in range(1, max_cycles+1):
        servos.cycle = cycle
        state = dict(cycle=cycle, preview_attempts=0)
        rounds.append(state)
        for i in range(35):
            servos.action('increase_Psi_0', 'approach')
            if trial.stopped or (i >= 8 and pipe_contacts(sim)[1]):
                break
        for _ in range(math.ceil((src['theta']+math.pi)/.08)):
            if trial.stopped:
                break
            servos.action('decrease_theta', 'unwind_arc')
        count = None
        used_reseat = False
        for _ in range(33):
            if trial.stopped or count is not None:
                break
            servos.action('reverse_single_step', 'unwind_line')
            before = src['Psi_0']
            count = choose_release(trial, servos, state, 0 if used_reseat else reseat_steps)
            used_reseat = used_reseat or src['Psi_0'] != before
        if count is None:
            state['failure'] = 'no supported right U with verified forward passive drop'
            break
        execute_crossing(trial, servos, count, state, roll_every)
        if 'failure' in state or trial.stopped:
            break
        if finish_if_clear(trial, servos):
            break
    result = recovery_metrics([r for r in trial.samples if r['stage'] == 'recover_roll'])
    save(trial, output, 'locked_gait', None, source_version='native v2', setup=setup,
         align_deg=align_deg, align_x=align_x, cycles=rounds, recovery=result,
         release_events=trial.release_events, sequential_writes=servos.dispatch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--align-deg', type=float, default=130.)
    parser.add_argument('--align-x', type=float, default=.414)
    parser.add_argument('--pitch-mm', type=float, default=140.)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--roll-every', type=int, default=2,
                        help='Insert one original roll key after this many shift keys')
    parser.add_argument('--reseat-steps', type=int, default=0,
                        help='At the first supported U, try this many roll keys before releasing')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-time', type=float, default=62.75)
    parser.add_argument('--final-radius-mm', type=float, default=94.)
    parser.add_argument('--rewrap-cap', type=float, help='Diagnostic torque limit after the passive drop only')
    args = parser.parse_args()
    if min(args.pitch_mm, args.final_radius_mm) <= 0. or args.cycles < 1:
        parser.error('pitch, final radius and cycle count must be positive')
    if min(args.roll_every, args.reseat_steps) < 0:
        parser.error('key counts cannot be negative')
    if args.checkpoint is not None:
        if args.rewrap_cap is not None and not 0. < args.rewrap_cap <= 7.85:
            parser.error('diagnostic torque cap must be in (0, 7.85] N m')
        checkpoint_run(args.archive, args.output, args.checkpoint, args.checkpoint_time, args.roll_every,
                       args.final_radius_mm, args.rewrap_cap)
    else:
        if args.rewrap_cap is not None:
            parser.error('diagnostic torque cap requires --checkpoint')
        run(args.archive, args.output, args.align_deg, args.align_x, args.pitch_mm, args.cycles, args.roll_every,
            args.reseat_steps)
