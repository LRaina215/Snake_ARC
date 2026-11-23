"""Exercise decoded Ctrl hotkeys on the real-time MuJoCo debug entry and save replay data."""
import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from codes import Rolling_Arc_Debug_Mujoco as debug
from validation.acute_arc_experiment import AcuteReview


def run(output, mode):
    session = debug.DebugSession('vertical_taper_return')
    controller, sim = session.controller, session.sim
    # AcuteReview uses the research scene key to select vertical evaluation.
    sim.scene = 'variable_return'
    review = AcuteReview(sim)
    route = sim.model.numeric('pipe_centerline').data.reshape(-1, 4)
    outlet = float(route[-2, 2])
    output.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(output / 'model.xml'), sim.model)
    # MuJoCo saves absolute asset names on this reconstructed model.
    frames, samples, statuses, presses = [], [], [], []

    def press(key, label):
        command = debug.decode_sim_hotkey(ord(key), True)
        presses.append(dict(command=command, time_s=session.elapsed, description=label))
        session.commands.put(command)

    def record():
        manual = controller.vertical_ring_transition
        target = np.radians(controller.targets)
        mask = session.bus.enabled.copy()
        samples.append(review.sample(manual.core, target, mask))
        statuses.append(manual.status())
        frames.append((session.elapsed, sim.data.qpos.copy(), sim.data.qvel.copy(), target, mask))

    press('8', 'Ctrl+8  接管已有竖直 Helix')
    session.advance()
    record()
    press('H', 'Ctrl+8 → Ctrl+H  逐节退尾')
    formed = branch_time = branch_phase = None
    state_at_switch = None
    for tick in range(18000):
        session.advance()
        manual = controller.vertical_ring_transition
        if manual.stage == 'vertical_ring' and formed is None:
            formed = session.elapsed, manual.core.phase
        # Let the carrier settle for a few seconds, using the identical prefix.
        if formed is not None and branch_time is None and session.elapsed - formed[0] >= 3.:
            branch_time, branch_phase = session.elapsed, manual.core.phase
            state_at_switch = dict(phase_rad=branch_phase, enabled=controller.enabled.copy(),
                                   target_deg=controller.targets.copy(), qpos=sim.data.qpos.tolist(),
                                   qvel=sim.data.qvel.tolist())
            key = '1' if mode == 'basic' else '3'
            text = '普通上滚' if mode == 'basic' else '圆形参考顺应'
            press(key, f'Ctrl+8 → Ctrl+H → Ctrl+{key}  {text}')
        if tick % 5 == 0:
            record()
        if branch_time is not None:
            if mode == 'basic' and abs(manual.core.phase - branch_phase) >= 4. * math.pi:
                break
            if mode == 'compliant' and session.elapsed >= 290.:
                break
    record()
    times, qpos, qvel, target, masks = map(np.asarray, zip(*frames))
    np.savez_compressed(output / 'trace.npz', time=times, qpos=qpos, qvel=qvel, targets=target, masks=masks)
    fields = dict(branch=mode, switch=state_at_switch, branch_time_s=branch_time,
                  formed_time_s=None if formed is None else formed[0],
                  commands=controller.control_commands, outlet_z_m=outlet, route=route.tolist())
    for name, payload in (
            ('run', dict(samples=samples, **fields)), ('controller', statuses), ('operator', presses),
            ('hypothesis', dict(case='straight', scene='variable_return', feature_exit_z_m=outlet,
                                debug_hotkeys=True, options={})),
    ):
        (output / f'{name}.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    if branch_time is None or statuses[-1]['vertical_roll_mode'] != mode:
        raise RuntimeError('No vertical mode handoff in the recorded trajectory')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('mode', choices=('basic', 'compliant'))
    args = parser.parse_args()
    run(args.output, args.mode)
