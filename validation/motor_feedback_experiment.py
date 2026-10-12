"""ONE planned natural-slip run; preserves all actual states on failure.

Run observation first. The closed-loop flag is gated by its saved review,
and uses the same model/initial parameters. No parameter sweep or side force.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from dataclasses import asdict
from datetime import datetime
import gzip
import json
import numpy as np
import mujoco
from snake_control.rolling_arc import ArcParameters
from snake_control.motor_feedback_simulation import MotorFeedbackSimulation
from snake_control.sliding_section_simulation import upward_support
from snake_control.simulation import ROOT, backbone

REPORT = ROOT/'validation/reports/motor_feedback_sliding'


def snapshot(sim):
    # All privileged observations below go into the archive/evaluation only.
    mujoco.mj_forward(sim.model, sim.data)
    return dict(time_s=float(sim.data.time), stage='settle' if sim.data.time < 2. else 'rolling',
                actual_qpos=sim.data.qpos.tolist(), actual_qvel=sim.data.qvel.tolist(),
                target_rad=sim.target.tolist(), enabled=sim.enabled.tolist(),
                actuator_force_nm=sim.data.actuator_force.tolist(),
                support_up_n=upward_support(sim).tolist(),
                body_center_m=backbone(sim.model, sim.data).mean(axis=0).tolist(),
                motor_input=sim.last_feedback, controller=sim.controller.snapshot(),
                events=list(sim.controller.events), external_force_world=sim.data.xfrc_applied.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--closed-loop', action='store_true')
    args = parser.parse_args()
    existing = list(REPORT.glob('observe_*/status.json'))+list(REPORT.glob('closed_*/status.json'))
    if len(existing) >= 2:
        raise RuntimeError('Planned maximum two dynamics attempts already used; do not rerun blindly')
    if args.closed_loop:
        review = json.loads((REPORT/'observation_review.json').read_text(encoding='utf8'))
        if not review['allow_closed_loop']:
            raise RuntimeError('Observation review does not authorize the second planned dynamics case')
    prefix = 'closed' if args.closed_loop else 'observe'
    output = REPORT/(prefix+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    p = ArcParameters(first_joint=5, last_joint=10)
    sim = MotorFeedbackSimulation(p, execute=args.closed_loop)
    count, error = 0, None
    limits = sim.model.jnt_range[sim.model.actuator_trnid[:, 0]]
    stats = dict(physical_steps=0, passive_torque_max_nm=0., active_torque_max_nm=0.,
                 joint_limit_excess_max_rad=0., saturation_joint_steps=0, mask_violation_steps=0,
                 floor_contact_steps=0, maximum_external_force_n=0., finite=True)
    try:
        with gzip.open(output/'actual_trace.jsonl.gz', 'wt', encoding='utf8') as stream:
            stream.write(json.dumps(snapshot(sim), ensure_ascii=False)+'\n')
            count += 1
            for frame in range(3100):
                for _ in range(20):
                    sim.step()
                    q = sim.data.qpos[sim.drive.qadr]
                    tau = sim.data.actuator_force
                    stats['physical_steps'] += 1
                    stats['mask_violation_steps'] += int(sim.enabled.sum() != 6)
                    stats['passive_torque_max_nm'] = max(stats['passive_torque_max_nm'], float(abs(tau[~sim.enabled]).max()))
                    stats['active_torque_max_nm'] = max(stats['active_torque_max_nm'], float(abs(tau[sim.enabled]).max()))
                    stats['saturation_joint_steps'] += int(np.count_nonzero(abs(tau[sim.enabled]) >= p.torque_limit_nm-1e-6))
                    stats['joint_limit_excess_max_rad'] = max(stats['joint_limit_excess_max_rad'],
                        float(np.maximum(np.maximum(limits[:, 0]-q, q-limits[:, 1]), 0.).max()))
                    stats['finite'] &= bool(np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all())
                    floor = sim.model.geom('floor').id
                    stats['floor_contact_steps'] += int(any(floor in (int(c.geom1), int(c.geom2)) and
                        bool({int(c.geom1), int(c.geom2)} & sim.robot_geoms) for c in sim.data.contact))
                    stats['maximum_external_force_n'] = max(stats['maximum_external_force_n'],
                        float(np.linalg.norm(sim.data.xfrc_applied[:, :3], axis=1).max()))
                stream.write(json.dumps(snapshot(sim), ensure_ascii=False)+'\n')
                count += 1
                if frame % 500 == 499:
                    print(f'{prefix}: saved {sim.data.time:.2f}s, window={sim.controller.window}, '
                          f"observer={None if sim.controller.observation is None else sim.controller.observation['state']}", flush=True)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        geom = sim.model.geom('pipe_straight')
        status = dict(status='complete' if error is None else 'interrupted', error=error,
            samples=count, final_time_s=float(sim.data.time), parameters=asdict(p),
            scene='straight', scene_geometry=dict(pipe_radius_m=float(geom.size[0]),
                pipe_position_world=sim.data.geom('pipe_straight').xpos.tolist(),
                pipe_axis_world=sim.data.geom('pipe_straight').xmat.reshape(3, 3)[:, 2].tolist(),
                pipe_half_length_m=float(geom.size[1]), model_gravity=sim.model.opt.gravity.tolist(),
                pipe_friction=geom.friction.tolist()), passive_model=dict(
                damping=sim.model.dof_damping[sim.drive.vadr].tolist(),
                armature=sim.model.dof_armature[sim.drive.vadr].tolist(), calibrated_to_hardware=False),
            statistics_1ms=stats, events=sim.controller.events,
            motor_observer=sim.controller.observer.configuration(), closed_loop=args.closed_loop)
        (output/'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf8')
        print('Actual-state archive:', output, flush=True)


if __name__ == '__main__':
    main()
