"""Three bounded length cases and at most one documented targeted repair."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import gzip
import json
import numpy as np
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation
from snake_control.continuous_feeding import ContinuousFeeding, ContinuousFeedingOptions
from validation.active_transfer_experiment import TransferSimulation, snapshot
from validation.in_place_experiment import spatial_truth

REPORT = ROOT/'validation/reports/continuous_feeding'
CASES = {'small': -.0645, 'head_large': -.258, 'tail_large': .258}


class ContinuousFeedingSimulation(TransferSimulation):
    def __init__(self, options):
        ArcSimulation.__init__(self, 'straight', ArcParameters(amplitude_deg=options.amplitude_deg), -.5)
        self.lengths = np.full(21, .0645)
        self.controller = ContinuousFeeding(self.lengths, options)
        self.target, self.enabled = self.controller.targets.copy(), self.controller.enabled.copy()
        self.next_control = self.next_feedback = 0.
        self.last_feedback = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    previous = list(REPORT.glob('*/status.json'))
    if len(previous) >= 4:
        raise RuntimeError('Three main cases and one repair budget exhausted')
    if args.repair:
        if list(REPORT.glob('repair_*/status.json')) or not (REPORT/'repair_plan.md').exists():
            raise RuntimeError('Only one repair, documented before its run')
    elif list(REPORT.glob(args.case+'_*/status.json')):
        raise RuntimeError('This main case already ran; do not overwrite evidence')
    options = replace(ContinuousFeedingOptions(), requested_transfer_m=CASES[args.case],
                      maximum_feed_s=60. if args.case=='small' else 100.)
    sim = ContinuousFeedingSimulation(options)
    label = ('repair_' if args.repair else '')+args.case
    folder = REPORT/(label+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder.mkdir()
    for name in ('snake_control/continuous_feeding.py', 'snake_control/feeding_length_observer.py',
                 'validation/continuous_feeding_experiment.py'):
        (folder/Path(name).name).write_text((ROOT/name).read_text(encoding='utf8'), encoding='utf8')
    statistics = dict(steps=0, passive_torque_max_nm=0., active_torque_max_nm=0.,
        joint_limit_excess_max_rad=0., saturation_joint_steps=0, floor_contact_steps=0,
        maximum_enabled_count=0, minimum_enabled_count=20, maximum_shared_target_step_deg=0.)
    limits = sim.model.jnt_range[sim.model.actuator_trnid[:, 0]]
    floor = sim.model.geom('floor').id
    baseline = None
    previous_target, previous_mask = sim.target.copy(), sim.enabled.copy()
    termination, error = 'duration', None
    try:
        with gzip.open(folder/'actual_trace.jsonl.gz', 'wt', encoding='utf8') as stream:
            duration = options.preparation_s+2*options.maximum_feed_s+2*options.review_s+2.
            for frame in range(int(duration/.02)):
                for _ in range(20):
                    sim.step()
                    statistics['steps'] += 1
                    torque = sim.data.actuator_force
                    q = sim.data.qpos[sim.drive.qadr]
                    statistics['passive_torque_max_nm'] = max(statistics['passive_torque_max_nm'], float(np.max(abs(torque[~sim.enabled]), initial=0.)))
                    statistics['active_torque_max_nm'] = max(statistics['active_torque_max_nm'], float(np.max(abs(torque))))
                    excess = np.maximum(np.maximum(limits[:, 0]-q, q-limits[:, 1]), 0.)
                    statistics['joint_limit_excess_max_rad'] = max(statistics['joint_limit_excess_max_rad'], float(excess.max()))
                    statistics['saturation_joint_steps'] += int(np.count_nonzero(abs(torque)>=sim.p.torque_limit_nm-1e-6))
                    statistics['floor_contact_steps'] += int(any(floor in (int(c.geom1), int(c.geom2)) and bool({int(c.geom1), int(c.geom2)}&sim.robot_geoms) for c in sim.data.contact))
                    count = int(sim.enabled.sum())
                    statistics['minimum_enabled_count'] = min(statistics['minimum_enabled_count'], count)
                    statistics['maximum_enabled_count'] = max(statistics['maximum_enabled_count'], count)
                row = snapshot(sim)
                spatial = spatial_truth(sim, row)
                if row['time_s'] >= 6. and baseline is None:
                    baseline = spatial.copy()
                if baseline:
                    for key in ('crown_world_m', 'support_point_world_m', 'mean_body_world_m'):
                        if spatial[key] is not None and baseline[key] is not None:
                            spatial[key+'_delta'] = (np.array(spatial[key])-baseline[key]).tolist()
                row['evaluation']['spatial'] = spatial
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                shared = sim.enabled & previous_mask
                statistics['maximum_shared_target_step_deg'] = max(statistics['maximum_shared_target_step_deg'], float(np.degrees(abs(sim.target[shared]-previous_target[shared])).max()))
                previous_target, previous_mask = sim.target.copy(), sim.enabled.copy()
                if statistics['floor_contact_steps'] or sim.controller.fault or not np.isfinite(sim.data.qpos).all():
                    termination = 'floor_or_controller_fault'
                    break
                if sim.controller.completed:
                    termination = 'task_complete'
                    break
                if frame%250 == 249:
                    m = row['evaluation']['material']
                    true = .5*(m['head_hanging_m']-m['tail_hanging_m']) if m['valid'] else np.nan
                    observed = sim.controller.observation
                    estimated = observed.get('transfer_coordinate_m', np.nan) if observed else np.nan
                    print(f"t={row['time_s']:.1f}s {sim.controller.stage} p={sim.controller.position:.3f} "
                          f"b={sim.controller.bias:.3f} true/estimated={true*1000:.1f}/{estimated*1000:.1f}mm "
                          f"velocity={sim.controller.velocity_m_s*1000:.1f}mm/s active={int(sim.enabled.sum())}", flush=True)
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        status = dict(scene='straight', route='continuous_feeding', case=args.case, repair=args.repair,
            parameters=asdict(sim.p), options=asdict(options), final_time_s=float(sim.data.time),
            termination=termination, error=error, statistics=statistics,
            controller=sim.controller.snapshot(), events=sim.controller.events, spatial_baseline=baseline,
            pipe_centerline=sim.model.numeric('pipe_centerline').data.reshape(-1, 4).tolist(),
            pipe_friction=sim.model.geom('pipe_straight').friction.tolist(), gravity=sim.model.opt.gravity.tolist(),
            hardware_run=False, physics_period_s=.001, control_period_s=.02, feedback_period_s=.2)
        (folder/'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf8')
        print(folder, termination, sim.controller.fault, flush=True)


if __name__ == '__main__':
    main()
