"""Bounded two-route dynamics comparison, with actual material-motion truth."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import gzip
import json
import math
import numpy as np
import mujoco
from snake_control.active_arc_transfer import ActiveArcTransfer, TransferOptions
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation, backbone
from snake_control.section_geometry import local_section
from snake_control.sliding_section_simulation import upward_support

REPORT = ROOT/'validation/reports/active_transfer'


class TransferSimulation(ArcSimulation):
    def __init__(self, route, rocking=False, options=TransferOptions()):
        p = ArcParameters()
        super().__init__('straight', p, -.5)
        self.lengths = np.full(21, .0645)
        self.controller = ActiveArcTransfer(self.lengths, route, options, rocking=rocking)
        self.target, self.enabled = self.controller.targets.copy(), self.controller.enabled.copy()
        self.next_feedback = self.next_control = 0.
        self.last_feedback = None

    def step(self, phase=None):
        if phase is not None:
            raise ValueError('Transfer controller owns phase')
        time_s = float(self.data.time)
        if time_s+1e-9 >= self.next_control:
            self.next_control += .02
            if time_s+1e-9 >= self.next_feedback:
                self.next_feedback += .2
                self.last_feedback = dict(angles_deg=np.degrees(self.data.qpos[self.drive.qadr]).tolist(),
                                          feedback_time_s=time_s)
            f = self.last_feedback
            self.target, self.enabled = self.controller.advance(f['angles_deg'], f['feedback_time_s'], time_s)
        self.last_phase = self.controller.phase
        self.drive.apply_targets(self.data, self.target, self.enabled)
        mujoco.mj_step(self.model, self.data)


def material_truth(points):
    """World-z crown and hanging lengths; evaluation ONLY, never controller input.

    Shoulders are interpolated crossings of pipe centre height, not an enable
    boundary. The highest connected above-centre span defines the top arc.
    """
    segment_length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.r_[0., np.cumsum(segment_length)]
    h = points[:, 2]-1.2
    peak = int(np.argmax(h))
    head, tail = peak, peak
    while head > 0 and h[head-1] > 0: head -= 1
    while tail < len(h)-1 and h[tail+1] > 0: tail += 1
    valid = bool(head > 0 and tail < len(h)-1 and h[peak] > 0)
    def crossing(i, j):
        fraction = -h[i]/(h[j]-h[i])
        return distance[i]+fraction*(distance[j]-distance[i])
    head_s = crossing(head-1, head) if valid else None
    tail_s = crossing(tail, tail+1) if valid else None
    locations, coordinates, weights = [], [], []
    for j, length in enumerate(segment_length):
        for u in (np.arange(16)+.5)/16:
            locations.append((1-u)*points[j]+u*points[j+1])
            coordinates.append(j+u)
            weights.append(length/16)
    z = np.asarray(locations)[:, 2]
    w = np.maximum(0., z-z.max()+.0080625)*np.asarray(weights)
    return dict(valid=valid, crown_joint=float(w@coordinates/w.sum()),
                head_hanging_m=head_s, tail_hanging_m=None if not valid else float(distance[-1]-tail_s),
                head_shoulder_joint=None if not valid else float(head_s/.0645),
                tail_shoulder_joint=None if not valid else float(tail_s/.0645),
                above_pipe_length_m=None if not valid else float(tail_s-head_s))


def actual_bend(points):
    """Mask-independent same-winding bend interval in the actual pipe projection.

    Threshold endpoints are reported alongside legacy mask-based wrap, never
    silently substituted for it. Reverse passive folds are excluded explicitly.
    """
    xy = points[:, [1, 2]]-np.array([0., 1.2])
    d = np.diff(xy, axis=0)
    turns = np.arctan2(d[:-1, 0]*d[1:, 1]-d[:-1, 1]*d[1:, 0], np.sum(d[:-1]*d[1:], axis=1))
    winding = 1 if turns[7:13].sum() >= 0 else -1
    js = np.arange(1, 21)
    selected = js[(js >= 5) & (js <= 16) & (winding*turns > math.radians(5.))]
    if not len(selected): return dict(valid=False)
    first, last = int(selected[0]), int(selected[-1])
    arc = np.vstack([.5*(xy[first-1]+xy[first]), xy[first:last+1], .5*(xy[last]+xy[last+1])])
    angle = np.unwrap(np.arctan2(arc[:, 1], arc[:, 0]))
    length = np.linalg.norm(np.diff(arc, axis=0), axis=1).sum()
    return dict(valid=True, interval=[first, last], threshold_deg=5.,
                wrap_deg=math.degrees(abs(angle[-1]-angle[0])),
                gap_fraction=float(np.linalg.norm(arc[-1]-arc[0])/length),
                total_turn_deg=math.degrees(float(np.maximum(0., winding*turns[(js >= first)&(js <= last)]).sum())))


def snapshot(sim):
    mujoco.mj_forward(sim.model, sim.data)
    points = backbone(sim.model, sim.data)
    active = np.flatnonzero(sim.enabled)+1
    first, last = int(active[0]), int(active[-1])
    arc = points[first:last+2]
    polar = np.unwrap(np.arctan2(arc[:, 2]-1.2, arc[:, 1]))
    q = sim.data.qpos[sim.drive.qadr].copy()
    shape = local_section(q, sim.lengths, first, last)
    fitting = shape['points'][1:-1]
    gap = np.linalg.norm(fitting[-1]-fitting[0])/np.linalg.norm(np.diff(fitting, axis=0), axis=1).sum()
    support = upward_support(sim)
    evaluation = dict(material=material_truth(points), actual_bend=actual_bend(points),
        legacy_enabled_wrap_deg=math.degrees(abs(polar[-1]-polar[0])),
        enabled_gap_fraction=float(gap), section_turn_deg=math.degrees(shape['total_turn']),
        plane_rms_m=shape['fit_rms'], maximum_curvature_per_m=max([1/s.radius for s in shape['curve'] if s.radius], default=0.),
        support_up_n=float(support.sum()), support_by_link_n=support.tolist(),
        min_z_m=float(points[:, 2].min()), mean_x_m=float(points[:, 0].mean()), mean_y_m=float(points[:, 1].mean()))
    return dict(time_s=float(sim.data.time), actual_qpos=sim.data.qpos.tolist(), actual_qvel=sim.data.qvel.tolist(),
        actual_angles_deg=np.degrees(q).tolist(), target_rad=sim.target.tolist(), enabled=sim.enabled.tolist(),
        actuator_force_nm=sim.data.actuator_force.tolist(), backbone_world_m=points.tolist(),
        motor_input=sim.last_feedback, controller=sim.controller.snapshot(), evaluation=evaluation,
        external_force_world=sim.data.xfrc_applied.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('route', choices=['loose', 'transfer'])
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    if len(list(REPORT.glob('*/status.json'))) >= 3:
        raise RuntimeError('Three planned dynamics runs used; review before further runs')
    if args.repair and not (REPORT/'repair_plan.md').exists():
        raise RuntimeError('Record a targeted repair plan first')
    if not args.repair and list(REPORT.glob(args.route+'_*/status.json')):
        raise RuntimeError('Main case already ran')
    options = replace(TransferOptions(), loose_amplitude_deg=35.) if args.repair and args.route == 'loose' else TransferOptions()
    sim = TransferSimulation(args.route, rocking=args.repair and args.route == 'transfer', options=options)
    folder = REPORT/(args.route+('_repair' if args.repair else '')+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder.mkdir()
    # Freeze exact controller source per dynamics run; later edits cannot erase failures.
    (folder/'controller_source.py').write_text((ROOT/'snake_control/active_arc_transfer.py').read_text(encoding='utf8'), encoding='utf8')
    (folder/'experiment_source.py').write_text(Path(__file__).read_text(encoding='utf8'), encoding='utf8')
    stats = dict(steps=0, passive_torque_max_nm=0., active_torque_max_nm=0.,
                 joint_limit_excess_max_rad=0., saturation_joint_steps=0, floor_contact_steps=0,
                 maximum_enabled_count=0, maximum_shared_target_step_deg=0.)
    limits = sim.model.jnt_range[sim.model.actuator_trnid[:, 0]]
    floor = sim.model.geom('floor').id
    termination, error = 'duration', None
    last_target, last_mask = sim.target.copy(), sim.enabled.copy()
    try:
        with gzip.open(folder/'actual_trace.jsonl.gz', 'wt', encoding='utf8') as stream:
            for frame in range(2750):
                for _ in range(20):
                    sim.step()
                    stats['steps'] += 1
                    q, tau = sim.data.qpos[sim.drive.qadr], sim.data.actuator_force
                    stats['passive_torque_max_nm'] = max(stats['passive_torque_max_nm'],float(np.max(abs(tau[~sim.enabled]),initial=0.)))
                    stats['active_torque_max_nm'] = max(stats['active_torque_max_nm'],float(np.max(abs(tau))))
                    stats['joint_limit_excess_max_rad'] = max(stats['joint_limit_excess_max_rad'],float(np.maximum(np.maximum(limits[:,0]-q,q-limits[:,1]),0.).max()))
                    stats['saturation_joint_steps'] += int(np.count_nonzero(abs(tau)>=sim.p.torque_limit_nm-1e-6))
                    stats['floor_contact_steps'] += int(any(floor in (int(c.geom1),int(c.geom2)) and bool({int(c.geom1),int(c.geom2)}&sim.robot_geoms) for c in sim.data.contact))
                    stats['maximum_enabled_count'] = max(stats['maximum_enabled_count'],int(sim.enabled.sum()))
                row = snapshot(sim)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                shared = last_mask & sim.enabled
                stats['maximum_shared_target_step_deg'] = max(stats['maximum_shared_target_step_deg'],float(np.degrees(abs(sim.target[shared]-last_target[shared])).max()))
                last_target,last_mask = sim.target.copy(),sim.enabled.copy()
                if not np.isfinite(sim.data.qpos).all() or stats['floor_contact_steps'] or sim.controller.fault:
                    termination='floor_or_controller_fault';break
                if sim.controller.completed:
                    termination='task_complete';break
                if frame%250==249:
                    e=row['evaluation'];print(f"{args.route} t={sim.data.time:.1f} stage={sim.controller.stage} crown={e['material']['crown_joint']:.2f} "
                        f"tails={e['material']['head_hanging_m']},{e['material']['tail_hanging_m']} wrap={e['legacy_enabled_wrap_deg']:.1f}",flush=True)
    except Exception as exc:
        error=repr(exc);raise
    finally:
        status=dict(route=args.route,repair=args.repair,scene='straight',parameters=asdict(sim.p),
            final_time_s=float(sim.data.time),termination=termination,error=error,statistics=stats,
            controller=sim.controller.snapshot(),events=sim.controller.events,
            pipe_centerline=sim.model.numeric('pipe_centerline').data.reshape(-1,4).tolist(),
            gravity=sim.model.opt.gravity.tolist(),pipe_friction=sim.model.geom('pipe_straight').friction.tolist(),
            control_period_s=.02,feedback_period_s=.2,physics_period_s=.001,hardware_run=False)
        (folder/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf8')
        print(folder,termination,error,flush=True)


if __name__=='__main__':
    main()
