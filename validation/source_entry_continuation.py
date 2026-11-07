"""Resume an exact successful checkpoint and check sustained Helix motion."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.mode_handoff_experiment import ExperimentController, real
from codes.Rolling_Arc_Debug_Mujoco import MujocoMotorBus
from validation.gm_locked_experiment import APPROACH_SETUP
from validation.gm_shift_cycle_experiment import CycleTrial
from validation.gm_replay_wrap_experiment import save
from snake_control.simulation import backbone


def run(folder, output):
    trace=np.load(folder/'locked_gait_trace.npz')
    trial=CycleTrial(setup=dict(APPROACH_SETUP,scene='straight'))
    sim,m,d=trial.sim,trial.model,trial.data
    d.qpos[:],d.qvel[:]=trace['qpos'][-1],trace['qvel'][-1]
    mujoco.mj_forward(m,d)
    c=ExperimentController(MujocoMotorBus(sim))
    c.enabled,c.targets=trace['masks'][-1].tolist(),np.degrees(trace['targets'][-1]).tolist()
    c.bus.prepare_enabled(c.targets,c.enabled)
    c.first,c.last,c.mode=1,18,'hold'
    c.refresh_compliance_feedback()
    # Restore the logical completed-mode-5 state as well as the physical state.
    c.helix_transition=real.GravityCrossing(real.LINK_LENGTHS_M,c.enabled,c.measured_angles(),
        real.GravityMomentOptions(r_h1=real.HELIX_RADIUS_M,r_h2=real.GRAVITY_OPEN_RADIUS_M,
                                  pitch=real.HELIX_PITCH_M),from_helix=True,helix_only=True)
    c.helix_transition.stage='complete'
    # Same hotkey route as the operator: 6 selects alignment, G rolls. No L,
    # obstacle release or world information is sent to the controller.
    c.command('gait_gravity');c.command('run')
    last_control=last_feedback=-1.
    collisions=[]
    def control(_):
        nonlocal last_control,last_feedback
        for contact in d.contact:
            a,b=m.geom_bodyid[[contact.geom1,contact.geom2]]-m.body('head').id
            if min(a,b)>=0 and abs(a-b)>2 and contact.dist<=0:
                collisions.append((float(d.time),int(a),int(b)))
        if d.time-last_control>=.02-1e-8:
            if d.time-last_feedback>=.2-1e-8:
                c.refresh_compliance_feedback();last_feedback=float(d.time)
            c.tick(.02);last_control=float(d.time)
            trial.phase=c.command_phase
        return np.radians(c.bus.targets_deg)
    trial.phase_run(1,'align',10.,control,stiffness=lambda:np.asarray(c.enabled,float))
    turns=[];radii=[]
    for state in trial.states:
        d.qpos[:]=state;mujoco.mj_forward(m,d)
        yz=backbone(m,d)[:,1:]-[0,1.2]
        turns.append(float(np.ptp(np.unwrap(np.arctan2(yz[:,1],yz[:,0])))/(2*np.pi)))
        radii.append(float(np.linalg.norm(yz,axis=1).max()))
    rows=trial.samples
    report=dict(checkpoint=str(folder),duration_s=rows[-1]['time_s']-rows[0]['time_s'],
        min_turns=min(turns),max_radius_m=max(radii),self_contact_events=collisions[:1],
        no_support_frames=sum(not r['main_links'] for r in rows),
        floor_frames=sum(r['ground_force_n']>1 for r in rows),
        head_advance_m=rows[-1]['head_x_m']-rows[0]['head_x_m'],
        max_tracking_deg=max(r['tracking_max_deg'] for r in rows),
        saturated_frames=sum(bool(r['saturated_joints']) for r in rows))
    report['passed']=bool(not collisions and not report['no_support_frames'] and not report['floor_frames']
        and report['min_turns']>=1.7 and report['max_radius_m']<.145 and report['head_advance_m']>.001)
    save(trial,output,'locked_gait',None,**report,binary_enable=True,
         parameters=asdict(sim.p),start_x=.1,commands=[dict(time_s=0,key='Ctrl+6, Ctrl+G')],
         hardware_run=False,controller_status=c.helix_transition.status())
    (output/'continuation_review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.folder,a.output)
