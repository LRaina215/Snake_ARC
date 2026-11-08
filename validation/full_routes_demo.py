"""Record the three requested current-controller hotkey routes, without real hardware."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validation import mode_handoff_experiment as experiment
from validation.gm_arc_handoff_experiment import helix_alignment_ready


ROUTES={
    'ctrl4_ctrl5':dict(window=(7,12),feed=-1),
    'ctrl4_ctrl5_ctrl6':dict(tee=True,feed=-1,sequential=True,complete_hold_s=3.,alignment_timeout_s=60.,confirmation_timeout_s=12.),
    'ctrl6':dict(tee=True,alignment_timeout_s=60.,confirmation_timeout_s=12.),
}
ALIGNMENT_RANGE=(90.,130.)


def visual_alignment(trial,head_alignment,unwind_phase=None,phase_tolerance=.1):
    # Operator/evaluation only: use actual shape and support, rather than the
    # old absolute motor-phase calibration from a different entry trajectory.
    return helix_alignment_ready(trial,ALIGNMENT_RANGE,None,phase_tolerance)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--route',choices=ROUTES,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--grip-steps',type=int,default=0,help='Operator presses Ctrl+O this many times before alignment')
    p.add_argument('--feed-direction',choices=['H','J'],default='H')
    p.add_argument('--head-angle-min',type=float,default=90.)
    p.add_argument('--head-angle-max',type=float,default=130.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    ROUTES[a.route]['grip_steps']=a.grip_steps
    if ROUTES[a.route].get('feed'):ROUTES[a.route]['feed']=-1 if a.feed_direction=='H' else 1
    ALIGNMENT_RANGE=(a.head_angle_min,a.head_angle_max)
    ROUTES[a.route]['operator_head_angle_range']=list(ALIGNMENT_RANGE)
    experiment.CASES[a.route]=ROUTES[a.route]
    experiment.helix_alignment_ready=visual_alignment
    shutil.copyfile(Path(experiment.real.__file__),a.output/'controller_source.py')
    (a.output/'demo_plan.json').write_text(json.dumps(dict(route=a.route,config=ROUTES[a.route],
        method='current real controller; commands through the shared manual-MuJoCo motor bus',
        hardware_run=False,operator='actual-shape/contact-based L confirmations; no absolute phase gate',
        acceptance='actual ARC/Helix shape, ordered handoff, supported right U, passive gravity drop, recovery and continued movement'),indent=2),encoding='utf-8')
    sys.exit(0 if experiment.run(a.route,a.output) else 1)
