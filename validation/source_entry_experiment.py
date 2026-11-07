"""Bounded ARC-to-source-return trials using the real motor command interface."""
import argparse
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation import mode_handoff_experiment as experiment
from snake_control.gravity_handoff import GravityCrossing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['basic5', 'head_short5', 'tail_short5', 'buffer6'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pitch', type=float, default=.13)
    parser.add_argument('--research-core', action='store_true')
    args = parser.parse_args()
    if args.research_core:
        experiment.real.GravityCrossing = GravityCrossing
    experiment.real.HELIX_PITCH_M = args.pitch
    # A source return has a tangent arc already cut out of the first turn.
    # These are operator/evaluation criteria, never inputs to the controller.
    experiment.MIN_FRONT_ATTACHED_TURNS = .8
    experiment.MAX_FRONT_ATTACHED_RADIUS_M = .120
    experiment.FRONT_SUPPORT_SETTLE_S = .6
    experiment.FRONT_MIN_CONTACT_SPAN = 0
    experiment.INTAKE_STAGES = ('return_seat', 'return_wind', 'return_finish')
    experiment.ASSERT_FRONT_FROZEN = False
    args.output.mkdir(parents=True, exist_ok=True)
    source=(Path(__file__).resolve().parents[1]/'snake_control/gravity_handoff.py' if args.research_core else
            Path(experiment.real.__file__))
    shutil.copyfile(source,args.output/'controller_source.py')
    return 0 if experiment.run(args.case, args.output) else 1


if __name__ == '__main__':
    sys.exit(main())
