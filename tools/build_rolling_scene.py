"""Extend only the straight cylinder and its route metadata for return test."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from tools.build_fixed_section_scenes import extend_exit
from snake_control.simulation import ROOT


if __name__ == '__main__':
    r = extend_exit('straight', 'rolling_straight_long', 2.)
    (ROOT/'mujoco_model/rolling_scene_manifest.json').write_text(json.dumps(r, indent=2), encoding='utf8')
    print(json.dumps(r))
