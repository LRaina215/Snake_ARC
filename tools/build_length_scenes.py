"""Actual solid 110->210->110mm straight pipe; metadata matches meshes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from tools.build_fixed_section_scenes import build
from snake_control.simulation import ROOT


def main():
    x = np.r_[[-1.2, -.2], np.linspace(-.2, .15, 15)[1:], [.55], np.linspace(.55, .9, 15)[1:], [3.5]]
    radius = np.interp(x, [-1.2, -.2, .15, .55, .9, 3.5], [.055, .055, .105, .105, .055, .055])
    report = build('length_taper_return', np.column_stack([x, np.zeros(len(x)), np.zeros(len(x)), radius]), [1., 2.1])
    report.update(diameters_mm=[110, 210, 110], continuous_solid=True, bend_deg=0.)
    constant = build('length_calibration210', np.array([[-1.2, 0., 0., .105], [2.5, 0., 0., .105]]), [0., 0.])
    constant.update(diameters_mm=[210, 210], bend_deg=0.)
    (ROOT/'mujoco_model/length_scene_manifest.json').write_text(json.dumps([report, constant], indent=2), encoding='utf8')
    print(json.dumps([report, constant]))


if __name__ == '__main__':
    main()
