"""110→210 mm continuous sharp elbow; taper occurs across the corner."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from tools.build_fixed_section_scenes import build
from snake_control.simulation import ROOT


def main():
    points = [[-1.1, 0., 0., .055], [-.15, 0., 0., .055]]
    for x in np.linspace(-.15, 0., 7)[1:]:
        points.append([x, 0., 0., .055 + .025 * (x + .15) / .15])
    for y in np.linspace(0., .15, 7)[1:]:
        points.append([0., y, 0., .080 + .025 * y / .15])
    points.extend([[0., y, 0., .105] for y in (.8, 1.6, 2.4)])
    report = build('elbow90_taper110_210', np.asarray(points), [.95, 1.25])
    report.update(bend_heading_deg=90., corner_radius_m=0.,
                  diameter_in_mm=110., diameter_out_mm=210.,
                  taper_length_m=.3, continuous_solid=True)
    (ROOT / 'mujoco_model/sliding_section_scene_manifest.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
