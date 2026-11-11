"""Adapt the neighboring repository's 2023 TRO control to the matched robot."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

from snake_control.section_geometry import continuous_section, forward, paper_axis, projection_basis


REFERENCE = Path(__file__).resolve().parents[2] / 'Snake_Robot_Gait/snake_control/pole_compliance.py'


def load_reference():
    spec = importlib.util.spec_from_file_location('reference_tro_pole_compliance', REFERENCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TroHelixBaseline:
    def __init__(self, lengths, gain=(.04, .08), pitch_angle=.25):
        self.lengths = np.asarray(lengths)
        self.module = load_reference()
        self.controller = self.module.ComplianceController(
            self.module.ComplianceParameters(gain, pitch_angle), self.lengths * 1000.)
        self.config = dict(head_length_mm=self.lengths[0] * 1000., joint_count=len(lengths) - 1,
            joints=[dict(joint_index=i + 1, axis='Yaw' if i % 2 == 0 else 'Pitch',
                         link_length_mm=length * 1000., minimum_deg=-90., maximum_deg=90.)
                    for i, length in enumerate(self.lengths[1:])])
        self.axis = None
        self.frame_time_s = None
        self.sequence = 0
        self.geometry = None

    def step(self, measured, roll, previous, feedback_time_s, now_s):
        if self.frame_time_s != feedback_time_s:
            self.frame_time_s = feedback_time_s
            self.sequence += 1
            points = forward(measured, self.lengths)
            axis = paper_axis(points)
            if self.axis is not None and axis['direction'] @ self.axis['direction'] < 0.:
                axis['direction'] *= -1.
            basis = projection_basis(axis['direction'], None if self.axis is None else self.axis['horizontal'])
            axis.update(horizontal=basis[:, 0], vertical=basis[:, 1])
            self.axis = axis
            xy = (points - axis['point']) @ basis[:, :2]
            count = len(measured)
            curve = continuous_section(xy, original=True)
            segments = [dict(kind='arc' if s.radius else 'line', length_mm=s.length * 1000.,
                             radius_mm=s.radius * 1000., turn_rad=s.turn,
                             joint_index=s.joint if 1 <= s.joint <= count else None,
                             start_mm=s.start * 1000., tangent=s.tangent) for s in curve]
            self.geometry = dict(config=self.config, curve=dict(valid=True, segments=segments),
                                 points_mm=points * 1000., feedback_usable=axis['confidence'] > 0.,
                                 age_ms=0., sample_unix_ms=feedback_time_s * 1000.,
                                 run_id='matched_simulation', config_id='robot', sequence=self.sequence,
                                 projection_axis=dict(point_mm=axis['point'] * 1000., direction=axis['direction'],
                                                      horizontal=axis['horizontal'], vertical=axis['vertical']))
        # The original integrator uses -sin/-cos rather than cos/sin. Their
        # rolling phase clocks have opposite signs after alignment.
        return self.controller.compute(self.geometry, -roll, previous, now_ms=now_s * 1000.)
