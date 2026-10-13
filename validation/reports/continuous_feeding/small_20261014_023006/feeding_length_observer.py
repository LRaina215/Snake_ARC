"""Encoder-only differential hanging-length proxy, with no pipe/world input.

The cut height is shared by both arms and is derived from the active arc.
It is not a measured pipe centre. For two nearly vertical straight arms a
common cut-height error largely cancels in their length difference. We log
both lengths, the cut height and the gravity-proxy quality rather than
calling these true contact boundaries.
"""
import numpy as np
from .section_geometry import forward, local_section
from .rolling_shape_observer import RollingShapeObserver


class FeedingLengthObserver:
    def __init__(self, lengths):
        self.lengths = np.asarray(lengths, float)
        self.distance = np.r_[0., np.cumsum(self.lengths)]
        self.down_helper = RollingShapeObserver(self.lengths, None)

    def observe(self, angles_deg, window):
        try:
            q = np.radians(angles_deg)
            points = forward(q, self.lengths)
            section = local_section(q, self.lengths, *window)
            down, quality = self.down_helper.passive_down(points, window, section)
            if down is None:
                return dict(valid=False, reason=quality['source'], end_quality=quality)
            heights = -points @ down
            first, last = window
            cut = .5*(heights[first]+heights[last+1])
            peak = int(np.argmax(heights))
            left = right = peak
            while left > 0 and heights[left-1] > cut:
                left -= 1
            while right < len(points)-1 and heights[right+1] > cut:
                right += 1
            if left == 0 or right == len(points)-1 or heights[peak] <= cut:
                return dict(valid=False, reason='no_two_hanging_crossings', end_quality=quality)

            def crossing(i, j):
                amount = (cut-heights[i])/(heights[j]-heights[i])
                return float(self.distance[i]+amount*(self.distance[j]-self.distance[i]))

            head = crossing(left-1, left)
            tail = float(self.distance[-1]-crossing(right, right+1))
            return dict(valid=True, head_hanging_m=head, tail_hanging_m=tail,
                        transfer_coordinate_m=.5*(head-tail),
                        above_cut_length_m=float(self.distance[-1]-head-tail),
                        cut_height_head_frame_m=float(cut), down_head=down.tolist(),
                        end_quality=quality, plane_rms_m=section['fit_rms'],
                        observes_pipe_contact=False, observes_world_position=False)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return dict(valid=False, reason=str(exc))
