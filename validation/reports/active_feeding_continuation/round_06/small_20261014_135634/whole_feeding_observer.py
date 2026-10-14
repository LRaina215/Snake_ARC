"""Whole-skeleton encoder observation independent of the enabled arc opening.

Distal torque-off arms still need three straight links. The global top and
an initial shape-derived cut depth replace the current-window height. None
of the inputs describes the actual pipe or its contacts/world orientation.
"""
import math
import numpy as np
from .section_geometry import forward, local_section


class WholeFeedingObserver:
    def __init__(self, lengths, initial_window):
        self.lengths = np.asarray(lengths, float)
        self.distance = np.r_[0., np.cumsum(self.lengths)]
        self.initial_window = initial_window
        self.cut_depth_m = None

    def down_direction(self, points, window):
        candidates = []
        for name, available in (('head', window[0]-1), ('tail', len(self.lengths)-1-window[1])):
            # Fixed distal material span avoids a different chord each time
            # the torque mask changes. Do not include a fourth, proximal
            # link that may still be transitioning out of the upper bend.
            count = min(3, available)
            if count < 3:
                continue
            samples = points[:count+1][::-1] if name=='head' else points[-count-1:]
            chord = samples[-1]-samples[0]
            length = np.linalg.norm(chord)
            straightness = length/np.linalg.norm(np.diff(samples, axis=0), axis=1).sum()
            direction = chord/max(length, 1e-12)
            heights = -points @ direction
            tip = 0 if name=='head' else len(points)-1
            below = float(heights.max()-heights[tip])
            if straightness >= .98 and below >= 2.5*float(np.mean(self.lengths)):
                candidates.append(dict(name=name, count=count, available=available,
                    straightness=float(straightness), tip_below_global_top_m=below,
                    direction=direction.tolist()))
        if not candidates:
            return None, dict(source='no_straight_distal_passive_arm', candidates=[])
        if len(candidates)==2:
            a,b = [np.array(c['direction']) for c in candidates]
            disagreement = math.degrees(math.acos(float(np.clip(a@b,-1.,1.))))
            if disagreement <= 25.:
                down = a+b; down /= np.linalg.norm(down)
                return down, dict(source='global_two_consistent_ends', candidates=candidates,
                    end_disagreement_deg=disagreement, observes_world_gravity=False)
        chosen = max(candidates, key=lambda c:(c['available'],c['straightness']))
        return np.array(chosen['direction']), dict(source='global_single_'+chosen['name'],
            candidates=candidates, observes_world_gravity=False)

    def observe(self, angles_deg, window):
        try:
            q = np.radians(angles_deg)
            points = forward(q, self.lengths)
            down, quality = self.down_direction(points, window)
            if down is None:
                return dict(valid=False, reason=quality['source'], end_quality=quality)
            heights = -points @ down
            if self.cut_depth_m is None:
                first,last = self.initial_window
                self.cut_depth_m = float(heights.max()-.5*(heights[first]+heights[last+1]))
            cut = float(heights.max()-self.cut_depth_m)
            peak = int(np.argmax(heights)); left=right=peak
            while left>0 and heights[left-1]>cut:
                left -= 1
            while right<len(points)-1 and heights[right+1]>cut:
                right += 1
            if left==0 or right==len(points)-1:
                return dict(valid=False, reason='global_cut_has_no_two_hanging_crossings', end_quality=quality)
            def crossing(i,j):
                fraction = (cut-heights[i])/(heights[j]-heights[i])
                return float(self.distance[i]+fraction*(self.distance[j]-self.distance[i]))
            head = crossing(left-1,left)
            tail = float(self.distance[-1]-crossing(right,right+1))
            locations=[]; coordinates=[]; weights=[]
            for j,length in enumerate(self.lengths):
                for u in (np.arange(16)+.5)/16:
                    locations.append((1-u)*heights[j]+u*heights[j+1])
                    coordinates.append(j+u); weights.append(length/16)
            levels=np.array(locations)
            w=np.maximum(0.,levels-levels.max()+np.mean(self.lengths)/8)*weights
            crest=float(w@coordinates/w.sum())
            # Cover the actual upper span plus neighbours. This is geometry,
            # not the contact support span; it cannot prove unloading.
            arc_window=(max(2,left-1),min(len(q)-1,right+1))
            section=local_section(q,self.lengths,*arc_window)
            return dict(valid=True, head_hanging_m=head, tail_hanging_m=tail,
                transfer_coordinate_m=.5*(head-tail), above_cut_length_m=float(self.distance[-1]-head-tail),
                cut_height_head_frame_m=cut, cut_depth_m=self.cut_depth_m, crest_joint=crest,
                actual_arc_window=list(arc_window), down_head=down.tolist(), end_quality=quality,
                plane_rms_m=section['fit_rms'], observes_pipe_contact=False, observes_world_position=False)
        except (ValueError,np.linalg.LinAlgError) as exc:
            return dict(valid=False,reason=str(exc))
