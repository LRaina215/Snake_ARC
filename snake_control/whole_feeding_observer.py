"""Whole-skeleton encoder observation independent of the enabled arc opening.

Straight segments are selected inside both torque-off arms. The global top and
an initial shape-derived cut depth replace the current-window height. None
of the inputs describes the actual pipe or its contacts/world orientation.
"""
import math
from dataclasses import dataclass, asdict
import numpy as np
from .section_geometry import forward, local_section


@dataclass(frozen=True)
class FeedingDirectionOptions:
    """Angle-only quality thresholds; stability is intrinsic, not world motion."""
    minimum_links: int = 3
    maximum_links: int = 5
    minimum_straightness: float = .98
    stability_s: float = .4
    minimum_samples: int = 3
    maximum_shape_change_deg: float = 3.
    maximum_sample_gap_s: float = .6
    minimum_tip_depth_links: float = 2.5
    minimum_segment_depth_links: float = .5
    maximum_end_disagreement_deg: float = 25.


class WholeFeedingObserver:
    def __init__(self, lengths, initial_window, options=FeedingDirectionOptions()):
        self.lengths = np.asarray(lengths, float)
        self.distance = np.r_[0., np.cumsum(self.lengths)]
        self.initial_window = initial_window
        self.cut_depth_m = None
        self.options = options
        self._histories = {}
        self._selected = {}
        self._single_side = None
        self._last_feedback = None
        self._last_angles = None
        self._reference = None
        self._reference_epoch = 0
        self._coordinate_offset_m = 0.
        self._last_valid_time = None
        self._current_candidates = {}

    def down_direction(self, points, window, angles_deg, feedback_time_s):
        """Search wholly passive spans; retain an eligible span to avoid chatter.

        History compares internal joint angles of each material span. Comparing
        directions in the head frame across time would mistake motion of the
        head frame for a changing gravity vector. Neither test observes gravity.
        """
        opt = self.options
        q = np.asarray(angles_deg, float)
        now = float(feedback_time_s)
        mean_length = float(np.mean(self.lengths))
        ends, selected, live_keys = {}, [], set()
        self._current_candidates = {}
        for name, available in (('head', window[0]-1), ('tail', len(self.lengths)-1-window[1])):
            lo, hi = (0, available) if name=='head' else (len(self.lengths)-available, len(self.lengths))
            tested, eligible, rejected = [], [], {}
            for count in range(opt.minimum_links, min(opt.maximum_links, available)+1):
                for start in range(lo, hi-count+1):
                    stop = start+count  # links [start, stop), points [start, stop]
                    key = (name, start, stop)
                    live_keys.add(key)
                    chord = points[stop]-points[start]
                    if name=='head':
                        chord = -chord  # always from the arc towards the free end
                    length = float(np.linalg.norm(chord))
                    straightness = length/float(self.lengths[start:stop].sum())
                    direction = chord/max(length, 1e-12)
                    heights = -points @ direction
                    tip = 0 if name=='head' else len(points)-1
                    below = float(heights.max()-heights[tip])
                    segment_depth = float(heights.max()-heights[start:stop+1].max())
                    reasons = []
                    if straightness < opt.minimum_straightness:
                        reasons.append('not_straight')
                    if below < opt.minimum_tip_depth_links*mean_length:
                        reasons.append('tip_not_below_arc')
                    if segment_depth < opt.minimum_segment_depth_links*mean_length:
                        reasons.append('segment_near_crest')
                    history = self._histories.get(key, [])
                    if reasons or (history and now-history[-1][0] > opt.maximum_sample_gap_s):
                        history = []
                    # A repeated encoder timestamp never counts as a new sample.
                    if not reasons and (not history or now > history[-1][0]):
                        history.append((now, q[start:stop-1].copy()))
                    # Keep BOTH the required time span and sample count. The
                    # previous time-only pruning could repeatedly discard the
                    # third good sample at real serial scan intervals (~.22s).
                    while len(history)>opt.minimum_samples and history[1][0] <= now-opt.stability_s+1e-9:
                        history.pop(0)
                    self._histories[key] = history
                    span = now-history[0][0] if history else 0.
                    change = float(np.ptp(np.array([h[1] for h in history]), axis=0).max()) if history else None
                    if not reasons:
                        if len(history)<opt.minimum_samples or span < opt.stability_s-1e-9:
                            reasons.append('collecting_stability')
                        if change is not None and change > opt.maximum_shape_change_deg:
                            reasons.append('shape_unstable')
                    candidate = dict(name=name, links=[start+1, stop],
                        internal_joints=[start+1, stop-1], count=count, available=available,
                        straightness=straightness, tip_below_global_top_m=below,
                        segment_below_global_top_m=segment_depth,
                        stable_span_s=span, samples=len(history), shape_change_deg=change,
                        direction=direction.tolist(), reasons=reasons)
                    tested.append(candidate)
                    self._current_candidates[key] = candidate
                    if not reasons:
                        eligible.append(candidate)
                    for reason in reasons:
                        rejected[reason] = rejected.get(reason, 0)+1
            def rank(c):
                return (c['count'], c['straightness'], -(c['shape_change_deg'] or 0.))
            prior = self._selected.get(name)
            chosen = next((c for c in eligible if c['links']==prior), None)
            if chosen is None and eligible:
                chosen = max(eligible, key=rank)
            if chosen is not None:
                self._selected[name] = chosen['links']
            if chosen is not None:
                selected.append(chosen)
            # Display one actionable candidate and rejection totals, not dozens
            # of nearly identical overlapping spans in every 20ms log row.
            best = chosen or (max(tested, key=lambda c: (
                not any(r in c['reasons'] for r in ('not_straight','tip_not_below_arc','segment_near_crest')),
                c['straightness'], c['count'])) if tested else None)
            ends[name] = dict(available=available, tested_count=len(tested),
                stable_count=len(eligible), selected=chosen, best=best,
                rejection_counts=rejected,
                reason='selected' if chosen else 'insufficient_passive_links' if not tested else 'no_eligible_segment')
        self._histories = {k:v for k,v in self._histories.items() if k in live_keys}
        quality = dict(source='no_stable_passive_segment', candidates=selected, ends=ends,
            thresholds=asdict(opt), observes_world_gravity=False,
            stability_basis='internal_joint_angles_in_material_segment')
        if not selected:
            self._single_side = None
            return None, quality
        if len(selected)==2:
            a,b = [np.array(c['direction']) for c in selected]
            disagreement = math.degrees(math.acos(float(np.clip(a@b,-1.,1.))))
            quality['end_disagreement_deg'] = disagreement
            if disagreement <= opt.maximum_end_disagreement_deg:
                weights = [c['count']*c['straightness'] for c in selected]
                down = weights[0]*a+weights[1]*b
                down /= np.linalg.norm(down)
                quality.update(source='global_two_consistent_segments', fusion_weights=weights)
                self._single_side = None
                return down, quality
            quality['fusion_rejected_reason'] = 'end_directions_disagree'
        chosen = next((c for c in selected if c['name']==self._single_side), None)
        if chosen is None:
            chosen = max(selected, key=rank)
        self._single_side = chosen['name']
        quality.update(source='global_single_'+chosen['name'], used_sides=[chosen['name']])
        return np.array(chosen['direction']), quality

    def reference_recipe(self, quality):
        candidates = quality['candidates']
        if quality['source']=='global_two_consistent_segments':
            weights = quality['fusion_weights']
        else:
            candidates = [c for c in candidates if c['name'] in quality['used_sides']]
            weights = [1.]
        return [(c['name'], tuple(c['links']), w) for c,w in zip(candidates,weights)]

    def previous_reference_now(self):
        """Reconstruct the old reference in the CURRENT head frame.

        Only still-passive, geometrically valid material spans are usable.
        Never freeze a head-frame vector or use a now-enabled arm as gravity.
        All alternative passive spans have been warming up in parallel.
        """
        down = np.zeros(3)
        for name,links,weight in self._reference:
            candidate = self._current_candidates.get((name,links[0]-1,links[1]))
            if candidate is None or any(r!='collecting_stability' for r in candidate['reasons']):
                return None
            down += weight*np.array(candidate['direction'])
        return down/np.linalg.norm(down)

    def cut_geometry(self, points, down):
        heights = -points @ down
        cut = float(heights.max()-self.cut_depth_m)
        peak = int(np.argmax(heights)); left=right=peak
        while left>0 and heights[left-1]>cut:
            left -= 1
        while right<len(points)-1 and heights[right+1]>cut:
            right += 1
        if left==0 or right==len(points)-1:
            raise ValueError('global_cut_has_no_two_hanging_crossings')
        def crossing(i,j):
            fraction = (cut-heights[i])/(heights[j]-heights[i])
            return float(self.distance[i]+fraction*(self.distance[j]-self.distance[i]))
        head = crossing(left-1,left)
        tail = float(self.distance[-1]-crossing(right,right+1))
        return heights,cut,left,right,head,tail

    def observe(self, angles_deg, window, feedback_time_s):
        try:
            t = float(feedback_time_s)
            angles = np.asarray(angles_deg, float)
            if not math.isfinite(t) or (self._last_feedback is not None and t < self._last_feedback):
                raise ValueError('nonmonotonic_or_invalid_feedback_time')
            if self._last_feedback == t and not np.array_equal(angles, self._last_angles):
                raise ValueError('changed_angles_with_same_feedback_time')
            self._last_feedback, self._last_angles = t, angles.copy()
            q = np.radians(angles_deg)
            points = forward(q, self.lengths)
            down, quality = self.down_direction(points, window, angles, t)
            if down is None:
                return dict(valid=False, reason=quality['source'], end_quality=quality)
            heights = -points @ down
            if self.cut_depth_m is None:
                first,last = self.initial_window
                self.cut_depth_m = float(heights.max()-.5*(heights[first]+heights[last+1]))
            heights,cut,left,right,head,tail = self.cut_geometry(points,down)
            raw_coordinate = .5*(head-tail)
            recipe = self.reference_recipe(quality)
            changed = self._reference is not None and [(a,b) for a,b,_ in recipe]!=[(a,b) for a,b,_ in self._reference]
            expired = self._last_valid_time is not None and t-self._last_valid_time>self.options.maximum_sample_gap_s
            event, correction = 'retained', 0.
            next_offset,next_epoch=self._coordinate_offset_m,self._reference_epoch
            if changed and not expired:
                old_down = self.previous_reference_now()
                compatible = old_down is not None and float(old_down@down)>=math.cos(math.radians(self.options.maximum_end_disagreement_deg))
                if compatible:
                    try:
                        *_,old_head,old_tail = self.cut_geometry(points,old_down)
                        correction = .5*(old_head-old_tail)-raw_coordinate
                        next_offset += correction
                        event = 'overlap_handover'
                    except ValueError:
                        compatible = False
                if not compatible:
                    event = 'reacquire_no_overlap'
            elif expired:
                event = 'reacquire_after_gap'
            if event.startswith('reacquire'):
                next_epoch += 1
                next_offset = 0.
            quality.update(reference_epoch=next_epoch, reference_event=event,
                handover_correction_m=correction, coordinate_offset_m=next_offset,
                reference_recipe=recipe)
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
            # Commit the reference only after the whole measurement succeeds.
            # Failed section reconstruction is not a fresh valid datum.
            self._reference, self._last_valid_time = recipe, t
            self._coordinate_offset_m,self._reference_epoch=next_offset,next_epoch
            return dict(valid=True, head_hanging_m=head, tail_hanging_m=tail,
                transfer_coordinate_m=raw_coordinate+self._coordinate_offset_m,
                raw_transfer_coordinate_m=raw_coordinate,
                above_cut_length_m=float(self.distance[-1]-head-tail),
                cut_height_head_frame_m=cut, cut_depth_m=self.cut_depth_m, crest_joint=crest,
                actual_arc_window=list(arc_window), down_head=down.tolist(), end_quality=quality,
                plane_rms_m=section['fit_rms'], observes_pipe_contact=False, observes_world_position=False)
        except (ValueError,np.linalg.LinAlgError) as exc:
            return dict(valid=False,reason=str(exc))
