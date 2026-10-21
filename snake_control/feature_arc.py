"""One full-body measured-arc selector for arming, mode changes and tracking.

Candidates come from measured link geometry, never the old window's opening.
Passive direction and the known feeding focus disambiguate candidates; neither
is required to detect an arc. This estimates a body feature, not pipe contact.
"""
import math
import numpy as np
from .section_geometry import forward,local_section


def feature_arc_candidates(angles_deg,lengths,down=None,focus=None,current_window=None):
    angles=np.asarray(angles_deg,float);lengths=np.asarray(lengths,float)
    n=len(lengths)-1
    if angles.shape!=(n,) or not np.isfinite(angles).all():
        raise ValueError('全关节角反馈必须完整且有限')
    points=forward(np.radians(angles),lengths)
    crest=None
    if down is not None:
        down=np.asarray(down,float);down=down/np.linalg.norm(down)
        heights=-points @ down
        crest=float(np.mean(np.flatnonzero(heights>=heights.max()-1e-8)))
    support=None
    if focus is not None:
        joints=np.arange(1,n+1)
        curved=joints[(joints>=focus[0]) & (joints<=focus[1]) & (abs(angles)>=5.)]
        if len(curved):support=(int(curved[0]),int(curved[-1]))
    candidates=[];rejections={}
    for first in range(2,n-1):
        for last in range(first+2,n):
            if support is not None and not (first<=support[0] and last>=support[1]):continue
            captured=float(abs(angles[first-1:last]).sum())
            if captured<90.:
                rejections['局部弯曲不足90°']=rejections.get('局部弯曲不足90°',0)+1
                continue
            # Prune only obvious multiple-fold spans before expensive fitting.
            if captured>540.:continue
            try:
                # Reuse one FK for every candidate. The absolute polyline
                # turns equal the smooth A/B/C construction's module turns;
                # constructing all those curve objects would waste serial time.
                arc=points[first:last+2]
                center=arc.mean(axis=0)
                _,_,basis=np.linalg.svd(arc-center,full_matrices=False)
                plane_rms=float(np.sqrt(np.mean(((arc-center)@basis[-1])**2)))
                projected=(points[first-1:last+3]-center)@basis[:2].T
                xy=projected[1:-1]
                gap=float(np.linalg.norm(arc[-1]-arc[0])/np.linalg.norm(np.diff(arc,axis=0),axis=1).sum())
                delta=np.diff(projected,axis=0);norm=np.linalg.norm(delta,axis=1)
                if norm.min()<1e-8:continue
                tangents=delta/norm[:,None]
                turns=np.arctan2(tangents[:-1,0]*tangents[1:,1]-tangents[:-1,1]*tangents[1:,0],
                                (tangents[:-1]*tangents[1:]).sum(axis=1))
                turns[abs(turns)<1e-7]=0.  # same straight-module tolerance as continuous_section
                turn=math.degrees(float(abs(turns[:last-first+1]).sum()))
                if not 90.<=turn<=240. or gap<.35 or plane_rms>.025:continue
                fit=np.linalg.lstsq(np.column_stack([2*xy,np.ones(len(xy))]),(xy*xy).sum(axis=1),rcond=None)[0]
                radius=math.sqrt(max(0.,fit[2]+fit[:2]@fit[:2]));radial=xy-fit[:2]
                rms=float(np.sqrt(np.mean((np.linalg.norm(radial,axis=1)-radius)**2)))
                polar=np.unwrap(np.arctan2(radial[:,1],radial[:,0]))
                coverage=math.degrees(abs(polar[-1]-polar[0]))
                if not 90.<=coverage<=240. or rms>.025:continue
                score=abs(turn-180.)/60.+abs(coverage-180.)/60.+rms/.015+.03*(last-first+1)
                alignment=None
                if down is not None:
                    opening=(arc[0]+arc[-1])/2-arc.mean(axis=0)
                    alignment=float(opening@down/max(np.linalg.norm(opening),1e-12))
                    # Direction is ranking evidence; the feature exists even
                    # without it. A reversed old arc must not veto this scan.
                    score+=3.*max(0.,.5-alignment)
                    if not first<crest<last+1:score+=3.
                if current_window is not None:
                    score+=.04*abs(sum((first,last))-sum(current_window))
                candidates.append(dict(window=[first,last],score=score,turn_deg=turn,coverage_deg=coverage,
                    gap_fraction=gap,plane_rms_m=plane_rms,circle_rms_m=rms,
                    opening_alignment=alignment,conditional_crest_joint=crest,
                    captured_bending_fraction=captured/max(float(abs(angles).sum()),1e-12)))
            except (ValueError,np.linalg.LinAlgError):continue
    candidates.sort(key=lambda c:(c['score'],c['window']))
    return dict(chosen=None if not candidates else candidates[0]['window'],
        best=None if not candidates else candidates[0],candidates=candidates[:5],crest_joint=crest,
        diagnostics=dict(eligible_count=len(candidates),rejection_counts=rejections),
        input='whole_measured_link_geometry',uses_down=down is not None,uses_feeding_focus=support is not None,
        observes_pipe_contact=False)
