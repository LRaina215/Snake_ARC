"""Choose the initial rolling section from full measured motor angles only."""
import math
import numpy as np
from .section_geometry import forward, local_section
from .rolling_shape_observer import RollingShapeObserver
from .posture_window import measured_half_circle, mechanically_feasible


def choose_posture_window(angles_deg, lengths, enabled_mask=None, diagnostics=None):
    """Conditional selection from an already draped body, not pipe detection.

    Evaluate contiguous arcs of any mechanically feasible width. Straight
    bodies have no identifiable draped arc and are rejected without enabling.
    """
    angles = np.asarray(angles_deg, float)
    lengths = np.asarray(lengths, float)
    n=len(lengths)-1
    if angles.shape!=(n,) or not np.isfinite(angles).all():
        raise ValueError('自动接管需要全关节有效角反馈')
    points=forward(np.radians(angles),lengths)
    passive_window=None
    if enabled_mask is not None:
        mask=np.asarray(enabled_mask,bool)
        if mask.shape!=(n,):
            raise ValueError('自动接管需要完整使能掩码')
        indices=np.flatnonzero(mask)+1
        if len(indices):
            passive_window=(int(indices[0]),int(indices[-1]))
    helper=RollingShapeObserver(lengths,None)
    total=float(abs(angles).sum())
    candidates=[]
    rejected={}
    nearest=None
    def reject(reason, stage, detail):
        nonlocal nearest
        rejected[reason]=rejected.get(reason,0)+1
        if nearest is None or stage>nearest['stage']:
            nearest=dict(detail,reason=reason,stage=stage)
    for first in range(2,n):
        for last in range(first+2,n):
            window=(first,last)
            if not mechanically_feasible(lengths,window):
                continue
            # Distal passive bends must not inflate a whole-body angle quota.
            # Locate the upper arc with a qualified down reference below.
            captured=float(abs(angles[first-1:last]).sum())
            detail=dict(window=list(window),captured_bending_deg=captured)
            if captured<90.:
                reject('局部弯曲不足90°',0,detail)
                continue
            try:
                section=local_section(np.radians(angles),lengths,*window)
                body=section['points'][1:-1]
                arc_length=float(np.linalg.norm(np.diff(body,axis=0),axis=1).sum())
                gap=float(np.linalg.norm(body[-1]-body[0])/max(arc_length,1e-12))
                turn=math.degrees(section['total_turn'])
                detail.update(turn_deg=turn,gap_fraction=gap,plane_rms_m=section['fit_rms'])
                if not 90.<=turn<=210. or gap<.35 or section['fit_rms']>.025:
                    reject('截面转角/开口/平面拟合不合格',1,detail)
                    continue
                circle=measured_half_circle(section)
                detail.update(coverage_deg=circle['coverage_deg'],circle_rms_m=circle['circle_rms_m'])
                if not 90.<=circle['coverage_deg']<=240. or circle['circle_rms_m']>.025:
                    reject('圆弧覆盖/圆拟合不合格',2,detail)
                    continue
                # During mode4 -> mode3, straightened feeding buffers are
                # still powered. They must not masquerade as passive gravity
                # references simply because the candidate window is narrower.
                down,quality=helper.passive_down(points,passive_window or window,section)
                detail['end_source']=quality['source']
                if down is None:
                    reject('缺少合格失能方向参考',3,detail)
                    continue
                heights=-points @ down
                # The candidate must straddle the highest reconstructed
                # backbone point (ties use the middle of the top plateau).
                # This is conditional on passive-end direction, not world Z.
                top=np.flatnonzero(heights>=heights.max()-1e-9)
                crest=float(np.mean(top))
                detail['conditional_crest_joint']=crest
                if not first<crest<last+1:
                    reject('候选未跨过重建顶部',4,detail)
                    continue
                score=(abs(circle['coverage_deg']-180.)/40.+abs(turn-180.)/60.+
                       circle['circle_rms_m']/.01+.03*(last-first+1))
                candidates.append(dict(window=list(window),score=score,
                    turn_deg=turn,coverage_deg=circle['coverage_deg'],gap_fraction=gap,
                    circle_rms_m=circle['circle_rms_m'],end_source=quality['source'],
                    conditional_crest_joint=crest,
                    captured_bending_fraction=captured/max(total,1e-12)))
            except (ValueError,np.linalg.LinAlgError) as exc:
                reject('截面重建失败：'+str(exc),0,detail)
                continue
    audit=dict(rejection_counts=rejected,nearest_rejected=nearest,eligible_count=len(candidates))
    if diagnostics is not None:
        diagnostics.update(audit)
    if not candidates:
        context='' if nearest is None else f"；最近候选J{nearest['window'][0]}–J{nearest['window'][1]}：{nearest['reason']}，数值={nearest}"
        raise ValueError('自动选窗未执行，原使能状态未改变'+context+'；拒绝统计='+str(rejected))
    candidates.sort(key=lambda x:(x['score'],x['window']))
    return dict(chosen=candidates[0]['window'],best=candidates[0],
        alternatives=candidates[1:3],input='all_measured_joint_angles',
        diagnostics=audit,
        observes_contact=False,observes_world_gravity=False)
