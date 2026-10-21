"""Choose the initial rolling section from full measured motor angles only."""
import math
import numpy as np
from .section_geometry import forward, local_section
from .rolling_shape_observer import RollingShapeObserver
from .posture_window import measured_half_circle, mechanically_feasible


def choose_posture_window(angles_deg, lengths):
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
    helper=RollingShapeObserver(lengths,None)
    total=float(abs(angles).sum())
    candidates=[]
    for first in range(2,n):
        for last in range(first+2,n):
            window=(first,last)
            if not mechanically_feasible(lengths,window):
                continue
            # Do not latch onto a small folded distal tip instead of the arc.
            captured=float(abs(angles[first-1:last]).sum())
            if captured<90. or captured<.65*total:
                continue
            try:
                section=local_section(np.radians(angles),lengths,*window)
                body=section['points'][1:-1]
                arc_length=float(np.linalg.norm(np.diff(body,axis=0),axis=1).sum())
                gap=float(np.linalg.norm(body[-1]-body[0])/max(arc_length,1e-12))
                turn=math.degrees(section['total_turn'])
                if not 90.<=turn<=210. or gap<.35 or section['fit_rms']>.025:
                    continue
                circle=measured_half_circle(section)
                if not 90.<=circle['coverage_deg']<=240. or circle['circle_rms_m']>.025:
                    continue
                down,quality=helper.passive_down(points,window,section)
                if down is None:
                    continue
                score=(abs(circle['coverage_deg']-180.)/40.+abs(turn-180.)/60.+
                       circle['circle_rms_m']/.01+.03*(last-first+1))
                candidates.append(dict(window=list(window),score=score,
                    turn_deg=turn,coverage_deg=circle['coverage_deg'],gap_fraction=gap,
                    circle_rms_m=circle['circle_rms_m'],end_source=quality['source'],
                    captured_bending_fraction=captured/max(total,1e-12)))
            except (ValueError,np.linalg.LinAlgError):
                continue
    if not candidates:
        raise ValueError('当前角反馈中没有可确认的开放圆弧和下垂参考；未使能。摆好搭管姿态后再按Ctrl+3，或Ctrl+U按配置形成。')
    candidates.sort(key=lambda x:(x['score'],x['window']))
    return dict(chosen=candidates[0]['window'],best=candidates[0],
        alternatives=candidates[1:3],input='all_measured_joint_angles',
        observes_contact=False,observes_world_gravity=False)
