"""Initial and feeding-mode entry share the full-body feature selector."""
from dataclasses import replace
import numpy as np
from .section_geometry import forward
from .feature_arc import feature_arc_candidates
from .whole_feeding_observer import WholeFeedingObserver,FeedingDirectionOptions


def choose_feeding_arc(angles_deg, lengths, enabled_mask, focus=None):
    """Acquire the established measured bend before releasing straight buffers."""
    mask=np.asarray(enabled_mask,bool)
    active=np.flatnonzero(mask)+1
    if not len(active):raise ValueError('模式4没有使能圆弧')
    if focus is None:focus=(int(active[0]),int(active[-1]))
    selection=feature_arc_candidates(angles_deg,lengths,focus=focus)
    if selection['chosen'] is None:raise ValueError('模式4实测形状中没有可重建的开放圆弧')
    selection['best']['end_source']='feeding_measured_arc_before_passive_release'
    selection['requires_down_before_switch']=False
    return selection


def choose_posture_window(angles_deg,lengths,enabled_mask=None,diagnostics=None):
    """Find measured arcs first; optional passive direction only ranks them."""
    selection=feature_arc_candidates(angles_deg,lengths)
    if diagnostics is not None:diagnostics.update(selection['diagnostics'])
    if selection['chosen'] is None:
        raise ValueError('没有可重建的开放圆弧；'+str(selection['diagnostics']))
    window=tuple(selection['chosen'])
    if enabled_mask is not None:
        active=np.flatnonzero(enabled_mask)+1
        if len(active):window=(int(active[0]),int(active[-1]))
    helper=WholeFeedingObserver(lengths,window,replace(FeedingDirectionOptions(),minimum_samples=1,stability_s=0.))
    down,quality=helper.down_direction(forward(np.radians(angles_deg),lengths),window,angles_deg,0.)
    if down is not None:
        selection=feature_arc_candidates(angles_deg,lengths,down=down)
    selection['best']['end_source']=quality['source']
    selection['requires_down_before_switch']=False
    if diagnostics is not None:diagnostics.update(selection['diagnostics'])
    return selection
