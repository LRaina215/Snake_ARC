"""Encoder/command-only vertical Helix -> bent-axis wrap -> horizontal ARC.

All targets are integrals of one stitched backbone. Scene state never enters
this module. Bend/horizontal transitions are explicit operator events.
"""
from dataclasses import dataclass,asdict
import math
import numpy as np
from .curve_gait import CurveGait,helix,arc,line
from .helix_arc_transition import HelixArcTransition,HelixArcOptions,HelixArcManual,ArcCurveReference
from .bent_helix_curve import BentHelixCurve


@dataclass(frozen=True)
class VerticalArcOptions:
    phase_rate_rad_s: float=-.5
    retained_turns: float=1.5
    exit_angle_rad: float=math.radians(40.)
    exit_transition_s: float=2.
    unwind_m_s: float=.016
    target_rate_deg_s: float=45.
    settle_s: float=2.
    baseline_s: float=8.
    local_roll_s: float=15.
    finish_after_pilot: bool=True
    bend_rate_m_s: float=.012
    bend_radius_m: float=.3


class VerticalArcTransition(HelixArcTransition):
    def __init__(self,measured,lengths,options=VerticalArcOptions()):
        super().__init__(measured,lengths,HelixArcOptions(target_rate_deg_s=options.target_rate_deg_s))
        self.pitch=self.fit['pitch']
        self.vo=options;self.stage='settle';self.elapsed=0.;self.time=0.
        self.straight=0.;self.helix_length=self.L;self.events=[];self.last_segments=[]
        self.bent= None;self.head_u=None;self.field_updated=-1e9;self.horizontal=None;self.arc_reference=None

    def enter(self,stage):
        self.events.append(dict(time_s=self.time,stage=stage));self.stage=stage;self.elapsed=0.

    def command_curve(self,segments,phase,dt,measured,mask=None):
        previous=self.enabled.copy();self.measured=measured
        self.set_curve(segments,phase,dt,mask)
        if np.any(previous!=self.enabled):self.events.append(dict(time_s=self.time,stage=self.stage,
            enabled=(np.flatnonzero(self.enabled)+1).tolist(),released=(np.flatnonzero(previous&~self.enabled)+1).tolist()))
        self.last_segments=segments

    def begin_bend(self,nominal_head_u=-.08):
        """Explicit operator event; nominal offset is configured, not a root measurement."""
        self.head_u=nominal_head_u;self.bent=BentHelixCurve(self.radius,self.pitch,self.vo.bend_radius_m)
        self.enter('bend_transfer')

    def bend_targets(self,dt):
        if self.time-self.field_updated>=.1-1e-8:
            bounds,components,end_phase=self.bent.field(self.head_u,self.helix_length)
            transition=self.radius*self.vo.exit_angle_rad
            self.field_bounds=np.r_[bounds,self.helix_length+transition,self.L]
            self.field_components=np.vstack([components,[math.cos(end_phase+math.pi/2)/self.radius,
                math.sin(end_phase+math.pi/2)/self.radius],[0.,0.]])
            self.field_updated=self.time
        desired=self.curve.sampled_targets(self.field_bounds,self.field_components,self.phase)
        mask=(self.curve.s[:-2]<self.helix_length+self.radius*self.vo.exit_angle_rad-1e-6)
        self.apply_reference(desired,dt,mask)

    def confirm_horizontal(self,measured):
        """Human support confirmation starts the existing horizontal converter."""
        count=max(5,int(np.searchsorted(self.curve.s[2:],self.helix_length)-1))
        manual=HelixArcManual(np.degrees(measured),self.lengths,
            end_reset_options=dict(rate_deg_s=60.,hold_s=.25,max_s=2.),fit_count=count)
        c=manual.core;c.targets=self.targets.copy();c.enabled=self.enabled.copy()
        c.helix_length=self.helix_length;c.straight=self.straight;c.lead=self.vo.exit_angle_rad
        c.enter('reverse_unwrap');manual.running=True;self.horizontal=manual
        self.enter('horizontal_reverse_unwrap')

    def reverse_curve(self,angle):
        transition=self.radius*angle
        self.helix_length=self.L-self.straight-transition
        from .curve_gait import Segment
        return [Segment(self.helix_length,self.kappa,self.tau),arc(transition,1/self.radius,math.pi/2),line(self.straight)]

    def step(self,dt,measured,feedback_time=None):
        self.time+=dt;self.elapsed+=dt;self.measured=np.asarray(measured)
        if self.stage=='arc_roll':
            self.phase-=.6*dt
            desired=self.arc_reference.targets(self.phase,60.)
            self.apply_reference(desired,dt,self.enabled)
            return self.targets.copy(),self.enabled.copy()
        if self.horizontal is not None:
            self.horizontal.advance(dt,self.time,self.time if feedback_time is None else feedback_time,np.degrees(measured))
            self.targets=self.horizontal.targets.copy();self.enabled=self.horizontal.enabled.copy();self.phase=self.horizontal.phase
            self.stage='horizontal_'+self.horizontal.stage
            if self.horizontal.completed:
                c=self.horizontal.core;self.arc_reference=ArcCurveReference(c.lengths,c.head,c.arc_length,c.radius,60.,c.center_feeding)
                self.enter('arc_roll')
            return self.targets.copy(),self.enabled.copy()
        if self.stage=='bend_transfer':
            self.phase+=self.vo.phase_rate_rad_s*dt
            self.head_u=min(self.vo.bend_radius_m*np.pi/2+.35,self.head_u+self.vo.bend_rate_m_s*dt)
            self.bend_targets(dt)
            return self.targets.copy(),self.enabled.copy()
        segments=self.reverse_curve(0.)
        if self.stage=='settle':
            if self.elapsed>=self.vo.settle_s:self.enter('baseline_climb')
        elif self.stage=='baseline_climb':
            self.phase+=self.vo.phase_rate_rad_s*dt
            if self.elapsed>=self.vo.baseline_s:self.enter('exit_transition')
        elif self.stage=='exit_transition':
            self.phase+=self.vo.phase_rate_rad_s*dt
            w=np.clip(self.elapsed/self.vo.exit_transition_s,0.,1.);w=w*w*(3-2*w)
            segments=self.reverse_curve(self.vo.exit_angle_rad*w)
            if w>=1:self.enter('reverse_unwrap')
        elif self.stage=='reverse_unwrap':
            self.phase+=self.vo.phase_rate_rad_s*dt
            kept=self.vo.retained_turns*math.hypot(2*math.pi*self.radius,self.pitch)
            max_straight=self.L-kept-self.radius*self.vo.exit_angle_rad
            self.straight=min(max_straight,self.straight+self.vo.unwind_m_s*dt)
            segments=self.reverse_curve(self.vo.exit_angle_rad)
            if self.straight>=max_straight-1e-9:self.enter('local_climb')
        elif self.stage in ('local_climb','complete','vertical_approach'):
            self.phase+=self.vo.phase_rate_rad_s*dt;segments=self.reverse_curve(self.vo.exit_angle_rad)
            if self.stage=='local_climb' and self.elapsed>=self.vo.local_roll_s:
                self.enter('complete' if self.vo.finish_after_pilot else 'vertical_approach')
        self.command_curve(segments,self.phase,dt,measured)
        return self.targets.copy(),self.enabled.copy()

    def status(self):
        return dict(stage=self.stage,time_s=self.time,phase_rad=self.phase,straight_m=self.straight,
                    helix_length_m=self.helix_length,enabled=(np.flatnonzero(self.enabled)+1).tolist(),
                    fit=self.fit,options=asdict(self.vo),segments=[asdict(s) for s in self.last_segments],
                    nominal_head_u_m=self.head_u,reference_gamma=None if self.bent is None else self.bent.gamma,
                    horizontal=None if self.horizontal is None else self.horizontal.status())
