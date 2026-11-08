"""Experimental reverse shift: sampled joint angles -> open ARC -> material centering.

No MuJoCo, pipe coordinates, contact forces or root orientation enter this module.
Units: metres, radians, seconds. The same declarations are bundled into the real entry.
"""
from dataclasses import dataclass
import math

import numpy as np

from .curve_gait import CurveGait, Segment, line, arc
from .gravity_moment import smooth
from .section_geometry import forward


def fit_helix(q, link):
    """Fit the two alternating curvature-integral families, including signed torsion."""
    from scipy.optimize import least_squares
    s=(np.arange(len(q))+1)*link
    odd=np.arange(len(q))%2==0
    def basis(tau):
        a=tau*s
        return np.column_stack((np.where(odd,np.cos(a),np.sin(a)),
                                np.where(odd,-np.sin(a),np.cos(a))))
    seeds=[]
    for tau in np.linspace(-10.,10.,101):
        c=np.linalg.lstsq(basis(tau),q,rcond=None)[0]
        seeds.append((float(np.linalg.norm(basis(tau)@c-q)),c,tau))
    _,c,tau=min(seeds,key=lambda x:x[0])
    fit=least_squares(lambda x:basis(x[2])@x[:2]-q,[*c,tau],max_nfev=60)
    x=fit.x;amplitude=float(np.hypot(*x[:2]));tau=float(x[2])
    if not math.isfinite(amplitude) or amplitude<1e-8:
        raise ValueError('当前角反馈没有可计算的螺旋曲率；Ctrl+7用于已搭管的Helix')
    kappa=amplitude/(2*link*np.sinc(tau*link/math.pi))
    radius=kappa/(kappa*kappa+tau*tau)
    return dict(kappa=kappa,tau=tau,radius=radius,phase=math.atan2(x[1],x[0]),
                pitch=2*math.pi*abs(tau)/(kappa*kappa+tau*tau),
                rms_deg=float(np.degrees(np.sqrt(np.mean(fit.fun**2)))))


def circle_frame(points):
    tangents=np.diff(points,axis=0);tangents/=np.linalg.norm(tangents,axis=1)[:,None]
    axis=(np.linalg.svd(points-points.mean(0))[2][-1] if len(points)<5 else
          np.linalg.eigh(np.cov(tangents.T))[1][:,0])
    e1=points[0]-points.mean(0);e1-=axis*(e1@axis);e1/=np.linalg.norm(e1)
    e2=np.cross(axis,e1)
    xy=(points-points.mean(0))@np.column_stack((e1,e2))
    c=np.linalg.lstsq(np.column_stack((2*xy,np.ones(len(xy)))),(xy*xy).sum(1),rcond=None)[0]
    center=points.mean(0)+e1*c[0]+e2*c[1]
    return center,axis


@dataclass(frozen=True)
class HelixArcOptions:
    speed_scale: float=1.                     # Motion multiplier; feedback and settling keep real time.
    retained_turns: float=1.                  # Leave this helical budget before forming the semicircle.
    exit_extra_rad: float=math.radians(30.)    # Exit bend beyond the helix lead angle; separates the falling tail.
    head_clearance_rad: float=math.radians(40.)# Temporary orthogonal head bend, removed before release.
    reverse_feed_m_s: float=.016               # Straight-tail growth, coupled to reverse rolling below.
    reverse_roll_rad_s: float=.112
    target_rate_deg_s: float=35.
    feedback_period_s: float=.2                # Caller sampling period; repeated calculations are not fresh angles.
    lead_s: float=1.5
    flatten_s: float=6.
    center_rate_m_s: float=.010                # Maximum material-coordinate speed during centering.
    center_tolerance_m: float=.025             # Observed tail length minus head length.
    roll_rate_rad_s: float=.35


class HelixArcTransition:
    """Reverse shift -> crown selection -> guided flattening -> release -> measured centering.

    The caller supplies sampled angles in radians and a fresh-sample flag.
    Outputs are target angles and a binary torque-enable mask. A fitting error
    is not a contact/load measurement; all physical acceptance lives outside.
    """
    def __init__(self,angles_rad,lengths,options=HelixArcOptions()):
        self.lengths=np.asarray(lengths,float);self.n=len(angles_rad)
        self.l=float(np.mean(lengths));self.L=float(sum(lengths));self.o=options
        if not math.isfinite(options.speed_scale) or options.speed_scale<=0.:
            raise ValueError('Ctrl+7速度倍率必须为有限正数')
        self.curve=CurveGait(self.n,self.l)
        self.fit=fit_helix(np.asarray(angles_rad),self.l)
        self.phase=self.fit['phase'];self.kappa=self.fit['kappa'];self.tau=self.fit['tau'];self.radius=self.fit['radius']
        self.lead=math.atan(self.fit['pitch']/(2*math.pi*self.radius))+options.exit_extra_rad
        self.targets=np.asarray(angles_rad,float).copy();self.enabled=np.ones(self.n,bool)
        self.stage='exit_transition';self.elapsed=0.;self.time=0.;self.straight=0.;self.head=0.
        self.helix_length=self.L;self.crown=None;self.down=None;self.geometry={}
        self.events=[];self.align_sign=1.
        self.arc_length=math.pi*self.radius;self.center_error=None;self.complete_since=None

    def enter(self,stage):
        self.events.append(dict(time_s=self.time,stage=stage));self.stage=stage;self.elapsed=0.

    def measured_geometry(self,q):
        """Infer down from the free distal span, then estimate crown or hanging lengths."""
        points=forward(q,self.lengths)
        distal=0
        for j in range(self.n-1,-1,-1):
            if self.enabled[j]:break
            distal+=1
        if distal>=3:
            tail=np.diff(points[-min(distal,5)-1:],axis=0)
            t=tail/np.linalg.norm(tail,axis=1)[:,None]
            direction=t.mean(0);quality=float(np.linalg.norm(direction))
            if quality>.90:self.down=direction/quality
        if self.down is None:return dict(valid=False,reason='gravity reference not yet free',distal=distal)
        if self.stage in ('arc_release','arc_settle','auto_center','center_settle','arc_roll'):
            lo=max(0,int(math.ceil(self.head/self.l)))
            hi=min(self.n+1,int(math.floor((self.head+self.arc_length)/self.l)))
            center,axis=circle_frame(points[lo:hi+1])
            up=-self.down+axis*(self.down@axis);up/=np.linalg.norm(up)
            heights=(points-center)@up
            top=lo+int(np.argmax(heights[lo:hi+1]))
            def crossing(indices):
                for a,b in zip(indices[:-1],indices[1:]):
                    if heights[a]>=0 and heights[b]<=0:
                        f=heights[a]/max(heights[a]-heights[b],1e-12)
                        return (a+f*(b-a))*self.l
                return None
            left=crossing(list(range(top,-1,-1)));right=crossing(list(range(top,self.n+2)))
            valid=left is not None and right is not None
            return dict(valid=valid,head_m=left,tail_m=None if right is None else self.L-right,
                        difference_m=None if not valid else self.L-right-left,
                        down=self.down.tolist(),distal=distal,reason=None if valid else 'both hanging crossings not found')
        count=max(5,min(self.n+1,int(self.helix_length/self.l)))
        try:
            center,axis=circle_frame(points[:count+1])
            up=-self.down+axis*(self.down@axis);up/=np.linalg.norm(up)
            side=np.cross(axis,up)
            angle=np.unwrap(np.arctan2((points[:count+1]-center)@side,(points[:count+1]-center)@up))
            coords=np.arange(count+1)*self.l
            crossings=[]
            for level in np.arange(-4,5)*2*math.pi:
                for i in range(len(angle)-1):
                    if (angle[i]-level)*(angle[i+1]-level)<=0 and abs(angle[i+1]-angle[i])>1e-6:
                        crossings.append(float(coords[i]+(coords[i+1]-coords[i])*(level-angle[i])/(angle[i+1]-angle[i])))
            target=self.helix_length/2
            crown=min(crossings,key=lambda x:abs(x-target)) if crossings else None
            return dict(valid=crown is not None,crown_m=crown,distal=distal,down=self.down.tolist(),
                        axis=axis.tolist(),reason=None if crossings else 'no crown in retained helix')
        except (ValueError,np.linalg.LinAlgError):
            return dict(valid=False,reason='short shape fit',distal=distal)

    def set_curve(self,segments,phase,dt,mask=None):
        desired=self.curve.targets(segments,phase)
        if mask is None:
            centers=(np.arange(self.n)+1)*self.l
            bounds=np.r_[0.,np.cumsum([s.length for s in segments])]
            mask=np.zeros(self.n,bool)
            for i,s in enumerate(segments):
                if s.kappa>1e-8:mask|=(centers+self.l>bounds[i]+1e-6)&(centers-self.l<bounds[i+1]-1e-6)
        entering=mask&~self.enabled
        self.targets[entering]=self.measured[entering]
        self.enabled=mask
        limit=math.radians(self.o.target_rate_deg_s)*self.o.speed_scale*dt
        self.targets+=np.clip(desired-self.targets,-limit,limit)
        self.targets=np.clip(self.targets,-math.radians(88),math.radians(88))
        self.targets[~mask]=self.measured[~mask]

    def reverse_segments(self,lead):
        """Grow the outgoing line at the expense of the fitted helix's material length."""
        transition=self.radius*lead
        self.helix_length=self.L-self.straight-transition
        return [Segment(self.helix_length,self.kappa,self.tau),
                arc(transition,1/self.radius,math.pi/2),line(self.straight)]

    def step(self,dt,time_s,angles_rad,fresh=False):
        self.time=time_s;self.elapsed+=dt;self.measured=np.asarray(angles_rad,float)
        motion_dt=dt*self.o.speed_scale
        if fresh:self.geometry=self.measured_geometry(self.measured)
        if self.stage=='exit_transition':
            self.set_curve(self.reverse_segments(self.lead*smooth(self.elapsed*self.o.speed_scale/self.o.lead_s)),self.phase,dt)
            if self.elapsed*self.o.speed_scale>=self.o.lead_s:self.enter('reverse_unwrap')
        elif self.stage=='reverse_unwrap':
            self.straight+=self.o.reverse_feed_m_s*motion_dt
            self.phase-=self.o.reverse_roll_rad_s*motion_dt
            self.set_curve(self.reverse_segments(self.lead),self.phase,dt)
            if self.helix_length<=self.o.retained_turns*math.hypot(2*math.pi*self.radius,self.fit['pitch']):self.enter('crown_align')
        elif self.stage=='crown_align':
            g=self.geometry
            if g.get('valid'):
                crown=g['crown_m']
                fits=(crown-self.arc_length/2>=self.l/2 and
                      crown+self.arc_length/2<=self.helix_length-self.l/2)
                if fits:
                    self.crown=g['crown_m'];self.start_phase=self.phase
                    self.flatten_mask=self.enabled.copy()
                    scores=[]
                    for sign in (-1.,1.):
                        segments,phase=self.flatten_segments(.5,sign)
                        predicted=self.measured.copy();predicted[self.flatten_mask]=self.curve.targets(segments,phase)[self.flatten_mask]
                        pts=forward(predicted,self.lengths)
                        head=pts[:max(2,int((self.crown-self.arc_length/2)/self.l)+1)]
                        tail=pts[int(self.helix_length/self.l)+2:]
                        scores.append((float(np.linalg.norm(head[:,None]-tail[None,:],axis=2).min()),sign))
                    self.clearance_sign=max(scores)[1];self.enter('flatten')
                else:
                    self.phase+=self.align_sign*.20*motion_dt
            else:self.phase+=self.align_sign*.20*motion_dt
            self.set_curve(self.reverse_segments(self.lead),self.phase,dt)
        elif self.stage=='flatten':
            w=smooth(self.elapsed*self.o.speed_scale/self.o.flatten_s)
            segments,phase=self.flatten_segments(w,self.clearance_sign)
            # Guide the short head prefix to the opposite side before letting
            # gravity take it. Early head torque-off sweeps it across the tail.
            self.set_curve(segments,phase,dt,mask=self.flatten_mask)
            if self.elapsed*self.o.speed_scale>=self.o.flatten_s:
                self.head=self.crown-self.arc_length/2;self.phase=self.start_phase+self.tau*self.crown
                self.release_mask=self.enabled.copy();self.enter('arc_release')
        elif self.stage=='arc_release':
            centers=(np.arange(self.n)+1)*self.l
            curved=(centers+self.l>self.head+1e-6)&(centers-self.l<self.head+self.arc_length-1e-6)
            leaving=np.flatnonzero(self.release_mask&~curved)
            allowed=min(len(leaving),int(self.elapsed*self.o.speed_scale/.2))
            mask=self.release_mask.copy();mask[leaving[:allowed]]=False
            self.arc_targets(dt,mask=mask)
            if allowed==len(leaving):self.enter('arc_settle')
        elif self.stage=='arc_settle':
            self.arc_targets(dt)
            if self.elapsed>=2.:self.enter('auto_center')
        elif self.stage=='auto_center':
            self.center_error=self.geometry.get('difference_m')
            if self.geometry.get('valid') and self.center_error is not None:
                self.head+=float(np.clip(.2*self.center_error,-self.o.center_rate_m_s,self.o.center_rate_m_s))*motion_dt
                self.head=float(np.clip(self.head,self.l/2,self.L-self.arc_length-self.l/2))
            self.arc_targets(dt)
            if self.center_error is not None and abs(self.center_error)<self.o.center_tolerance_m:
                self.complete_since=self.time if self.complete_since is None else self.complete_since
                if self.time-self.complete_since>=.8:self.enter('center_settle')
            else:self.complete_since=None
        elif self.stage=='center_settle':
            self.arc_targets(dt)
            if self.elapsed>=3.:self.enter('arc_roll')
        elif self.stage=='arc_roll':
            self.phase+=self.o.roll_rate_rad_s*motion_dt;self.arc_targets(dt)
        return self.targets.copy(),self.enabled.copy()

    def arc_targets(self,dt,mask=None):
        self.set_curve([line(self.head),arc(self.arc_length,1/self.radius),
                        line(self.L-self.head-self.arc_length)],self.phase,dt,mask=mask)

    def flatten_segments(self,w,sign):
        start=w*(self.crown-self.arc_length/2)
        end=(1-w)*self.helix_length+w*(self.crown+self.arc_length/2)
        tau=(1-w)*self.tau;kappa=(1-w)*self.kappa+w/self.radius
        phase=self.start_phase+self.tau*self.crown-tau*(self.crown-start)
        bow=sign*self.o.head_clearance_rad*math.sin(math.pi*w)
        head_phase=phase+math.pi/2
        # A temporary orthogonal arc separates the outgoing head from the
        # already hanging tail along the pipe; its curvature vanishes at both ends.
        return [Segment(start,abs(bow)/max(start,1e-9),0.,head_phase+(math.pi if bow<0 else 0.)),
                Segment(end-start,kappa,tau,phase-head_phase-(math.pi if bow<0 else 0.)),
                line(self.L-end)],0.

    def status(self):
        return dict(stage=self.stage,phase=self.phase,helix_length_m=self.helix_length,
                    speed_scale=self.o.speed_scale,
                    straight_m=self.straight,arc_head_m=self.head,arc_length_m=self.arc_length,
                    crown_m=self.crown,fit=self.fit,geometry=self.geometry,center_error_m=self.center_error,
                    enabled_joints=(np.flatnonzero(self.enabled)+1).tolist())


class HelixArcManual:
    """J/G scheduling around the validated geometry; pauses never advance curve time.

    Angle samples are degrees at the entry boundary, radians inside the core.
    Centering completion pauses once. A further J/G explicitly starts ARC rolling.
    """
    def __init__(self,measured_deg,lengths,options=HelixArcOptions(),step_s=1.,max_feedback_age_s=.8):
        self.core=HelixArcTransition(np.radians(measured_deg),lengths,options)
        self.step_s=step_s;self.max_feedback_age_s=max_feedback_age_s
        self.running=False;self.started=False;self.step_left=0.;self.control_time=0.
        self.feedback_time=None;self.completed=False;self.feedback_wait=False

    @property
    def targets(self):return self.core.targets

    @property
    def enabled(self):return self.core.enabled

    @property
    def stage(self):return self.core.stage

    @property
    def phase(self):return self.core.phase

    def pause(self):
        self.running=False;self.step_left=0.

    def step_once(self):
        self.running=False;self.step_left=self.step_s

    def advance(self,dt,now,feedback_time,measured_deg):
        if not self.running and self.step_left<=0.:return False
        self.feedback_wait=feedback_time is None or now-feedback_time>self.max_feedback_age_s
        if self.feedback_wait:return False
        fresh=self.feedback_time is None or feedback_time>self.feedback_time
        self.feedback_time=feedback_time;self.started=True
        used=dt if self.running else min(dt,self.step_left)
        self.core.step(used,self.control_time,np.radians(measured_deg),fresh)
        self.control_time+=used
        self.step_left=max(0.,self.step_left-used)
        if self.step_left<1e-9:self.step_left=0.
        if self.stage=='arc_roll' and not self.completed:
            self.completed=True;self.pause()
        return True

    def status(self):
        return dict(self.core.status(),running=self.running,step_remaining_s=self.step_left,
                    control_time_s=self.control_time,started=self.started,completed=self.completed,
                    feedback_wait=self.feedback_wait)
