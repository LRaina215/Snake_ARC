"""Local geometry, ported from Snake_Robot_Gait's C# visualizer.

Reference: Kinematics.cs, RollingAxis.cs, CrossSectionCurve.cs (2023 paper).
Lengths are metres, angles radians. MuJoCo -> visualizer signs: odd +, even -.
No MuJoCo, pipe geometry, contact or root pose is imported here.
"""
from dataclasses import dataclass
import math
import numpy as np


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-14)


def forward(q, lengths):
    """Head-frame skeleton, including head and tail endpoints."""
    q, lengths = np.asarray(q), np.asarray(lengths)
    if len(lengths) != len(q) + 1 or not np.isfinite(q).all():
        raise ValueError('Mechanical dimensions or angles are invalid')
    points = [np.zeros(3), np.array([-lengths[0], 0., 0.])]
    orientation = np.eye(3)
    for i, angle in enumerate(q):
        a = angle if i % 2 == 0 else -angle
        c, s = math.cos(a), math.sin(a)
        rotation = (np.array([[c,-s,0],[s,c,0],[0,0,1.]]) if i % 2 == 0 else
                    np.array([[c,0,s],[0,1.,0],[-s,0,c]]))
        orientation = orientation @ rotation
        points.append(points[-1] + orientation @ [-lengths[i+1],0,0])
    return np.asarray(points)


def pitch_variance(tangents, directions):
    a = np.arcsin(np.clip(np.abs(np.asarray(directions) @ tangents.T),0,1))
    return np.var(a, axis=-1)


def refine_axis(tangents, seed):
    direction = unit(seed); variance = float(pitch_variance(tangents,direction)); step=.35
    for _ in range(100):
        if step <= 1e-7 or variance <= 1e-18: break
        reference = [0,0,1] if abs(direction[2]) < .85 else [0,1,0]
        a=unit(np.cross(reference,direction)); b=unit(np.cross(direction,a))
        offsets=np.array([a,-a,b,-b,(a+b)*.7071,(a-b)*.7071,(-a+b)*.7071,(-a-b)*.7071])
        candidates=direction+step*offsets
        candidates/=np.linalg.norm(candidates,axis=1)[:,None]
        values=pitch_variance(tangents,candidates); improved=False
        # Preserve C# candidate-order tie breaking.
        for candidate, value in zip(candidates,values):
            if value + 1e-18 < variance:
                direction,variance,improved=candidate,float(value),True
        if not improved: step *= .5
    return direction,variance


def paper_axis(points):
    """Original pitch-variance objective and coarse/refine search."""
    points=np.asarray(points); center=points.mean(axis=0)
    first,last=1,len(points)-2
    if len(points)>=12:
        typical=np.sort(np.linalg.norm(np.diff(points[1:-1],axis=0),axis=1))
        typical=typical[len(typical)//2]
        if np.linalg.norm(points[1]-points[0])<typical*.8: first+=2
        if np.linalg.norm(points[-1]-points[-2])<typical*.8: last-=2
    tangents=np.asarray([unit(points[i+1]-points[i-1]) for i in range(first,last+1)])
    pv,pe=np.linalg.eigh((points-center).T@(points-center))
    mean=tangents.mean(axis=0); centered=tangents-mean
    tv,te=np.linalg.eigh(centered.T@centered/len(tangents)); spread=max(0,float(tv.sum()))
    if len(tangents)<3:
        direction,variance,confidence=pe[:,-1],0.,.25
    elif spread<.015:
        direction=unit(mean) if np.linalg.norm(mean)>1e-8 else pe[:,-1]
        variance=float(pitch_variance(tangents,direction)); confidence=0.
    else:
        i=np.arange(256); z=1-2*(i+.5)/256; r=np.sqrt(1-z*z); az=i*2.399963229728653
        coarse=np.column_stack([r*np.cos(az),r*np.sin(az),z])
        values=pitch_variance(tangents,coarse)
        seeds=[*te.T,pe[:,-1],unit(points[-1]-points[0]),*coarse[np.argsort(values)[:4]]]
        candidates=[refine_axis(tangents,s) for s in seeds]
        direction,variance=min(candidates,key=lambda item:item[1])
        second,small=max(0,tv[1]),max(0,tv[0])
        gap=0 if second<1e-9 else np.clip((second-small)/second,0,1)
        confidence=float(gap*np.clip(spread/.08,0,1)*np.clip(1-math.sqrt(variance)/.35,0,1))
        if second<max(1e-8,spread*1e-4): direction,confidence=pe[:,-1],min(confidence,.25)
    direction=unit(direction); end=unit(points[0]-points[-1]); alignment=direction@end
    if abs(alignment)>.08: direction*=1 if alignment>=0 else -1
    else:
        alignment=-float(tangents@direction @ np.ones(len(tangents)))
        if abs(alignment)>len(tangents)*.02: direction*=1 if alignment>=0 else -1
        elif direction[np.argmax(abs(direction))]<0: direction*=-1
    return dict(point=center,direction=direction,confidence=confidence,variance=variance)


def projection_basis(direction, previous=None):
    h=np.zeros(3) if previous is None else previous-direction*(previous@direction)
    if np.linalg.norm(h)<1e-8:
        h=np.cross([0,0,1] if abs(direction[2])<.85 else [0,1,0],direction)
    h=unit(h)
    return np.column_stack([h,unit(np.cross(direction,h)),direction])


@dataclass
class Segment:
    start: np.ndarray
    tangent: np.ndarray
    length: float
    radius: float
    turn: float
    joint: int
    kind: str

    def at(self,t):
        if self.radius==0: return self.start + self.tangent*(self.length*t)
        a=self.turn*t; r=math.copysign(self.radius,self.turn)
        normal=np.array([-self.tangent[1],self.tangent[0]])
        return self.start+r*(math.sin(a)*self.tangent+2*math.sin(a/2)**2*normal)

    def end_tangent(self):
        c,s=math.cos(self.turn),math.sin(self.turn)
        return np.array([[c,-s],[s,c]])@self.tangent


def continuous_section(points, gamma_min=math.pi*.01, original=False):
    """A/B/C port; original=True reproduces D for migration comparison only.

    Open mode keeps straight turns zero and outward arcs signed. It never
    inserts an inward arc merely because a joint is straight/outward.
    """
    points=np.asarray(points); delta=np.diff(points,axis=0)
    lengths=np.linalg.norm(delta,axis=1)
    if np.min(lengths)<1e-8: raise ValueError('Zero projected link')
    tangents=delta/lengths[:,None]
    turns=np.arctan2(tangents[:-1,0]*tangents[1:,1]-tangents[:-1,1]*tangents[1:,0],
                     np.sum(tangents[:-1]*tangents[1:],axis=1))
    winding=1 if turns.sum()>=0 else -1
    position=points[0].copy(); tangent=tangents[0].copy(); result=[]
    def add(length,radius,turn,joint,kind):
        nonlocal position,tangent
        if length<1e-12: return
        row=Segment(position.copy(),tangent.copy(),length,radius,turn,joint,kind)
        result.append(row); position=row.at(1); tangent=row.end_tangent()
    add(lengths[0]/2,0,0,0,'end')
    for i,turn in enumerate(turns,1):
        a,b=lengths[i-1]/2,lengths[i]/2
        if original and turn*winding<=gamma_min:
            add(a+b,(a+b)/gamma_min,winding*gamma_min,i,'D')
        elif abs(turn)<1e-7:
            add(a+b,0,0,i,'straight')
        else:
            radius=min(a,b)/math.tan(abs(turn)/2)
            kind='C' if abs(a-b)<=1e-10*max(a,b) else 'A' if a>b else 'B'
            if a>b: add(a-b,0,0,i,kind)
            add(radius*abs(turn),radius,turn,i,kind)
            if b>a: add(b-a,0,0,i,kind)
    # The tail half-link is a separate module after the final interior joint.
    add(lengths[-1]/2,0,0,len(points)-1,'end')
    return result


def sample_section(segments, step=.005):
    return np.asarray([segments[0].start]+[s.at(t) for s in segments for t in
                      np.linspace(0,1,max(2,int(math.ceil(s.length/step))+1))[1:]])


def local_section(q,lengths,first,last,previous=None, method='plane'):
    points=forward(q,lengths)[first-1:last+3]  # one entering/exiting boundary link
    if method=='paper': axis=paper_axis(points)
    else:
        # Open near-planar arc extension: directly fit its section plane.
        fitting=points[1:-1]  # passive boundary links cannot determine the section plane
        center=fitting.mean(axis=0); _,_,vh=np.linalg.svd(fitting-center,full_matrices=False)
        direction=vh[-1]
        if previous is not None and direction@previous['direction']<0: direction=-direction
        elif previous is None and direction[np.argmax(abs(direction))]<0: direction=-direction
        distances=(fitting-center)@direction
        axis=dict(point=center,direction=direction,confidence=float(np.clip(1-np.std(distances)/.03,0,1)),
                  variance=float(np.mean(distances**2)))
    if previous is not None and axis['direction']@previous['direction']<0: axis['direction']*=-1
    basis=projection_basis(axis['direction'],None if previous is None else previous['basis'][:,0])
    axis['basis']=basis
    projected=(points-axis['point'])@basis[:,:2]
    curve=continuous_section(projected)
    return dict(points=points,projected=projected,curve=curve,axis=axis,
                total_turn=sum(abs(s.turn) for s in curve if 1<=s.joint<=last-first+1),
                fit_rms=math.sqrt(axis['variance']))
