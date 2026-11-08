"""Nominal bent-axis wrap, resampled in BODY arc length, then Bishop curvature.

The nominal radius and elbow shape are explicit planning inputs. Nothing here
queries simulation geometry or physical root pose. Frame transport has no
gravity projection singularity at a vertical axis.
"""
import math
import numpy as np


def transport(normal,previous_tangent,tangent):
    v=np.cross(previous_tangent,tangent);c=np.sum(previous_tangent*tangent,axis=-1)
    return normal+np.cross(v,normal)+np.cross(v,np.cross(v,normal))/(1+c)[...,None]


class BentHelixCurve:
    def __init__(self,radius,pitch,bend_radius=.3):
        self.r=radius;self.pitch=pitch;self.R=bend_radius
        self.previous_tangent=None;self.reference_normal=None;self.gamma=0.

    def axis(self,u):
        u=np.asarray(u);a=np.clip(u/self.R,0,np.pi/2);end=self.R*np.pi/2
        p=np.column_stack([self.R*(1-np.cos(a))+np.maximum(u-end,0),np.zeros_like(u),self.R*np.sin(a)+np.minimum(u,0)])
        n=np.column_stack([np.cos(a),np.zeros_like(u),-np.sin(a)])
        return p,n

    def field(self,head_u,length,theta_head=math.pi):
        # Constant helix angle: theta' includes the offset-curve metric. This
        # keeps tangent continuous where straight pipe enters constant bend.
        w=2*math.pi/self.pitch;factor=math.hypot(1,self.r*w)
        vmax=length/factor/(1-self.r/self.R)*1.12
        v=np.linspace(0,vmax,max(100,int(vmax/.0003)+1));dv=v[1]-v[0]
        theta=np.empty(len(v));theta[0]=theta_head
        for i in range(1,len(v)):
            k=1/self.R if 0<head_u-(v[i]+v[i-1])/2<self.R*np.pi/2 else 0.
            f=-w*(1-self.r*k*math.cos(theta[i-1]))
            theta[i]=theta[i-1]-dv*w*(1-self.r*k*math.cos(theta[i-1]+.5*dv*f))
        p,n=self.axis(head_u-v);p+=self.r*(np.cos(theta)[:,None]*n+np.sin(theta)[:,None]*np.array([0.,1.,0.]))
        material=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        s=np.linspace(0,length,max(40,int(length/.0015)+1))
        p=np.column_stack([np.interp(s,material,p[:,j]) for j in range(3)])
        t=np.gradient(p,s,axis=0,edge_order=2);t/=np.linalg.norm(t,axis=1)[:,None]
        kvec=np.gradient(t,s,axis=0,edge_order=2);kvec-=t*(kvec*t).sum(1)[:,None]
        kappa=np.linalg.norm(kvec,axis=1);norm=kvec/np.maximum(kappa[:,None],1e-10)
        transported=transport(norm[:-1],t[:-1],t[1:])
        delta=np.arctan2((np.cross(transported,norm[1:])*t[1:]).sum(1),(transported*norm[1:]).sum(1))
        psi=np.r_[0.,np.cumsum(delta)]
        if self.previous_tangent is None:self.reference_normal=norm[0].copy()
        else:self.reference_normal=transport(self.reference_normal,self.previous_tangent,t[0])
        self.reference_normal-=t[0]*(self.reference_normal@t[0]);self.reference_normal/=np.linalg.norm(self.reference_normal)
        self.previous_tangent=t[0].copy()
        gamma=math.atan2(np.cross(self.reference_normal,norm[0])@t[0],self.reference_normal@norm[0])
        self.gamma+=math.atan2(math.sin(gamma-self.gamma),math.cos(gamma-self.gamma))
        psi+=self.gamma
        components=np.column_stack([kappa*np.cos(psi),kappa*np.sin(psi)])
        return s,(components[:-1]+components[1:])/2,float(psi[-1])
