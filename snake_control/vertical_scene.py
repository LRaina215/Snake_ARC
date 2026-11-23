"""Research scene; all world/force information is evaluation-only."""
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .rolling_arc import ArcParameters,LocalMotorDrive
from .simulation import backbone

ROOT=Path(__file__).resolve().parents[1]


class VerticalScene:
    def __init__(self,scene,recorded_model=None):
        self.scene=scene
        self.model_path=ROOT/f'mujoco_model/Snake_Robot_o18_Vertical_ARC_{scene}.xml'
        if recorded_model is None:
            self.model=mujoco.MjModel.from_xml_path(str(self.model_path))
        else:
            asset_dir=self.model_path.parent/'asset'
            recorded_model=Path(recorded_model)
            xml=recorded_model.read_text(encoding='utf-8').replace('./asset/',asset_dir.as_posix()+'/')
            mesh = recorded_model.with_suffix('.obj')
            assets = {mesh.name: mesh.read_bytes()} if mesh.exists() else None
            self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data=mujoco.MjData(self.model)
        self.p=ArcParameters(joints=18,first_joint=1,last_joint=18)
        self.drive=LocalMotorDrive(self.model,self.p)
        self.lengths=self.model.numeric('robot_lengths').data.copy()
        self.pipe_geoms=set(np.flatnonzero(self.model.geom_bodyid==self.model.body('pipe_network').id))
        self.head=self.model.body('head').id

    def place_helix(self,q,head_height=1.42,beta=0.):
        """One initial placement; axis from tail toward head points upward."""
        m,d=self.model,self.data;d.qpos[:]=0.;d.qpos[3]=1.;d.qpos[self.drive.qadr]=q
        mujoco.mj_forward(m,d);pts=backbone(m,d);t=np.diff(pts,axis=0);t/=np.linalg.norm(t,axis=1)[:,None]
        axis=np.linalg.eigh(np.cov(t.T))[1][:,0]
        if (pts[0]-pts[-1])@axis<0:axis=-axis
        radial=pts-pts.mean(0);radial-=np.outer(radial@axis,axis)
        a=np.linalg.svd(radial)[2][0];a-=axis*(a@axis);a/=np.linalg.norm(a);b=np.cross(axis,a)
        xy=(pts-pts.mean(0))@np.column_stack([a,b]);v=np.linalg.lstsq(np.c_[2*xy,np.ones(len(xy))],(xy*xy).sum(1),rcond=None)[0]
        center=pts.mean(0)+a*v[0]+b*v[1]
        # rows a,b,axis give a proper rotation into the vertical world frame.
        rot=Rotation.from_rotvec(np.array([0.,0.,beta])).as_matrix()@np.vstack([a,b,axis])
        d.qpos[:3]=np.array([0.,0.,head_height-(pts[0]-center)@axis])-rot@center
        d.qpos[3:7]=Rotation.from_matrix(rot).as_quat()[[3,0,1,2]]
        d.qvel[:]=0.;mujoco.mj_forward(m,d)
        return dict(radius_m=float(np.sqrt(v[2]+v[0]**2+v[1]**2)),mass_kg=float(m.body_mass[self.head:].sum()))

    def evaluate(self,target,mask):
        m,d=self.model,self.data;pts=backbone(m,d);forces=np.zeros(3);links=set();floor=0.;self_contacts=[]
        robot=set(np.flatnonzero(m.geom_bodyid>=self.head))
        for i,c in enumerate(d.contact):
            a,b=int(c.geom1),int(c.geom2);f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
            if (a in self.pipe_geoms and b in robot) or (b in self.pipe_geoms and a in robot):
                rg=b if b in robot else a;links.add(int(m.geom_bodyid[rg]-self.head))
                forces+=(1 if rg==b else -1)*c.frame.reshape(3,3).T@f[:3]
            if m.geom('floor').id in (a,b) and (a in robot or b in robot):floor+=abs(f[0])
            if a in robot and b in robot and abs(m.geom_bodyid[a]-m.geom_bodyid[b])>2 and c.dist<0:
                self_contacts.append([int(m.geom_bodyid[a]-self.head),int(m.geom_bodyid[b]-self.head)])
        q=d.qpos[self.drive.qadr]
        return dict(head_z_m=float(pts[0,2]),head_x_m=float(pts[0,0]),center_m=pts.mean(0).tolist(),
                    min_height_m=float(pts[:,2].min()),pipe_force_n=forces.tolist(),contact_links=sorted(links),
                    floor_force_n=floor,self_contacts=self_contacts,max_joint_deg=float(np.max(abs(np.degrees(q)))),
                    tracking_deg=float(np.max(abs(np.degrees(q[mask]-target[mask])),initial=0.)),
                    passive_actuator_nm=float(np.max(abs(d.actuator_force[~mask]),initial=0.)),
                    saturated_joints=(np.flatnonzero(mask & (abs(d.actuator_force)>2.35))+1).tolist())
