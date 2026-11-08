"""18 measured-spacing hinges on a bare vertical/90-degree/horizontal pipe."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LENGTHS=np.array([.025]+[.072]*17+[.091])
BEND_HEIGHT=1.8
BEND_RADIUS=.3
PIPE_RADIUS=.055


def axis(u):
    if u<0:return np.array([0.,0.,BEND_HEIGHT+u]),np.array([1.,0.,0.])
    if u<=np.pi*BEND_RADIUS/2:
        a=u/BEND_RADIUS
        return np.array([BEND_RADIUS*(1-np.cos(a)),0.,BEND_HEIGHT+BEND_RADIUS*np.sin(a)]),np.array([np.cos(a),0.,-np.sin(a)])
    return np.array([BEND_RADIUS+u-np.pi*BEND_RADIUS/2,0.,BEND_HEIGHT+BEND_RADIUS]),np.array([0.,0.,-1.])


def build(scene):
    root=ET.parse(ROOT/'mujoco_model/Snake_Robot_o18_Tee_straight.xml').getroot()
    root.set('model','vertical-arc18-'+scene)
    world=root.find('worldbody');assets=root.find('asset')
    for mesh in assets.findall('mesh'):
        scale=np.array(mesh.get('scale','1 1 1').split(),float);scale[2]*=.072/.0645
        mesh.set('scale',' '.join(map(str,scale)))
    tail_mesh=ET.fromstring(ET.tostring(assets.find("mesh[@name='v_link']")))
    tail_mesh.set('name','v_tail');scale=np.array(tail_mesh.get('scale').split(),float);scale[2]*=.091/.072
    tail_mesh.set('scale',' '.join(map(str,scale)));assets.append(tail_mesh)
    head=world.find("body[@name='head']")
    head.find("site[@name='s_head']").set('pos','0 0 .025')
    shell=head.find('geom');shell.set('size','.029 .0125');shell.set('pos','0 0 .0125')
    for j in range(1,19):
        body=next(b for b in head.iter('body') if b.find(f"joint[@name='joint{j}']") is not None)
        body.set('pos',f'0 0 {-LENGTHS[j]:.9g}')
        body.find('joint').set('pos',f'0 0 {LENGTHS[j]:.9g}')
        if j==18:body.find('geom').set('mesh','v_tail')
    pipe=world.find("body[@name='pipe_network']")
    for child in list(pipe):pipe.remove(child)
    common=dict(friction='.5 .005 .002',condim='6',priority='1',material='pipe_bare',group='1')
    ET.SubElement(pipe,'geom',name='pipe_vertical',type='cylinder',size='.055',fromto=f'0 0 -.2 0 0 {BEND_HEIGHT}',**common)
    if scene=='elbow':
        angles=np.linspace(0,np.pi/2,25);theta=np.arange(48)*2*np.pi/48
        for i,(a,b) in enumerate(zip(angles[:-1],angles[1:])):
            rings=[]
            for x in (a,b):
                center,normal=axis(x*BEND_RADIUS)
                rings.extend(center+PIPE_RADIUS*(np.cos(theta)[:,None]*normal+np.sin(theta)[:,None]*np.array([0.,1.,0.])))
            mesh=f'pipe_bend_{i:02d}'
            ET.SubElement(assets,'mesh',name=mesh,vertex=' '.join(f'{x:.10g}' for x in np.asarray(rings).ravel()))
            ET.SubElement(pipe,'geom',name=mesh,type='mesh',mesh=mesh,**common)
        ET.SubElement(pipe,'geom',name='pipe_horizontal',type='cylinder',size='.055',fromto=f'.3 0 {BEND_HEIGHT+.3} 2.3 0 {BEND_HEIGHT+.3}',**common)
        u=np.r_[np.linspace(-2.,0.,41),np.linspace(0,np.pi*.3/2,49)[1:],np.linspace(np.pi*.3/2,np.pi*.3/2+2.,25)[1:]]
        route=np.array([np.r_[axis(x)[0],PIPE_RADIUS] for x in u])
    else:
        pipe.find('geom').set('fromto','0 0 -.2 0 0 4')
        route=np.array([[0.,0.,z,.055] for z in np.linspace(-.2,4.,85)])
    custom=root.find('custom')
    for c in list(custom):custom.remove(c)
    ET.SubElement(custom,'numeric',name='pipe_centerline',data=' '.join(map(str,route.ravel())))
    ET.SubElement(custom,'numeric',name='feature_span',data=f'2 {2+np.pi*.3/2}')
    ET.SubElement(custom,'numeric',name='robot_lengths',data=' '.join(map(str,LENGTHS)))
    root.find('statistic').set('center','0 0 1.6');root.find('statistic').set('extent','2')
    path=ROOT/f'mujoco_model/Snake_Robot_o18_Vertical_ARC_{scene}.xml'
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    return str(path.relative_to(ROOT))


if __name__=='__main__':
    paths=[build(name) for name in ('straight','elbow')]
    print(json.dumps(dict(models=paths,lengths_m=LENGTHS.tolist(),pipe_diameter_m=.11,bend_radius_m=.3)))
