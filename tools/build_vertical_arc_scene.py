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
SHARP_SCALE=PIPE_RADIUS/.0565
SHARP_LEG=.15*SHARP_SCALE
SHORT_BEND_RADIUS=.105


def axis(u,radius=BEND_RADIUS,entry=BEND_HEIGHT):
    if u<0:return np.array([0.,0.,entry+u]),np.array([1.,0.,0.])
    if u<=np.pi*radius/2:
        a=u/radius
        return np.array([radius*(1-np.cos(a)),0.,entry+radius*np.sin(a)]),np.array([np.cos(a),0.,-np.sin(a)])
    return np.array([radius+u-np.pi*radius/2,0.,entry+radius]),np.array([0.,0.,-1.])


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
    if scene=='acute':
        source=ET.parse(ROOT/'mujoco_model/Snake_Robot_o20_Rolling_Arc_elbow90.xml').getroot()
        # Preserve the source's two seam planes at z=0.95 and x=0.15,
        # rather than starting either cylinder inside the miter meshes.
        height=BEND_HEIGHT+SHARP_LEG/2
        entry=height-SHARP_LEG
        pipe.find('geom').set('fromto',f'0 0 -.2 0 0 {entry}')
        for source_geom in source.findall(".//geom[@type='mesh']"):
            name=source_geom.get('mesh','')
            if not name.startswith('pipe_sharp_'):
                continue
            original=source.find(f"asset/mesh[@name='{name}']")
            vertices=np.fromstring(original.get('vertex'),sep=' ')*SHARP_SCALE
            ET.SubElement(assets,'mesh',name=name,vertex=' '.join(f'{x:.12g}' for x in vertices))
            pos=np.fromstring(source_geom.get('pos'),sep=' ')
            pos=(pos-[0,0,1.1])*SHARP_SCALE+[0,0,height]
            ET.SubElement(pipe,'geom',name=name,type='mesh',mesh=name,pos=' '.join(map(str,pos)),**common)
        ET.SubElement(pipe,'geom',name='pipe_horizontal',type='cylinder',size='.055',
                      fromto=f'{SHARP_LEG} 0 {height} 2.3 0 {height}',**common)
        route=np.array([[0,0,-.2,.055],[0,0,entry,.055],
                        [0,0,height,.055],[SHARP_LEG,0,height,.055],[2.3,0,height,.055]])
    elif scene in ('elbow','short'):
        radius=SHORT_BEND_RADIUS if scene=='short' else BEND_RADIUS
        entry=BEND_HEIGHT+SHARP_LEG/2-radius if scene=='short' else BEND_HEIGHT
        pipe.find('geom').set('fromto',f'0 0 -.2 0 0 {entry}')
        angles=np.linspace(0,np.pi/2,25);theta=np.arange(48)*2*np.pi/48
        for i,(a,b) in enumerate(zip(angles[:-1],angles[1:])):
            rings=[]
            for x in (a,b):
                center,normal=axis(x*radius,radius,entry)
                rings.extend(center+PIPE_RADIUS*(np.cos(theta)[:,None]*normal+np.sin(theta)[:,None]*np.array([0.,1.,0.])))
            mesh=f'pipe_bend_{i:02d}'
            ET.SubElement(assets,'mesh',name=mesh,vertex=' '.join(f'{x:.10g}' for x in np.asarray(rings).ravel()))
            ET.SubElement(pipe,'geom',name=mesh,type='mesh',mesh=mesh,**common)
        ET.SubElement(pipe,'geom',name='pipe_horizontal',type='cylinder',size='.055',
                      fromto=f'{radius} 0 {entry+radius} 2.3 0 {entry+radius}',**common)
        u=np.r_[np.linspace(-entry-.2,0.,41),np.linspace(0,np.pi*radius/2,49)[1:],
                np.linspace(np.pi*radius/2,np.pi*radius/2+2.,25)[1:]]
        route=np.array([np.r_[axis(x,radius,entry)[0],PIPE_RADIUS] for x in u])
    else:
        pipe.find('geom').set('fromto','0 0 -.2 0 0 4')
        route=np.array([[0.,0.,z,.055] for z in np.linspace(-.2,4.,85)])
    custom=root.find('custom')
    for c in list(custom):custom.remove(c)
    ET.SubElement(custom,'numeric',name='pipe_centerline',data=' '.join(map(str,route.ravel())))
    if scene=='acute':
        feature=(entry+.2,entry+.2+2*SHARP_LEG)
    elif scene in ('elbow','short'):
        feature=(entry+.2,entry+.2+np.pi*radius/2)
    else:
        feature=(2,2+np.pi*BEND_RADIUS/2)
    ET.SubElement(custom,'numeric',name='feature_span',data=' '.join(map(str,feature)))
    ET.SubElement(custom,'numeric',name='robot_lengths',data=' '.join(map(str,LENGTHS)))
    root.find('statistic').set('center','0 0 1.6');root.find('statistic').set('extent','2')
    path=ROOT/f'mujoco_model/Snake_Robot_o18_Vertical_ARC_{scene}.xml'
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    return str(path.relative_to(ROOT))


if __name__=='__main__':
    paths=[build(name) for name in ('straight','elbow','acute','short')]
    print(json.dumps(dict(models=paths,lengths_m=LENGTHS.tolist(),pipe_diameter_m=.11,bend_radius_m=.3)))
