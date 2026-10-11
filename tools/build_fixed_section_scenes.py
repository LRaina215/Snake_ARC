"""Solid convex frusta with explicit radius/centerline metadata; same robot."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import xml.etree.ElementTree as ET
import numpy as np
import json
from scipy.spatial.transform import Rotation
from snake_control.simulation import ROOT


def build(name,points,feature):
    root=ET.parse(ROOT/'mujoco_model/Snake_Robot_o20_Rolling_Arc_straight.xml').getroot()
    root.set('model','fixed-section-'+name)
    pipe=root.find("worldbody/body[@name='pipe_network']")
    pipe.set('pos','0 0 1.2');pipe.set('quat','1 0 0 0')
    for g in list(pipe):pipe.remove(g)
    for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
        # Ring bases along horizontal centerline; each pair is a convex solid.
        direction=b[:3]-a[:3];direction/=np.linalg.norm(direction)
        lateral=np.cross(direction,[0,0,1]);up=np.cross(lateral,direction)
        theta=np.arange(48)*2*np.pi/48
        rings=[]
        for p in (a,b):
            rings.extend(p[:3]+p[3]*(np.cos(theta)[:,None]*lateral+np.sin(theta)[:,None]*up))
        mesh=f'{name}_solid_{i}'
        ET.SubElement(root.find('asset'),'mesh',name=mesh,vertex=' '.join(f'{v:.9g}' for v in np.asarray(rings).ravel()))
        ET.SubElement(pipe,'geom',name=mesh,type='mesh',mesh=mesh,material='pipe_rubber',group='1',
                      friction='2 .005 .002',condim='6')
    points=np.asarray(points).copy();points[:,2]+=1.2
    numeric=root.find("custom/numeric[@name='pipe_centerline']")
    numeric.set('data',' '.join(f'{v:.10g}' for v in points.ravel()))
    route=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points[:,:3],axis=0),axis=1))]
    root.find("custom/numeric[@name='feature_span']").set('data',' '.join(map(str,feature)))
    ET.ElementTree(root).write(ROOT/'mujoco_model'/f'Snake_Robot_o20_Rolling_Arc_{name}.xml',encoding='utf-8',xml_declaration=True)
    return dict(scene=name,sections=len(points)-1,radius_min_m=float(points[:,3].min()),
                radius_max_m=float(points[:,3].max()),feature_span_m=feature,route_end_m=float(route[-1]))


def extend_exit(source, name, extra_length=1.2):
    """Copy an existing obstacle unchanged and extend only its outgoing cylinder.

    The long outlet gives the whole robot room for three post-obstacle cycles.
    Convert world metadata to the original pipe body's local coordinates.
    """
    root=ET.parse(ROOT/'mujoco_model'/f'Snake_Robot_o20_Rolling_Arc_{source}.xml').getroot()
    root.set('model','fixed-section-'+name)
    pipe=root.find("worldbody/body[@name='pipe_network']")
    numeric=root.find("custom/numeric[@name='pipe_centerline']")
    points=np.fromstring(numeric.get('data'),sep=' ').reshape(-1,4)
    heading=points[-1,:3]-points[-2,:3];heading/=np.linalg.norm(heading)
    endpoint=points[-1].copy();endpoint[:3]+=extra_length*heading
    origin=np.fromstring(pipe.get('pos'),sep=' ')
    quaternion=np.fromstring(pipe.get('quat'),sep=' ')
    rotation=Rotation.from_quat(quaternion[[1,2,3,0]])
    outlet=pipe.findall('geom')[-1]
    ends=np.fromstring(outlet.get('fromto'),sep=' ')
    ends[3:]=rotation.inv().apply(endpoint[:3]-origin)
    outlet.set('fromto',' '.join(f'{v:.12g}' for v in ends))
    points=np.vstack([points,endpoint])
    numeric.set('data',' '.join(f'{v:.12g}' for v in points.ravel()))
    ET.ElementTree(root).write(ROOT/'mujoco_model'/f'Snake_Robot_o20_Rolling_Arc_{name}.xml',encoding='utf-8',xml_declaration=True)
    return dict(scene=name,source=source,exit_extension_m=extra_length,
                bend_heading_deg=float(np.degrees(np.arctan2(heading[1],heading[0]))),
                feature_span_m=list(np.fromstring(root.find("custom/numeric[@name='feature_span']").get('data'),sep=' ')),
                route_end_m=float(np.linalg.norm(np.diff(points[:,:3],axis=0),axis=1).sum()))


if __name__=='__main__':
    # Small increase and return, followed by a long exit for >=3 rolling cycles.
    x=np.r_[[-1.2,-.15],np.linspace(-.15,.15,16)[1:],[.45],np.linspace(.45,.75,16)[1:],[2.]]
    radii=np.interp(x,[-1.2,-.15,.15,.45,.75,2.],[.0565,.0565,.06215,.06215,.0565,.0565])
    results=[build('small_taper_return',np.column_stack([x,np.zeros(len(x)),np.zeros(len(x)),radii]),[1.05,1.95])]
    # True 90° bend; gradual taper happens in the bend, not an abrupt step.
    x=np.array([-1.2,-.2]);points=[np.array([a,0,0,.0565]) for a in x]
    for a in np.linspace(0,np.pi/2,25)[1:]:
        points.append(np.array([-.2+.3*np.sin(a),.3*(1-np.cos(a)),0,.0565+.025*a/(np.pi/2)]))
    points.extend([np.array([.1,y,0,.0815]) for y in (.8,1.4,2.0)])
    results.append(build('taper_true90',np.asarray(points),[1.,1.+.3*np.pi/2]))
    for source,name in [('bend90','bend90_section_exit'),('elbow90','elbow90_section_exit'),
                        ('taper_bend90','taper_bend45_section_exit')]:
        results.append(extend_exit(source,name))
    (ROOT/'mujoco_model/fixed_section_scene_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results))
