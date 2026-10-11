"""Reuse the reference robot and pipe collision geometry; horizontal Step1 scenes."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {'straight': 'Snake_Robot_o20_single_horizontal_pipe_long.xml',
           'bend90': 'Snake_Robot_o20_Pipe_bend90.xml',
           'elbow90': 'Snake_Robot_o20_Pipe_elbow90_horizontal.xml'}
ROTATION = Rotation.from_quat([.5, .5, .5, .5]).as_matrix()


def build(name):
    tree = ET.parse(ROOT / 'mujoco_model/reference' / SOURCES[name])
    root = tree.getroot()
    root.set('model', 'rolling-arc-' + name)
    root.find('option').set('integrator', 'implicitfast')
    world = root.find('worldbody')
    for site in list(world.findall('site')):
        world.remove(site)
    pipe = world.find("body[@name='pipe_network']")
    original = np.array([float(x) for x in root.find("custom/numeric[@name='pipe_centerline']").get('data').split()]).reshape(-1, 4)
    if name == 'bend90':
        pipe.set('quat', '.5 .5 .5 .5')
        pipe.set('pos', '-1.2 0 1.2')
        route = original.copy()
        route[:, :3] = original[:, :3] @ ROTATION.T + [-1.2, 0, 1.2]
        feature = [1.0, 1.0 + .3 * np.pi / 2]
    elif name == 'elbow90':
        pipe.set('pos', '-1.1 0 1.2')
        route = original.copy()
        route[:, :3] += [-1.1, 0, .2]
        # Miter collision geometry spans 0.95–1.25 m in source route coordinates.
        feature = [.95, 1.25]
    else:
        # Use only the original straight collision geom; discard unused elbow meshes.
        for geom in list(pipe):
            pipe.remove(geom)
        ET.SubElement(pipe, 'geom', name='pipe_straight', type='cylinder', size='.0565',
                      fromto='0 0 -1.2 0 0 1.8', friction='2 .005 .002', condim='6', material='pipe_rubber', group='1')
        pipe.set('pos', '0 0 1.2')
        route = np.column_stack([np.linspace(-1.2, 1.8, 61), np.zeros(61), np.full(61, 1.2), np.full(61, .0565)])
        feature = [0., 0.]
    used_meshes = {g.get('mesh') for g in root.iter('geom') if g.get('mesh')}
    for mesh in list(root.find('asset').findall('mesh')):
        if mesh.get('name') not in used_meshes:
            root.find('asset').remove(mesh)
    for joint in world.iter('joint'):
        joint.set('damping', '1.5')
    actuator = root.find('actuator')
    for item in list(actuator):
        joint = item.get('joint')
        actuator.remove(item)
        ET.SubElement(actuator, 'motor', name='motor_' + joint, joint=joint, gear='1',
                      ctrllimited='true', ctrlrange='-2.355 2.355', forcelimited='true', forcerange='-2.355 2.355')
    for tag in ('custom', 'keyframe'):
        item = root.find(tag)
        if item is not None:
            root.remove(item)
    custom = ET.SubElement(root, 'custom')
    ET.SubElement(custom, 'numeric', name='pipe_centerline', data=' '.join(f'{v:.10g}' for v in route.ravel()))
    ET.SubElement(custom, 'numeric', name='feature_span', data=' '.join(map(str, feature)))
    tree.write(ROOT / 'mujoco_model' / f'Snake_Robot_o20_Rolling_Arc_{name}.xml', encoding='utf-8', xml_declaration=True)
    return dict(scene=name, source=SOURCES[name], feature_span=feature, pipe_radius_m=.0565,
                robot_joints=20, shell_radius_m=.029, link_length_m=.0645)


if __name__ == '__main__':
    manifest = [build(name) for name in SOURCES]
    (ROOT / 'mujoco_model/scene_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Built:', ', '.join(SOURCES))
