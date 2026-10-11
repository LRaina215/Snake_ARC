"""Reuse the reference 20-joint tapered-pipe collision assets with Step1 robot."""
from pathlib import Path
import copy
import json
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'Snake_Robot_Gait/mujoco_model'
SOURCES = {'taper_straight': 'Snake_Robot_o20_Pipe_taper_horizontal.xml',
           'taper_bend90': 'Snake_Robot_o20_Pipe_taper_bend.xml'}


def build(name, source):
    local = ROOT/'mujoco_model/reference'/source
    original = ET.parse(local if local.exists() else REFERENCE/source).getroot()
    # Save the original as provenance; generated XML is self-contained.
    ET.ElementTree(original).write(ROOT/'mujoco_model/reference'/source, encoding='utf-8', xml_declaration=True)
    root = ET.parse(ROOT/'mujoco_model/Snake_Robot_o20_Rolling_Arc_straight.xml').getroot()
    root.set('model', 'rolling-arc-'+name)
    asset = root.find('asset')
    for mesh in original.findall('asset/mesh'):
        if mesh.get('file') is None:
            asset.append(copy.deepcopy(mesh))
    world = root.find('worldbody')
    world.remove(world.find("body[@name='pipe_network']"))
    pipe = copy.deepcopy(original.find("worldbody/body[@name='pipe_network']"))
    pipe.set('quat', '.5 .5 .5 .5')
    pipe.set('pos', '-1.1 0 1.2')
    world.insert(2, pipe)
    route = np.fromstring(original.find("custom/numeric[@name='pipe_centerline']").get('data'),sep=' ').reshape(-1,4)
    if name == 'taper_straight':
        # Source metadata is already in world x even though geoms use local z.
        route[:,:3] += [-1.1,0,.2]
    else:
        route[:,:3] = route[:,:3] @ np.array([[0,0,1],[1,0,0],[0,1,0]]).T + [-1.1,0,1.2]
    custom = root.find('custom')
    custom.find("numeric[@name='pipe_centerline']").set('data',' '.join(f'{v:.10g}' for v in route.ravel()))
    feature = original.find("custom/numeric[@name='feature_span']").get('data')
    custom.find("numeric[@name='feature_span']").set('data', feature)
    ET.ElementTree(root).write(ROOT/'mujoco_model'/f'Snake_Robot_o20_Rolling_Arc_{name}.xml',encoding='utf-8',xml_declaration=True)
    return dict(scene=name, source=source, radii_m=[.0565,.0815], feature_span_m=feature)


if __name__ == '__main__':
    results=[build(name,source) for name,source in SOURCES.items()]
    (ROOT/'mujoco_model/step2_scene_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(results)
