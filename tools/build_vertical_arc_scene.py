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
SQUARE_HALF_WIDTH=2. * np.pi * PIPE_RADIUS / 8.
GENERATED_STRAIGHT_SCENES = (
    'straight_wide', 'variable', 'variable_return', 'square', 'round_to_square', 'round_to_ellipse', 'collar', 'twin',
)


def axis(u,radius=BEND_RADIUS,entry=BEND_HEIGHT):
    if u<0:return np.array([0.,0.,entry+u]),np.array([1.,0.,0.])
    if u<=np.pi*radius/2:
        a=u/radius
        return np.array([radius*(1-np.cos(a)),0.,entry+radius*np.sin(a)]),np.array([np.cos(a),0.,-np.sin(a)])
    return np.array([radius+u-np.pi*radius/2,0.,entry+radius]),np.array([0.,0.,-1.])


def add_vertical_section(assets, pipe, name, first, last, common):
    vertices = [(*point, z) for z, section in (first, last) for point in section]
    ET.SubElement(assets, 'mesh', name=name, vertex=' '.join(f'{v:.10g}' for v in np.ravel(vertices)))
    ET.SubElement(pipe, 'geom', name=name, type='mesh', mesh=name, **common)


def add_short_elbow_sdf(root, assets, pipe, radius, entry, leg, common, mesh_path):
    """One closed swept surface spanning the elbow and its two tangent legs."""
    extension = root.find('extension')
    if extension is None:
        extension = ET.SubElement(root, 'extension')
    plugin = ET.SubElement(extension, 'plugin', plugin='mujoco.sdf.sdflib')
    instance = ET.SubElement(plugin, 'instance', name='elbow_sdf')
    ET.SubElement(instance, 'config', key='aabb', value='0')
    end = radius * np.pi / 2.
    samples = np.r_[-leg, np.linspace(0., end, 49), end + leg]
    theta = np.arange(96) * 2. * np.pi / 96
    points = []
    for u in samples:
        center, normal = axis(u, radius, entry)
        points.extend(center + PIPE_RADIUS * (np.cos(theta)[:, None] * normal +
                                              np.sin(theta)[:, None] * [0., 1., 0.]))
    faces = []
    count = len(theta)
    for start in range(0, (len(samples) - 1) * count, count):
        for j in range(count):
            a, b = start + j, start + (j + 1) % count
            faces.extend(((a, b, b + count), (a, b + count, a + count)))
    last = (len(samples) - 1) * count
    for j in range(1, count - 1):
        faces.extend(((0, j + 1, j), (last, last + j, last + j + 1)))
    # MuJoCo 3.2.7 uses the file-backed mesh path for this SDF plugin.
    with mesh_path.open('w', encoding='ascii') as stream:
        for x, y, z in points:
            stream.write(f'v {x:.10g} {y:.10g} {z:.10g}\n')
        for a, b, c in faces:
            stream.write(f'f {a + 1} {b + 1} {c + 1}\n')
    mesh = ET.SubElement(assets, 'mesh', name='pipe_bend_sdf', file=mesh_path.name)
    ET.SubElement(mesh, 'plugin', instance='elbow_sdf')
    geom = ET.SubElement(pipe, 'geom', name='pipe_bend_sdf', type='sdf', mesh='pipe_bend_sdf', **common)
    ET.SubElement(geom, 'plugin', instance='elbow_sdf')


def build(scene, output=None, *, taper_length_m=.15, upper_diameter_m=.140, lower_diameter_m=2.*PIPE_RADIUS,
          outlet_height_m=4.,
          embed_return_outlet=False, short_bend_radius_m=SHORT_BEND_RADIUS, short_elbow_sdf=False):
    if short_elbow_sdf and scene != 'short':
        raise ValueError('The continuous elbow collision surface requires the short bend scene')
    path=Path(output) if output is not None else ROOT/f'mujoco_model/Snake_Robot_o18_Vertical_ARC_{scene}.xml'
    if scene == 'short' and not PIPE_RADIUS < short_bend_radius_m < BEND_HEIGHT:
        raise ValueError('The bend centreline radius must exceed the pipe radius and fit above the inlet')
    if lower_diameter_m != 2.*PIPE_RADIUS and scene not in ('variable', 'variable_return'):
        raise ValueError('A custom inlet diameter requires a variable-diameter scene')
    if embed_return_outlet and scene != 'variable_return':
        raise ValueError('Embedding the outlet requires an expanded section followed by a diameter return')
    if scene in ('variable', 'variable_return'):
        if not 0. < lower_diameter_m < upper_diameter_m < np.inf:
            raise ValueError('Variable-diameter scenes require finite, positive and increasing diameters')
        feature_exit = 1.9 + taper_length_m
        if scene == 'variable_return':
            feature_exit += .30 + taper_length_m
        if not np.isfinite(outlet_height_m) or outlet_height_m <= feature_exit:
            raise ValueError('The outlet must extend above the last diameter transition')
    elif scene == 'round_to_ellipse':
        if not np.isfinite(taper_length_m) or not 0. < taper_length_m:
            raise ValueError('The circle-to-ellipse transition length must be finite and positive')
        if not np.isfinite(outlet_height_m) or outlet_height_m <= 1.9 + taper_length_m:
            raise ValueError('The outlet must extend above the ellipse transition')
    elif outlet_height_m != 4.:
        raise ValueError('A custom outlet height requires a variable-diameter or circle-to-ellipse scene')
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
        radius=short_bend_radius_m if scene=='short' else BEND_RADIUS
        entry=BEND_HEIGHT+SHARP_LEG/2-radius if scene=='short' else BEND_HEIGHT
        leg = .18 if short_elbow_sdf else 0.
        pipe.find('geom').set('fromto',f'0 0 -.2 0 0 {entry-leg}')
        if short_elbow_sdf:
            add_short_elbow_sdf(root, assets, pipe, radius, entry, leg, common, path.with_suffix('.obj'))
        else:
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
                      fromto=f'{radius+leg} 0 {entry+radius} 2.3 0 {entry+radius}',**common)
        u=np.r_[np.linspace(-entry-.2,0.,41),np.linspace(0,np.pi*radius/2,49)[1:],
                np.linspace(np.pi*radius/2,np.pi*radius/2+2.,25)[1:]]
        route=np.array([np.r_[axis(x,radius,entry)[0],PIPE_RADIUS] for x in u])
    elif scene == 'straight_wide':
        upper_radius = .070
        cylinder = pipe.find('geom')
        cylinder.set('size', str(upper_radius))
        cylinder.set('fromto', '0 0 -.2 0 0 4')
        route = np.array([[0.,0.,-.2,upper_radius], [0.,0.,4.,upper_radius]])
    elif scene == 'round_to_ellipse':
        start, end = 1.9, 1.9 + taper_length_m
        theta = np.arange(96) * 2. * np.pi / 96
        unit = np.column_stack([np.cos(theta), np.sin(theta)])
        circle_section = PIPE_RADIUS * unit
        ellipse_ratio = np.array([1.30, 1.])
        # Scale the ellipse to match the perimeter of the discretized circular inlet.
        ring = unit * ellipse_ratio
        scale = (np.linalg.norm(np.roll(circle_section, 1, axis=0) - circle_section, axis=1).sum() /
                 np.linalg.norm(np.roll(ring, 1, axis=0) - ring, axis=1).sum())
        ellipse_axes = scale * ellipse_ratio
        ellipse_section = unit * ellipse_axes
        # Matching vertices on both sides of each seam give a continuous exterior.
        pipe.remove(pipe.find('geom'))
        add_vertical_section(assets, pipe, 'pipe_vertical', (-.2, circle_section), (start, circle_section), common)
        add_vertical_section(assets, pipe, 'pipe_taper', (start, circle_section), (end, ellipse_section), common)
        add_vertical_section(assets, pipe, 'pipe_upper', (end, ellipse_section),
                             (outlet_height_m, ellipse_section), common)
        route = np.array([[0., 0., z, PIPE_RADIUS] for z in (-.2, start, end, outlet_height_m)])
    elif scene == 'twin':
        # A scaled analogue of the published 200 -> 100/150 mm specimen.
        # The spacing and transition length are declared simulation geometry.
        start, end = 1.9, 2.2
        pipe.find('geom').set('fromto', f'0 0 -.2 0 0 {start}')
        radii = np.array([.0275, .04125])
        separation = radii.sum() + .008
        small_center = -.5 * (separation + radii[1] - radii[0])
        centers = (small_center, small_center + separation)
        theta = np.arange(64) * 2 * np.pi / 64
        directions = np.column_stack([np.cos(theta), np.sin(theta)])
        planes = np.linspace(start, end, 5)
        blend = (planes - start) / (end - start)
        blend = (blend**2 * (3. - 2. * blend))[:, None]
        plane_centers = blend * centers
        plane_radii = PIPE_RADIUS + blend * (radii - PIPE_RADIUS)
        for branch, label in enumerate(('small', 'large')):
            sections = [radius * directions + [x, 0.] for x, radius in zip(plane_centers[:, branch], plane_radii[:, branch])]
            for index, (z0, z1) in enumerate(zip(planes[:-1], planes[1:])):
                add_vertical_section(assets, pipe, f'pipe_twin_{label}_{index}',
                                     (z0, sections[index]), (z1, sections[index + 1]), common)
            final_x, final_r = centers[branch], radii[branch]
            ET.SubElement(pipe, 'geom', name=f'pipe_twin_{label}', type='cylinder', size=str(final_r),
                          fromto=f'{final_x} 0 {end} {final_x} 0 4', **common)
        outer_radius = centers[1] + radii[1]
        route = np.array([[0., 0., -.2, PIPE_RADIUS], [0., 0., start, PIPE_RADIUS],
                          [0., 0., end, outer_radius], [0., 0., 4., outer_radius]])
    elif scene in ('variable', 'variable_return', 'square', 'round_to_square', 'collar'):
        start, upper_radius = 1.9, upper_diameter_m * .5
        lower_radius = lower_diameter_m * .5
        if scene == 'collar':
            taper_length_m, upper_radius = .015, .065
        end = start + taper_length_m
        cylinder = pipe.find('geom')
        if scene == 'square':
            pipe.remove(cylinder)
            ET.SubElement(pipe, 'geom', name='pipe_vertical', type='box',
                          size=f'{SQUARE_HALF_WIDTH} {SQUARE_HALF_WIDTH} 2.1', pos='0 0 1.9', **common)
            route = np.array([[0.,0.,-.2,PIPE_RADIUS], [0.,0.,4.,PIPE_RADIUS]])
        else:
            cylinder.set('fromto', f'0 0 -.2 0 0 {start}')
            if lower_radius != PIPE_RADIUS:
                cylinder.set('size', str(lower_radius))
            theta = np.arange(64) * 2 * np.pi / 64
            directions = np.column_stack([np.cos(theta), np.sin(theta)])
            if scene in ('variable', 'variable_return', 'collar'):
                section = upper_radius * directions
                plateau_end = end + .060 if scene == 'collar' else end + .30 if scene == 'variable_return' else outlet_height_m
                ET.SubElement(pipe, 'geom', name='pipe_upper', type='cylinder', size=str(upper_radius),
                              fromto=f'0 0 {end} 0 0 {plateau_end}', **common)
            else:
                end = 2.2
                upper_radius = PIPE_RADIUS  # perimeter-equivalent, only for route metadata
                section = SQUARE_HALF_WIDTH * np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
                ET.SubElement(pipe, 'geom', name='pipe_upper', type='box',
                              size=f'{SQUARE_HALF_WIDTH} {SQUARE_HALF_WIDTH} {(4.-end)/2.}',
                              pos=f'0 0 {(4.+end)/2.}', **common)
            add_vertical_section(assets, pipe, 'pipe_taper', (start, lower_radius * directions), (end, section), common)
            route = [(-.2, lower_radius), (start, lower_radius), (end, upper_radius)]
            if scene in ('variable_return', 'collar'):
                end = plateau_end + taper_length_m
                add_vertical_section(assets, pipe, 'pipe_return', (plateau_end, section),
                                     (end, lower_radius * directions), common)
                # The cylinder's lower cap can sit inside the taper, where the
                # solid pipe already covers it. This removes the exposed cap at
                # the join while retaining the same exterior radius profile.
                outlet_start = plateau_end if embed_return_outlet else end
                ET.SubElement(pipe, 'geom', name='pipe_exit', type='cylinder', size=str(lower_radius),
                              fromto=f'0 0 {outlet_start} 0 0 {outlet_height_m}', **common)
                route += [(plateau_end, upper_radius), (end, lower_radius), (outlet_height_m, lower_radius)]
            else:
                route += [(outlet_height_m, upper_radius)]
            route = np.array([[0.,0.,z,r] for z, r in route])
    else:
        pipe.find('geom').set('fromto','0 0 -.2 0 0 4')
        route=np.array([[0.,0.,z,.055] for z in np.linspace(-.2,4.,85)])
    custom=root.find('custom')
    for c in list(custom):custom.remove(c)
    ET.SubElement(custom,'numeric',name='pipe_centerline',data=' '.join(map(str,route.ravel())))
    if scene == 'round_to_ellipse':
        ET.SubElement(custom, 'numeric', name='pipe_ellipse_semiaxes', data=' '.join(map(str, ellipse_axes)))
    if scene == 'twin':
        sections = np.column_stack((planes, plane_centers, plane_radii))
        ET.SubElement(custom, 'numeric', name='pipe_twin_cross_sections', data=' '.join(map(str, sections.ravel())))
    if scene=='acute':
        feature=(entry+.2,entry+.2+2*SHARP_LEG)
    elif scene in ('elbow','short'):
        feature=(entry+.2,entry+.2+np.pi*radius/2)
    elif scene in ('variable', 'variable_return', 'round_to_square', 'round_to_ellipse', 'collar', 'twin'):
        feature=(start+.2,end+.2)
    elif scene in ('square', 'straight_wide'):
        feature=(0.,0.)
    else:
        feature=(2,2+np.pi*BEND_RADIUS/2)
    ET.SubElement(custom,'numeric',name='feature_span',data=' '.join(map(str,feature)))
    if short_elbow_sdf:
        ET.SubElement(custom, 'numeric', name='bend_planes', data=f'{entry:.10g} {radius:.10g}')
    ET.SubElement(custom,'numeric',name='robot_lengths',data=' '.join(map(str,LENGTHS)))
    root.find('statistic').set('center','0 0 1.6');root.find('statistic').set('extent','2')
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    return str(path if output is not None else path.relative_to(ROOT))


if __name__=='__main__':
    paths=[build(name) for name in ('straight','elbow','acute','short')]
    print(json.dumps(dict(models=paths,lengths_m=LENGTHS.tolist(),pipe_diameter_m=.11,bend_radius_m=.3)))
