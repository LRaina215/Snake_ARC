"""Movie and representative/worst frames from archives; zero mj_step calls."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import gzip
import json
import numpy as np
import mujoco
import glfw
import cv2
from PIL import Image, ImageDraw
from snake_control.simulation import ROOT, backbone


def load(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


def manual_panel(path, rows):
    """Verify the existing manual HUD using an archived pose/observation."""
    from codes.Rolling_Arc_Debug_Mujoco import DebugSession, DebugViewer
    session, viewer = DebugSession(), None
    try:
        row = rows[-1]
        sim, c = session.sim, session.controller
        sim.data.qpos[:], sim.data.qvel[:], sim.data.time = row['actual_qpos'], row['actual_qvel'], row['time_s']
        mujoco.mj_forward(sim.model, sim.data)
        active = np.flatnonzero(row['enabled'])+1
        c.first, c.last = int(active[0]), int(active[-1])
        c.enabled, c.targets = row['enabled'], np.degrees(row['target_rad']).tolist()
        session.bus.enabled[:] = c.enabled; session.bus.targets_deg[:] = c.targets
        c.mode, c.forming, c.settle_remaining = 'hold', False, 0.
        c.phase, session.elapsed = row['controller']['phase_rad'], row['time_s']
        c.select_gait('sliding')
        c.sliding.observation = row['controller']['observation']
        c.sliding.pending, c.sliding.phase_hold = row['controller']['pending'], row['controller']['phase_hold']
        session.physics_started = True
        session.initial_center = np.array(rows[0]['body_center_m'])
        session.colour_active()
        viewer = DebugViewer(session, visible=False)
        viewer.camera.lookat[:] = [backbone(sim.model, sim.data).mean(axis=0)[0], 0., .95]
        viewer.camera.distance, viewer.camera.azimuth, viewer.camera.elevation = 1.8, 125., -12.
        viewer.render(); viewer.render()
        rgb = np.empty((800, 1280, 3), np.uint8)
        mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, 1280, 800), viewer.context)
        Image.fromarray(np.flipud(rgb)).save(path/'manual_panel.png')
        print('Manual HUD rendered from saved pose/observation; no dynamics or physical hotkeys')
    finally:
        if viewer is not None:
            viewer.close()
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('--manual-panel-only', action='store_true')
    args = parser.parse_args()
    rows = load(args.path/'actual_trace.jsonl.gz')
    if args.manual_panel_only:
        manual_panel(args.path, rows)
        return
    evaluation = load(args.path/'evaluation.jsonl.gz')
    status = json.loads((args.path/'status.json').read_text(encoding='utf8'))
    scene_name = status.get('scene', 'straight')
    model = mujoco.MjModel.from_xml_path(str(ROOT/'mujoco_model'/f'Snake_Robot_o20_Rolling_Arc_{scene_name}.xml'))
    data = mujoco.MjData(model)
    if not glfw.init():
        raise RuntimeError('Cannot initialize archived-state renderer')
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    surface = glfw.create_window(960, 600, 'Saved actual states', None, None)
    if surface is None:
        glfw.terminate()
        raise RuntimeError('Cannot create hidden renderer')
    glfw.make_context_current(surface)
    scene = mujoco.MjvScene(model, maxgeom=5000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    option, camera = mujoco.MjvOption(), mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.8, 125., -12.
    viewport = mujoco.MjrRect(0, 0, 960, 600)
    rgb = np.empty((600, 960, 3), np.uint8)
    movie = ROOT/'validation/results'/args.path.parent.name/(args.path.name+'.mp4')
    movie.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*'mp4v'), 10, (960, 600))
    if not writer.isOpened():
        raise RuntimeError('Cannot create movie writer')
    frames = 0
    def render(row):
        data.qpos[:], data.qvel[:], data.time = row['actual_qpos'], row['actual_qvel'], row['time_s']
        mujoco.mj_forward(model, data)
        center = backbone(model, data).mean(axis=0)
        camera.lookat[:] = [center[0], center[1], .95]
        model.geom_rgba[model.geom_bodyid >= model.body('head').id] = [.50, .26, .70, 1]
        active = np.flatnonzero(row['enabled'])+1
        for j in active:
            bid = model.jnt_bodyid[model.joint(f'joint{j}').id]
            model.geom_rgba[model.geom_bodyid == bid] = [.10, .48, .85, 1]
        mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        for _ in range(2):
            mujoco.mjr_render(viewport, scene, context)
            o = row['controller']['observation']
            text = f"t={row['time_s']:.2f}s | enabled J{active[0]}-J{active[-1]} | {len(active)} joints\n"
            text += f"observer: {'settle' if o is None else o['state']} | handover hold={row['controller']['phase_hold']}"
            c = row['controller']
            text += f"\nphase={c['phase_rad']:.3f}rad | fault={c.get('fault') or c.get('resize_fault') or 'none'}"
            force = np.asarray(row.get('external_force_world', []))
            magnitude = float(np.linalg.norm(force[:, :3], axis=1).sum()) if force.size else 0.
            text += f"\nApplied force sum={magnitude:.1f}N | feedback={status.get('motor_feedback_period_s', .02):g}s"
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               viewport, 'Saved actual motion\nBlue: enabled  Purple: disabled', text, context)
            glfw.swap_buffers(surface)
        mujoco.mjr_readPixels(rgb, None, viewport, context)
        return np.flipud(rgb).copy()
    panels = []
    try:
        for row in rows[::5]:
            writer.write(cv2.cvtColor(render(row), cv2.COLOR_RGB2BGR)); frames += 1
        wanted = [0., 2.]
        complete = [e for e in status['events'] if e['kind'] in ('complete', 'resize_step', 'resize_complete')]
        if complete:
            for event in complete:
                switch = event['time_s']
                wanted += [switch-.02, switch+.02]
        else:
            wanted += [next((r['time_s'] for r in evaluation if (r.get('observation') or {}).get('direction')), rows[-1]['time_s'])]
        stimulus = status.get('stimulation', {})
        if stimulus.get('kind', 'none') != 'none':
            start, end = stimulus['interval_s']
            wanted += [start, (start+end)/2, end, end+2, 50.]
        wanted += [max(evaluation, key=lambda r: r.get('pipe_angular_span_deg', r.get('pipe_shape', {}).get('wrap_deg', 0.)))['time_s'],
                   max(evaluation, key=lambda r: r['body_geometry']['turn_deg'])['time_s'], rows[-1]['time_s']]
        for index, timestamp in enumerate(wanted):
            row = min(rows, key=lambda r: abs(r['time_s']-timestamp))
            picture = Image.fromarray(render(row))
            picture.save(args.path/f'actual_{index+1}.png')
            small = Image.new('RGB', (640, 426), 'white')
            ImageDraw.Draw(small).text((8, 6), f"Actual state {row['time_s']:.2f}s", fill='black')
            small.paste(picture.resize((640, 400)), (0, 26)); panels.append(small)
    finally:
        writer.release(); context.free(); glfw.destroy_window(surface); glfw.terminate()
    montage = Image.new('RGB', (1280, ((len(panels)+1)//2)*426), 'white')
    for i, panel in enumerate(panels):
        montage.paste(panel, ((i % 2)*640, (i//2)*426))
    montage.save(args.path/'actual_montage.png')
    (args.path/'render.json').write_text(json.dumps(dict(movie=str(movie), frames=frames, fps=10,
        full_time_range_s=[rows[0]['time_s'], rows[-1]['time_s']], snapshot_times_s=wanted,
        method='mj_forward saved actual qpos/qvel; no mj_step'), indent=2), encoding='utf8')
    print(movie)


if __name__ == '__main__':
    main()
