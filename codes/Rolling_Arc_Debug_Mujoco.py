"""手动调试版MuJoCo：直接运行，Ctrl热键与实机完全共用。

仅把实机总线替换为MuJoCo关节读写，使用真实动力学反馈而非目标回显。
修改下方场景参数；步态、顺应与键位修改Rolling_Arc_Real.py顶部。
自动研究实验仍使用原入口。本文件不打开真实串口。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import json
import queue
import threading
import time
import numpy as np
import mujoco
from codes import Rolling_Arc_Real as real
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation, ROOT, backbone


# ==================== 1. 手动测试参数 ====================
SCENE = 'straight'              # straight / bend90 / elbow90 / taper_true90 / elbow90_taper110_210
START_X_M = -.5                 # 弯管常用-.35，急弯常用-.25
SAVE_LOG = True
LOG_DIRECTORY = Path(__file__).resolve().parent / 'logs'
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 800


# ==================== 2. 关节接口：真实状态读角，显式使能+位置目标 ====================
class MujocoMotorBus:
    def __init__(self, sim):
        self.sim = sim
        self.enabled = np.zeros(real.NUM_JOINTS, dtype=bool)
        self.targets_deg = np.degrees(sim.data.qpos[sim.drive.qadr]).copy()
        sim.data.ctrl[:] = 0.
        mujoco.mj_forward(sim.model, sim.data)

    def read_angles(self, indices):
        measured = np.degrees(self.sim.data.qpos[self.sim.drive.qadr])
        return {i: float(measured[i]) for i in indices}

    def read_states(self):
        angles = self.read_angles(range(real.NUM_JOINTS))
        # 模型没有电流/温度/协议负载寄存器，不伪造成真实舵机的0反馈。
        return {i: dict(angle_deg=angles[i], load_raw=None, current_raw=None,
                        voltage_v=None, temperature_c=None, status_raw=None, moving_raw=None)
                for i in range(real.NUM_JOINTS)}

    def send_targets(self, angles, enabled):
        angles, enabled = np.asarray(angles, float), np.asarray(enabled, bool)
        if angles.shape != self.targets_deg.shape or enabled.shape != self.enabled.shape:
            raise ValueError('MuJoCo关节目标/使能长度错误')
        if not np.isfinite(angles).all():
            raise ValueError('MuJoCo关节目标必须有限')
        self.targets_deg[enabled] = angles[enabled]

    def set_enabled(self, enabled, verify=True):
        enabled = np.asarray(enabled, bool)
        if enabled.shape != self.enabled.shape:
            raise ValueError('MuJoCo使能掩码长度错误')
        self.enabled = enabled.copy()
        self.sim.data.ctrl[~enabled] = 0.
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def prepare_enabled(self, angles, enabled, configure_mask=None):
        configure = enabled if configure_mask is None else configure_mask
        self.send_targets(angles, configure)  # 与实机一样先预置实测角，再使能。
        self.set_enabled(enabled)

    def apply_drive(self):
        self.sim.drive.apply_targets(self.sim.data, np.radians(self.targets_deg), self.enabled)


class DebugSession:
    """两端共用控制器、命令分派和反馈周期；仅后端执行器不同。"""
    def __init__(self, scene=SCENE, start_x=START_X_M):
        real.validate_setup()
        p = replace(ArcParameters.load(ROOT/'configs/step1.json'),
                    joints=real.NUM_JOINTS, first_joint=real.FIRST_JOINT, last_joint=real.LAST_JOINT,
                    amplitude_deg=real.AMPLITUDE_DEG, omega_rad_s=real.OMEGA_RAD_S,
                    initial_phase_rad=real.INITIAL_PHASE_RAD, settle_s=real.SETTLE_S)
        self.sim = ArcSimulation(scene, p, start_x)
        lengths = np.linalg.norm(np.diff(backbone(self.sim.model, self.sim.data), axis=0), axis=1)
        if not np.allclose(lengths, real.LINK_LENGTHS_M, atol=1e-8):
            raise ValueError('当前模型骨架与实机代码LINK_LENGTHS_M不同，请先核对机械尺寸')
        steps = real.CONTROL_PERIOD_S / self.sim.model.opt.timestep
        if abs(steps-round(steps)) > 1e-8:
            raise ValueError('控制周期须是MuJoCo时间步长的整数倍')
        self.substeps = int(round(steps))
        self.bus = MujocoMotorBus(self.sim)
        self.controller = real.RollingArcController(self.bus)
        self.commands = queue.Queue()
        self.next_feedback, self.elapsed = 0., 0.
        self.start_wall = time.monotonic()
        self.physics_started = False
        self.colour_active()

    def advance(self):
        c = self.controller
        real.process_control_commands(c, self.commands, self.elapsed)
        if c.quit:
            return
        # 首次接管前只是初始姿态预览，避免读操作说明期间掉管；
        # 接管/释放后再不冻结物理，暂停、保持与失能都按真实动力学执行。
        self.physics_started |= c.touched_hardware
        if not self.physics_started:
            return
        before = (c.mode, c.forming, c.settle_remaining > 0)
        self.next_feedback = real.poll_control_feedback(c, self.elapsed, self.next_feedback)
        c.tick(real.CONTROL_PERIOD_S)
        if before != (c.mode, c.forming, c.settle_remaining > 0):
            c.show_status()
        for _ in range(self.substeps):
            self.bus.apply_drive()
            mujoco.mj_step(self.sim.model, self.sim.data)
        self.elapsed = float(self.sim.data.time)
        mujoco.mj_forward(self.sim.model, self.sim.data)
        if not np.isfinite(self.sim.data.qpos).all() or not np.isfinite(self.sim.data.qvel).all():
            raise RuntimeError('MuJoCo动力学状态不是有限值')
        self.colour_active()

    def colour_active(self):
        model = self.sim.model
        model.geom_rgba[list(self.sim.robot_geoms)] = [.50, .26, .70, 1.]
        for i in np.flatnonzero(self.bus.enabled):
            body = model.jnt_bodyid[model.actuator_trnid[i, 0]]
            model.geom_rgba[model.geom_bodyid == body] = [.10, .48, .85, 1.]

    def snapshot(self):
        data = self.sim.data
        return dict(time_s=float(data.time), wall_elapsed_s=time.monotonic()-self.start_wall,
                    execution='mujoco_debug', scene=self.sim.scene,
                    physics_started=self.physics_started, **self.controller.snapshot(),
                    actual_qpos=data.qpos.tolist(), actual_qvel=data.qvel.tolist(),
                    actuator_force_nm=data.actuator_force.tolist(),
                    motor_feedback='continuous_model_joint_angles')

    def close(self):
        self.controller.release()
        self.bus.apply_drive()
        mujoco.mj_forward(self.sim.model, self.sim.data)


# ==================== 3. 三维查看器：只有鼠标相机，没有MuJoCo原生键盘菜单 ====================
class DebugViewer:
    """使用MuJoCo渲染API，避开默认viewer的键盘快捷键；热键由实机监听器处理。"""
    def __init__(self, session, visible=True):
        import glfw
        self.glfw, self.session = glfw, session
        if not glfw.init():
            raise RuntimeError('GLFW初始化失败')
        glfw.window_hint(glfw.VISIBLE, int(visible))
        self.window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT,
                                        'Rolling Arc | MuJoCo manual debug | Ctrl hotkeys', None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('无法创建MuJoCo调试窗口')
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.camera, self.options = mujoco.MjvCamera(), mujoco.MjvOption()
        self.camera.lookat[:] = backbone(session.sim.model, session.sim.data).mean(axis=0)
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.1, 135, -20
        self.scene = mujoco.MjvScene(session.sim.model, maxgeom=4000)
        self.context = mujoco.MjrContext(session.sim.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.last_mouse = glfw.get_cursor_pos(self.window)
        glfw.set_cursor_pos_callback(self.window, self.on_mouse)
        glfw.set_scroll_callback(self.window, self.on_scroll)
        # 不注册单字母/空格回调，也不调用mujoco.viewer的原生按键处理。

    def on_mouse(self, window, x, y):
        glfw = self.glfw
        dx, dy = x-self.last_mouse[0], y-self.last_mouse[1]
        self.last_mouse = x, y
        height = max(1, glfw.get_window_size(window)[1])
        shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        mujoco.mjv_moveCamera(self.session.sim.model, action, dx/height, dy/height, self.scene, self.camera)

    def on_scroll(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(self.session.sim.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                             0., -.05*yoffset, self.scene, self.camera)

    def running(self):
        return not self.glfw.window_should_close(self.window)

    def render(self):
        width, height = self.glfw.get_framebuffer_size(self.window)
        if not width or not height:
            self.glfw.poll_events()
            return
        sim, c = self.session.sim, self.session.controller
        mujoco.mjv_updateScene(sim.model, sim.data, self.options, None, self.camera,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, self.scene, self.context)
        left = 'Gait\nMode\nWindow\nPhase\nPhysics\nTime'
        right = (f'{"fixed section" if c.compliance else "basic arc"}\n{c.mode}'
                 f'\nJ{c.first}-J{c.last}\n{c.phase:.3f} rad'
                 f'\n{"running" if self.session.physics_started else "initial preview; Ctrl+U"}'
                 f'\n{sim.data.time:.2f} s')
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          viewport, left, right, self.context)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                          viewport, 'Ctrl+U form | Ctrl+J/H step | Ctrl+G run/pause | Ctrl+E compliance\nCtrl+X release | Ctrl+Q quit',
                          'Ctrl+Space pause | Ctrl+N freeze targets\nCtrl+T reverse | Ctrl+arrows tune', self.context)
        self.glfw.swap_buffers(self.window)
        self.glfw.poll_events()

    def close(self):
        self.glfw.make_context_current(self.window)
        self.context.free()
        self.scene = None  # MjvScene由Python绑定析构释放，没有free()方法。
        self.glfw.destroy_window(self.window)
        self.glfw.terminate()


# ==================== 4. 直接运行；与实机共用Ctrl热键和控制台 ====================
def main():
    from pynput import keyboard
    session = DebugSession()
    viewer = listener = console = log_file = None
    stop = threading.Event()
    try:
        viewer = DebugViewer(session)
        listener = real.global_key_listener(session.commands, keyboard)
        console = threading.Thread(target=real.console_listener, args=(session.commands, stop), daemon=True)
        console.start()
        if SAVE_LOG:
            path, log_file = real.open_run_log(LOG_DIRECTORY, prefix='mujoco_debug')
            print(f'MuJoCo调试日志：{path}', flush=True)
        print(real.HELP, flush=True)
        print('启动为静态预览；Ctrl+U接管或Ctrl+X释放后开始物理，暂停相位不暂停物理。蓝色使能，紫色失能。', flush=True)
        next_log = 0.
        while viewer.running() and not session.controller.quit:
            started = time.monotonic()
            session.advance()
            viewer.render()
            wall_elapsed = time.monotonic()-session.start_wall
            if log_file and wall_elapsed >= next_log:
                log_file.write(json.dumps(session.snapshot(), ensure_ascii=False)+'\n')
                log_file.flush()
                next_log = wall_elapsed+.1
            time.sleep(max(0., real.CONTROL_PERIOD_S-(time.monotonic()-started)))
    finally:
        stop.set()
        try:
            session.close()
            if log_file:
                log_file.write(json.dumps(session.snapshot(), ensure_ascii=False)+'\n')
        finally:
            if listener:
                listener.stop()
                listener.join(timeout=1.)
            if console:
                console.join(timeout=.1)
            if log_file:
                log_file.close()
            if viewer:
                viewer.close()


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            raise ValueError('手动调试版直接运行即可，场景等参数在文件顶部修改。')
        main()
    except KeyboardInterrupt:
        print('用户中止。')
    except Exception as exc:
        print(f'MuJoCo手动调试停止：{exc}', flush=True)
        raise SystemExit(1)
