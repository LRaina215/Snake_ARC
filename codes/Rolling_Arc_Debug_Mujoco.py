"""手动调试版MuJoCo：基本Ctrl操作共用，仿真额外支持三模式和物理扰动。

基础与固定顺应共用实机控制器；滑动模式默认仅角度反馈，原接触版本可选。
MuJoCo关节接口提供真实动力学反馈而非目标回显。
修改下方仿真场景、窗口和幅值参数；基本算法与键位共用实机代码。
自动研究实验仍使用原入口。本文件不打开真实串口。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import copy
import math
import json
import queue
import threading
import time
import numpy as np
import mujoco
from codes import Rolling_Arc_Real as real
from snake_control.rolling_arc import ArcParameters
from snake_control.simulation import ArcSimulation, ROOT, backbone
from snake_control.sliding_section_compliance import SlidingSectionCompliance, WindowOptions
from snake_control.sliding_section_simulation import upward_support
MotorFeedbackSliding = real.MotorFeedbackSliding


# ==================== 1. 手动测试参数 ====================
SCENE = 'straight'              # straight / bend90 / elbow90 / taper_true90 / elbow90_taper110_210 / length_taper_return
START_X_M = -.5                 # 弯管常用-.35，急弯常用-.25
SIM_FIRST_JOINT = real.FIRST_JOINT  # 本轮偏置搭管实验设5 / 10，不修改实机文件
SIM_LAST_JOINT = real.LAST_JOINT
SIM_AMPLITUDE_DEG = 60.0         # 与本轮冻结基准一致；独立于用户的实机幅值调试
SIM_OMEGA_RAD_S = .6
SLIDING_FEEDBACK = 'motor_angles'  # motor_angles：自动位置+数量；contact：历史固定宽度接触版本
SAVE_LOG = True
SHOW_SHAPE_PREVIEW = True       # 与实机相同的关节重建截面预览，无世界位姿输入
LOG_DIRECTORY = Path(__file__).resolve().parent / 'logs'
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 800
PUSH_FORCE_N = 8.0              # Ctrl+A/D按住拉动左/右悬垂端、松开停止；总力，单位N
PUSH_FORCE_STEP_N = 2.0         # Ctrl+[ / ] 调整扰动力；不会直接移动窗口或qpos
PUSH_WORLD_DIRECTION = np.array([0., 1., 0.])
PUSH_DOWNWARD_RATIO = 1.0        # 水平向外+向下拉；0时只水平拉，均归一化为上面的总力

SIM_HELP = '''MuJoCo新增操作：
  Ctrl+1 / 2 / 3：基础圆弧 / 固定截面顺应 / 滑动窗口截面顺应
  Ctrl+A / D：按住拉动左(-Y) / 右(+Y)悬垂端；松开字母或Ctrl即停止
  Ctrl+[ / ]：扰动总力减小 / 增大；Ctrl+W：自动移窗与扩缩开/关（仅滑动模式）
  Ctrl+R：明确结束当前试验，重建初始预览；不是连续运动中的位姿修正
窗口外头/尾关节数是使能窗口的分布，不等于实际接触/悬垂关节数。
默认滑动模式仅用实测角和已知相位；接触力只记录用于评价。
DIY调参策略与实机相同；持续侧拉可能失去方向观测。
'''


# ==================== 2. 仿真扩展键位与三模式控制器 ====================
def decode_sim_hotkey(vk, control_down):
    if not control_down:
        return None
    extra = {49: 'gait_basic', 50: 'gait_fixed', 51: 'gait_sliding',
             97: 'gait_basic', 98: 'gait_fixed', 99: 'gait_sliding',
             65: 'push_left', 68: 'push_right', 82: 'reset_preview',
             87: 'window_auto', 219: 'push_weaker', 221: 'push_stronger'}
    return extra.get(vk) or real.decode_hotkey(vk, control_down)


def decode_sim_release(vk, is_control):
    if is_control:
        return 'push_stop'
    return {65: 'push_left_stop', 68: 'push_right_stop'}.get(vk)


class DebugArcController(real.RollingArcController):
    """三种仅角度模式共用实机控制；保留历史接触控制供已有实验复现。"""
    def __init__(self, bus, session):
        super().__init__(bus)
        self.session = session
        self.first, self.last = session.sim.p.first_joint, session.sim.p.last_joint
        self.length_normalized = self.last-self.first+1 != 6
        self.amplitude, self.omega = session.sim.p.amplitude_deg, session.sim.p.omega_rad_s
        self.feeding_origin = (self.first, self.last)
        self.sliding = None
        self.pending_gait = None
        self.sliding_fault = False
        self.help_text = real.HELP+'\n'+SIM_HELP

    def select_gait(self, kind):
        if kind=='feeding':
            self.session.stop_push()
            return super().select_gait(kind)
        if self.feeding_selected:
            super().select_gait('basic')
        if SLIDING_FEEDBACK == 'motor_angles':
            super().select_gait(kind)
            self.session.record_event('gait_change', to=kind)
            return
        if kind not in ('basic', 'fixed', 'sliding'):
            raise ValueError('未知模式')
        if kind == 'basic':
            self.pending_gait = None
            if self.compliance is not None:
                self.session.record_event('gait_change', to=kind)
                self.sliding = None
                self.sliding_fault = False
                super().toggle_compliance()  # 保留相位/目标，限速恢复当前窗口基础波。
            return
        if self.mode == 'released' or self.forming or self.settle_remaining > 0:
            self.pending_gait = kind
            print(f'已选择{kind}，形成与静置完成后尝试接入。', flush=True)
            return
        if kind == 'fixed':
            if self.sliding is None:
                if self.compliance is None:
                    super().toggle_compliance()
                    self.pending_gait = None
                    self.session.record_event('gait_change', to=kind)
                return
            # 先检查，再替换。失败时保留原滑动模式、窗口与目标。
            measured = np.radians(list(self.bus.read_angles(range(real.NUM_JOINTS)).values()))
            valid, _ = self.check_actual_section(np.degrees(measured))
            if not valid:
                raise ValueError('实际形态不满足固定截面条件，未切换')
            reference = real.SectionReference(self.first, self.last, self.amplitude)
            core = real.FixedSectionCompliance(reference, real.LINK_LENGTHS_M, self.options)
            nominal = measured.copy()
            nominal[core.enabled] = reference.targets(self.phase)[core.enabled]
            if not core._check_goal(nominal)[0]:
                raise ValueError('当前基础参考不满足开放约束，未切换')
            core.targets = np.radians(self.targets)
            core.next_geometry_s = self.section_clock_s
            self.sliding, self.compliance = None, core
            self.refresh_compliance_feedback()
            self.section_checked_sequence = -1
            self.sliding_fault = False
        else:
            if self.sliding is not None and not self.sliding_fault:
                return
            if self.last-self.first+1 != 6:
                raise ValueError('当前滑动核心要求连续6关节')
            measured = np.radians(list(self.bus.read_angles(range(real.NUM_JOINTS)).values()))
            valid, _ = self.check_actual_section(np.degrees(measured))
            if not valid:
                raise ValueError('实际形态不满足开放圆弧条件，未切换')
            p = replace(self.session.sim.p, first_joint=self.first, last_joint=self.last,
                        amplitude_deg=self.amplitude, omega_rad_s=self.omega, direction=1)
            if SLIDING_FEEDBACK == 'contact':
                slider = SlidingSectionCompliance(p, real.LINK_LENGTHS_M, self.options, WindowOptions())
                support = upward_support(self.session.sim)
                slider.window = (self.first, self.last)
                slider.weights = np.asarray(self.enabled, float)
                slider.support[:] = support
                slider.raw_support[:] = support
                if support.sum() < slider.options.minimum_support_n or slider._coverage(slider.window) < slider.options.minimum_support_coverage:
                    raise ValueError('当前窗口承重或覆盖不足，未切换到滑动模式')
                slider._make_core(slider.window, measured)
                slider.last_measured = measured.copy()
                slider._reset_phase_load(self.phase)
                reason = 'take over currently driven supported window'
            else:
                raise ValueError('SLIDING_FEEDBACK应为motor_angles或contact')
            slider.core.targets = np.radians(self.targets)
            slider.core.next_geometry_s = self.section_clock_s
            slider.events.append(dict(time_s=self.session.elapsed, kind='acquire',
                                      to=list(slider.window), reason=reason))
            if not isinstance(slider, MotorFeedbackSliding):
                slider.cooldown_until = self.session.elapsed+slider.options.cooldown_s
            self.sliding, self.compliance = slider, slider.core
            self.sliding_fault = False
        self.pending_gait = None
        self.section_block_reason = None
        self.freeze_targets = False
        self.session.record_event('gait_change', to=kind)

    def toggle_compliance(self):
        if SLIDING_FEEDBACK == 'motor_angles':
            return super().toggle_compliance()
        self.select_gait('fixed' if self.compliance is None else 'basic')

    def release(self, verify=True):
        self.sliding = None
        self.pending_gait = None
        self.sliding_fault = False
        self.session.stop_push()
        super().release(verify)

    def assign_parameters(self, command):
        if not real.DIY_PARAMETERS and self.sliding is not None and any(token.split('=')[0] in ('amplitude', 'window') for token in command.split()):
            raise ValueError('滑动模式使用半圆归一化预算；调幅值/窗口先切回基础模式')
        if not real.DIY_PARAMETERS and isinstance(self.sliding, MotorFeedbackSliding):
            for item in command.split():
                key, value = item.split('=', 1)
                if key == 'omega' and float(value) != .6:
                    raise ValueError('角度滑窗的相位基准只标定了omega=0.6')
        super().assign_parameters(command)
        if self.sliding is not None:
            self.sliding.section_options = self.options
            self.sliding.core.options = self.options

    def command(self, command):
        if not real.DIY_PARAMETERS and isinstance(self.sliding, MotorFeedbackSliding) and command.strip().lower() == 'reverse':
            raise ValueError('冻结角度基准未标定反向滚动；先切换基础/固定模式')
        command = command.strip().lower()
        if command.startswith('gait_'):
            self.select_gait(command[5:])
        elif command in ('push_left', 'push_right'):
            if not self.session.physics_started:
                raise ValueError('先Ctrl+U接管再做本体扰动')
            self.session.begin_push(-1 if command == 'push_left' else 1)
        elif command in ('push_left_stop', 'push_right_stop'):
            self.session.push_keys.discard(-1 if command == 'push_left_stop' else 1)
        elif command == 'push_stop':
            self.session.stop_push()
        elif command in ('push_weaker', 'push_stronger'):
            step = -PUSH_FORCE_STEP_N if command == 'push_weaker' else PUSH_FORCE_STEP_N
            self.session.push_force = max(0., self.session.push_force+step)
        elif command == 'reset_preview':
            self.session.reset_preview()
        elif command == 'window_auto':
            if self.sliding is None:
                raise ValueError('先切换到滑动模式')
            self.sliding.automatic = not self.sliding.automatic
            if not isinstance(self.sliding, MotorFeedbackSliding):
                self.sliding.pending = None
        elif command in ('quit', 'exit'):
            self.session.stop_push()
            super().command(command)
        else:
            super().command(command)

    def tick(self, dt):
        if self.sliding is None:
            super().tick(dt)
        elif self.mode != 'released':
            self.tick_sliding(min(dt, real.CONTROL_PERIOD_S))
        if self.pending_gait and self.mode != 'released' and not self.forming and self.settle_remaining <= 0:
            kind, self.pending_gait = self.pending_gait, None
            try:
                self.select_gait(kind)
            except ValueError as exc:
                print(f'模式接入未执行：{exc}；可再次按模式热键重试。', flush=True)

    def tick_sliding(self, dt):
        if isinstance(self.sliding, MotorFeedbackSliding):
            return self.tick_motor_sliding(dt)
        self.section_clock_s += dt
        self.refresh_compliance_feedback()  # 新实际角，每20ms采集，不能拿保持读数假装卸载速度为0。
        self.feedback_time_s = self.session.elapsed
        measured = np.radians(self.measured_angles())
        _, self.actual_section = self.check_actual_section(np.degrees(measured))
        if self.freeze_targets or self.sliding_fault:
            self.bus.send_targets(self.targets, self.enabled)
            return
        previous_phase, previous_pending = self.phase, self.pending_phase
        previous_targets = self.targets.copy()
        old = self.sliding
        old_core, old_window, old_weights = old.core, old.window, old.weights.copy()
        if not old.phase_hold:
            if self.mode == 'run':
                self.phase += self.direction*self.omega*dt
            elif self.mode == 'step':
                increment = max(-self.omega*dt, min(self.omega*dt, self.pending_phase))
                self.phase += increment
                self.pending_phase -= increment
        try:
            target, weights = old.update(measured, self.phase, upward_support(self.session.sim),
                                         self.session.elapsed, dt, self.session.sim.data.actuator_force.copy())
            enabled = weights > 1e-12
            if enabled.sum() != 6 or old.core.status != 'running' or not old.core._check_goal(target)[0]:
                raise ValueError('滑动模式发送目标或使能不满足开放约束')
            targets = np.degrees(target).tolist()
            for i in np.flatnonzero(enabled):
                real.angle_to_raw(i, targets[i])
        except (ValueError, np.linalg.LinAlgError) as exc:
            old.window, old.weights, old.core = old_window, old_weights, old_core
            old.core.targets = np.radians(previous_targets)
            old.preparation = old.transfer = None
            old.phase_hold = False
            self.compliance = old_core
            self.phase, self.pending_phase = previous_phase, previous_pending
            self.sliding_fault = True
            self.block_section(str(exc)+'，重新选择模式后才能继续')
            self.session.record_event('sliding_guard_reject', reason=str(exc))
            self.bus.send_targets(previous_targets, self.enabled)
            return
        if enabled.tolist() != self.enabled:
            entering = enabled & ~np.asarray(self.enabled, bool)
            self.bus.prepare_enabled(np.degrees(measured), enabled, configure_mask=entering)
            self.first, self.last = old.window
            self.enabled = enabled.tolist()
        self.compliance = old.core
        self.targets = targets
        self.bus.send_targets(targets, self.enabled)
        self.command_phase = self.phase
        if self.mode == 'step' and abs(self.pending_phase) < 1e-10:
            self.mode = 'hold'

    def tick_motor_sliding(self, dt):
        """仿真按实机200ms读角，调用同一个20ms目标/使能更新方法。"""
        if self.feedback_time_s is None or self.session.elapsed-self.feedback_time_s >= real.COMPLIANCE_FEEDBACK_PERIOD_S-1e-9:
            self.refresh_compliance_feedback()
        return super().tick_motor_sliding(dt)

    def refresh_compliance_feedback(self):
        super().refresh_compliance_feedback()
        self.feedback_time_s = self.session.elapsed

    def motor_sample_clock(self):
        return self.session.elapsed, self.feedback_time_s

    def change_parameters(self, amplitude=None, omega=None):
        if not real.DIY_PARAMETERS and isinstance(self.sliding, MotorFeedbackSliding) and (
                (amplitude is not None and amplitude != 60.) or (omega is not None and omega != .6)):
            raise ValueError('先切换基础/固定模式再调幅值或频率；角度滑窗使用冻结的60度/0.6基准')
        return super().change_parameters(amplitude, omega)

    def snapshot(self):
        row = super().snapshot()
        if self.sliding is not None:
            row['gait'] = 'sliding_section_compliance'
            if row['compliance'] is not None:
                row['compliance']['window_fixed'] = False
        row.update(pending_gait=self.pending_gait,
                   outside_window_head_joints=self.first-1,
                   outside_window_tail_joints=real.NUM_JOINTS-self.last,
                   sliding=None if self.sliding is None else copy.deepcopy(self.sliding.snapshot()),
                   sliding_events=[] if self.sliding is None else copy.deepcopy(self.sliding.events))
        if row['sliding'] is not None:
            row['sliding']['automatic'] = self.sliding.automatic
        return row

    def show_status(self, include_feedback=False):
        if self.sliding is None or isinstance(self.sliding, MotorFeedbackSliding):
            super().show_status(include_feedback)
            return
        label = {'released': '已释放', 'hold': '保持', 'run': '连续', 'step': '单步'}[self.mode]
        print(f'滑动窗口截面顺应 | {label} | J{self.first}–J{self.last} | phase={self.phase:.4f}rad | '
              f'omega={self.omega:g}rad/s | 自动={self.sliding.automatic} | '
              f'交接保持={self.sliding.phase_hold}', flush=True)
        print(f'实测截面直接顺应 K={self.options.gain:g} | '
              f'半圆预算={180*self.sliding.budget_scale:g}° | '
              f'{self.sliding.selection_reason} | 阻塞原因={self.section_block_reason or "无"}', flush=True)
        if include_feedback:
            print('实测角：'+' '.join(f'J{i+1}={v:.1f}°' for i, v in self.last_feedback.items()), flush=True)


# ==================== 3. 关节接口：真实状态读角，显式使能+位置目标 ====================
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


# ==================== 4. 物理场景、端点扰动与记录 ====================
class ManualArcParameters(ArcParameters):
    def __post_init__(self):
        if real.DIY_PARAMETERS:
            real.validate_parameters(self.amplitude_deg, self.omega_rad_s, self.first_joint, self.last_joint)
        else:
            super().__post_init__()

    def targets(self, phase):
        return np.radians(real.joint_angles_deg(phase, self.amplitude_deg, self.first_joint,
                          self.last_joint, length_normalized=self.last_joint-self.first_joint+1 != 6))


def make_manual_simulation(scene, start_x):
    p = replace(ManualArcParameters.load(ROOT/'configs/step1.json'),
                joints=real.NUM_JOINTS, first_joint=SIM_FIRST_JOINT, last_joint=SIM_LAST_JOINT,
                amplitude_deg=SIM_AMPLITUDE_DEG, omega_rad_s=SIM_OMEGA_RAD_S,
                initial_phase_rad=real.INITIAL_PHASE_RAD, settle_s=real.SETTLE_S)
    sim = ArcSimulation(scene, p, start_x)
    if scene in ('length_taper_return', 'elbow90_taper110_210'):
        route = sim.model.numeric('pipe_centerline').data.reshape(-1, 4)
        radius = .055 if scene == 'elbow90_taper110_210' else float(np.interp(start_x, route[:, 0], route[:, 3]))
        sim.data.qpos[2] += radius-.0565  # 只修正初始搭管；控制器不读取管径。
        mujoco.mj_forward(sim.model, sim.data)
    return sim


class DebugSession:
    """共用基础/固定控制；仿真扩展滑动模式、物理扰动和实际状态记录。"""
    def __init__(self, scene=SCENE, start_x=START_X_M):
        self.start_x = start_x
        real.validate_setup()
        self.sim = make_manual_simulation(scene, start_x)
        lengths = np.linalg.norm(np.diff(backbone(self.sim.model, self.sim.data), axis=0), axis=1)
        if not np.allclose(lengths, real.LINK_LENGTHS_M, atol=1e-8):
            raise ValueError('当前模型骨架与实机代码LINK_LENGTHS_M不同，请先核对机械尺寸')
        steps = real.CONTROL_PERIOD_S / self.sim.model.opt.timestep
        if abs(steps-round(steps)) > 1e-8:
            raise ValueError('控制周期须是MuJoCo时间步长的整数倍')
        self.substeps = int(round(steps))
        self.bus = MujocoMotorBus(self.sim)
        self.push_keys, self.push_force = set(), PUSH_FORCE_N
        self.events, self.episode_index = [], 0
        self._configure_perturbation()
        self.controller = DebugArcController(self.bus, self)
        self.commands = queue.Queue()
        self.next_feedback, self.elapsed = 0., 0.
        self.start_wall = time.monotonic()
        self.physics_started = False
        self.colour_active()

    def _configure_perturbation(self):
        direction = np.asarray(PUSH_WORLD_DIRECTION, float)
        if not np.isfinite(direction).all() or direction.shape != (3,) or np.linalg.norm(direction) <= 1e-12:
            raise ValueError('扰动方向必须是有限非零三维向量')
        if not math.isfinite(self.push_force) or self.push_force < 0:
            raise ValueError('扰动力必须有限且非负')
        self.push_direction = direction/np.linalg.norm(direction)
        self.robot_bodies = np.arange(self.sim.model.body('head').id, self.sim.model.nbody)
        self.push_targets = {}
        self.initial_center = backbone(self.sim.model, self.sim.data).mean(axis=0)

    def stop_push(self):
        self.push_keys.clear()
        self.push_targets.clear()
        self.sim.data.xfrc_applied[:] = 0.

    def begin_push(self, sign):
        model = self.sim.model
        last_body = int(model.jnt_bodyid[model.actuator_trnid[-1, 0]])
        endpoints = [model.body('head').id, last_body]
        projected = [self.sim.data.xpos[b] @ self.push_direction for b in endpoints]
        body = endpoints[int(np.argmin(projected) if sign < 0 else np.argmax(projected))]
        direction = sign*self.push_direction+np.array([0., 0., -PUSH_DOWNWARD_RATIO])
        if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 1e-12:
            raise ValueError('拉动方向配置退化')
        self.push_targets[sign] = (body, direction/np.linalg.norm(direction))
        self.push_keys.add(sign)  # 按下时选定物理端点，不在滚动中反复更换施力端。

    def apply_push(self):
        self.sim.data.xfrc_applied[:] = 0.
        if len(self.push_keys) != 1:
            return  # 同时按住两个相反方向时不产生额外向下合力。
        for sign in self.push_keys:
            body, direction = self.push_targets[sign]
            self.sim.data.xfrc_applied[body, :3] += self.push_force*direction

    def record_event(self, kind, **fields):
        self.events.append(dict(episode=self.episode_index, time_s=self.elapsed, kind=kind,
                                phase_rad=self.controller.phase,
                                window=[self.controller.first, self.controller.last], **fields))

    def reset_preview(self):
        # 只有明确Ctrl+R才重新设置位姿，保存上一段终态，不能冒充连续运动。
        before = dict(actual_qpos=self.sim.data.qpos.tolist(), actual_qvel=self.sim.data.qvel.tolist(),
                      targets=self.controller.targets.copy(), enabled=self.controller.enabled.copy())
        self.record_event('reset_episode', previous_state=before)
        self.controller.release()
        self.sim = make_manual_simulation(self.sim.scene, self.start_x)
        self.bus = MujocoMotorBus(self.sim)
        self._configure_perturbation()
        fresh = DebugArcController(self.bus, self)
        self.controller.__dict__.clear()
        self.controller.__dict__.update(fresh.__dict__)  # 保持队列分派中的对象身份。
        self.next_feedback, self.elapsed = 0., 0.
        self.physics_started = False
        self.episode_index += 1
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
        if c.sliding is None:
            self.next_feedback = real.poll_control_feedback(c, self.elapsed, self.next_feedback)
        c.tick(real.CONTROL_PERIOD_S)
        if before != (c.mode, c.forming, c.settle_remaining > 0):
            c.show_status()
        for _ in range(self.substeps):
            self.apply_push()
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
        center = backbone(self.sim.model, data).mean(axis=0)
        support = upward_support(self.sim)
        return dict(time_s=float(data.time), wall_elapsed_s=time.monotonic()-self.start_wall,
                    execution='mujoco_debug', scene=self.sim.scene,
                    physics_started=self.physics_started, **self.controller.snapshot(),
                    actual_qpos=data.qpos.tolist(), actual_qvel=data.qvel.tolist(),
                    actuator_force_nm=data.actuator_force.tolist(),
                    motor_feedback='200ms actual joint samples / 20ms control in compliance modes', episode=self.episode_index,
                    events=copy.deepcopy(self.events), body_center_m=center.tolist(),
                    body_displacement_m=(center-self.initial_center).tolist(),
                    support_up_n=support.tolist(),
                    instantaneous_support_center_joint=(None if support.sum() <= 1e-12 else
                        float(support @ np.arange(1, real.NUM_JOINTS+1)/support.sum())),
                    perturb_force_setting_n=self.push_force,
                    perturb_active_bodies=([self.sim.model.body(self.push_targets[sign][0]).name
                        for sign in sorted(self.push_keys)] if len(self.push_keys) == 1 else []),
                    perturb_force_world_n=data.xfrc_applied[self.robot_bodies, :3].sum(axis=0).tolist())

    def close(self):
        self.stop_push()
        self.controller.release()
        self.bus.apply_drive()
        mujoco.mj_forward(self.sim.model, self.sim.data)


# ==================== 5. 三维查看器：只有鼠标相机，没有MuJoCo原生键盘菜单 ====================
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
        gait = 'active feeding' if c.feeding_selected else 'sliding section' if c.sliding else 'fixed section' if c.compliance else 'basic arc'
        delta = backbone(sim.model, sim.data).mean(axis=0)-self.session.initial_center
        slider = c.sliding.snapshot() if c.sliding is not None else None
        if slider is None:
            cycle, handover = '-', '-'
            if c.feeding is not None:
                f = c.feeding
                o = f.observation or {}
                amount = float('nan') if f.baseline is None or not o.get('valid') else 1000*(o['transfer_coordinate_m']-f.baseline)
                cycle = f'{f.stage}: encoder delta {amount:+.1f} mm'
                handover = f'paused={c.feeding_paused}; {f.fault or "running"}'
        elif isinstance(c.sliding, MotorFeedbackSliding):
            o = slider['observation']
            cycle = 'waiting motor angles' if o is None else f"{o['state']} offset={o.get('filtered_error_joints', float('nan')):+.2f}"
            selection = slider.get('selection') or {}
            estimate = selection.get('filtered_count_joints')
            count = '-' if estimate is None else f'{estimate:.2f}'
            cycle += f" | {selection.get('state', 'collecting')} count~{count}"
            pending = slider['pending']
            handover = f"{pending.get('operation', 'slide')} -> {pending['to']} (rolling)" if pending else 'none'
        else:
            center = slider['cycle_support_center_joint']
            cycle = f'J{center:.2f}' if center is not None else f"collecting {slider['load_observation_phase_rad']:.1f}/6.3 rad"
            handover = 'unloading' if slider['phase_hold'] else 'waiting zero phase' if slider['preparation'] else 'none'
        pull = 'off'
        if len(self.session.push_keys) == 1:
            sign = next(iter(self.session.push_keys))
            body = self.session.sim.model.body(self.session.push_targets[sign][0]).name
            pull = f'{"left" if sign < 0 else "right"} {self.session.push_force:.1f} N @{body}'
        observation_label = 'Motor shape observer' if isinstance(c.sliding, MotorFeedbackSliding) else 'Cycle load center'
        left = 'Gait\nMode\nWindow\nOutside joints\nPhase\nPhysics\nTime\nPull\nSide displacement\n'+observation_label+'\nHandover'
        right = (f'{gait}{" -> "+c.pending_gait if c.pending_gait else ""}\n{c.mode}'
                 f'\nJ{c.first}-J{c.last} ({sum(c.enabled)} enabled)\nHead:{c.first-1}  Tail:{real.NUM_JOINTS-c.last}'
                 f'\n{c.phase:.3f} rad'
                 f'\n{"running" if self.session.physics_started else "initial preview; Ctrl+U"}'
                 f'\n{sim.data.time:.2f} s'
                 f'\n{pull}\n{float(delta @ self.session.push_direction):+.3f} m'
                 f'\n{cycle}\n{handover}')
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          viewport, left, right, self.context)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                          viewport, 'Ctrl+1/2/3/4: modes | Ctrl+U: form | Ctrl+G: run/pause\nHold Ctrl+A/D: pull left/right | Ctrl+[ / ]: force\nCtrl+W: auto window | Ctrl+R: reset preview',
                          'Ctrl+J/H: step / feed action | Ctrl+Y: return\nCtrl+L: feed frame | Ctrl+Space: pause\nCtrl+X: release | Ctrl+Q: quit', self.context)
        self.glfw.swap_buffers(self.window)
        self.glfw.poll_events()

    def close(self):
        self.glfw.make_context_current(self.window)
        self.context.free()
        self.scene = None  # MjvScene由Python绑定析构释放，没有free()方法。
        self.glfw.destroy_window(self.window)
        self.glfw.terminate()


# ==================== 6. 直接运行；基本操作共用Ctrl热键和控制台 ====================
def main():
    from pynput import keyboard
    session = DebugSession()
    viewer = listener = console = log_file = shape_preview = None
    stop = threading.Event()
    try:
        viewer = DebugViewer(session)
        if SHOW_SHAPE_PREVIEW:
            shape_preview = real.JointPreviewWindow()
        listener = real.global_key_listener(session.commands, keyboard, decode_sim_hotkey, decode_sim_release)
        console = threading.Thread(target=real.console_listener, args=(session.commands, stop), daemon=True)
        console.start()
        if SAVE_LOG:
            path, log_file = real.open_run_log(LOG_DIRECTORY, prefix='mujoco_debug')
            print(f'MuJoCo调试日志：{path}', flush=True)
        print(real.HELP, flush=True)
        print(SIM_HELP, flush=True)
        print('启动为静态预览；Ctrl+U接管或Ctrl+X释放后开始物理，暂停相位不暂停物理。蓝色使能，紫色失能。', flush=True)
        next_log = 0.
        while viewer.running() and not session.controller.quit:
            started = time.monotonic()
            if shape_preview is not None:
                session.controller.preview_feedback = shape_preview.active
                if shape_preview.active and session.controller.feedback_time_s is None:
                    session.controller.refresh_compliance_feedback()
            session.advance()
            if shape_preview is not None:
                # 独立显示仅接收读角快照；主MuJoCo查看器仍显示实际世界场景。
                shape_preview.publish(session.controller)
            viewer.render()
            wall_elapsed = time.monotonic()-session.start_wall
            if log_file and (session.controller.feeding_selected or wall_elapsed >= next_log):
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
            if shape_preview is not None:
                shape_preview.close()


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
