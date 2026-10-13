"""圆弧滚动实机控制：只运行本文件，通过热键直接控制飞特串口舵机。

直接运行本文件，自动查找CH340串口；所有实验配置在文件顶部修改。
DRY_RUN=True时使用内存串口，不连接机器人；日志自动保存在本文件旁的logs目录。
默认启动只读全身预览；显示实测角、使能和算法候选，估计上下方向，不测世界位置或水平朝向。
Ctrl+U 形成圆弧，Ctrl+J/H 正/反单步，Ctrl+G 连续/暂停，Ctrl+X 全部失能。
Ctrl+1/2/3 切换基础圆弧、固定截面顺应、仅角度反馈自动移窗与数量自适应。
不依赖 MuJoCo、EXE、相邻仓库或本仓库的其他 Python 模块。
"""
import json
import copy
import math
import statistics
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path
import queue
import sys
import threading
import time
import numpy as np


# ==================== 1. 经常修改的实验参数 ====================
PORT = None                    # None：和CaseA一样自动查找CH340；手动指定时改为'COM9'
BAUD_RATE = 115200
DRY_RUN = False                # False：真实机器人；True：内存串口演练，不打开设备
SAVE_LOG = True                # 自动保存实测角、目标、使能与模式；每次运行生成新文件
SHOW_PREVIEW = True            # 随实机脚本启动全身构型窗口；关闭窗口不停止控制
PREVIEW_GRAVITY_ALIGN = True    # 双侧重力代理对齐；窗口按钮可切回头部坐标
PREVIEW_PERIOD_S = .2          # 独立进程刷新，绘图不占用串口控制循环
LOG_DIRECTORY = Path(__file__).resolve().parent / 'logs'  # 不依赖终端/IDE工作目录
NUM_JOINTS = 20
FIRST_JOINT = 8                 # 关节从 1 开始编号，包含两个端点
LAST_JOINT = 13                 # 窗口外明确关闭舵机扭矩，不发送零角位置目标
AMPLITUDE_DEG = 60.0            # 圆弧幅值，单位：度
OMEGA_RAD_S = 0.6               # 连续滚动角频率，单位：rad/s
INITIAL_PHASE_RAD = 0.0
PHASE_STEP_RAD = 0.10           # Ctrl+J/H 每次累计的相位增量，单步也平滑执行
CONTROL_PERIOD_S = 0.02         # 50 Hz，同步发送窗口内所有目标
FORM_RATE_DEG_S = 15.0          # 从实测姿态形成圆弧时的最大目标角速度
MAX_TARGET_RATE_DEG_S = 45.0    # 正常滚动/修改幅值时的最大目标角速度
SETTLE_S = 2.0                 # 圆弧目标形成后静置，再允许滚动
SERVO_SPEED_RAW = 800          # 舵机速度寄存器原始值，沿用原目标块布局
TORQUE_LIMIT_RAW = 300         # 0..1000 的寄存器值，参考端默认值；不是 N·m
ACCELERATION_RAW = 20          # 加速度寄存器原始值，参考端默认值
FEEDBACK_PERIOD_S = 1.0        # 每秒读取主动关节位置；失败即结束并尝试失能
SERIAL_TIMEOUT_S = 0.10
AMPLITUDE_KEY_STEP_DEG = 2.0    # Ctrl+上/下：幅值增加/减小
OMEGA_KEY_STEP_RAD_S = 0.05     # Ctrl+右/左：角频率增加/减小

# 固定截面顺应：调实机主要改这里；原始负载/电流不作为控制输入。
COMPLIANCE_GAIN = 0.02          # 连续截面长度不变的曲率偏置，不是舵机PD增益
SECTION_YIELDING = 0.7          # 实际形状反馈权重；保持小于1，保留名义恢复参考
SECTION_GEOMETRY_PERIOD_S = .05 # 最短50 ms；20 ms循环通常量化为60 ms，使用最近角反馈
SECTION_TARGET_FILTER_S = .15
SECTION_MAX_TURN_DEG = 190.0    # 发送目标的截面转角上限
SECTION_MIN_GAP_FRACTION = .4   # 端距/主动骨架长度的最低开口比例
SECTION_MIN_ACTUAL_TURN_DEG = 150.0
SECTION_MAX_ACTUAL_TURN_DEG = 218.0
SECTION_MAX_PLANE_RMS_M = .015
LINK_LENGTHS_M = [.0645] * (NUM_JOINTS + 1)  # 仿真骨架尺寸，按实机铰链中心距修改
COMPLIANCE_FEEDBACK_PERIOD_S = .20  # 全20舵机读角；不把重复计算当20 Hz新实测
MAX_FEEDBACK_AGE_S = .8

# 自动移窗与数量自适应：同步 phase_normalized_count_v3，真实设备效果待用户验证。
# 冻结的相位基准对应 A=60°、omega=0.6、21段64.5mm、正向滚动。
# 其余模式仍可调幅值/频率。移窗或扩缩每次只改变相邻边界，不指定6/8白名单。
MOTOR_SLIDING_AUTO = True
MOTOR_SLIDING_COOLDOWN_S = 5.0
MOTOR_SLIDING_SWITCH_ANGLE_DEG = 4.0
# 以下是最常调的数量判断参数；不添加未经验证的回正修正。
POSTURE_COUNT_HYSTERESIS_JOINTS = .5
POSTURE_PERSISTENCE_S = 1.0
POSTURE_CONTRACT_EXIT_ANGLE_DEG = 12.0
POSTURE_EXPANSION_TURN_ERROR_DEG = -5.0
POSTURE_CONTRACTION_TURN_ERROR_DEG = 5.0

# 历史幅值/滑动窗口原型参数：只供旧仿真导入第3A段，当前实机入口不用。
MIN_COMPLIANCE_AMPLITUDE_DEG = 25.0
CURVATURE_FILTER_S = 0.5
MIN_WINDOW_JOINTS = 6
MAX_WINDOW_JOINTS = 12
WINDOW_GUARD_JOINTS = 2
WINDOW_DWELL_S = 0.6           # 新支撑段持续足够久才转移窗口
WINDOW_TRANSFER_S = 0.8        # 新旧使能交叠，先预置实测角，再渐入目标
SUPPORT_LOAD_THRESHOLD_RAW = 40 # 原始负载阈值，未标定为 N 或 N·m，需实机调整
SUPPORT_CURVATURE_THRESHOLD_DEG = 20.0
# auto 仅用负载+姿态估计支撑，无法观测断电舵机的真实管道接触。
# 不可靠时输入 support=9:11，人工指定实际承担支撑的关节区间；与使能窗口不同。
DEFAULT_MANUAL_SUPPORT = None

# 从 configs/hardware_20j_reference.json 复制，顺序是 J1..J20。
# 单文件即可运行。改变实机校准时修改本表，不要再用统一的 2048 零位。
SERVO_IDS = list(range(1, 21))
ZERO_RAW = [2027, 1953, 2013, 2088, 2057, 1985, 2022, 2111, 2068, 2027,
            2089, 2103, 1990, 2031, 2073, 2021, 2095, 2062, 2057, 2070]
SERVO_DIRECTION = [1] * NUM_JOINTS
# 原项目控制器角 -> 舵机协议角：奇数同号，偶数反号。与零位方向是两层换算。
MODEL_TO_SERVO_SIGN = [1 if j % 2 else -1 for j in range(1, NUM_JOINTS + 1)]
MIN_ANGLE_DEG = [-90.0] * NUM_JOINTS
MAX_ANGLE_DEG = [90.0] * NUM_JOINTS

HOTKEY_COMMANDS = {
    'U': 'apply', 'J': 'step', 'H': 'back', 'G': 'toggle', 'N': 'hold',
    'K': 'status', 'X': 'release', 'Q': 'quit', 'T': 'reverse', 'E': 'compliance',
    'W': 'window_auto',
    '-': 'delta-', '=': 'delta+',
}
HELP = '''圆弧滚动调试（Ctrl+字母全局热键；任何窗口均会响应）：
  Ctrl+U 读取当前位置，使能局部并缓慢形成圆弧；默认完成后保持
  Ctrl+J / Ctrl+H 正向 / 反向单步；Ctrl+-/= 相位步长减半/加倍
  Ctrl+G 连续滚动/暂停；Ctrl+空格 暂停；Ctrl+N 保持并重发当前目标
  Ctrl+T 反转连续滚动方向；Ctrl+K 读取所有关节位置
  Ctrl+E 固定窗口截面顺应开/关：先形成圆弧；保留相位与连续/暂停状态
  Ctrl+1 / 2 / 3 基础圆弧 / 固定截面顺应 / 角度反馈自动滑窗
  Ctrl+W 自动移窗与扩缩开/关（仅滑动模式）；外侧扭矩关闭
  Ctrl+上/下 幅值±2°；Ctrl+右/左 角频率±0.05 rad/s
  Ctrl+X 全部失能（机器人会失去主动支撑）；Ctrl+Q 退出，接管后会失能
控制台输入后回车：apply / step / back / run / pause / hold / reverse
                  compliance / gait_basic / gait_fixed / gait_sliding / window_auto
                  release / status / help / quit
参数：amplitude=60 omega=0.6 delta=0.1 window=8:13 gain=0.02 yielding=0.7 filter=0.15
window只在release后修改；自动滑窗只用全20关节角、采样时间和已知相位。
滑窗基准只标定了60度/0.6/正向；移窗不等于已从电机观测到支撑或卸载。
顺应性下pause冻结相位、继续形状反馈；hold冻结全部目标。
'''


# ==================== 2. 圆弧公式与角度换算 ====================
def joint_angles_deg(phase_rad, amplitude_deg, first_joint, last_joint, length_normalized=False):
    """这里是调步态时主要修改的公式；返回控制器坐标中的角度，单位为度。"""
    if length_normalized:
        # 60是半圆参考的缩放基准；七关节时各轴族分摊π，不会变成7×60°。
        return np.degrees(nominal_angles(LINK_LENGTHS_M, (first_joint, last_joint), phase_rad)
                          * (amplitude_deg / 60.)).tolist()
    angles = [0.0] * NUM_JOINTS
    for joint in range(first_joint, last_joint + 1):
        if joint % 2:           # 奇数关节：cos(phi)
            angle = amplitude_deg * math.cos(phase_rad)
        else:                   # 偶数关节：sin(phi)，与奇数相差 90°
            angle = amplitude_deg * math.sin(phase_rad)
        angles[joint - 1] = angle
    return angles               # 窗口外的 0 仅占位，驱动函数不发送这些目标


def angle_to_raw(index, angle_deg):
    """控制器角(度) -> 协议角(度) -> 已校准的 12 bit 单圈位置。"""
    protocol_deg = MODEL_TO_SERVO_SIGN[index] * angle_deg
    if not math.isfinite(protocol_deg) or not MIN_ANGLE_DEG[index] <= protocol_deg <= MAX_ANGLE_DEG[index]:
        raise ValueError(f'J{index + 1} 目标角超出档案限位: {protocol_deg}')
    raw = round(ZERO_RAW[index] + SERVO_DIRECTION[index] * protocol_deg * 4096 / 360)
    if not 0 <= raw <= 4095:
        raise ValueError(f'J{index + 1} 目标超出单圈位置范围: {raw}')
    return raw


def raw_to_angle(index, raw):
    if not 0 <= raw <= 4095:
        raise RuntimeError(f'J{index + 1} 实测位置 {raw} 不属于单圈位置模式')
    protocol_deg = (raw - ZERO_RAW[index]) * 360 / 4096 / SERVO_DIRECTION[index]
    return protocol_deg * MODEL_TO_SERVO_SIGN[index]


def validate_parameters(amplitude, omega, first, last):
    if type(first) is not int or type(last) is not int or not 1 <= first <= last <= NUM_JOINTS:
        raise ValueError('使能窗口必须满足 1 <= first <= last <= 20')
    if not math.isfinite(amplitude) or not 0 < amplitude <= 90:
        raise ValueError('幅值必须在 (0,90] 度范围内')
    if not math.isfinite(omega) or omega < 0 or amplitude * omega > MAX_TARGET_RATE_DEG_S:
        raise ValueError('角频率非负，且 amplitude * omega 不能超过目标角速度上限')
    for i in range(first - 1, last):
        angle_to_raw(i, amplitude)
        angle_to_raw(i, -amplitude)  # 预先检查完整滚动周期的两端，避免运行中才越界


def validate_setup():
    tables = (SERVO_IDS, ZERO_RAW, SERVO_DIRECTION, MODEL_TO_SERVO_SIGN, MIN_ANGLE_DEG, MAX_ANGLE_DEG)
    if any(len(table) != NUM_JOINTS for table in tables) or len(set(SERVO_IDS)) != NUM_JOINTS:
        raise ValueError('舵机表长度或 ID 唯一性错误')
    if any(type(v) is not int or not 1 <= v <= 253 for v in SERVO_IDS):
        raise ValueError('舵机 ID 必须为 1..253 的整数')
    if any(v not in (-1, 1) for v in SERVO_DIRECTION + MODEL_TO_SERVO_SIGN):
        raise ValueError('方向必须为 +1 或 -1')
    positive = (CONTROL_PERIOD_S, PHASE_STEP_RAD, FORM_RATE_DEG_S, MAX_TARGET_RATE_DEG_S,
                FEEDBACK_PERIOD_S, SERIAL_TIMEOUT_S,COMPLIANCE_FEEDBACK_PERIOD_S,MAX_FEEDBACK_AGE_S)
    if any(not math.isfinite(v) or v <= 0 for v in positive) or not math.isfinite(SETTLE_S) or SETTLE_S < 0:
        raise ValueError('控制周期、速度和反馈周期必须为正，静置时间非负')
    if not math.isfinite(INITIAL_PHASE_RAD) or not 1 <= SERVO_SPEED_RAW <= 32766:
        raise ValueError('初相位或舵机速度错误')
    if not 0 <= TORQUE_LIMIT_RAW <= 1000 or not 0 <= ACCELERATION_RAW <= 254:
        raise ValueError('力矩限制或加速度寄存器值错误')
    if MAX_FEEDBACK_AGE_S<=COMPLIANCE_FEEDBACK_PERIOD_S:
        raise ValueError('反馈过期时间必须大于全量反馈周期')
    make_section_options()  # 打开串口前检查当前固定截面参数
    if len(LINK_LENGTHS_M) != NUM_JOINTS+1 or not all(math.isfinite(v) and v>0 for v in LINK_LENGTHS_M):
        raise ValueError('LINK_LENGTHS_M须为21个有限正的铰链中心/端点距离')
    if not 0 < SECTION_MIN_ACTUAL_TURN_DEG < SECTION_MAX_ACTUAL_TURN_DEG < 360 or SECTION_MAX_PLANE_RMS_M <= 0:
        raise ValueError('实际截面有效范围或平面误差门槛错误')
    validate_parameters(AMPLITUDE_DEG, OMEGA_RAD_S, FIRST_JOINT, LAST_JOINT)
    validate_arc_reference(AMPLITUDE_DEG, OMEGA_RAD_S, FIRST_JOINT, LAST_JOINT,
                           LAST_JOINT-FIRST_JOINT+1 != 6)


def validate_arc_reference(amplitude, omega, first, last, normalized):
    """数量改变后实际峰值不一定等于顶部幅值；使能前检查两个轴族。"""
    peaks = np.maximum(np.abs(joint_angles_deg(0., amplitude, first, last, normalized)),
                       np.abs(joint_angles_deg(math.pi/2, amplitude, first, last, normalized)))
    if float(peaks.max())*omega > MAX_TARGET_RATE_DEG_S+1e-8:
        raise ValueError('当前数量的半圆参考超过目标角速度上限')
    for i in range(first-1, last):
        angle_to_raw(i, float(peaks[i]))
        angle_to_raw(i, -float(peaks[i]))


# ==================== 3A. 历史原型：旧仿真对照所需，当前实机入口不使用 ====================
# 核心流程：实测角 -> 奇偶关节曲率包络 -> 平滑幅值 -> 正交滚动目标。
# 支撑观测 -> 延时锁存 -> 保持/迁移局部窗口 -> 新旧关节目标渐变。
# 这里不读取串口、不使用管道半径，不改变机器人根部位置。
# 旧版曾用碰撞反力/硬件估计或人工支撑；当前实机不再接入这套接口。
# 注意：这是一种位置目标让顺控制，不是力传感器闭环或完整 Yamada 曲线重构。
@dataclass(frozen=True)
class SlidingArcOptions:
    gain: float = .02
    curvature_filter_s: float = .5
    minimum_amplitude_deg: float = 25.
    maximum_amplitude_deg: float = 60.
    guard_joints: int = 2
    minimum_width: int = 6
    maximum_width: int = 12
    contact_on_s: float = .12
    contact_off_s: float = .8
    window_dwell_s: float = .6
    transfer_s: float = .8
    target_rate_deg_s: float = 45.
    contact_loss_hold_s: float = .5

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Compliance options must be finite')
        if not 0 <= self.gain < .5 or not 0 < self.minimum_amplitude_deg <= self.maximum_amplitude_deg <= 90:
            raise ValueError('Invalid compliance gain or amplitude bounds')
        if not 2 <= self.minimum_width <= self.maximum_width or self.guard_joints < 0:
            raise ValueError('Invalid window bounds')
        if any(type(v) is not int for v in (self.minimum_width,self.maximum_width,self.guard_joints)) or self.minimum_width%2 or self.maximum_width%2:
            raise ValueError('Window widths must be even integers; guard must be integer')
        if min(self.curvature_filter_s, self.contact_on_s, self.contact_off_s, self.window_dwell_s,
               self.transfer_s, self.target_rate_deg_s, self.contact_loss_hold_s) <= 0:
            raise ValueError('Time constants and target speed must be positive')


def support_seed(scores, minimum_width=6, maximum_width=12, guard=2):
    """Pick the strongest connected support region; returns one-based bounds or None."""
    contacts = [i for i,value in enumerate(scores) if value>0]
    if not contacts:
        return None
    groups = [[contacts[0]]]
    for i in contacts[1:]:
        if i-groups[-1][-1]>2:
            groups.append([])
        groups[-1].append(i)
    region = max(groups,key=lambda group:sum(scores[i] for i in group))
    width = max(minimum_width,region[-1]-region[0]+1+2*guard)
    width = min(len(scores),maximum_width,width+width%2)
    lo = max(0,min(len(scores)-width,round(statistics.mean(region)-(width-1)/2)))
    return lo+1,lo+width


class SlidingArcCompliance:
    def __init__(self, measured_deg, phase, seed_window, options=SlidingArcOptions(), initial_enabled=None):
        self.options = options
        self.n = len(measured_deg)
        first, last = seed_window
        if not 1 <= first <= last <= self.n or self.n < options.minimum_width:
            raise ValueError('Invalid compliance seed window')
        if not math.isfinite(phase) or not all(math.isfinite(q) for q in measured_deg):
            raise ValueError('Invalid measured angles or phase')
        self.phase = phase
        self.window = (first - 1, last - 1)
        self.weights = ([float(e) for e in initial_enabled] if initial_enabled is not None else
                        [float(first <= j <= last) for j in range(1,self.n+1)])
        if len(self.weights)!=self.n:
            raise ValueError('Initial enable mask size mismatch')
        self.targets = list(measured_deg)
        self.amplitudes = [options.maximum_amplitude_deg] * self.n
        self.contact_age = [0.] * self.n
        self.absent_age = [0.] * self.n
        self.support = [False] * self.n
        self.proposal, self.proposal_age = None, 0.
        self.unsupported_s = 0.
        self.events = []
        self.time_s = 0.
        self.phase_paused = False
        self.transfer_remaining_s = 0.

    def update(self, measured_deg, support, dt, phase):
        if len(measured_deg) != self.n or len(support) != self.n or not all(math.isfinite(q) for q in measured_deg):
            raise ValueError('Incomplete or nonfinite compliance observations')
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(phase):
            raise ValueError('Invalid compliance time or phase')
        p = self.options
        self.time_s += dt
        self.phase = phase
        for i, observed in enumerate(support):
            self.contact_age[i] = self.contact_age[i] + dt if observed else 0.
            self.absent_age[i] = 0. if observed else self.absent_age[i] + dt
            if self.contact_age[i] >= p.contact_on_s:
                self.support[i] = True
            elif self.absent_age[i] >= p.contact_off_s:
                self.support[i] = False
        self.unsupported_s = 0. if any(support) else self.unsupported_s + dt
        self.transfer_remaining_s = max(0.,self.transfer_remaining_s-dt)
        self.phase_paused = self.unsupported_s >= p.contact_loss_hold_s or self.transfer_remaining_s>0

        # Orthogonal adjacent joints form a measured curvature envelope.
        # Rebuild from ACTUAL angles each update, never recursively deform a target.
        previous = [i for i, w in enumerate(self.weights) if w > .95]
        alpha = 1 - math.exp(-dt / p.curvature_filter_s)
        for i in previous:
            others = [j for j in previous if (j - i) % 2]
            if not others:
                continue
            opposite = min(others, key=lambda j: abs(j - i))
            observed = math.hypot(measured_deg[i], measured_deg[opposite])
            desired = min(p.maximum_amplitude_deg, max(p.minimum_amplitude_deg, observed * (1 + p.gain)))
            self.amplitudes[i] += alpha * (desired - self.amplitudes[i])

        contacts = [i for i, active in enumerate(self.support) if active]
        if contacts:
            # Preserve one support-connected region; isolated distant collisions
            # must not activate all intervening hanging joints.
            groups = [[contacts[0]]]
            for i in contacts[1:]:
                if i - groups[-1][-1] > 2:
                    groups.append([])
                groups[-1].append(i)
            center = sum(self.window) / 2
            contacts = min(groups, key=lambda g: abs(statistics.mean(g) - center))
            amplitude = statistics.median(self.amplitudes[i] for i in previous) if previous else p.maximum_amplitude_deg
            width = max(p.minimum_width, contacts[-1] - contacts[0] + 1 + 2 * p.guard_joints)
            width = min(self.n, p.maximum_width, width + width % 2)
            lo = max(0, min(self.n - width, round(statistics.mean(contacts) - (width - 1) / 2)))
            candidate = (lo, lo + width - 1)
            if self.window[0] <= contacts[0] and self.window[1] >= contacts[-1]:
                candidate = self.window  # Do not recenter an already sufficient support window.
            if candidate == self.window:
                self.proposal, self.proposal_age = None, 0.
            elif candidate != self.proposal:
                self.proposal, self.proposal_age = candidate, dt
            else:
                self.proposal_age += dt
                if self.proposal_age >= p.window_dwell_s:
                    old = self.window
                    self.window = candidate
                    self.transfer_remaining_s = p.transfer_s + .4
                    self.phase_paused = True
                    self.events.append(dict(time_s=self.time_s, old=[old[0]+1,old[1]+1],
                                            new=[lo+1,lo+width], support=[i+1 for i in contacts]))
                    self.proposal, self.proposal_age = None, 0.
                    old_width = old[1]-old[0]+1
                    scale = min(1.,old_width/width)
                    for i in range(self.n):
                        if self.weights[i]>0:
                            self.amplitudes[i] = max(p.minimum_amplitude_deg,self.amplitudes[i]*scale)
                    for i in range(lo, lo + width):
                        if self.weights[i] == 0:
                            self.targets[i] = measured_deg[i]
                            self.amplitudes[i] = max(p.minimum_amplitude_deg,amplitude*scale)

        for i in range(self.n):
            desired_weight = float(self.window[0] <= i <= self.window[1])
            self.weights[i] += max(-dt/p.transfer_s, min(dt/p.transfer_s, desired_weight-self.weights[i]))
            if self.weights[i] < 1e-9:
                self.weights[i] = 0.
            wave = math.cos(phase) if i%2==0 else math.sin(phase)
            goal = self.amplitudes[i] * wave
            # During handover follow measured pose -> rolling shape; exiting
            # joints yield to measurement before their actuator is switched off.
            goal = measured_deg[i] + self.weights[i] * (goal - measured_deg[i])
            limit = p.target_rate_deg_s * dt
            self.targets[i] += max(-limit, min(limit, goal-self.targets[i]))
        return self.targets.copy(), [w > 0 for w in self.weights]

    def snapshot(self):
        return dict(window=[self.window[0]+1,self.window[1]+1], weights=self.weights.copy(),
                    support=[i+1 for i,flag in enumerate(self.support) if flag],
                    amplitude_deg=self.amplitudes.copy(), phase_paused=self.phase_paused,
                    unsupported_s=self.unsupported_s,transfer_remaining_s=self.transfer_remaining_s)


def make_compliance_options(amplitude):
    """集中把顶部参数送入核心。Step1 幅值低于 25° 时同步降低顺应性下限。"""
    return SlidingArcOptions(gain=COMPLIANCE_GAIN,curvature_filter_s=CURVATURE_FILTER_S,
        minimum_amplitude_deg=min(MIN_COMPLIANCE_AMPLITUDE_DEG,amplitude),maximum_amplitude_deg=amplitude,
        guard_joints=WINDOW_GUARD_JOINTS,minimum_width=MIN_WINDOW_JOINTS,maximum_width=MAX_WINDOW_JOINTS,
        window_dwell_s=WINDOW_DWELL_S,transfer_s=WINDOW_TRANSFER_S,target_rate_deg_s=MAX_TARGET_RATE_DEG_S)


# ==================== 3B. 固定截面核心：与当前仿真方法一致，单文件内可修改 ====================
# BEGIN FIXED_SECTION_CORE
def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-14)


def forward(q, lengths):
    """Head-frame skeleton, including head and tail endpoints."""
    q, lengths = np.asarray(q), np.asarray(lengths)
    if len(lengths) != len(q) + 1 or not np.isfinite(q).all():
        raise ValueError('Mechanical dimensions or angles are invalid')
    points = [np.zeros(3), np.array([-lengths[0], 0., 0.])]
    orientation = np.eye(3)
    for i, angle in enumerate(q):
        a = angle if i % 2 == 0 else -angle
        c, s = math.cos(a), math.sin(a)
        rotation = (np.array([[c,-s,0],[s,c,0],[0,0,1.]]) if i % 2 == 0 else
                    np.array([[c,0,s],[0,1.,0],[-s,0,c]]))
        orientation = orientation @ rotation
        points.append(points[-1] + orientation @ [-lengths[i+1],0,0])
    return np.asarray(points)


def pitch_variance(tangents, directions):
    a = np.arcsin(np.clip(np.abs(np.asarray(directions) @ tangents.T),0,1))
    return np.var(a, axis=-1)


def refine_axis(tangents, seed):
    direction = unit(seed); variance = float(pitch_variance(tangents,direction)); step=.35
    for _ in range(100):
        if step <= 1e-7 or variance <= 1e-18: break
        reference = [0,0,1] if abs(direction[2]) < .85 else [0,1,0]
        a=unit(np.cross(reference,direction)); b=unit(np.cross(direction,a))
        offsets=np.array([a,-a,b,-b,(a+b)*.7071,(a-b)*.7071,(-a+b)*.7071,(-a-b)*.7071])
        candidates=direction+step*offsets
        candidates/=np.linalg.norm(candidates,axis=1)[:,None]
        values=pitch_variance(tangents,candidates); improved=False
        # Preserve C# candidate-order tie breaking.
        for candidate, value in zip(candidates,values):
            if value + 1e-18 < variance:
                direction,variance,improved=candidate,float(value),True
        if not improved: step *= .5
    return direction,variance


def paper_axis(points):
    """Original pitch-variance objective and coarse/refine search."""
    points=np.asarray(points); center=points.mean(axis=0)
    first,last=1,len(points)-2
    if len(points)>=12:
        typical=np.sort(np.linalg.norm(np.diff(points[1:-1],axis=0),axis=1))
        typical=typical[len(typical)//2]
        if np.linalg.norm(points[1]-points[0])<typical*.8: first+=2
        if np.linalg.norm(points[-1]-points[-2])<typical*.8: last-=2
    tangents=np.asarray([unit(points[i+1]-points[i-1]) for i in range(first,last+1)])
    pv,pe=np.linalg.eigh((points-center).T@(points-center))
    mean=tangents.mean(axis=0); centered=tangents-mean
    tv,te=np.linalg.eigh(centered.T@centered/len(tangents)); spread=max(0,float(tv.sum()))
    if len(tangents)<3:
        direction,variance,confidence=pe[:,-1],0.,.25
    elif spread<.015:
        direction=unit(mean) if np.linalg.norm(mean)>1e-8 else pe[:,-1]
        variance=float(pitch_variance(tangents,direction)); confidence=0.
    else:
        i=np.arange(256); z=1-2*(i+.5)/256; r=np.sqrt(1-z*z); az=i*2.399963229728653
        coarse=np.column_stack([r*np.cos(az),r*np.sin(az),z])
        values=pitch_variance(tangents,coarse)
        seeds=[*te.T,pe[:,-1],unit(points[-1]-points[0]),*coarse[np.argsort(values)[:4]]]
        candidates=[refine_axis(tangents,s) for s in seeds]
        direction,variance=min(candidates,key=lambda item:item[1])
        second,small=max(0,tv[1]),max(0,tv[0])
        gap=0 if second<1e-9 else np.clip((second-small)/second,0,1)
        confidence=float(gap*np.clip(spread/.08,0,1)*np.clip(1-math.sqrt(variance)/.35,0,1))
        if second<max(1e-8,spread*1e-4): direction,confidence=pe[:,-1],min(confidence,.25)
    direction=unit(direction); end=unit(points[0]-points[-1]); alignment=direction@end
    if abs(alignment)>.08: direction*=1 if alignment>=0 else -1
    else:
        alignment=-float(tangents@direction @ np.ones(len(tangents)))
        if abs(alignment)>len(tangents)*.02: direction*=1 if alignment>=0 else -1
        elif direction[np.argmax(abs(direction))]<0: direction*=-1
    return dict(point=center,direction=direction,confidence=confidence,variance=variance)


def projection_basis(direction, previous=None):
    h=np.zeros(3) if previous is None else previous-direction*(previous@direction)
    if np.linalg.norm(h)<1e-8:
        h=np.cross([0,0,1] if abs(direction[2])<.85 else [0,1,0],direction)
    h=unit(h)
    return np.column_stack([h,unit(np.cross(direction,h)),direction])


@dataclass
class Segment:
    start: np.ndarray
    tangent: np.ndarray
    length: float
    radius: float
    turn: float
    joint: int
    kind: str

    def at(self,t):
        if self.radius==0: return self.start + self.tangent*(self.length*t)
        a=self.turn*t; r=math.copysign(self.radius,self.turn)
        normal=np.array([-self.tangent[1],self.tangent[0]])
        return self.start+r*(math.sin(a)*self.tangent+2*math.sin(a/2)**2*normal)

    def end_tangent(self):
        c,s=math.cos(self.turn),math.sin(self.turn)
        return np.array([[c,-s],[s,c]])@self.tangent


def continuous_section(points, gamma_min=math.pi*.01, original=False):
    """A/B/C port; original=True reproduces D for migration comparison only.

    Open mode keeps straight turns zero and outward arcs signed. It never
    inserts an inward arc merely because a joint is straight/outward.
    """
    points=np.asarray(points); delta=np.diff(points,axis=0)
    lengths=np.linalg.norm(delta,axis=1)
    if np.min(lengths)<1e-8: raise ValueError('Zero projected link')
    tangents=delta/lengths[:,None]
    turns=np.arctan2(tangents[:-1,0]*tangents[1:,1]-tangents[:-1,1]*tangents[1:,0],
                     np.sum(tangents[:-1]*tangents[1:],axis=1))
    winding=1 if turns.sum()>=0 else -1
    position=points[0].copy(); tangent=tangents[0].copy(); result=[]
    def add(length,radius,turn,joint,kind):
        nonlocal position,tangent
        if length<1e-12: return
        row=Segment(position.copy(),tangent.copy(),length,radius,turn,joint,kind)
        result.append(row); position=row.at(1); tangent=row.end_tangent()
    add(lengths[0]/2,0,0,0,'end')
    for i,turn in enumerate(turns,1):
        a,b=lengths[i-1]/2,lengths[i]/2
        if original and turn*winding<=gamma_min:
            add(a+b,(a+b)/gamma_min,winding*gamma_min,i,'D')
        elif abs(turn)<1e-7:
            add(a+b,0,0,i,'straight')
        else:
            radius=min(a,b)/math.tan(abs(turn)/2)
            kind='C' if abs(a-b)<=1e-10*max(a,b) else 'A' if a>b else 'B'
            if a>b: add(a-b,0,0,i,kind)
            add(radius*abs(turn),radius,turn,i,kind)
            if b>a: add(b-a,0,0,i,kind)
    add(lengths[-1]/2,0,0,len(points)-2,'end')
    return result


def sample_section(segments, step=.005):
    return np.asarray([segments[0].start]+[s.at(t) for s in segments for t in
                      np.linspace(0,1,max(2,int(math.ceil(s.length/step))+1))[1:]])


def local_section(q,lengths,first,last,previous=None, method='plane'):
    points=forward(q,lengths)[first-1:last+3]  # one entering/exiting boundary link
    if method=='paper': axis=paper_axis(points)
    else:
        # Open near-planar arc extension: directly fit its section plane.
        fitting=points[1:-1]  # passive boundary links cannot determine the section plane
        center=fitting.mean(axis=0); _,_,vh=np.linalg.svd(fitting-center,full_matrices=False)
        direction=vh[-1]
        if previous is not None and direction@previous['direction']<0: direction=-direction
        elif previous is None and direction[np.argmax(abs(direction))]<0: direction=-direction
        distances=(fitting-center)@direction
        axis=dict(point=center,direction=direction,confidence=float(np.clip(1-np.std(distances)/.03,0,1)),
                  variance=float(np.mean(distances**2)))
    if previous is not None and axis['direction']@previous['direction']<0: axis['direction']*=-1
    basis=projection_basis(axis['direction'],None if previous is None else previous['basis'][:,0])
    axis['basis']=basis
    projected=(points-axis['point'])@basis[:,:2]
    curve=continuous_section(projected)
    return dict(points=points,projected=projected,curve=curve,axis=axis,
                total_turn=sum(abs(s.turn) for s in curve if 1<=s.joint<=last-first+1),
                fit_rms=math.sqrt(axis['variance']))

@dataclass(frozen=True)
class SectionOptions:
    gain: float = .02
    yielding: float = .7
    geometry_period_s: float = .05
    target_rate_deg_s: float = 45.
    target_filter_s: float = .15
    maximum_section_turn_deg: float = 190.
    minimum_gap_fraction: float = .4

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Nonfinite parameter')
        if not 0 <= self.gain < .2 or not 0 <= self.yielding < 1:
            raise ValueError('Invalid compliance gain')
        if min(self.geometry_period_s, self.target_rate_deg_s, self.target_filter_s) <= 0:
            raise ValueError('Geometry period, rate limit and filter time must be positive')
        if not 0 < self.minimum_gap_fraction < 1 or not 0 < self.maximum_section_turn_deg < 360:
            raise ValueError('Invalid open-shape bounds')


def yamada_section(curve, link_lengths, first, count, phase, gain=0., direction=1):
    """Exact piecewise κ integral over neighbouring links, τ=0 for Step1.

    θ=κ Δs is the planar limit of the reference's sinc integration. Global
    yaw/pitch indexing and phase match Step1, including windows starting Pitch.
    The same-module line/arc lengths are normalized together to physical length.
    """
    lengths=np.asarray(link_lengths)
    modules=np.r_[lengths[0]/2,(lengths[:-1]+lengths[1:])/2,lengths[-1]/2]
    winding=1 if sum(s.turn for s in curve if 1<=s.joint<=count)>=0 else -1
    total=[sum(s.length for s in curve if s.joint==j) for j in range(len(modules))]
    rows=[]; cursor=0.
    for s in curve:
        length=s.length*modules[s.joint]/max(total[s.joint],1e-12)
        taper=.5 if s.joint in (1,count) else 1.
        factor=1+gain*taper if 1<=s.joint<=count else 1.
        turn=s.turn*winding*factor
        rows.append((cursor,cursor+length,turn/length if s.radius else 0.))
        cursor+=length
    joints=np.r_[0,np.cumsum(lengths)]
    targets=np.zeros(count)
    for j in range(1,count+1):
        integral=sum(k*max(0,min(hi,joints[j+1])-max(lo,joints[j-1])) for lo,hi,k in rows)
        number=first+j-1
        targets[j-1]=integral*(math.cos(phase) if number%2 else direction*math.sin(phase))
    return targets,rows


class FixedSectionCompliance:
    def __init__(self, parameters, lengths, options=SectionOptions()):
        self.p = parameters
        self.lengths = np.asarray(lengths)
        self.options = options
        self.first, self.last = parameters.first_joint, parameters.last_joint
        self.count = self.last - self.first + 1
        # This implementation needs one passive boundary link at each end and
        # enough interior points to estimate an open section plane.
        if self.first < 2 or self.last >= parameters.joints or self.count < 3:
            raise ValueError('Section controller requires 3+ active joints inside J2..J(N-1)')
        if self.lengths.shape != (parameters.joints+1,) or not np.isfinite(self.lengths).all() or np.any(self.lengths <= 0):
            raise ValueError('Expected one positive physical length per backbone link')
        self.enabled = parameters.enabled.copy()
        self.targets = None
        self.previous_axis = None
        self.geometry = None
        self.target_geometry = None
        self.reference_geometry = None
        self.last_goal = None
        self.next_geometry_s = 0.
        self.status = 'initializing'
        self.updates = 0
        self.clipped = 0

    def _check_goal(self, angles):
        """Evaluate the open shape from angles, without any pipe information."""
        geometry = local_section(angles, self.lengths, self.first, self.last)
        active = geometry['points'][1:-1]
        length = np.linalg.norm(np.diff(active, axis=0), axis=1).sum()
        gap = np.linalg.norm(active[-1] - active[0]) / length
        valid = (
            geometry['total_turn'] <= math.radians(self.options.maximum_section_turn_deg) + 1e-8
            and gap >= self.options.minimum_gap_fraction
        )
        return valid, geometry

    def update(self, measured, phase, time_s, dt, reference_targets=None):
        """Read actual shape and return one full-size angle command.

        Only self.enabled is driven by the adapter. Passive entries carry the
        measured angles so that later geometry checks use the actual boundary.
        """
        measured = np.asarray(measured)
        if self.targets is None:
            self.targets = measured.copy()
        if time_s + 1e-10 >= self.next_geometry_s:
            self.next_geometry_s = time_s + self.options.geometry_period_s
            self.updates += 1
            geometry = local_section(measured, self.lengths, self.first, self.last, self.previous_axis)
            reference = measured.copy()
            nominal_targets = self.p.targets(phase) if reference_targets is None else np.asarray(reference_targets)
            if nominal_targets.shape != measured.shape or not np.isfinite(nominal_targets).all():
                raise ValueError('Invalid external open-arc reference')
            reference[self.enabled] = nominal_targets[self.enabled]
            ref_geometry = local_section(reference, self.lengths, self.first, self.last)
            self.geometry, self.reference_geometry = geometry, ref_geometry
            self.previous_axis = geometry['axis']
            links = self.lengths[self.first-1:self.last+2]
            observed, _ = yamada_section(
                geometry['curve'], links, self.first, self.count, phase,
                self.options.gain, self.p.direction,
            )
            nominal, _ = yamada_section(
                ref_geometry['curve'], links, self.first, self.count, phase,
                direction=self.p.direction,
            )
            goal = reference.copy()
            # Calibrate the inverse approximation on the SAME phase/reference.
            # Thus an exact reference observation with K=0 gives exact Step1.
            goal[self.enabled] += self.options.yielding * (observed - nominal)
            goal[self.enabled] = np.clip(goal[self.enabled], -math.pi/2 + .01, math.pi/2 - .01)
            valid, goal_geometry = self._check_goal(goal)
            if not valid:
                self.clipped += 1
                # Search only the proposed correction, never actuate passive joints.
                lower, upper = 0., 1.
                for _ in range(8):
                    fraction = (lower + upper) / 2
                    candidate = reference + fraction * (goal - reference)
                    if self._check_goal(candidate)[0]:
                        lower = fraction
                    else:
                        upper = fraction
                goal = reference + lower * (goal - reference)
                valid, goal_geometry = self._check_goal(goal)
            self.last_goal, self.target_geometry = goal, goal_geometry
            self.status = 'running' if valid else 'reference constraint infeasible'
        # Hold the candidate between geometry updates; filter it at every
        # physics/control tick. The next geometry update uses the new phase.
        if self.last_goal is not None:
            filter_gain = 1 - math.exp(-dt / self.options.target_filter_s)
            delta = filter_gain * (self.last_goal - self.targets)
            limit = math.radians(self.options.target_rate_deg_s) * dt
            self.targets[self.enabled] += np.clip(delta[self.enabled], -limit, limit)
        self.targets[~self.enabled] = measured[~self.enabled]
        return self.targets.copy()

    def snapshot(self):
        g=self.geometry
        if g is None:return dict(status=self.status)
        curve_points=sample_section(g['curve'])
        xy=g['projected'][1:-1]
        circle=np.linalg.lstsq(np.column_stack([2*xy,np.ones(len(xy))]),np.sum(xy*xy,axis=1),rcond=None)[0]
        radius=math.sqrt(max(0,circle[2]+circle[:2]@circle[:2]))
        return dict(status=self.status,window=[self.first,self.last],geometry_updates=self.updates,
                    constrained_updates=self.clipped,fit_rms_m=g['fit_rms'],
                    measured_section_turn_deg=math.degrees(g['total_turn']),
                    target_section_turn_deg=math.degrees(self.target_geometry['total_turn']),
                    fitted_body_radius_m=radius,axis=g['axis']['direction'].tolist(),
                    axis_confidence=g['axis']['confidence'],projected_m=g['projected'].tolist(),
                    section_curve_m=curve_points.tolist())
# END FIXED_SECTION_CORE


@dataclass(frozen=True)
class SectionReference:
    """只提供几何核心需要的步态参考；相位反向由控制器推进方向实现。"""
    first_joint: int
    last_joint: int
    amplitude_deg: float
    joints: int = NUM_JOINTS
    direction: int = 1
    length_normalized: bool = False

    @property
    def enabled(self):
        indices = np.arange(1, self.joints+1)
        return (indices >= self.first_joint) & (indices <= self.last_joint)

    def targets(self, phase):
        return np.radians(joint_angles_deg(phase, self.amplitude_deg, self.first_joint, self.last_joint,
                                          self.length_normalized))


def make_section_options():
    return SectionOptions(gain=COMPLIANCE_GAIN, yielding=SECTION_YIELDING,
        geometry_period_s=SECTION_GEOMETRY_PERIOD_S, target_rate_deg_s=MAX_TARGET_RATE_DEG_S,
        target_filter_s=SECTION_TARGET_FILTER_S, maximum_section_turn_deg=SECTION_MAX_TURN_DEG,
        minimum_gap_fraction=SECTION_MIN_GAP_FRACTION)


# BEGIN MOTOR_SLIDING_CORE
@dataclass(frozen=True)
class ArcParameters:
    joints: int = 20
    first_joint: int = 8
    last_joint: int = 13
    amplitude_deg: float = 60.0
    omega_rad_s: float = 0.6
    direction: int = 1
    initial_phase_rad: float = 0.0
    settle_s: float = 2.0
    kp: float = 127.0
    kd: float = 1.0
    torque_limit_nm: float = 2.355
    passive_damping: float = 1.5

    def __post_init__(self):
        if any(type(v) is not int for v in (self.joints, self.first_joint, self.last_joint, self.direction)):
            raise ValueError('Joint counts, bounds and direction must be integers')
        if not 1 <= self.first_joint <= self.last_joint <= self.joints:
            raise ValueError('Invalid one-based active window')
        numeric = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not np.isfinite(numeric).all():
            raise ValueError('Parameters must be finite')
        if self.direction not in (-1, 1) or not 0 < self.amplitude_deg <= 90:
            raise ValueError('direction must be ±1; amplitude must be in (0,90]')
        if min(self.kp, self.torque_limit_nm) <= 0 or min(self.omega_rad_s, self.kd, self.settle_s, self.passive_damping) < 0:
            raise ValueError('Invalid frequency, gain or damping')
        if self.torque_limit_nm > 2.355:
            raise ValueError('Torque exceeds the reused model rating (2.355 Nm)')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))

    @property
    def enabled(self):
        return (np.arange(1, self.joints + 1) >= self.first_joint) & (np.arange(1, self.joints + 1) <= self.last_joint)

    def targets(self, phase):
        if not math.isfinite(phase):
            raise ValueError('Phase must be finite')
        odd = np.arange(self.joints) % 2 == 0
        angles = math.radians(self.amplitude_deg) * np.where(odd, np.cos(phase), self.direction * np.sin(phase))
        return np.where(self.enabled, angles, 0.0)

    def phase(self, time_s):
        return self.initial_phase_rad + self.omega_rad_s * max(0.0, time_s - self.settle_s)


@dataclass(frozen=True)
class ObserverOptions:
    width: int = 6
    end_links: int = 4
    minimum_end_links: int = 3
    minimum_straightness: float = .98
    maximum_end_disagreement_deg: float = 25.
    minimum_turn_deg: float = 150.
    maximum_turn_deg: float = 218.
    minimum_gap: float = .4
    maximum_plane_rms_m: float = .015
    boundary_links: int = 2
    top_band_link_fraction: float = .125
    samples_per_module: int = 12
    minimum_projected_down: float = .25
    error_filter_s: float = .2
    offset_threshold_joints: float = .5
    persistence_s: float = .5


class AngleShapeObserver:
    """One causal observer; current enabled mask always comes from feedback.

    Angles use controller/model signs, in degrees. Phase is a known commanded
    phase in radians, and timestamps refer to actual motor feedback acquisition.
    A recommendation never updates the enabled window inside the observer.
    """
    def __init__(self, link_lengths_m, amplitude_deg=60., options=ObserverOptions()):
        self.lengths = np.asarray(link_lengths_m, float).copy()
        if self.lengths.ndim != 1 or len(self.lengths) < 9 or not np.isfinite(self.lengths).all() or np.any(self.lengths <= 0):
            raise ValueError('Finite positive mechanical link lengths required')
        self.n = len(self.lengths) - 1
        self.amplitude = math.radians(amplitude_deg)
        self.options = options
        self.last_time = None
        self.window = None
        self.filtered_error = None
        self.pending_direction = 0
        self.pending_since = None

    def infer_down(self, points, window):
        first, last = window
        head_count = min(self.options.end_links, first - 1)
        tail_count = min(self.options.end_links, self.n - last)
        if min(head_count, tail_count) < self.options.minimum_end_links:
            raise ValueError('insufficient passive end links')
        head = points[:head_count + 1][::-1]
        tail = points[-tail_count - 1:]
        vectors, straightness = [], []
        for end in (head, tail):
            chord = end[-1] - end[0]
            length = np.linalg.norm(chord)
            if length <= 1e-12:
                raise ValueError('degenerate end direction')
            vectors.append(chord / length)
            straightness.append(float(length / np.linalg.norm(np.diff(end, axis=0), axis=1).sum()))
        disagreement = math.degrees(math.acos(float(np.clip(vectors[0] @ vectors[1], -1., 1.))))
        summed = vectors[0] + vectors[1]
        if np.linalg.norm(summed) <= 1e-12:
            raise ValueError('opposite end directions')
        down = summed / np.linalg.norm(summed)
        quality = dict(head_straightness=straightness[0], tail_straightness=straightness[1],
                       end_disagreement_deg=disagreement, head_direction=vectors[0].tolist(),
                       tail_direction=vectors[1].tolist(), end_link_counts=[head_count, tail_count])
        return down, quality

    def geometry(self, q, window):
        section = local_section(q, self.lengths, *window)
        points = section['points'][1:-1]
        length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        info = dict(turn_deg=math.degrees(section['total_turn']), fit_rms_m=section['fit_rms'],
                    gap_fraction=float(np.linalg.norm(points[-1] - points[0]) / length))
        good = (self.options.minimum_turn_deg <= info['turn_deg'] <= self.options.maximum_turn_deg
                and info['gap_fraction'] >= self.options.minimum_gap
                and info['fit_rms_m'] <= self.options.maximum_plane_rms_m)
        return good, info, section

    def crest(self, points, window, section, down, return_curve=False):
        """Smooth projected curve with a continuous physical joint coordinate.

        Each curved mechanical module is mapped to j-0.5 .. j+0.5, including
        all A/B/C pieces. This prevents short-radius smoothing from changing
        its joint identity. End half-links retain their half-intervals.
        """
        start = max(0, window[0] - self.options.boundary_links)
        stop = min(len(points), window[1] + self.options.boundary_links + 2)
        axis = section['axis']
        basis = axis['basis'][:, :2]
        projected_down = down @ basis
        down_norm = float(np.linalg.norm(projected_down))
        if down_norm < self.options.minimum_projected_down:
            raise ValueError('down direction nearly normal to section')
        projected = (points[start:stop] - axis['point']) @ basis
        curve = continuous_section(projected)
        groups = []
        for segment in curve:
            group = (segment.joint, segment.kind == 'end')
            if not groups or groups[-1][0] != group:
                groups.append((group, []))
            groups[-1][1].append(segment)
        locations, coordinates, arc_weights = [], [], []
        last_point = stop - 1
        for (joint, is_end), pieces in groups:
            if is_end and joint == 0:
                lo, hi = start, start + .5
            elif is_end:
                lo, hi = last_point - .5, last_point
            else:
                lo, hi = start + joint - .5, start + joint + .5
            module_length = sum(piece.length for piece in pieces)
            traversed = 0.
            for piece in pieces:
                # Midpoint quadrature in arclength. Tiny A/B straight remnants
                # cannot receive the same total weight as a long curved piece.
                count = self.options.samples_per_module
                for fraction in (np.arange(count) + .5) / count:
                    coordinates.append(lo + (hi - lo) * (traversed + piece.length * fraction) / module_length)
                    locations.append(piece.at(fraction))
                    arc_weights.append(piece.length / count)
                traversed += piece.length
        locations = np.asarray(locations)
        coordinates = np.asarray(coordinates)
        heights = -locations @ projected_down
        band = float(np.median(self.lengths) * self.options.top_band_link_fraction)
        weight = np.maximum(0., heights - heights.max() + band) * np.asarray(arc_weights)
        crest_joint = float(weight @ coordinates / weight.sum())
        top_xy = (weight[:, None] * locations).sum(axis=0) / weight.sum()
        result = dict(joint_coordinate=crest_joint, point_head_m=(axis['point'] + basis @ top_xy).tolist(),
                      top_band_m=band, projected_down_norm=down_norm,
                      top_coordinate_span=coordinates[weight > 0][[0, -1]].tolist())
        if return_curve:
            result.update(curve_head_m=(locations @ basis.T + axis['point']).tolist(),
                          curve_joint_coordinates=coordinates.tolist())
        return result

    def reference_crest(self, window, phase):
        q = np.zeros(self.n)
        indices = np.arange(window[0] - 1, window[1])
        if len(indices) == 6:
            q[indices] = self.amplitude * np.where(indices % 2 == 0, math.cos(phase), math.sin(phase))
        else:
            # Variable-width observers preserve each axis family's half-turn.
            for parity, factor in ((0, math.cos(phase)), (1, math.sin(phase))):
                family = indices[indices % 2 == parity]
                q[family] = math.pi/len(family)*factor
        points = forward(q, self.lengths)
        down, _ = self.infer_down(points, window)
        _, _, section = self.geometry(q, window)
        return self.crest(points, window, section, down)['joint_coordinate']

    def _clear_confirmation(self):
        self.filtered_error = None
        self.pending_direction = 0
        self.pending_since = None

    def update(self, angles_deg, phase_rad, enabled, feedback_time_s):
        angles = np.asarray(angles_deg, float)
        mask = np.asarray(enabled, bool)
        if angles.shape != (self.n,) or mask.shape != angles.shape or not np.isfinite(angles).all():
            raise ValueError('Expected one finite measured angle and enable flag per joint')
        if not math.isfinite(phase_rad) or not math.isfinite(feedback_time_s):
            raise ValueError('Finite phase and feedback timestamp required')
        if self.last_time is not None and feedback_time_s <= self.last_time:
            raise ValueError('Observer requires fresh, increasing feedback timestamps')
        active = np.flatnonzero(mask)
        if len(active) != self.options.width or np.any(np.diff(active) != 1):
            raise ValueError('Expected one contiguous enable mask matching observer width')
        window = (int(active[0] + 1), int(active[-1] + 1))
        dt = 0. if self.last_time is None else feedback_time_s - self.last_time
        changed = self.window is not None and window != self.window
        if changed:
            self._clear_confirmation()
        self.window, self.last_time = window, feedback_time_s
        q = np.radians(angles)
        result = dict(execution='observation_only', executes_window_update=False,
                      feedback_time_s=feedback_time_s, current_window=list(window),
                      external_window_change=changed, valid=False, direction=0,
                      suggested_window=None, state='unobservable', reasons=[])
        try:
            points = forward(q, self.lengths)
            down, quality = self.infer_down(points, window)
            result.update(down_head=down.tolist(), end_quality=quality)
            good, geometry, section = self.geometry(q, window)
            result['current_geometry'] = geometry
            crest = self.crest(points, window, section, down)
            reference = self.reference_crest(window, phase_rad)
            error = crest['joint_coordinate'] - reference
            result.update(crest=crest, reference_joint_coordinate=reference, raw_error_joints=error)
            reasons = result['reasons']
            if min(quality['head_straightness'], quality['tail_straightness']) < self.options.minimum_straightness:
                reasons.append('passive end not straight enough')
            if quality['end_disagreement_deg'] > self.options.maximum_end_disagreement_deg:
                reasons.append('passive ends disagree on down direction')
            if not good:
                reasons.append('current section outside open-shape limits')
            if reasons:
                self._clear_confirmation()
                return result
            result['valid'] = True
            if self.filtered_error is None:
                self.filtered_error = error
            else:
                self.filtered_error += (1 - math.exp(-dt / self.options.error_filter_s)) * (error - self.filtered_error)
            result['filtered_error_joints'] = self.filtered_error
            if abs(self.filtered_error) <= self.options.offset_threshold_joints:
                self.pending_direction, self.pending_since = 0, None
                result['state'] = 'retain'
                return result
            direction = 1 if self.filtered_error > 0 else -1
            candidate = (window[0] + direction, window[1] + direction)
            if candidate[0] < 2 or candidate[1] >= self.n:
                result['state'] = 'blocked_boundary'
                self.pending_direction, self.pending_since = 0, None
                return result
            candidate_good, candidate_geometry, _ = self.geometry(q, candidate)
            result['candidate_window'] = list(candidate)
            result['candidate_geometry'] = candidate_geometry
            if not candidate_good:
                result['state'] = 'blocked_candidate_shape'
                self.pending_direction, self.pending_since = 0, None
                return result
            if self.pending_direction != direction:
                self.pending_direction, self.pending_since = direction, feedback_time_s
            persistent = feedback_time_s - self.pending_since
            result.update(state='confirming', pending_direction=direction, persistence_s=persistent)
            if persistent + 1e-9 >= self.options.persistence_s:
                result.update(state='recommend_head' if direction < 0 else 'recommend_tail',
                              direction=direction, suggested_window=list(candidate))
            return result
        except (ValueError, np.linalg.LinAlgError) as exc:
            self._clear_confirmation()
            result['reasons'].append(str(exc))
            return result

    def configuration(self):
        return dict(options=asdict(self.options), mechanical_lengths_m=self.lengths.tolist(),
                    nominal_amplitude_deg=math.degrees(self.amplitude), input_units='controller degrees; phase radians')


def local_phase(phase, first):
    """Local first hinge is cos; an even first hinge swaps the two families."""
    return (phase if first % 2 else math.pi/2-phase) % (2*math.pi)


def bending_features(angles_deg, lengths, window):
    """Same-winding arc curvature, excluding reverse folds at passive ends.

    The actual active section supplies its plane and winding, not gravity.
    Integrating signed curvature over each physical module gives its turn.
    Taking absolute motor angles would count a reverse tail fold as an arc.
    """
    first, last = window
    n = len(angles_deg)
    q = np.radians(angles_deg)
    section = local_section(q, lengths, first, last)
    points = forward(q, lengths)
    start, stop = max(0, first-3), min(n+2, last+4)
    projected = (points[start:stop]-section['axis']['point']) @ section['axis']['basis'][:, :2]
    curve = continuous_section(projected)
    indices = np.arange(max(1, first-2), min(n, last+2)+1)
    signed = np.array([sum(segment.turn for segment in curve if start+segment.joint == j)
                       for j in indices])
    winding = 1 if signed[(indices >= first) & (indices <= last)].sum() >= 0 else -1
    turns = np.degrees(np.maximum(0., winding*signed))
    mass = turns
    return dict(bending_center_joint=float(indices @ mass / max(mass.sum(), 1e-12)),
                head_boundary_turn_deg=float(turns[indices < first].sum()),
                tail_boundary_turn_deg=float(turns[indices > last].sum()),
                reverse_boundary_turn_deg=float(np.degrees(np.maximum(0., -winding*signed[
                    (indices < first) | (indices > last)])).sum()))


@dataclass(frozen=True)
class MotorObserverOptions:
    crest_threshold_joints: float = .5
    bending_threshold_joints: float = .2
    boundary_threshold_deg: float = 8.
    filter_s: float = .2
    persistence_s: float = .5
    maximum_feedback_gap_s: float = .1


class PhaseShapeReference:
    """Periodic shape offsets in relative body coordinates, with provenance."""
    def __init__(self, values, metadata):
        self.values = np.asarray(values, float)
        if self.values.ndim != 2 or self.values.shape[1] != 3 or not np.isfinite(self.values).all():
            raise ValueError('Reference requires finite crest/bending/boundary phase bins')
        self.metadata = dict(metadata)

    def at(self, phase, first):
        count = len(self.values)
        position = local_phase(phase, first)/(2*math.pi)*count-.5
        left = math.floor(position)
        fraction = position-left
        return (1-fraction)*self.values[left % count] + fraction*self.values[(left+1) % count]

    def to_dict(self):
        return dict(values=self.values.tolist(), metadata=self.metadata)

    @classmethod
    def from_dict(cls, value):
        return cls(value['values'], value['metadata'])


class MotorShapeObserver:
    """Angle-only corroborated recommendation; cannot operate any actuator."""
    def __init__(self, lengths, reference, amplitude_deg=60., options=MotorObserverOptions(), width=6):
        self.lengths = np.asarray(lengths, float)
        self.reference = reference
        self.options = options
        self.geometry_observer = AngleShapeObserver(lengths, amplitude_deg, ObserverOptions(width=width))
        self.filtered = None
        self.pending_direction = 0
        self.pending_since = None
        self.last_time = None

    def update(self, angles_deg, phase_rad, enabled, feedback_time_s):
        # The legacy object supplies geometry/down quality only. Its nominal
        # offset and recommendation are NOT used as migration evidence.
        original = self.geometry_observer.update(angles_deg, phase_rad, enabled, feedback_time_s)
        dt = 0. if self.last_time is None else feedback_time_s-self.last_time
        self.last_time = feedback_time_s
        if dt > self.options.maximum_feedback_gap_s:
            self.filtered, self.pending_direction, self.pending_since = None, 0, None
        window = tuple(original['current_window'])
        result = {key: original[key] for key in ('feedback_time_s', 'current_window',
                  'external_window_change', 'valid', 'reasons', 'execution', 'executes_window_update')}
        result.update(direction=0, suggested_window=None, state='unobservable')
        for key in ('down_head', 'end_quality', 'current_geometry', 'crest'):
            if key in original:
                result[key] = original[key]
        if not original['valid']:
            self.filtered, self.pending_direction, self.pending_since = None, 0, None
            return result
        f = bending_features(angles_deg, self.lengths, window)
        center = sum(window)/2
        values = np.array([original['crest']['joint_coordinate']-center,
                           f['bending_center_joint']-center,
                           f['tail_boundary_turn_deg']-f['head_boundary_turn_deg']])
        baseline = self.reference.at(phase_rad, window[0])
        error = values-baseline
        if self.filtered is None or original['external_window_change']:
            self.filtered, self.pending_direction, self.pending_since = error.copy(), 0, None
        else:
            self.filtered += (1-math.exp(-dt/self.options.filter_s))*(error-self.filtered)
        result.update(features=f, phase_reference=baseline.tolist(), feature_error=error.tolist(),
                      filtered_feature_error=self.filtered.tolist(),
                      filtered_error_joints=float(self.filtered[0]))
        if abs(self.filtered[0]) <= self.options.crest_threshold_joints:
            self.pending_direction, self.pending_since = 0, None
            result['state'] = 'retain'
            return result
        direction = 1 if self.filtered[0] > 0 else -1
        corroborated = (direction*self.filtered[1] >= self.options.bending_threshold_joints or
                        direction*self.filtered[2] >= self.options.boundary_threshold_deg)
        if not corroborated:
            self.pending_direction, self.pending_since = 0, None
            result['state'] = 'reject_crest_without_bending_migration'
            return result
        candidate = (window[0]+direction, window[1]+direction)
        # Both far ends need at least three passive links for the down proxy.
        if candidate[0] < 4 or candidate[1] > len(angles_deg)-3:
            self.pending_direction, self.pending_since = 0, None
            result['state'] = 'blocked_passive_end_length'
            return result
        good, geometry, _ = self.geometry_observer.geometry(np.radians(angles_deg), candidate)
        result.update(candidate_window=list(candidate), candidate_geometry=geometry)
        if not good:
            self.pending_direction, self.pending_since = 0, None
            result['state'] = 'blocked_candidate_shape'
            return result
        if self.pending_direction != direction:
            self.pending_direction, self.pending_since = direction, feedback_time_s
        persistent = feedback_time_s-self.pending_since
        result.update(state='confirming', persistence_s=persistent)
        if persistent+1e-9 >= self.options.persistence_s:
            result.update(state='recommend_head' if direction < 0 else 'recommend_tail',
                          direction=direction, suggested_window=list(candidate))
        return result

    def configuration(self):
        return dict(options=asdict(self.options), frozen_reference=self.reference.to_dict(),
                    gravity_assumption='unforced sufficiently straight passive ends',
                    inputs=['angles_deg', 'phase_rad', 'enabled', 'feedback_time_s'])


@dataclass(frozen=True)
class MotorSlidingOptions:
    cooldown_s: float = 5.
    phase_zero_tolerance: float = .015
    switch_angle_deg: float = 4.
    switch_velocity_deg_s: float = 8.
    stable_feedback_s: float = .06
    minimum_hold_s: float = .1
    handover_timeout_s: float = 20.
    maximum_feedback_age_s: float = .8


class MotorFeedbackSliding:
    def __init__(self, parameters, lengths, reference, execute=True,
                 section_options=SectionOptions(), options=MotorSlidingOptions(), feedback_period_s=.02,
                 allowed_widths=(6,)):
        width = parameters.last_joint-parameters.first_joint+1
        if width not in allowed_widths:
            raise ValueError('Enabled width is outside the explicitly allowed set')
        self.allowed_widths = tuple(allowed_widths)
        self.p, self.lengths, self.options = parameters, np.asarray(lengths), options
        self.section_options, self.execute = section_options, execute
        self.window = (parameters.first_joint, parameters.last_joint)
        self.core = FixedSectionCompliance(parameters, lengths, section_options)
        self.observer = MotorShapeObserver(lengths, reference, parameters.amplitude_deg,
            MotorObserverOptions(maximum_feedback_gap_s=max(.1, 2.5*feedback_period_s)), width=width)
        self.feedback_period_s = feedback_period_s
        self.enabled = parameters.enabled.copy()
        self.phase = parameters.initial_phase_rad
        self.phase_hold = False
        self.targets = parameters.targets(self.phase)
        self.previous_angles, self.last_time = None, None
        self.last_control_time = None
        self.feedback_phase = self.phase
        self.fresh_feedback = False
        self.pending = None
        self.cooldown_until = parameters.settle_s
        self.events = []
        self.observation = None
        self.angle_velocity_deg_s = np.zeros(parameters.joints)
        self.angle_velocity_available = False
        self.last_gate = None
        self.fault = None

    def _handover(self, angles, time_s):
        request = self.pending
        new = tuple(request['to'])
        new_mask = (np.arange(1, self.p.joints+1) >= new[0]) & (np.arange(1, self.p.joints+1) <= new[1])
        entering = np.flatnonzero(new_mask & ~self.enabled)
        exiting = np.flatnonzero(self.enabled & ~new_mask)
        changed = np.r_[entering, exiting]
        factor = math.cos(self.phase) if exiting[0] % 2 == 0 else math.sin(self.phase)
        if not self.phase_hold and abs(factor) <= self.options.phase_zero_tolerance:
            self.phase_hold = True
            request['hold_since'] = time_s
            self.events.append(dict(kind='phase_hold', time_s=time_s, phase_rad=self.phase,
                                    entering=(entering+1).tolist(), exiting=(exiting+1).tolist()))
        if self.phase_hold and self.fresh_feedback:
            q = np.radians(angles)
            good, geometry, _ = self.observer.geometry_observer.geometry(q, new)
            preload = self.targets.copy()
            preload[entering] = q[entering]
            preload[~new_mask] = q[~new_mask]
            next_core = FixedSectionCompliance(replace(self.p, first_joint=new[0], last_joint=new[1]),
                                              self.lengths, self.section_options)
            command_good, _ = next_core._check_goal(preload)
            self.last_gate = dict(time_s=time_s, actual_candidate=geometry,
                switch_angles_deg=angles[changed].tolist(),
                switch_velocity_deg_s=self.angle_velocity_deg_s[changed].tolist(),
                velocity_from_continuous_fresh_feedback=self.angle_velocity_available,
                candidate_open=good, command_open=bool(command_good),
                unloading_observed=False)
            direction = new[0]-self.window[0]
            evidence = self.observation is not None and self.observation['valid'] and self.observation['direction'] == direction
            ready = (good and command_good and evidence and self.angle_velocity_available and
                np.max(abs(angles[changed])) <= self.options.switch_angle_deg and
                np.max(abs(self.angle_velocity_deg_s[changed])) <= self.options.switch_velocity_deg_s and
                time_s-request['hold_since'] >= self.options.minimum_hold_s)
            if ready:
                if request['ready_since'] is None:
                    request['ready_since'] = self.last_time
                if self.last_time-request['ready_since']+1e-9 >= self.options.stable_feedback_s:
                    old = self.window
                    next_core.targets = preload.copy()
                    next_core.previous_axis = self.core.previous_axis
                    self.core, self.window, self.enabled = next_core, new, new_mask
                    self.targets = preload.copy()
                    self.events.append(dict(kind='complete', time_s=time_s, old=list(old), to=list(new),
                                            gate=self.last_gate, strategy='motor-angle geometric trial'))
                    self.pending, self.phase_hold = None, False
                    self.cooldown_until = time_s+self.options.cooldown_s
            else:
                request['ready_since'] = None
        if self.pending is not None and time_s-request['time_s'] >= self.options.handover_timeout_s:
            self.events.append(dict(kind='reject', time_s=time_s, to=list(new),
                                    reason='motor-angle handover timed out', last_gate=self.last_gate))
            self.pending, self.phase_hold = None, False
            self.cooldown_until = time_s+self.options.cooldown_s

    @property
    def automatic(self):
        return self.execute

    @automatic.setter
    def automatic(self, value):
        self.execute = bool(value)
        if not self.execute and self.pending is not None:
            self.events.append(dict(kind='cancel', time_s=self.last_control_time,
                                    to=self.pending['to'], reason='automatic selection disabled'))
            self.pending, self.phase_hold = None, False

    def update(self, angles_deg, feedback_time_s, commanded_phase_rad=None):
        """Compatibility: one fresh feedback and one control tick together."""
        if self.last_time is not None and feedback_time_s <= self.last_time:
            raise ValueError('Fresh increasing motor-feedback time required')
        return self.advance(angles_deg, feedback_time_s, feedback_time_s, commanded_phase_rad)

    def advance(self, angles_deg, feedback_time_s, control_time_s,
                commanded_phase_rad=None, phase_at_feedback=None, freeze_targets=False):
        """Control tick may hold a motor sample; observation may not invent one.

        Both times use the caller's same clock. The known phase at acquisition
        is separate from the next commanded phase. Mask changes require NEW
        motor feedback; target filtering and phase gating run at control rate.
        """
        angles = np.asarray(angles_deg, float)
        if angles.shape != (self.p.joints,) or not np.isfinite(angles).all():
            raise ValueError('One finite actual motor angle per joint required')
        if not all(math.isfinite(t) for t in (feedback_time_s, control_time_s)) or feedback_time_s > control_time_s+1e-8:
            raise ValueError('Finite, nonfuture feedback/control times required')
        if self.last_time is not None and feedback_time_s < self.last_time:
            raise ValueError('Motor-feedback time cannot run backwards')
        if self.last_control_time is not None and control_time_s <= self.last_control_time:
            raise ValueError('Control clock must increase')
        if control_time_s-feedback_time_s > self.options.maximum_feedback_age_s:
            raise TimeoutError('Motor angles too old for sliding control')
        dt = .02 if self.last_control_time is None else min(.02, control_time_s-self.last_control_time)
        previous_time = control_time_s-dt if self.last_control_time is None else self.last_control_time
        observed_phase = self.phase
        self.fresh_feedback = self.last_time is None or feedback_time_s > self.last_time
        if self.fresh_feedback:
            feedback_dt = .02 if self.last_time is None else feedback_time_s-self.last_time
            gap = feedback_dt > self.observer.options.maximum_feedback_gap_s
            self.angle_velocity_available = self.previous_angles is not None and not gap
            if gap and self.pending is not None:
                self.pending['ready_since'] = None
            if self.previous_angles is not None:
                self.angle_velocity_deg_s = (angles-self.previous_angles)/feedback_dt
            self.previous_angles, self.last_time = angles.copy(), feedback_time_s
            self.feedback_phase = observed_phase if phase_at_feedback is None else phase_at_feedback
            if control_time_s >= self.p.settle_s:
                self.observation = self.observer.update(angles, self.feedback_phase, self.enabled, feedback_time_s)
        elif not np.array_equal(angles, self.previous_angles):
            raise ValueError('Changed angles with an unchanged feedback timestamp')
        self.last_control_time = control_time_s
        if freeze_targets:
            if self.pending is not None:
                self.pending['ready_since'] = None
            return self.targets.copy(), self.enabled.copy()
        if commanded_phase_rad is not None and not math.isfinite(commanded_phase_rad):
            raise ValueError('Finite commanded phase required')
        if commanded_phase_rad is not None:
            if not self.phase_hold:
                self.phase = float(commanded_phase_rad)
        elif not self.phase_hold:
            rolling_dt = min(dt, max(0., control_time_s-max(previous_time, self.p.settle_s)))
            self.phase += self.p.omega_rad_s*rolling_dt
        if control_time_s < self.p.settle_s:
            self.targets = self.p.targets(self.phase)
            return self.targets.copy(), self.enabled.copy()
        if (self.execute and self.observation is not None and self.pending is None and control_time_s >= self.cooldown_until
                and self.observation['suggested_window'] is not None and self.fault is None):
            self.pending = dict(to=self.observation['suggested_window'], time_s=control_time_s,
                                ready_since=None, hold_since=None)
            self.events.append(dict(kind='prepare', time_s=control_time_s, old=list(self.window),
                                    to=self.pending['to'], observation=self.observation))
        if self.pending is not None:
            self._handover(angles, control_time_s)
        previous_targets = self.targets.copy()
        proposed = self.core.update(np.radians(angles), self.phase, control_time_s, dt)
        valid, _ = self.core._check_goal(proposed)
        if self.core.status != 'running' or not valid:
            if self.fault is None:
                self.events.append(dict(kind='shape_hold', time_s=control_time_s, reason=self.core.status))
            self.fault = self.core.status if self.core.status != 'running' else 'filtered target outside opening'
            self.targets[self.enabled] = previous_targets[self.enabled]
            self.targets[~self.enabled] = np.radians(angles)[~self.enabled]
        else:
            self.fault = None
            self.targets = proposed
        if self.enabled.sum() not in self.allowed_widths:
            raise RuntimeError('Allowed-width mask invariant violated')
        return self.targets.copy(), self.enabled.copy()

    def snapshot(self):
        return dict(window=list(self.window), phase_rad=self.phase, phase_hold=self.phase_hold,
                    pending=self.pending, observation=self.observation, last_gate=self.last_gate,
                    fault=self.fault, compliance=self.core.snapshot(), options=asdict(self.options),
                    feedback_time_s=self.last_time, control_time_s=self.last_control_time,
                    new_feedback_this_tick=self.fresh_feedback, feedback_phase_rad=self.feedback_phase,
                    feedback_period_s=self.feedback_period_s,
                    feedback_source='measured joint angles; qdot from fresh angle differences',
                    observes_contact_or_unloading=False, execute=self.execute)


@dataclass(frozen=True)
class HalfArcParameters(ArcParameters):
    mechanical_lengths_m: tuple = (.0645,)*21

    def __post_init__(self):
        super().__post_init__()
        if self.amplitude_deg != 60. or self.direction != 1 or self.omega_rad_s != .6:
            raise ValueError('This experiment uses a fixed pi budget and the forward 0.6rad/s calibration')
        if len(self.mechanical_lengths_m) != self.joints+1 or not np.isfinite(self.mechanical_lengths_m).all() or min(self.mechanical_lengths_m) <= 0:
            raise ValueError('One positive physical length per backbone segment required')

    def targets(self, phase):
        if not math.isfinite(phase):
            raise ValueError('Finite phase required')
        lengths = np.asarray(self.mechanical_lengths_m)
        module = .5*(lengths[:-1]+lengths[1:])
        target = np.zeros(self.joints)
        active = np.flatnonzero(self.enabled)
        for parity, factor in ((0, math.cos(phase)), (1, math.sin(phase))):
            family = active[active % 2 == parity]
            target[family] = math.pi*module[family]/module[family].sum()*factor
        return target


def circle_shape(section):
    xy = section['projected'][1:-1]
    matrix = np.column_stack([2*xy, np.ones(len(xy))])
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.sum(xy*xy, axis=1), rcond=None)
    if rank < 3:
        raise ValueError('Circle fit degenerate')
    radius = math.sqrt(max(0., solution[2]+solution[:2]@solution[:2]))
    distance = np.linalg.norm(xy-solution[:2], axis=1)
    return dict(radius_m=radius, circle_rms_m=float(np.sqrt(np.mean((distance-radius)**2))),
                turn_deg=math.degrees(section['total_turn']))


class RollingShapeObserver:
    def __init__(self, lengths, reference, amplitude_deg=60.,
                 options=MotorObserverOptions(), width=6):
        self.lengths = np.asarray(lengths, float)
        self.reference, self.options = reference, options
        self.geometry_observer = AngleShapeObserver(lengths, amplitude_deg, ObserverOptions(width=width))
        self.width = width
        self.last_time = self.window = self.filtered = self.pending_since = None
        self.pending_direction = 0
        self.filtered_crest = None

    def passive_down(self, points, window, section):
        """Conditional gravity proxy; one straight long end is not a sensor.

        The open-side hemisphere rejects a pulled-up end, but cannot rule out
        every force/contact that tilts a straight arm. Truth is evaluated later.
        """
        ends = []
        arc = section['points'][1:-1]
        opening = (arc[0]+arc[-1])/2-arc.mean(axis=0)
        if np.linalg.norm(opening) < 1e-12:
            return None, dict(source='degenerate_opening_direction', candidates=[])
        opening /= np.linalg.norm(opening)
        for name, available in (('head', window[0]-1), ('tail', len(self.lengths)-1-window[1])):
            count = min(4, available)
            if count < 3: continue
            samples = points[:count+1][::-1] if name == 'head' else points[-count-1:]
            chord = samples[-1]-samples[0]
            length = np.linalg.norm(chord)
            if length < 1e-12: continue
            straightness = float(length/np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())
            direction = chord/length
            alignment = float(direction@opening)
            if straightness >= .98 and alignment >= .5:
                ends.append(dict(name=name, count=count, straightness=straightness,
                                 opening_alignment=alignment, direction=direction.tolist()))
        if not ends: return None, dict(source='no_trusted_passive_end', candidates=[])
        if len(ends) == 2:
            a, b = [np.asarray(e['direction']) for e in ends]
            disagreement = math.degrees(math.acos(float(np.clip(a@b, -1., 1.))))
            if disagreement <= 25.:
                direction = a+b; direction /= np.linalg.norm(direction)
                return direction, dict(source='two_consistent_ends', end_disagreement_deg=disagreement, candidates=ends)
        chosen = max(ends, key=lambda e: (e['count'], e['straightness']))
        return np.asarray(chosen['direction']), dict(source='single_'+chosen['name'], candidates=ends,
                                                   does_not_prove_gravity=True)

    def update(self, angles_deg, phase_rad, enabled, feedback_time_s):
        angles, mask = np.asarray(angles_deg, float), np.asarray(enabled, bool)
        active = np.flatnonzero(mask)
        if (angles.shape != (len(self.lengths)-1,) or mask.shape != angles.shape or
                not np.isfinite(angles).all() or not math.isfinite(phase_rad) or not math.isfinite(feedback_time_s)):
            raise ValueError('Finite measured angles, phase and timestamp required')
        if len(active) != self.width or np.any(np.diff(active) != 1):
            raise ValueError('Contiguous mask must match observer width')
        if self.last_time is not None and feedback_time_s <= self.last_time:
            raise ValueError('New increasing feedback required')
        window = (int(active[0]+1), int(active[-1]+1))
        dt = 0. if self.last_time is None else feedback_time_s-self.last_time
        changed = self.window is not None and window != self.window
        if changed or dt > self.options.maximum_feedback_gap_s:
            self.filtered, self.pending_since, self.pending_direction = None, None, 0
            self.filtered_crest = None
        self.window, self.last_time = window, feedback_time_s
        r = dict(feedback_time_s=feedback_time_s, current_window=list(window), external_window_change=changed,
                 execution='observation_only', executes_window_update=False, valid=False, reasons=[],
                 direction=0, suggested_window=None, state='unobservable',
                 localisation='intrinsic_same_winding_bending', observes_absolute_down=False)
        try:
            good, geometry, section = self.geometry_observer.geometry(np.radians(angles), window)
            r['current_geometry'] = geometry
            if not good:
                raise ValueError('Current body outside open-shape limits')
            features = bending_features(angles, self.lengths, window)
            center = sum(window)/2
            values = np.array([features['bending_center_joint']-center,
                               features['tail_boundary_turn_deg']-features['head_boundary_turn_deg']])
            baseline = self.reference.at(phase_rad, window[0])[1:]
            error = values-baseline
            if self.filtered is None:
                self.filtered = error.copy()
            else:
                self.filtered += (1-math.exp(-dt/self.options.filter_s))*(error-self.filtered)
            r.update(valid=True, features=features, phase_reference=baseline.tolist(), feature_error=error.tolist(),
                     filtered_feature_error=self.filtered.tolist(), filtered_error_joints=float(self.filtered[0]),
                     estimated_bending_center_joint=float(center+self.filtered[0]))
            r['position_valid'] = not self.reference.metadata.get('transition_only', False)
            if not r['position_valid']:
                r['state'] = 'transition_geometry_only'
                return r
            down, quality = self.passive_down(forward(np.radians(angles), self.lengths), window, section)
            r['end_quality'] = quality
            primary, corroborated = float(self.filtered[0]), False
            if down is not None:
                crest = self.geometry_observer.crest(forward(np.radians(angles), self.lengths), window, section, down)
                crest_error = crest['joint_coordinate']-center-self.reference.at(phase_rad, window[0])[0]
                if self.filtered_crest is None:
                    self.filtered_crest = crest_error
                else:
                    self.filtered_crest += (1-math.exp(-dt/self.options.filter_s))*(crest_error-self.filtered_crest)
                primary = float(self.filtered_crest)
                r.update(down_head=down.tolist(), crest=crest, localisation='conditional_crest_with_bending',
                         gravity_proxy_assumption='qualified passive end hangs down', filtered_error_joints=primary,
                         observes_absolute_down=False)
                sign = 1 if primary > 0 else -1
                corroborated = (sign*self.filtered[0] >= self.options.bending_threshold_joints or
                                sign*self.filtered[1] >= self.options.boundary_threshold_deg)
            else:
                self.filtered_crest = None
                sign = 1 if primary > 0 else -1
                corroborated = sign*self.filtered[1] >= self.options.boundary_threshold_deg
            direction = 1 if primary > 0 else -1
            if abs(primary) < self.options.crest_threshold_joints or not corroborated:
                self.pending_since, self.pending_direction = None, 0
                r['state'] = 'retain'
                return r
            candidate = (window[0]+direction, window[1]+direction)
            if candidate[0] < 2 or candidate[1] > len(angles)-1:
                self.pending_since, self.pending_direction = None, 0
                r['state'] = 'blocked_robot_boundary'
                return r
            candidate_good, candidate_geometry, _ = self.geometry_observer.geometry(np.radians(angles), candidate)
            r.update(candidate_window=list(candidate), candidate_geometry=candidate_geometry)
            if not candidate_good:
                self.pending_since, self.pending_direction = None, 0
                r['state'] = 'blocked_candidate_shape'
                return r
            if self.pending_direction != direction:
                self.pending_direction, self.pending_since = direction, feedback_time_s
            persistent = feedback_time_s-self.pending_since
            r.update(state='confirming', persistence_s=persistent)
            if persistent+1e-9 >= self.options.persistence_s:
                r.update(state='recommend_head' if direction < 0 else 'recommend_tail', direction=direction,
                         suggested_window=list(candidate))
            return r
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.filtered, self.pending_since, self.pending_direction = None, None, 0
            self.filtered_crest = None
            r['valid'], r['position_valid'] = False, False
            r['reasons'].append(str(exc))
            return r

    def configuration(self):
        return dict(options=asdict(self.options), frozen_reference=self.reference.to_dict(),
                    gravity_assumption='qualified long passive end; otherwise strong intrinsic bending only',
                    position_definition='conditional gravity crest with same-winding corroboration',
                    inputs=['angles_deg', 'phase_rad', 'enabled', 'feedback_time_s'])


class PhaseBalancedCount:
    def __init__(self, groups=4, minimum_samples_per_group=2,
                 phase_span_rad=math.pi, maximum_feedback_gap_s=.5):
        self.groups = groups
        self.minimum_samples_per_group = minimum_samples_per_group
        self.phase_span_rad = phase_span_rad
        self.maximum_feedback_gap_s = maximum_feedback_gap_s
        self.reset()

    def reset(self):
        self.samples = []
        self.last_time = self.last_phase = None

    def update(self, count, turn_deg, phase_rad, time_s, nominal_turn_deg=180.):
        if not all(math.isfinite(v) for v in (count, turn_deg, phase_rad, time_s, nominal_turn_deg)):
            raise ValueError('Finite angle-derived count, turn, phase and time required')
        if self.last_time is not None:
            if time_s <= self.last_time:
                raise ValueError('Only new feedback may update phase statistics')
            if time_s-self.last_time > self.maximum_feedback_gap_s or phase_rad < self.last_phase-1e-8:
                self.reset()
        sample = dict(count=float(count), turn_deg=float(turn_deg), nominal_turn_deg=float(nominal_turn_deg),
                      phase_rad=float(phase_rad), time_s=float(time_s))
        if self.samples and abs(phase_rad-self.samples[-1]['phase_rad']) <= 1e-8:
            self.samples[-1] = sample
        else:
            self.samples.append(sample)
        self.last_time, self.last_phase = time_s, phase_rad
        self.samples = [s for s in self.samples if s['phase_rad'] >= phase_rad-self.phase_span_rad-1e-8]
        return self.summary()

    def summary(self):
        buckets = [[] for _ in range(self.groups)]
        for sample in self.samples:
            index = int((sample['phase_rad'] % self.phase_span_rad)/self.phase_span_rad*self.groups) % self.groups
            buckets[index].append(sample)
        count_medians = [None if not b else float(np.median([s['count'] for s in b])) for b in buckets]
        turn_medians = [None if not b else float(np.median([s['turn_deg'] for s in b])) for b in buckets]
        turn_error_medians = [None if not b else float(np.median([s['turn_deg']-s['nominal_turn_deg'] for s in b])) for b in buckets]
        ready = all(len(b) >= self.minimum_samples_per_group for b in buckets)
        return dict(ready=ready, phase_period_rad=self.phase_span_rad, group_sample_counts=[len(b) for b in buckets],
                    group_count_medians=count_medians, group_turn_medians_deg=turn_medians,
                    balanced_count_joints=None if not ready else float(np.mean(count_medians)),
                    balanced_turn_deg=None if not ready else float(np.mean(turn_medians)),
                    group_turn_error_medians_deg=turn_error_medians,
                    balanced_turn_error_deg=None if not ready else float(np.mean(turn_error_medians)),
                    maximum_group_count_joints=None if not ready else max(count_medians),
                    represented_phase_span_rad=0. if not self.samples else self.samples[-1]['phase_rad']-self.samples[0]['phase_rad'],
                    new_samples=len(self.samples), observes_contact=False)


@dataclass(frozen=True)
class PostureWindowOptions:
    count_hysteresis_joints: float = .5
    expansion_turn_error_deg: float = -5.
    contraction_turn_error_deg: float = 5.
    circle_rms_max_m: float = .012
    persistence_s: float = 1.
    cooldown_s: float = 5.
    trial_grace_s: float = 1.
    trial_review_s: float = 5.
    nominal_joint_margin_rad: float = .01
    shorter_trial_count_margin_joints: float = .25
    shorter_trial_turn_error_deg: float = -2.
    contract_exit_angle_deg: float = 12.


def nominal_angles(lengths, window, phase):
    """Distribute each alternating axis family's pi budget by module length."""
    lengths = np.asarray(lengths, float)
    module = .5*(lengths[:-1]+lengths[1:])
    indices = np.arange(window[0]-1, window[1])
    q = np.zeros(len(module))
    for parity, factor in ((0, math.cos(phase)), (1, math.sin(phase))):
        family = indices[indices % 2 == parity]
        if not len(family):
            raise ValueError('A rolling section requires both axis families')
        q[family] = math.pi*module[family]/module[family].sum()*factor
    return q


def mechanically_feasible(lengths, window, margin=.01):
    """A geometric boundary and full-cycle joint-limit check, not a width list."""
    n = len(lengths)-1
    if window[0] < 2 or window[1] >= n or window[1]-window[0]+1 < 3:
        return False
    try:
        peak = np.maximum(abs(nominal_angles(lengths, window, 0.)),
                          abs(nominal_angles(lengths, window, math.pi/2)))
        return bool(peak.max() <= math.pi/2-margin)
    except ValueError:
        return False


class GeneralPhaseReference:
    """Reuse any available calibration; transfer its residual to other widths.

    There is no stable-width whitelist. Unseen widths use their own nominal
    geometry plus an interpolated (or nearest) frozen calibration residual.
    Transfer across widths is an explicit assumption to audit in simulation.
    """
    def __init__(self, lengths, width, calibrated):
        self.lengths = np.asarray(lengths, float)
        self.width = width
        self.calibrated = calibrated
        self.helper = AngleShapeObserver(lengths)
        self.cache = {}
        self.metadata = dict(width=width, method='nominal geometry plus frozen residual',
                             calibrated_widths=sorted(calibrated),
                             transferred=width not in calibrated, online_learning=False)

    def ideal(self, width, phase, first):
        # Use the same starting parity, with enough space for boundary modules.
        start = 2 if first % 2 == 0 else 3
        if start+width-1 >= len(self.lengths)-1:
            start = 2
        window = (start, start+width-1)
        q = nominal_angles(self.lengths, window, phase)
        _, _, section = self.helper.geometry(q, window)
        points = forward(q, self.lengths)
        arc = section['points'][1:-1]
        down = (arc[0]+arc[-1])/2-arc.mean(axis=0)
        down /= np.linalg.norm(down)
        crest = self.helper.crest(points, window, section, down)['joint_coordinate']
        f = bending_features(np.degrees(q), self.lengths, window)
        center = sum(window)/2
        return np.array([crest-center, f['bending_center_joint']-center,
                         f['tail_boundary_turn_deg']-f['head_boundary_turn_deg']])

    def at(self, phase, first):
        if self.width in self.calibrated:
            return self.calibrated[self.width].at(phase, first)
        key = (round(phase % (2*math.pi), 8), first % 2)
        if key not in self.cache:
            widths = sorted(self.calibrated)
            residuals = np.array([self.calibrated[w].at(phase, first)-self.ideal(w, phase, first)
                                  for w in widths])
            residual = np.array([np.interp(self.width, widths, residuals[:, j]) for j in range(3)])
            self.cache[key] = self.ideal(self.width, phase, first)+residual
        return self.cache[key].copy()

    def to_dict(self):
        return dict(metadata=self.metadata,
                    calibrations={str(w): r.to_dict() for w, r in self.calibrated.items()})


class PostureWindow(MotorFeedbackSliding):
    """Adjacent position/length selection with continuous rolling handover."""
    def __init__(self, parameters, lengths, references, feedback_period_s=.2,
                 posture_options=PostureWindowOptions(), **kwargs):
        if not references:
            raise ValueError('At least one frozen normal-rolling reference is required')
        self.references = dict(references)
        self.posture_options = posture_options
        window = (parameters.first_joint, parameters.last_joint)
        if not mechanically_feasible(lengths, window, posture_options.nominal_joint_margin_rad):
            raise ValueError('Initial arc cannot meet the full-cycle mechanical limits')
        for phase in (0., math.pi/2):
            if not np.allclose(parameters.targets(phase), nominal_angles(lengths, window, phase), atol=1e-10):
                raise ValueError('Use length-normalized half-arc parameters for every width')
        width = window[1]-window[0]+1
        reference = GeneralPhaseReference(lengths, width, references)
        super().__init__(parameters, lengths, reference, execute=False,
                         allowed_widths=range(3, parameters.joints-1),
                         feedback_period_s=feedback_period_s, **kwargs)
        self.selection = None
        self.filtered_count = None
        self.intent = self.intent_since = None
        self.trial = None
        self.resize_fault = None
        self.adaptive_enabled = True
        self._new_observer()

    @property
    def automatic(self):
        return self.adaptive_enabled

    @automatic.setter
    def automatic(self, value):
        self.adaptive_enabled = bool(value)
        if not self.adaptive_enabled:
            self.pending = None
            self.intent = self.intent_since = None

    def _new_observer(self):
        width = self.window[1]-self.window[0]+1
        reference = GeneralPhaseReference(self.lengths, width, self.references)
        self.observer = RollingShapeObserver(self.lengths, reference, width=width,
            options=MotorObserverOptions(maximum_feedback_gap_s=2.5*self.feedback_period_s))
        self.observation = None
        self.filtered_count = None
        self.count_history = PhaseBalancedCount(maximum_feedback_gap_s=2.5*self.feedback_period_s)
        self.count_observation = None
        self.intent = self.intent_since = None

    def _candidate(self, angles, window):
        if not mechanically_feasible(self.lengths, window, self.posture_options.nominal_joint_margin_rad):
            return dict(window=list(window), eligible=False, reason='mechanical boundary or peak joint limit')
        try:
            good, geometry, section = self.observer.geometry_observer.geometry(np.radians(angles), window)
            down, quality = self.observer.passive_down(forward(np.radians(angles), self.lengths), window, section)
            return dict(window=list(window), eligible=bool(good and down is not None), geometry=geometry,
                        end_source=quality['source'], reason=None if good and down is not None else 'shape or passive reference')
        except (ValueError, np.linalg.LinAlgError) as exc:
            return dict(window=list(window), eligible=False, reason=str(exc))

    def _observe_count(self, angles, time_s):
        """Collect even while a handover, trial or cooldown is in progress."""
        o = self.observation
        if not o or tuple(o['current_window']) != self.window:
            return
        if not o['valid']:
            self.count_history.reset()
            self.count_observation = dict(valid=False, reason='current shape not reliable')
            return
        width = self.window[1]-self.window[0]+1
        q = np.radians(angles)
        actual = circle_shape(local_section(q, self.lengths, *self.window))
        nominal = q.copy()
        nominal[self.enabled] = nominal_angles(self.lengths, self.window, self.feedback_phase)[self.enabled]
        reference = circle_shape(local_section(nominal, self.lengths, *self.window))
        radius_ratio = actual['radius_m']/reference['radius_m']
        turn_ratio = reference['turn_deg']/max(actual['turn_deg'], 1e-8)
        estimated_count = width*.5*(radius_ratio+turn_ratio)
        if actual['circle_rms_m'] > self.posture_options.circle_rms_max_m:
            self.count_history.reset()
            self.count_observation = dict(valid=False, reason='body circle fit not reliable', actual_body_circle=actual)
            return
        phase_statistics = self.count_history.update(estimated_count, actual['turn_deg'], self.feedback_phase,
                                                    self.last_time, reference['turn_deg'])
        self.filtered_count = phase_statistics['balanced_count_joints']
        self.count_observation = dict(valid=True, time_s=self.last_time, window=list(self.window),
            raw_count_joints=estimated_count, actual_body_circle=actual, nominal_body_circle=reference,
            phase_statistics=phase_statistics, observes_support=False, observes_pipe_radius=False)

    def _select(self, angles, time_s):
        o, count = self.observation, self.count_observation
        if not o or not o['valid'] or 'down_head' not in o:
            self.intent = self.intent_since = None
            self.selection = dict(state='retain_untrusted_shape_or_down', observes_support=False)
            return
        width = self.window[1]-self.window[0]+1
        option = self.posture_options
        stats = count.get('phase_statistics', {}) if count and count['valid'] else {}
        ready = stats.get('ready', False)
        actual = None if not count else count.get('actual_body_circle')
        reference = None if not count else count.get('nominal_body_circle')
        delta_width = 0
        length_reason = 'multi_phase_coverage_incomplete' if not ready else 'no_persistent_count_error'
        if ready and self.filtered_count >= width+option.count_hysteresis_joints and stats['balanced_turn_error_deg'] <= option.expansion_turn_error_deg:
            delta_width = 1
            length_reason = 'multi_phase_arc_deficit'
        elif ready:
            strongly_compact = self.filtered_count <= width-option.count_hysteresis_joints and stats['balanced_turn_error_deg'] >= option.contraction_turn_error_deg
            shorter_trial = (stats['maximum_group_count_joints'] <= width+option.shorter_trial_count_margin_joints and
                             stats['balanced_turn_error_deg'] >= option.shorter_trial_turn_error_deg)
            first, last = self.window
            can_shorten = any(mechanically_feasible(self.lengths, w, option.nominal_joint_margin_rad)
                              for w in ((first+1, last), (first, last-1)))
            if (strongly_compact or shorter_trial) and can_shorten:
                delta_width = -1
                length_reason = 'multi_phase_compact_arc' if strongly_compact else 'guarded_adjacent_shorter_trial'
        offset = float(o['filtered_error_joints'])
        sign = 1 if offset > 0 else -1
        bending = o['filtered_feature_error']
        corroborated = sign*bending[0] >= self.observer.options.bending_threshold_joints or sign*bending[1] >= self.observer.options.boundary_threshold_deg
        migration = sign if abs(offset) >= self.observer.options.crest_threshold_joints and corroborated else 0
        first, last = self.window
        if delta_width > 0:
            windows = [(first-1, last), (first, last+1)]
            operation = 'expand'
        elif delta_width < 0:
            windows = [(first+1, last), (first, last-1)]
            operation = 'contract'
        elif migration:
            windows = [(first+migration, last+migration)]
            operation = 'slide'
        else:
            windows, operation = [], 'retain'
        candidates = [self._candidate(angles, w) for w in windows]
        desired_center = sum(self.window)/2+(offset if migration else 0.)
        for c in candidates:
            w = c['window']
            c['center_error_joints'] = abs(sum(w)/2-desired_center)
            c['score'] = c['center_error_joints']+.01*abs(c.get('geometry', {}).get('turn_deg', 180.)-180.)
        usable = [c for c in candidates if c['eligible']]
        best = min(usable, key=lambda c: c['score']) if usable else None
        self.selection = dict(state=operation if best else 'retain' if not windows else 'blocked_candidates',
            current_window=list(self.window), estimated_count_joints=None if not count else count.get('raw_count_joints'),
            filtered_count_joints=self.filtered_count, actual_body_circle=actual, nominal_body_circle=reference,
            phase_statistics=stats, length_reason=length_reason,
            migration_direction=migration, conditional_crest_error_joints=offset,
            candidates=candidates, chosen=None if best is None else best['window'],
            observes_pipe_radius=False, observes_support=False)
        # A length decision persists across a phase-dependent choice of side.
        # The fresh candidate and continuity gate decide which side can switch.
        intent = None if best is None else (operation, width+delta_width) if delta_width else (operation, *best['window'])
        if intent != self.intent:
            self.intent, self.intent_since = intent, time_s
        if intent is not None and time_s-self.intent_since+1e-8 >= option.persistence_s:
            self.pending = dict(kind='slide' if operation == 'slide' else 'resize',
                                operation=operation, to=best['window'], time_s=time_s,
                                ready_since=None, evidence=copy.deepcopy(self.selection))
            self.events.append(dict(kind='prepare', time_s=time_s, old=list(self.window),
                                    to=best['window'], operation=operation, selection=copy.deepcopy(self.selection)))
            self.selection['state'] = 'waiting_moving_handover'

    def _handover(self, angles, time_s):
        request, new = self.pending, tuple(self.pending['to'])
        if self.fresh_feedback:
            p = replace(self.p, first_joint=new[0], last_joint=new[1])
            entering = np.flatnonzero(p.enabled & ~self.enabled)
            exiting = np.flatnonzero(self.enabled & ~p.enabled)
            changed = len(entering)+len(exiting)
            if (request['kind'] == 'slide' and (len(entering), len(exiting)) != (1, 1)) or (request['kind'] != 'slide' and changed != 1):
                raise RuntimeError('Only adjacent position or single-boundary length changes are allowed')
            candidate = self._candidate(angles, new)
            q = np.radians(angles)
            preload = self.targets.copy()
            preload[entering], preload[~p.enabled] = q[entering], q[~p.enabled]
            core = FixedSectionCompliance(p, self.lengths, self.section_options)
            core.targets, core.previous_axis = preload.copy(), self.core.previous_axis
            preview_core = copy.deepcopy(core)
            preview = preview_core.update(q, self.phase, time_s, .02)
            step_deg = float(np.degrees(abs(preview[p.enabled]-preload[p.enabled])).max())
            exit_bound = self.posture_options.contract_exit_angle_deg if request['operation'] == 'contract' else self.options.switch_angle_deg
            compatible = bool(np.max(abs(angles[exiting])) <= exit_bound) if len(exiting) else abs(math.cos(self.phase) if entering[0] % 2 == 0 else math.sin(self.phase)) <= .1
            current_trusted = self.observation is not None and self.observation['valid'] and 'down_head' in self.observation
            evidence = request.get('rollback', False) or current_trusted
            shared = self.enabled & p.enabled
            self.last_gate = dict(time_s=time_s, candidate_geometry=candidate.get('geometry'),
                entering_joint=(entering+1).tolist(), exiting_joint=(exiting+1).tolist(),
                entering_angle_deg=angles[entering].tolist(), exiting_angle_deg=angles[exiting].tolist(),
                exiting_velocity_deg_s=self.angle_velocity_deg_s[exiting].tolist(),
                exit_angle_bound_deg=exit_bound,
                command_open=bool(core._check_goal(preload)[0]), next_command_open=bool(preview_core._check_goal(preview)[0]),
                next_target_step_deg=step_deg, phase_compatible=compatible, evidence_valid=bool(evidence),
                shared_preload_jump_rad=float(abs(preload[shared]-self.targets[shared]).max()),
                unloading_observed=False, phase_paused=False)
            ready = (candidate['eligible'] and evidence and compatible and self.angle_velocity_available and
                     self.last_gate['command_open'] and self.last_gate['next_command_open'] and preview_core.status == 'running' and
                     step_deg <= self.section_options.target_rate_deg_s*.02+1e-8)
            if ready:
                old = self.window
                observation = copy.deepcopy(self.observation)
                self.core, self.window, self.enabled, self.targets = core, new, p.enabled.copy(), preload
                kind = 'complete' if request['kind'] == 'slide' else 'resize_complete'
                self.events.append(dict(kind=kind, operation=request['operation'], time_s=time_s,
                    old=list(old), to=list(new), phase_rad=self.phase, observation=observation,
                    gate=copy.deepcopy(self.last_gate), selection=request['evidence'], rollback=request.get('rollback', False)))
                self.pending = None
                self.cooldown_until = time_s+self.posture_options.cooldown_s
                self.fault = None
                self.trial = None if request.get('rollback') else dict(old=list(old), new=list(new),
                    start_time_s=time_s, bad_samples=0, samples=0, good=True)
                self._new_observer()
        if self.pending is not None and time_s-request['time_s'] >= self.options.handover_timeout_s:
            self.events.append(dict(kind='reject', time_s=time_s, transaction=copy.deepcopy(request), gate=self.last_gate))
            if request.get('rollback'):
                self.resize_fault = 'shape rollback could not meet the moving gate'
            self.pending = None
            self.intent = self.intent_since = None
            self.cooldown_until = time_s+self.posture_options.cooldown_s

    def _review_trial(self, angles, time_s):
        if not self.trial or not self.fresh_feedback:
            return
        trial = self.trial
        good, geometry, _ = self.observer.geometry_observer.geometry(np.radians(angles), self.window)
        trial['samples'] += 1
        age = time_s-trial['start_time_s']
        if age >= self.posture_options.trial_grace_s:
            trial['good'] &= good
            trial['bad_samples'] = 0 if good else trial['bad_samples']+1
        if trial['bad_samples'] >= 2 or self.fault:
            old = trial['old']
            kind = 'slide' if old[1]-old[0] == self.window[1]-self.window[0] else 'resize'
            self.pending = dict(kind=kind, operation='rollback', to=old, time_s=time_s,
                                rollback=True, evidence=dict(actual_geometry=geometry, trial=copy.deepcopy(trial)))
            self.events.append(dict(kind='rollback_request', time_s=time_s, window=list(self.window), trial=copy.deepcopy(trial)))
            self.trial = None
        elif age >= self.posture_options.trial_review_s:
            self.events.append(dict(kind='shape_trial_complete', time_s=time_s, trial=copy.deepcopy(trial),
                                    observes_support=False, full_cycle_claim=False))
            self.trial = None

    def advance(self, angles_deg, feedback_time_s, control_time_s, commanded_phase_rad=None, **kwargs):
        rolling_back = self.pending is not None and self.pending.get('rollback', False)
        if self.resize_fault or (self.fault and not rolling_back):
            kwargs['freeze_targets'] = True
        result = super().advance(angles_deg, feedback_time_s, control_time_s, commanded_phase_rad, **kwargs)
        if self.fresh_feedback:
            self._observe_count(np.asarray(angles_deg), control_time_s)
        self._review_trial(np.asarray(angles_deg), control_time_s)
        if (self.automatic and self.fresh_feedback and self.pending is None and self.trial is None and
                control_time_s >= self.cooldown_until and not self.fault and not self.resize_fault and
                not kwargs.get('freeze_targets', False)):
            self._select(np.asarray(angles_deg), control_time_s)
        if self.phase_hold:
            raise RuntimeError('Normal posture-window exchanges must keep rolling')
        return result

    def snapshot(self):
        result = super().snapshot()
        result.update(active_count=int(self.enabled.sum()), selection=copy.deepcopy(self.selection),
                      length_observation=copy.deepcopy(self.selection), trial=copy.deepcopy(self.trial),
                      resize_fault=self.resize_fault, posture_options=asdict(self.posture_options),
                      observer_configuration=self.observer.configuration(),
                      automatic=self.automatic,
                      count_observation=copy.deepcopy(self.count_observation),
                      stable_width_whitelist=None, preferred_count=None,
                      controller_revision='phase_normalized_count_v3')
        return result

MOTOR_PHASE_REFERENCE = {'values': [[0.15833807751140405, -0.06707766383423497, 15.790626070439448],
            [0.24920276274168796, 0.02536057701555894, 12.395211690197813],
            [0.3734541008434089, 0.16845054934214154, 7.62153022032187],
            [0.46378255925176504, 0.3253755415483619, 3.199743136734469],
            [0.5073863314267077, 0.4513296148095325, 0.9978033754624366],
            [0.5223977266067124, 0.5125031733178709, 0.8439353914176058],
            [0.39561665739814167, 0.3747303690784296, -1.0122521644161462],
            [0.37505650923980927, 0.2605016598342882, 2.3391707944094295],
            [0.2993804563979765, 0.09126427422450512, 6.112216197237271],
            [0.14734398772633384, -0.0703727243346286, 9.45566373288018],
            [0.0013173140603726097, -0.16453627632445222, 11.938442674644946],
            [-0.06634367647607498, -0.2061481609866309, 11.69737589944446],
            [-0.03201483675658423, -0.17334666333776294, 10.126628677912105],
            [0.094737682550603, -0.06629309598288913, 7.279149714181012],
            [0.24696620944067682, 0.0957196623174319, 3.6334985715118986],
            [0.3704103527859024, 0.27905585980702163, 0.5598734460687046],
            [0.4310940894138815, 0.3936009166417138, -1.4012551228543793],
            [0.4575229058899257, 0.4695048988132493, -0.3977261241876268],
            [0.4680110247611182, 0.4439960793038127, 2.098362837401867],
            [0.460983373691235, 0.33745983417685643, 6.004498340048969],
            [0.4195904217878894, 0.1743310317474167, 10.402662608307672],
            [0.31802774120819066, 0.013682720649617508, 14.091099107139492],
            [0.21647529307361868, -0.07238521748343807, 16.925851990920634],
            [0.14392091508250182, -0.0988697887929968, 17.241840646977693]],
 'metadata': {'source': 'validation/reports/angle_observer/whole_body_observer_trace.jsonl.gz',
              'stage': 'shared no_force prefix; first full commanded cycle only',
              'samples': 524,
              'start_time_s': 2.1999999999998687,
              'end_time_s': 12.659999999998423,
              'phase_span_rad': 6.275999999999921,
              'start_phase_rad': 0.0,
              'bins': 24,
              'bin_counts': [21,
                             22,
                             22,
                             22,
                             22,
                             21,
                             23,
                             22,
                             21,
                             22,
                             22,
                             22,
                             22,
                             21,
                             22,
                             22,
                             22,
                             22,
                             22,
                             21,
                             22,
                             22,
                             22,
                             22],
              'first_joint': 8,
              'calibration_window': [8, 13],
              'link_lengths_m': [0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645],
              'adaptation': 'frozen offline; no online learning',
              'fields': ['angle-only crest relative to window center',
                         'angle bending center relative to window center',
                         'tail minus head passive-boundary absolute turns'],
              'unused_inputs': ['contact',
                                'root pose',
                                'gravity truth',
                                'motor torque',
                                'side-pull labels']}}

MOTOR_PHASE_REFERENCE8 = {'values': [[-0.6570665260128861, -0.5265259979957095, -1.3146355254500173],
            [-0.6028652118794033, -0.4631916158970686, -1.3526659035445499],
            [-0.49921537171850794, -0.3406073039009714, -2.1491573806117783],
            [-0.4186763809658025, -0.20296725017434092, -4.6355452357763625],
            [-0.39157326688499516, -0.04010396200467703, -8.450767539914994],
            [-0.41416288823033387, 0.05076215598159539, -11.693360524739553],
            [-0.43698910304997707, 0.06327629957845549, -13.278122318298301],
            [-0.46404746245798734, 0.023273998431257326, -13.727317063303389],
            [-0.5047603114825412, -0.0660861634572365, -12.969561031143986],
            [-0.5741496298787379, -0.18958092255968317, -10.715228067873472],
            [-0.6924622596224399, -0.3609726670296567, -7.0138700336612825],
            [-0.7374358237686716, -0.47028177498347734, -3.5719283120227763],
            [-0.7092689899273159, -0.4813457162493986, -1.5686128401228594],
            [-0.6268196662237662, -0.408838463467041, -0.2140931251314856],
            [-0.5274004460417085, -0.2970778418678135, -0.21680187819838714],
            [-0.43514876602542785, -0.14816722207897115, -3.5986461159757313],
            [-0.42671510821487146, -0.008928153233595637, -7.95426406007654],
            [-0.44749373619845745, 0.04954940841145472, -11.052050022685007],
            [-0.4733382201678813, 0.057012452947115655, -13.08963584279277],
            [-0.5053410770785813, 0.02117847437030651, -13.71947502752245],
            [-0.5556567108796804, -0.06726650754962016, -13.322533530589425],
            [-0.6653805089619276, -0.22632471749949268, -10.713674478597168],
            [-0.8090300706248659, -0.3992724243285739, -7.242202381307166],
            [-0.8735685407734213, -0.4929743462334386, -4.710881953080088]],
 'metadata': {'width': 8,
              'source': 'calibration_20261013_081414_692753',
              'control_feedback_period_s': 0.2,
              'phase_interval_rad': [0.0, 6.283185307179586],
              'fields': ['crest', 'same-winding bending', 'same-winding tail minus head boundary'],
              'link_lengths_m': [0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645,
                                 0.0645],
              'adaptation': 'frozen first cycle; next cycle held out'}}

# END MOTOR_SLIDING_CORE

# ==================== 4. 飞特串口协议（只有主循环操作串口） ====================
def instruction_packet(servo_id, instruction, parameters):
    body = bytes([servo_id, len(parameters) + 2, instruction, *parameters])
    return b'\xff\xff' + body + bytes([(~sum(body)) & 255])


def sync_packet(address, width, values):
    parameters = bytearray([address, width])
    for servo_id, value in values:
        if len(value) != width:
            raise ValueError('同步写入寄存器长度错误')
        parameters.extend([servo_id, *value])
    if len(parameters) + 2 > 255:
        raise ValueError('同步写报文过长')
    return instruction_packet(0xFE, 0x83, parameters)


class FeetechBus:
    def __init__(self, serial_port):
        self.port = serial_port

    def write(self, packet):
        if self.port.write(packet) != len(packet):
            raise OSError('串口写入不完整')

    def read_register(self, servo_id, address, width):
        """逐 ID 读取；检查 ID、长度、协议校验和舵机错误位，拒绝伪造反馈。"""
        request = instruction_packet(servo_id, 0x02, [address, width])
        self.port.reset_input_buffer()
        self.write(request)
        deadline = time.monotonic() + SERIAL_TIMEOUT_S
        buffer = bytearray()
        while time.monotonic() < deadline:
            buffer.extend(self.port.read(1))
            while len(buffer) >= 4:
                if buffer[:2] != b'\xff\xff':
                    del buffer[0]
                    continue
                length = buffer[3]
                if not 2 <= length <= 64:
                    del buffer[0]
                    continue
                total = length + 4
                if len(buffer) < total:
                    break
                packet = bytes(buffer[:total])
                del buffer[:total]
                if packet == request or packet[2] != servo_id:
                    continue  # 适配器回显或其他 ID 的旧应答
                if sum(packet[2:]) & 255 != 255:
                    raise RuntimeError(f'舵机 {servo_id} 应答校验和错误')
                if packet[4] != 0:
                    raise RuntimeError(f'舵机 {servo_id} 返回错误位 0x{packet[4]:02x}')
                if length != width + 2:
                    raise RuntimeError(f'舵机 {servo_id} 应答长度错误')
                return packet[5:-1]
        raise TimeoutError(f'舵机 {servo_id} 地址 0x{address:02x} 读取超时')

    def read_angles(self, indices):
        return {i: raw_to_angle(i, int.from_bytes(self.read_register(SERVO_IDS[i], 0x38, 2), 'little'))
                for i in indices}

    def read_states(self):
        """读取 56..70 原始反馈：位置、负载、电压、温度、状态、运动、原始电流。

        负载 bit10、电流 bit15 是符号位。没有标定系数，不转换成力矩或力。
        位置反馈统一转回控制器坐标，再送给与仿真相同的顺应性核心。
        """
        states = {}
        for i, sid in enumerate(SERVO_IDS):
            raw = self.read_register(sid, 0x38, 15)
            load = int.from_bytes(raw[4:6], 'little')
            current = int.from_bytes(raw[13:15], 'little')
            states[i] = dict(angle_deg=raw_to_angle(i,int.from_bytes(raw[:2],'little')),
                             load_raw=-(load & 1023) if load & 1024 else load,
                             current_raw=-(current & 32767) if current & 32768 else current,
                             voltage_v=raw[6]/10,temperature_c=raw[7],
                             status_raw=raw[9],moving_raw=raw[10])
        return states

    def send_targets(self, angles, enabled):
        values = []
        for i, active in enumerate(enabled):
            if active:
                raw = angle_to_raw(i, angles[i])
                block = raw.to_bytes(2, 'little') + b'\x00\x00' + SERVO_SPEED_RAW.to_bytes(2, 'little')
                values.append((SERVO_IDS[i], block))
        if values:
            self.write(sync_packet(0x2A, 6, values))

    def set_enabled(self, enabled, verify=True):
        # 向每个 ID 明确写入 0/1；失能不是位置目标设成 0。
        self.write(sync_packet(0x28, 1, [(sid, bytes([int(e)])) for sid, e in zip(SERVO_IDS, enabled)]))
        if verify:
            actual = [self.read_register(sid, 0x28, 1)[0] for sid in SERVO_IDS]
            if actual != [int(e) for e in enabled]:
                raise RuntimeError(f'使能掩码读回不一致: {actual}')

    def prepare_enabled(self, angles, enabled, configure_mask=None):
        # 窗口迁移只预置新接管的关节，正在支撑的关节保留原目标。
        configure = enabled if configure_mask is None else configure_mask
        active_ids = [sid for sid, e in zip(SERVO_IDS, configure) if e]
        for sid in active_ids:
            if self.read_register(sid, 0x21, 1)[0] != 0:
                raise RuntimeError(f'舵机 {sid} 不是单圈位置模式；脚本不会修改 EEPROM 模式')
        # 先把当前位置预置为目标，再开扭矩，避免恢复上一次运行遗留的目标。
        self.send_targets(angles, configure)
        if active_ids:
            self.write(sync_packet(0x30, 2, [(sid, TORQUE_LIMIT_RAW.to_bytes(2, 'little')) for sid in active_ids]))
            self.write(sync_packet(0x29, 1, [(sid, bytes([ACCELERATION_RAW])) for sid in active_ids]))
        self.set_enabled(enabled)


# ==================== 5. 实验状态与热键功能 ====================
class RollingArcController:
    def __init__(self, bus):
        self.bus = bus
        self.amplitude, self.omega = AMPLITUDE_DEG, OMEGA_RAD_S
        self.first, self.last = FIRST_JOINT, LAST_JOINT
        self.length_normalized = self.last-self.first+1 != 6
        self.phase = self.command_phase = INITIAL_PHASE_RAD
        self.direction, self.mode = 1, 'released'
        self.enabled, self.targets = [False]*NUM_JOINTS, [0.]*NUM_JOINTS
        self.pending_phase, self.phase_step = 0., PHASE_STEP_RAD
        self.forming, self.settle_remaining = False, 0.
        self.touched_hardware = self.quit = False
        self.last_feedback, self.states = {}, {}
        self.feedback_time_s = self.feedback_wall_s = None
        self.feedback_sequence = 0
        self.compliance, self.options = None, make_section_options()
        self.freeze_targets = False
        self.section_clock_s = 0.
        self.section_checked_sequence = -1
        self.section_block_reason, self.actual_section = None, None
        self.sliding, self.pending_gait = None, None
        self.sliding_fault = False
        self.motor_clock_origin_s = time.monotonic()
        self.feedback_phase_rad = self.command_phase
        self.preview_feedback = False

    def refresh_compliance_feedback(self):
        started = time.monotonic()
        acquisition_phase = self.command_phase  # 扫读期间控制线程不改相位
        states = self.bus.read_states()
        if time.monotonic()-started > MAX_FEEDBACK_AGE_S:
            raise TimeoutError('全关节反馈读取过慢，最早样本已经过期')
        if set(states) != set(range(NUM_JOINTS)) or not all(
                math.isfinite(s['angle_deg']) for s in states.values()):
            raise RuntimeError('全关节位置反馈不完整或不是有限值')
        self.states = states
        self.last_feedback = {i: state['angle_deg'] for i, state in states.items()}
        self.feedback_wall_s = started  # 顺序扫读，按最早样本计龄
        self.feedback_phase_rad = acquisition_phase
        self.feedback_sequence += 1

    def motor_sample_clock(self):
        """控制时间/最早反馈时间共用单调时钟；MuJoCo子类改用仿真时钟。"""
        now = time.monotonic()-self.motor_clock_origin_s
        sample = None if self.feedback_wall_s is None else self.feedback_wall_s-self.motor_clock_origin_s
        return now, sample

    def section_reference(self):
        """扩缩后切回基础/固定模式，继续使用当前长度的半圆参考。"""
        return SectionReference(self.first, self.last, self.amplitude,
                                length_normalized=self.length_normalized)

    def basic_targets(self):
        return np.degrees(self.section_reference().targets(self.phase)).tolist()

    def select_gait(self, kind):
        """保留当前窗口、相位和已发目标；切模式不会重新搭管或重置机器人。"""
        if kind not in ('basic', 'fixed', 'sliding'):
            raise ValueError('模式：basic / fixed / sliding')
        if kind == 'basic':
            self.pending_gait = None
            if self.compliance is not None:
                self.sliding, self.compliance = None, None
                self.sliding_fault = False
                self.section_block_reason = None
                self.freeze_targets = False
                self.forming, self.settle_remaining = True, 0.
            return
        if self.mode == 'released' or self.forming or self.settle_remaining > 0:
            self.pending_gait = kind
            print(f'已选择{kind}，形成与静置完成后尝试接入。', flush=True)
            return
        if kind == 'fixed' and self.sliding is None:
            self.pending_gait = None
            if self.compliance is None:
                self.toggle_compliance()
            return
        if kind == 'sliding' and self.sliding is not None and not self.sliding_fault:
            return
        self.refresh_compliance_feedback()
        measured = self.measured_angles()
        valid, geometry = self.check_actual_section(measured)
        if not valid:
            raise ValueError('实际形态不满足开放圆弧条件，未切换：'+str(geometry))
        if kind == 'fixed':
            reference = self.section_reference()
            core = FixedSectionCompliance(reference, LINK_LENGTHS_M, self.options)
            nominal = np.radians(measured)
            nominal[core.enabled] = reference.targets(self.phase)[core.enabled]
            if not core._check_goal(nominal)[0]:
                raise ValueError('当前名义参考不满足开放约束，未切换')
            core.targets = np.radians(self.targets)
            core.next_geometry_s = self.section_clock_s
            self.sliding, self.compliance = None, core
        else:
            if self.amplitude != 60. or self.omega != .6 or self.direction != 1:
                raise ValueError('冻结角度基准只标定了amplitude=60 omega=0.6正向；先在基础模式恢复')
            if not np.allclose(LINK_LENGTHS_M, MOTOR_PHASE_REFERENCE['metadata']['link_lengths_m'], rtol=0., atol=1e-9):
                raise ValueError('实际连杆尺寸与相位基准不一致，需要重新标定')
            now, _ = self.motor_sample_clock()
            p = HalfArcParameters(joints=NUM_JOINTS, first_joint=self.first, last_joint=self.last,
                              amplitude_deg=self.amplitude, omega_rad_s=self.omega,
                              initial_phase_rad=self.phase, settle_s=0.)
            slider = PostureWindow(p, LINK_LENGTHS_M,
                {6: PhaseShapeReference.from_dict(MOTOR_PHASE_REFERENCE),
                 8: PhaseShapeReference.from_dict(MOTOR_PHASE_REFERENCE8)},
                section_options=self.options,
                options=MotorSlidingOptions(cooldown_s=MOTOR_SLIDING_COOLDOWN_S,
                    switch_angle_deg=MOTOR_SLIDING_SWITCH_ANGLE_DEG,
                    maximum_feedback_age_s=MAX_FEEDBACK_AGE_S),
                posture_options=PostureWindowOptions(
                    count_hysteresis_joints=POSTURE_COUNT_HYSTERESIS_JOINTS,
                    persistence_s=POSTURE_PERSISTENCE_S, cooldown_s=MOTOR_SLIDING_COOLDOWN_S,
                    contract_exit_angle_deg=POSTURE_CONTRACT_EXIT_ANGLE_DEG,
                    expansion_turn_error_deg=POSTURE_EXPANSION_TURN_ERROR_DEG,
                    contraction_turn_error_deg=POSTURE_CONTRACTION_TURN_ERROR_DEG),
                feedback_period_s=COMPLIANCE_FEEDBACK_PERIOD_S)
            slider.automatic = MOTOR_SLIDING_AUTO
            slider.targets = np.radians(self.targets)
            slider.core.targets = slider.targets.copy()
            slider.core.next_geometry_s = now
            slider.cooldown_until = now
            slider.events.append(dict(kind='acquire', time_s=now, to=list(slider.window),
                                      reason='take over current window; angle-only posture and length selection'))
            self.sliding, self.compliance = slider, slider.core
            self.length_normalized = True
        self.pending_gait = None
        self.sliding_fault = False
        self.actual_section = geometry
        self.section_checked_sequence = -1
        self.section_block_reason = None
        self.freeze_targets = False

    def tick_motor_sliding(self, dt):
        """20ms目标更新；只有新的200ms全关节角反馈可以完成交接。"""
        slider = self.sliding
        control_time, feedback_time = self.motor_sample_clock()
        if feedback_time is None:
            raise TimeoutError('自动滑窗尚未取得全关节读角')
        if control_time-feedback_time > MAX_FEEDBACK_AGE_S:
            raise TimeoutError('自动滑窗全关节反馈已过期，结束运行并尝试失能')
        measured = np.asarray(self.measured_angles(), float)
        if self.sliding_fault:
            self.bus.send_targets(self.targets, self.enabled)
            return
        self.section_clock_s += dt
        previous_phase, previous_pending = self.phase, self.pending_phase
        previous_core, previous_window = slider.core, slider.window
        previous_targets = self.targets.copy()
        if not slider.phase_hold and not self.freeze_targets:
            if self.mode == 'run':
                self.phase += self.direction*self.omega*dt
            elif self.mode == 'step':
                increment = max(-self.omega*dt, min(self.omega*dt, self.pending_phase))
                self.phase += increment
                self.pending_phase -= increment
        try:
            if slider.last_time is None or feedback_time > slider.last_time:
                valid, self.actual_section = self.check_actual_section(measured)
                if not valid:
                    raise ValueError('新实测形态超出开放圆弧/平面范围 '+str(self.actual_section))
            target, enabled = slider.advance(measured, feedback_time, control_time,
                commanded_phase_rad=self.phase, phase_at_feedback=self.feedback_phase_rad,
                freeze_targets=self.freeze_targets)
            expected_mask = (np.arange(1, NUM_JOINTS+1) >= slider.window[0]) & (np.arange(1, NUM_JOINTS+1) <= slider.window[1])
            if (not np.array_equal(enabled, expected_mask)
                    or not mechanically_feasible(LINK_LENGTHS_M, slider.window, slider.posture_options.nominal_joint_margin_rad)
                    or slider.fault is not None or slider.resize_fault is not None):
                raise ValueError('滑窗使能或开放目标异常：'+str(slider.fault or slider.resize_fault))
            targets = np.degrees(target).tolist()
            for i in np.flatnonzero(enabled):
                angle_to_raw(i, targets[i])
        except (ValueError, np.linalg.LinAlgError) as exc:
            slider.core, slider.window = previous_core, previous_window
            slider.enabled = np.asarray(self.enabled, bool)
            slider.targets = np.radians(previous_targets)
            slider.pending, slider.phase_hold = None, False
            self.phase, self.pending_phase = previous_phase, previous_pending
            self.sliding_fault = True
            self.block_section(str(exc)+'，重新选择模式后继续')
            self.bus.send_targets(previous_targets, self.enabled)
            return
        if enabled.tolist() != self.enabled:
            entering = enabled & ~np.asarray(self.enabled, bool)
            # 预置新进入舵机的实测角，再同步写0x28开/关并读回确认。
            # 公共关节不重新配置；缩窗只关闭退出ID，窗口外不发送位置目标。
            self.bus.prepare_enabled(measured, enabled, configure_mask=entering)
            self.first, self.last = slider.window
            self.enabled = enabled.tolist()
            print(f'自动窗口更新：J{self.first}–J{self.last}，{sum(self.enabled)}关节使能。', flush=True)
        self.phase, self.compliance = slider.phase, slider.core
        if slider.fresh_feedback:
            _, self.actual_section = self.check_actual_section(measured)
        self.targets = targets
        self.bus.send_targets(targets, self.enabled)
        self.command_phase = self.phase
        if self.mode == 'step' and abs(self.pending_phase) < 1e-10:
            self.mode = 'hold'

    def measured_angles(self):
        return [self.states[i]['angle_deg'] for i in range(NUM_JOINTS)]

    def check_actual_section(self, measured):
        """只评价实测角重建的身体形态，不识别管道支撑或绝对对地朝向。"""
        try:
            g = local_section(np.radians(measured), LINK_LENGTHS_M, self.first, self.last)
            points = g['points'][1:-1]
            length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            result = dict(turn_deg=math.degrees(g['total_turn']),
                          gap_fraction=float(np.linalg.norm(points[-1]-points[0])/length),
                          fit_rms_m=g['fit_rms'])
            valid = (SECTION_MIN_ACTUAL_TURN_DEG <= result['turn_deg'] <= SECTION_MAX_ACTUAL_TURN_DEG
                     and result['gap_fraction'] >= self.options.minimum_gap_fraction
                     and result['fit_rms_m'] <= SECTION_MAX_PLANE_RMS_M)
            return valid, result
        except (ValueError, np.linalg.LinAlgError) as exc:
            return False, dict(error=str(exc))

    def block_section(self, reason):
        # 形态计算不适用时保持最近目标和主动支撑，不能继续盲目滚动。
        if reason != self.section_block_reason:
            print('截面顺应暂停：'+reason+'；保持最近目标，可Ctrl+E返回基础模式。', flush=True)
        self.section_block_reason = reason
        self.mode, self.pending_phase = 'hold', 0.



    def toggle_compliance(self):
        if self.sliding is not None:
            self.select_gait('basic')
            return
        if self.compliance is not None:
            # 不重新使能、不归零、不跳相位；下个基础帧按正常限速恢复名义圆弧。
            self.compliance = None
            self.section_block_reason = None
            self.freeze_targets = False
            self.forming, self.settle_remaining = True, 0.
            return
        if self.mode == 'released':
            raise ValueError('先Ctrl+U形成基础圆弧，再Ctrl+E开启固定截面顺应')
        if self.forming or self.settle_remaining > 0:
            raise ValueError('先等待基础圆弧形成和静置完成，再Ctrl+E')
        reference = self.section_reference()
        if reference.enabled.tolist() != self.enabled:
            raise RuntimeError('当前使能掩码与固定窗口不一致')
        self.refresh_compliance_feedback()
        measured = self.measured_angles()
        valid, geometry = self.check_actual_section(measured)
        if not valid:
            raise ValueError('实测固定窗口不是有效开放圆弧，未开启顺应：'+str(geometry))
        core = FixedSectionCompliance(reference, LINK_LENGTHS_M, self.options)
        nominal = np.radians(measured)
        nominal[core.enabled] = reference.targets(self.phase)[core.enabled]
        if not core._check_goal(nominal)[0]:
            raise ValueError('当前幅值的名义参考不满足开放约束，未开启顺应')
        core.targets = np.radians(self.targets)  # 保留最近发送目标，软接入反馈
        core.next_geometry_s = self.section_clock_s
        self.compliance = core
        self.actual_section = geometry
        self.section_checked_sequence = -1
        self.section_block_reason = None
        self.freeze_targets = False
        # 保留run/hold/step状态及剩余单步；开关不构造新的运动。

    def tick_compliance(self, dt):
        if self.feedback_wall_s is None or time.monotonic()-self.feedback_wall_s > MAX_FEEDBACK_AGE_S:
            raise TimeoutError('截面顺应位置反馈已过期，结束运行并尝试失能')
        if self.compliance.enabled.tolist() != self.enabled:
            raise RuntimeError('固定截面控制器不能改变使能窗口')
        self.section_clock_s += dt
        measured = self.measured_angles()
        if self.section_checked_sequence != self.feedback_sequence:
            valid, self.actual_section = self.check_actual_section(measured)
            self.section_checked_sequence = self.feedback_sequence
            if not valid:
                self.block_section('实测形态超出开放圆弧/平面范围 '+str(self.actual_section))
            else:
                self.section_block_reason = None
        if self.freeze_targets or self.section_block_reason is not None:
            self.bus.send_targets(self.targets, self.enabled)
            return
        previous_phase = self.phase
        previous_targets = self.targets.copy()
        step_finished = False
        if self.mode == 'run':
            self.phase += self.direction*self.omega*dt
        elif self.mode == 'step':
            increment = max(-self.omega*dt, min(self.omega*dt, self.pending_phase))
            self.phase += increment
            self.pending_phase -= increment
            step_finished = abs(self.pending_phase) < 1e-10
        try:
            # 全角反馈目标间隔200 ms；中间保持观测，按几何周期计算、按控制周期过滤。
            target = self.compliance.update(np.radians(measured), self.phase, self.section_clock_s, dt)
            valid_command = self.compliance._check_goal(target)[0]
            if self.compliance.status != 'running' or not valid_command:
                raise ValueError('名义参考或过滤后发送目标不满足开放约束')
            targets = np.degrees(target).tolist()
            for i, active in enumerate(self.enabled):
                if active:
                    angle_to_raw(i, targets[i])
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.phase = previous_phase
            self.compliance.targets = np.radians(previous_targets)
            self.block_section(str(exc))
            self.bus.send_targets(previous_targets, self.enabled)
            return
        self.targets = targets
        self.bus.send_targets(self.targets, self.enabled)  # 包只含当前使能ID
        self.command_phase = self.phase
        if step_finished:
            self.mode = 'hold'

    def arm(self):
        if self.mode != 'released':
            return              # 重复 apply 不重新初始化，不把机器人拉回相位 0
        validate_arc_reference(self.amplitude, self.omega, self.first, self.last, self.length_normalized)
        enabled = [self.first <= j <= self.last for j in range(1, NUM_JOINTS + 1)]
        measured = self.bus.read_angles(range(self.first - 1, self.last))
        targets = [measured.get(i, 0.0) for i in range(NUM_JOINTS)]
        for i, e in enumerate(enabled):
            if e:
                angle_to_raw(i, targets[i])
        self.touched_hardware = True  # 从第一次写开始，异常/退出都尝试关闭扭矩
        self.bus.prepare_enabled(targets, enabled)
        self.targets, self.enabled = targets, enabled
        self.phase = INITIAL_PHASE_RAD
        self.forming, self.settle_remaining = True, SETTLE_S
        self.pending_phase, self.mode = 0.0, 'hold'

    def release(self, verify=True):
        self.sliding, self.pending_gait = None, None
        self.sliding_fault = False
        self.compliance = None
        self.mode, self.pending_phase = 'released', 0.0
        self.enabled = [False] * NUM_JOINTS
        self.touched_hardware = True
        self.bus.set_enabled(self.enabled, verify=verify)

    def change_parameters(self, amplitude=None, omega=None):
        values = []
        if amplitude is not None:
            values.append(f'amplitude={amplitude}')
        if omega is not None:
            values.append(f'omega={omega}')
        self.assign_parameters(' '.join(values))

    def assign_parameters(self, command):
        """先验证整行，再一起更新；固定窗口只在release后允许人工修改。"""
        if isinstance(self.sliding, MotorFeedbackSliding):
            for item in command.split():
                key, value = item.split('=', 1)
                if key in ('amplitude', 'window') or (key == 'omega' and float(value) != .6):
                    raise ValueError('自动滑窗使用60度缩放/0.6正向相位参考；改幅值/频率/窗口先切回基础模式')
        amplitude, omega, delta = self.amplitude, self.omega, self.phase_step
        first, last, options = self.first, self.last, self.options
        option_keys = {'gain': 'gain', 'yielding': 'yielding', 'filter': 'target_filter_s'}
        for assignment in command.split():
            key, value = assignment.split('=', 1)
            if key == 'amplitude':
                amplitude = float(value)
            elif key == 'omega':
                omega = float(value)
            elif key == 'delta':
                delta = float(value)
            elif key == 'window':
                first, last = map(int, value.split(':'))
            elif key in option_keys:
                options = replace(options, **{option_keys[key]: float(value)})
            else:
                raise ValueError('参数：amplitude omega delta window gain yielding filter；本版不用support/load')
        validate_parameters(amplitude, omega, first, last)
        normalized = last-first+1 != 6 if self.mode == 'released' else self.length_normalized
        validate_arc_reference(amplitude, omega, first, last, normalized)
        if not math.isfinite(delta) or not 0 < delta <= math.pi:
            raise ValueError('单步相位增量必须在(0,pi] rad内')
        if (first, last) != (self.first, self.last) and self.mode != 'released':
            raise ValueError('手动修改window前先release')
        if self.compliance is not None:
            reference = replace(self.compliance.p, amplitude_deg=amplitude)
            # 半圆预算随amplitude/60缩放；保留既有保守参考上限。
            if 3*amplitude > options.maximum_section_turn_deg:
                raise ValueError('该幅值使名义半圆参考超过截面开放上限')
        if amplitude != self.amplitude and self.mode != 'released' and self.compliance is None:
            self.forming, self.settle_remaining = True, SETTLE_S
        self.amplitude, self.omega, self.phase_step = amplitude, omega, delta
        self.first, self.last, self.options = first, last, options
        if self.mode == 'released':
            self.length_normalized = last-first+1 != 6
        if self.compliance is not None:
            self.compliance.p = reference
            self.compliance.options = options
            self.compliance.next_geometry_s = self.section_clock_s
            if self.sliding is not None:
                self.sliding.section_options = options

    def command(self, command):
        command = command.strip().lower()
        if isinstance(self.sliding, MotorFeedbackSliding) and (command in ('reverse', 'back') or
                (command == 'step' and self.pending_phase < 0)):
            raise ValueError('角度相位基准未标定反向滚动；先切换基础/固定模式')
        if '=' in command:
            self.assign_parameters(command)
        elif command.startswith('gait_'):
            self.select_gait(command[5:])
        elif command == 'window_auto':
            if self.sliding is None:
                raise ValueError('先Ctrl+3切换到自动滑窗模式')
            self.sliding.automatic = not self.sliding.automatic
        elif command in ('quit', 'exit'):
            self.quit = True
        elif command in ('arm', 'apply'):
            self.arm()
        elif command == 'release':
            self.release()
        elif command == 'status':
            self.refresh_compliance_feedback()
        elif command == 'compliance':
            self.toggle_compliance()
        elif command == 'hold':
            if self.mode == 'released':
                raise ValueError('还没有接管实机；先 apply')
            self.mode, self.pending_phase = 'hold', 0.0
            self.freeze_targets = True
            self.forming, self.settle_remaining = False, 0.0
            self.bus.send_targets(self.targets, self.enabled)
        elif command in ('delta-', 'delta+'):
            self.phase_step = max(.01, self.phase_step / 2) if command == 'delta-' else min(.6, self.phase_step * 2)
        elif command in ('amplitude_up', 'amplitude_down'):
            sign = 1 if command.endswith('up') else -1
            self.change_parameters(amplitude=self.amplitude + sign * AMPLITUDE_KEY_STEP_DEG)
        elif command in ('omega_up', 'omega_down'):
            sign = 1 if command.endswith('up') else -1
            self.change_parameters(omega=self.omega + sign * OMEGA_KEY_STEP_RAD_S)
        elif command == 'reverse':
            self.direction *= -1
        elif command in ('step', 'back', 'toggle', 'pause', 'run'):
            if self.mode == 'released':
                raise ValueError('先 Ctrl+U 或 apply 使能并形成圆弧')
            if command in ('step', 'back'):
                if self.omega == 0:
                    raise ValueError('单步执行需要 omega > 0')
                self.mode = 'step'
                self.pending_phase += self.phase_step * (1 if command == 'step' else -1)
            else:
                running = command == 'run' or (command == 'toggle' and self.mode != 'run')
                self.mode, self.pending_phase = ('run' if running else 'hold'), 0.0
                if not running:
                    self.forming, self.settle_remaining = False, 0.0
            self.freeze_targets = False
            if self.mode != 'hold' and not self.forming and self.compliance is None:
                goal = self.basic_targets()
                if any(abs(goal[i] - self.targets[i]) > 1e-6 for i, e in enumerate(self.enabled) if e):
                    self.forming, self.settle_remaining = True, SETTLE_S
        else:
            raise ValueError(f'未知命令: {command}')

    def tick(self, dt):
        if self.mode == 'released':
            return
        step_finished = False
        # 不追赶阻塞期间漏发的帧，防止串口延迟后突然跳过一大段相位。
        dt = min(dt, CONTROL_PERIOD_S)
        if self.sliding is not None:
            self.tick_motor_sliding(dt)
            return
        if self.compliance is not None:
            self.tick_compliance(dt)
            return
        if not self.forming and self.settle_remaining <= 0:
            if self.mode == 'run':
                self.phase += self.direction * self.omega * dt
            elif self.mode == 'step':
                delta = max(-self.omega * dt, min(self.omega * dt, self.pending_phase))
                self.phase += delta
                self.pending_phase -= delta
                if abs(self.pending_phase) < 1e-10:
                    step_finished = True
        goal = self.basic_targets()
        if len(goal) != NUM_JOINTS or any(not math.isfinite(angle) for angle in goal):
            raise ValueError('步态公式必须返回 20 个有限角度值（单位：度）')
        for i, active in enumerate(self.enabled):
            if active:
                angle_to_raw(i, goal[i])  # 修改公式后也先检查最终目标，不能靠截断掩盖越界
        if self.mode == 'hold' and not self.forming:
            goal = self.targets.copy()  # hold 立即冻结已发目标，未到位的形成过程也可以中断
        rate = FORM_RATE_DEG_S if self.forming else MAX_TARGET_RATE_DEG_S
        for i, active in enumerate(self.enabled):
            if active:
                delta = max(-rate * dt, min(rate * dt, goal[i] - self.targets[i]))
                self.targets[i] += delta
        self.bus.send_targets(self.targets, self.enabled)
        self.command_phase = self.phase
        if step_finished:
            self.mode = 'hold'  # 最后一小步的目标发送后再保持，避免停在倒数一帧
        if self.forming:
            self.forming = any(abs(goal[i] - self.targets[i]) > 1e-8 for i, e in enumerate(self.enabled) if e)
        else:
            self.settle_remaining = max(0.0, self.settle_remaining - dt)
        if self.pending_gait and not self.forming and self.settle_remaining <= 0:
            kind, self.pending_gait = self.pending_gait, None
            try:
                self.select_gait(kind)
            except ValueError as exc:
                print(f'模式接入未执行：{exc}；可再次按模式热键重试。', flush=True)

    def snapshot(self):
        section = None
        if self.compliance is not None:
            core = self.compliance
            section = (core.snapshot() if core.geometry is not None and core.target_geometry is not None
                       else dict(status=core.status, window=[core.first, core.last]))
            section.update(block_reason=self.section_block_reason, actual_body_geometry=self.actual_section,
                           options=asdict(self.options), input='measured_joint_angles', window_fixed=self.sliding is None)
        age = None if self.feedback_wall_s is None else time.monotonic()-self.feedback_wall_s
        gait = ('sliding_section_compliance' if self.sliding is not None else
                'basic_arc' if self.compliance is None else 'fixed_section_compliance')
        return dict(mode=self.mode, gait=gait,
                    targets_frozen=self.freeze_targets,
                    forming=self.forming, phase_rad=self.phase, command_phase_rad=self.command_phase,
                    amplitude_deg=self.amplitude, omega_rad_s=self.omega, direction=self.direction,
                    phase_step_rad=self.phase_step, pending_phase_rad=self.pending_phase,
                    settle_remaining_s=self.settle_remaining, window=[self.first, self.last],
                    enabled=self.enabled.copy(), target_controller_deg=self.targets.copy(),
                    active_count=sum(self.enabled), length_normalized=self.length_normalized,
                    enable_state_known=self.touched_hardware,
                    feedback_time_s=self.feedback_time_s, feedback_age_s=age,
                    feedback_sequence=self.feedback_sequence, compliance=section,
                    pending_gait=self.pending_gait,
                    sliding=None if self.sliding is None else copy.deepcopy(self.sliding.snapshot()),
                    sliding_events=[] if self.sliding is None else copy.deepcopy(self.sliding.events),
                    servo_feedback_raw=self.states,
                    measured_controller_deg={f'J{i+1}': v for i, v in self.last_feedback.items()})

    def show_status(self, include_feedback=False):
        label = {'released': '未接管/已释放', 'hold': '保持', 'run': '连续', 'step': '单步'}[self.mode]
        if self.forming:
            label += '，正在形成基础圆弧'
        elif self.settle_remaining > 0:
            label += f'，静置余{self.settle_remaining:.1f}s'
        gait = ('角度反馈滑窗顺应' if self.sliding is not None else
                '基础圆弧' if self.compliance is None else '固定截面顺应')
        print(f'{gait} | {label} | J{self.first}–J{self.last} | A={self.amplitude:g}° '
              f'omega={self.omega:g}rad/s | phase={self.phase:.4f}rad | delta={self.phase_step:g}rad', flush=True)
        if self.compliance is not None:
            window_state = '窗口固定' if self.sliding is None else f'自动选窗={self.sliding.automatic} 等相位={self.sliding.phase_hold}'
            print(f'截面K={self.options.gain:g} yielding={self.options.yielding:g} | '
                  f'{window_state} | 阻塞原因={self.section_block_reason or "无"}', flush=True)
        if self.sliding is not None and self.sliding.observation is not None:
            o = self.sliding.observation
            print(f'角度观察器={o["state"]} 建议={o["suggested_window"]} '
                  f'不可观测原因={o.get("reasons", [])}', flush=True)
        if isinstance(self.sliding, PostureWindow):
            s = self.sliding.selection or {}
            print(f'使能数量={sum(self.enabled)}，数量估计={s.get("filtered_count_joints")} '
                  f'决策={s.get("state", "收集相位")}，候选={s.get("chosen")}，'
                  f'原因={s.get("length_reason", "等待反馈")}', flush=True)
        if include_feedback:
            print('实测角：'+' '.join(f'J{i+1}={v:.1f}°' for i, v in self.last_feedback.items()), flush=True)
            print('原始负载/电流（只记录）：'+' '.join(
                f'J{i+1}={s["load_raw"]}/{s["current_raw"]}' for i, s in self.states.items()), flush=True)


# ==================== 6. 实测全身构型预览（单独进程，无控制输出） ====================
# BEGIN GRAVITY_PREVIEW_CORE
def gravity_unit(v):
    return np.asarray(v, float)/max(float(np.linalg.norm(v)), 1e-12)


def gravity_angle(a, b):
    return math.degrees(math.acos(float(np.clip(np.dot(gravity_unit(a), gravity_unit(b)), -1., 1.))))


def gravity_link_frames(q):
    frames = [np.eye(3)]
    for i, angle in enumerate(q):
        a = angle if i % 2 == 0 else -angle
        c, s = math.cos(a), math.sin(a)
        rotation = (np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) if i % 2 == 0 else
                    np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]]))
        frames.append(frames[-1]@rotation)
    return frames


@dataclass(frozen=True)
class GravityPreviewOptions:
    maximum_end_links: int = 4
    minimum_end_links: int = 3
    minimum_straightness: float = .98
    maximum_disagreement_deg: float = 25.
    filter_time_s: float = .4
    correction_rate_deg_s: float = 30.
    maximum_feedback_gap_s: float = .8


class DualEndGravityPreview:
    def __init__(self, lengths, options=GravityPreviewOptions()):
        self.lengths = np.asarray(lengths, float)
        self.options = options
        self.down = self.basis = self.last_time = None
        self.previous_frames = self.previous_candidates = None
        self.result = dict(state='waiting', reliable=False, down_head=None, basis=None)

    def update(self, angles_deg, window, feedback_time_s):
        q = np.radians(angles_deg)
        if q.shape != (len(self.lengths)-1,) or not np.isfinite(q).all() or not math.isfinite(feedback_time_s):
            raise ValueError('Finite full-body angles and sample time required')
        if self.last_time is not None and feedback_time_s <= self.last_time:
            return self.result  # Repainting an old sample must not advance the estimator.
        dt = 0. if self.last_time is None else feedback_time_s-self.last_time
        gap = dt > self.options.maximum_feedback_gap_s
        points, frames = forward(q, self.lengths), gravity_link_frames(q)
        arc = points[window[0]:window[1]+2]
        opening = (arc[0]+arc[-1])*.5-arc.mean(axis=0)
        opening_valid = np.linalg.norm(opening) > 1e-6
        opening = gravity_unit(opening)
        candidates = {}
        for name, available, anchor in (('head', window[0]-1, 0), ('tail', len(q)-window[1], len(q))):
            count = min(self.options.maximum_end_links, available)
            if count < self.options.minimum_end_links:
                continue
            samples = points[:count+1][::-1] if name == 'head' else points[-count-1:]
            chord = samples[-1]-samples[0]
            straight = float(np.linalg.norm(chord)/np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())
            direction = gravity_unit(chord)
            alignment = float(direction@opening)
            eligible = opening_valid and straight >= self.options.minimum_straightness and alignment >= .5
            weight = count/self.options.maximum_end_links * float(np.clip((straight-.97)/.03, 0., 1.)) if eligible else 0.
            # Curvature/bending change in that end's own frame discounts an unstable end.
            instability = 0.
            old = (self.previous_candidates or {}).get(name)
            if old and self.previous_frames is not None and not gap:
                transported = frames[anchor]@self.previous_frames[anchor].T@np.array(old['direction'])
                instability = gravity_angle(transported, direction)
                weight /= 1.+(instability/8.)**2
            candidates[name] = dict(direction=direction.tolist(), count=count, anchor=anchor,
                straightness=straight, alignment=alignment, weight=weight, instability_deg=instability)
        usable = [c for c in candidates.values() if c['weight'] > 0.]
        disagreement = None if len(usable) != 2 else gravity_angle(usable[0]['direction'], usable[1]['direction'])
        consistent = disagreement is None or disagreement <= self.options.maximum_disagreement_deg
        # Both ends predict the previous estimate in today's head frame. No root pose is used.
        predictions, weights = [], []
        if self.down is not None and self.previous_frames is not None and not gap:
            for name, c in candidates.items():
                old = (self.previous_candidates or {}).get(name)
                if old and c['weight'] > 0. and old['weight'] > 0.:
                    predictions.append(frames[c['anchor']]@self.previous_frames[c['anchor']].T@self.down)
                    weights.append(min(c['weight'], old['weight']))
        prediction = self.down
        if predictions and consistent:
            prediction = gravity_unit(np.average(predictions, axis=0, weights=weights))
        raw = gravity_unit(sum(c['weight']*np.array(c['direction']) for c in usable)) if usable and consistent else None
        state = 'conflict_hold' if not consistent else 'unobservable_hold'
        reliable = raw is not None
        if raw is not None:
            state = 'dual' if len(usable) == 2 else 'single_degraded'
            if self.down is None:
                self.down = raw
            else:
                # After a gap, keep the visible history but explicitly reacquire slowly.
                reference = prediction if prediction is not None else self.down
                alpha = 1.-math.exp(-min(dt, .2)/self.options.filter_time_s)
                angle = gravity_angle(reference, raw)
                alpha = min(alpha, self.options.correction_rate_deg_s*min(dt, .2)/max(angle, 1e-9))
                self.down = gravity_unit((1.-alpha)*reference+alpha*raw)
                if gap:
                    state, reliable = 'reacquiring', False
        if self.down is not None:
            up = -self.down
            # Horizontal heading is an arbitrary display convention, NOT magnetic/world yaw.
            x = np.array([0., 1., 0.]) if self.basis is None else self.basis[0].copy()
            x -= (x@up)*up
            if np.linalg.norm(x) < .1:
                x = np.eye(3)[int(np.argmin(abs(up)))]; x -= (x@up)*up
            x = gravity_unit(x)
            self.basis = np.array([x, gravity_unit(np.cross(up, x)), up])
        self.last_time, self.previous_frames, self.previous_candidates = feedback_time_s, frames, candidates
        self.result = dict(state=state, reliable=reliable, down_head=None if self.down is None else self.down.tolist(),
            raw_down_head=None if raw is None else raw.tolist(), basis=None if self.basis is None else self.basis.tolist(),
            candidates=candidates, disagreement_deg=disagreement, feedback_time_s=feedback_time_s,
            observes_world_pose=False, conditional_transport=True)
        return self.result
# END GRAVITY_PREVIEW_CORE

def preview_payload(controller):
    """仅拷贝显示所需数据；目标线在失能段沿用实测角，不画虚假的零角尾部。"""
    complete = set(controller.states) == set(range(NUM_JOINTS))
    angles = controller.measured_angles() if complete else None
    enabled = list(controller.enabled)
    target = None if angles is None else [controller.targets[i] if enabled[i] else angles[i]
                                         for i in range(NUM_JOINTS)]
    slider = controller.sliding
    selection = getattr(slider, 'selection', None) or {}
    observation = getattr(slider, 'observation', None) or {}
    return dict(angles_deg=angles, target_deg=target, enabled=enabled,
        gravity_time_s=controller.motor_sample_clock()[1],
        enable_known=controller.touched_hardware, feedback_wall_s=controller.feedback_wall_s,
        window=[controller.first, controller.last], phase_rad=controller.phase,
        mode=controller.mode, gait='自动位置/数量' if slider else '固定顺应' if controller.compliance else '基础圆弧',
        automatic=None if slider is None else slider.automatic,
        selection=copy.deepcopy(selection), pending=copy.deepcopy(getattr(slider, 'pending', None)),
        observer_state=observation.get('state', '等待反馈'),
        reasons=observation.get('reasons', []), block_reason=controller.section_block_reason,
        lengths=list(LINK_LENGTHS_M))


class JointPreviewFigure:
    """实测/目标骨架与使能；可选双侧重力对齐，始终没有世界定位输入。"""
    def __init__(self):
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        self.plt = plt
        self.figure = plt.figure(figsize=(13, 7), facecolor='#f4f6fa')
        self.figure.canvas.manager.set_window_title('Rolling Arc — 实测构型与使能预览')
        grid = self.figure.add_gridspec(3, 2, width_ratios=[1.3, 1], height_ratios=[1, 1, .45])
        self.body = self.figure.add_subplot(grid[:, 0], projection='3d')
        self.angles = self.figure.add_subplot(grid[:2, 1])
        self.mask = self.figure.add_subplot(grid[2, 1])
        self.header = self.figure.suptitle('等待实测角度；预览不会使能或驱动舵机', fontsize=12)
        self.note = self.figure.text(.04, .04, '', fontsize=10, va='bottom')
        self.figure.subplots_adjust(left=.05, right=.97, top=.88, bottom=.20, wspace=.24, hspace=.65)
        self.body.view_init(elev=22, azim=-65)
        from matplotlib.widgets import Button
        self.gravity = DualEndGravityPreview(LINK_LENGTHS_M)
        self.gravity_align = PREVIEW_GRAVITY_ALIGN
        self.latest_row = None
        self.view_button = Button(self.figure.add_axes([.79, .93, .16, .035]), 'Gravity / Head')
        self.view_button.on_clicked(self.toggle_view)

    def toggle_view(self, _event):
        self.gravity_align = not self.gravity_align
        if self.latest_row is not None:
            self.update(self.latest_row)

    def update(self, row):
        self.latest_row = row
        elev, azim = self.body.elev, self.body.azim
        self.body.clear(); self.angles.clear(); self.mask.clear()
        self.body.view_init(elev=elev, azim=azim)
        known = row['enable_known']
        colours = ['#147bb5' if e else '#9259b5' for e in row['enabled']] if known else ['#a3a9b2']*NUM_JOINTS
        ids = np.arange(1, NUM_JOINTS+1)
        self.mask.bar(ids, np.ones(NUM_JOINTS), color=colours, width=.85)
        self.mask.set(xticks=ids, yticks=[], xlim=(.3, NUM_JOINTS+.7), xlabel='关节编号：小 → 头端；大 → 尾端')
        self.mask.set_title('蓝：使能 / 紫：失能' if known else '灰：尚未接管，使能状态未知', fontsize=10)
        window = row['window']
        age = None if row['feedback_wall_s'] is None else max(0., time.monotonic()-row['feedback_wall_s'])
        stale = age is None or age > MAX_FEEDBACK_AGE_S
        age_text = '无反馈' if age is None else f'反馈年龄 {age:.2f}s'+('（陈旧）' if stale else '')
        self.header.set_text(f"{row['gait']} | {row['mode']} | J{window[0]}–J{window[1]} | "
                             f"使能 {sum(row['enabled']) if known else '?'} | 相位 {row['phase_rad']:.2f} | {age_text}")
        self.header.set_color('#b34322' if stale else '#17354b')
        selection, pending = row['selection'], row['pending']
        candidate = pending['to'] if pending else selection.get('chosen')
        decision = selection.get('state', row['observer_state'])
        count = selection.get('filtered_count_joints')
        count_text = '收集相位中' if count is None else f'{count:.2f}'
        direction = ''
        if candidate:
            shift = sum(candidate)-sum(window)
            direction = '向头端' if shift < 0 else '向尾端' if shift > 0 else '中心不变'
            direction += f' → J{candidate[0]}–J{candidate[1]}'
        gravity = self.gravity.result
        sample_time = row.get('gravity_time_s', row['feedback_wall_s'])
        if row['angles_deg'] is not None and sample_time is not None and not stale:
            if self.gravity.last_time is not None and sample_time < self.gravity.last_time:
                self.gravity = DualEndGravityPreview(row['lengths'])  # 仿真明确重置为新episode。
            gravity = self.gravity.update(row['angles_deg'], window, sample_time)
        aligned = self.gravity_align and gravity['basis'] is not None
        states = {'dual': '双侧融合', 'single_degraded': '单侧补充（降级）', 'conflict_hold': '双侧冲突，保留方向',
                  'unobservable_hold': '无可信悬垂段，保留方向', 'reacquiring': '重新获取', 'waiting': '等待悬垂参考'}
        view_label = '重力对齐' if aligned else '头部坐标'
        gravity_label = '反馈陈旧' if stale else states[gravity['state']]
        self.note.set_text(f'{view_label} | {gravity_label}；仅估计上下，不测世界位置/水平朝向。实线实测，虚线目标。\n'
            f"自动={row['automatic']} | 决策={decision} {direction} | 数量估计={count_text}\n"
            f"状态：{row['block_reason'] or ', '.join(row['reasons']) or '正常'}；关闭此窗口不停止控制，Ctrl+X失能。")
        if row['angles_deg'] is not None:
            points = forward(np.radians(row['angles_deg']), row['lengths'])
            target = forward(np.radians(row['target_deg']), row['lengths'])
            if aligned:
                basis = np.array(gravity['basis'])
                points, target = points@basis.T, target@basis.T
            self.body.plot(*target.T, '--', color='#a3a9b2', lw=1.5, label='目标')
            for i in range(NUM_JOINTS+1):
                colour = colours[max(0, i-1)]
                self.body.plot(*points[i:i+2].T, color=colour, lw=3, alpha=.45 if stale else 1.)
            self.body.scatter(*points[1:-1].T, c=colours, s=24)
            for j, point in enumerate(points[1:-1], 1):
                self.body.text(*point, str(j), fontsize=8)
            self.body.text(*points[0], ' HEAD', color='#17354b')
            self.body.text(*points[-1], ' TAIL', color='#17354b')
            if candidate:
                a = points[window[0]:window[1]+2].mean(axis=0)
                b = points[candidate[0]:candidate[1]+2].mean(axis=0)
                self.body.quiver(*a, *(b-a), color='#e38b27', linewidth=2, arrow_length_ratio=.3)
            both = np.r_[points, target]
            center = (both.min(axis=0)+both.max(axis=0))/2
            radius = max(.1, float(np.ptp(both, axis=0).max())*.55)
            self.body.set(xlim=(center[0]-radius, center[0]+radius),
                          ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius))
            self.angles.plot(ids, row['angles_deg'], 'o-', color='#17354b', label='实测')
            active = np.asarray(row['enabled'], bool)
            targets = np.where(active, row['target_deg'], np.nan)
            self.angles.plot(ids, targets, 's--', color='#e38b27', label='使能关节目标')
            self.angles.legend(loc='upper right')
        else:
            self.body.text2D(.15, .5, '等待全20关节实测角，不用目标伪装实测', transform=self.body.transAxes)
        frame_label = 'display' if aligned else 'head'
        self.body.set(xlabel=f'X {frame_label} / m', ylabel=f'Y {frame_label} / m',
                      zlabel='Z up (estimated) / m' if aligned else 'Z head / m')
        self.body.set_box_aspect((1, 1, 1))
        self.angles.set(ylabel='控制器坐标角度 / °', xticks=ids, ylim=(-100, 100))
        self.angles.grid(alpha=.2)
        self.figure.canvas.draw_idle()


def _preview_worker(inbox, stopped, ready):
    """Windows spawn子进程，只读队列，不导入串口、不写设备。"""
    try:
        figure = JointPreviewFigure()
        latest = [None]
        def refresh():
            if stopped.is_set():
                figure.plt.close(figure.figure)
                return
            while True:
                try:
                    latest[0] = inbox.get_nowait()
                except queue.Empty:
                    break
            if latest[0] is not None:
                figure.update(latest[0])
                if not ready.is_set():
                    figure.figure.canvas.draw()
                    ready.set()
        timer = figure.figure.canvas.new_timer(interval=int(PREVIEW_PERIOD_S*1000))
        timer.add_callback(refresh)
        timer.start()
        figure.plt.show()
    finally:
        stopped.set()


class JointPreviewWindow:
    def __init__(self):
        import multiprocessing as mp
        context = mp.get_context('spawn')
        self.inbox, self.stopped, self.ready = context.Queue(maxsize=1), context.Event(), context.Event()
        self.process = context.Process(target=_preview_worker, args=(self.inbox, self.stopped, self.ready), daemon=True)
        self.process.start()
        self.next_update = 0.

    @property
    def active(self):
        return self.process.is_alive() and not self.stopped.is_set()

    def publish(self, controller):
        now = time.monotonic()
        if not self.active or now < self.next_update:
            return
        self.next_update = now+PREVIEW_PERIOD_S
        try:
            self.inbox.put_nowait(preview_payload(controller))
        except queue.Full:
            pass  # 显示慢时丢弃新显示帧；绝不阻塞控制线程。

    def close(self):
        self.stopped.set()
        self.process.join(timeout=1.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.)
        self.inbox.cancel_join_thread()
        self.inbox.close()


# ==================== 7. 离线串口，仅验证协议与交互，不代表物理运动 ====================
class DryRunSerial:
    def __init__(self):
        self.registers = {sid: bytearray(80) for sid in SERVO_IDS}
        for sid, zero in zip(SERVO_IDS, ZERO_RAW):
            self.registers[sid][0x38:0x3A] = zero.to_bytes(2, 'little')
        self.received = bytearray()

    def reset_input_buffer(self):
        self.received.clear()

    def write(self, packet):
        if sum(packet[2:]) & 255 != 255 or packet[3] != len(packet) - 4:
            raise RuntimeError('离线串口收到非法报文')
        if packet[4] == 0x02:
            sid, address, width = packet[2], packet[5], packet[6]
            payload = self.registers[sid][address:address + width]
            self.received.extend(instruction_packet(sid, 0, payload))
        elif packet[4] == 0x83:
            address, width = packet[5], packet[6]
            for offset in range(7, len(packet) - 1, width + 1):
                sid = packet[offset]
                self.registers[sid][address:address + width] = packet[offset + 1:offset + 1 + width]
                if address == 0x2A:
                    self.registers[sid][0x38:0x3A] = packet[offset + 1:offset + 3]
        return len(packet)

    def read(self, size):
        value = bytes(self.received[:size])
        del self.received[:size]
        return value

    def close(self):
        pass


# ==================== 7. Ctrl 热键与控制台，参照 Adaptive_Pole_Climbing_Real ====================
def decode_hotkey(vk, control_down):
    """Windows Ctrl+字母产生控制字符，因此使用虚拟键码，不用 key.char。"""
    if not control_down:
        return None
    special = {32: 'pause', 38: 'amplitude_up', 40: 'amplitude_down',
               39: 'omega_up', 37: 'omega_down', 189: 'delta-', 187: 'delta+',
               49: 'gait_basic', 50: 'gait_fixed', 51: 'gait_sliding',
               97: 'gait_basic', 98: 'gait_fixed', 99: 'gait_sliding'}
    return special.get(vk) or HOTKEY_COMMANDS.get(chr(vk) if vk is not None and 65 <= vk <= 90 else '')


def global_key_listener(commands, keyboard, command_decoder=decode_hotkey, release_decoder=None):
    held = set()
    control_keys = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}

    def on_press(key):
        if key in held:
            return              # 按住不放不累计自动重复单步
        held.add(key)
        vk = getattr(key, 'vk', None)
        if vk is None:
            vk = getattr(getattr(key, 'value', None), 'vk', None)
        command = command_decoder(vk, bool(held & control_keys))
        if command:
            commands.put(command)

    def on_release(key):
        held.discard(key)
        if release_decoder is not None:
            vk = getattr(key, 'vk', None)
            if vk is None:
                vk = getattr(getattr(key, 'value', None), 'vk', None)
            command = release_decoder(vk, key in control_keys)
            if command:
                commands.put(command)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    return listener


def console_listener(commands, stop):
    """控制台也只排队。Windows 终端轮询，退出时不会遗留阻塞 input() 线程。"""
    if sys.platform == 'win32' and sys.stdin.isatty():
        import msvcrt
        line = ''
        while not stop.wait(.02):
            if not msvcrt.kbhit():
                continue
            char = msvcrt.getwch()
            if char in ('\x00', '\xe0'):
                msvcrt.getwch()  # 忽略方向键的第二字节；热键由 pynput 入队
            elif char == '\x03':
                commands.put('quit')
                return
            elif char in ('\r', '\n'):
                print('', flush=True)
                if line.strip():
                    commands.put(line)
                if line.strip().lower() in ('quit', 'exit'):
                    return
                line = ''
            elif char == '\b':
                if line:
                    line = line[:-1]
                    print('\b \b', end='', flush=True)
            elif char.isprintable():
                line += char
                print(char, end='', flush=True)
    else:
        for line in sys.stdin:  # 重定向输入；离线检查可在run_controller中关闭控制台
            if stop.is_set():
                return
            commands.put(line)
            if line.strip().lower() in ('quit', 'exit'):
                return


# ==================== 8. 运行入口与退出清理 ====================
def find_robot_port(available_ports):
    """参考CaseA的CH340识别；多个适配器时由顶部PORT明确选择。"""
    candidates = [p for p in available_ports if 'CH340' in (p.description or '').upper()]
    if not candidates:
        raise RuntimeError("未找到CH340串口，请检查连接；其他适配器可在文件顶部设置PORT。")
    if len(candidates) > 1:
        names = ', '.join(p.device for p in candidates)
        raise RuntimeError(f"找到多个CH340串口：{names}；请在文件顶部设置PORT选择机器人。")
    return candidates[0].device


def open_robot_serial():
    """真实连接只在运行入口调用；导入文件和内存串口检查不会打开设备。"""
    import serial
    import serial.tools.list_ports
    name = PORT if PORT else find_robot_port(serial.tools.list_ports.comports())
    return serial.Serial(name, BAUD_RATE, timeout=0.01, write_timeout=SERIAL_TIMEOUT_S)


def open_run_log(folder, prefix='rolling_arc'):
    """每次启动新建日志；即使在同一时间戳启动，也不覆盖既有记录。"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    number = 0
    while True:
        suffix = '' if number == 0 else f'_{number}'
        path = folder / f'{prefix}_{stamp}{suffix}.jsonl'
        try:
            return path, path.open('x', encoding='utf-8')
        except FileExistsError:
            number += 1


def process_control_commands(controller, commands, elapsed):
    """实机与MuJoCo手调共用同一命令分派和状态提示。"""
    for _ in range(commands.qsize()):
        command = commands.get_nowait()
        try:
            if command.strip().lower() == 'help':
                print(getattr(controller, 'help_text', HELP), flush=True)
                continue
            controller.command(command)
            if command.strip().lower() == 'status':
                controller.feedback_time_s = elapsed
            controller.show_status(include_feedback=command.strip().lower() == 'status')
        except ValueError as exc:
            print(f'参数/操作未执行: {exc}', flush=True)


def poll_control_feedback(controller, elapsed, next_feedback):
    """顺应或构型预览都读全关节；预览读角也可在未接管/失能时进行。"""
    full_feedback = controller.compliance is not None or controller.preview_feedback
    compliance_due = (full_feedback and
                      (controller.feedback_time_s is None or
                       elapsed-controller.feedback_time_s >= COMPLIANCE_FEEDBACK_PERIOD_S))
    if (elapsed >= next_feedback or compliance_due) and (controller.mode != 'released' or controller.preview_feedback):
        if full_feedback:
            controller.refresh_compliance_feedback()
            period = COMPLIANCE_FEEDBACK_PERIOD_S
        else:
            active = [i for i, e in enumerate(controller.enabled) if e]
            controller.last_feedback = controller.bus.read_angles(active)
            period = FEEDBACK_PERIOD_S
        controller.feedback_time_s = elapsed
        return elapsed + period
    return next_feedback


def run_controller(*, dry_run=False, hotkeys=True, console=True, auto=False,
                   auto_compliance=False, duration_s=None, fast=False,
                   log_directory=None, preview=False):
    """运行控制循环；关键字参数供内存串口检查，日常直接运行文件即可。"""
    if fast and not dry_run:
        raise ValueError('加速循环只能用于内存串口检查。')
    if auto_compliance and not auto:
        raise ValueError('离线自动接入顺应需先自动形成基础圆弧。')
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError('运行时长必须为有限正数。')
    if BAUD_RATE <= 0 or (not hotkeys and not console and duration_s is None):
        raise ValueError('波特率必须为正；关闭全部交互输入的离线检查需设置时长。')
    validate_setup()
    # 打开设备前检查热键依赖，避免连接成功后才因缺包退出。
    keyboard_module = None
    if hotkeys:
        from pynput import keyboard as keyboard_module
    if dry_run:
        port = DryRunSerial()
    else:
        port = open_robot_serial()
    controller = RollingArcController(FeetechBus(port))
    commands = queue.Queue()
    listener, console_thread, log_file, preview_window = None, None, None, None
    stop_console = threading.Event()
    try:
        if preview:
            preview_window = JointPreviewWindow()
            controller.preview_feedback = True
            preview_window.publish(controller)
        if log_directory is not None:
            log_path, log_file = open_run_log(log_directory)
            print(f'实验日志：{log_path}', flush=True)
        if keyboard_module:
            listener = global_key_listener(commands, keyboard_module)
        if console:
            console_thread = threading.Thread(target=console_listener, args=(commands, stop_console), daemon=True)
            console_thread.start()
        print('离线串口（不是动力学仿真）' if dry_run else f'实机串口 {port.port} / {BAUD_RATE}', flush=True)
        print(HELP, flush=True)
        print('启动不改变扭矩；预览开启时只读实际角，尚未接管的舵机原使能状态未知。', flush=True)
        if auto:
            commands.put('arm')
            commands.put('run')
        elapsed, next_feedback, next_log = 0.0, 0.0, 0.0
        auto_compliance_pending = auto and auto_compliance
        while not controller.quit and (duration_s is None or elapsed < duration_s):
            cycle_start = time.monotonic()
            # 热键线程只排队；状态改变、读反馈、写目标都在这个线程依次执行。
            process_control_commands(controller, commands, elapsed)
            if controller.quit:
                break
            if preview_window is not None:
                controller.preview_feedback = preview_window.active
            before = (controller.mode, controller.forming, controller.settle_remaining > 0)
            next_feedback = poll_control_feedback(controller, elapsed, next_feedback)
            if auto_compliance_pending and not controller.forming and controller.settle_remaining<=0:
                controller.command('compliance');controller.command('run')
                controller.show_status()
                auto_compliance_pending = False
            controller.tick(CONTROL_PERIOD_S)
            if preview_window is not None:
                preview_window.publish(controller)
            after = (controller.mode, controller.forming, controller.settle_remaining > 0)
            if before != after:
                controller.show_status()
            if log_file and elapsed >= next_log:
                record = dict(time_s=elapsed, execution='dry_run' if dry_run else 'serial', **controller.snapshot())
                log_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                log_file.flush()
                next_log = elapsed + 0.1
            if fast:
                elapsed += CONTROL_PERIOD_S
            else:
                time.sleep(max(0.0, CONTROL_PERIOD_S - (time.monotonic() - cycle_start)))
                elapsed += time.monotonic() - cycle_start
    finally:
        stop_console.set()
        if controller.touched_hardware:
            try:
                controller.release()  # 正常退出、Ctrl+C、通信异常都走这条路径
                print('全部舵机失能已读回确认。', flush=True)
            except Exception as exc:
                print(f'退出时无法确认失能: {exc}；断连情况下软件不能保证关闭舵机。', flush=True)
        try:
            port.close()
        finally:
            if listener:
                listener.stop()
                listener.join(timeout=1.)
            if console_thread:
                console_thread.join(timeout=.1)
            if log_file:
                log_file.close()
            if preview_window is not None:
                preview_window.close()


def main():
    """直接运行本文件：使用顶部配置连接，等待Ctrl热键，不自动使能或滚动。"""
    run_controller(dry_run=DRY_RUN, log_directory=LOG_DIRECTORY if SAVE_LOG else None, preview=SHOW_PREVIEW)


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            raise ValueError('请直接运行本文件，不附加启动参数；设置在文件顶部修改，离线用DRY_RUN=True。')
        main()
    except KeyboardInterrupt:
        print('用户中止。')
    except Exception as exc:
        print(f'实机控制停止: {exc}', flush=True)
        raise SystemExit(1)
