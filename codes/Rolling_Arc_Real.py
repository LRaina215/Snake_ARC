"""圆弧滚动实机控制：只运行本文件，通过热键直接控制飞特串口舵机。

直接运行本文件，自动查找CH340串口；所有实验配置在文件顶部修改。
DRY_RUN=True时使用内存串口，不连接机器人；日志自动保存在本文件旁的logs目录。
Ctrl+U 形成圆弧，Ctrl+J/H 正/反单步，Ctrl+G 连续/暂停，Ctrl+X 全部失能。
Ctrl+E 开关固定窗口的实际截面顺应；J8–J13保持使能，其余关节失能。
不依赖 MuJoCo、EXE、相邻仓库或本仓库的其他 Python 模块。
"""
import json
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
    '-': 'delta-', '=': 'delta+',
}
HELP = '''圆弧滚动调试（Ctrl+字母全局热键；任何窗口均会响应）：
  Ctrl+U 读取当前位置，使能局部并缓慢形成圆弧；默认完成后保持
  Ctrl+J / Ctrl+H 正向 / 反向单步；Ctrl+-/= 相位步长减半/加倍
  Ctrl+G 连续滚动/暂停；Ctrl+空格 暂停；Ctrl+N 保持并重发当前目标
  Ctrl+T 反转连续滚动方向；Ctrl+K 读取所有关节位置
  Ctrl+E 固定窗口截面顺应开/关：先形成圆弧；保留相位与连续/暂停状态
  Ctrl+上/下 幅值±2°；Ctrl+右/左 角频率±0.05 rad/s
  Ctrl+X 全部失能（机器人会失去主动支撑）；Ctrl+Q 退出，接管后会失能
控制台输入后回车：apply / step / back / run / pause / hold / reverse
                  compliance / release / status / help / quit
参数：amplitude=60 omega=0.6 delta=0.1 window=8:13 gain=0.02 yielding=0.7 filter=0.15
window只在release后修改；本版不自动移窗、不使用support/load判据。
顺应性下pause冻结相位、继续形状反馈；hold冻结全部目标。
'''


# ==================== 2. 圆弧公式与角度换算 ====================
def joint_angles_deg(phase_rad, amplitude_deg, first_joint, last_joint):
    """这里是调步态时主要修改的公式；返回控制器坐标中的角度，单位为度。"""
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

    @property
    def enabled(self):
        indices = np.arange(1, self.joints+1)
        return (indices >= self.first_joint) & (indices <= self.last_joint)

    def targets(self, phase):
        return np.radians(joint_angles_deg(phase, self.amplitude_deg, self.first_joint, self.last_joint))


def make_section_options():
    return SectionOptions(gain=COMPLIANCE_GAIN, yielding=SECTION_YIELDING,
        geometry_period_s=SECTION_GEOMETRY_PERIOD_S, target_rate_deg_s=MAX_TARGET_RATE_DEG_S,
        target_filter_s=SECTION_TARGET_FILTER_S, maximum_section_turn_deg=SECTION_MAX_TURN_DEG,
        minimum_gap_fraction=SECTION_MIN_GAP_FRACTION)


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

    def refresh_compliance_feedback(self):
        started = time.monotonic()
        states = self.bus.read_states()
        if time.monotonic()-started > MAX_FEEDBACK_AGE_S:
            raise TimeoutError('全关节反馈读取过慢，最早样本已经过期')
        if set(states) != set(range(NUM_JOINTS)) or not all(
                math.isfinite(s['angle_deg']) for s in states.values()):
            raise RuntimeError('全关节位置反馈不完整或不是有限值')
        self.states = states
        self.last_feedback = {i: state['angle_deg'] for i, state in states.items()}
        self.feedback_wall_s = started  # 顺序扫读，按最早样本计龄
        self.feedback_sequence += 1

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
        if self.last-self.first+1 != 6:
            raise ValueError('当前截面实机版本只验证连续6关节；释放后设window=8:13再形成')
        reference = SectionReference(self.first, self.last, self.amplitude)
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
        self.bus.send_targets(self.targets, self.enabled)  # 包只含固定6个ID
        self.command_phase = self.phase
        if step_finished:
            self.mode = 'hold'

    def arm(self):
        if self.mode != 'released':
            return              # 重复 apply 不重新初始化，不把机器人拉回相位 0
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
        if not math.isfinite(delta) or not 0 < delta <= math.pi:
            raise ValueError('单步相位增量必须在(0,pi] rad内')
        if (first, last) != (self.first, self.last) and self.mode != 'released':
            raise ValueError('修改固定window前先release；顺应性不会自动移动窗口')
        if self.compliance is not None:
            reference = replace(self.compliance.p, amplitude_deg=amplitude)
            # 六关节均匀名义圆弧的峰值为3*幅值；参考必须留在目标开放范围。
            if 3*amplitude > options.maximum_section_turn_deg:
                raise ValueError('该幅值使名义六关节参考超过截面开放上限')
        if amplitude != self.amplitude and self.mode != 'released' and self.compliance is None:
            self.forming, self.settle_remaining = True, SETTLE_S
        self.amplitude, self.omega, self.phase_step = amplitude, omega, delta
        self.first, self.last, self.options = first, last, options
        if self.compliance is not None:
            self.compliance.p = reference
            self.compliance.options = options
            self.compliance.next_geometry_s = self.section_clock_s

    def command(self, command):
        command = command.strip().lower()
        if '=' in command:
            self.assign_parameters(command)
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
                goal = joint_angles_deg(self.phase, self.amplitude, self.first, self.last)
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
        goal = joint_angles_deg(self.phase, self.amplitude, self.first, self.last)
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

    def snapshot(self):
        section = None
        if self.compliance is not None:
            core = self.compliance
            section = (core.snapshot() if core.geometry is not None and core.target_geometry is not None
                       else dict(status=core.status, window=[core.first, core.last]))
            section.update(block_reason=self.section_block_reason, actual_body_geometry=self.actual_section,
                           options=asdict(self.options), input='measured_joint_angles', window_fixed=True)
        age = None if self.feedback_wall_s is None else time.monotonic()-self.feedback_wall_s
        return dict(mode=self.mode, gait='basic_arc' if self.compliance is None else 'fixed_section_compliance',
                    targets_frozen=self.freeze_targets,
                    forming=self.forming, phase_rad=self.phase, command_phase_rad=self.command_phase,
                    amplitude_deg=self.amplitude, omega_rad_s=self.omega, direction=self.direction,
                    phase_step_rad=self.phase_step, pending_phase_rad=self.pending_phase,
                    settle_remaining_s=self.settle_remaining, window=[self.first, self.last],
                    enabled=self.enabled.copy(), target_controller_deg=self.targets.copy(),
                    feedback_time_s=self.feedback_time_s, feedback_age_s=age,
                    feedback_sequence=self.feedback_sequence, compliance=section,
                    servo_feedback_raw=self.states,
                    measured_controller_deg={f'J{i+1}': v for i, v in self.last_feedback.items()})

    def show_status(self, include_feedback=False):
        label = {'released': '未接管/已释放', 'hold': '保持', 'run': '连续', 'step': '单步'}[self.mode]
        if self.forming:
            label += '，正在形成基础圆弧'
        elif self.settle_remaining > 0:
            label += f'，静置余{self.settle_remaining:.1f}s'
        gait = '基础圆弧' if self.compliance is None else '固定截面顺应'
        print(f'{gait} | {label} | J{self.first}–J{self.last} | A={self.amplitude:g}° '
              f'omega={self.omega:g}rad/s | phase={self.phase:.4f}rad | delta={self.phase_step:g}rad', flush=True)
        if self.compliance is not None:
            print(f'截面K={self.options.gain:g} yielding={self.options.yielding:g} | '
                  f'窗口固定 | 阻塞原因={self.section_block_reason or "无"}', flush=True)
        if include_feedback:
            print('实测角：'+' '.join(f'J{i+1}={v:.1f}°' for i, v in self.last_feedback.items()), flush=True)
            print('原始负载/电流（只记录）：'+' '.join(
                f'J{i+1}={s["load_raw"]}/{s["current_raw"]}' for i, s in self.states.items()), flush=True)


# ==================== 6. 离线串口，仅验证协议与交互，不代表物理运动 ====================
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
               39: 'omega_up', 37: 'omega_down', 189: 'delta-', 187: 'delta+'}
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
    """共用实机反馈节奏；顺应读全关节，基础模式读主动关节。"""
    compliance_due = (controller.compliance is not None and
                      (controller.feedback_time_s is None or
                       elapsed-controller.feedback_time_s >= COMPLIANCE_FEEDBACK_PERIOD_S))
    if (elapsed >= next_feedback or compliance_due) and controller.mode != 'released':
        if controller.compliance is not None:
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
                   log_directory=None):
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
    listener, console_thread, log_file = None, None, None
    stop_console = threading.Event()
    try:
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
        print('启动仅打开串口，不改变扭矩；尚未接管的舵机原使能状态未知。', flush=True)
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
            before = (controller.mode, controller.forming, controller.settle_remaining > 0)
            next_feedback = poll_control_feedback(controller, elapsed, next_feedback)
            if auto_compliance_pending and not controller.forming and controller.settle_remaining<=0:
                controller.command('compliance');controller.command('run')
                controller.show_status()
                auto_compliance_pending = False
            controller.tick(CONTROL_PERIOD_S)
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


def main():
    """直接运行本文件：使用顶部配置连接，等待Ctrl热键，不自动使能或滚动。"""
    run_controller(dry_run=DRY_RUN, log_directory=LOG_DIRECTORY if SAVE_LOG else None)


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
