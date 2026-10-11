"""圆弧滚动实机控制：只运行本文件，通过热键直接控制飞特串口舵机。

默认连接真实串口；--dry-run 使用内存串口，不连接机器人。
Ctrl+U 形成圆弧，Ctrl+J/H 正/反单步，Ctrl+G 连续/暂停，Ctrl+X 全部失能。
Ctrl+E 接入/退出局部滑动窗口顺应性；支撑可用原始负载估计或人工指定。
不依赖 MuJoCo、EXE、相邻仓库或本仓库的其他 Python 模块。
"""
import argparse
import json
import math
import statistics
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import queue
import sys
import threading
import time


# ==================== 1. 经常修改的实验参数 ====================
PORT = 'COM9'                  # 原设备档案；实际端口变化时改这里或用 --port
BAUD_RATE = 115200
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

# Step2：这些是调试时最常修改的顺应性参数。全部角度仍为控制器角（度）。
COMPLIANCE_GAIN = 0.02          # 实测曲率包络 * (1+gain)，不是舵机 PD 增益
MIN_COMPLIANCE_AMPLITUDE_DEG = 25.0
CURVATURE_FILTER_S = 0.5
MIN_WINDOW_JOINTS = 6
MAX_WINDOW_JOINTS = 12
WINDOW_GUARD_JOINTS = 2
WINDOW_DWELL_S = 0.6           # 新支撑段持续足够久才转移窗口
WINDOW_TRANSFER_S = 0.8        # 新旧使能交叠，先预置实测角，再渐入目标
COMPLIANCE_FEEDBACK_PERIOD_S = 0.20  # 读取 20 舵机，不假装能在 115200 下 50 Hz 全量反馈
MAX_FEEDBACK_AGE_S = 0.8
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
HELP = '''圆弧滚动实机调试（Ctrl+字母全局热键；任何窗口均会响应）：
  Ctrl+U 读取当前位置，使能局部并缓慢形成圆弧；默认完成后保持
  Ctrl+J / Ctrl+H 正向 / 反向单步；Ctrl+-/= 相位步长减半/加倍
  Ctrl+G 连续滚动/暂停；Ctrl+空格 暂停；Ctrl+N 保持并重发当前目标
  Ctrl+T 反转连续滚动方向；Ctrl+K 读取所有关节位置
  Ctrl+E 顺应性开/关：从实测姿态与当前相位接管，默认保持相位
  Ctrl+上/下 幅值±2°；Ctrl+右/左 角频率±0.05 rad/s
  Ctrl+X 全部失能（机器人会失去主动支撑）；Ctrl+Q 退出，接管后会失能
控制台输入后回车：apply / step / back / run / pause / hold / reverse
                  compliance / release / status / help / quit
参数：amplitude=60 omega=0.6 delta=0.1 window=8:13 gain=0.02 load=40
支撑输入：support=auto 或 support=9:11（实际支撑段，控制窗口会自动加边界）。
window 只在 release 后修改。顺应性下 pause 冻结相位、继续顺应；hold 冻结全部目标。
实机 auto 是原始负载+姿态估计，不是接触力传感器；无可信支撑观测时停止相位推进。
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
                FEEDBACK_PERIOD_S, SERIAL_TIMEOUT_S,COMPLIANCE_FEEDBACK_PERIOD_S,MAX_FEEDBACK_AGE_S,
                SUPPORT_CURVATURE_THRESHOLD_DEG)
    if any(not math.isfinite(v) or v <= 0 for v in positive) or not math.isfinite(SETTLE_S) or SETTLE_S < 0:
        raise ValueError('控制周期、速度和反馈周期必须为正，静置时间非负')
    if not math.isfinite(INITIAL_PHASE_RAD) or not 1 <= SERVO_SPEED_RAW <= 32766:
        raise ValueError('初相位或舵机速度错误')
    if not 0 <= TORQUE_LIMIT_RAW <= 1000 or not 0 <= ACCELERATION_RAW <= 254:
        raise ValueError('力矩限制或加速度寄存器值错误')
    if type(SUPPORT_LOAD_THRESHOLD_RAW) is not int or not 1<=SUPPORT_LOAD_THRESHOLD_RAW<=1023:
        raise ValueError('原始负载阈值必须为 1..1023 的整数')
    if MAX_FEEDBACK_AGE_S<=COMPLIANCE_FEEDBACK_PERIOD_S:
        raise ValueError('反馈过期时间必须大于全量反馈周期')
    if DEFAULT_MANUAL_SUPPORT is not None and (len(DEFAULT_MANUAL_SUPPORT)!=2 or
            not all(type(v) is int for v in DEFAULT_MANUAL_SUPPORT) or
            not 1<=DEFAULT_MANUAL_SUPPORT[0]<=DEFAULT_MANUAL_SUPPORT[1]<=NUM_JOINTS):
        raise ValueError('默认人工支撑必须为空或有效首:尾关节')
    make_compliance_options(AMPLITUDE_DEG)  # 打开串口前检查顶部顺应性参数
    validate_parameters(AMPLITUDE_DEG, OMEGA_RAD_S, FIRST_JOINT, LAST_JOINT)


# ==================== 3. 局部顺应性共享核心（仿真也直接导入此实现） ====================
# 核心流程：实测角 -> 奇偶关节曲率包络 -> 平滑幅值 -> 正交滚动目标。
# 支撑观测 -> 延时锁存 -> 保持/迁移局部窗口 -> 新旧关节目标渐变。
# 这里不读取串口、不使用管道半径，不改变机器人根部位置。
# 在仿真里 support 来自真实碰撞反力；在实机里由下方适配层提供估计或人工输入。
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
        self.amplitude = AMPLITUDE_DEG
        self.omega = OMEGA_RAD_S
        self.first, self.last = FIRST_JOINT, LAST_JOINT
        self.phase = INITIAL_PHASE_RAD
        self.command_phase = self.phase
        self.direction = 1      # 用相位推进方向反转，不在当前姿态突然翻转偶数角
        self.mode = 'released'
        self.enabled = [False] * NUM_JOINTS
        self.targets = [0.0] * NUM_JOINTS
        self.pending_phase = 0.0
        self.phase_step = PHASE_STEP_RAD
        self.forming = False
        self.settle_remaining = 0.0
        self.touched_hardware = False
        self.quit = False
        self.last_feedback = {}
        self.feedback_time_s = None
        self.compliance = None
        self.options = make_compliance_options(self.amplitude)
        self.manual_support = DEFAULT_MANUAL_SUPPORT
        self.load_threshold = SUPPORT_LOAD_THRESHOLD_RAW
        self.states = {}
        self.feedback_wall_s = None
        self.freeze_targets = False

    def refresh_compliance_feedback(self):
        self.states = self.bus.read_states()
        self.last_feedback = {i:state['angle_deg'] for i,state in self.states.items()}
        self.feedback_wall_s = time.monotonic()

    def support_observations(self):
        """适配层：人工支撑段或估计量。核心不把负载当真实接触力。"""
        if self.manual_support is not None:
            first,last = self.manual_support
            return [first<=j<=last for j in range(1,NUM_JOINTS+1)]
        scores = [False]*NUM_JOINTS
        for i,state in self.states.items():
            neighbor = i+1 if i+1<NUM_JOINTS else i-1
            envelope = math.hypot(state['angle_deg'],self.states[neighbor]['angle_deg'])
            scores[i] = abs(state['load_raw'])>=self.load_threshold and envelope>=SUPPORT_CURVATURE_THRESHOLD_DEG
        return scores

    def toggle_compliance(self):
        if self.compliance is not None:
            # 返回基础圆弧时保留当前局部使能区域；下次 run 再限速形成统一幅值。
            active = [i for i,e in enumerate(self.enabled) if e]
            self.first,self.last = active[0]+1,active[-1]+1
            self.compliance = None
            self.mode,self.pending_phase = 'hold',0.
            self.forming,self.settle_remaining = False,0.
            return
        self.refresh_compliance_feedback()
        observed = self.support_observations()
        seed = support_seed([float(e) for e in observed],self.options.minimum_width,
                            self.options.maximum_width,self.options.guard_joints)
        if seed is None:
            raise ValueError('没有可信支撑观测；检查负载阈值，或按实际搭管处输入 support=首:尾 后再 Ctrl+E')
        measured = [self.states[i]['angle_deg'] for i in range(NUM_JOINTS)]
        if self.mode == 'released':
            # 任意已有搭管姿态接管：保留当前角度，从奇偶实测角拟合相位。
            # 不是自动把未受支撑的机器人抓到管上，支撑须由用户或反馈确认。
            lo,hi = seed
            odd = [measured[i] for i in range(lo-1,hi) if i%2==0]
            even = [measured[i] for i in range(lo-1,hi) if i%2==1]
            self.phase = math.atan2(statistics.mean(even),statistics.mean(odd))
            mask = [lo<=j<=hi for j in range(1,NUM_JOINTS+1)]
            for i,e in enumerate(mask):
                if e:angle_to_raw(i,measured[i])
            self.touched_hardware = True
            self.bus.prepare_enabled(measured,mask)
            self.enabled,self.targets = mask,measured.copy()
        else:
            # 已使能时不重新归零、不重新形成 Step1。
            seed = (self.first,self.last) if all(not e or self.enabled[i] for i,e in enumerate(observed)) else seed
        self.compliance = SlidingArcCompliance(measured,self.phase,seed,self.options,self.enabled)
        # 继续使能的关节保留已发目标，再通过核心限速靠近实测包络。
        self.compliance.targets = self.targets.copy()
        self.forming,self.settle_remaining,self.pending_phase = False,0.,0.
        self.mode,self.freeze_targets = 'hold',False

    def tick_compliance(self, dt):
        if self.feedback_wall_s is None or time.monotonic()-self.feedback_wall_s>MAX_FEEDBACK_AGE_S:
            raise TimeoutError('顺应性全关节反馈已过期，结束运行并尝试失能')
        if self.freeze_targets:
            self.bus.send_targets(self.targets,self.enabled)
            return
        measured = [self.states[i]['angle_deg'] for i in range(NUM_JOINTS)]
        target,mask = self.compliance.update(measured,self.support_observations(),dt,self.phase)
        self.command_phase = self.phase
        if mask!=self.enabled:
            new = [e and not self.enabled[i] for i,e in enumerate(mask)]
            # 更新前立即重读新关节位置。不能用 0.2 s 前的角度开扭矩。
            if any(new):
                fresh = self.bus.read_angles([i for i,e in enumerate(new) if e])
                for i,value in fresh.items():
                    measured[i] = target[i] = value
                    self.compliance.targets[i] = value
            for i,e in enumerate(mask):
                if e:angle_to_raw(i,target[i])
            self.bus.prepare_enabled(measured,mask,configure_mask=new)
            self.enabled = mask
        self.targets = target
        self.first,self.last = self.compliance.window[0]+1,self.compliance.window[1]+1
        self.bus.send_targets(self.targets,self.enabled)
        if not self.compliance.phase_paused:
            if self.mode=='run':self.phase += self.direction*self.omega*dt
            elif self.mode=='step':
                delta = max(-self.omega*dt,min(self.omega*dt,self.pending_phase))
                self.phase += delta;self.pending_phase -= delta
                # Send the final phase on the next tick before entering hold.
                if abs(self.pending_phase)<1e-10:self.mode='hold'

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
        candidate_a = self.amplitude if amplitude is None else amplitude
        candidate_w = self.omega if omega is None else omega
        validate_parameters(candidate_a, candidate_w, self.first, self.last)
        options = replace(self.options,maximum_amplitude_deg=candidate_a,
                          minimum_amplitude_deg=min(MIN_COMPLIANCE_AMPLITUDE_DEG,candidate_a))
        if candidate_a != self.amplitude and self.mode != 'released' and self.compliance is None:
            self.forming, self.settle_remaining = True, SETTLE_S
        self.amplitude, self.omega = candidate_a, candidate_w
        self.options = options
        if self.compliance is not None:self.compliance.options = options

    def assign_parameters(self, command):
        """先验证整行，再一起更新；无效输入不留下部分修改。"""
        amplitude, omega, delta, first, last = self.amplitude, self.omega, self.phase_step, self.first, self.last
        options,manual,threshold = self.options,self.manual_support,self.load_threshold
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
            elif key == 'gain':options = replace(options,gain=float(value))
            elif key == 'load':
                threshold = int(value)
                if not 1<=threshold<=1023:raise ValueError('load 必须在 1..1023 内')
            elif key == 'support':
                manual = None if value=='auto' else tuple(map(int,value.split(':')))
                if manual is not None and (len(manual)!=2 or not 1<=manual[0]<=manual[1]<=NUM_JOINTS):
                    raise ValueError('support 必须为 auto 或 1..20 内的首:尾')
            else:
                raise ValueError('参数：amplitude omega delta window gain load support')
        validate_parameters(amplitude, omega, first, last)
        if not math.isfinite(delta) or not 0 < delta <= math.pi:
            raise ValueError('单步相位增量必须在 (0,pi] rad 内')
        if (first, last) != (self.first, self.last) and self.mode != 'released':
            raise ValueError('修改初始 window 前先 release；顺应性运行时改 support 控制支撑观测')
        options = replace(options,maximum_amplitude_deg=amplitude,
                          minimum_amplitude_deg=min(MIN_COMPLIANCE_AMPLITUDE_DEG,amplitude))
        if amplitude != self.amplitude and self.mode != 'released' and self.compliance is None:
            self.forming, self.settle_remaining = True, SETTLE_S
        self.amplitude, self.omega, self.phase_step = amplitude, omega, delta
        self.first, self.last = first, last
        self.options,self.manual_support,self.load_threshold = options,manual,threshold
        if self.compliance is not None:self.compliance.options = options

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
        return dict(mode=self.mode, forming=self.forming, phase_rad=self.phase,
                    command_phase_rad=self.command_phase,
                    amplitude_deg=self.amplitude, omega_rad_s=self.omega, direction=self.direction,
                    phase_step_rad=self.phase_step, pending_phase_rad=self.pending_phase,
                    settle_remaining_s=self.settle_remaining,
                    window=[self.first, self.last], enabled=self.enabled.copy(),
                    target_controller_deg=self.targets.copy(),
                    feedback_time_s=self.feedback_time_s,
                    compliance=None if self.compliance is None else self.compliance.snapshot(),
                    support_source='load_pose_estimate' if self.manual_support is None else 'manual',
                    manual_support=self.manual_support,load_threshold_raw=self.load_threshold,
                    servo_feedback_raw=self.states,
                    measured_controller_deg={f'J{i + 1}': v for i, v in self.last_feedback.items()})

    def show_status(self, include_feedback=False):
        label = {'released': '未接管/已释放', 'hold': '保持', 'run': '连续', 'step': '单步'}[self.mode]
        if self.forming:
            label += '，正在形成圆弧'
        elif self.settle_remaining > 0:
            label += f'，静置余 {self.settle_remaining:.1f}s'
        print(f'{label} | J{self.first}–J{self.last} | A={self.amplitude:g}° '
              f'omega={self.omega:g}rad/s | phase={self.phase:.4f}rad '
              f'delta={self.phase_step:g}rad | direction={self.direction:+d}', flush=True)
        if self.compliance is not None:
            print(f'顺应性 gain={self.options.gain:g} | 支撑={"负载+姿态估计" if self.manual_support is None else self.manual_support} '
                  f'| 暂停相位={self.compliance.phase_paused} | 窗口交接余 {self.compliance.transfer_remaining_s:.2f}s',flush=True)
        if include_feedback:
            values = ' '.join(f'J{i + 1}={v:.1f}°' for i, v in self.last_feedback.items())
            print(f'实测控制器角：{values}', flush=True)
            loads=' '.join(f'J{i+1}={state["load_raw"]}' for i,state in self.states.items())
            print(f'原始负载（未标定）：{loads}',flush=True)


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


def global_key_listener(commands, keyboard):
    held = set()
    control_keys = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}

    def on_press(key):
        if key in held:
            return              # 按住不放不累计自动重复单步
        held.add(key)
        vk = getattr(key, 'vk', None)
        if vk is None:
            vk = getattr(getattr(key, 'value', None), 'vk', None)
        command = decode_hotkey(vk, bool(held & control_keys))
        if command:
            commands.put(command)

    def on_release(key):
        held.discard(key)

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
        for line in sys.stdin:  # 重定向输入；单纯离线批处理使用 --no-console
            if stop.is_set():
                return
            commands.put(line)
            if line.strip().lower() in ('quit', 'exit'):
                return


# ==================== 8. 运行入口与退出清理 ====================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default=PORT)
    parser.add_argument('--baud', type=int, default=BAUD_RATE)
    parser.add_argument('--dry-run', action='store_true', help='内存串口，不打开真实设备')
    parser.add_argument('--no-hotkeys', action='store_true', help='用于离线批量验证')
    parser.add_argument('--no-console', action='store_true', help='不读取控制台输入')
    parser.add_argument('--auto', action='store_true', help='连接后直接形成圆弧并连续滚动')
    parser.add_argument('--compliance',action='store_true',help='与 --auto 一起使用，形成后接入顺应性')
    parser.add_argument('--support',help='auto 或 人工支撑首:尾，例如 9:11')
    parser.add_argument('--duration', type=float, help='到时自动结束；默认等待 Ctrl+Q/quit')
    parser.add_argument('--fast', action='store_true', help='仅离线模式：取消实际等待')
    parser.add_argument('--log', type=Path, help='JSONL，记录发出的目标和实测反馈')
    args = parser.parse_args()
    if args.fast and not args.dry_run:
        parser.error('--fast 只能用于 --dry-run')
    if args.compliance and not args.auto:
        parser.error('--compliance 自动接入需同时指定 --auto；手动接入用 Ctrl+E')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration 必须为有限正数')
    if args.baud <= 0 or (args.no_hotkeys and args.no_console and args.duration is None):
        parser.error('波特率必须为正；关闭全部交互输入时必须指定 --duration')
    validate_setup()
    # 打开设备前检查热键依赖，避免连接成功后才因缺包退出。
    keyboard_module = None
    if not args.no_hotkeys:
        from pynput import keyboard as keyboard_module
    if args.dry_run:
        port = DryRunSerial()
    else:
        import serial
        port = serial.Serial(args.port, args.baud, timeout=0.01, write_timeout=SERIAL_TIMEOUT_S)
    controller = RollingArcController(FeetechBus(port))
    commands = queue.Queue()
    listener, console, log_file = None, None, None
    stop_console = threading.Event()
    try:
        if args.support:controller.command('support='+args.support)
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log_file = args.log.open('w', encoding='utf-8')
        if keyboard_module:
            listener = global_key_listener(commands, keyboard_module)
        if not args.no_console:
            console = threading.Thread(target=console_listener, args=(commands, stop_console), daemon=True)
            console.start()
        print('离线串口（不是动力学仿真）' if args.dry_run else f'实机串口 {args.port} / {args.baud}', flush=True)
        print(HELP, flush=True)
        print('启动仅打开串口，不改变扭矩；尚未接管的舵机原使能状态未知。', flush=True)
        if args.auto:
            commands.put('arm')
            commands.put('run')
        elapsed, next_feedback, next_log = 0.0, 0.0, 0.0
        auto_compliance_pending = args.auto and args.compliance
        while not controller.quit and (args.duration is None or elapsed < args.duration):
            cycle_start = time.monotonic()
            # 热键线程只排队；状态改变、读反馈、写目标都在这个线程依次执行。
            for _ in range(commands.qsize()):
                command = commands.get_nowait()
                try:
                    if command.strip().lower() == 'help':
                        print(HELP, flush=True)
                        continue
                    controller.command(command)
                    if command.strip().lower() == 'status':
                        controller.feedback_time_s = elapsed
                    controller.show_status(include_feedback=command.strip().lower() == 'status')
                except ValueError as exc:
                    print(f'参数/操作未执行: {exc}', flush=True)
            if controller.quit:
                break
            before = (controller.mode, controller.forming, controller.settle_remaining > 0)
            compliance_due = (controller.compliance is not None and
                              (controller.feedback_time_s is None or elapsed-controller.feedback_time_s>=COMPLIANCE_FEEDBACK_PERIOD_S))
            if (elapsed>=next_feedback or compliance_due) and controller.mode!='released':
                if controller.compliance is not None:
                    controller.refresh_compliance_feedback()
                    period = COMPLIANCE_FEEDBACK_PERIOD_S
                else:
                    active = [i for i,e in enumerate(controller.enabled) if e]
                    controller.last_feedback = controller.bus.read_angles(active)
                    period = FEEDBACK_PERIOD_S
                controller.feedback_time_s = elapsed
                next_feedback = elapsed+period
            if auto_compliance_pending and not controller.forming and controller.settle_remaining<=0:
                controller.command('compliance');controller.command('run')
                controller.show_status()
                auto_compliance_pending = False
            controller.tick(CONTROL_PERIOD_S)
            after = (controller.mode, controller.forming, controller.settle_remaining > 0)
            if before != after:
                controller.show_status()
            if log_file and elapsed >= next_log:
                record = dict(time_s=elapsed, execution='dry_run' if args.dry_run else 'serial', **controller.snapshot())
                log_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                log_file.flush()
                next_log = elapsed + 0.1
            if args.fast:
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
            if console:
                console.join(timeout=.1)
            if log_file:
                log_file.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('用户中止。')
    except Exception as exc:
        print(f'实机控制停止: {exc}', flush=True)
        raise SystemExit(1)
