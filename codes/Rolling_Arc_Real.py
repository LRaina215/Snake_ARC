"""圆弧滚动实机控制：只运行本文件，通过热键直接控制飞特串口舵机。

默认连接真实串口；--dry-run 使用内存串口，不连接机器人。
Ctrl+U 形成圆弧，Ctrl+J/H 正/反单步，Ctrl+G 连续/暂停，Ctrl+X 全部失能。
不依赖 MuJoCo、EXE、相邻仓库或本仓库的其他 Python 模块。
"""
import argparse
import json
import math
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
    'K': 'status', 'X': 'release', 'Q': 'quit', 'T': 'reverse',
    '-': 'delta-', '=': 'delta+',
}
HELP = '''圆弧滚动实机调试（Ctrl+字母全局热键；任何窗口均会响应）：
  Ctrl+U 读取当前位置，使能局部并缓慢形成圆弧；默认完成后保持
  Ctrl+J / Ctrl+H 正向 / 反向单步；Ctrl+-/= 相位步长减半/加倍
  Ctrl+G 连续滚动/暂停；Ctrl+空格 暂停；Ctrl+N 保持并重发当前目标
  Ctrl+T 反转连续滚动方向；Ctrl+K 读取所有关节位置
  Ctrl+上/下 幅值±2°；Ctrl+右/左 角频率±0.05 rad/s
  Ctrl+X 全部失能（机器人会失去主动支撑）；Ctrl+Q 退出，接管后会失能
控制台输入后回车：apply / step / back / run / pause / hold / reverse
                  release / status / help / quit
参数：amplitude=60 omega=0.6 delta=0.1 window=8:13
窗口只能在 release 后修改。本阶段不实现顺应性或其他步态切换。
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
                FEEDBACK_PERIOD_S, SERIAL_TIMEOUT_S)
    if any(not math.isfinite(v) or v <= 0 for v in positive) or not math.isfinite(SETTLE_S) or SETTLE_S < 0:
        raise ValueError('控制周期、速度和反馈周期必须为正，静置时间非负')
    if not math.isfinite(INITIAL_PHASE_RAD) or not 1 <= SERVO_SPEED_RAW <= 32766:
        raise ValueError('初相位或舵机速度错误')
    if not 0 <= TORQUE_LIMIT_RAW <= 1000 or not 0 <= ACCELERATION_RAW <= 254:
        raise ValueError('力矩限制或加速度寄存器值错误')
    validate_parameters(AMPLITUDE_DEG, OMEGA_RAD_S, FIRST_JOINT, LAST_JOINT)


# ==================== 3. 飞特串口协议（只有主循环操作串口） ====================
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

    def prepare_enabled(self, angles, enabled):
        active_ids = [sid for sid, e in zip(SERVO_IDS, enabled) if e]
        for sid in active_ids:
            if self.read_register(sid, 0x21, 1)[0] != 0:
                raise RuntimeError(f'舵机 {sid} 不是单圈位置模式；脚本不会修改 EEPROM 模式')
        # 先把当前位置预置为目标，再开扭矩，避免恢复上一次运行遗留的目标。
        self.send_targets(angles, enabled)
        self.write(sync_packet(0x30, 2, [(sid, TORQUE_LIMIT_RAW.to_bytes(2, 'little')) for sid in active_ids]))
        self.write(sync_packet(0x29, 1, [(sid, bytes([ACCELERATION_RAW])) for sid in active_ids]))
        self.set_enabled(enabled)


# ==================== 4. 实验状态与热键功能 ====================
class RollingArcController:
    def __init__(self, bus):
        self.bus = bus
        self.amplitude = AMPLITUDE_DEG
        self.omega = OMEGA_RAD_S
        self.first, self.last = FIRST_JOINT, LAST_JOINT
        self.phase = INITIAL_PHASE_RAD
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
        self.mode, self.pending_phase = 'released', 0.0
        self.enabled = [False] * NUM_JOINTS
        self.touched_hardware = True
        self.bus.set_enabled(self.enabled, verify=verify)

    def change_parameters(self, amplitude=None, omega=None):
        candidate_a = self.amplitude if amplitude is None else amplitude
        candidate_w = self.omega if omega is None else omega
        validate_parameters(candidate_a, candidate_w, self.first, self.last)
        if candidate_a != self.amplitude and self.mode != 'released':
            self.forming, self.settle_remaining = True, SETTLE_S
        self.amplitude, self.omega = candidate_a, candidate_w

    def assign_parameters(self, command):
        """先验证整行，再一起更新；无效输入不留下部分修改。"""
        amplitude, omega, delta, first, last = self.amplitude, self.omega, self.phase_step, self.first, self.last
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
            else:
                raise ValueError('参数：amplitude=度 omega=rad/s delta=rad window=首:尾')
        validate_parameters(amplitude, omega, first, last)
        if not math.isfinite(delta) or not 0 < delta <= math.pi:
            raise ValueError('单步相位增量必须在 (0,pi] rad 内')
        if (first, last) != (self.first, self.last) and self.mode != 'released':
            raise ValueError('修改窗口前先 release；Step1 不做运动中窗口滑移')
        if amplitude != self.amplitude and self.mode != 'released':
            self.forming, self.settle_remaining = True, SETTLE_S
        self.amplitude, self.omega, self.phase_step = amplitude, omega, delta
        self.first, self.last = first, last

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
            self.last_feedback = self.bus.read_angles(range(NUM_JOINTS))
        elif command == 'hold':
            if self.mode == 'released':
                raise ValueError('还没有接管实机；先 apply')
            self.mode, self.pending_phase = 'hold', 0.0
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
            if self.mode != 'hold' and not self.forming:
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
        if self.mode == 'hold' and not self.forming:
            goal = self.targets.copy()  # hold 立即冻结已发目标，未到位的形成过程也可以中断
        rate = FORM_RATE_DEG_S if self.forming else MAX_TARGET_RATE_DEG_S
        for i, active in enumerate(self.enabled):
            if active:
                delta = max(-rate * dt, min(rate * dt, goal[i] - self.targets[i]))
                self.targets[i] += delta
        self.bus.send_targets(self.targets, self.enabled)
        if step_finished:
            self.mode = 'hold'  # 最后一小步的目标发送后再保持，避免停在倒数一帧
        if self.forming:
            self.forming = any(abs(goal[i] - self.targets[i]) > 1e-8 for i, e in enumerate(self.enabled) if e)
        else:
            self.settle_remaining = max(0.0, self.settle_remaining - dt)

    def snapshot(self):
        return dict(mode=self.mode, forming=self.forming, phase_rad=self.phase,
                    amplitude_deg=self.amplitude, omega_rad_s=self.omega, direction=self.direction,
                    phase_step_rad=self.phase_step, pending_phase_rad=self.pending_phase,
                    settle_remaining_s=self.settle_remaining,
                    window=[self.first, self.last], enabled=self.enabled.copy(),
                    target_controller_deg=self.targets.copy(),
                    feedback_time_s=self.feedback_time_s,
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
        if include_feedback:
            values = ' '.join(f'J{i + 1}={v:.1f}°' for i, v in self.last_feedback.items())
            print(f'实测控制器角：{values}', flush=True)


# ==================== 5. 离线串口，仅验证协议与交互，不代表物理运动 ====================
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


# ==================== 6. Ctrl 热键与控制台，参照 Adaptive_Pole_Climbing_Real ====================
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


# ==================== 7. 运行入口与退出清理 ====================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default=PORT)
    parser.add_argument('--baud', type=int, default=BAUD_RATE)
    parser.add_argument('--dry-run', action='store_true', help='内存串口，不打开真实设备')
    parser.add_argument('--no-hotkeys', action='store_true', help='用于离线批量验证')
    parser.add_argument('--no-console', action='store_true', help='不读取控制台输入')
    parser.add_argument('--auto', action='store_true', help='连接后直接形成圆弧并连续滚动')
    parser.add_argument('--duration', type=float, help='到时自动结束；默认等待 Esc')
    parser.add_argument('--fast', action='store_true', help='仅离线模式：取消实际等待')
    parser.add_argument('--log', type=Path, help='JSONL，记录发出的目标和实测反馈')
    args = parser.parse_args()
    if args.fast and not args.dry_run:
        parser.error('--fast 只能用于 --dry-run')
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
            controller.tick(CONTROL_PERIOD_S)
            after = (controller.mode, controller.forming, controller.settle_remaining > 0)
            if before != after:
                controller.show_status()
            if elapsed >= next_feedback and controller.mode != 'released':
                active = [i for i, enabled in enumerate(controller.enabled) if enabled]
                controller.last_feedback = controller.bus.read_angles(active)
                controller.feedback_time_s = elapsed
                next_feedback = elapsed + FEEDBACK_PERIOD_S
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
