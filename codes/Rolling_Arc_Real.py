"""圆弧滚动实机控制：只运行本文件，通过热键直接控制飞特串口舵机。

直接运行本文件，自动查找CH340串口；所有实验配置在文件顶部修改。
DRY_RUN=True时使用内存串口，不连接机器人；日志自动保存在本文件旁的logs目录。
默认启动只读预览；正视由关节重建的局部截面，显示全身实测角、使能和算法候选。
Ctrl+U 形成圆弧，Ctrl+J/H 正/反单步，Ctrl+G 连续/暂停，Ctrl+X 全部失能。
Ctrl+1/2/3 切换基础圆弧、固定截面顺应、仅角度反馈自动移窗与数量自适应。
失能时Ctrl+3按全身实测姿态选窗接管并进入顺应；Ctrl+U始终按配置窗口形成。
模式4转3会按实测重新选窗并撤去送链缓冲，成功后Ctrl+G开始滚动。
Ctrl+4 主动送链；Ctrl+J/H 各执行一段头侧增长/缩短，Ctrl+Y 提前返回，Ctrl+L沿任务方向强制推进一步。
Ctrl+5 从当前圆弧接入 Helix 调试；Ctrl+J 单步，Ctrl+G 连续/暂停。
Ctrl+6 接入 T 管重力矩越障；暂停时 Ctrl+L 确认实际姿态，Ctrl+J/G 再推进。
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
PREVIEW_PERIOD_S = .2          # 独立进程刷新，绘图不占用串口控制循环
LOG_DIRECTORY = Path(__file__).resolve().parent / 'logs'  # 不依赖终端/IDE工作目录
NUM_JOINTS = 20
FIRST_JOINT = 8                 # 关节从 1 开始编号，包含两个端点
LAST_JOINT = 13                 # 窗口外明确关闭舵机扭矩，不发送零角位置目标
AMPLITUDE_DEG = 60.0            # 圆弧幅值，单位：度
OMEGA_RAD_S = 0.6               # 连续滚动角频率，单位：rad/s
DIY_PARAMETERS = True          # 手调：关闭建形/接入的速度、幅值、开放形态和60°/0.6基准拦截
                              # 保留编码范围/有效读角；形成斜率、目标滤波仍按下面参数执行
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
# DIY默认按当前参数生成名义几何参考；关闭DIY才用冻结的60°/0.6正向标定。
# 移窗或扩缩每次只改变相邻边界，不指定6/8白名单。
MOTOR_SLIDING_AUTO = True
MOTOR_SLIDING_COOLDOWN_S = 5.0
MOTOR_SLIDING_SWITCH_ANGLE_DEG = 4.0
# 以下是最常调的数量判断参数；不添加未经验证的回正修正。
POSTURE_COUNT_HYSTERESIS_JOINTS = .5
POSTURE_PERSISTENCE_S = 1.0
POSTURE_CONTRACT_EXIT_ANGLE_DEG = 12.0
POSTURE_EXPANSION_TURN_ERROR_DEG = -5.0
POSTURE_CONTRACTION_TURN_ERROR_DEG = 5.0

# 主动送链：只用角度/时间，暂停滚动，首次按旧版模板建形；不是管径顺应。
FEED_DISTANCE_M = .0645        # Ctrl+J/H一次任务的材料量；四连杆试验改为.258
FEED_AMPLITUDE_DEG = 60.0      # 送链起态自己的幅值，不覆盖你的基础/顺应参数
FEED_PREPARATION_S = 6.0       # 首次形成；之后另等接管展直稳定才标定本步基线
FEED_AUTOMATIC_RETURN = False  # 手调默认单程，到位保持；不自动反向
FEED_FORCE_STEP_JOINTS = .05   # Ctrl+L强制推进的弯曲分布位移；单位关节间距，不是实测毫米
FEED_OPTIONS_OVERRIDE = {
    'tracking_tolerance_deg': 8.0,  # 手调接管/运动跟踪容差；原5°，不改目标或使能交接逻辑
}                             # 其他ContinuousFeedingOptions字段也可在这里覆盖
FEED_DIRECTION_OVERRIDE = {}   # 第3C段ManualFeedingDirectionOptions：选段/直线度/稳定时间
FEED_DIAGNOSTIC_PERIOD_S = 2.0 # 自动显示两侧判断数值；Ctrl+K随时查看

# ARC→Helix。半径是蛇身中心线的半径，螺距为每圈沿主管的前进量。
HELIX_RADIUS_M = .094
HELIX_PITCH_M = .13
HELIX_MIN_END_DEPTH_LINKS = 1.5 # 截面短端低于拟合圆心的深度不足时，先向内移一节
HELIX_STEP_S = 1.0             # Ctrl+J 单次最多推进的控制时间；等待新反馈时不消耗

# T 管重力矩越障，初值来自 18 关节、主管/支管 110/60 mm 的仿真。
# J1 在前。调这些参数时，以实物的右侧小 U 仍搭在主管、尾段可落向轴下为准。
GRAVITY_RETAIN_JOINTS = 8       # 释放时保持 J1 到此关节，其余关闭扭矩
GRAVITY_RELEASE_LINE_M = .32201388
GRAVITY_RELEASE_RADIUS_M = .09554165
GRAVITY_OPEN_RADIUS_M = .098    # 释放后保留的前部稍微放松，尾部自然下落
GRAVITY_ALIGN_RATE_RAD_S = .14  # Ctrl+J/H 对准支管时的滚动速度
GRAVITY_SHIFT_RATE_M_S = .012   # Shift Rolling 最大曲线进给速度
GRAVITY_SHIFT_ROLL_RAD_S = .35 # 每 50 mm 曲线进给叠加的滚动相位，按进给比例缩放
GRAVITY_INTAKE_ERROR_DEG = 16.  # 尾侧相邻关节跟踪到螺旋的确认角差

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
    'W': 'window_auto', 'Y': 'feed_return', 'L': 'feed_tick',
    '-': 'delta-', '=': 'delta+',
}
HELP = '''圆弧滚动调试（Ctrl+字母全局热键；任何窗口均会响应）：
  Ctrl+U 读取当前位置，使能局部并缓慢形成圆弧；默认完成后保持
  Ctrl+J / Ctrl+H 正向 / 反向单步；Ctrl+-/= 相位步长减半/加倍
  Ctrl+G 连续滚动/暂停；Ctrl+空格 暂停；Ctrl+N 保持并重发当前目标
  Ctrl+T 反转连续滚动方向；Ctrl+K 读取所有关节位置
  Ctrl+E 固定窗口截面顺应开/关：先形成圆弧；保留相位与连续/暂停状态
  Ctrl+1 / 2 / 3 基础圆弧 / 固定截面顺应 / 角度反馈自动滑窗
  失能时Ctrl+3：按实测姿态选初始窗口并形成；Ctrl+U：按配置窗口形成
  模式4→3：按实测重新选窗接管，不继承送链缓冲使能；成功后Ctrl+G滚动
  Ctrl+4 主动送链：J/H各执行一段头侧增长/缩短，到位保持，-/=目标量减半/加倍
  送链中Ctrl+G暂停/继续；Ctrl+L沿任务方向强制推进一步，目标发完保持；Ctrl+Y提前返回起态
  Ctrl+5 选择 ARC→Helix，选择时保持现状；Ctrl+J 单步，Ctrl+G 连续/暂停
  Ctrl+6 选择 T 管重力矩越障（J1 为前端）；Ctrl+J/G 推进，Ctrl+L 确认实物姿态
  T 管模式对准支管时 Ctrl+J/H 正反单步；Ctrl+空格暂停，Ctrl+X 全部失能
  Ctrl+W 自动移窗与扩缩开/关（仅滑动模式）；外侧扭矩关闭
  Ctrl+上/下 幅值±2°；Ctrl+右/左 角频率±0.05 rad/s
  Ctrl+X 全部失能（机器人会失去主动支撑）；Ctrl+Q 退出，接管后会失能
控制台输入后回车：apply / step / back / run / pause / hold / reverse
                  compliance / gait_basic / gait_fixed / gait_sliding / gait_feeding / gait_helix / window_auto
                  feed_return / feed_tick；feed_mm=258设置送链量（不立即运动）
                  release / status / help / quit
参数：amplitude=60 omega=0.6 delta=0.1 window=8:13 gain=0.02 filter=0.15
window只在release后修改；自动滑窗只用全20关节角、采样时间和已知相位。
DIY_PARAMETERS=True时三模式均可调幅值/频率，滑窗采用当前参数几何参考。
关闭DIY才使用60度/0.6/正向冻结标定；移窗不等于已观测到支撑或卸载。
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
    if not math.isfinite(protocol_deg):
        raise ValueError(f'J{index + 1} 目标角必须为有限值')
    if not DIY_PARAMETERS and not MIN_ANGLE_DEG[index] <= protocol_deg <= MAX_ANGLE_DEG[index]:
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
    if not math.isfinite(amplitude) or amplitude < 0:
        raise ValueError('幅值必须为有限非负数')
    if not math.isfinite(omega) or omega < 0:
        raise ValueError('角频率必须为有限非负数')
    if DIY_PARAMETERS:
        return  # 不根据幅值推算整周期峰值；逐帧仍核对实际要编码的目标。
    if not 0 < amplitude <= 90:
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
    if DIY_PARAMETERS:
        return
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
    # The tail half-link is a separate module after the final interior joint.
    add(lengths[-1]/2,0,0,len(points)-1,'end')
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
    yielding: float = .7  # Legacy configuration field; ignored by measured-section control.
    geometry_period_s: float = .05
    target_rate_deg_s: float = 45.
    target_filter_s: float = .15
    maximum_section_turn_deg: float = 190.
    minimum_gap_fraction: float = .4
    correct_discrete_turn: bool = True

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Nonfinite parameter')
        if not 0 <= self.gain < .2:
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
        self.last_goal = None
        self.next_geometry_s = 0.
        self.status = 'initializing'
        self.updates = 0
        self.clipped = 0
        self.inverse_diagnostic = None

    def _match_measured_deformation(self, measured, geometry, candidate):
        """Correct only the curve-to-hinge discretization, using measured shape.

        The desired turn is that of the measured curve AFTER the original
        length-preserving deformation, never pi or a nominal gait amplitude.
        A scalar inverse solve preserves the integrator's relative joint shape
        and rolling phase. It prevents a positive curvature bias being erased
        by projection/integration/discretization losses.
        """
        desired = sum(abs(s.turn)*(1+self.options.gain*(.5 if s.joint in (1, self.count) else 1.))
                      for s in geometry['curve'] if 1 <= s.joint <= self.count)
        def evaluate(scale):
            q = measured.copy()
            q[self.enabled] = candidate[self.enabled]*scale
            shape = local_section(q, self.lengths, self.first, self.last, geometry['axis'])
            return shape['total_turn'], q
        raw_turn, best = evaluate(1.)
        best_scale, best_error = 1., abs(raw_turn-desired)
        lower, upper = .25, 2.
        # Do not search across a hinge fold; this is an inverse conversion,
        # not a request to drive the motors past the existing joint range.
        peak = float(np.max(abs(candidate[self.enabled])))
        if peak > 1e-12:
            upper = min(upper, (math.pi/2-.01)/peak)
        upper = max(lower, upper)
        for _ in range(10):
            scale = .5*(lower+upper)
            turn, q = evaluate(scale)
            error = abs(turn-desired)
            if error < best_error:
                best, best_error, best_scale = q, error, scale
            if turn < desired:
                lower = scale
            else:
                upper = scale
        self.inverse_diagnostic = dict(source='measured_deformed_curve', scale=best_scale,
            requested_turn_deg=math.degrees(desired), raw_turn_deg=math.degrees(raw_turn),
            residual_deg=math.degrees(best_error), nominal_used=False)
        return best

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
        if getattr(self.p, 'manual_tuning', False):
            valid = True  # DIY只记录形态，不按研究阈值拒绝目标。
        return valid, geometry

    def update(self, measured, phase, time_s, dt, reference_targets=None):
        """Read actual shape and return one full-size angle command.

        Only self.enabled is driven by the adapter. Passive entries carry the
        measured angles so that later geometry checks use the actual boundary.
        reference_targets is accepted for old callers but deliberately ignored:
        an initial/nominal arc must not restore or replace measured geometry.
        """
        measured = np.asarray(measured)
        if self.targets is None:
            self.targets = measured.copy()
        if time_s + 1e-10 >= self.next_geometry_s:
            self.next_geometry_s = time_s + self.options.geometry_period_s
            self.updates += 1
            geometry = local_section(measured, self.lengths, self.first, self.last, self.previous_axis)
            self.geometry = geometry
            self.previous_axis = geometry['axis']
            links = self.lengths[self.first-1:self.last+2]
            observed, _ = yamada_section(
                geometry['curve'], links, self.first, self.count, phase,
                self.options.gain, self.p.direction,
            )
            # Reference repository: deform the ACTUAL section at constant arc
            # length, integrate its curvature, and use those angles directly.
            goal = measured.copy()
            goal[self.enabled] = observed
            if self.options.correct_discrete_turn:
                goal = self._match_measured_deformation(measured, geometry, goal)
            if not getattr(self.p, 'manual_tuning', False):
                goal[self.enabled] = np.clip(goal[self.enabled], -math.pi/2 + .01, math.pi/2 - .01)
            valid, goal_geometry = self._check_goal(goal)
            if not valid:
                self.clipped += 1
            self.last_goal, self.target_geometry = goal, goal_geometry
            self.status = 'running' if valid else 'measured-section target outside opening'
        # Hold the candidate between geometry updates; filter it at every
        # physics/control tick. The next geometry update uses the new phase.
        if self.last_goal is not None and self.status == 'running':
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
                    control_law='direct_measured_section', nominal_restoring=False,
                    inverse_conversion=self.inverse_diagnostic,
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
    manual_tuning: bool = False

    @property
    def enabled(self):
        indices = np.arange(1, self.joints+1)
        return (indices >= self.first_joint) & (indices <= self.last_joint)

    def targets(self, phase):
        return np.radians(joint_angles_deg(phase, self.amplitude_deg, self.first_joint, self.last_joint,
                                          self.length_normalized))


def make_section_options():
    return SectionOptions(gain=COMPLIANCE_GAIN,
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
        if self.last_control_time is not None and control_time_s < self.last_control_time:
            raise ValueError('Control clock cannot run backwards')
        if self.last_control_time == control_time_s:
            # A duplicate timestamp has no elapsed time. Do not invent dt,
            # feedback, phase progress or another window transaction.
            self.fresh_feedback = False
            return self.targets.copy(), self.enabled.copy()
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


def feature_arc_candidates(angles_deg,lengths,down=None,focus=None,current_window=None):
    angles=np.asarray(angles_deg,float);lengths=np.asarray(lengths,float)
    n=len(lengths)-1
    if angles.shape!=(n,) or not np.isfinite(angles).all():
        raise ValueError('全关节角反馈必须完整且有限')
    points=forward(np.radians(angles),lengths)
    crest=None
    if down is not None:
        down=np.asarray(down,float);down=down/np.linalg.norm(down)
        heights=-points @ down
        crest=float(np.mean(np.flatnonzero(heights>=heights.max()-1e-8)))
    support=None
    if focus is not None:
        joints=np.arange(1,n+1)
        curved=joints[(joints>=focus[0]) & (joints<=focus[1]) & (abs(angles)>=5.)]
        if len(curved):support=(int(curved[0]),int(curved[-1]))
    candidates=[];rejections={}
    for first in range(2,n-1):
        for last in range(first+2,n):
            if support is not None and not (first<=support[0] and last>=support[1]):continue
            captured=float(abs(angles[first-1:last]).sum())
            if captured<90.:
                rejections['局部弯曲不足90°']=rejections.get('局部弯曲不足90°',0)+1
                continue
            # Prune only obvious multiple-fold spans before expensive fitting.
            if captured>540.:continue
            try:
                # Reuse one FK for every candidate. The absolute polyline
                # turns equal the smooth A/B/C construction's module turns;
                # constructing all those curve objects would waste serial time.
                arc=points[first:last+2]
                center=arc.mean(axis=0)
                _,_,basis=np.linalg.svd(arc-center,full_matrices=False)
                plane_rms=float(np.sqrt(np.mean(((arc-center)@basis[-1])**2)))
                projected=(points[first-1:last+3]-center)@basis[:2].T
                xy=projected[1:-1]
                gap=float(np.linalg.norm(arc[-1]-arc[0])/np.linalg.norm(np.diff(arc,axis=0),axis=1).sum())
                delta=np.diff(projected,axis=0);norm=np.linalg.norm(delta,axis=1)
                if norm.min()<1e-8:continue
                tangents=delta/norm[:,None]
                turns=np.arctan2(tangents[:-1,0]*tangents[1:,1]-tangents[:-1,1]*tangents[1:,0],
                                (tangents[:-1]*tangents[1:]).sum(axis=1))
                turns[abs(turns)<1e-7]=0.  # same straight-module tolerance as continuous_section
                turn=math.degrees(float(abs(turns[:last-first+1]).sum()))
                if not 90.<=turn<=240. or gap<.35 or plane_rms>.025:continue
                fit=np.linalg.lstsq(np.column_stack([2*xy,np.ones(len(xy))]),(xy*xy).sum(axis=1),rcond=None)[0]
                radius=math.sqrt(max(0.,fit[2]+fit[:2]@fit[:2]));radial=xy-fit[:2]
                rms=float(np.sqrt(np.mean((np.linalg.norm(radial,axis=1)-radius)**2)))
                polar=np.unwrap(np.arctan2(radial[:,1],radial[:,0]))
                coverage=math.degrees(abs(polar[-1]-polar[0]))
                if not 90.<=coverage<=240. or rms>.025:continue
                score=abs(turn-180.)/60.+abs(coverage-180.)/60.+rms/.015+.03*(last-first+1)
                alignment=None
                if down is not None:
                    opening=(arc[0]+arc[-1])/2-arc.mean(axis=0)
                    alignment=float(opening@down/max(np.linalg.norm(opening),1e-12))
                    # Direction is ranking evidence; the feature exists even
                    # without it. A reversed old arc must not veto this scan.
                    score+=3.*max(0.,.5-alignment)
                    if not first<crest<last+1:score+=3.
                if current_window is not None:
                    score+=.04*abs(sum((first,last))-sum(current_window))
                candidates.append(dict(window=[first,last],score=score,turn_deg=turn,coverage_deg=coverage,
                    gap_fraction=gap,plane_rms_m=plane_rms,circle_rms_m=rms,
                    opening_alignment=alignment,conditional_crest_joint=crest,
                    captured_bending_fraction=captured/max(float(abs(angles).sum()),1e-12)))
            except (ValueError,np.linalg.LinAlgError):continue
    candidates.sort(key=lambda c:(c['score'],c['window']))
    return dict(chosen=None if not candidates else candidates[0]['window'],
        best=None if not candidates else candidates[0],candidates=candidates[:5],crest_joint=crest,
        diagnostics=dict(eligible_count=len(candidates),rejection_counts=rejections),
        input='whole_measured_link_geometry',uses_down=down is not None,uses_feeding_focus=support is not None,
        observes_pipe_contact=False)


def locate_measured_arc(angles_deg, lengths, width, down, current_window=None):
    """Compatibility entry: width is no longer a constraint on feature search."""
    return feature_arc_candidates(angles_deg,lengths,down=down,current_window=current_window)


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
        self.direction_observer = WholeFeedingObserver(lengths,(2,3))
        self.runtime_down = None
        self.runtime_quality = None
        self.feature_cache = None

    def passive_down(self, points, window, section):
        """Conditional gravity proxy; one straight long end is not a sensor.

        The open-side hemisphere rejects a pulled-up end, but cannot rule out
        every force/contact that tilts a straight arm. Truth is evaluated later.
        """
        if self.runtime_quality is not None:
            # Candidate evaluation may not borrow a reference it would enable.
            for name,links,_ in self.direction_observer.reference_recipe(self.runtime_quality) if self.runtime_down is not None else []:
                if not (links[1]-1 < window[0] or links[0] > window[1]):
                    return None,dict(source='candidate_would_enable_direction_reference')
            return self.runtime_down,self.runtime_quality
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
        points=forward(np.radians(angles),self.lengths)
        self.runtime_down,self.runtime_quality=self.direction_observer.down_direction(
            points,window,angles,feedback_time_s)
        r = dict(feedback_time_s=feedback_time_s, current_window=list(window), external_window_change=changed,
                 execution='observation_only', executes_window_update=False, valid=False, reasons=[],
                 direction=0, suggested_window=None, state='unobservable',
                 localisation='intrinsic_same_winding_bending', observes_absolute_down=False)
        try:
            key=(tuple(angles),window,None if self.runtime_down is None else tuple(self.runtime_down))
            if self.feature_cache is None or self.feature_cache[0]!=key:
                self.feature_cache=(key,feature_arc_candidates(angles,self.lengths,down=self.runtime_down,current_window=window))
            feature=self.feature_cache[1]
            r.update(global_arc=feature,end_quality=self.runtime_quality,
                     localisation='whole_body_measured_arc_feature')
            if feature['chosen'] is None:
                raise ValueError('No reconstructed open arc feature in full measured body')
            try:
                good,geometry,section=self.geometry_observer.geometry(np.radians(angles),window)
            except (ValueError,np.linalg.LinAlgError) as exc:
                good,geometry,section=False,dict(error=str(exc)),None
            r['current_geometry']=geometry
            center=sum(window)/2
            desired=tuple(feature['chosen']);offset=sum(desired)/2-center
            r.update(valid=True,position_valid=True,features={},
                     phase_reference=None,feature_error=[offset,0.],filtered_feature_error=[offset,0.],
                     filtered_error_joints=offset,estimated_bending_center_joint=sum(desired)/2)
            aligned=None
            if self.runtime_down is not None:
                r['down_head']=self.runtime_down.tolist()
                if section is not None:
                    arc=section['points'][1:-1]
                    opening=(arc[0]+arc[-1])/2-arc.mean(axis=0)
                    aligned=float(opening@self.runtime_down/max(np.linalg.norm(opening),1e-12))
                    r['current_opening_alignment']=aligned
            disjoint=desired[0]>window[1] or desired[1]<window[0]
            if desired!=window and (disjoint or not good or abs(offset)>2. or (aligned is not None and aligned<0.)):
                r.update(state='relocalize',relocalize_window=list(desired))
                return r
            # Half-joint changes can reflect alternate hinge families during
            # rolling. Position shifts require a full-joint centre difference;
            # width changes are separately persisted by the count estimator.
            direction=1 if offset>0 else -1
            if abs(offset)<1.-1e-8:
                self.pending_since,self.pending_direction=None,0
                r['state']='retain'
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
                    gravity_assumption='optional stable passive material segments',
                    position_definition='whole-body measured open-arc feature; phase reference is legacy metadata',
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
        self.travel = 0.
        self.direction = 0

    def update(self, count, turn_deg, phase_rad, time_s, nominal_turn_deg=180.):
        if not all(math.isfinite(v) for v in (count, turn_deg, phase_rad, time_s, nominal_turn_deg)):
            raise ValueError('Finite angle-derived count, turn, phase and time required')
        if self.last_time is not None:
            if time_s <= self.last_time:
                raise ValueError('Only new feedback may update phase statistics')
            delta = phase_rad-self.last_phase
            direction = 0 if abs(delta) <= 1e-8 else (1 if delta > 0 else -1)
            if (time_s-self.last_time > self.maximum_feedback_gap_s or
                    (direction and self.direction and direction != self.direction)):
                self.reset()
            else:
                self.travel += abs(delta)
            if direction:
                self.direction = direction
        sample = dict(count=float(count), turn_deg=float(turn_deg), nominal_turn_deg=float(nominal_turn_deg),
                      phase_rad=float(phase_rad), travel_rad=self.travel, time_s=float(time_s))
        if self.samples and abs(phase_rad-self.samples[-1]['phase_rad']) <= 1e-8:
            self.samples[-1] = sample
        else:
            self.samples.append(sample)
        self.last_time, self.last_phase = time_s, phase_rad
        self.samples = [s for s in self.samples if s['travel_rad'] >= self.travel-self.phase_span_rad-1e-8]
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
                    represented_phase_span_rad=0. if not self.samples else self.samples[-1]['travel_rad']-self.samples[0]['travel_rad'],
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


def measured_half_circle(section):
    """Body-circle coverage from measured points, independent of gait amplitude.

    The circle is not a pipe measurement. Polar-angle sweep describes covered
    arc, whereas tangent turning includes boundary effects of the reconstruction.
    """
    xy = section['projected'][1:-1]
    fit = np.linalg.lstsq(np.column_stack([2*xy, np.ones(len(xy))]), np.sum(xy*xy, axis=1), rcond=None)[0]
    center = fit[:2]
    radius = math.sqrt(max(0., fit[2]+center@center))
    radial = xy-center
    angles = np.unwrap(np.arctan2(radial[:, 1], radial[:, 0]))
    coverage = float(abs(angles[-1]-angles[0]))
    rms = float(np.sqrt(np.mean((np.linalg.norm(radial, axis=1)-radius)**2)))
    return dict(radius_m=radius, circle_rms_m=rms, coverage_deg=math.degrees(coverage),
                arc_length_m=radius*coverage, turn_deg=math.degrees(section['total_turn']))


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
        if not (2 <= window[0] <= window[1] < parameters.joints and window[1]-window[0]+1 >= 3):
            raise ValueError('Initial window requires 3+ interior joints')
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
        self.relocalization = None
        self.relocation_intent = None
        self.relocation_since = None
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
        # Over/under-covered arcs must remain observable so they can be resized.
        self.observer.geometry_observer.options = replace(self.observer.geometry_observer.options,
            minimum_turn_deg=30., maximum_turn_deg=330., minimum_gap=.08, maximum_plane_rms_m=.025)
        self.observation = None
        self.filtered_count = None
        self.count_history = PhaseBalancedCount(maximum_feedback_gap_s=2.5*self.feedback_period_s)
        self.count_observation = None
        self.intent = self.intent_since = None

    def _position_evidence(self, window=None):
        """Use the observer's persistent intrinsic recommendation when down is absent.

        This localises bending along the chain; it does not infer gravity or
        support. Without a down proxy this fallback authorises only the exact
        recommended slide, never a resize or an unrelated candidate.
        """
        o = self.observation
        if not o or not o.get('valid') or not o.get('position_valid', False):
            return None
        if o.get('global_arc',{}).get('chosen'):
            return 'whole_body_measured_arc_feature'
        if 'down_head' in o:
            return 'passive_down_and_bending'
        suggested = o.get('suggested_window')
        if o.get('state') in ('recommend_head', 'recommend_tail') and suggested:
            if window is None or tuple(window) == tuple(suggested):
                return 'persistent_intrinsic_bending_and_boundary'
        return None

    def _candidate(self, angles, window):
        if not (2 <= window[0] <= window[1] < self.p.joints and window[1]-window[0]+1 >= 3):
            return dict(window=list(window), eligible=False, reason='robot boundary')
        try:
            good, geometry, section = self.observer.geometry_observer.geometry(np.radians(angles), window)
            down, quality = self.observer.passive_down(forward(np.radians(angles), self.lengths), window, section)
            measured = measured_half_circle(section)
            intrinsic = self._position_evidence(window) == 'persistent_intrinsic_bending_and_boundary'
            resizing = window[1]-window[0] != self.window[1]-self.window[0]
            reason = ('candidate_shape_unreliable' if not good else
                      'candidate_circle_fit_unreliable' if resizing and measured['circle_rms_m'] > self.posture_options.circle_rms_max_m else
                      'candidate_has_no_position_reference' if down is None and not intrinsic and self._position_evidence(window)!='whole_body_measured_arc_feature' else None)
            return dict(window=list(window), eligible=reason is None,
                        geometry=geometry, actual_body_circle=measured,
                        circle_fit_required=resizing,
                        end_source=quality['source'], position_evidence=self._position_evidence(window), reason=reason)
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
        actual = measured_half_circle(local_section(q, self.lengths, *self.window))
        # L_half = pi*R_body. The active arc length is R_body*coverage;
        # using the ratio avoids confusing a body-radius fit with pipe radius.
        estimated_count = width*180./max(actual['coverage_deg'], 1e-8)
        if actual['circle_rms_m'] > self.posture_options.circle_rms_max_m:
            self.count_history.reset()
            self.count_observation = dict(valid=False, reason='body circle fit not reliable', actual_body_circle=actual)
            return
        phase_statistics = self.count_history.update(estimated_count, actual['coverage_deg'], self.feedback_phase,
                                                    self.last_time, 180.)
        self.filtered_count = phase_statistics['balanced_count_joints']
        self.count_observation = dict(valid=True, time_s=self.last_time, window=list(self.window),
            raw_count_joints=estimated_count, actual_body_circle=actual, nominal_body_circle=None,
            target_coverage_deg=180., required_arc_length_m=math.pi*actual['radius_m'],
            phase_statistics=phase_statistics, observes_support=False, observes_pipe_radius=False)

    def _select(self, angles, time_s):
        o, count = self.observation, self.count_observation
        if o and o.get('state')=='relocalize':
            destination=tuple(o['relocalize_window'])
            if self.relocation_intent!=destination:
                self.relocation_intent,self.relocation_since=destination,time_s
            self.selection=dict(state='confirming_global_relocation',chosen=list(destination),
                global_arc=o['global_arc'],length_reason='current_window_lost_upper_arc')
            if time_s-self.relocation_since>=self.posture_options.persistence_s:
                union=(min(self.window[0],destination[0]),max(self.window[1],destination[1]))
                _,quality=self.observer.passive_down(forward(np.radians(angles),self.lengths),union,None)
                if quality.get('source')=='candidate_would_enable_direction_reference':
                    self.selection['state']='relocation_reference_conflict'
                    return
                mask=(np.arange(1,self.p.joints+1)>=union[0]) & (np.arange(1,self.p.joints+1)<=union[1])
                goal=self.targets.copy();goal[~self.enabled]=np.radians(angles)[~self.enabled]
                keep=(np.arange(1,self.p.joints+1)>=destination[0]) & (np.arange(1,self.p.joints+1)<=destination[1])
                goal[mask & ~keep]=0.
                self.targets[mask & ~self.enabled]=np.radians(angles)[mask & ~self.enabled]
                self.relocalization=dict(old=list(self.window),to=list(destination),goal=goal,
                    keep=keep,start_time=time_s,ready_since=None)
                self.enabled,self.window=mask,union
                self.phase_hold=True
                self.pending,self.trial=None,None
                self.events.append(dict(kind='relocalization_start',time_s=time_s,to=list(destination),
                    old=self.relocalization['old'],phase_paused=True,observes_support=False))
            return
        self.relocation_intent=self.relocation_since=None
        evidence = self._position_evidence()
        if evidence is None:
            self.intent = self.intent_since = None
            reason = ('waiting_angle_feedback' if not o else 'current_shape_unreliable' if not o['valid'] else
                      'no_passive_down_or_persistent_intrinsic_recommendation')
            self.selection = dict(state='retain_untrusted_position', length_reason=reason,
                                  observer_reasons=[] if not o else o.get('reasons', []), observes_support=False)
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
            length_reason = 'measured_coverage_below_half_circle'
        elif ready and self.filtered_count <= width-option.count_hysteresis_joints and stats['balanced_turn_error_deg'] >= option.contraction_turn_error_deg:
            delta_width = -1
            length_reason = 'measured_coverage_above_half_circle'
        if evidence == 'persistent_intrinsic_bending_and_boundary':
            delta_width = 0
            length_reason = 'intrinsic_slide_only_resize_waits_for_down'
        offset = float(o['filtered_error_joints'])
        sign = 1 if offset > 0 else -1
        bending = o['filtered_feature_error']
        corroborated = sign*bending[0] >= self.observer.options.bending_threshold_joints or sign*bending[1] >= self.observer.options.boundary_threshold_deg
        migration = sign if abs(offset) >= self.observer.options.crest_threshold_joints and corroborated else 0
        if evidence=='whole_body_measured_arc_feature':
            migration=int(o['direction']) if o.get('state') in ('recommend_head','recommend_tail') else 0
        if evidence == 'persistent_intrinsic_bending_and_boundary':
            migration = int(o['direction'])
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
            coverage = c.get('actual_body_circle', {}).get('coverage_deg', float('inf'))
            c['score'] = abs(coverage-180.)/30.+.25*c['center_error_joints']
            if operation == 'contract' and actual is not None:
                c['eligible'] &= abs(coverage-180.)+1. < abs(actual['coverage_deg']-180.)
        usable = [c for c in candidates if c['eligible']]
        best = min(usable, key=lambda c: c['score']) if usable else None
        self.selection = dict(state=operation if best else 'retain' if not windows else 'blocked_candidates',
            current_window=list(self.window), estimated_count_joints=None if not count else count.get('raw_count_joints'),
            filtered_count_joints=self.filtered_count, actual_body_circle=actual, nominal_body_circle=reference,
            phase_statistics=stats, length_reason=length_reason, position_evidence=evidence,
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
        if self.observation and self.observation.get('state')=='relocalize':
            self.pending=None
            return
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
            current_trusted = self._position_evidence(new) is not None
            evidence = request.get('rollback', False) or current_trusted
            shared = self.enabled & p.enabled
            self.last_gate = dict(time_s=time_s, candidate_geometry=candidate.get('geometry'),
                entering_joint=(entering+1).tolist(), exiting_joint=(exiting+1).tolist(),
                entering_angle_deg=angles[entering].tolist(), exiting_angle_deg=angles[exiting].tolist(),
                exiting_velocity_deg_s=self.angle_velocity_deg_s[exiting].tolist(),
                exit_angle_bound_deg=exit_bound,
                command_open=bool(core._check_goal(preload)[0]), next_command_open=bool(preview_core._check_goal(preview)[0]),
                next_target_step_deg=step_deg, phase_compatible=compatible, evidence_valid=bool(evidence),
                candidate_eligible=candidate['eligible'], candidate_reason=candidate.get('reason'),
                position_evidence=self._position_evidence(new), angle_velocity_available=self.angle_velocity_available,
                shared_preload_jump_rad=float(abs(preload[shared]-self.targets[shared]).max()),
                unloading_observed=False, phase_paused=False)
            self.last_gate['waiting_reasons'] = [name for name, ok in (
                (candidate.get('reason') or 'candidate_rejected', candidate['eligible']),
                ('position_evidence_changed', evidence), ('exit_angle_or_entry_phase', compatible),
                ('waiting_second_angle_sample', self.angle_velocity_available),
                ('preload_shape', self.last_gate['command_open']),
                ('next_command_shape', self.last_gate['next_command_open']),
                ('compliance_blocked', preview_core.status == 'running'),
                ('target_rate', step_deg <= self.section_options.target_rate_deg_s*.02+1e-8)) if not ok]
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
        if self.relocalization is not None:
            return self._advance_relocalization(angles_deg,feedback_time_s,control_time_s,kwargs.get('freeze_targets',False))
        rolling_back = self.pending is not None and self.pending.get('rollback', False)
        if self.resize_fault or (self.fault and not rolling_back):
            kwargs['freeze_targets'] = True
        try:
            result = super().advance(angles_deg, feedback_time_s, control_time_s, commanded_phase_rad, **kwargs)
        except (ValueError,np.linalg.LinAlgError):
            if not self.observation or self.observation.get('state')!='relocalize':raise
            # A straight/degenerate old section cannot veto a different valid
            # full-body feature. Keep its last commands while acquiring that arc.
            result=self.targets.copy(),self.enabled.copy()
        relocating=self.observation is not None and self.observation.get('state')=='relocalize'
        if self.fresh_feedback and not relocating:
            self._observe_count(np.asarray(angles_deg), control_time_s)
        if not relocating:self._review_trial(np.asarray(angles_deg), control_time_s)
        if (self.automatic and self.fresh_feedback and self.pending is None and self.trial is None and
                control_time_s >= self.cooldown_until and not self.fault and not self.resize_fault and
                not kwargs.get('freeze_targets', False)):
            self._select(np.asarray(angles_deg), control_time_s)
        if self.phase_hold and self.relocalization is None:
            raise RuntimeError('Normal posture-window exchanges must keep rolling')
        return self.targets.copy(),self.enabled.copy()

    def _advance_relocalization(self, angles_deg, feedback_time, time_s, frozen):
        """Acquire the measured upper arc, then straighten/release old motors.

        This exceptional recovery holds phase; normal adjacent slides still
        roll continuously. Every entering joint starts at its measured angle.
        """
        if not 0.<=time_s-feedback_time<=self.options.maximum_feedback_age_s:
            raise TimeoutError('Relocation needs fresh motor feedback')
        if self.last_control_time is not None and time_s<=self.last_control_time:
            if time_s<self.last_control_time:raise ValueError('Control clock cannot run backwards')
            return self.targets.copy(),self.enabled.copy()
        fresh=self.last_time is None or feedback_time>self.last_time
        if self.last_time is not None and feedback_time<self.last_time:raise ValueError('Feedback time cannot run backwards')
        angles=np.asarray(angles_deg,float);q=np.radians(angles)
        if not fresh and not np.array_equal(angles,self.previous_angles):raise ValueError('Changed angles with an unchanged feedback timestamp')
        dt=min(.02,time_s-self.last_control_time)
        self.last_control_time=time_s;self.fresh_feedback=fresh
        if fresh:self.previous_angles,self.last_time=angles.copy(),feedback_time
        r=self.relocalization
        error=float(abs(np.degrees(self.targets[self.enabled])-angles[self.enabled]).max())
        retiring=self.enabled & ~r['keep']
        if not frozen and error<=5. and self.fault is None:
            step=math.radians(self.section_options.target_rate_deg_s)*dt
            self.targets[self.enabled]+=np.clip(r['goal'][self.enabled]-self.targets[self.enabled],-step,step)
        self.targets[~self.enabled]=q[~self.enabled]
        exit_error=float(abs(angles[retiring]).max()) if retiring.any() else 0.
        target_error=float(abs(np.degrees(self.targets[retiring])).max()) if retiring.any() else 0.
        self.selection=dict(state='relocalizing_upper_arc',chosen=r['to'],tracking_error_deg=error,
            retiring_max_angle_deg=exit_error,phase_paused=True,length_reason='controlled_old_arc_release')
        if fresh and not frozen and self.fault is None:
            ready=exit_error<=self.options.switch_angle_deg and target_error<=self.options.switch_angle_deg and error<=5.
            r['ready_since']=(feedback_time if r['ready_since'] is None else r['ready_since']) if ready else None
            if r['ready_since'] is not None and feedback_time-r['ready_since']>=.2:
                self.window=tuple(r['to']);self.enabled=r['keep'].copy()
                p=replace(self.p,first_joint=self.window[0],last_joint=self.window[1])
                self.core=FixedSectionCompliance(p,self.lengths,self.section_options)
                self.core.targets=self.targets.copy()
                self.targets[~self.enabled]=q[~self.enabled]
                self.events.append(dict(kind='relocalization_complete',time_s=time_s,to=list(self.window),
                    max_exit_angle_deg=exit_error,observes_unloading=False))
                self.relocalization=None;self.phase_hold=False
                self.relocation_intent=self.relocation_since=None
                self.cooldown_until=time_s+self.posture_options.cooldown_s
                self._new_observer()
        if self.relocalization is not None and time_s-r['start_time']>10.:
            self.fault='global relocation tracking did not settle; hold and reselect mode'
        return self.targets.copy(),self.enabled.copy()

    def snapshot(self):
        result = super().snapshot()
        result['relocalization']=None if self.relocalization is None else {
            k:v for k,v in self.relocalization.items() if k not in ('goal','keep')}
        result.update(active_count=int(self.enabled.sum()), selection=copy.deepcopy(self.selection),
                      length_observation=copy.deepcopy(self.selection), trial=copy.deepcopy(self.trial),
                      resize_fault=self.resize_fault, posture_options=asdict(self.posture_options),
                      observer_configuration=self.observer.configuration(),
                      automatic=self.automatic,
                      count_observation=copy.deepcopy(self.count_observation),
                      stable_width_whitelist=None, preferred_count=None,
                      controller_revision='measured_half_circle_v4')
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


@dataclass(frozen=True)
class ManualArcReference(ArcParameters):
    """手调入口的真实名义目标；不复用60°冻结标定来冒充当前参数。"""
    manual_tuning: bool = True

    def __post_init__(self):
        validate_parameters(self.amplitude_deg, self.omega_rad_s, self.first_joint, self.last_joint)
        if self.joints != NUM_JOINTS or self.direction not in (-1, 1) or not math.isfinite(self.initial_phase_rad):
            raise ValueError('关节数、方向或相位错误')

    def targets(self, phase):
        if not math.isfinite(phase):
            raise ValueError('相位必须为有限值')
        return np.radians(joint_angles_deg(phase, self.amplitude_deg, self.first_joint, self.last_joint,
                                          length_normalized=self.last_joint-self.first_joint+1 != 6))


def parameter_window_feasible(parameters, lengths, window, margin=.01):
    if getattr(parameters, 'manual_tuning', False):
        # 截面算法仍需要内部3+关节和各端一根边界连杆；不预测速度/峰值来拦截。
        return 2 <= window[0] <= window[1] < len(lengths)-1 and window[1]-window[0]+1 >= 3
    return mechanically_feasible(lengths, window, margin)

# ==================== 3C. 主动送链核心：角度/时间输入，可直接调参 ====================
# BEGIN ACTIVE_FEEDING_CORE
class FeedingArcGeometry:
    """Original transfer mask/template; no unused observer or constructor gates."""
    def __init__(self, lengths):
        self.n = len(lengths)-1

    def mask(self, window):
        if not 2 <= window[0] <= window[1] < self.n:
            raise ValueError('Active arc needs passive boundary joints')
        j = np.arange(1, self.n+1)
        return (j >= window[0]) & (j <= window[1])

    def template(self, window, phase, amplitude, bias=0.):
        factors = np.where(np.arange(self.n) % 2 == 0, math.cos(phase), math.sin(phase))
        target = np.where(self.mask(window), math.radians(amplitude)*factors, 0.)
        # Move bending within the same arc; keep each alternating family's
        # total angle unchanged. This is a position action, not radius feedback.
        for parity in (0, 1):
            family = np.flatnonzero(self.mask(window) & (np.arange(self.n) % 2 == parity))
            if len(family) >= 2:
                target[family[0]] *= 1.-bias
                target[family[-1]] *= 1.+bias
        return target


@dataclass(frozen=True)
class FeedingDirectionOptions:
    """Angle-only quality thresholds; stability is intrinsic, not world motion."""
    minimum_links: int = 3
    maximum_links: int = 5
    minimum_straightness: float = .98
    stability_s: float = .4
    minimum_samples: int = 3
    maximum_shape_change_deg: float = 3.
    maximum_sample_gap_s: float = .6
    minimum_tip_depth_links: float = 2.5
    minimum_segment_depth_links: float = .5
    maximum_end_disagreement_deg: float = 25.


class WholeFeedingObserver:
    def __init__(self, lengths, initial_window, options=FeedingDirectionOptions()):
        self.lengths = np.asarray(lengths, float)
        self.distance = np.r_[0., np.cumsum(self.lengths)]
        self.initial_window = initial_window
        self.cut_depth_m = None
        self.options = options
        self._histories = {}
        self._selected = {}
        self._single_side = None
        self._last_feedback = None
        self._last_angles = None
        self._reference = None
        self._reference_epoch = 0
        self._coordinate_offset_m = 0.
        self._last_valid_time = None
        self._current_candidates = {}
        self._feature_cache = None

    def down_direction(self, points, window, angles_deg, feedback_time_s):
        """Search wholly passive spans; retain an eligible span to avoid chatter.

        History compares internal joint angles of each material span. Comparing
        directions in the head frame across time would mistake motion of the
        head frame for a changing gravity vector. Neither test observes gravity.
        """
        opt = self.options
        q = np.asarray(angles_deg, float)
        now = float(feedback_time_s)
        mean_length = float(np.mean(self.lengths))
        ends, selected, live_keys = {}, [], set()
        self._current_candidates = {}
        for name, available in (('head', window[0]-1), ('tail', len(self.lengths)-1-window[1])):
            lo, hi = (0, available) if name=='head' else (len(self.lengths)-available, len(self.lengths))
            tested, eligible, rejected = [], [], {}
            for count in range(opt.minimum_links, min(opt.maximum_links, available)+1):
                for start in range(lo, hi-count+1):
                    stop = start+count  # links [start, stop), points [start, stop]
                    key = (name, start, stop)
                    live_keys.add(key)
                    chord = points[stop]-points[start]
                    if name=='head':
                        chord = -chord  # always from the arc towards the free end
                    length = float(np.linalg.norm(chord))
                    straightness = length/float(self.lengths[start:stop].sum())
                    direction = chord/max(length, 1e-12)
                    heights = -points @ direction
                    tip = 0 if name=='head' else len(points)-1
                    below = float(heights.max()-heights[tip])
                    segment_depth = float(heights.max()-heights[start:stop+1].max())
                    reasons = []
                    if straightness < opt.minimum_straightness:
                        reasons.append('not_straight')
                    if below < opt.minimum_tip_depth_links*mean_length:
                        reasons.append('tip_not_below_arc')
                    if segment_depth < opt.minimum_segment_depth_links*mean_length:
                        reasons.append('segment_near_crest')
                    history = self._histories.get(key, [])
                    if reasons or (history and now-history[-1][0] > opt.maximum_sample_gap_s):
                        history = []
                    # A repeated encoder timestamp never counts as a new sample.
                    if not reasons and (not history or now > history[-1][0]):
                        history.append((now, q[start:stop-1].copy()))
                    # Keep BOTH the required time span and sample count. The
                    # previous time-only pruning could repeatedly discard the
                    # third good sample at real serial scan intervals (~.22s).
                    while len(history)>opt.minimum_samples and history[1][0] <= now-opt.stability_s+1e-9:
                        history.pop(0)
                    self._histories[key] = history
                    span = now-history[0][0] if history else 0.
                    change = float(np.ptp(np.array([h[1] for h in history]), axis=0).max()) if history else None
                    if not reasons:
                        if len(history)<opt.minimum_samples or span < opt.stability_s-1e-9:
                            reasons.append('collecting_stability')
                        if change is not None and change > opt.maximum_shape_change_deg:
                            reasons.append('shape_unstable')
                    candidate = dict(name=name, links=[start+1, stop],
                        internal_joints=[start+1, stop-1], count=count, available=available,
                        straightness=straightness, tip_below_global_top_m=below,
                        segment_below_global_top_m=segment_depth,
                        stable_span_s=span, samples=len(history), shape_change_deg=change,
                        direction=direction.tolist(), reasons=reasons)
                    tested.append(candidate)
                    self._current_candidates[key] = candidate
                    if not reasons:
                        eligible.append(candidate)
                    for reason in reasons:
                        rejected[reason] = rejected.get(reason, 0)+1
            def rank(c):
                return (c['count'], c['straightness'], -(c['shape_change_deg'] or 0.))
            prior = self._selected.get(name)
            chosen = next((c for c in eligible if c['links']==prior), None)
            if chosen is None and eligible:
                chosen = max(eligible, key=rank)
            if chosen is not None:
                self._selected[name] = chosen['links']
            if chosen is not None:
                selected.append(chosen)
            # Display one actionable candidate and rejection totals, not dozens
            # of nearly identical overlapping spans in every 20ms log row.
            best = chosen or (max(tested, key=lambda c: (
                not any(r in c['reasons'] for r in ('not_straight','tip_not_below_arc','segment_near_crest')),
                c['straightness'], c['count'])) if tested else None)
            ends[name] = dict(available=available, tested_count=len(tested),
                stable_count=len(eligible), selected=chosen, best=best,
                rejection_counts=rejected,
                reason='selected' if chosen else 'insufficient_passive_links' if not tested else 'no_eligible_segment')
        self._histories = {k:v for k,v in self._histories.items() if k in live_keys}
        quality = dict(source='no_stable_passive_segment', candidates=selected, ends=ends,
            thresholds=asdict(opt), observes_world_gravity=False,
            stability_basis='internal_joint_angles_in_material_segment')
        if not selected:
            self._single_side = None
            return None, quality
        if len(selected)==2:
            a,b = [np.array(c['direction']) for c in selected]
            disagreement = math.degrees(math.acos(float(np.clip(a@b,-1.,1.))))
            quality['end_disagreement_deg'] = disagreement
            if disagreement <= opt.maximum_end_disagreement_deg:
                weights = [c['count']*c['straightness'] for c in selected]
                down = weights[0]*a+weights[1]*b
                down /= np.linalg.norm(down)
                quality.update(source='global_two_consistent_segments', fusion_weights=weights)
                self._single_side = None
                return down, quality
            quality['fusion_rejected_reason'] = 'end_directions_disagree'
        chosen = next((c for c in selected if c['name']==self._single_side), None)
        if chosen is None:
            chosen = max(selected, key=rank)
        self._single_side = chosen['name']
        quality.update(source='global_single_'+chosen['name'], used_sides=[chosen['name']])
        return np.array(chosen['direction']), quality

    def reference_recipe(self, quality):
        candidates = quality['candidates']
        if quality['source']=='global_two_consistent_segments':
            weights = quality['fusion_weights']
        else:
            candidates = [c for c in candidates if c['name'] in quality['used_sides']]
            weights = [1.]
        return [(c['name'], tuple(c['links']), w) for c,w in zip(candidates,weights)]

    def previous_reference_now(self):
        """Reconstruct the old reference in the CURRENT head frame.

        Only still-passive, geometrically valid material spans are usable.
        Never freeze a head-frame vector or use a now-enabled arm as gravity.
        All alternative passive spans have been warming up in parallel.
        """
        down = np.zeros(3)
        for name,links,weight in self._reference:
            candidate = self._current_candidates.get((name,links[0]-1,links[1]))
            if candidate is None or any(r!='collecting_stability' for r in candidate['reasons']):
                return None
            down += weight*np.array(candidate['direction'])
        return down/np.linalg.norm(down)

    def cut_geometry(self, points, down):
        heights = -points @ down
        cut = float(heights.max()-self.cut_depth_m)
        peak = int(np.argmax(heights)); left=right=peak
        while left>0 and heights[left-1]>cut:
            left -= 1
        while right<len(points)-1 and heights[right+1]>cut:
            right += 1
        if left==0 or right==len(points)-1:
            raise ValueError('global_cut_has_no_two_hanging_crossings')
        def crossing(i,j):
            fraction = (cut-heights[i])/(heights[j]-heights[i])
            return float(self.distance[i]+fraction*(self.distance[j]-self.distance[i]))
        head = crossing(left-1,left)
        tail = float(self.distance[-1]-crossing(right,right+1))
        return heights,cut,left,right,head,tail

    def observe(self, angles_deg, window, feedback_time_s):
        try:
            t = float(feedback_time_s)
            angles = np.asarray(angles_deg, float)
            if not math.isfinite(t) or (self._last_feedback is not None and t < self._last_feedback):
                raise ValueError('nonmonotonic_or_invalid_feedback_time')
            if self._last_feedback == t and not np.array_equal(angles, self._last_angles):
                raise ValueError('changed_angles_with_same_feedback_time')
            self._last_feedback, self._last_angles = t, angles.copy()
            q = np.radians(angles_deg)
            points = forward(q, self.lengths)
            down, quality = self.down_direction(points, window, angles, t)
            if down is None:
                return dict(valid=False, reason=quality['source'], end_quality=quality)
            heights = -points @ down
            if self.cut_depth_m is None:
                first,last = self.initial_window
                self.cut_depth_m = float(heights.max()-.5*(heights[first]+heights[last+1]))
            heights,cut,left,right,head,tail = self.cut_geometry(points,down)
            raw_coordinate = .5*(head-tail)
            recipe = self.reference_recipe(quality)
            changed = self._reference is not None and [(a,b) for a,b,_ in recipe]!=[(a,b) for a,b,_ in self._reference]
            expired = self._last_valid_time is not None and t-self._last_valid_time>self.options.maximum_sample_gap_s
            event, correction = 'retained', 0.
            next_offset,next_epoch=self._coordinate_offset_m,self._reference_epoch
            if changed and not expired:
                old_down = self.previous_reference_now()
                compatible = old_down is not None and float(old_down@down)>=math.cos(math.radians(self.options.maximum_end_disagreement_deg))
                if compatible:
                    try:
                        *_,old_head,old_tail = self.cut_geometry(points,old_down)
                        correction = .5*(old_head-old_tail)-raw_coordinate
                        next_offset += correction
                        event = 'overlap_handover'
                    except ValueError:
                        compatible = False
                if not compatible:
                    event = 'reacquire_no_overlap'
            elif expired:
                event = 'reacquire_after_gap'
            if event.startswith('reacquire'):
                next_epoch += 1
                next_offset = 0.
            quality.update(reference_epoch=next_epoch, reference_event=event,
                handover_correction_m=correction, coordinate_offset_m=next_offset,
                reference_recipe=recipe)
            locations=[]; coordinates=[]; weights=[]
            for j,length in enumerate(self.lengths):
                for u in (np.arange(16)+.5)/16:
                    locations.append((1-u)*heights[j]+u*heights[j+1])
                    coordinates.append(j+u); weights.append(length/16)
            levels=np.array(locations)
            w=np.maximum(0.,levels-levels.max()+np.mean(self.lengths)/8)*weights
            crest=float(w@coordinates/w.sum())
            # Use the same measured feature as initial arming and rolling.
            # The length cut must not also define an expanding torque buffer.
            key=(tuple(angles),tuple(down))
            if self._feature_cache is None or self._feature_cache[0]!=key:
                self._feature_cache=(key,feature_arc_candidates(angles,self.lengths,down=down))
            feature=self._feature_cache[1]
            if feature['chosen'] is None:raise ValueError('no_open_arc_feature_for_feeding')
            arc_window=tuple(feature['chosen'])
            section=local_section(q,self.lengths,*arc_window)
            # Commit the reference only after the whole measurement succeeds.
            # Failed section reconstruction is not a fresh valid datum.
            self._reference, self._last_valid_time = recipe, t
            self._coordinate_offset_m,self._reference_epoch=next_offset,next_epoch
            return dict(valid=True, head_hanging_m=head, tail_hanging_m=tail,
                transfer_coordinate_m=raw_coordinate+self._coordinate_offset_m,
                raw_transfer_coordinate_m=raw_coordinate,
                above_cut_length_m=float(self.distance[-1]-head-tail),
                cut_height_head_frame_m=cut, cut_depth_m=self.cut_depth_m, crest_joint=crest,
                actual_arc_window=list(arc_window), down_head=down.tolist(), end_quality=quality,
                arc_feature=feature,
                plane_rms_m=section['fit_rms'], observes_pipe_contact=False, observes_world_position=False)
        except (ValueError,np.linalg.LinAlgError) as exc:
            return dict(valid=False,reason=str(exc))


@dataclass(frozen=True)
class ManualFeedingDirectionOptions:
    """Angle-only quality thresholds; stability is intrinsic, not world motion."""
    minimum_links: int = 3
    maximum_links: int = 5
    minimum_straightness: float = .98
    stability_s: float = .4
    minimum_samples: int = 3
    maximum_shape_change_deg: float = 3.
    maximum_sample_gap_s: float = .6
    minimum_tip_depth_links: float = 2.5
    minimum_segment_depth_links: float = .5
    maximum_end_disagreement_deg: float = 25.


class ManualFeedingObserver:
    def __init__(self, lengths, initial_window, options=ManualFeedingDirectionOptions()):
        self.lengths = np.asarray(lengths, float)
        self.distance = np.r_[0., np.cumsum(self.lengths)]
        self.initial_window = initial_window
        self.cut_depth_m = None
        self.options = options
        self._histories = {}
        self._selected = {}
        self._single_side = None
        self._last_feedback = None
        self._last_angles = None

    def down_direction(self, points, window, angles_deg, feedback_time_s):
        """Search wholly passive spans; retain an eligible span to avoid chatter.

        History compares internal joint angles of each material span. Comparing
        directions in the head frame across time would mistake motion of the
        head frame for a changing gravity vector. Neither test observes gravity.
        """
        opt = self.options
        q = np.asarray(angles_deg, float)
        now = float(feedback_time_s)
        mean_length = float(np.mean(self.lengths))
        ends, selected, live_keys = {}, [], set()
        for name, available in (('head', window[0]-1), ('tail', len(self.lengths)-1-window[1])):
            lo, hi = (0, available) if name=='head' else (len(self.lengths)-available, len(self.lengths))
            tested, eligible, rejected = [], [], {}
            for count in range(opt.minimum_links, min(opt.maximum_links, available)+1):
                for start in range(lo, hi-count+1):
                    stop = start+count  # links [start, stop), points [start, stop]
                    key = (name, start, stop)
                    live_keys.add(key)
                    chord = points[stop]-points[start]
                    if name=='head':
                        chord = -chord  # always from the arc towards the free end
                    length = float(np.linalg.norm(chord))
                    straightness = length/float(self.lengths[start:stop].sum())
                    direction = chord/max(length, 1e-12)
                    heights = -points @ direction
                    tip = 0 if name=='head' else len(points)-1
                    below = float(heights.max()-heights[tip])
                    segment_depth = float(heights.max()-heights[start:stop+1].max())
                    reasons = []
                    if straightness < opt.minimum_straightness:
                        reasons.append('not_straight')
                    if below < opt.minimum_tip_depth_links*mean_length:
                        reasons.append('tip_not_below_arc')
                    if segment_depth < opt.minimum_segment_depth_links*mean_length:
                        reasons.append('segment_near_crest')
                    history = self._histories.get(key, [])
                    if reasons or (history and now-history[-1][0] > opt.maximum_sample_gap_s):
                        history = []
                    # A repeated encoder timestamp never counts as a new sample.
                    if not reasons and (not history or now > history[-1][0]):
                        history.append((now, q[start:stop-1].copy()))
                    while len(history)>1 and history[1][0] <= now-opt.stability_s+1e-9:
                        history.pop(0)
                    self._histories[key] = history
                    span = now-history[0][0] if history else 0.
                    change = float(np.ptp(np.array([h[1] for h in history]), axis=0).max()) if history else None
                    if not reasons:
                        if len(history)<opt.minimum_samples or span < opt.stability_s-1e-9:
                            reasons.append('collecting_stability')
                        if change is not None and change > opt.maximum_shape_change_deg:
                            reasons.append('shape_unstable')
                    candidate = dict(name=name, links=[start+1, stop],
                        internal_joints=[start+1, stop-1], count=count, available=available,
                        straightness=straightness, tip_below_global_top_m=below,
                        segment_below_global_top_m=segment_depth,
                        stable_span_s=span, samples=len(history), shape_change_deg=change,
                        direction=direction.tolist(), reasons=reasons)
                    tested.append(candidate)
                    if not reasons:
                        eligible.append(candidate)
                    for reason in reasons:
                        rejected[reason] = rejected.get(reason, 0)+1
            def rank(c):
                return (c['count'], c['straightness'], -(c['shape_change_deg'] or 0.))
            prior = self._selected.get(name)
            chosen = next((c for c in eligible if c['links']==prior), None)
            if chosen is None and eligible:
                chosen = max(eligible, key=rank)
            self._selected[name] = None if chosen is None else chosen['links']
            if chosen is not None:
                selected.append(chosen)
            # Display one actionable candidate and rejection totals, not dozens
            # of nearly identical overlapping spans in every 20ms log row.
            best = chosen or (max(tested, key=lambda c: (
                not any(r in c['reasons'] for r in ('not_straight','tip_not_below_arc','segment_near_crest')),
                c['straightness'], c['count'])) if tested else None)
            ends[name] = dict(available=available, tested_count=len(tested),
                stable_count=len(eligible), selected=chosen, best=best,
                rejection_counts=rejected,
                reason='selected' if chosen else 'insufficient_passive_links' if not tested else 'no_eligible_segment')
        self._histories = {k:v for k,v in self._histories.items() if k in live_keys}
        quality = dict(source='no_stable_passive_segment', candidates=selected, ends=ends,
            thresholds=asdict(opt), observes_world_gravity=False,
            stability_basis='internal_joint_angles_in_material_segment')
        if not selected:
            self._single_side = None
            return None, quality
        if len(selected)==2:
            a,b = [np.array(c['direction']) for c in selected]
            disagreement = math.degrees(math.acos(float(np.clip(a@b,-1.,1.))))
            quality['end_disagreement_deg'] = disagreement
            if disagreement <= opt.maximum_end_disagreement_deg:
                weights = [c['count']*c['straightness'] for c in selected]
                down = weights[0]*a+weights[1]*b
                down /= np.linalg.norm(down)
                quality.update(source='global_two_consistent_segments', fusion_weights=weights)
                self._single_side = None
                return down, quality
            quality['fusion_rejected_reason'] = 'end_directions_disagree'
        chosen = next((c for c in selected if c['name']==self._single_side), None)
        if chosen is None:
            chosen = max(selected, key=rank)
        self._single_side = chosen['name']
        quality.update(source='global_single_'+chosen['name'], used_sides=[chosen['name']])
        return np.array(chosen['direction']), quality

    def observe(self, angles_deg, window, feedback_time_s):
        try:
            t = float(feedback_time_s)
            angles = np.asarray(angles_deg, float)
            if not math.isfinite(t) or (self._last_feedback is not None and t < self._last_feedback):
                raise ValueError('nonmonotonic_or_invalid_feedback_time')
            if self._last_feedback == t and not np.array_equal(angles, self._last_angles):
                raise ValueError('changed_angles_with_same_feedback_time')
            self._last_feedback, self._last_angles = t, angles.copy()
            q = np.radians(angles_deg)
            points = forward(q, self.lengths)
            down, quality = self.down_direction(points, window, angles, t)
            if down is None:
                return dict(valid=False, reason=quality['source'], end_quality=quality)
            heights = -points @ down
            if self.cut_depth_m is None:
                first,last = self.initial_window
                self.cut_depth_m = float(heights.max()-.5*(heights[first]+heights[last+1]))
            cut = float(heights.max()-self.cut_depth_m)
            peak = int(np.argmax(heights)); left=right=peak
            while left>0 and heights[left-1]>cut:
                left -= 1
            while right<len(points)-1 and heights[right+1]>cut:
                right += 1
            if left==0 or right==len(points)-1:
                return dict(valid=False, reason='global_cut_has_no_two_hanging_crossings', end_quality=quality)
            def crossing(i,j):
                fraction = (cut-heights[i])/(heights[j]-heights[i])
                return float(self.distance[i]+fraction*(self.distance[j]-self.distance[i]))
            head = crossing(left-1,left)
            tail = float(self.distance[-1]-crossing(right,right+1))
            locations=[]; coordinates=[]; weights=[]
            for j,length in enumerate(self.lengths):
                for u in (np.arange(16)+.5)/16:
                    locations.append((1-u)*heights[j]+u*heights[j+1])
                    coordinates.append(j+u); weights.append(length/16)
            levels=np.array(locations)
            w=np.maximum(0.,levels-levels.max()+np.mean(self.lengths)/8)*weights
            crest=float(w@coordinates/w.sum())
            # Cover the actual upper span plus neighbours. This is geometry,
            # not the contact support span; it cannot prove unloading.
            arc_window=(max(2,left-1),min(len(q)-1,right+1))
            section=local_section(q,self.lengths,*arc_window)
            return dict(valid=True, head_hanging_m=head, tail_hanging_m=tail,
                transfer_coordinate_m=.5*(head-tail), above_cut_length_m=float(self.distance[-1]-head-tail),
                cut_height_head_frame_m=cut, cut_depth_m=self.cut_depth_m, crest_joint=crest,
                actual_arc_window=list(arc_window), down_head=down.tolist(), end_quality=quality,
                plane_rms_m=section['fit_rms'], observes_pipe_contact=False, observes_world_position=False)
        except (ValueError,np.linalg.LinAlgError) as exc:
            return dict(valid=False,reason=str(exc))


@dataclass(frozen=True)
class ContinuousFeedingOptions:
    first_joint: int = 8
    last_joint: int = 13
    amplitude_deg: float = 60.
    requested_transfer_m: float = -.0645
    preparation_s: float = 6.
    length_gain_s: float = .8
    speed_damping: float = .7
    maximum_position_rate_joints_s: float = .1
    velocity_filter_s: float = .4
    length_tolerance_m: float = .004
    speed_tolerance_m_s: float = .001
    persistence_s: float = .4
    review_s: float = 2.
    maximum_feed_s: float = 60.
    target_rate_deg_s: float = 45.
    exit_angle_deg: float = 8.
    bearing_bias_limit: float = .18
    # Retained for archived option dictionaries; current bias is periodic,
    # not the old integrated shoulder-height controller.
    bearing_bias_rate_s: float = .125
    bearing_gain: float = .4
    maximum_untrusted_s: float = 3.
    trailing_buffer_joints: int = 2
    release_persistence_s: float = .4
    tracking_tolerance_deg: float = 5.
    tracked_turn_soft_limit_deg: float = 192.
    maximum_tracking_wait_s: float = 10.
    travelling_bias_amplitude: float = .12
    fractional_middle_boost_deg: float = 15.
    middle_boost_activation_joints: float = 1.5
    middle_boost_ramp_joints: float = 1.
    adjacent_fold_angle_deg: float = 5.
    adjacent_fold_distance_joints: int = 4
    fold_retention_distance_joints: float = 1.
    release_down_alignment: float = .97
    minimum_passive_reference_links: int = 3
    automatic_return: bool = True  # Research round trips; manual entries set False.
    acquisition_stability_s: float = .6
    acquisition_angle_span_deg: float = 3.
    acquisition_length_span_m: float = .002


class ContinuousFeeding:
    def __init__(self, lengths, options=ContinuousFeedingOptions()):
        self.lengths = np.asarray(lengths, float)
        self.n = len(self.lengths)-1
        self.options = options
        self.reference_direction = 1 if options.requested_transfer_m > 0 else -1
        self.helper = FeedingArcGeometry(self.lengths)
        self.length_observer = ManualFeedingObserver(self.lengths, (options.first_joint, options.last_joint))
        self.origin = (options.first_joint, options.last_joint)
        # Each transition advances two alternating hinges. The last valid
        # pair gives a coarse profile range, not a guarantee of support.
        self.minimum_bin = math.ceil((2-options.first_joint)/2)
        self.maximum_bin = math.floor((self.n-1-options.last_joint-2)/2)
        self.position_bounds = (2*self.minimum_bin, 2*(self.maximum_bin+1))
        self.position = self.bias = self.phase = 0.
        # Equal relative settling speed for one-link and larger actions.
        # This is a motor-proxy gate, not a claim of true length stability.
        self.goal_speed_tolerance_m_s = options.speed_tolerance_m_s*max(
            1., abs(options.requested_transfer_m)/float(np.mean(self.lengths)))
        self.enabled = self.helper.mask(self.origin)
        self.window = self.origin
        self.targets = self.helper.template(self.origin, 0., options.amplitude_deg)
        self.stage = 'prepare'
        self.stage_since = self.ready_since = None
        self.last_time = self.last_feedback = self.last_length_time = None
        self.previous_angles = self.previous_coordinate = None
        self.velocity_m_s = 0.
        self.observation = self.baseline = self.goal_m = None
        self.bearing_error_m = None
        self.untrusted_since = None
        self.release_below_peak_m = None
        self.release_since = np.full(self.n, np.nan)
        self.fold_capture_position = np.full(self.n, np.nan)
        self.tracking_blocked_since = None
        self.tracking_error_deg = self.actual_enabled_turn_deg = 0.
        self.completed = False
        self.fault = None
        self.events = []
        self.acquisition_samples = []
        self.acquisition_reason = None
        self.acquisition_metrics = {}
        self.forced_step = None

    def request_step(self, distance_m, control_time_s):
        """Start one relative action without resetting the travelling profile.

        The start datum is measured after taking over/straightening neighbours.
        Existing targets, mask, profile position and release history survive.
        """
        if not math.isfinite(distance_m) or distance_m == 0:
            raise ValueError('A finite nonzero feeding distance is required')
        self.forced_step = None
        self.options = replace(self.options, requested_transfer_m=distance_m)
        self.goal_speed_tolerance_m_s = self.options.speed_tolerance_m_s*max(
            1., abs(distance_m)/float(np.mean(self.lengths)))
        self.stage, self.stage_since = 'acquire', control_time_s
        self.completed, self.fault = False, None
        self.baseline = self.goal_m = None
        self.ready_since = self.untrusted_since = self.tracking_blocked_since = None
        self.acquisition_samples = []
        self.acquisition_metrics = {}
        self.acquisition_reason = 'waiting_for_takeover_and_stability'
        self.velocity_m_s = 0.
        self.last_length_time = None
        self.events.append(dict(kind='manual_relative_step', time_s=control_time_s,
            requested_transfer_m=distance_m, position_joints=self.position))

    def start_forced_step(self, angles_deg, step_joints, control_time_s):
        """One manual profile increment, independent of length/tracking gates.

        Use the active task's intended direction, not the signed length error.
        Hold the existing torque mask; preload any newly needed joints. The
        target is sent gradually by advance(), without waiting for tracking.
        """
        angles = np.asarray(angles_deg, float)
        if angles.shape != (self.n,) or not np.isfinite(angles).all():
            raise ValueError('Finite complete encoder sample required')
        if not math.isfinite(step_joints) or step_joints <= 0:
            raise ValueError('Forced step must be a positive joint-coordinate increment')
        direction = 1 if self.options.requested_transfer_m > 0 else -1
        if self.stage in ('feed_return', 'review_return'):
            direction *= -1
        previous = self.position
        position = float(np.clip(previous+direction*step_joints, *self.position_bounds))
        if abs(position-previous) < 1e-12:
            self.events.append(dict(kind='forced_step_boundary', time_s=control_time_s,
                position_joints=previous, direction=direction))
            return False
        bias = -self.options.travelling_bias_amplitude*math.sin(math.pi*position)
        desired, focus, _, _ = self.profile(position, bias)
        # Only the new profile/buffer is needed for this explicit command;
        # do not require a passive direction or a measured length baseline.
        required = self.required_with_buffer(focus) | self.helper.mask(focus)
        entering = required & ~self.enabled
        self.targets[entering] = np.radians(angles)[entering]
        self.enabled |= required
        active = np.flatnonzero(self.enabled)+1
        self.window = (int(active[0]), int(active[-1]))
        self.position, self.bias = position, bias
        previous_fault = self.fault
        self.completed, self.fault = False, None
        if self.stage in ('prepare', 'complete'):
            self.stage = 'acquire'
            self.baseline = self.goal_m = None
        self.acquisition_samples = []
        self.acquisition_metrics = {}
        self.ready_since = self.untrusted_since = self.tracking_blocked_since = None
        self.last_length_time = None
        self.velocity_m_s = 0.
        self.stage_since = control_time_s
        self.last_time = control_time_s-.02
        self.forced_step = dict(target_rad=desired.tolist(), from_position_joints=previous,
            to_position_joints=position, direction=direction)
        self.events.append(dict(kind='forced_step_requested', time_s=control_time_s,
            from_position_joints=previous, to_position_joints=position,
            direction=direction, previous_fault=previous_fault,
            entering=(np.flatnonzero(entering)+1).tolist(), measured_distance_claim=False))
        return True

    def acquire_start(self, angles, feedback_time, control_time, fresh):
        """Settle the actual takeover before defining zero transfer.

        No commanded profile travel and no release while collecting the datum.
        Geometric readings during this stage are diagnostics, not progress.
        """
        desired, focus, _, _ = self.profile(self.position, self.bias)
        required = self.required_mask(focus, angles, self.position)
        entering = required & ~self.enabled
        if np.any(entering):
            self.targets[entering] = np.radians(angles)[entering]
            self.enabled |= required
            active = np.flatnonzero(self.enabled)+1
            self.window = (int(active[0]), int(active[-1]))
            self.acquisition_samples = []
            self.acquisition_metrics = {}
            self.events.append(dict(kind='acquisition_takeover', time_s=control_time,
                entering=(np.flatnonzero(entering)+1).tolist(), position_joints=self.position))
        opt = self.options
        self.tracking_error_deg = float(np.max(abs(np.degrees(desired[self.enabled])-angles[self.enabled])))
        self.acquisition_reason = 'waiting_for_stable_shape_and_length'
        if fresh:
            if entering.any() or not self.observation['valid'] or self.tracking_error_deg > opt.tracking_tolerance_deg:
                self.acquisition_samples = []
                self.acquisition_metrics = {}
                self.acquisition_reason = ('waiting_for_takeover_tracking' if entering.any() or self.observation['valid']
                                           else self.observation.get('reason', 'untrusted_direction'))
            else:
                coordinate = self.observation['transfer_coordinate_m']
                quality = self.observation.get('end_quality', {})
                # Changing the reference segments is a new measurement frame.
                # Ignore time-varying quality metrics; compare material spans.
                reference = (quality.get('source'), tuple(
                    (k, str(v.get('selected', {}).get('links')) if isinstance(v.get('selected'), dict) else str(v.get('selected')))
                    for k,v in quality.get('ends', {}).items()))
                if self.acquisition_samples and (feedback_time-self.acquisition_samples[-1][0]>.6 or
                                                  reference != self.acquisition_samples[-1][3]):
                    self.acquisition_samples = []
                self.acquisition_samples.append((feedback_time, angles.copy(), coordinate, reference))
                while len(self.acquisition_samples)>2 and feedback_time-self.acquisition_samples[1][0]>=opt.acquisition_stability_s:
                    self.acquisition_samples.pop(0)
                samples = self.acquisition_samples
                self.acquisition_metrics = dict(samples=len(samples),
                    stable_span_s=feedback_time-samples[0][0],
                    angle_span_deg=float(np.ptp(np.array([s[1] for s in samples]), axis=0).max()),
                    length_span_mm=float(1000*np.ptp([s[2] for s in samples])),
                    tracking_error_deg=self.tracking_error_deg)
                if (len(samples)>=3 and feedback_time-samples[0][0]>=opt.acquisition_stability_s-1e-8 and
                    np.ptp(np.array([s[1] for s in samples]), axis=0).max()<=opt.acquisition_angle_span_deg and
                    np.ptp([s[2] for s in samples])<=opt.acquisition_length_span_m):
                    self.baseline = coordinate
                    self.goal_m = coordinate+opt.requested_transfer_m
                    self.velocity_m_s = 0.
                    self.previous_coordinate, self.last_length_time = coordinate, feedback_time
                    points = forward(np.radians(angles), self.lengths)
                    heights = -points @ np.array(self.observation['down_head'])
                    first, last = focus
                    self.release_below_peak_m = float(heights.max()-heights[first:last+2].min()+.5*np.mean(self.lengths))
                    self.stage, self.stage_since = 'feed_out', control_time
                    self.acquisition_reason = None
                    self.events.append(dict(kind='start_feeding', time_s=control_time,
                        baseline_m=coordinate, goal_transfer_m=opt.requested_transfer_m,
                        position_joints=self.position, window=list(self.window),
                        datum='after_takeover_stable_encoder_geometry'))
        if self.stage=='acquire' and control_time-self.stage_since>opt.maximum_tracking_wait_s:
            self.fault = 'takeover shape/length did not settle; no transfer baseline recorded'
        return desired

    def profile(self, position, bias):
        """Interpolate the travelling profile; totals of each axis stay fixed."""
        cell = int(np.clip(math.floor(position/2), self.minimum_bin, self.maximum_bin))
        fraction = float(np.clip(position/2-cell, 0., 1.))
        old = tuple(j+2*cell for j in self.origin)
        new = tuple(j+2 for j in old)
        left = self.helper.template(old, 0., self.options.amplitude_deg, bias)
        right = self.helper.template(new, 0., self.options.amplitude_deg, bias)
        curved = np.flatnonzero(abs(left)+abs(right)>1e-10)+1
        focus = (int(curved[0]), int(curved[-1]))
        target = (1-fraction)*left+fraction*right
        # Four-axis interpolation can retain extra material in the top arc.
        # Coordinate both middle bends while preserving the pi turn budget.
        offset = float(np.clip((abs(position)-self.options.middle_boost_activation_joints)/
                               self.options.middle_boost_ramp_joints, 0., 1.))
        allocation = offset*offset*(3.-2.*offset)
        boost = math.radians(self.options.fractional_middle_boost_deg)*allocation*math.sin(math.pi*fraction)**2
        family = curved-1
        if len(family)==4:
            target[family[[0,-1]]] -= boost
            target[family[1:3]] += boost
        return target, focus, cell, fraction

    def required_with_buffer(self, focus):
        direction = 1 if self.options.requested_transfer_m > 0 else -1
        if self.stage in ('feed_return', 'review_return'):
            direction *= -1
        first, last = focus
        if direction > 0:
            first = max(2, first-self.options.trailing_buffer_joints)
        else:
            last = min(self.n-1, last+self.options.trailing_buffer_joints)
        return self.helper.mask((first, last))

    def required_mask(self, focus, angles, position):
        required = self.required_with_buffer(focus)
        actual_arc = self.observation.get('actual_arc_window')
        if actual_arc is not None:
            required |= self.helper.mask(tuple(actual_arc))
        active = np.flatnonzero(required)+1
        radius = self.options.adjacent_fold_distance_joints
        joints = np.arange(1,self.n+1)
        folds = joints[(abs(angles)>=self.options.adjacent_fold_angle_deg) &
                       (joints>=active[0]-radius) & (joints<=active[-1]+radius)]
        self.fold_capture_position[folds-1] = position
        # A straightened fold often returns under gravity immediately
        # after torque-off. Keep its temporary straightening control
        # until the travelling arc leaves, rather than cycling masks
        # at a stationary length goal. This is material distance,
        # never pipe contact or a world unloading observation.
        retained = joints[np.isfinite(self.fold_capture_position) &
            (abs(position-self.fold_capture_position)<self.options.fold_retention_distance_joints)]
        folds = np.union1d(folds, retained)
        if len(folds):
            first=max(2,min(int(active[0]),int(folds.min())))
            last=min(self.n-1,max(int(active[-1]),int(folds.max())))
            required |= self.helper.mask((first,last))
        # Keep one distal torque-off reference observable. The end
        # being lengthened on the outbound leg has the larger reserve;
        # use the same end through return to avoid switching the
        # observation geometry. Reserved folds remain passive and are
        # still fully included in external whole-body shape audits.
        reserve = self.options.minimum_passive_reference_links
        if self.reference_direction > 0:
            required[:reserve] = False
        else:
            required[self.n-reserve:] = False
        return required

    def release_ready(self, angles, required, feedback_time, fresh):
        """Encoder geometry below the bearing arc; not a contact-force gate."""
        leaving = self.enabled & ~required
        if not self.observation['valid'] or self.release_below_peak_m is None:
            self.release_since[:] = np.nan
            return np.zeros(self.n, bool)
        points = forward(np.radians(angles), self.lengths)
        heights = -points @ np.array(self.observation['down_head'])
        level = float(heights.max()-self.release_below_peak_m)
        below = np.array([heights[max(0,j):min(len(heights),j+3)].max()<level for j in range(self.n)])
        directions = np.diff(points,axis=0)
        directions /= np.maximum(np.linalg.norm(directions,axis=1)[:,None],1e-12)
        indices = np.arange(self.n)
        active = np.flatnonzero(required)
        towards_tip = directions[1:].copy()
        towards_tip[indices<active[0]] *= -1
        aligned = towards_tip @ np.array(self.observation['down_head']) >= self.options.release_down_alignment
        eligible = leaving & (abs(angles)<=self.options.exit_angle_deg) & below & aligned
        if fresh:
            self.release_since[~eligible] = np.nan
            starting = eligible & np.isnan(self.release_since)
            self.release_since[starting] = feedback_time
        return eligible & (feedback_time-self.release_since>=self.options.release_persistence_s-1e-8)

    def advance(self, angles_deg, feedback_time_s, control_time_s):
        angles = np.asarray(angles_deg, float)
        if angles.shape != (self.n,) or not np.isfinite(angles).all():
            raise ValueError('Finite complete encoder sample required')
        age = control_time_s-feedback_time_s
        if not math.isfinite(age) or not -.00000001 <= age <= .5:
            raise ValueError('Current, nonfuture feedback required')
        if self.last_time is not None and control_time_s <= self.last_time:
            raise ValueError('Increasing control time required')
        if self.last_feedback is not None and feedback_time_s < self.last_feedback:
            raise ValueError('Feedback time cannot run backwards')
        fresh = self.last_feedback is None or feedback_time_s > self.last_feedback
        if not fresh and not np.array_equal(angles, self.previous_angles):
            raise ValueError('Changed angles with same feedback time')
        dt = .02 if self.last_time is None else control_time_s-self.last_time
        self.last_time = control_time_s
        opt = self.options
        if fresh:
            self.last_feedback = feedback_time_s
            self.previous_angles = angles.copy()
            self.observation = self.length_observer.observe(angles, self.window, feedback_time_s)
            if 'crest_joint' not in self.observation:
                self.observation['crest_joint'] = None
            if self.observation['valid']:
                self.untrusted_since = None
                coordinate = self.observation['transfer_coordinate_m']
                if self.last_length_time is not None:
                    sample_dt = feedback_time_s-self.last_length_time
                    raw_velocity = (coordinate-self.previous_coordinate)/sample_dt
                    amount = 1-math.exp(-sample_dt/opt.velocity_filter_s)
                    self.velocity_m_s += amount*(raw_velocity-self.velocity_m_s)
                self.previous_coordinate = coordinate
                self.last_length_time = feedback_time_s
                # A periodic intrinsic allocation coordinates the middle
                # hinges without an accumulated orientation bias at rest.
            elif self.stage != 'prepare':
                self.ready_since = None
                if self.untrusted_since is None:
                    self.untrusted_since = control_time_s

        if self.forced_step is not None:
            desired = np.array(self.forced_step['target_rad'])
            limit = math.radians(opt.target_rate_deg_s)*min(dt, .02)
            self.targets[self.enabled] += np.clip(desired-self.targets, -limit, limit)[self.enabled]
            self.targets[~self.enabled] = np.radians(angles)[~self.enabled]
            if np.max(abs(desired[self.enabled]-self.targets[self.enabled])) < 1e-10:
                self.events.append(dict(kind='forced_step_targets_sent', time_s=control_time_s,
                    position_joints=self.position, actual_motion_verified=False))
                self.forced_step = None
                self.stage_since = control_time_s
                self.last_length_time = None
                self.velocity_m_s = 0.
            return self.targets.copy(), self.enabled.copy()

        desired = self.targets.copy()
        if self.stage == 'prepare':
            desired = self.helper.template(self.origin, 0., opt.amplitude_deg)
            if control_time_s >= opt.preparation_s-1e-8 and fresh:
                if not self.observation['valid']:
                    # A fresh sample at the nominal preparation boundary may
                    # still be gathering stable passive geometry. Use the same
                    # bounded observation wait as during feeding, not one frame.
                    if control_time_s >= opt.preparation_s+opt.maximum_untrusted_s:
                        self.fault = 'initial differential length unavailable'
                else:
                    points = forward(np.radians(angles), self.lengths)
                    heights = -points @ np.array(self.observation['down_head'])
                    first, last = self.origin
                    arc_depth = float(heights.max()-heights[first:last+2].min())
                    self.release_below_peak_m = arc_depth+.5*float(np.mean(self.lengths))
                    self.stage, self.stage_since = 'acquire', control_time_s
                    desired = self.acquire_start(angles, feedback_time_s, control_time_s, fresh)
        elif self.stage == 'acquire':
            desired = self.acquire_start(angles, feedback_time_s, control_time_s, fresh)
        elif self.stage in ('feed_out', 'review_out', 'feed_return', 'review_return'):
            # Pausing the material coordinate must not freeze a newly enabled
            # joint at its preload angle. Finish the already commanded shape
            # while waiting for the length/gravity proxy to become usable.
            desired, _, _, _ = self.profile(self.position, self.bias)
            if self.observation['valid']:
                error = self.goal_m-self.observation['transfer_coordinate_m']
                speed = opt.length_gain_s*error-opt.speed_damping*self.velocity_m_s
                rate = np.clip(speed/np.mean(self.lengths),
                               -opt.maximum_position_rate_joints_s, opt.maximum_position_rate_joints_s)
                proposed_position = float(np.clip(self.position+rate*dt, *self.position_bounds))
                current, _, _, _ = self.profile(self.position, self.bias)
                self.tracking_error_deg = float(np.max(abs(np.degrees(current[self.enabled])-angles[self.enabled])))
                self.actual_enabled_turn_deg = float(abs(angles[self.enabled]).sum())
                blocked = (self.tracking_error_deg>opt.tracking_tolerance_deg or
                           self.actual_enabled_turn_deg>opt.tracked_turn_soft_limit_deg)
                if blocked:
                    proposed_position = self.position
                    if self.tracking_blocked_since is None:
                        self.tracking_blocked_since = control_time_s
                else:
                    self.tracking_blocked_since = None
                self.bias = -opt.travelling_bias_amplitude*math.sin(math.pi*proposed_position)
                proposal, focus, _, _ = self.profile(proposed_position, self.bias)
                required = self.required_mask(focus, angles, proposed_position)
                leaving = self.enabled & ~required
                release = self.release_ready(angles, required, feedback_time_s, fresh)
                next_mask = required | (leaving & ~release)
                entering = next_mask & ~self.enabled
                # A completed release ends this temporary fold capture. Keep
                # no dormant record that can re-enable it when position moves
                # back across the retention boundary. Fresh nearby bends and
                # the current arc can still request a new takeover normally.
                self.fold_capture_position[release] = np.nan
                self.release_since[release] = np.nan
                if np.any(entering) or np.any(release):
                    self.targets[entering] = np.radians(angles)[entering]
                    self.events.append(dict(kind='mask_handover', time_s=control_time_s,
                        entering=(np.flatnonzero(entering)+1).tolist(),
                        released=(np.flatnonzero(release)+1).tolist(),
                        exit_angles_deg=angles[release].tolist(), enabled_count=int(next_mask.sum()),
                        release_means='persistent_angle_and_below_arc_geometry_not_measured_unloading'))
                self.position = proposed_position
                self.enabled = next_mask
                active = np.flatnonzero(self.enabled)+1
                self.window = (int(active[0]), int(active[-1]))
                desired = proposal
                if fresh and self.stage in ('feed_out', 'feed_return'):
                    reached = abs(error)<=opt.length_tolerance_m and abs(self.velocity_m_s)<=self.goal_speed_tolerance_m_s
                    self.ready_since = (feedback_time_s if self.ready_since is None else self.ready_since) if reached else None
                    if self.ready_since is not None and feedback_time_s-self.ready_since >= opt.persistence_s-1e-8:
                        self.stage = 'review_out' if self.stage == 'feed_out' else 'review_return'
                        self.stage_since = control_time_s
                        if self.stage=='review_out' and not opt.automatic_return:
                            self.stage, self.completed = 'complete', True
                        self.events.append(dict(kind='length_goal_observed', time_s=control_time_s,
                            goal_delta_m=self.goal_m-self.baseline, observed_delta_m=coordinate-self.baseline,
                            velocity_m_s=self.velocity_m_s, position_joints=self.position))
            if self.tracking_blocked_since is not None and control_time_s-self.tracking_blocked_since>opt.maximum_tracking_wait_s:
                self.fault = 'measured shape tracking did not recover within bounded handover'
            elif self.untrusted_since is not None and control_time_s-self.untrusted_since > opt.maximum_untrusted_s:
                self.fault = 'length observation did not recover during bounded shape completion'
            elif self.stage in ('feed_out', 'feed_return') and control_time_s-self.stage_since > opt.maximum_feed_s:
                self.fault = 'length goal not reached within bounded feeding'
            elif self.stage in ('review_out', 'review_return') and control_time_s-self.stage_since >= opt.review_s:
                if self.stage == 'review_out':
                    self.stage, self.stage_since, self.ready_since = 'feed_return', control_time_s, None
                    self.goal_m = self.baseline
                    self.events.append(dict(kind='start_return', time_s=control_time_s))
                else:
                    self.stage, self.completed = 'complete', True
        delta = np.clip(desired-self.targets, -math.radians(opt.target_rate_deg_s)*dt,
                        math.radians(opt.target_rate_deg_s)*dt)
        self.targets[self.enabled] += delta[self.enabled]
        self.targets[~self.enabled] = np.radians(angles)[~self.enabled]
        return self.targets.copy(), self.enabled.copy()

    def snapshot(self):
        return dict(stage=self.stage, route='continuous_feeding', window=list(self.window),
            forced_step=None if self.forced_step is None else dict(self.forced_step),
            acquisition_reason=self.acquisition_reason,
            acquisition_samples=len(self.acquisition_samples),
            acquisition_metrics=self.acquisition_metrics,
            phase_rad=0., position_joints=self.position, bearing_bias=self.bias,
            bearing_error_m=self.bearing_error_m, goal_m=self.goal_m, baseline_m=self.baseline,
            goal_delta_m=None if self.goal_m is None else self.goal_m-self.baseline,
            observation=self.observation, velocity_m_s=self.velocity_m_s, enabled=self.enabled.tolist(),
            goal_speed_tolerance_m_s=self.goal_speed_tolerance_m_s,
            untrusted_since_s=self.untrusted_since,
            release_below_peak_m=self.release_below_peak_m,
            tracking_blocked_since_s=self.tracking_blocked_since,
            tracking_error_deg=self.tracking_error_deg, actual_enabled_turn_deg=self.actual_enabled_turn_deg,
            retained_fold_joints=(np.flatnonzero(np.isfinite(self.fold_capture_position) &
                (abs(self.position-self.fold_capture_position)<self.options.fold_retention_distance_joints))+1).tolist(),
            completed=self.completed, fault=self.fault, options=asdict(self.options),
            profile_position_bounds_joints=list(self.position_bounds), compliance=False,
            passive_reference_end='head' if self.reference_direction>0 else 'tail',
            completion_means='encoder_length_and_speed_goal_not_world_truth_acceptance',
            inputs=['angles_deg', 'feedback_time_s', 'control_time_s'])


def choose_feeding_arc(angles_deg, lengths, enabled_mask, focus=None):
    """Acquire the established measured bend before releasing straight buffers."""
    mask=np.asarray(enabled_mask,bool)
    active=np.flatnonzero(mask)+1
    if not len(active):raise ValueError('模式4没有使能圆弧')
    if focus is None:focus=(int(active[0]),int(active[-1]))
    selection=feature_arc_candidates(angles_deg,lengths,focus=focus)
    if selection['chosen'] is None:raise ValueError('模式4实测形状中没有可重建的开放圆弧')
    selection['best']['end_source']='feeding_measured_arc_before_passive_release'
    selection['requires_down_before_switch']=False
    return selection


def choose_posture_window(angles_deg,lengths,enabled_mask=None,diagnostics=None):
    """Find measured arcs first; optional passive direction only ranks them."""
    selection=feature_arc_candidates(angles_deg,lengths)
    if diagnostics is not None:diagnostics.update(selection['diagnostics'])
    if selection['chosen'] is None:
        raise ValueError('没有可重建的开放圆弧；'+str(selection['diagnostics']))
    window=tuple(selection['chosen'])
    if enabled_mask is not None:
        active=np.flatnonzero(enabled_mask)+1
        if len(active):window=(int(active[0]),int(active[-1]))
    helper=WholeFeedingObserver(lengths,window,replace(FeedingDirectionOptions(),minimum_samples=1,stability_s=0.))
    down,quality=helper.down_direction(forward(np.radians(angles_deg),lengths),window,angles_deg,0.)
    if down is not None:
        selection=feature_arc_candidates(angles_deg,lengths,down=down)
    selection['best']['end_source']=quality['source']
    selection['requires_down_before_switch']=False
    if diagnostics is not None:diagnostics.update(selection['diagnostics'])
    return selection
# END ACTIVE_FEEDING_CORE

# BEGIN ARC_HELIX_CORE
@dataclass(frozen=True)
class HelixCurveSegment:
    length: float      # body arc length, m
    kappa: float = 0.  # curvature, 1/m
    tau: float = 0.    # torsion, 1/m (sign = handedness)
    psi: float = 0.    # torsion-angle offset added at this segment's start, rad

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.length, self.kappa, self.tau, self.psi)):
            raise ValueError('nonfinite segment')
        if self.length < 0 or self.kappa < 0:
            raise ValueError('length and curvature must be non-negative')


def line(length):
    return HelixCurveSegment(length)


def arc(length, kappa, psi=0.):
    return HelixCurveSegment(length, kappa, 0., psi)


def helix(length, radius, pitch, handed=1, psi=0.):
    """Circular helix around a straight axis; pitch is axial advance per turn.

    `handed` is the sign of the torsion term in this joint convention. The
    physical handedness also depends on `CurveGait.direction`: with
    direction=+1, handed=+1 gives a LEFT-handed helix in the MuJoCo model
    (checked by forward kinematics); use `physical_handedness` to convert.
    """
    if radius <= 0 or pitch < 0 or handed not in (-1, 1):
        raise ValueError('invalid helix')
    c = pitch / (2 * math.pi)
    denom = radius * radius + c * c
    return HelixCurveSegment(length, radius / denom, handed * c / denom, psi)


class CurveGait:
    def __init__(self, joints, link_length=.0645, direction=1):
        if joints < 3 or link_length <= 0 or direction not in (-1, 1):
            raise ValueError('invalid robot')
        self.joints, self.l, self.direction = joints, link_length, direction
        self.body_length = (joints + 1) * link_length

    def targets(self, segments, roll=0., offset=0.):
        """Joint angles (rad). `offset` shifts the body along the curve (shift control):
        the head tip sits at curve coordinate `offset`."""
        bounds = np.r_[0., np.cumsum([s.length for s in segments])]
        # psi at each segment start: accumulate torsion and explicit offsets.
        psi0, starts = roll, []
        for s in segments:
            psi0 += s.psi
            starts.append(psi0)
            psi0 += s.tau * s.length
        out = np.zeros(self.joints)
        for j in range(1, self.joints + 1):
            lo, hi = offset + (j - 1) * self.l, offset + (j + 1) * self.l
            cos_part = sin_part = 0.
            for k, s in enumerate(segments):
                a, b = max(lo, bounds[k]), min(hi, bounds[k + 1])
                if b <= a or s.kappa == 0:
                    continue
                span = b - a
                mid = starts[k] + s.tau * ((a + b) / 2 - bounds[k])
                weight = s.kappa * span * np.sinc(s.tau * span / (2 * math.pi))
                cos_part += weight * math.cos(mid)
                sin_part += weight * math.sin(mid)
            out[j - 1] = cos_part if j % 2 else self.direction * sin_part
        return out

    def window_arc(self, first, last, amplitude_rad, extra_links=0.):
        """ARC as one arc segment covering joints first..last (plus optional extra links)."""
        start = (first - 1) * self.l - extra_links * self.l
        stop = (last + 1) * self.l + extra_links * self.l
        kappa = amplitude_rad / (2 * self.l)
        return [line(max(start, 0.)), arc(stop - max(start, 0.), kappa), line(max(self.body_length - stop, 0.))]


class ArcHelixShift:
    """Convert an ARC to a helix by moving the joins of a single body curve."""

    front_extension_rate = 1.75

    def __init__(self, gait, window, amplitude, radius, pitch, handed):
        self.gait = gait
        first, last = window
        self.head_length = (first-1)*gait.l
        self.tail_length = gait.body_length-(last+1)*gait.l
        self.arc_length = gait.body_length-self.head_length-self.tail_length
        self.arc_curvature = amplitude/(2.*gait.l)
        self.helix = helix(0., radius, pitch, handed)

    def targets(self, head_shift, tail_shift, arc_length, anchor_phase):
        """Keep the bending phase at the original ARC midpoint continuous."""
        head_line = self.head_length-head_shift
        tail_line = self.tail_length-tail_shift
        converted = .5*(self.arc_length-arc_length)
        head_helix = replace(self.helix, length=head_shift+converted)
        tail_helix = replace(self.helix, length=tail_shift+converted)
        phase = anchor_phase-self.helix.tau*head_helix.length
        segments = (line(head_line), head_helix, arc(arc_length, self.arc_curvature),
                    tail_helix, line(tail_line))
        return self.gait.targets(segments, phase), phase

    def shifted_window(self, shift, arc_length, anchor_phase, leading_extension=0., leading_helix=None):
        """Sample a fixed-length interval in coordinates anchored to the initial ARC.

        Moving the sample relative to the joins gives the same relative
        coordinates as moving the joins in a frame attached to the head.
        """
        head = -shift
        arc_start = self.head_length
        arc_end = arc_start+arc_length
        rear_join = arc_start+self.arc_length
        # Grow the front helix toward the head from the original ARC shoulder.
        front_join = arc_start-leading_extension-self.front_extension_rate*shift
        origin = min(-self.gait.body_length, front_join, head-self.gait.l)
        front_helix = replace(self.helix if leading_helix is None else leading_helix, length=arc_start-front_join)
        rear_helix = replace(self.helix, length=rear_join-arc_end)
        segments = (line(front_join-origin), front_helix, arc(arc_length, self.arc_curvature),
                    rear_helix, line(self.tail_length))
        # Compensate from a fixed curve coordinate. Adding upstream helix
        # does not change the phase at the old ARC midpoint.
        midpoint = arc_start+.5*self.arc_length
        rear_overlap = max(0., midpoint-arc_end)
        if leading_helix is None:
            phase = anchor_phase-self.helix.tau*(front_helix.length+rear_overlap)
        else:
            phase = anchor_phase-(front_helix.tau*front_helix.length+self.helix.tau*rear_overlap)
        q = self.gait.targets(segments, phase, offset=head-origin)
        head_phase = phase+front_helix.tau*float(np.clip(head-front_join, 0., front_helix.length))
        centers = head+np.arange(1, self.gait.joints+1)*self.gait.l
        overlap = np.minimum(centers+self.gait.l, rear_join)-np.maximum(centers-self.gait.l, front_join)
        stiffness = np.clip(overlap/self.gait.l-1., 0., 1.)
        return q, head_phase, stiffness


class MeasuredGravityShift:
    """Translate one curve reconstructed from the measured hanging pose.

    The two alternating hinge families each measure a curvature integral over
    two links. A piecewise-constant correction to a helical reference matches
    those measured integrals. Upstream of the recorded body the correction is
    zero, so the new material coordinates enter the helix continuously.
    """

    def __init__(self, gait, radius_m, phase_rad, angles_rad, support_count, *, upstream_m=0.):
        self.link = gait.l
        n = gait.gait.joints
        angles = np.asarray(angles_rad, float)
        if angles.shape != (n,) or not np.isfinite(angles).all() or not 1 <= support_count < n:
            raise ValueError('Expected one valid hanging pose and a nonempty support and tail')
        if not math.isfinite(upstream_m) or upstream_m < 0:
            raise ValueError('Upstream extension must be finite and non-negative')
        self.shift_limit_m = (n-support_count)*self.link+upstream_m
        self.support_end_m = (support_count+1)*self.link
        self.joint_coordinates = (np.arange(n)+1)*self.link
        self.phase_rad = phase_rad
        self.direction = gait.gait.direction
        segment = helix(1., radius_m, gait.o.pitch, gait.o.handed)
        self.torsion = segment.tau
        self.angle_scale = segment.kappa*2*self.link*np.sinc(segment.tau*self.link/math.pi)
        residual = angles-self.helical_targets(0.)
        self.corrections = []
        for axis in (0, 1):
            coords = self.joint_coordinates[axis::2]
            boundaries = np.r_[coords-self.link, coords[-1]+self.link]
            self.corrections.append((boundaries, np.r_[0., np.cumsum(residual[axis::2])]))

    def helical_targets(self, shift_m, roll_rad=0.):
        phase = self.phase_rad+self.torsion*(self.joint_coordinates-shift_m)+roll_rad
        angles = np.cos(phase)
        angles[1::2] = self.direction*np.sin(phase[1::2])
        return self.angle_scale*angles

    def targets(self, shift_m, front_roll_rad=0.):
        if not math.isfinite(shift_m) or not 0 <= shift_m <= self.shift_limit_m:
            raise ValueError('Shift lies outside the available measured curve')
        q = self.helical_targets(shift_m)
        for axis, (boundaries, integral) in enumerate(self.corrections):
            centers = self.joint_coordinates[axis::2]-shift_m
            q[axis::2] += np.interp(centers+self.link, boundaries, integral)
            q[axis::2] -= np.interp(centers-self.link, boundaries, integral)
        if front_roll_rad:
            # Tie the transition to the curve coordinate. The hanging tail
            # follows the measured curve until it reaches the supported part.
            distance = self.support_end_m-(self.joint_coordinates-shift_m)
            blend = np.clip(distance/(2*self.link), 0., 1.)
            weight = blend**2*(3-2*blend)
            q += weight*(self.helical_targets(shift_m, front_roll_rad)-self.helical_targets(shift_m))
        return q


def ease(value):
    value = float(np.clip(value, 0., 1.))
    return value*value*(3.-2.*value)


@dataclass(frozen=True)
class TransferOptions:
    first_joint: int = 8
    last_joint: int = 13
    amplitude_deg: float = 60.
    loose_amplitude_deg: float = 45.
    hold_phase_rad: float = .7
    preparation_s: float = 6.
    transfer_s: float = 5.
    review_s: float = 3.
    rolling_speed_rad_s: float = .6
    exit_angle_deg: float = 12.
    target_rate_deg_s: float = 45.
    desired_direction: int = -1
    desired_shift_joints: float = 1.
    drift_persistence_s: float = .6
    maximum_transfer_wait_s: float = 20.
    rocking_amplitude_rad: float = .12
    rocking_period_s: float = 2.
    feedback_align: bool = False
    alignment_tolerance_joints: float = .12
    alignment_persistence_s: float = .4
    alignment_gain_s: float = .45
    alignment_bias_limit: float = .4
    maximum_alignment_s: float = 12.

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Finite action parameters required')
        if self.desired_direction not in (-1, 1) or not 0 < self.loose_amplitude_deg <= self.amplitude_deg <= 90:
            raise ValueError('Direction or configured arc amplitudes invalid')
        if min(self.transfer_s, self.review_s, self.exit_angle_deg, self.target_rate_deg_s,
               self.desired_shift_joints, self.drift_persistence_s, self.rocking_period_s) <= 0:
            raise ValueError('Action timing, release and rate parameters must be positive')


class ActiveArcTransfer:
    def __init__(self, lengths, route, options=TransferOptions(), rocking=False):
        if route not in ('loose', 'transfer'):
            raise ValueError('route must be loose or transfer')
        self.lengths = np.asarray(lengths, float)
        self.n = len(self.lengths)-1
        self.options, self.route, self.rocking = options, route, rocking
        self.window = (options.first_joint, options.last_joint)
        self.initial_window = self.window
        self.enabled = self.mask(self.window)
        self.targets = self.template(self.window, 0., options.amplitude_deg)
        self.phase = 0.
        self.stage = 'prepare'
        self.last_control_time = self.last_feedback_time = None
        self.previous_angles = None
        self.observation = None
        self.events = []
        self.transaction = None
        self.stage_since = None
        self.baseline = None
        self.drift_since = None
        self.direction_index = 0
        self.completed = False
        self.fault = None
        self.last_gate = None
        self.bias = 0.
        self.alignment_since = None
        self.geometry = AngleShapeObserver(self.lengths)
        self.end_helper = RollingShapeObserver(self.lengths, None)

    def mask(self, window):
        if not 2 <= window[0] <= window[1] < self.n:
            raise ValueError('Active arc needs passive boundary joints')
        j = np.arange(1, self.n+1)
        return (j >= window[0]) & (j <= window[1])

    def template(self, window, phase, amplitude, bias=0.):
        factors = np.where(np.arange(self.n) % 2 == 0, math.cos(phase), math.sin(phase))
        target = np.where(self.mask(window), math.radians(amplitude)*factors, 0.)
        # Move bending within the same arc; keep each alternating family's
        # total angle unchanged. This is a position action, not radius feedback.
        for parity in (0, 1):
            family = np.flatnonzero(self.mask(window) & (np.arange(self.n) % 2 == parity))
            if len(family) >= 2:
                target[family[0]] *= 1.-bias
                target[family[-1]] *= 1.+bias
        return target

    def position(self, angles, window):
        """Conditional gravity crest in material coordinates, not world position."""
        try:
            q = np.radians(angles)
            points = forward(q, self.lengths)
            section = local_section(q, self.lengths, *window)
            down, quality = self.end_helper.passive_down(points, window, section)
            if down is None:
                return dict(valid=False, reason=quality['source'], end_quality=quality)
            crest = self.geometry.crest(points, window, section, down)
            return dict(valid=True, crest_joint=crest['joint_coordinate'],
                        end_quality=quality, down_head=down.tolist(),
                        plane_rms_m=section['fit_rms'], observes_world_position=False)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return dict(valid=False, reason=str(exc))

    def ideal_crest(self, phase, amplitude):
        q = self.template(self.initial_window, phase, amplitude)
        points = forward(q, self.lengths)
        section = local_section(q, self.lengths, *self.initial_window)
        arc = section['points'][1:-1]
        down = (arc[0]+arc[-1])/2-arc.mean(axis=0)
        down /= np.linalg.norm(down)
        return self.geometry.crest(points, self.initial_window, section, down)['joint_coordinate']

    def begin_transfer(self, direction, angles, time_s, reason):
        old = self.window
        new = (old[0]+direction, old[1]+direction)
        new_mask = self.mask(new)
        old_mask = self.enabled.copy()
        entering = new_mask & ~old_mask
        preload = self.targets.copy()
        preload[entering] = np.radians(angles)[entering]
        self.targets = preload
        self.enabled = old_mask | new_mask
        self.transaction = dict(old=list(old), new=list(new), start_s=time_s,
                                entering=np.flatnonzero(entering).tolist(),
                                exiting=np.flatnonzero(old_mask & ~new_mask).tolist(),
                                preload=preload.tolist(), start_bias=self.bias)
        self.stage, self.stage_since = 'transfer', time_s
        self.events.append(dict(kind='begin_transfer', time_s=time_s, direction=direction,
                                old=list(old), new=list(new), reason=reason,
                                overlap_count=int(self.enabled.sum()), phase_rad=self.phase))

    def advance(self, angles_deg, feedback_time_s, control_time_s):
        angles = np.asarray(angles_deg, float)
        if angles.shape != (self.n,) or not np.isfinite(angles).all():
            raise ValueError('Finite actual motor angles required')
        if not math.isfinite(feedback_time_s) or not math.isfinite(control_time_s):
            raise ValueError('Finite feedback and control times required')
        if feedback_time_s > control_time_s+1e-8 or control_time_s-feedback_time_s > .5:
            raise ValueError('Feedback is future or stale')
        if self.last_control_time is not None and control_time_s <= self.last_control_time:
            raise ValueError('Increasing control time required')
        if self.last_feedback_time is not None:
            if feedback_time_s < self.last_feedback_time:
                raise ValueError('Feedback time cannot run backwards')
            if feedback_time_s == self.last_feedback_time and not np.array_equal(angles, self.previous_angles):
                raise ValueError('Changed angles with unchanged feedback time')
        dt = .02 if self.last_control_time is None else control_time_s-self.last_control_time
        self.last_control_time = control_time_s
        fresh = self.last_feedback_time is None or feedback_time_s > self.last_feedback_time
        if fresh:
            self.last_feedback_time = feedback_time_s
            self.previous_angles = angles.copy()
            focus = self.window if self.transaction is None else (
                min(self.transaction['old'][0], self.transaction['new'][0]),
                max(self.transaction['old'][1], self.transaction['new'][1]))
            self.observation = self.position(angles, focus)
        opt = self.options
        desired = self.targets.copy()
        if self.stage == 'prepare':
            self.phase = opt.hold_phase_rad*ease((control_time_s-2.)/2.)
            desired = self.template(self.window, self.phase, opt.amplitude_deg)
            if control_time_s >= opt.preparation_s and fresh:
                if not self.observation['valid']:
                    self.fault = 'initial material position unobservable'
                else:
                    self.baseline = self.observation['crest_joint']
                    self.events.append(dict(kind='baseline', time_s=control_time_s,
                        crest_joint=self.baseline, phase_rad=self.phase))
                    if self.route == 'loose':
                        self.stage, self.stage_since = 'loose_roll', control_time_s
                    else:
                        self.begin_transfer(opt.desired_direction, angles, control_time_s, 'active_command')
                        desired = self.targets.copy()
        elif self.stage == 'loose_roll':
            age = control_time_s-self.stage_since
            self.phase = opt.hold_phase_rad+opt.rolling_speed_rad_s*age
            amplitude = opt.amplitude_deg+(opt.loose_amplitude_deg-opt.amplitude_deg)*ease(age/2.)
            desired = self.template(self.window, self.phase, amplitude)
            if fresh and self.observation['valid']:
                predicted = self.ideal_crest(self.phase, amplitude)-self.ideal_crest(opt.hold_phase_rad, opt.amplitude_deg)
                shift = self.observation['crest_joint']-self.baseline-predicted
                self.observation.update(phase_corrected_shift_joints=shift,
                                        ideal_phase_offset_joints=predicted)
                reached = opt.desired_direction*shift >= opt.desired_shift_joints
                self.drift_since = (control_time_s if self.drift_since is None else self.drift_since) if reached else None
                if self.drift_since is not None and control_time_s-self.drift_since >= opt.drift_persistence_s-1e-8:
                    self.events.append(dict(kind='observed_drift', time_s=control_time_s,
                                            shift_joints=shift, observation=self.observation.copy()))
                    self.begin_transfer(opt.desired_direction, angles, control_time_s, 'follow_observed_drift')
                    desired = self.targets.copy()
        elif self.stage == 'transfer':
            tr = self.transaction
            age = control_time_s-tr['start_s']
            if self.rocking:
                self.phase = opt.hold_phase_rad+opt.rocking_amplitude_rad*math.sin(2*math.pi*age/opt.rocking_period_s)
            old = self.template(tuple(tr['old']), self.phase, opt.amplitude_deg, tr['start_bias'])
            new = self.template(tuple(tr['new']), self.phase, opt.amplitude_deg)
            amount = ease(age/opt.transfer_s)
            desired = (1.-amount)*old+amount*new
            entering = tr['entering']
            desired[entering] = (1.-amount)*np.array(tr['preload'])[entering]+amount*new[entering]
            exiting = tr['exiting']
            self.last_gate = dict(exit_angle_deg=angles[exiting].tolist(),
                                  exit_angle_ready=bool(np.max(abs(angles[exiting])) <= opt.exit_angle_deg),
                                  trajectory_finished=amount >= 1., fresh_feedback=fresh)
            if amount >= 1. and fresh and self.last_gate['exit_angle_ready']:
                self.window = tuple(tr['new'])
                self.enabled = self.mask(self.window)
                self.events.append(dict(kind='transfer_complete', time_s=control_time_s,
                    old=tr['old'], new=tr['new'], gate=self.last_gate.copy(),
                    observation=None if self.observation is None else self.observation.copy()))
                self.transaction = None
                self.bias = 0.
                self.alignment_since = None
                self.stage, self.stage_since = ('align' if opt.feedback_align else 'review'), control_time_s
                if self.rocking:
                    self.phase = opt.hold_phase_rad
                desired = self.template(self.window, self.phase, opt.amplitude_deg)
            elif age > opt.maximum_transfer_wait_s:
                self.fault = 'exit angle did not reach release condition'
        elif self.stage == 'align':
            goal = self.baseline+(opt.desired_direction*opt.desired_shift_joints if self.direction_index == 0 else 0.)
            if self.observation and self.observation['valid']:
                error = goal-self.observation['crest_joint']
                self.bias = float(np.clip(self.bias+opt.alignment_gain_s*error*dt,
                                          -opt.alignment_bias_limit, opt.alignment_bias_limit))
                if fresh:
                    reached = abs(error) <= opt.alignment_tolerance_joints
                    self.alignment_since = (feedback_time_s if self.alignment_since is None else self.alignment_since) if reached else None
                    if self.alignment_since is not None and feedback_time_s-self.alignment_since >= opt.alignment_persistence_s-1e-8:
                        self.events.append(dict(kind='material_goal_observed', time_s=control_time_s,
                            goal_joint=goal, measured_crest_joint=self.observation['crest_joint'], bias=self.bias,
                            uses_world_truth=False))
                        self.stage, self.stage_since = 'review', control_time_s
            desired = self.template(self.window, self.phase, opt.amplitude_deg, self.bias)
            if control_time_s-self.stage_since > opt.maximum_alignment_s:
                self.fault = 'material position did not reach configured tolerance'
        elif self.stage == 'review':
            desired = self.template(self.window, self.phase, opt.amplitude_deg, self.bias)
            if control_time_s-self.stage_since >= opt.review_s:
                if self.route == 'transfer' and self.direction_index == 0:
                    self.direction_index += 1
                    self.begin_transfer(-opt.desired_direction, angles, control_time_s, 'active_return')
                    desired = self.targets.copy()
                else:
                    self.completed, self.stage = True, 'complete'
        delta = np.clip(desired-self.targets, -math.radians(opt.target_rate_deg_s)*dt,
                        math.radians(opt.target_rate_deg_s)*dt)
        self.targets[self.enabled] += delta[self.enabled]
        self.targets[~self.enabled] = np.radians(angles)[~self.enabled]
        return self.targets.copy(), self.enabled.copy()

    def snapshot(self):
        return dict(stage=self.stage, route=self.route, window=list(self.window), phase_rad=self.phase,
                    enabled=self.enabled.tolist(), transaction=self.transaction, observation=self.observation,
                    baseline_crest_joint=self.baseline, completed=self.completed, fault=self.fault,
                    position_bias=self.bias,
                    completion_means='prescribed_sequence_finished_not_material_goal_verified',
                    gate=self.last_gate, options=asdict(self.options), rocking=self.rocking,
                    compliance=False, inputs=['angles_deg','feedback_time_s','control_time_s'])


@dataclass(frozen=True)
class ArcHelixOptions:
    radius_m: float = .094
    pitch_m: float = .13
    shift_m_s: float = .026
    roll_rad_s: float = .23
    target_rate_deg_s: float = 23.
    intake_error_deg: float = 16.
    min_end_depth_links: float = 1.5
    max_feedback_age_s: float = .8
    step_s: float = 1.
    staged_shift_m_s: float = .022
    staged_roll_rad_s: float = .14
    staged_target_rate_deg_s: float = 14.
    prepare_front: bool = False
    front_entry_max_links: float = 3.15


class ArcHelixTransition:
    """One curve interval and adjacent motor handoffs, with optional ARC transfer."""

    def __init__(self, lengths, window, measured_deg, direction=1, options=ArcHelixOptions()):
        self.lengths = np.asarray(lengths, float)
        self.n = len(self.lengths)-1
        self.l = float(np.mean(self.lengths))
        self.options = options
        self.direction = direction
        mask = self._mask(window)
        q = np.radians(measured_deg)
        section = local_section(q, self.lengths, *window)
        self.arc_turn_deg = math.degrees(section['total_turn'])
        if not 125. <= self.arc_turn_deg <= 232. or section['fit_rms'] > .025:
            raise ValueError('当前使能区域没有可拟合的开放圆弧')
        head_length, tail_length, head_depth, tail_depth = self.estimate_end_geometry(q, self.lengths, section, window)
        self.end_lengths_m = (head_length, tail_length)
        self.end_depths_m = (head_depth, tail_depth)
        near_equal = abs(head_length-tail_length) <= 2*self.l
        self.staged = options.prepare_front or (near_equal and min(head_length, tail_length) >= 3*self.l)
        # Tail first for the near-equal mounting convention (tail on the left).
        self.reversed = not options.prepare_front and (self.staged or head_length > tail_length+2*self.l)
        self.roll_direction = -direction if self.reversed else direction
        if self.reversed:
            self.lengths = self.lengths[::-1].copy()
        self.window = self.physical_window(window)
        self._enabled = mask[::-1].copy() if self.reversed else mask
        self._targets = self._map_angles(q)
        self.measured = self._targets.copy()
        self.feedback_time = None
        self.stage = 'shift'
        self.running = False
        self.started = False
        self.step_left = 0.
        self.events = [dict(stage='entry', reversed_intake=self.reversed, staged=self.staged, window=list(window),
                            head_length_m=head_length, tail_length_m=tail_length,
                            head_depth_m=head_depth, tail_depth_m=tail_depth)]
        self.transfer = None
        odd = np.arange(self.n)%2 == 0
        components = [np.mean(self.measured[self._enabled & parity]) for parity in (odd, ~odd)]
        self.amplitude = float(np.linalg.norm(components))
        self.phase = math.atan2(components[1], components[0])
        leading_depth = tail_depth if self.reversed else head_depth
        if options.prepare_front:
            self._check_front_entry()
        else:
            self._plan_entry(leading_depth)

    @classmethod
    def estimate_end_geometry(cls, q, lengths, section, window):
        """Estimate arm lengths and end depths using a cut through the fitted ARC center."""
        points = section['projected'][1:-1]
        system = np.column_stack((2.*points, np.ones(len(points))))
        center = np.linalg.lstsq(system, np.sum(points**2, axis=1), rcond=None)[0][:2]
        crown = points.mean(axis=0)-center
        direction = crown/np.linalg.norm(crown)
        backbone = forward(q, lengths)
        projected = (backbone-section['axis']['point'])@section['axis']['basis'][:,:2]
        height = (projected-center)@direction
        first, last = window
        top = first+int(np.argmax(height[first:last+2]))
        material = np.r_[0., np.cumsum(lengths)]
        head = cls._length_after_cut(height[top::-1], material[top]-material[top::-1])
        tail = cls._length_after_cut(height[top:], material[top:]-material[top])
        return head, tail, -float(height[0]), -float(height[-1])

    @staticmethod
    def _length_after_cut(height, material):
        below = np.flatnonzero(height <= 0.)
        if not len(below):
            return 0.
        i = below[0]
        if i == 0:
            return float(material[-1])
        crossing = material[i-1]+(material[i]-material[i-1])*height[i-1]/(height[i-1]-height[i])
        return float(material[-1]-crossing)

    def _mask(self, window):
        j = np.arange(1, self.n+1)
        if not 2 <= window[0] < window[1] < self.n:
            raise ValueError('圆弧窗口应连续且两端至少保留一个失能关节')
        return (j >= window[0]) & (j <= window[1])

    def _map_angles(self, angles):
        angles = np.asarray(angles)
        if not self.reversed:
            return angles.copy()
        out = angles[::-1].copy()
        if self.n%2:
            out[1::2] *= -1.
        return out

    def physical_window(self, window):
        return ((self.n+1-window[1],self.n+1-window[0]) if self.reversed else tuple(window))

    @property
    def targets(self):
        return self._map_angles(self._targets)

    @property
    def enabled(self):
        return self._enabled[::-1].copy() if self.reversed else self._enabled.copy()

    def _plan_entry(self, leading_depth):
        first, last = self.window
        # Add a hinge at the leading end if its measured section has little
        # room for the first helical shoulder.
        if leading_depth < self.options.min_end_depth_links*self.l and last < self.n-1:
            self._prepare_transfer(1)
        elif last == self.n-1 and first > 2:
            self._prepare_transfer(-1)
        else:
            self._prepare_shift()

    def _check_front_entry(self):
        section = local_section(self.measured, self.lengths, *self.window)
        head_length, _, head_depth, _ = self.estimate_end_geometry(self.measured, self.lengths, section, self.window)
        self.events.append(dict(stage='front_check', window=list(self.window),
                                head_length_m=head_length, head_depth_m=head_depth))
        if head_length <= self.options.front_entry_max_links*self.l:
            self._prepare_shift()
        elif self.window[0] > 2:
            self._prepare_transfer(-1)
        else:
            self.stage = 'blocked'
            self.pause()
            self.events.append(dict(stage=self.stage, reason='front geometry unreachable at the transfer boundary'))

    def _prepare_transfer(self, direction):
        first, last = self.window
        self.stage = 'adjust'
        options = TransferOptions(first_joint=first, last_joint=last, hold_phase_rad=self.phase,
                                  amplitude_deg=math.degrees(self.amplitude), desired_direction=direction,
                                  transfer_s=6. if self.options.prepare_front else 5.,
                                  loose_amplitude_deg=min(45., math.degrees(self.amplitude)))
        self.transfer = ActiveArcTransfer(self.lengths, 'transfer', options)
        self.transfer.phase = self.phase
        self.transfer.targets = self._targets.copy()
        self.transfer_clock = 0.
        self.transfer_feedback_sample = None
        reason = 'shorten the leading arm' if self.options.prepare_front else 'leave two receiving hinges'
        self.transfer.begin_transfer(direction, np.degrees(self.measured), 0., reason)
        self.events.append(dict(stage='adjust', before=list(self.physical_window(self.window)),
                                after=list(self.physical_window((first+direction,last+direction)))))

    def _prepare_shift(self):
        self.stage = 'lead' if self.staged else 'shift'
        self.shift = self.roll = self.shift_time = 0.
        self.leading_extension = 0.
        self.leading_pose = None
        self.receive_start = 0.
        self.receive_shift = 0.
        self.accepted = self.window[1]
        self.confirmed = 0
        self.ready = 0
        opt = self.options
        self.curve = ArcHelixShift(CurveGait(self.n, self.l), self.window, self.amplitude,
                                   opt.radius_m, opt.pitch_m, self.direction if self.reversed else -self.direction)
        self.curve.front_extension_rate = max(1.75, 1.+self.curve.head_length/max(.40,self.curve.tail_length))
        nominal, _, _ = self.curve.shifted_window(0., self.curve.arc_length, self.phase)
        self.residual = np.where(self._enabled, self.measured-nominal, 0.)
        self.events.append(dict(stage=self.stage, window=list(self.physical_window(self.window))))

    def step_once(self):
        if self.stage not in ('complete','blocked'):
            self.running = False
            self.step_left = self.options.step_s

    def pause(self):
        self.running = False
        self.step_left = 0.

    def status(self):
        accepted = None if self.staged and not self.options.prepare_front else getattr(self, 'accepted', None)
        if accepted is not None and self.reversed:
            accepted = self.n+1-accepted
        return dict(stage=self.stage, arc_window=list(self.physical_window(self.window)),
                    leading_end='tail' if self.reversed else 'head',
                    intake_end='head' if self.reversed else 'tail',
                    accepted_joint=accepted,
                    entry_end_lengths_m=self.end_lengths_m, entry_end_depths_m=self.end_depths_m,
                    shift_m=getattr(self, 'shift', 0.), step_remaining_s=self.step_left,
                    receive_shift_m=getattr(self, 'receive_shift', 0.),
                    leading_pose=getattr(self, 'leading_pose', None),
                    transfer_stage=None if self.transfer is None or self.stage != 'adjust' else self.transfer.stage,
                    events=self.events.copy())

    def advance(self, dt, now, feedback_time, measured_deg):
        fresh = self.feedback_time is None or feedback_time > self.feedback_time
        if fresh:
            self.measured = self._map_angles(np.radians(measured_deg))
            self.feedback_time = feedback_time
        max_age = min(.5, self.options.max_feedback_age_s) if self.stage == 'adjust' else self.options.max_feedback_age_s
        if (self.stage in ('complete','blocked') or (not self.running and self.step_left <= 0.) or
                now-feedback_time > max_age):
            return False
        self.started = True
        dt = dt if self.running else min(dt,self.step_left)
        self.step_left = max(0., self.step_left-dt)
        if self.step_left < 1e-9:
            self.step_left = 0.
        if self.stage == 'adjust':
            self._advance_transfer(dt)
        elif self.stage == 'front_settle':
            self.front_settle -= dt
            if self.front_settle <= 0. and fresh:
                self._check_front_entry()
        elif self.options.prepare_front and self.stage == 'lead':
            self._advance_prepared_lead(dt, fresh)
        elif self.staged and not self.options.prepare_front:
            self._advance_staged(dt, fresh)
        else:
            self._advance_shift(dt, fresh)
        return True

    def _advance_transfer(self, dt):
        transfer = self.transfer
        self.transfer_clock += dt
        clock = self.transfer_clock
        if getattr(self, 'transfer_feedback_sample', None) != self.feedback_time:
            self.transfer_feedback_clock = clock
            self.transfer_feedback_sample = self.feedback_time
        transfer.advance(np.degrees(self.measured), self.transfer_feedback_clock, clock)
        self._targets = transfer.targets.copy()
        self._enabled = transfer.enabled.copy()
        if transfer.fault:
            self.stage = 'blocked'
            self.events.append(dict(stage=self.stage, reason=transfer.fault))
        elif transfer.stage == 'review':
            self.window = transfer.window
            if self.options.prepare_front:
                self.stage = 'front_settle'
                self.front_settle = .4
            else:
                self._prepare_shift()

    def _tracking_pace(self):
        error = np.max(np.abs(self._targets[self._enabled]-self.measured[self._enabled]))
        return float(np.clip((math.radians(25.)-error)/math.radians(18.), .15, 1.))

    def _advance_prepared_lead(self, dt, fresh):
        opt, curve = self.options, self.curve
        first, last = self.window
        pace = self._tracking_pace()
        distance = opt.staged_shift_m_s*pace*dt
        self.shift_time += dt
        self.roll += self.roll_direction*opt.staged_roll_rad_s*ease(self.shift_time/1.5)*pace*dt
        phase = self.phase+self.roll
        if self.leading_extension < curve.head_length:
            self.leading_extension = min(curve.head_length, self.leading_extension+distance)
            goal, _ = curve.targets(self.leading_extension, 0., curve.arc_length, phase)
            coverage = np.clip(self.leading_extension/self.l+np.arange(1,self.n+1)-first+.999, 0., 1.)
        else:
            self.shift = min(1.7*self.l, self.shift+distance)
            arc_length = curve.arc_length*(1.-ease(self.shift/.16))
            goal, _, coverage = curve.shifted_window(self.shift, arc_length, phase, self.leading_extension,
                                                    self._prepared_helix())
        coverage[last:] = 0.
        goal = self._apply_curve(goal, coverage, opt.staged_target_rate_deg_s, dt)
        if not fresh:
            return
        self.leading_pose = self._leading_pose(goal)
        error = np.max(abs(goal[:last]-self.measured[:last]))
        ready = (self.leading_extension == curve.head_length and self.leading_pose['axial_m'] >= .022
                 and -.024 <= self.leading_pose['radial_offset_m'] <= -.0135
                 and self.leading_pose['axial_tangent'] >= .35
                 and error <= math.radians(opt.intake_error_deg))
        self.ready = self.ready+1 if ready else 0
        if self.ready >= 3:
            self.ready = 0
            self.stage = 'receive'
            self.receive_start = self.shift
            self.events.append(dict(stage=self.stage, shift_m=self.shift, leading_pose=self.leading_pose.copy()))

    def _leading_pose(self, goal):
        first, last = self.window
        section = local_section(self.measured, self.lengths, first, last)
        points = section['projected'][1:-1]
        system = np.column_stack((2.*points, np.ones(len(points))))
        center = np.linalg.lstsq(system, np.sum(points**2, axis=1), rcond=None)[0][:2]
        radius = np.median(np.linalg.norm(points-center, axis=1))
        points = forward(self.measured, self.lengths)
        head_link = (points[:2]-section['axis']['point'])@section['axis']['basis']
        head = head_link.mean(axis=0)
        shoulder = (points[first]-section['axis']['point'])@section['axis']['basis']
        reference = forward(goal, self.lengths)
        axis = section['axis']['basis'][:,2]
        expected_axial = (reference[:2].mean(axis=0)-reference[first])@axis
        forward_sign = math.copysign(1., expected_axial)
        return dict(axial_m=float(forward_sign*(head[2]-shoulder[2])),
                    radial_offset_m=float(np.linalg.norm(head[:2]-center)-radius),
                    axial_tangent=float(forward_sign*(head_link[0,2]-head_link[1,2])/self.lengths[0]))

    def _prepared_helix(self):
        if not self.options.prepare_front:
            return None
        amount = ease(self.shift/.075)
        if self.stage == 'receive':
            amount *= 1.-ease((self.shift-self.receive_start)/.14)
        opt = self.options
        return helix(0., opt.radius_m-.012*amount, opt.pitch_m, -self.direction)

    def _advance_staged(self, dt, fresh):
        opt, curve = self.options, self.curve
        first, last = self.window
        leading = self.stage == 'lead'
        pace = self._tracking_pace()
        distance = opt.staged_shift_m_s*pace*dt
        self.shift_time += dt
        self.roll += self.roll_direction*opt.staged_roll_rad_s*ease(self.shift_time/1.5)*pace*dt
        if leading:
            self.shift = min(curve.head_length, self.shift+distance)
        else:
            self.receive_shift = min(curve.tail_length, self.receive_shift+distance)
        # Keep the initial supporting ARC while only the first straight arm
        # is converted. Shorten the ARC as the second arm enters the curve.
        arc_length = curve.arc_length*(1.-ease(self.receive_shift/min(.12,curve.tail_length)))
        goal, _ = curve.targets(self.shift, self.receive_shift, arc_length, self.phase+self.roll)
        j = np.arange(1,self.n+1)
        coverage = np.clip(np.where(j < first, self.shift/self.l+j-first,
                                    np.where(j > last, self.receive_shift/self.l+last-j, 1.))+.999, 0., 1.)
        goal = self._apply_curve(goal, coverage, opt.staged_target_rate_deg_s, dt)
        if not fresh:
            return
        if leading:
            error = np.max(abs(goal[:last]-self.measured[:last]))
            settled = self.shift == curve.head_length and error <= math.radians(opt.intake_error_deg)
        else:
            error = np.max(abs(goal-self.measured))
            settled = self.receive_shift == curve.tail_length and error <= math.radians(10.)
        self.ready = self.ready+1 if settled else 0
        if self.ready >= 3:
            self.ready = 0
            self.stage = 'receive' if leading else 'complete'
            self.events.append(dict(stage=self.stage, shift_m=self.shift, receive_shift_m=self.receive_shift))
            if not leading:
                self.pause()

    def _advance_shift(self, dt, fresh):
        opt, curve = self.options, self.curve
        pace = self._tracking_pace()
        limit = (self.accepted-self.window[1]+1.5)*self.l if self.accepted < self.n else math.inf
        self.shift_time += dt
        if opt.prepare_front:
            self.shift += min(max(0., limit-self.shift), opt.shift_m_s*pace*dt)
            self.receive_shift = self.shift-self.receive_start
        else:
            self.shift = min(limit, self.shift+opt.shift_m_s*pace*dt)
        self.roll += self.roll_direction*opt.roll_rad_s*ease(self.shift_time/1.5)*pace*dt
        arc_length = curve.arc_length*(1.-ease(self.shift/.16))
        goal, _, coverage = curve.shifted_window(self.shift, arc_length, self.phase+self.roll,
                                                self.leading_extension, self._prepared_helix())
        if self.options.prepare_front:
            coverage[self.accepted+1:] = 0.
        goal = self._apply_curve(goal, coverage, opt.target_rate_deg_s, dt)
        if fresh and self.accepted < self.n:
            i = self.accepted
            follows = np.max(np.abs(goal[i-1:i+1]-self.measured[i-1:i+1])) <= math.radians(opt.intake_error_deg)
            self.confirmed = self.confirmed+1 if coverage[i] >= .65 and follows else 0
            if self.confirmed >= 2:
                self.accepted += 1
                self.confirmed = 0
                joint = self.n+1-self.accepted if self.reversed else self.accepted
                self.events.append(dict(stage='intake', joint=joint, shift_m=self.shift))
        if fresh and self.accepted == self.n and np.all(coverage >= .999):
            self.ready = self.ready+1 if np.max(np.abs(goal-self.measured)) <= math.radians(10.) else 0
            if self.ready >= 3:
                self.stage = 'complete'
                self.pause()
                self.events.append(dict(stage=self.stage, shift_m=self.shift))

    def _apply_curve(self, goal, coverage, rate_deg_s, dt):
        goal = np.clip(goal+self.residual*(1.-ease(self.shift_time/2.)), -math.radians(88.), math.radians(88.))
        entering = (coverage > 1.e-6) & ~self._enabled
        self._targets[entering] = self.measured[entering]
        self._enabled |= entering
        # Apply the overlap as a position blend. Both backends use the same
        # binary enable mask; no simulated stiffness or contact is supplied.
        receive = self._enabled & (coverage < .999)
        blended = goal.copy()
        blended[receive] = self.measured[receive]+coverage[receive]*(goal[receive]-self.measured[receive])
        rate = math.radians(rate_deg_s)*dt
        self._targets += np.clip(blended-self._targets,-rate,rate)
        self._targets[~self._enabled] = self.measured[~self._enabled]
        return goal


@dataclass(frozen=True)
class GravityMomentOptions:
    r_h1: float = .092
    r_h2: float = .096
    pitch: float = .13
    r_c: float = .09
    handed: int = -1                 # torsion sign; with direction=+1 this is a right-handed helix
    roll_rate: float = .6            # rad/s on psi(0); sign chosen so the robot advances toward +x
    t1: float = 1.0                  # contact thresholds on mean |torque| (N*m), set by calibration
    t2: float = 1.125
    settle_s: float = 2.
    detect_after_s: float = 1.5      # ignore the detector right after (re)entering phase 1
    arc_s: float = 3.                # phase 2: theta1 0 -> pi
    line_s: float = 4.               # phase 2: l_s 0 -> one helix turn - pi*r_c
    loosen_s: float = 2.             # phase 2: r_h1 -> r_h2
    drop_s: float = 3.               # phase 2: tail disabled, hangs under gravity
    extend_s: float = 2.             # phase 3: theta2 = alpha_h, l_s = (n_ext+1) l
    extend_hold_s: float = 1.
    rewind_s: float = 6.             # phase 4: l_s -> keep_links*l, theta2 -> 0, r_h2 -> r_h1, rolling
    keep_links: float = 2.
    straighten_s: float = 2.         # back in phase 1: l_s -> 0
    blend_s: float = 1.5             # re-enabled joints blend from measured angles
    crossing_roll: bool = True       # False: no rolling during phases 2-4 (user 11-06: advance comes from
                                     # re-winding against the branch, not from rolling)
    direct_cycle: bool = False       # True: after settle and after each rewind go straight to phase 2
    cycle_hold_s: float = 1.
    rewind_loose: bool = False       # keep r_h2 while re-forming the helix; tighten to r_h1 in cycle_hold
    line_rate: float = .1            # m/s, l_s growth when the operator stops the unwinding by eye
    shift_fit: bool = False          # phase 4 by shift control: the helix phase stays fixed at the tail
                                     # junction (the front is an extended helix), so as the tail re-forms
                                     # the front helix moves along the curve. Phases 2-3 stay compensated
                                     # (eqs. (3), (7)); phase 4 has no compensation in the paper.
    shift_rate: float = 0.           # m/s > 0: phase 4 slides the body along ONE fixed curve (helix, arc
                                     # theta2 = alpha_h kept, line) at this rate, radius kept at r_h2;
                                     # implies shift_fit. 0: old timed rewind (arc shrinks, rewind_s).
    max_target_deg: float = 88.


def smooth(x):
    x = min(max(x, 0.), 1.)
    return x * x * (3 - 2 * x)


class GravityMomentGait:
    def __init__(self, joints=18, options=GravityMomentOptions(), link_length=.0645):
        self.o = options
        self.gait = CurveGait(joints, link_length)
        self.l = link_length
        self.L = (joints + 1) * link_length

    def turn_length(self, r):
        return math.hypot(2 * math.pi * r, self.o.pitch)

    def lead_angle(self, r):
        return math.atan(self.o.pitch / (2 * math.pi * r))

    def n_ext(self, r):
        return int(self.turn_length(r) // self.l)

    def segments(self, r_h, ls=0., th1=0., th2=0.):
        """Return (segments, helix_length). Seg 3's +pi/2 twist also applies when theta2 = 0."""
        o = self.o
        a2, a3 = o.r_c * th1, o.r_c * th2
        lh = self.L - ls - a2 - a3
        if lh <= 0:
            raise ValueError('tail segments longer than the body')
        h = helix(1., r_h, o.pitch, o.handed)
        segs = [HelixCurveSegment(lh, h.kappa, h.tau)]
        carry = math.pi / 2
        if a3 > 0:
            segs.append(HelixCurveSegment(a3, 1 / o.r_c, 0., carry))
            carry = 0.
        if a2 > 0:
            segs.append(HelixCurveSegment(a2, 1 / o.r_c, 0., carry - math.pi))
        if ls > 0:
            segs.append(HelixCurveSegment(ls))
        return segs, lh

    def targets(self, roll, r_h, ls=0., th1=0., th2=0.):
        segs, lh = self.segments(r_h, ls, th1, th2)
        q = self.gait.targets(segs, roll)
        lim = math.radians(self.o.max_target_deg)
        return np.clip(q, -lim, lim), lh

    def helix_joints(self, lh):
        """Joints whose two neighbouring links lie entirely on the helix part."""
        j = np.arange(1, self.gait.joints + 1)
        return (j + 1) * self.l <= lh + 1e-9


class GravityFrontTransition(ArcHelixTransition):
    """保持承载 ARC，只延长头端螺旋，直至操作者确认贴管。"""

    def _check_front_entry(self):
        section = local_section(self.measured, self.lengths, *self.window)
        head, tail, _, _ = self.estimate_end_geometry(self.measured, self.lengths, section, self.window)
        self.events.append(dict(stage='front_check', head_length_m=head, tail_length_m=tail))
        if head+.5*self.l <= tail:
            self._prepare_shift()
        elif self.window[0] > 2:
            self._prepare_transfer(-1)
        else:
            self.stage = 'blocked'
            self.pause()
            self.events.append(dict(stage='blocked', reason='调到头侧边界后仍未形成头短尾长的起卷姿态'))

    def _advance_prepared_lead(self, dt, fresh):
        first, last = self.window
        pace = self._tracking_pace()
        self.shift_time += dt
        self.leading_extension = min(self.curve.head_length, self.leading_extension+.035*pace*dt)
        self.roll += .23*smooth(self.shift_time/1.5)*pace*dt
        goal, _ = self.curve.targets(self.leading_extension, 0., self.curve.arc_length, self.phase+self.roll)
        coverage = np.clip(self.leading_extension/self.l+np.arange(1, self.n+1)-first+1., 0., 1.)
        coverage[last:] = 0.
        self._apply_curve(goal, coverage, 23., dt)


@dataclass(frozen=True)
class GravityCrossingOptions:
    retain_joints: int = 8
    release_line_m: float = .32201388
    release_radius_m: float = .09554165
    alignment_rate_rad_s: float = .14
    shift_rate_m_s: float = .012
    shift_roll_rad_s: float = .35
    intake_error_deg: float = 16.
    step_s: float = 1.
    max_feedback_age_s: float = .8


class GravityCrossing:
    """头侧成形 → Shift Rolling → 解旋成右 U → 失能 → 回卷。"""

    def __init__(self, lengths, enabled, measured_deg, gait_options=GravityMomentOptions(r_h1=.094, r_h2=.098),
                 options=GravityCrossingOptions(), from_helix=False):
        lengths = np.asarray(lengths, float)
        self.n, self.l = len(lengths)-1, float(np.mean(lengths))
        if not np.allclose(lengths, self.l):
            raise ValueError('T 管模式的曲线积分需要等间距关节')
        if not 4 <= options.retain_joints < self.n:
            raise ValueError('请检查 T 管模式的前部保留关节数')
        self.gait = GravityMomentGait(self.n, gait_options, self.l)
        self.gait.segments(options.release_radius_m, ls=options.release_line_m, th1=math.pi)
        self.options = options
        self._enabled = np.asarray(enabled, bool).copy()
        self.measured = np.radians(measured_deg)
        self._targets = self.measured.copy()
        self.running = self.started = False
        self.reversed = False  # 本模式固定 J1 为前端，右 U 的方向与此绑定。
        self.step_left = self.elapsed = self.phase = self.offset = self.roll = 0.
        self.feedback_time = None
        self.accepted = None
        self.front = None
        self.front_extended = False
        self.align_direction = 1
        self.events = []
        if from_helix:
            if not self._enabled.all():
                raise ValueError('从 Helix 接入 T 管模式需要已成形的全身螺旋')
            self.phase = self._fit_phase(self.n, gait_options.r_h1)
            error = np.sqrt(np.mean((self._helix(self.phase, gait_options.r_h1)-self.measured)**2))
            if error > math.radians(16.):
                raise ValueError('实测 Helix 的绕向或形状与 J1 在前的 T 管曲线不符')
            self._enter('align')
        else:
            active = np.flatnonzero(self._enabled)+1
            if len(active) < 3 or np.any(np.diff(active) != 1):
                raise ValueError('T 管模式需要先形成连续的局部承载圆弧')
            arc_options = ArcHelixOptions(radius_m=gait_options.r_h1, pitch_m=gait_options.pitch,
                                         prepare_front=True)
            self.front = GravityFrontTransition(lengths, (int(active[0]), int(active[-1])),
                                                measured_deg, options=arc_options)
            self._enter('front')

    @property
    def targets(self):
        return self._targets.copy()

    @property
    def enabled(self):
        return self._enabled.copy()

    def _enter(self, stage):
        self.stage, self.elapsed = stage, 0.
        self.events.append(dict(stage=stage, phase_rad=self.phase, enabled=(np.flatnonzero(self._enabled)+1).tolist()))

    def pause(self):
        self.running = False
        self.step_left = 0.

    def step_once(self):
        self.running = False
        self.step_left = self.options.step_s

    def _helix(self, phase, radius):
        return self.gait.targets(phase, radius)[0]

    def _fit_phase(self, count, radius):
        basis = np.column_stack([self._helix(p, radius)[:count] for p in (0., math.pi/2)])
        weights = np.linalg.lstsq(basis, self.measured[:count], rcond=None)[0]
        return math.atan2(weights[1], weights[0])

    def _start_shift(self, support, stage):
        radius = self.gait.o.r_h2
        self.initial_prefix = self.accepted = support
        self.phase = self._fit_phase(support, radius)
        self.curve = MeasuredGravityShift(self.gait, radius, self.phase, self.measured, support,
                                         upstream_m=(support+1)*self.l)
        self.offset = self.roll = 0.
        self.tracking_since = None
        self._enter(stage)

    def confirm(self, feedback_time, measured_deg):
        """Ctrl+L 标记操作者看到的几何事件，之后仍保持到 J/G。"""
        if self.running or self.step_left > 0.:
            raise ValueError('请先暂停，再按 Ctrl+L 确认实际姿态')
        self.measured = np.radians(measured_deg)
        self.feedback_time = feedback_time
        self.started = True
        if self.stage == 'front':
            if self.front.stage != 'lead' or self.front.leading_extension < self.front.curve.head_length:
                raise ValueError('前部曲线还在形成，请先用 Ctrl+J/G 完成前卷')
            self._start_shift(self.front.window[1], 'first_shift')
        elif self.stage == 'align':
            self.anchor = self._targets.copy()
            self._enter('unwind')
        elif self.stage == 'right_u_hold' or (self.stage == 'unwind' and self.elapsed >= 3.):
            self.anchor = self._targets.copy()
            self._enter('drop')
        elif self.stage == 'drop_hold':
            self._start_shift(self.options.retain_joints, 'rewrap')
        else:
            raise ValueError('当前阶段用 Ctrl+J 单步或 Ctrl+G 连续推进')

    def _set_targets(self, goal, dt, rate=18.):
        limit = math.radians(rate)*dt
        goal = np.clip(goal, -math.radians(88.), math.radians(88.))
        self._targets += np.clip(goal-self._targets, -limit, limit)
        self._targets[~self._enabled] = self.measured[~self._enabled]

    def advance(self, dt, now, feedback_time, measured_deg):
        if (feedback_time is None or now-feedback_time > self.options.max_feedback_age_s or
                (not self.running and self.step_left <= 0.)):
            return False
        fresh = self.feedback_time is None or feedback_time > self.feedback_time
        if fresh:
            self.measured = np.radians(measured_deg)
            self.feedback_time = feedback_time
        self.started = True
        dt = dt if self.running else min(dt, self.step_left)
        self.step_left = max(0., self.step_left-dt)
        if self.step_left < 1e-9:
            self.step_left = 0.
        self.elapsed += dt

        if self.stage == 'front':
            self.front.running = True
            self.front.advance(dt, now, feedback_time, measured_deg)
            self._enabled, self._targets = self.front.enabled, self.front.targets
            self.phase = self.front.phase+self.front.roll if self.front.stage == 'lead' else self.front.phase
            if self.front.stage == 'blocked':
                self.pause()
                self._enter('blocked')
            elif (not self.front_extended and self.front.stage == 'lead' and
                  self.front.leading_extension >= self.front.curve.head_length):
                self.front_extended = True
                self.pause()
        elif self.stage in ('first_shift', 'rewrap'):
            self._advance_shift(dt, fresh)
        elif self.stage == 'align':
            self.phase += self.align_direction*self.options.alignment_rate_rad_s*dt
            self._set_targets(self._helix(self.phase, self.gait.o.r_h1), dt)
        elif self.stage == 'unwind':
            self._advance_unwind(dt)
        elif self.stage == 'drop':
            self._enabled[self.options.retain_joints:] = False
            opened = self._helix(self.phase, self.gait.o.r_h2)
            self._targets = self.anchor+smooth(self.elapsed/.40)*(opened-self.anchor)
            self._targets[~self._enabled] = self.measured[~self._enabled]
            if self.elapsed >= 1.-1e-9:
                self._enter('drop_hold')
                self.pause()
        elif self.stage in ('first_recover', 'recover'):
            self._advance_recover(dt)
        elif self.stage == 'roll':
            self.phase += self.gait.o.roll_rate*dt
            self._set_targets(self._helix(self.phase, self.gait.o.r_h1), dt, rate=45.)
        return True

    def _advance_unwind(self, dt):
        t, opt, gait = self.elapsed, self.options, self.gait
        radius = gait.o.r_h1+(opt.release_radius_m-gait.o.r_h1)*smooth((t-7.)/.85)
        goal = gait.targets(self.phase, radius, ls=opt.release_line_m*smooth((t-3.)/4.),
                            th1=math.pi*smooth(t/3.))[0]
        residual = self.anchor-self._helix(self.phase, gait.o.r_h1)
        self._set_targets(goal+(1.-smooth(t))*residual, dt, rate=30.)
        if t >= 7.85:
            self._enter('right_u_hold')
            self.pause()

    def _advance_shift(self, dt, fresh):
        """采样窗向前移动，滚动相位同时推进，尾侧从已接管边界逐节加入。"""
        opt, count = self.options, self.accepted
        stop = min(self.n, count+3)
        error = np.max(abs(self._targets[self._enabled]-self.measured[self._enabled]))
        pace = float(np.clip((math.radians(23.)-error)/math.radians(17.), .10, 1.))
        limit = (count-self.initial_prefix+2)*self.l if count < self.n else self.curve.shift_limit_m
        advance = min(opt.shift_rate_m_s*pace*dt, max(0., limit-self.offset))
        self.offset += advance
        turn = advance*opt.shift_roll_rad_s/.05
        if count < self.n and self.offset >= limit:
            turn = max(turn, .055*pace*dt)
        self.roll += turn
        self.phase = self.curve.phase_rad-self.curve.torsion*self.offset+self.roll
        goal = self.curve.targets(self.offset, self.roll)
        nominal = self.curve.helical_targets(self.offset, self.roll)
        blend = smooth(self.offset/self.gait.turn_length(self.gait.o.r_h2))
        goal[:count] += blend*(nominal[:count]-goal[:count])

        # 舵机是二值使能。边界后最多三节以位置目标渐入，未到边界的尾段仍失能。
        for i in range(count, stop):
            reach = smooth((self.offset/self.l-(i-self.initial_prefix)+1.)/.75)
            if reach > 0.:
                if not self._enabled[i]:
                    self._targets[i] = self.measured[i]
                    self._enabled[i] = True
                weight = reach*[.55, .45, .30][i-count]
                goal[i] = self.measured[i]+weight*(goal[i]-self.measured[i])
        self._set_targets(goal, dt)
        if not fresh:
            return
        if count < self.n:
            arrived = self.offset >= (count-self.initial_prefix+.75)*self.l
            follows = np.max(abs(nominal[count-1:count+1]-self.measured[count-1:count+1])) <= math.radians(opt.intake_error_deg)
            ready = arrived and follows
        else:
            ready = np.max(abs(nominal-self.measured)) <= math.radians(10.)
        self.tracking_since = (self.elapsed if self.tracking_since is None else self.tracking_since) if ready else None
        if self.tracking_since is not None and self.elapsed-self.tracking_since >= .30:
            self.tracking_since = None
            if count < self.n:
                self.accepted += 1
                self.events.append(dict(stage='intake', joint=self.accepted, shift_m=self.offset))
            else:
                self.recover_phase, self.anchor = self.phase, self._targets.copy()
                self._enter('first_recover' if self.stage == 'first_shift' else 'recover')

    def _advance_recover(self, dt):
        t, gait = self.elapsed, self.gait
        start_rate = self.options.shift_rate_m_s*(self.options.shift_roll_rad_s/.05-self.curve.torsion)
        rate = start_rate+(gait.o.roll_rate-start_rate)*smooth((t-3.)/2.)
        self.phase += rate*dt
        radius = gait.o.r_h2+(gait.o.r_h1-gait.o.r_h2)*smooth(t/3.)
        residual = self.anchor-self._helix(self.recover_phase, gait.o.r_h2)
        self._set_targets(self._helix(self.phase, radius)+(1.-smooth(t/2.))*residual, dt, rate=45.)
        if t >= 5.:
            if self.stage == 'first_recover':
                self._enter('align')
                self.pause()
            else:
                self._enter('roll')

    def status(self):
        front = None if self.front is None else self.front.status()
        return dict(stage=self.stage, elapsed_s=self.elapsed, phase_rad=math.remainder(self.phase, math.tau),
                    accepted_joint=self.accepted, shift_m=self.offset, step_remaining_s=self.step_left,
                    front=front, events=self.events.copy())
# END ARC_HELIX_CORE

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
        self.configured_window = (self.first, self.last)
        self.initial_window_selection = None
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
        self.motor_clock_origin_s = time.perf_counter()
        self.feedback_motor_clock_s = None
        self.feedback_phase_rad = self.command_phase
        self.preview_feedback = False
        self.feeding_selected, self.feeding = False, None
        self.feeding_origin = (self.first, self.last)
        self.feed_distance_m, self.feed_direction = abs(FEED_DISTANCE_M), 1 if FEED_DISTANCE_M>0 else -1
        self.feeding_paused, self.feeding_pause_clock = True, None
        self.feeding_clock_origin = 0.
        self.feeding_action_id = 0
        self.feeding_inputs, self.feeding_last_stage = None, None
        self.feeding_last_diagnostic_s = None
        self.helix_transition = None

    def read_helix_entry(self):
        """用同一帧全身舵机角重建当前搭管姿态并规划切换。"""
        active = np.flatnonzero(self.enabled)+1
        if len(active) < 3 or np.any(np.diff(active) != 1):
            raise ValueError('ARC→Helix 需要一个连续使能的圆弧')
        self.refresh_compliance_feedback()
        options = ArcHelixOptions(radius_m=HELIX_RADIUS_M, pitch_m=HELIX_PITCH_M,
                                  min_end_depth_links=HELIX_MIN_END_DEPTH_LINKS, step_s=HELIX_STEP_S)
        return ArcHelixTransition(LINK_LENGTHS_M, (int(active[0]), int(active[-1])),
                                  self.measured_angles(), self.direction, options)

    def select_helix(self):
        """接管时保持已有目标，第一次推进时按最新角度重新规划。"""
        if self.helix_transition is not None and self.helix_transition.started:
            self.helix_transition.pause()
            self.mode, self.freeze_targets = 'hold', True
            return
        if self.mode == 'released' or self.forming or self.settle_remaining > 0:
            raise ValueError('请先形成并稳定搭管圆弧，再按Ctrl+5')
        self.helix_transition = self.read_helix_entry()
        self.hold_for_helix_mode()

    def hold_for_helix_mode(self):
        self.sliding, self.compliance, self.pending_gait = None, None, None
        self.feeding_selected, self.feeding = False, None
        self.feeding_inputs = None
        self.section_block_reason = None
        self.pending_phase, self.mode, self.freeze_targets = 0., 'hold', True

    def read_gravity_entry(self, from_helix):
        self.refresh_compliance_feedback()
        gait = GravityMomentOptions(r_h1=HELIX_RADIUS_M, r_h2=GRAVITY_OPEN_RADIUS_M, pitch=HELIX_PITCH_M)
        options = GravityCrossingOptions(retain_joints=GRAVITY_RETAIN_JOINTS, release_line_m=GRAVITY_RELEASE_LINE_M,
                                         release_radius_m=GRAVITY_RELEASE_RADIUS_M,
                                         alignment_rate_rad_s=GRAVITY_ALIGN_RATE_RAD_S,
                                         shift_rate_m_s=GRAVITY_SHIFT_RATE_M_S, shift_roll_rad_s=GRAVITY_SHIFT_ROLL_RAD_S,
                                         intake_error_deg=GRAVITY_INTAKE_ERROR_DEG, step_s=HELIX_STEP_S)
        return GravityCrossing(LINK_LENGTHS_M, self.enabled, self.measured_angles(), gait, options, from_helix)

    def select_gravity(self):
        current = self.helix_transition
        if isinstance(current, GravityCrossing):
            current.pause()
            self.mode, self.freeze_targets = 'hold', True
            return
        if self.mode == 'released' or self.forming or self.settle_remaining > 0.:
            raise ValueError('请先将开放圆弧稳定搭在主管上，再按 Ctrl+6')
        if current is not None and current.stage != 'complete':
            raise ValueError('请先完成当前 ARC→Helix，再从已成形的 Helix 接入 T 管模式')
        self.helix_transition = self.read_gravity_entry(current is not None)
        self.hold_for_helix_mode()

    def command_helix(self, command):
        core = self.helix_transition
        if command in ('release', 'quit', 'exit', 'status'):
            return False
        gravity = isinstance(core, GravityCrossing)
        if gravity and command == 'feed_tick':
            if self.mode != 'hold':
                raise ValueError('请先暂停，再按 Ctrl+L 确认实际姿态')
            self.refresh_compliance_feedback()
            _, sample_time = self.motor_sample_clock()
            core.confirm(sample_time, self.measured_angles())
            return True
        if gravity and core.stage == 'align' and command in ('step', 'back', 'run', 'toggle'):
            core.align_direction = -1 if command == 'back' else 1
        if command in ('pause', 'hold') or (command == 'toggle' and core.running):
            core.pause()
            self.mode = 'hold'
        elif command in ('step', 'run', 'toggle') or (gravity and core.stage == 'align' and command == 'back'):
            if core.stage in ('complete', 'blocked'):
                raise ValueError('查看Ctrl+K的切换结果；重新形成圆弧前先按Ctrl+X')
            if gravity and core.stage in ('right_u_hold', 'drop_hold'):
                raise ValueError('当前等待实物姿态确认；先查看 Ctrl+K，再按 Ctrl+L')
            if not core.started and self.mode == 'hold':
                align_direction = core.align_direction if gravity else 1
                core = self.read_gravity_entry(core.front is None) if gravity else self.read_helix_entry()
                self.helix_transition = core
                if gravity:
                    core.align_direction = align_direction
            if command in ('step', 'back'):
                if self.mode != 'step':
                    core.step_once()
                self.mode = 'step'
            else:
                core.pause()
                core.running = True
                self.mode = 'run'
        else:
            if gravity:
                raise ValueError('T 管模式用 Ctrl+J/G 推进、Ctrl+L 确认、Ctrl+空格暂停、Ctrl+X 失能')
            raise ValueError('ARC→Helix 使用Ctrl+J单步、Ctrl+G连续、Ctrl+空格暂停；退出用Ctrl+X')
        self.freeze_targets = self.mode == 'hold'
        return True

    def tick_helix(self, dt):
        core = self.helix_transition
        if self.mode == 'hold':
            return
        previous_stage = core.stage
        now, sample_time = self.motor_sample_clock()
        measured = self.measured_angles()
        if not core.advance(dt, now, sample_time, measured):
            return
        enabled = core.enabled.tolist()
        # 回读可能在软件行程外一个编码步。用最近的可写位置预置，再开扭矩。
        targets = np.clip(np.degrees(core.targets), MIN_ANGLE_DEG, MAX_ANGLE_DEG).tolist()
        for i, active in enumerate(enabled):
            if active:
                angle_to_raw(i, targets[i])
        entering = [new and not old for old, new in zip(self.enabled, enabled)]
        if any(entering):
            preload = np.clip(measured, MIN_ANGLE_DEG, MAX_ANGLE_DEG).tolist()
            self.bus.prepare_enabled(preload, enabled, configure_mask=entering)
        elif enabled != self.enabled:
            self.bus.set_enabled(enabled)
        self.bus.send_targets(targets, enabled)
        self.enabled, self.targets = enabled, targets
        active = np.flatnonzero(enabled)+1
        self.first, self.last = int(active[0]), int(active[-1])
        gravity = isinstance(core, GravityCrossing)
        if core.stage != 'adjust':
            phase = core.phase if gravity else core.phase+getattr(core, 'roll', 0.)
            self.command_phase = self.phase = math.pi/2-phase if core.reversed else phase
        if not core.running and core.step_left <= 0.:
            self.mode, self.freeze_targets = 'hold', True
        if gravity and (core.stage != previous_stage or (self.mode == 'hold' and core.stage == 'front' and core.front_extended)):
            self.show_status()

    def select_feeding(self):
        """选择只读就绪状态；任务由J/H/G明确启动，不在切模式时移窗。"""
        self.sliding, self.compliance = None, None
        self.sliding_fault = False
        self.feeding_selected, self.feeding = True, None
        self.feeding_paused, self.freeze_targets = True, False
        self.section_block_reason = None
        if self.mode == 'released' or self.forming or self.settle_remaining>0:
            self.pending_gait = 'feeding'
        else:
            self.mode, self.pending_phase = 'hold', 0.
            self.pending_gait = None

    def start_feeding(self, direction=None, running=True):
        """每按一次执行一个相对调长段；已有任务保留当前弯曲分布。"""
        if self.mode=='released' or self.forming or self.settle_remaining>0:
            raise ValueError('先Ctrl+U形成圆弧，等待形成/静置完成，再启动送链')
        self.refresh_compliance_feedback()
        now, feedback = self.motor_sample_clock()
        self.feed_direction = self.feed_direction if direction is None else direction
        if self.feeding is not None:
            core = self.feeding
            # Restore the clock before starting a new relative action. Do not
            # regenerate the seed arc or reset p when changing direction.
            core.completed = False
            self.resume_feeding()
            now, _ = self.motor_sample_clock()
            core.request_step(self.feed_direction*self.feed_distance_m, now-self.feeding_clock_origin)
            core.options = replace(core.options, automatic_return=FEED_AUTOMATIC_RETURN,
                maximum_feed_s=60. if self.feed_distance_m<=float(np.mean(LINK_LENGTHS_M))+1e-9 else 100.)
            self.feeding_action_id += 1
            self.feeding_inputs = None
            self.feeding_last_stage = None
            if not running:
                self.pause_feeding()
            return
        options = ContinuousFeedingOptions(first_joint=self.feeding_origin[0], last_joint=self.feeding_origin[1],
            amplitude_deg=FEED_AMPLITUDE_DEG, requested_transfer_m=self.feed_direction*self.feed_distance_m,
            preparation_s=FEED_PREPARATION_S,
            maximum_feed_s=60. if self.feed_distance_m<=float(np.mean(LINK_LENGTHS_M))+1e-9 else 100.)
        options = replace(options, **FEED_OPTIONS_OVERRIDE)
        options = replace(options, automatic_return=FEED_AUTOMATIC_RETURN)
        core = ContinuousFeeding(LINK_LENGTHS_M, options)
        core.length_observer.options = replace(core.length_observer.options, **FEED_DIRECTION_OVERRIDE)
        measured = np.asarray(self.measured_angles())
        mask = core.enabled | np.asarray(self.enabled)
        entering = mask & ~np.asarray(self.enabled)
        self.touched_hardware = True
        if np.any(entering):
            self.bus.prepare_enabled(measured, mask, configure_mask=entering)
            for i in np.flatnonzero(entering):
                self.targets[i] = float(measured[i])
        core.enabled = mask
        active = np.flatnonzero(mask)+1
        core.window = (int(active[0]), int(active[-1]))
        core.targets = np.radians(self.targets)
        self.enabled = mask.tolist()
        self.first, self.last = core.window
        self.feeding, self.feeding_selected = core, True
        self.feeding_clock_origin = feedback
        self.feeding_action_id += 1
        self.feeding_last_stage = None
        self.feeding_last_diagnostic_s = None
        self.feeding_inputs = None
        self.phase = self.command_phase = self.pending_phase = 0.
        self.forming, self.settle_remaining = False, 0.
        self.freeze_targets, self.feeding_paused = False, not running
        self.feeding_pause_clock = None if running else now
        self.mode = 'run' if running else 'hold'

    def pause_feeding(self):
        if not self.feeding_paused:
            self.feeding_pause_clock = self.motor_sample_clock()[0]
        self.feeding_paused, self.freeze_targets = True, True
        self.mode = 'hold'
        self.bus.send_targets(self.targets, self.enabled)

    def resume_feeding(self):
        """保留真实反馈时间戳；暂停时长不算等待期限、不变成控制dt。"""
        if self.feeding is None:
            self.start_feeding()
            return
        if self.feeding.completed:
            print('本步已完成并保持；Ctrl+J/H执行下一段，Ctrl+Y手动返回本步起态。', flush=True)
            return
        self.refresh_compliance_feedback()
        now, _ = self.motor_sample_clock()
        core = self.feeding
        retrying = core.fault is not None
        if core.fault:
            core.events.append(dict(kind='manual_retry', time_s=now-self.feeding_clock_origin, previous_fault=core.fault))
            core.fault = None
            core.ready_since = core.untrusted_since = core.tracking_blocked_since = None
        if self.feeding_paused and self.feeding_pause_clock is not None:
            gap = max(0., now-self.feeding_pause_clock)
            if core.stage=='prepare':
                core.options = replace(core.options, preparation_s=core.options.preparation_s+gap)
            if core.stage_since is not None:
                core.stage_since += gap
            core.ready_since = core.untrusted_since = core.tracking_blocked_since = None
            if core.last_time is not None:
                core.last_time = now-self.feeding_clock_origin-CONTROL_PERIOD_S
            core.events.append(dict(kind='manual_resume', time_s=now-self.feeding_clock_origin, paused_s=gap))
        if retrying:
            core.stage_since = now-self.feeding_clock_origin
        self.feeding_paused, self.freeze_targets = False, False
        self.feeding_pause_clock = None
        self.mode = 'run'

    def force_feeding_step(self):
        """Ctrl+L：沿当前任务方向强制推进一小步，目标发完后保持。"""
        if self.feeding is None:
            self.start_feeding(running=False)
        self.refresh_compliance_feedback()
        now, feedback = self.motor_sample_clock()
        if feedback is None or not 0. <= now-feedback <= MAX_FEEDBACK_AGE_S:
            raise TimeoutError('强制步需要当前全关节角反馈')
        core = self.feeding
        measured = self.measured_angles()
        control_time = now-self.feeding_clock_origin
        if not core.start_forced_step(measured, FEED_FORCE_STEP_JOINTS, control_time):
            self.pause_feeding()
            print('强制步未推进：已到当前模板位置边界；可用反向J/H建立反向任务。', flush=True)
            return
        self.feeding_paused, self.freeze_targets = False, False
        self.feeding_pause_clock = None
        self.mode = 'run'
        step = core.forced_step
        print(f'强制步：位置参数 {step["from_position_joints"]:.4f} → '
              f'{step["to_position_joints"]:.4f}，绕过跟踪/稳定等待；目标发完后保持。', flush=True)

    def command_feeding(self, command):
        """J/H仍为整段任务；L沿任务方向强制推进，目标发完后保持。"""
        if command in ('step', 'back'):
            self.start_feeding(1 if command=='step' else -1)
        elif command in ('toggle', 'run', 'pause', 'hold'):
            if command in ('pause', 'hold') or command=='toggle' and not self.feeding_paused:
                self.pause_feeding()
            else:
                self.resume_feeding()
        elif command=='feed_tick':
            self.force_feeding_step()
        elif command=='feed_return':
            core = self.feeding
            if core is None or core.baseline is None:
                raise ValueError('尚未取得送链起态长度基线；可Ctrl+G暂停或Ctrl+X失能')
            core.completed, core.fault = False, None
            self.resume_feeding()
            now, _ = self.motor_sample_clock()
            core.stage, core.stage_since, core.ready_since = 'feed_return', now-self.feeding_clock_origin, None
            core.goal_m, core.completed, core.fault = core.baseline, False, None
            core.events.append(dict(kind='manual_return', time_s=core.stage_since))
        elif command in ('delta-', 'delta+'):
            self.feed_distance_m *= .5 if command=='delta-' else 2.
        elif command=='reverse':
            self.feed_direction *= -1
        elif command in ('compliance', 'window_auto'):
            print('主动送链独立于顺应；Ctrl+1/2/3可切回原模式。', flush=True)
        else:
            return False
        return True

    def tick_feeding(self, dt):
        core = self.feeding
        if self.feeding_paused or core.completed or core.fault:
            # Holding motor targets must not stop passive-shape observation.
            # Otherwise human-paced Ctrl+L (>0.6s between keys) could never
            # collect three consecutive samples, despite continuous full reads.
            now, feedback = self.motor_sample_clock()
            observer = core.length_observer
            sample_time = None if feedback is None else feedback-self.feeding_clock_origin
            if (feedback is not None and 0. <= now-feedback <= MAX_FEEDBACK_AGE_S and
                    (observer._last_feedback is None or sample_time>observer._last_feedback)):
                if core.stage=='prepare':
                    observer.cut_depth_m = None
                core.observation = observer.observe(self.measured_angles(), core.window,
                    sample_time)
            self.bus.send_targets(self.targets, self.enabled)
            return
        now, feedback = self.motor_sample_clock()
        if feedback is None:
            raise TimeoutError('主动送链需要包括失能关节在内的全角反馈')
        control_time, feedback_time = now-self.feeding_clock_origin, feedback-self.feeding_clock_origin
        if core.last_time==control_time:
            return
        if self.mode=='step' and core.last_time is not None:
            core.last_time = control_time-min(dt, CONTROL_PERIOD_S)
        measured = np.asarray(self.measured_angles())
        self.feeding_inputs = dict(angles_deg=measured.tolist(), feedback_time_s=feedback_time, control_time_s=control_time)
        before = core.stage
        forced_before = core.forced_step is not None
        if before=='prepare':
            # 实机可以从不同姿态启动。形成过程中不固化尚未建立的切片。
            core.length_observer.cut_depth_m = None
        target, mask = core.advance(measured, feedback_time, control_time)
        if before=='prepare' and core.stage=='feed_out':
            core.velocity_m_s = 0.
            core.previous_coordinate = core.observation['transfer_coordinate_m']
            core.last_length_time = feedback_time
        targets = np.degrees(target).tolist()
        for i in np.flatnonzero(mask):
            angle_to_raw(i, targets[i])
        if mask.tolist()!=self.enabled:
            entering = mask & ~np.asarray(self.enabled)
            self.bus.prepare_enabled(measured, mask, configure_mask=entering)
        self.targets, self.enabled = targets, mask.tolist()
        self.first, self.last = core.window
        self.length_normalized = True
        self.bus.send_targets(targets, self.enabled)
        self.phase = self.command_phase = 0.
        diagnostics_due = (self.feeding_last_diagnostic_s is None or
                           now-self.feeding_last_diagnostic_s >= FEED_DIAGNOSTIC_PERIOD_S)
        if core.stage!=self.feeding_last_stage or core.fault or diagnostics_due:
            self.feeding_last_stage = core.stage
            self.feeding_last_diagnostic_s = now
            self.show_status()
        if forced_before and core.forced_step is None:
            print('强制步目标已发完，保持；Ctrl+L再推进，Ctrl+G继续原任务，J/H开始相对任务。', flush=True)
            self.pause_feeding()
        elif self.mode=='step' or core.completed or core.fault:
            self.pause_feeding()

    def refresh_compliance_feedback(self):
        started = time.monotonic()
        sample_clock = time.perf_counter()  # 最早读角时刻，与控制时刻共用高精度计时器
        acquisition_phase = self.command_phase  # 扫读期间控制线程不改相位
        states = self.bus.read_states()
        if time.monotonic()-started > MAX_FEEDBACK_AGE_S:
            raise TimeoutError('全关节反馈读取过慢，最早样本已经过期')
        if set(states) != set(range(NUM_JOINTS)) or not all(
                math.isfinite(s['angle_deg']) for s in states.values()):
            raise RuntimeError('全关节位置反馈不完整或不是有限值')
        self.states = states
        self.last_feedback = {i: state['angle_deg'] for i, state in states.items()}
        self.feedback_motor_clock_s = sample_clock-self.motor_clock_origin_s
        self.feedback_wall_s = started  # 顺序扫读，按最早样本计龄
        self.feedback_phase_rad = acquisition_phase
        self.feedback_sequence += 1

    def motor_sample_clock(self):
        """控制时间/最早反馈时间共用单调时钟；MuJoCo子类改用仿真时钟。"""
        return time.perf_counter()-self.motor_clock_origin_s, self.feedback_motor_clock_s

    def section_reference(self):
        """扩缩后切回基础/固定模式，继续使用当前长度的半圆参考。"""
        return SectionReference(self.first, self.last, self.amplitude,
                                length_normalized=self.length_normalized, manual_tuning=DIY_PARAMETERS)

    def basic_targets(self):
        return np.degrees(self.section_reference().targets(self.phase)).tolist()

    def select_gait(self, kind):
        """模式4转3按实测重选；其他已使能模式保留当前窗口接入。"""
        if kind=='sliding' and self.mode=='released':
            self.refresh_compliance_feedback()
            audit={}
            try:
                selection=choose_posture_window(self.measured_angles(),LINK_LENGTHS_M,diagnostics=audit)
            except ValueError as exc:
                self.initial_window_selection=dict(accepted=False,diagnostics=audit,angles_deg=self.measured_angles())
                raise ValueError('自动初始选窗失败，保持失能（未使能）：'+str(exc)) from exc
            self.initial_window_selection=selection
            self.arm(window_override=tuple(selection['chosen']))
            self.pending_gait='sliding'
            best=selection['best']
            print(f'按实测姿态接管J{self.first}–J{self.last}（{sum(self.enabled)}关节）；'
                  f'截面转角={best["turn_deg"]:.1f}°，覆盖角={best["coverage_deg"]:.1f}°，'
                  f'下垂依据={best["end_source"]}。形成完成后进入自动滑窗顺应。',flush=True)
            return
        if kind=='feeding':
            self.select_feeding()
            return
        from_feeding = self.feeding_selected and kind=='sliding'
        if from_feeding:
            # Stop the old action before inspecting the pose. A rejected
            # selection retains its torque mask and holds, never silently
            # falls back to the feeding buffer as the rolling window.
            self.feeding_paused=True
            self.mode,self.pending_phase='hold',0.
            self.refresh_compliance_feedback()
            measured=self.measured_angles()
            audit={}
            try:
                focus=None
                if self.feeding is not None and hasattr(self.feeding,'profile'):
                    _,focus,_,_=self.feeding.profile(self.feeding.position,self.feeding.bias)
                selection=choose_feeding_arc(measured,LINK_LENGTHS_M,self.enabled,focus)
            except ValueError as exc:
                self.initial_window_selection=dict(accepted=False,diagnostics=audit,angles_deg=measured,from_mode='active_feeding')
                raise ValueError('模式4→3重选未完成，保留原使能并保持：'+str(exc)) from exc
            selected_window=tuple(selection['chosen'])
        else:
            selected_window=(self.first,self.last)
        if self.feeding_selected and not from_feeding:
            self.feeding_selected, self.feeding = False, None
            self.feeding_inputs = None
            self.mode, self.pending_phase = 'hold', 0.
            self.freeze_targets = False
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
        if not from_feeding and (self.mode == 'released' or self.forming or self.settle_remaining > 0):
            self.pending_gait = kind
            print(f'已选择{kind}，形成与静置完成后尝试接入。', flush=True)
            return
        if kind == 'fixed' and self.sliding is None:
            self.pending_gait = None
            if self.compliance is None:
                self.toggle_compliance()
            return
        if kind == 'sliding' and self.sliding is not None and not self.sliding_fault and not from_feeding:
            return
        if not from_feeding:
            self.refresh_compliance_feedback()
        measured = self.measured_angles()
        valid, geometry = self.check_actual_section(measured,selected_window)
        if not valid:
            raise ValueError('实际形态不满足开放圆弧条件，未切换：'+str(geometry))
        if kind == 'fixed':
            reference = self.section_reference()
            core = FixedSectionCompliance(reference, LINK_LENGTHS_M, self.options)
            core.targets = np.radians(self.targets)
            core.next_geometry_s = self.section_clock_s
            self.sliding, self.compliance = None, core
        else:
            if not DIY_PARAMETERS and (self.amplitude != 60. or self.omega != .6 or self.direction != 1):
                raise ValueError('冻结角度基准只标定了amplitude=60 omega=0.6正向；先在基础模式恢复')
            if not DIY_PARAMETERS and not np.allclose(LINK_LENGTHS_M, MOTOR_PHASE_REFERENCE['metadata']['link_lengths_m'], rtol=0., atol=1e-9):
                raise ValueError('实际连杆尺寸与相位基准不一致，需要重新标定')
            now, _ = self.motor_sample_clock()
            parameter_type = ManualArcReference if DIY_PARAMETERS else HalfArcParameters
            p = parameter_type(joints=NUM_JOINTS, first_joint=selected_window[0], last_joint=selected_window[1],
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
            slider.targets = np.radians(measured if from_feeding else self.targets)
            slider.core.targets = slider.targets.copy()
            slider.core.next_geometry_s = now
            slider.cooldown_until = now
            slider.events.append(dict(kind='acquire', time_s=now, to=list(slider.window),
                                      reason='remeasured_after_feeding' if from_feeding else 'take over current window; angle-only posture and length selection'))
            if from_feeding:
                enabled=slider.enabled
                entering=enabled & ~np.asarray(self.enabled,bool)
                for i in np.flatnonzero(enabled):
                    angle_to_raw(i,measured[i])
                # No all-off interval. Preload entering joints from actual
                # angles, then apply exactly the newly measured rolling mask.
                self.touched_hardware=True
                self.bus.prepare_enabled(measured,enabled,configure_mask=entering)
                selection.update(from_mode='active_feeding',previous_window=[self.first,self.last])
                self.initial_window_selection=selection
                self.first,self.last=selected_window
                self.enabled=enabled.tolist()
                self.targets=list(measured)
                self.feeding_selected,self.feeding=False,None
                self.feeding_inputs=None
                self.forming,self.settle_remaining=False,0.
                self.command_phase=self.phase
                print(f'模式4→3已按实测重选：J{self.first}–J{self.last}，{sum(self.enabled)}关节；'
                      '已退出送链缓冲使能，保持当前姿态，Ctrl+G开始滚动。',flush=True)
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
        if slider.last_control_time == control_time:
            # 同时刻重复调用（例如暂停的仿真）：不消耗单步相位、不重复交接。
            slider.fresh_feedback = False
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
                if not valid and not DIY_PARAMETERS:
                    raise ValueError('新实测形态超出开放圆弧/平面范围 '+str(self.actual_section))
            target, enabled = slider.advance(measured, feedback_time, control_time,
                commanded_phase_rad=self.phase, phase_at_feedback=self.feedback_phase_rad,
                freeze_targets=self.freeze_targets)
            expected_mask = (np.arange(1, NUM_JOINTS+1) >= slider.window[0]) & (np.arange(1, NUM_JOINTS+1) <= slider.window[1])
            if (not np.array_equal(enabled, expected_mask)
                    or not parameter_window_feasible(slider.p, LINK_LENGTHS_M, slider.window, slider.posture_options.nominal_joint_margin_rad)
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

    def check_actual_section(self, measured, window=None):
        """只评价实测角重建的身体形态，不识别管道支撑或绝对对地朝向。"""
        try:
            first,last=(self.first,self.last) if window is None else window
            g = local_section(np.radians(measured), LINK_LENGTHS_M, first,last)
            points = g['points'][1:-1]
            length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            result = dict(turn_deg=math.degrees(g['total_turn']),
                          gap_fraction=float(np.linalg.norm(points[-1]-points[0])/length),
                          fit_rms_m=g['fit_rms'])
            valid = (SECTION_MIN_ACTUAL_TURN_DEG <= result['turn_deg'] <= SECTION_MAX_ACTUAL_TURN_DEG
                     and result['gap_fraction'] >= self.options.minimum_gap_fraction
                     and result['fit_rms_m'] <= SECTION_MAX_PLANE_RMS_M)
            if DIY_PARAMETERS:
                valid = True
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
                raise ValueError('实测截面生成的目标不满足开放约束')
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

    def arm(self, window_override=None):
        if self.mode != 'released':
            return              # 重复 apply 不重新初始化，不把机器人拉回相位 0
        self.first, self.last = self.configured_window if window_override is None else window_override
        self.feeding_origin = (self.first,self.last)
        if window_override is None:
            self.initial_window_selection=None
        self.length_normalized = self.last-self.first+1!=6
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
        self.helix_transition = None
        self.feeding_selected, self.feeding = False, None
        self.feeding_inputs = None
        self.sliding, self.pending_gait = None, None
        self.sliding_fault = False
        self.compliance = None
        self.mode, self.pending_phase = 'released', 0.0
        self.enabled = [False] * NUM_JOINTS
        self.touched_hardware = True
        self.bus.set_enabled(self.enabled, verify=verify)
        self.first, self.last = self.configured_window
        self.feeding_origin = self.configured_window
        self.length_normalized = self.last-self.first+1!=6
        self.freeze_targets = False

    def change_parameters(self, amplitude=None, omega=None):
        values = []
        if amplitude is not None:
            values.append(f'amplitude={amplitude}')
        if omega is not None:
            values.append(f'omega={omega}')
        self.assign_parameters(' '.join(values))

    def assign_parameters(self, command):
        """先验证整行，再一起更新；固定窗口只在release后允许人工修改。"""
        if command.startswith('feed_mm='):
            value = float(command.split('=', 1)[1])/1000.
            if not math.isfinite(value) or value==0:
                raise ValueError('feed_mm需要有限非零的毫米数')
            self.feed_distance_m, self.feed_direction = abs(value), 1 if value>0 else -1
            return
        if not DIY_PARAMETERS and isinstance(self.sliding, MotorFeedbackSliding):
            for item in command.split():
                key, value = item.split('=', 1)
                if key in ('amplitude', 'window') or (key == 'omega' and float(value) != .6):
                    raise ValueError('自动滑窗使用60度缩放/0.6正向相位参考；改幅值/频率/窗口先切回基础模式')
        amplitude, omega, delta = self.amplitude, self.omega, self.phase_step
        first, last, options = self.first, self.last, self.options
        option_keys = {'gain': 'gain', 'filter': 'target_filter_s'}
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
            elif key == 'yielding':
                raise ValueError('yielding恢复权重已取消；顺应直接使用实测截面，只需调整gain/filter')
            elif key in option_keys:
                options = replace(options, **{option_keys[key]: float(value)})
            else:
                raise ValueError('参数：amplitude omega delta window gain filter；本版不用support/load')
        validate_parameters(amplitude, omega, first, last)
        normalized = last-first+1 != 6 if self.mode == 'released' else self.length_normalized
        validate_arc_reference(amplitude, omega, first, last, normalized)
        if not math.isfinite(delta) or not 0 < delta <= math.pi:
            raise ValueError('单步相位增量必须在(0,pi] rad内')
        if (first, last) != (self.first, self.last) and self.mode != 'released':
            raise ValueError('手动修改window前先release')
        if self.compliance is not None:
            reference = replace(self.compliance.p, amplitude_deg=amplitude)
        if amplitude != self.amplitude and self.mode != 'released' and self.compliance is None and not self.feeding_selected:
            self.forming, self.settle_remaining = True, SETTLE_S
        self.amplitude, self.omega, self.phase_step = amplitude, omega, delta
        self.first, self.last, self.options = first, last, options
        if self.mode == 'released':
            self.length_normalized = last-first+1 != 6
            self.feeding_origin = (first, last)
            self.configured_window = (first, last)
        if self.compliance is not None:
            self.compliance.p = reference
            self.compliance.options = options
            self.compliance.next_geometry_s = self.section_clock_s
            if self.sliding is not None:
                self.sliding.section_options = options
                if DIY_PARAMETERS:
                    self.sliding.p = replace(self.sliding.p, amplitude_deg=amplitude, omega_rad_s=omega,
                                             first_joint=first, last_joint=last)
                    self.compliance.p = self.sliding.p
                    self.sliding.pending = None
                    self.sliding._new_observer()  # 参数改变后重新建立对应的形状参考。

    def command(self, command):
        command = command.strip().lower()
        if command == 'gait_gravity':
            self.select_gravity()
            return
        if command == 'gait_helix':
            if isinstance(self.helix_transition, GravityCrossing):
                raise ValueError('当前为 T 管越障，结束时用 Ctrl+X')
            self.select_helix()
            return
        if self.helix_transition is not None and self.command_helix(command):
            return
        if self.feeding_selected and self.command_feeding(command):
            return
        if not DIY_PARAMETERS and isinstance(self.sliding, MotorFeedbackSliding) and (command in ('reverse', 'back') or
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
        if self.helix_transition is not None:
            self.tick_helix(dt)
            return
        if self.feeding_selected and self.feeding is not None:
            self.tick_feeding(dt)
            return
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
        gait = ('gravity_crossing' if isinstance(self.helix_transition, GravityCrossing) else
                'arc_to_helix' if self.helix_transition is not None else
                'active_feeding' if self.feeding_selected else 'sliding_section_compliance' if self.sliding is not None else
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
                    helix_transition=None if self.helix_transition is None else self.helix_transition.status(),
                    feeding=None if self.feeding is None else copy.deepcopy(self.feeding.snapshot()),
                    feeding_motor_input=copy.deepcopy(self.feeding_inputs),
                    feeding_action_id=self.feeding_action_id, feed_requested_m=self.feed_direction*self.feed_distance_m,
                    configured_window=list(self.configured_window), initial_window_selection=copy.deepcopy(self.initial_window_selection),
                    feeding_paused=self.feeding_paused,
                    sliding=None if self.sliding is None else copy.deepcopy(self.sliding.snapshot()),
                    sliding_events=[] if self.sliding is None else copy.deepcopy(self.sliding.events),
                    servo_feedback_raw=self.states,
                    measured_controller_deg={f'J{i+1}': v for i, v in self.last_feedback.items()})

    def show_status(self, include_feedback=False):
        label = {'released': '未接管/已释放', 'hold': '保持', 'run': '连续', 'step': '单步'}[self.mode]
        if self.feeding_selected and self.mode=='run':
            label = '执行本步'
        if self.forming:
            label += '，正在形成基础圆弧'
        elif self.settle_remaining > 0:
            label += f'，静置余{self.settle_remaining:.1f}s'
        gravity = isinstance(self.helix_transition, GravityCrossing)
        gait = ('T 管重力矩越障' if gravity else 'ARC→Helix' if self.helix_transition is not None else
                '主动送链' if self.feeding_selected else '角度反馈滑窗顺应' if self.sliding is not None else
                '基础圆弧' if self.compliance is None else '固定截面顺应')
        amplitude = (FEED_AMPLITUDE_DEG if self.feeding is None else self.feeding.options.amplitude_deg) if self.feeding_selected else self.amplitude
        print(f'{gait} | {label} | J{self.first}–J{self.last} | A={amplitude:g}° '
              f'omega={self.omega:g}rad/s | phase={self.phase:.4f}rad | delta={self.phase_step:g}rad', flush=True)
        if gravity:
            h = self.helix_transition.status()
            instruction = {
                'front': '先头部绕上主管；尾部保持失能。头部贴稳后暂停并按 Ctrl+L',
                'first_shift': 'Shift Rolling 逐节接入尾部',
                'first_recover': '收紧螺旋，继续滚动至对准阶段',
                'align': 'Ctrl+J/H 微调。确认前部过支管、螺旋已对准后按 Ctrl+L',
                'unwind': '形成右侧小 U；从尾朝头观察，确认 U 搭稳后暂停并按 Ctrl+L',
                'right_u_hold': '确认小 U 搭在主管右侧，前部贴稳后按 Ctrl+L，再用 Ctrl+J/G 释放尾段',
                'drop': '尾段关闭扭矩，自然下落',
                'drop_hold': '观察尾段是否落向轴线下方。确认后 Ctrl+L，再用 Ctrl+J/G 回卷',
                'rewrap': 'Shift Rolling 回卷尾部并继续推进前部',
                'recover': '回卷接收完成，恢复常速滚动',
                'roll': '继续滚动，现场观察是否整身清障',
            }.get(h['stage'], '查看调长状态和反馈')
            print(f"阶段={h['stage']} | 曲线进给={1000*h['shift_m']:.1f} mm | "
                  f"已接收边界={h['accepted_joint']} | 相位={h['phase_rad']:.3f} rad | {instruction}", flush=True)
            if h['front'] and h['stage'] in ('front', 'blocked'):
                front = h['front']
                print(f"头部接入={front['stage']} | 调长阶段={front['transfer_stage']} | "
                      f"圆弧窗口={front['arc_window']} | 最近事件={front['events'][-1]}", flush=True)
        elif self.helix_transition is not None:
            h = self.helix_transition.status()
            head, tail = h['entry_end_lengths_m']
            head_depth, tail_depth = h['entry_end_depths_m']
            print(f'起态头/尾侧长度估计 {1000*head:.0f}/{1000*tail:.0f} mm | '
                  f'端点距截面圆心深度 {1000*head_depth:.0f}/{1000*tail_depth:.0f} mm', flush=True)
            print(f"阶段={h['stage']} | 起卷端={h['leading_end']} | 已接收边界={h['accepted_joint']} | "
                  f"曲线移位={h['shift_m']*1000:.1f} mm | 调长阶段={h['transfer_stage']}", flush=True)
            if h['stage'] in ('lead', 'receive'):
                print(f"分段进给 起卷侧/接收侧={h['shift_m']*1000:.1f}/{h['receive_shift_m']*1000:.1f} mm", flush=True)
            if h['stage'] == 'blocked':
                print(f"停在当前目标：{h['events'][-1]['reason']}", flush=True)
        if self.feeding_selected:
            core = self.feeding
            o = {} if core is None else core.observation or {}
            if core is not None and core.forced_step is not None:
                print('强制步正在下发目标：绕过跟踪误差/稳定采样等待，发完后保持。', flush=True)
            delta = None if core is None or core.baseline is None or not o.get('valid') else 1000*(o['transfer_coordinate_m']-core.baseline)
            print(f'下一任务={1000*self.feed_direction*self.feed_distance_m:+g}mm | '
                  f'阶段={"就绪" if core is None else core.stage} | 几何长度代理变化={delta}mm | '
                  f'目标={None if core is None or core.goal_m is None else 1000*(core.goal_m-core.baseline)}mm | '
                  f'暂停={self.feeding_paused} | 原因={None if core is None else core.fault or core.acquisition_reason or o.get("reason")}', flush=True)
            if core is not None and core.stage=='acquire':
                m=core.acquisition_metrics
                print(f'接管展直中，尚未开始计量 | 跟踪误差={core.tracking_error_deg:.2f}° | '
                      f'稳定采样={len(core.acquisition_samples)} | '
                      f'稳定跨度={m.get("stable_span_s",0.):.2f}s | '
                      f'角度波动={m.get("angle_span_deg")}° | 长度代理波动={m.get("length_span_mm")}mm', flush=True)
            for line in feeding_direction_lines(o):
                print(line, flush=True)
        if self.compliance is not None:
            window_state = '窗口固定' if self.sliding is None else f'自动选窗={self.sliding.automatic} 待交接={self.sliding.pending is not None}'
            print(f'实测截面直接顺应 K={self.options.gain:g} | '
                  f'{window_state} | 阻塞原因={self.section_block_reason or "无"}', flush=True)
        if self.sliding is not None and self.sliding.observation is not None:
            o = self.sliding.observation
            print(f'角度观察器={o["state"]} 建议={o["suggested_window"]} '
                  f'不可观测原因={o.get("reasons", [])}', flush=True)
        if isinstance(self.sliding, PostureWindow):
            s = self.sliding.selection or {}
            count = self.sliding.count_observation or {}
            phase_stats = count.get('phase_statistics', {})
            print(f'使能数量={sum(self.enabled)}，数量估计={self.sliding.filtered_count} '
                  f'决策={s.get("state", "收集相位")}，候选={s.get("chosen")}，'
                  f'原因={s.get("length_reason", count.get("reason", "收集实测角与相位"))}', flush=True)
            print(f'数量观测={count.get("reason", "可靠" if count.get("valid") else "等待实测角")} '
                  f'相位分组样本={phase_stats.get("group_sample_counts", [])} '
                  f'位置依据={s.get("position_evidence") or (self.sliding.observation or {}).get("end_quality", {}).get("source", "暂无")}', flush=True)
            if self.sliding.pending is not None:
                gate = self.sliding.last_gate or {}
                print(f'交接目标={self.sliding.pending["to"]} '
                      f'等待条件={gate.get("waiting_reasons", ["等待下一帧实测角"])}', flush=True)
        if include_feedback:
            print('实测角：'+' '.join(f'J{i+1}={v:.1f}°' for i, v in self.last_feedback.items()), flush=True)
            print('原始负载/电流（只记录）：'+' '.join(
                f'J{i+1}={s["load_raw"]}/{s["current_raw"]}' for i, s in self.states.items()), flush=True)


def feeding_direction_lines(observation):
    """同一份数值供实机/仿真控制台使用；L为连杆号，J为关节号。"""
    quality = observation.get('end_quality') or {}
    if not quality.get('ends'):
        return ['向下估计：等待全关节新反馈']
    thresholds = quality['thresholds']
    reason_text = dict(not_straight='直线度不足', tip_not_below_arc='端点下垂距离不足',
        segment_near_crest='区段靠近顶部', collecting_stability='收集稳定样本', shape_unstable='局部角仍在变化')
    lines = []
    for side, label in (('head','头侧'),('tail','尾侧')):
        end = quality['ends'][side]
        c = end['selected'] or end['best']
        if c is None:
            lines.append(f"{label}：失能连杆仅{end['available']}段，至少需要{thresholds['minimum_links']}段")
            continue
        drift = '--' if c['shape_change_deg'] is None else f"{c['shape_change_deg']:.2f}"
        reason = ' / '.join(reason_text.get(r,r) for r in c['reasons']) or '可用'
        lines.append(f"{label}：{'选中' if end['selected'] else '候选'}L{c['links'][0]}–L{c['links'][1]} "
            f"(内部J{c['internal_joints'][0]}–J{c['internal_joints'][1]}) "
            f"直线度={c['straightness']:.4f}/{thresholds['minimum_straightness']:.2f}；"
            f"稳定={c['stable_span_s']:.2f}/{thresholds['stability_s']:.2f}s ({c['samples']}样本)；"
            f"局部角变化={drift}/{thresholds['maximum_shape_change_deg']:g}°；{reason}")
        lines.append(f"  下垂端距估计顶部={1000*c['tip_below_global_top_m']:.1f}mm，"
            f"区段距估计顶部={1000*c['segment_below_global_top_m']:.1f}mm；"
            f"合格区段={end['stable_count']}/{end['tested_count']}；"
            + ' '.join(f"{reason_text.get(k,k)}:{v}" for k,v in end['rejection_counts'].items()))
    angle = quality.get('end_disagreement_deg')
    disagreement = '--' if angle is None else f'{angle:.2f}°'
    lines.append(f"向下估计={quality['source']}；两侧夹角={disagreement}/"
        f"{thresholds['maximum_end_disagreement_deg']:g}°；"
        + ('方向不一致，使用单侧' if quality.get('fusion_rejected_reason') else '仅角度几何估计'))
    if 'reference_event' in quality:
        event={'retained':'沿用合格参考','overlap_handover':'同帧重建交接',
            'reacquire_no_overlap':'无共同参考，重新标定','reacquire_after_gap':'缺样后重新标定'}
        lines.append(f"参考交接={event.get(quality['reference_event'],quality['reference_event'])}；"
            f"基准序号={quality['reference_epoch']}；同帧校正={1000*quality['handover_correction_m']:.2f}mm；"
            f"累计坐标补偿={1000*quality['coordinate_offset_m']:.2f}mm")
    return lines


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

    def update(self, angles_deg, window, feedback_time_s, hint=None):
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
        # A controller may already have qualified a straight passive subspan
        # (the distal tip itself can be bent). Copy that angle-only observation;
        # never ask the controller to run or change its state for the preview.
        hint_used = False
        if hint:
            direction = np.asarray(hint['down_head'], float)
            if direction.shape == (3,) and np.isfinite(direction).all() and np.linalg.norm(direction) > .5:
                raw = gravity_unit(direction)
                hint_used = True
        reliable = raw is not None
        if raw is not None:
            state = 'observer_'+hint['source'] if hint_used else 'dual' if len(usable) == 2 else 'single_degraded'
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


def preview_section_geometry(geometry):
    """只复制控制器已有的线/圆弧与截面坐标；采样留给显示进程。"""
    if geometry is None:
        return None
    return dict(origin=geometry['axis']['point'].tolist(),
                basis=geometry['axis']['basis'][:, :2].tolist(),
                segments=[dict(start=s.start.tolist(), tangent=s.tangent.tolist(),
                               length=s.length, radius=s.radius, turn=s.turn,
                               joint=s.joint, kind=s.kind) for s in geometry['curve']])


def preview_smooth_points(geometry):
    """把控制截面的采样点还原到头部坐标，再交由共同的显示投影处理。"""
    segments = [Segment(np.array(s['start']), np.array(s['tangent']), s['length'],
                        s['radius'], s['turn'], s['joint'], s['kind']) for s in geometry['segments']]
    return sample_section(segments)@np.array(geometry['basis']).T+np.array(geometry['origin'])


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
    feeding = controller.feeding
    if controller.feeding_selected:
        observation = {} if feeding is None else feeding.observation or {}
    core = controller.compliance
    section = None if core is None else dict(
        source='control', window=[core.first, core.last], updates=core.updates,
        measured=preview_section_geometry(core.geometry),
        candidate=preview_section_geometry(core.target_geometry))
    sample_time = controller.motor_sample_clock()[1]
    observed_time = (getattr(feeding, 'last_feedback', None) if controller.feeding_selected else
                     observation.get('feedback_time_s'))
    hint = None
    if (observation.get('valid') and observation.get('down_head') is not None and
            observed_time is not None and observed_time == sample_time):
        hint = dict(down_head=list(observation['down_head']),
                    source=observation.get('end_quality', {}).get('source', 'passive_end'))
    return dict(angles_deg=angles, target_deg=target, enabled=enabled,
        smooth_section=section, gravity_hint=hint,
        gravity_time_s=sample_time,
        enable_known=controller.touched_hardware, feedback_wall_s=controller.feedback_wall_s,
        window=[controller.first, controller.last], phase_rad=controller.phase,
        mode=controller.mode,
        gait='T 管重力矩越障' if isinstance(controller.helix_transition, GravityCrossing) else 'ARC→Helix' if controller.helix_transition is not None else '主动送链' if controller.feeding_selected else '自动位置/数量' if slider else '固定顺应' if controller.compliance else '基础圆弧',
        feeding=None if feeding is None else copy.deepcopy(feeding.snapshot()),
        automatic=None if slider is None else slider.automatic,
        selection=copy.deepcopy(selection), pending=copy.deepcopy(getattr(slider, 'pending', None)),
        observer_state=observation.get('state', '等待反馈'),
        reasons=observation.get('reasons', []), block_reason=controller.section_block_reason,
        lengths=list(LINK_LENGTHS_M))


class JointPreviewFigure:
    """全身投影到实测截面；仅在该平面内旋转，使估计重力朝屏幕下方。"""
    def __init__(self):
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        self.plt = plt
        self.figure = plt.figure(figsize=(14, 8), facecolor='#f4f6fa')
        self.figure.canvas.manager.set_window_title('Rolling Arc — 实测构型与使能预览')
        grid = self.figure.add_gridspec(3, 2, width_ratios=[1.3, 1], height_ratios=[1, 1, .45])
        self.body = self.figure.add_subplot(grid[:, 0])
        self.angles = self.figure.add_subplot(grid[:2, 1])
        self.mask = self.figure.add_subplot(grid[2, 1])
        self.header = self.figure.suptitle('等待实测角度；预览不会使能或驱动舵机', fontsize=12)
        self.note = self.figure.text(.04, .04, '', fontsize=10, va='bottom')
        self.figure.subplots_adjust(left=.05, right=.97, top=.88, bottom=.24, wspace=.24, hspace=.65)
        self.section_origin = self.section_basis = None
        self.gravity = None
        self.gravity_status = '等待重力估计'
        self.body_points = self.target_points = None
        self.smooth_display = {}
        self.latest_row = None

    def update(self, row):
        self.latest_row = row
        self.body.clear(); self.angles.clear(); self.mask.clear()
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
        self.note.set_text('关节重建截面平面；目标与实测共用实测截面坐标。\n'
            f"自动={row['automatic']} | 决策={decision} {direction} | 数量估计={count_text}\n"
            f"状态：{row['block_reason'] or ', '.join(row['reasons']) or '正常'}；关闭此窗口不停止控制，Ctrl+X失能。")
        if row.get('feeding') is not None:
            f = row['feeding']
            o = f.get('observation') or {}
            delta = None if f['baseline_m'] is None or not o.get('valid') else 1000*(o['transfer_coordinate_m']-f['baseline_m'])
            self.note.set_text(f"主动送链：{f['stage']} | 目标={None if f['goal_delta_m'] is None else 1000*f['goal_delta_m']}mm\n"
                f"几何长度代理变化={delta}mm | 实测支撑/管轴不可得\n"
                f"原因：{f['fault'] or f.get('acquisition_reason') or o.get('reason') or '无'}；Ctrl+G暂停/继续，Ctrl+Y返回，Ctrl+X失能。")
        if row['angles_deg'] is not None:
            smooth_note = self.draw_section(row, colours, stale)
            self.note.set_text(self.note.get_text()+'\n'+smooth_note)
            self.angles.plot(ids, row['angles_deg'], 'o-', color='#17354b', label='实测')
            active = np.asarray(row['enabled'], bool)
            targets = np.where(active, row['target_deg'], np.nan)
            self.angles.plot(ids, targets, 's--', color='#e38b27', label='使能关节目标')
            self.angles.legend(loc='upper right')
            error = np.abs(np.asarray(row['angles_deg'])-np.asarray(row['target_deg']))[active]
            self.angles.set_title('使能关节最大角差：'+(f'{error.max():.2f}°' if len(error) else '无使能关节'), fontsize=11)
        else:
            self.body.text(.05, .5, '等待全20关节实测角', transform=self.body.transAxes)
        self.body.set(xlabel='截面水平 / m', ylabel='截面向上 / m',
                      title='全身关节重建截面（蓝：使能；紫：失能）')
        self.body.set_aspect('equal', adjustable='box')
        self.body.grid(alpha=.2)
        self.angles.set(ylabel='控制器坐标角度 / °', xticks=ids, ylim=(-100, 100))
        self.angles.grid(alpha=.2)
        self.figure.canvas.draw_idle()

    def orient_section(self, row, measured_basis, points, stale):
        """保留截面平面，消除拟合基底的任意转角与法向正负翻转。"""
        if self.gravity is None:
            self.gravity = DualEndGravityPreview(row['lengths'])
        if not stale and row['enable_known'] and any(row['enabled']) and row['gravity_time_s'] is not None:
            result = self.gravity.update(row['angles_deg'], row['window'], row['gravity_time_s'],
                                         hint=row.get('gravity_hint'))
        else:
            result = self.gravity.result
        normal = gravity_unit(np.cross(measured_basis[:, 0], measured_basis[:, 1]))
        previous = self.section_basis
        if previous is not None and normal @ np.cross(previous[:, 0], previous[:, 1]) < 0:
            normal = -normal
        down = result.get('down_head')
        projected = None if down is None else np.asarray(down)-normal*(normal@down)
        aligned = projected is not None and np.linalg.norm(projected) >= .15
        if aligned:
            up = -gravity_unit(projected)
            source = result['state']
            self.gravity_status = ('重力对齐：双侧融合' if source == 'dual' else
                '重力对齐：单侧暂用' if source == 'single_degraded' else
                '重力对齐：双侧被动段融合' if source.startswith('observer_') and 'two_consistent' in source else
                '重力对齐：单侧被动段暂用' if source.startswith('observer_') else
                '重力不可靠：保留上次方向')
        else:
            # 无可靠重力时不随SVD基底转动。第一次用开口定向只是显示约定。
            if previous is not None:
                up = previous[:, 1]-normal*(normal@previous[:, 1])
            else:
                first, last = row['window']
                arc = points[first:last+2]
                up = arc.mean(axis=0)-(arc[0]+arc[-1])*.5
                up -= normal*(normal@up)
            if np.linalg.norm(up) < 1e-8:
                up = measured_basis[:, 1]
            up = gravity_unit(up)
            self.gravity_status = ('重力近垂直于截面：保持方向' if down is not None else
                                   '重力未确定：保持显示方向（首次按圆弧开口定向）')
        if stale:
            self.gravity_status = '反馈陈旧：方向保持，等待新实测角'
        horizontal = gravity_unit(np.cross(up, normal))
        if previous is None and (points[-1]-points[0])@horizontal < 0:
            horizontal = -horizontal  # 首次约定头端在左；不声称观测到世界水平朝向。
        return np.column_stack((horizontal, up))

    def draw_section(self, row, colours, stale):
        """所有关节共用实测截面；局部控制平滑曲线保持原数据，仅变换显示坐标。"""
        window = row['window']
        self.smooth_display = {}
        section = row.get('smooth_section')
        try:
            if section is None:
                # 基础模式使用完全相同的关节重建方法，明确不是控制缓存。
                q = np.radians(row['angles_deg'])
                arc = forward(q, row['lengths'])[window[0]:window[1]+2]
                if np.linalg.svd(arc-arc.mean(axis=0), compute_uv=False)[1] < 1e-7:
                    raise ValueError('圆弧尚未形成，不能确定截面平面')
                section = dict(source='reconstructed', window=window,
                    measured=preview_section_geometry(local_section(q, row['lengths'], *window)),
                    candidate=preview_section_geometry(local_section(np.radians(row['target_deg']), row['lengths'], *window)))
            if list(section['window']) != list(window) or section.get('measured') is None:
                self.body.text(.08, .5, '等待控制器更新截面', transform=self.body.transAxes)
                return '平滑截面：等待控制器更新，不使用其他平面替代。'
            measured = section['measured']
            self.section_origin = origin = np.array(measured['origin'])
            full = forward(np.radians(row['angles_deg']), row['lengths'])
            basis = self.orient_section(row, np.array(measured['basis']), full, stale)
            self.section_basis = basis
            self.body_points = points = (full-origin)@basis
            self.target_points = targets = (forward(np.radians(row['target_deg']), row['lengths'])-origin)@basis
            # 22个节点 = 头端点、J1..J20、尾端点；不再按使能窗口裁剪。
            for i in range(len(points)-1):
                self.body.plot(*points[i:i+2].T, color=colours[max(0, i-1)], lw=1.5, alpha=.35 if stale else .7)
            self.body.plot([], [], color='#147bb5', lw=2, label='使能实测')
            self.body.plot([], [], color='#9259b5', lw=2, label='失能实测')
            self.body.plot(*targets.T, '--', color='#e38b27', lw=1.3, alpha=.65, label='已发目标连杆')
            self.body.scatter(*points[1:-1].T, c=colours, s=24, zorder=4)
            for i in range(1, NUM_JOINTS+1):
                self.body.annotate(f'J{i}', points[i], xytext=(4 if i % 2 else -22, 3),
                                   textcoords='offset points', fontsize=8, color=colours[i-1])
            for i, label in ((0, '头端'), (-1, '尾端')):
                self.body.annotate(label, points[i], xytext=(4, -9), textcoords='offset points', fontsize=8)
            used = section['source'] == 'control'
            for key, color, style, label in (
                ('measured', '#008d82', '-', '实测平滑（控制输入）' if used else '实测平滑（重建）'),
                ('candidate', '#c33786', '--', '顺应目标平滑（滤波前）' if used else '已发目标平滑（重建）')):
                if section.get(key) is not None:
                    curve = (preview_smooth_points(section[key])-origin)@basis
                    self.smooth_display[key] = curve
                    self.body.plot(*curve.T, style, color=color, lw=2.3, alpha=.4 if stale else 1., label=label)
            self.body.legend(loc='best', fontsize=8)
            both = np.concatenate([points, targets]+list(self.smooth_display.values()))
            center = (both.min(axis=0)+both.max(axis=0))/2
            radius = max(.04, float(np.ptp(both, axis=0).max())*.65)
            self.body.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius))
            if self.gravity.result.get('down_head') is not None:
                self.body.annotate('', xy=(.94, .71), xytext=(.94, .88), xycoords='axes fraction',
                                   arrowprops=dict(arrowstyle='->', color='#33485b', lw=1.8))
                self.body.text(.92, .9, '估计重力', transform=self.body.transAxes, ha='right', fontsize=9)
            plane_normal = np.cross(basis[:, 0], basis[:, 1])
            off_plane = float(np.max(np.abs((full-origin)@plane_normal)))
            return (self.gravity_status+f'；全身投影，离截面最大 {off_plane*1000:.1f}mm。\n'+
                    (f"控制实测截面 #{section.get('updates', '?')}；洋红：滤波前候选，橙色：已发目标。"
                    if used else '送链预览：平滑曲线仅作显示；长度由连杆几何估计。'
                    if row.get('feeding') is not None else '平滑截面：同法重建，当前未参与顺应控制。'))
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.body.text(.08, .5, '等待有效的关节重建截面', transform=self.body.transAxes)
            return '截面不可用：'+str(exc)


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
               49: 'gait_basic', 50: 'gait_fixed', 51: 'gait_sliding', 52: 'gait_feeding', 53: 'gait_helix', 54: 'gait_gravity',
               97: 'gait_basic', 98: 'gait_fixed', 99: 'gait_sliding', 100: 'gait_feeding', 101: 'gait_helix', 102: 'gait_gravity'}
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
    full_feedback = (controller.compliance is not None or controller.feeding_selected or
                     controller.helix_transition is not None or controller.preview_feedback)
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
            if log_file and (controller.feeding_selected or elapsed >= next_log):
                record = dict(time_s=elapsed, execution='dry_run' if dry_run else 'serial', **controller.snapshot())
                log_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                log_file.flush()
                next_log = elapsed + (CONTROL_PERIOD_S if controller.feeding_selected else .1)
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
