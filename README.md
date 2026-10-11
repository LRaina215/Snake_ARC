# Snake_ARC

圆弧滚动管外运动研究。当前新增局部实际形状截面顺应与滑动窗口仿真：固定6关节、整周期承重定位、卸载相位交接，其余关节主动失能。已完成有限实验计划，包含110→210 mm连续90°急弯变径，全身通过且观察内未累计收紧。急弯严格管轴覆盖、4 N载荷触发迁移仍有未通过项，详见逐轮报告。实机代码仍为旧版本，等待用户验收后再移植。

参考 `Snake_Robot_Gait` 的目录结构：`codes/` 为入口，`snake_control/` 为共享控制模块，`mujoco_model/` 为模型，`tools/` 为模型构建工具，`validation/` 为验证，`configs/` 为参数，`docs/` 为交接和复用说明。

Step1 默认 20 关节、固定使能 J8–J13。首版 Step2 根据支撑观测接管和迁移局部窗口，并从实测角调整滚动幅值；窗口外关闭主动驱动并保留机械阻尼。弯管仍使用同一滚动公式，没有步态转换或根部位姿牵引，但部分变径案例的实际形态偏离了开放圆弧目标。

## 滑动窗口截面顺应：当前入口（2026-10-12）

实际关节角重建局部截面并反馈顺应；窗口依据实际形态和MuJoCo向上接触承重选取，迁移在同奇偶族接近零且实际卸载后进行，任一时刻只使能6关节。按整周期F·dφ定位排除正常滚动中的承重点摆动；控制器不读取管道半径、中心线。等待卸载会暂时停相位，物理和顺应继续，没有改变步态或牵引根位姿。

本轮共17次动力学运行：初始11次＋交接修复3次＋出口实际周期计时修复1次＋周期定位修复/回归2次。失败完整保留，每轮审查全采样、全程时序侧/端视和最差状态。固定6关节在210 mm出口仍保持承重与开放，未触发永久扩窗；临时6→5→6交接尝试失败，最终采用6→6卸载交接。

最终大变径急弯只发生1次自动交接，出口后三个实际周期不再往返切窗，身体转角160.93–186.97°。拐角管轴包绕最低142.84°且有47个不可观测采样，完整验收仍未通过。4 N实际载荷回归保持承重/形态但未触发迁移，原测试项保留未覆盖。基础对照也完成运动与开放，尚无性能优势证据；大管径持续触及限矩。最终周期定位器只运行上述2例，早期其余病例不是最终定位器的全矩阵回归。

[实验复盘和视频](validation/reports/sliding_section/README.md) · [全部17次结果](validation/reports/sliding_section/结果总表.md) · [设计与参数](docs/滑动窗口截面顺应设计.md) · [冻结计划与增加测试的原因](docs/滑动窗口实验计划.md)

```powershell
# 空格保持/恢复相位，A/D向头/尾交接，U切换自动定位，R反向；不使用F键
py -3.12 -X utf8 codes/Rolling_Arc_Sliding_Section_Mujoco.py --scene elbow90_taper110_210 --start-x -.25 --duration 150
# 单次实验，出口后观察3个实际周期；独立目录避免覆盖归档
py -3.12 -X utf8 validation/sliding_section_experiment.py --scene elbow90_taper110_210 --label my_sliding --start -.25 --duration 150 --exit-cycles 3 --output validation/results/my_sliding
# 仅重放保存状态，不跑新动力学
py -3.12 -X utf8 validation/render_sliding_section.py --labels s4_sliding_cyclemean --video
```

![110→210 mm连续90°急弯真实运动](validation/reports/sliding_section/s4_sliding_cyclemean_overview.png)

## 固定窗口截面顺应：前阶段入口（2026-10-12）

使用关节实测角在Python中完成正运动学、局部平面/截面投影、连续线圆弧和Yamada积分；不依赖可视化EXE，也不把接触或管径真值送入控制器。J8–J13固定使能，其余关节主动力矩为零。取消原D规则的强制内弯，并用参考形态校准和总转角/开口约束防止累计收紧。

已运行18次动力学实验：原计划16次，另加2次替换不合格的90°相位初始化。没有追加反馈延迟、低阻尼或统计重复。通过的变化场景全身越过后观察至少3个周期；急弯也保留完整出口记录和未验收原因。基础对照同样通过直管变径，尚未证明顺应控制的性能优势或精确管径跟踪。

方法、参数和与2023方法的区别见[固定窗口截面顺应设计](docs/固定窗口截面顺应设计.md)，逐轮问题、实际状态和复现命令见[实验记录](validation/reports/fixed_section/README.md)。新方法尚未移植到实机，以下入口只运行MuJoCo。

```powershell
# 三维交互：空格冻结相位，E切换几何增益，R反向；物理和顺应仍运行
py -3.12 codes/Rolling_Arc_Section_Mujoco.py --scene taper_true90 --start-x -.35 --duration 75
# 一次有明确条件的实验，不自动展开矩阵；全身出口后观察3周期即结束
py -3.12 -X utf8 validation/fixed_section_experiment.py --scene taper_true90 --label my_taper90 --start -.35 --duration 120 --exit-cycles 3 --output validation/results/my_section
# 从保存的真实动力学状态生成全程侧视/端视，不重新跑仿真
py -3.12 -X utf8 validation/render_fixed_section.py --labels r5_taper90 r5_elbow90 --video
```

![真90°变径弯完整时程](validation/reports/fixed_section/r5_taper90_review.png)

## 首版 Step2 历史结果与运行

10 个仿真案例通过旧的推进/支撑/失能/限矩检查；该检查遗漏开放形态。前部变径弯管末态包绕角约 295°，因此不能视为完整 Step2 通过。复盘与原始状态见 [Step2 实验记录](validation/reports/step2/README.md)，修正幅值反馈、窗口协调及分阶段实验安排见 [设计说明](docs/Step2设计说明.md)。普通管道推进量与同口径 Step1 相差小于 1.8%。基线也能通过当前变径场景，未证明通过率或效率优势；变径弯管机械正功增加约 27.46%。

场景兼容 ID `taper_bend90` 实际为复用的 **45°变径弯管**，此前 90°标注有误；普通 `bend90` 仍为 90°。下图记录首版原型的实际状态，不表示开放形态验收已通过。

![Step2 实际运动](validation/reports/step2/actual_motion.png)

```powershell
# 3D 交互：空格冻结相位，E 顺应性开/关，R 反向
py -3.12 codes/Rolling_Arc_Adaptive_Mujoco.py --scene taper_bend90 --duration 90
py -3.12 codes/Rolling_Arc_Adaptive_Mujoco.py --scene taper_bend90 --window 4:9 --duration 90
# 从已保存的非标准搭管姿态接管
py -3.12 codes/Rolling_Arc_Adaptive_Mujoco.py --scene taper_bend90 --initial-state validation/reports/step2/pose_initial_state.npz --duration 65
# 历史 10 个案例与 5 个固定窗口对照；旧检查不包含开放形态
py -3.12 -X utf8 validation/step2_experiment.py
py -3.12 -X utf8 validation/step2_experiment.py --baseline --cases straight bend90 elbow90 taper_straight taper_bend90
```

实机仍运行 `codes/Rolling_Arc_Real.py`：Ctrl+U 基础形成，Ctrl+E 接入顺应性，Ctrl+J/H 单步，Ctrl+G 连续。输入 `gain=0.02 load=40` 调参；`support=auto` 用原始负载+姿态估计，无法保证失能区域真实接触可观测；`support=9:11` 按实物人工给定支撑。详细操作见 [实机说明](docs/实机调试代码说明.md)。代码移植和离线协议检查已完成，本轮没有连接真实机器人。

## Step1 基础结果

![实际动力学运动](validation/reports/step1/actual_motion.png)

| 场景 | 仿真时间，含 2 s 静置 | 实际沿管进度增量 | 结果 |
| --- | ---: | ---: | --- |
| 水平直管 | 30 s | 0.526 m | 持续滚动推进 |
| 水平柔弯，90°、中心线半径 300 mm | 70 s | 1.296 m | 整体通过，沿出口继续推进 |
| 实体连续的水平 90° 急弯 | 60 s | 1.070 m | 整体通过，全程保持同一滚动步态 |

结果来自当前参数的一次确定性运行，不代表任意摩擦、姿态和尺寸下的通过率。急弯过程中主动驱动力矩触及 2.355 N·m 上限，仍需关注实际驱动裕量。完整验收、静止/反向对照和阻尼对照见 [实验记录](validation/reports/step1/README.md)。

## 运行

本轮使用本机已有的 Windows Python 3.12.4、MuJoCo 3.3.0 环境。以下命令在仓库根目录执行；换用其他 Python 时将 `py -3.12` 替换为对应解释器。

```powershell
# 新环境安装；本机已有依赖可跳过
py -3.12 -m pip install -r requirements-tested.txt

# 三维交互仿真：空格保持/恢复滚动相位，物理仿真仍继续
py -3.12 codes/Rolling_Arc_Mujoco.py --scene straight --duration 30
py -3.12 codes/Rolling_Arc_Mujoco.py --scene bend90 --start-x -0.35 --duration 70
py -3.12 codes/Rolling_Arc_Mujoco.py --scene elbow90 --start-x -0.25 --duration 60

# 无窗口动力学仿真
py -3.12 codes/Rolling_Arc_Mujoco.py --headless --scene elbow90 --start-x -0.25 --duration 60 --output validation/results/elbow90.json

# 正式实验及结果重现；默认运行三场景、静止、反向和两种低阻尼对照
py -3.12 validation/step1_experiment.py

# 只重跑选定场景，单独输出，避免覆盖完整验收报告
py -3.12 validation/step1_experiment.py --cases elbow90 --output validation/results/elbow90_repeat

# 控制/失能/配置和实机映射的必要检查
py -3.12 -m unittest discover -s validation -p "test_*.py" -v

# 从已保存实际状态渲染截图；不会重新执行试验
py -3.12 validation/render_step1.py

# 本轮不连接硬件；仅预览原设备档案对应的目标与使能报文
py -3.12 codes/Rolling_Arc_Real_Preview.py --phase 0.5 --output validation/results/hardware_preview.json

# 实机单文件入口：真实串口，Ctrl+字母热键/控制台，不使用外部可视化程序
py -3.12 -m pip install -r requirements-real.txt
py -3.12 codes/Rolling_Arc_Real.py --port COM9 --log logs/rolling_arc_real.jsonl

# 同一实机控制流程的离线演练，不连接机器人
py -3.12 codes/Rolling_Arc_Real.py --dry-run
```

默认参数在 [configs/step1.json](configs/step1.json)，关节编号从 1 开始。需要调整方向时将 `direction` 改为 `-1`；默认幅值 60°，角频率 0.6 rad/s，初始相位 0。当前自动搭管初始化按相位 0 的圆弧构型设计；更改初始相位或窗口/幅值组合后需要重新检查初始构型与碰撞，不能直接沿用本次验收结论。

场景由参考模型派生，已随仓库保存；复建命令为 `py -3.12 tools/build_step1_scenes.py`，Step2 变径模型用 `py -3.12 tools/build_step2_scenes.py`。XML 单独载入不会自动搭管，应通过步态入口设置初始姿态并驱动电机。

## 文件位置

| 文件/目录 | 用途 |
| --- | --- |
| `codes/Rolling_Arc_Mujoco.py` | 可视化及无窗口仿真入口 |
| `snake_control/rolling_arc.py` | 共享圆弧目标、固定使能窗口、限矩驱动 |
| `snake_control/simulation.py` | 初始搭管、物理步进与实际状态测量 |
| `codes/Rolling_Arc_Sliding_Section_Mujoco.py` | 当前滑动窗口截面顺应交互入口，仅MuJoCo |
| `snake_control/sliding_section_compliance.py` | 选窗、整周期承重定位、卸载相位交接、单半圆预算 |
| `snake_control/sliding_section_simulation.py` | 实际管道接触/主动矩观测、局部驱动与相位保持 |
| `tools/build_sliding_section_scene.py` | 110→210 mm连续90°急弯变径模型构建 |
| `validation/sliding_section_experiment.py` | 单次动力学运行、实际状态存档及分项判定 |
| `validation/reports/sliding_section/` | 当前17次运行证据、全时程、视频与复盘 |
| `codes/Rolling_Arc_Adaptive_Mujoco.py` | Step2 交互/无窗口仿真、已有姿态接管入口 |
| `snake_control/arc_compliance.py` | 导出实机文件中的同一顺应性核心，避免双份公式 |
| `snake_control/adaptive_simulation.py` | 管道接触支撑提取、动态驱动掩码与相位推进 |
| `mujoco_model/` | 复用的 20 关节机器人、网格和三种管道 |
| `tools/build_step1_scenes.py` | 保留原碰撞几何，生成水平管场景 |
| `validation/step1_experiment.py` | 动力学验收、对照和实际状态存档 |
| `validation/reports/step1/` | 已提交的结果、原始状态、截图和指标图 |
| `validation/step2_experiment.py`、`validation/render_step2.py` | Step2 验收、基线对照、实际轨迹渲染 |
| `validation/reports/step2/`、`docs/Step2设计说明.md` | Step2 原始证据、方法和硬件限制 |
| `codes/Rolling_Arc_Real_Preview.py` | 原设备档案对应的离线报文预览 |
| `codes/Rolling_Arc_Real.py` | 独立实机串口入口，参数/公式集中，Ctrl 热键和控制台调试 |
| `docs/实机调试代码说明.md` | 真实运行、调试键位、常改位置、反馈与验证边界 |
| `docs/Step1设计说明.md` | 步态公式、失能含义、初始化及验证边界 |
| `docs/复用来源.md` | 源仓库文件与修改说明 |
| `交接记录.md` | 当前完成内容、命令、未完成部分和后续接口 |

本机实际状态回放视频位于 `validation/results/videos/`，Git 保存对应的完整状态文件和生成脚本，视频本身不入库。生成视频需安装 `requirements-media.txt`，执行 `py -3.12 validation/render_step1.py --video`，默认四倍速回放。

实机代码已补齐：一份 Python 文件直接连接真实舵机，按 Ctrl+U 形成局部圆弧，Ctrl+J/H 正反单步，Ctrl+G 连续/暂停，Ctrl+X 全部失能；不使用 F 键。代码顶部可直接改参数，`joint_angles_deg()` 可直接改公式，运行中也可输入 `amplitude=58 omega=0.5 delta=0.05`。详见 [实机调试说明](docs/实机调试代码说明.md)。

三类管道已完成的是动力学仿真验证。实机程序完成了离线串口检查和 Windows 热键监听启动/退出检查，真实机器人通过实验尚未执行。运行实机入口时，由该脚本直接占用串口，原 EXE 等其他串口程序应关闭。
