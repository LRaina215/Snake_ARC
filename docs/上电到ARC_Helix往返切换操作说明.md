# 上电后，连续完成ARC → Helix → ARC

接好机器人电源和USB串口，托好中部搭管姿态，在仓库根目录启动一次程序：

    conda activate mujoco
    python .\codes\Rolling_Arc_Real.py

**下面每一行对应一次需要你按键的操作。同一行内的自动动作，按完后让程序连续执行，不再拆开按键。**整套往返保持同一个程序运行，中间不重新建弧或释放。

| 顺序：什么时候操作 | 具体按键，按先后顺序 | 这一次操作会连续完成什么，等到什么再接下一行 | 怎么调这一次动作的速度 |
| --- | --- | --- | --- |
| **1．上电搭管、程序已启动** | **Ctrl+U，按一次** | 中部形成初始ARC → 静置 → 两端短时归零再失能。等全部结束、ARC搭稳，再做第2行 | 建弧速度：FORM_RATE_DEG_S；静置：SETTLE_S；端部归零：END_ZERO_RATE_DEG_S，保持／最长时间为END_ZERO_HOLD_S / END_ZERO_MAX_S |
| **2．初始ARC已搭稳，开始切Helix** | **Ctrl+5，然后Ctrl+G一次** | 程序根据当前姿态自动调两侧长度 → 前部逐渐形成螺旋并就位。等前部实际搭牢、进入搭稳确认处，再做第3行 | 主要改HELIX_TRANSITION_SPEED_SCALE；其中自动调长上限为FEED_RATE_JOINTS_S × 此倍率；前部小幅就位速度另看GRAVITY_ALIGN_RATE_RAD_S |
| **3．前部已经实际搭牢，确认送尾** | **Ctrl+L，然后Ctrl+G一次** | 确认并接管尾段 → 前部配合螺旋推进、尾部沿曲线逐段上管 → 整理成完整Helix → complete并保持。全部属于这一次Ctrl+G，等Helix完成再做第4行 | 主要仍改HELIX_TRANSITION_SPEED_SCALE；送尾的直尾进给／配合滚动为GRAVITY_SHIFT_RATE_M_S / GRAVITY_RETURN_ROLL_RAD_S，首次送尾乘上述倍率 |
| **4．刚才的完整Helix已搭稳，直接切回ARC** | **Ctrl+7，然后Ctrl+G一次** | 尾部逆向解缠 → 导出头端、形成开放半圆 → 两端归零 → 自动归中 → 按最终范围再归零并失能两端 → 提示“已交回基础圆弧并保持”。全部自动衔接，只需这一次Ctrl+G；等完整结束再做第5行 | 整套回程改HELIX_ARC_SPEED_SCALE；端部归零速度也乘此倍率。内部速度细项见下文，不需要为这些内部阶段再按键 |
| **5．已返回ARC，继续运动** | **Ctrl+J正向单步／Ctrl+H反向单步；需要连续走就Ctrl+G** | 使用刚形成的ARC运动，保留当前半径、使能范围和相位。这时整套往返切换已经结束 | 每次行程：PHASE_STEP_RAD；单步与连续速度：OMEGA_RAD_S；目标角变化上限：MAX_TARGET_RATE_DEG_S |

**整套按键串起来就是：Ctrl+U → ARC搭稳 → Ctrl+5、Ctrl+G → 前部搭牢 → Ctrl+L、Ctrl+G → Helix完成 → Ctrl+7、Ctrl+G → 返回ARC → Ctrl+J／Ctrl+H。**

第2、3、4行的Ctrl+G各按一次启动该段连续动作；运行中再按Ctrl+G会暂停。中途需要暂停用Ctrl+空格，从原进度继续用Ctrl+G；查看状态用Ctrl+K。只有结束整次实验时才用Ctrl+X全部失能、Ctrl+Q退出。

如果要先手动预调两侧长度，在第1、2行之间插入 **Ctrl+4 → Ctrl+J使头侧增长／Ctrl+H使头侧缩短 → 本步结束或Ctrl+G暂停**，然后直接接第2行Ctrl+5、Ctrl+G，不用回中。预调速度是FEED_RATE_JOINTS_S，每次请求量是FEED_DISTANCE_M；在线Ctrl+=增大请求量、Ctrl+-减小请求量。Ctrl+5本身包含按当前姿态自动调长，直接走主流程即可。

## 调速补充

参数都在 codes/Rolling_Arc_Real.py 顶部；改文件后重启程序才生效。Ctrl+5与Ctrl+7的倍率独立，Ctrl+7完成后的普通ARC运动改OMEGA_RAD_S，不改切换倍率。

- Ctrl+4手动调长、Ctrl+5自动预调、Ctrl+7自动归中共用同一个调长执行器与FEED配置。调长速度统一改FEED_RATE_JOINTS_S；若在FEED_OPTIONS_OVERRIDE设置maximum_position_rate_joints_s，以覆盖值为准。Ctrl+5再乘自己的切换倍率，Ctrl+7归中再乘自己的回程倍率。
- 第3行末尾仍有约5秒的recover过渡，目前没有顶部独立调速入口；最终统一Helix的目标限速为18×Ctrl+5倍率度/秒。因此提高倍率不代表整段耗时严格按比例缩短。
- 第4行若只想调某一种动作，改HELIX_ARC_OPTIONS：reverse_feed_m_s / reverse_roll_rad_s控制解缠进给与滚动；lead_s / flatten_s控制过渡和半圆成形基础时长；target_rate_deg_s控制这些阶段的目标变化速度。归中使用公共FEED参数：target_rate_deg_s是调长目标限速，length_tolerance_m是长度代理容差，均写在FEED_OPTIONS_OVERRIDE。速度实际乘Ctrl+7倍率，基础时长随倍率缩短；静置和反馈时间仍按实际时间。
- 第5行每次单步太短，可把PHASE_STEP_RAD从0.10改成0.30，每次行程即原来的三倍；在线Ctrl+=增大、Ctrl+-减小。动作本身要更快则改OMEGA_RAD_S，在线Ctrl+右增加、Ctrl+左减小，每次0.05 rad/s。
- 想对第2、3、4行改为单步观察，用Ctrl+J代替Ctrl+G。Ctrl+5／Ctrl+7切换中每次执行HELIX_STEP_S / HELIX_ARC_STEP_S规定的秒数，默认各1秒；只有返回第5行的基础ARC后，Ctrl+J／Ctrl+H才按PHASE_STEP_RAD累计正反行程。

仿真演练只将启动命令换为 python .\codes\Rolling_Arc_Debug_Mujoco.py，整套按键不变；基础幅值／频率在仿真文件顶部SIM_AMPLITUDE_DEG / SIM_OMEGA_RAD_S设置，其他上述参数共用实机文件。

预览中蓝色使能、紫色失能；角度图黑色为实测、橙色为使能目标。端部归零释放后仍可能受接触或摩擦再次弯曲，用Ctrl+K核对释放前最大角与当前角。控制完成仍需对照实物贴管情况；[已有验证范围](../validation/reports/helix_arc_basic_return_20261108/README.md)与[按键修复记录](../validation/reports/basic_hotkey_response_20261108/README.md)另存。
