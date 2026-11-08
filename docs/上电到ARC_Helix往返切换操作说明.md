# 连续操作：上电 → Ctrl+U → ARC → Helix → ARC

**按下表从第1步一直做到最后，整套往返只启动一次程序。**先用Ctrl+U建立ARC，接着用Ctrl+5切成Helix，再直接用Ctrl+7切回ARC。中间保持当前姿态接续，不退出程序、不按Ctrl+X释放，也不重新按Ctrl+U建弧。所有操作均写出完整组合键。

下表第2步在仓库根目录执行：

    conda activate mujoco
    python .\codes\Rolling_Arc_Real.py

仿真演练只将第二行换为 python .\codes\Rolling_Arc_Debug_Mujoco.py，其余仍按这同一张表顺序操作。

| 顺序与接续条件 | 此时具体按什么 | 接下来应看到什么 | 这一段怎么调速 |
| --- | --- | --- | --- |
| **1．上电、搭管** | 接好机器人电源和USB串口，托好中部搭管姿态 | 中部跨在管上，两端有可下垂余长；J1是头部，J18是尾部 | — |
| **2．保持搭管，启动程序一次** | 执行上方命令 | 控制台和预览打开，显示实测角；此时尚未接管 | — |
| **3．在当前程序中建立初始ARC** | **Ctrl+U，按一次** | 默认J8–J13逐渐形成承载圆弧，其余关节失能 | FORM_RATE_DEG_S越大，建弧指令越快 |
| **4．接着等本次建弧收尾** | **无需按键，等形成、静置和两端归零结束** | 两端短时使能到零再失能，预览可能短暂全蓝；结束后承载弧蓝色、两端紫色，程序保持 | 静置改SETTLE_S；归零速度改END_ZERO_RATE_DEG_S，保持和最长时间改END_ZERO_HOLD_S / END_ZERO_MAX_S |
| **5．初始ARC已搭稳，直接开始切Helix** | **Ctrl+5，然后Ctrl+G** | 从刚形成的ARC读取姿态；需要时自动调两侧长度，再卷前部。Ctrl+5只选择，Ctrl+G才开始连续执行 | 整段主要改HELIX_TRANSITION_SPEED_SCALE；自动调长还受FEED_RATE_JOINTS_S影响 |
| **6．保持这次连续执行，等前部搭管** | **无需反复按Ctrl+G**；用**Ctrl+K**查看阶段 | 前部逐渐形成螺旋并小幅就位，尾部尚未整段上管；观察前部是否实际贴稳、有支撑 | 前部动作随Ctrl+5倍率；小幅就位速度另看GRAVITY_ALIGN_RATE_RAD_S |
| **7．前部实际搭牢，阶段为front_seat或front_hold** | **Ctrl+L确认，然后Ctrl+G继续** | Ctrl+L确认后先保持；随后Ctrl+G进入尾部接管。只按Ctrl+L不会开始送尾 | 首次送尾随HELIX_TRANSITION_SPEED_SCALE；直尾进给和配合滚动分别为GRAVITY_SHIFT_RATE_M_S / GRAVITY_RETURN_ROLL_RAD_S |
| **8．沿当前进度继续送尾，直到Helix完成** | **无需再切模式，让本次Ctrl+G继续运行** | 尾段沿拼接曲线逐段上管，前部配合螺旋推进；随后统一成全身Helix。出现complete后自动保持 | 首次接管与最终归形随Ctrl+5倍率；最后recover仍有约5秒过渡，倍率不会把整个流程等比例缩短 |
| **9．刚才的Helix已完成并搭稳，立即接反向切换** | **直接Ctrl+7，然后Ctrl+G** | 从这同一个Helix读取实测角，开始Helix→ARC；不用重启、重新建弧或重新上电 | 整段主要改HELIX_ARC_SPEED_SCALE，它独立于Ctrl+5倍率 |
| **10．继续本次连续执行，逆向解缠尾段** | **无需按键** | 尾部逐段解缠、失能下垂，前部保留承载螺旋 | Ctrl+7倍率；单独微调看HELIX_ARC_OPTIONS中的reverse_feed_m_s / reverse_roll_rad_s，分别控制送出直尾和配合滚动 |
| **11．解缠后接着导出头端、形成开放半圆** | **无需按键** | 前部剩余螺旋逐渐消除螺距，成为开放半圆，两端释放 | Ctrl+7倍率；退出过渡与半圆成形基础时长为lead_s / flatten_s，增大时长会更慢；目标变化速度为target_rate_deg_s |
| **12．半圆形成后，接着做第一次端部归零** | **无需按键，不需要Ctrl+L确认** | 承载半圆保持，两端归零后失能；随后进入自动归中 | END_ZERO_RATE_DEG_S乘Ctrl+7倍率；保持和最长时间仍为END_ZERO_HOLD_S / END_ZERO_MAX_S |
| **13．接着自动调节两侧余长** | **无需按键，让流程继续** | 圆弧在蛇身上的位置逐渐调整，头尾两侧长度接近均衡 | Ctrl+7倍率；单独调归中上限看HELIX_ARC_OPTIONS中的center_rate_m_s |
| **14．归中结束，再按最终范围归零两端** | **无需按键，等这一小段结束** | 新释放的端部也短时使能到零、失能；承载半圆保持。接着提示“已交回基础圆弧并保持” | 与第12步使用同一组归零参数 |
| **15．整套ARC→Helix→ARC已经完成，继续用返回的ARC运动** | **Ctrl+J正向单步，Ctrl+H反向单步；Ctrl+G连续／暂停** | 使用刚形成的ARC继续走，保留当前半径、使能范围和相位，不跳回初始模板 | 每次行程改PHASE_STEP_RAD；执行速度改OMEGA_RAD_S；目标角变化上限为MAX_TARGET_RATE_DEG_S |
| **16．本次实验确实要结束时** | **Ctrl+空格暂停，Ctrl+X全部失能，再Ctrl+Q退出** | 控制结束 | — |

**表中的“无需按键”表示沿前面那次Ctrl+G接着运行。运行中再按Ctrl+G会暂停，不要把每一行都当成要再按一次Ctrl+G。**中途想立即停住用Ctrl+空格，想看当前到了哪一步用Ctrl+K；暂停后从原进度按Ctrl+G继续。

第7步若已在front_finish提前看到前部搭牢，可以先Ctrl+L记录确认，再Ctrl+G完成衔接；进入front_seat后需要直接接尾时，再Ctrl+L、Ctrl+G即可。确认依据是实物前部贴管，不只是目标角跟上。

**如果想先手动预调两侧长度，将这一小段插在第4、5步之间：Ctrl+4 → Ctrl+J使头侧增长／Ctrl+H使头侧缩短 → 本步结束或Ctrl+G暂停 → 直接续第5步Ctrl+5、Ctrl+G。**Ctrl+5会根据调整后的当前姿态继续判断，不用回中间。预调速度为FEED_RATE_JOINTS_S，每次请求量为FEED_DISTANCE_M；在线用Ctrl+=增大请求量、Ctrl+-减小请求量。这里的Ctrl+J／Ctrl+H是一段调长任务，不是第15步的ARC相位步长。

调速参数在 codes/Rolling_Arc_Real.py 顶部，改文件后重启程序，再从第1步开始。Ctrl+5自动调长上限为 **FEED_RATE_JOINTS_S × HELIX_TRANSITION_SPEED_SCALE**；如果FEED_OPTIONS_OVERRIDE设置了maximum_position_rate_joints_s，以覆盖值为准。Ctrl+5最后recover阶段目前没有顶部独立调速入口，约5秒过渡在_advance_recover中；最终归形目标限速为18×Ctrl+5倍率度/秒。静置和反馈时间仍按实际时间，不会全部随倍率缩短。

第15步想**每次走更多**：Ctrl+=增大步长、Ctrl+-减小步长，或将PHASE_STEP_RAD从0.10改为0.30，每次相位行程就是原来的三倍。想**动作本身更快**：调OMEGA_RAD_S，也可在线Ctrl+右增加、Ctrl+左减小，每次0.05 rad/s。Ctrl+7完成后的滚动用这个基础速度，不用HELIX_ARC_OPTIONS中的roll_rate_rad_s。

整条路线也可中途改用Ctrl+J逐段观察：在Ctrl+5／Ctrl+7尚未完成时，Ctrl+J每次执行HELIX_STEP_S / HELIX_ARC_STEP_S规定的秒数，默认各1秒，执行中重复按不会追加；Ctrl+G负责连续。**等第14步交回基础ARC后，Ctrl+J／Ctrl+H才按PHASE_STEP_RAD累计正反行程。**

预览中蓝色为使能、紫色为失能，角度图黑色为实测、橙色为使能目标。两端失能后仍可能受接触或摩擦再次弯曲，用Ctrl+K核对释放前最大角与当前角；zero_then_disabled表示释放前已接近零，timeout_then_disabled表示到时已释放。控制阶段完成不代替实物贴管观察；[验证范围](../validation/reports/helix_arc_basic_return_20261108/README.md)和[按键修复](../validation/reports/basic_hotkey_response_20261108/README.md)另有记录。
