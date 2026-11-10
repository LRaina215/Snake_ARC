# 竖直 Helix → 整圈 ARC 曲线设计复盘

目标是 18 关节蛇在竖直直段从 Helix 接管成紧凑整圈，保持整圈跨连续急弯，直到最后一个实体连杆出弯，再净前进 72 mm。使用已测的 110 mm 等径、105 mm 中心线半径短弯，沿用已更正的[全身使能 Helix 对照](../acute_arc_20261109/README.md)。本轮只比较 ARC 内部的方案。

## 曲线结构

原弯轴参考给一圈内的不同材料位置分配不同管轴截面。弯头附近，这会同时拉长承载环并改变首尾相对位置。`SectionRingCurve` 将一圈布置在同一个随名义管轴移动的截面上，主体保留浅轴向导程，主要的轴向错位由两端短曲线提供。错位至少取模型外壳直径，两端以连续曲线连接。离开承载环的肩部曲线向名义下垂方向过渡，后段仍然失能。这样可分别检查承载主体的形态和首尾实体净空。

原弯轴参考与新截面在材料坐标上连续交接，仍由 `CurveGait` 积分生成全部关节目标。结构对照的滚动角速度保持 0.5 rad/s，名义进给保持 7 mm/s。试过按实测环面倾转触发交接，也试过在同一交接区分别插值曲率大小与方向。后一种方式可以避免直接混合向量时，方向不同的分量抵消并临时削弱承载曲率。

## 实验复盘

| 病例 | 改动与观察 |
| --- | --- |
| `late_section` | 旧扫掠曲线延后恢复轴向错位，弯中包角最低约 335°，跨度仍达到 128 mm。全身出弯后有一段合规续行达 175 mm，弯内形态未通过，交接附近有 58 帧自接触。 |
| `encoder_spacing` / `outlet_spacing` | 只从角反馈重建宽度和首尾间距。全弯补偿导致失载。限制到出口附近后，参考起态成功出弯且自接触为零，但尾端回摆且后续环宽超限。 |
| `manual_outlet` / `encoder_compaction` | 用实机入口从现有 Helix 角和目标接管。出弯间距补偿的版本虽完整出弯，整圈续行只有 9.6 mm。加弯中压缩后未出弯，在水平支撑建立前失载。 |
| `manual_late` / `manual_slow` | 单独验证实机接管接口及较慢对照。旧扫掠曲线仍受首尾干涉。减速对照失载，现有研究入口已移除本轮新增的速度倍率。 |
| `manual_tip` | 仅对前 150 mm 曲线的曲率方向让位，候选由关节角重建后的相对间距比较选出。118.80 s 完整出弯，122.60 s 净前进 72.5 mm。弯区最大跨度约 131 mm，弯中仍有 45 帧自接触。 |
| `manual_tip_phase` / `manual_feed_sync` | 相同模型和控制参数，初始相位从 0.40 改到 1.15 rad。两个病例均在水平接管之前失载。按观测形态降低名义曲线进给也未恢复承载。 |
| `section_ring` | 固定速度改为同一截面承载环。移交初期下滑约 0.6 m，随后重新爬升。145 s 时头端已转至水平侧附近，最后连杆仍差约 57 mm。弯控制阶段轴向跨度最大约 106 mm，自接触为零。此例终止时仍在迁移，因此保留原例并延长运行复核。 |
| `section_complete` / `section_endpoints` | 延长同一病例后，前者在 148.65 s 完整出弯，152.40 s 达到 72 mm，全程无自接触，弯中最低约 310°。后者按半个轴距增加端部覆盖，虽也完整越弯，但发生 224 帧自接触，未通过合规续行。 |
| `section_tilt_handoff` / `section_tilt_phase` | 由角反馈估计环面相对下垂尾段的倾角，开始倾转后才交接。两起始相位都完整越弯，无明显下滑或自接触。0.40 rad 起态合规续行 66.8 mm。1.15 rad 起态合规续行超过 72 mm。两个病例都交接偏晚，弯中最大跨度约 169–171 mm。 |
| `section_curvature_handoff` | 将曲率大小和方向分开交接。参考起态下滑由约 596 mm 降到 31 mm，无自接触。103.85 s 全身出弯，107.65 s 达到 72 mm。后续合规推进约 1.51 m。弯中最大跨度约 107 mm，最小包角约 304°。 |

越弯以最后一个实体连杆和水平管独立承载为准。72 mm 从出弯后用同一批实测连杆的位置差计算。弯中的整圈形态独立按 325–395° 和 100 mm 的轴向跨度上限检查。曲率交接版本把夹持问题和弯中开口区分开了，仍需在出弯前处理包角不足。在几乎所有保存帧中均有至少一关节达到力矩饱和，仿真也未确认有实机扭矩余量。

## 实机接管接口

`snake_control/vertical_ring_manual.py` 的 `VerticalRingManual` 是研究控制器的实机接管接口。它已在 MuJoCo 的正常动力学回路中从已搭好的全身 Helix 接管。输入为当前一帧全身标定角、当时已发送的 Helix 目标和现有 `LINK_LENGTHS_M`。接管后使用现有全身角读数周期，仿真采用 200 ms。`targets` 已是控制器角度制，`enabled` 是布尔掩码，沿用现有 `prepare_enabled`、`set_enabled`、`send_targets` 的顺序即可接线。

| 操作 | 接口 | 现场观察 |
| --- | --- | --- |
| 已搭稳在竖直管上，选择新模式 | `VerticalRingManual(measured_deg, lengths, options, helix_targets_deg=current_targets)` | 接管保持现有目标 |
| 单步或连续形成承载环 | `step_once()` 或 `toggle_continuous()` | 从全身螺旋逐段退出余尾，保留近一圈 |
| 看到承载环到达入弯标记 | `begin_bend()` | 开始按名义弯轴迁移 |
| 电机循环 | `advance(dt, now, feedback_time, measured_deg)` | 只在控制时钟推进时更新目标，暂停或等待读角时保持现有目标 |

模拟操作者发出入弯指令的位置记录在 `operator.json`，实机由操作者观察。控制器没有真值位姿、接触或世界轴向位移输入。`advance()` 的时间戳须来自同一时钟。继续使用当前实机脚本的释放逻辑。联合验收目前没有通过，以上接口作为接线和台架验证候选。

## 按审核图实现尾部逐节失能

[审核通过的示意图](../../../figures/fig_vertical_helix_arc_stitch_review.png)要求单圈持续使能，尾部从末端向圈内依次释放。`--progressive-tail-release` 先用新鲜实测角重建前侧一圈，确认包角后才允许释放。每次只关闭当前最后一个使能关节，重复使用同一帧读角时不追加失能。曲线积分在仍使能的材料区间内连续退缩，释放后目标值仅用于记录，驱动扭矩为零。最终使能区按单圈曲线覆盖的相邻连杆积分范围产生。退出完成后才接受入弯指令。

| 病例 | 观测与判定 |
| --- | --- |
| `approved_release01` | J18 至 J11 各自单独失能，完成时刻 46.94 s。入弯 52.40 s 后，原有截面插值缩短了单圈，又释放了 J10。86.10 s 完整出弯，90.75 s 达到 72 mm。弯中最小包角 316.7°，全过程无自接触。 |
| `approved_release_phase` | 1.15 rad 起始相位重复同样的尾部释放。94.60 s 完整出弯，98.80 s 达到 72 mm。弯中最小包角 314.0°，全过程无自接触。 |
| `approved_release_endpoints` | 延伸单圈端部材料使 J10 在入弯前后保持使能。最小包角提高至 322.3°，但出现 288 个自接触帧。完整出弯后合规续行约 29 mm。单纯延伸端部材料使首尾净空变差。 |
| `approved_release_fixed` | 在尾部释放期间拼完弯中所用的单截面曲线。J18 至 J10 依次失能，J1–J9 从入弯前一直保持使能。92.75 s 全身完整出弯，97.35 s 达到 72 mm。全程无自接触，合规续行到约 130 mm。弯中最低包角 312.8°，仍不满足整圈越弯。 |
| `approved_release_fixed_phase` | 仅将起始相位改为 1.15 rad。逐节失能后同样保持 J1–J9 使能，89.25 s 完整出弯，全程无自接触。弯中最低包角 314.5°，出弯后虽有约 1.03 m 实际平移，整圈形态仅连续合格约 1.5 mm，续行门槛未通过。 |
| `approved_release_seam` | 把端部轴向净空延伸到重叠区。初版平台和斜坡的切向衔接不连续，尾部逐节释放完成后未能继续爬到弯头。 |
| `approved_release_seam_smooth` | 以五次曲线消除平台两端的切向跳变。能进入弯头，但出现 201 个自接触帧，最后连杆的外形离出口面仍至少差 43 mm。更长的轴向避让区没有解决离散连杆的干涉。 |

各病例沿用 0.5 rad/s 滚动与 7 mm/s 名义进给。通过实际扭矩掩码和仿真驱动力矩核对逐节释放。`approved_release_fixed` 在尾部释放完成后保持同一使能窗口，越过出口并通过 72 mm 续行门槛。整圈越弯的形态判据仍未通过。`review.json` 逐项记录每次扭矩掩码变化，带事件的运行还记录 `tail_release.json`，其中反馈时刻使用实验全局时钟。[固定使能边界完整回放](../../../artifacts/acute_arc_20261110/approved_release_fixed/arc/process_4x.mp4)

下一项最小验证应先用离散连杆外壳和参考曲率做一整周期的净空核对，再选择满足包角与净空的两端过渡曲线进入动力学。当前数据支持保留逐节失能和入弯前确定使能边界，尚未验证同一单圈几何在不同相位都能连续维持形态。[逐节失能结果摘要](approved_release_results.json)

## 复现与记录

```powershell
conda activate mujoco
python validation/acute_arc_experiment.py --output artifacts/my_vertical_section --scene short --modes arc --reference-radius .13 --ring-turns 1.12 --ring-radius-delta -0.003 --section-shell-diameter .058 --section-keep-curvature --manual-takeover --duration 184
python validation/acute_arc_experiment.py --review artifacts/my_vertical_section/arc
python validation/render_vertical_arc.py artifacts/my_vertical_section/arc --speed 4
```

逐节退尾、入弯前固定单圈的复现命令如下。

```powershell
python validation/acute_arc_experiment.py --output artifacts/my_approved_release --scene short --modes arc --reference-radius .13 --ring-turns 1.12 --ring-radius-delta -0.003 --section-shell-diameter .058 --section-keep-curvature --progressive-tail-release --section-before-bend --manual-takeover --duration 173
python validation/acute_arc_experiment.py --review artifacts/my_approved_release/arc
python validation/render_vertical_arc.py artifacts/my_approved_release/arc --speed 4
```

本轮完整状态、角反馈、阶段目标、掩码、接触、模型及代码快照保存在本机 `artifacts/acute_arc_20261110/`。每次仿真保留原始 `run.json`，离线复核另写 `review.json`。现场部署前还需要验证不同起始相位下的承载交接，并复测出弯后至少 72 mm。

## 出弯续行和手调入口

水平段仅修改环两端的拼接。利用最后一个使能关节在材料坐标上剩余的约 18 mm，把单圈端部稍向前延伸，同时保留原有的轴向错位。材料边界限制在 J10 的积分区间之前。J18 至 J10 的逐节失能、弯中 J1–J9 的使能掩码、滚动速度 0.5 rad/s 和名义进给 7 mm/s 均沿用原设置。

| 起始相位 | 首次完整出弯 | 连续保持整圈达到 72 mm | 计程段最小包角 | 计程段最大轴向跨度 |
| --- | --- | --- | --- | --- |
| 0.40 rad | 92.70 s | 97.20 s | 335.3° | 98.7 mm |
| 1.15 rad | 89.45 s | 93.75 s | 333.8° | 96.2 mm |

验收从最后一个实体外壳完全过面且水平直管独立承载的首次保存帧开始。此后持续检查包角、跨度、接触和出弯状态。两例完整运行均无自接触、无触地，圈外驱动力矩为零。单独增强端部曲率的对照两例只连续达成约 57 和 20 mm。弯内包角仍曾低于 325°，更长的水平续行也仍有个别形态超限，结论限于本次首次出弯后的 72 mm 任务。23 项相关离线测试通过。旧手调测试中的固定 20 关节读角和归零阶段最多 6 关节断言，在同步前的代码上也失败。

日常 MuJoCo 入口以真实动力学预置两种竖直 Helix 起态，用共享 Ctrl 命令路径复测。0.40 和 1.15 rad 两次都完整出弯，并从首次出弯分别继续 4.6 和 4.3 s 达到 72 mm。实机单文件内嵌同一控制核心，使用离线方式核对输出。实机物理结果待测。[研究与日常入口结果](outlet_results.json)

| 操作时机 | 按键与观察 |
| --- | --- |
| 竖直管上已有搭稳且全身使能的 Helix | **Ctrl+8 → Ctrl+G**。观察成环、尾部依次失能、单圈上行。日常 MuJoCo 设 `SCENE='vertical_short'`，启动时已预置该 Helix，使用同一组按键。 |
| 退尾已完成，头端到入弯标记 | **Ctrl+L** 开始跨弯。110 mm 短弯仿真的标记约为水平管轴线下方 185 mm。 |
| 承载圈已经搭稳在水平直管 | **Ctrl+L** 接入水平端部曲线并继续走。仿真用头端超过出口面约 90 mm 且承载转到水平段作为提示。完整出弯计程仍须等最后一个连杆越过出口面。 |

**Ctrl+J** 单步，**Ctrl+G** 连续或暂停，**Ctrl+空格** 暂停，**Ctrl+K** 查看阶段与实际角，**Ctrl+X** 全部失能。`VERTICAL_INITIAL_PHASE_RAD` 可在日常 MuJoCo 文件顶部切换到 1.15。实机配置适用当前 18 关节、110 mm 管道与已标定角度，实机命令由操作者确认入弯和水平接管。
