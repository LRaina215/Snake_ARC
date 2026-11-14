# 渐扩坡度与水平出口包角复测

本轮的竖直对照保持 18 关节、72 mm 轴距、反馈 200 ms、滚动角速度 0.5 rad/s 和 110→140 mm 管径变化。先只改变渐扩长度，确认停滞是否随坡度改变。长渐扩整身通过后，在相同几何中检查整圈续行，并针对环的轴向跨度和端部实体净空改曲线。水平部分保持 20 关节和原有三个碰撞场景，只改由角反馈选择窗口的依据。接触和世界坐标均用于验收。

## 竖直 110→140 mm

| 渐扩长度 | 起态 | 结果 |
| --- | ---: | --- |
| 450 mm | 0.40 rad | 190.651 s 整身持续清空，随后一周期净上升 126.6 mm。该周期最低包角 329.14°、最大轴向跨度 98.44 mm |
| 450 mm | 1.15 rad | 192.151 s 整身持续清空，随后一周期净上升 126.9 mm。该周期最低包角 329.05°、最大轴向跨度 99.21 mm |
| 150 mm | 0.40 rad | 同一配置下，环中心最高 1.947 m，低于 2.050 m 渐扩出口。随后出现 10 帧自碰并失载触地 |

450 mm 两组使用相同参数，全程均无自碰或触地，最大回滑分别约 4.0 mm 和 10.5 mm。整身出区后的两个续行区间均持续有接触。变径过渡期间的最大轴向跨度分别为 106.1 和 102.8 mm，分别保留为全过程形态超限。两组几乎所有保存帧均有至少一关节达到力矩饱和，未验证实机力矩余量。

同一 450 mm 场景、同一 0.40 rad 起态的 2023 TRO Helix 对照在 107.151 s 整身清空，随后一周期净上升 194.4 mm。两种步态均通过此场景。Helix 在当前模型中更早清空，续行周期的净上升也更大。

首个 450 mm 对照使用原有 44 mm 环主体导程，完成整身通过并净上升 115 mm，续行最低包角约 324°，轴向跨度超过 105 mm。把主体导程逐步压小并增加环的有效覆盖后，整圈包角改善。重叠区出现的 51 帧自接触集中在头外壳与连杆 9、10，单看曲线两个端点的间距会漏掉相邻实体干涉。复用原弯道端部避碰：用角反馈比较头部与重叠连杆的最小距离，并在几个局部曲率方向候选中选择净空较大的一个。只在环已追加弧长时介入，保持滚动速度。24 mm 主体导程配合此方法消除了自碰，续行跨度仍约 102 mm。最后将主体导程减为 14 mm，两个起态均通过同一续行门槛。

对短坡额外试过从同相位舵机角误差识别受阻位置，把局部闭合曲率向其他区域分配。环中心仍止于约 2.022 m，随后失载。该试验改动已撤出，原始记录及源码留档。150 mm 的负结果保留，450 mm 的通过只证明缓坡工况。

## 水平出口

水平开启截面内部圆心估计时，在全窗圆拟合残差过大后才尝试。边界点不参与圆心拟合，仍参与计算窗口的实际周向覆盖。这样能在变径直管回到细管时继续收集缩窗依据。三场景采用同一组新配置。

| 工况 | 整身持续清空 | 保持开放圆弧净前进 | 出口最大管轴包角 |
| --- | ---: | ---: | ---: |
| 等径连续 90° 急弯 | 32.35 s | 222.0 mm | 247.7° |
| 110→210 mm 连续 90° 急弯 | 35.50 s | 189.5 mm | 206.1° |
| 110→210→110 mm 直管 | 81.05 s | 211.9 mm | 231.2° |

三组均通过身体截面的开放形态要求。直管在 64.4 s 与 70.6 s 分两次缩短窗口，清空出口时由 6 关节承载，最大管轴包角从约 257° 降到约 231°。等径弯与变径直管仍超出 218° 管轴门槛。单独试过按身体拟合圆的极角去放松边界，等径弯出口最大值降到 221.6°，但直管形态退化，此改动未纳入当前控制器。

11 月 14 日用同一配置复跑三场景，新增 `sustained_exit_pipe_open` 审计。该项从持续清空起算，只要出口续行有样本无法确认单一管轴，或包角超过 218°，就不给通过。三组的出口单轴样本均可观测，只有变径急弯的严格出口形态通过。三组均无自碰，失能关节命令力矩为零。检查实际连杆的周向位置发现，等径急弯出口包角中位数 242.9°，去掉头侧边界连杆后降至 203.0°。变径直管从 229.5° 降至 189.7°。这只用于定位下一次改动应关注的材料位置，验收仍包含边界连杆。

## 回放与复现

- [450 mm 渐扩，0.40 rad](../../../artifacts/unified_section_followup/variable_140_slope450_compact_clearance/arc/process_4x.mp4) · [1.15 rad](../../../artifacts/unified_section_followup/variable_140_slope450_compact_phase115/arc/process_4x.mp4)
- [水平双向变径直管](../../../artifacts/unified_horizontal/taper_straight_contraction/process_4x.mp4)

```powershell
conda activate mujoco
$ring = @('--manual-takeover', '--reference-radius', '.13', '--ring-turns', '1.12',
    '--ring-radius-delta', '-.003', '--section-shell-diameter', '.058', '--section-keep-curvature',
    '--progressive-tail-release', '--section-before-bend', '--outlet-use-spare-length',
    '--outlet-end-correction', '--section-feedback-weight', '.35', '--vertical-use-spare-length',
    '--vertical-adaptive-material', '--vertical-material-minimum-wrap', '350',
    '--vertical-material-handoff', '.036', '--section-bearing-advance', '.014',
    '--vertical-material-radius-gain', '.85', '--vertical-material-max-extra', '.105', '--vertical-end-clearance')
python validation/acute_arc_experiment.py @ring --scene variable --modes arc --phase .4 --taper-length .450 --upper-pipe-diameter .140 --duration 244 --output artifacts/my_wider_taper
python validation/horizontal_shared_section_experiment.py taper_straight --expansion-turn 160 --anticipate --inner-circle-resize --output artifacts/my_taper_straight_inner_fit
```
