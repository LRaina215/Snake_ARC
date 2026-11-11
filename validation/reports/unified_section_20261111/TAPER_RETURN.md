# 竖直 110→140→110 mm 双向变径

验证同一承载环能否在扩大和缩小管径时增减曲线材料，并在整身通过后保持形态继续上升。沿用 18 关节、72 mm 轴距、200 ms 读角、20 ms 控制周期、0.5 rad/s 滚动速度和相同的曲线拼接控制。渐扩区为 1.90–2.35 m，粗管段长 300 mm，渐缩区为 2.65–3.10 m。每次在最后一个实体外壳持续完全离开渐缩区后检查一个滚动周期。接触力、世界位姿和管道几何只用于布置与复盘。

## 碰撞场景复核

1. **4.0 m 管顶，0.40 rad。** ARC 的 9 关节承载区在渐扩时扩到 11，缩径后退回 9。最后一个实体外壳在 260.60 s 首次越过 3.10 m。其后曾短暂回到过渡区，随后环到达模型管顶并触地，未完成持续合格周期。TRO Helix 对照在 152.75 s 整身清空并在合格周期内上升 207.0 mm，之后继续上升至模型管顶并触地。
2. **5.7 m 管顶，出口下端盖与缩径上缘对接。** 只延长最后一段细管后，ARC 在两个起相位都停在缩径接缝。环中心最高约 3.093 m，最后一个外壳未清空过渡区。按保存的姿态、速度和执行器目标复算接触，ARC 在 230–235 s 的接缝附近，出口圆柱贡献的平均轴向法向力为 −15.6 N，渐缩外壁贡献 +5.5 N。用同一批状态、只把出口下端盖埋到渐缩起点，分别约为 0 和 +3.0 N。该复算是对接缝接触的诊断，完整运动由重跑验收。

两个管顶版本的缩径段、出口平面和名义直径一致。4.0 m 版本能越过接缝，5.7 m 直接对接版本在接缝卡住。这个差异说明直接拼接圆柱端盖会改变接触结果，不能把该停滞直接归因于顺应控制。埋入版本只将出口圆柱下端由 3.10 m 移至 2.65 m，该段 110 mm 内柱处于原有渐缩实体内部，外形和评价出口高度保持原定义。

## 埋入端盖后的对照

原控制下，0.40 和 1.15 rad 的 ARC 最终从 260.35 和 258.95 s 开始持续清空过渡区。清空后的首周期内承载连续，无自碰和触地，最大轴向跨度分别为 95.78 和 95.75 mm。最低包角分别为 324.98 和 324.90°，两组均低于 325°，所以未通过完整周期验收。两组都在缩径期间出现了头部外壳与重叠连杆的短时自碰。TRO Helix 在配对的 0.40 rad 场景中于 151.85 s 整身清空，随后一周期净上升 207.2 mm，全记录无自碰或触地。

ARC 退回 9 关节后，同相位包角低点重复出现。控制重建的环包角比管轴验收略大，但仍有相对于配置目标的缺口。继续用实测包角的滤波误差保留少量曲线材料，将退窗后的额外弧长限制在 0.1 个轴距内。验收门槛和滚动速度沿用原设置。

| 步态及起始相位 | 整身持续清空 | 清空后合格周期净上升 | 该周期最小包角／最大轴向跨度 |
| --- | ---: | ---: | ---: |
| ARC，0.40 rad | 260.15 s | 139.6 mm | 326.20°／95.70 mm |
| ARC，1.15 rad | 255.50 s | 139.6 mm | 326.22°／95.72 mm |
| 2023 TRO Helix，0.40 rad | 151.85 s | 207.2 mm | Helix 形态合格 |

同相位 ARC 与 Helix 使用相同的碰撞 XML 和数值相同的分支初始状态。ARC 两组均从清空时保持单圈形态到 285.95 s，失能关节的驱动力矩为零，清空后未重新进入变径区，无自碰或触地。两组出区周期保留约 2.7–4.6 mm 的额外弧长，成环后的最大回滑分别约 4.0 和 4.1 mm。此缓坡工况下两种方法均通过，Helix 更早清空，且合格周期净上升更大。

保留原碰撞几何的 110→120→110 mm、0.40 rad 回归也通过。207.95 s 整身清空，随后一个合格周期净上升 135.2 mm，最低包角 327.41°、最大跨度 98.21 mm，全记录无自碰或触地。复用 [原小变径复现命令](FOLLOWUP.md#回放与复现) 即可。

全过程仍有改进空间。0.40 和 1.15 rad 起态分别在缩径期间记录到 6 和 3 帧头端外壳与连杆 9 自接触，当时仍有约 20–31 mm 的追加材料，半径修正已归零。成环到整身清空前也出现包角和跨度短暂超限。两组几乎每个采样帧都至少有一个关节达到模型力矩上限。当前结果验证的是这个缓坡场景的整身通过及出口续行。下一项最小验证是在同一模型中根据实测端部净空协调缩径曲率和材料退回，减少已观测到的自碰，然后再考虑更陡的变径。

## 复现

```powershell
conda activate mujoco
$ring = @('--manual-takeover', '--reference-radius', '.13', '--ring-turns', '1.12',
    '--ring-radius-delta', '-.003', '--section-shell-diameter', '.058', '--section-keep-curvature',
    '--progressive-tail-release', '--section-before-bend', '--outlet-use-spare-length',
    '--outlet-end-correction', '--section-feedback-weight', '.35', '--vertical-use-spare-length',
    '--vertical-adaptive-material', '--vertical-material-minimum-wrap', '350',
    '--vertical-material-handoff', '.036', '--section-bearing-advance', '.014',
    '--vertical-material-radius-gain', '.85', '--vertical-material-max-extra', '.105',
    '--vertical-end-clearance', '--vertical-material-return-phase', '6.2831853',
    '--vertical-return-advance', '.012')
python validation/acute_arc_experiment.py @ring --scene variable_return --taper-length .450 --upper-pipe-diameter .140 --vertical-outlet-height 5.7 --embed-return-outlet --phase .4 --modes arc tro --duration 286 --output artifacts/my_vertical_return_040
python validation/acute_arc_experiment.py @ring --scene variable_return --taper-length .450 --upper-pipe-diameter .140 --vertical-outlet-height 5.7 --embed-return-outlet --phase 1.15 --modes arc --duration 286 --output artifacts/my_vertical_return_115
```

- [ARC 0.40 rad 完整回放](../../../artifacts/unified_section_followup/variable_140_return450_reserve_040/arc/process_4x.mp4) · [ARC 1.15 rad 完整回放](../../../artifacts/unified_section_followup/variable_140_return450_reserve_115/arc/process_4x.mp4)
- [直接对接记录](../../../artifacts/unified_section_followup/variable_140_return450_outlet_040/) · [埋入端盖及 TRO 对照记录](../../../artifacts/unified_section_followup/variable_140_return450_embedded_040/)
