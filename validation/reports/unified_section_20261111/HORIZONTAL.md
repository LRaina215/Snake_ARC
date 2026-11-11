# 共享截面顺应的水平回归

使用仓库现有 20 关节水平模型和角度反馈滑窗。水平与竖直复用 `local_section`、`section_curvature_field` 和 `section_integral`，分别把实测截面变形写入开放半圆和浅导程整圈的关节目标。三个水平场景使用同一组控制参数。反馈周期 200 ms，命令周期 20 ms，动力学步长 1 ms。接触力、根位姿、管径和实际出口只用于独立评价。

## 同一配置的三场景结果

| 工况 | 持续清空开始 | 清空后一个滚动周期净前进 | 全程实测身体转角 | 出口续行的管轴包角 |
| --- | ---: | ---: | ---: | ---: |
| 113 mm 连续 90° 等径急弯 | 32.35 s | 214.2 mm | 167.3–189.5° | 219.4–257.5° |
| 110→210 mm 连续 90° 急弯 | 35.50 s | 189.5 mm | 154.2–189.2° | 174.1–206.1° |
| 110→210→110 mm 直管 | 80.95 s | 217.3 mm | 153.5–189.7° | 243.0–257.0° |

圆弧形态从 3 s 起按实际关节重建截面检查，身体转角须在 150–218°、开口距离与弧长之比至少 0.4、离面 RMS 不超过 15 mm。三组各自的最小开口比为 0.407、0.479、0.419，无自碰或触地记录，失能关节驱动力矩为零。完整出弯用碰撞外壳通过出口平面判定。两种变径场景在首次过线后有回摆，分别从 35.50 s、80.95 s 再次持续清空时计程。续行期间一直由管道承载，位移用清空时选定的同一批实体连杆计算。

管轴投影作为独立检查。急弯上同时跨两个局部轴向的样本分别有 223、118 帧无法用单轴衡量。表中的续行区间均有单轴结果。等径弯和变径直管出口仍有超过 218° 的管轴投影包角。对保存状态复算，三组在出口续行时管道接触点的周向最大覆盖约 185°、158°、111°。接触范围与使能段骨架绕管范围描述不同量，尚不能给出三组严格管轴形态均合格的结论。

## 复盘

1. 直接测得的候选在 2.8 s 左右轻微超出 190° 命令上限，原试验入口将其判为故障。现在只缩放实测变形所生成的候选至可用范围，保留相位和关节之间的相对形状。控制未使用恢复到名义圆弧的项。
2. 初版缩窗只依赖拟合圆周向覆盖。薄管上过早缩到五关节，后续某些相位实际身体转角降至约 109°。候选窗现在也接受全相位关节可达检查。即将低于半圆形态下限时，以实测切向转角的分相位统计和近期下降量提前扩窗。110→210 急弯曾有 107 帧低于形态下限。只提前静态阈值后还有 64 帧。加入近期下降量后降为零，等径急弯和变径直管复测也为零。
3. 全部场景都有至少一关节达到力矩饱和的样本。等径弯和变径直管的出口骨架包角偏大，需要检查窗口选取和身体相对管轴的偏移。管轴投影仍只用于评价。

## 同轮竖直检查

在正常滚动参考中减去同相位跟踪误差后，110→140 mm 渐扩中 J4、J5、J9 的误差残差较明显，双管过渡中主要集中于 J6–J8。凸环的残差与正常滚动部分重叠。按残差把轴向进给从这些关节重新分配后，渐扩承载环最高 2.023 m，双管最高 2.138 m，均未整身清空。让轴向进给高点随相位绕圈迁移，也未通过 110→140 mm，并出现 7 帧自接触。两种试验改动均已撤出当前控制器，原始记录和源码快照仍保留。大变径、方管及其过渡、凸环和双管的竖直通过能力仍未达到研究目标。

## 回放与复现

- [水平等径急弯](../../../artifacts/unified_horizontal/elbow_anticipate/process_4x.mp4)
- [水平 110→210 mm 变径急弯](../../../artifacts/unified_horizontal/taper_elbow_anticipate/process_4x.mp4)
- [水平 110→210→110 mm 变径直管](../../../artifacts/unified_horizontal/taper_straight_anticipate/process_4x.mp4)

```powershell
conda activate mujoco
python validation/horizontal_shared_section_experiment.py elbow --expansion-turn 160 --anticipate --output artifacts/my_elbow
python validation/horizontal_shared_section_experiment.py taper_elbow --expansion-turn 160 --anticipate --output artifacts/my_taper_elbow
python validation/horizontal_shared_section_experiment.py taper_straight --expansion-turn 160 --anticipate --output artifacts/my_taper_straight
python validation/render_motor_feedback.py artifacts/my_taper_elbow
```
