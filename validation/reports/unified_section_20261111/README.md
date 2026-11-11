# 水平与竖直共享截面顺应 · 首轮仿真

[后续竖直复测、通过范围及视频](FOLLOWUP.md) · [圆转方、凸环和双管对照](VERTICAL_OBSTACLES.md)

本轮以 18 关节竖直 Helix→单圈 ARC 为入口，先验证实测截面变形能否与现有三维拼接同时工作，再以同起态比较 2023 TRO Helix 顺应。验收以真实连杆、接触和运动为准。等径管从成环稳定后检查一个完整滚动周期。渐扩管从最后一个碰撞外壳通过过渡上缘后开始检查，要求维持形态并继续净上升。方管另外检查四个角部的实际接触。

## 实现与条件

- 水平半圆和竖直承载环使用同一个实测截面重建、模块长度归一化、局部转角变形和材料坐标积分。竖直适配在原三维曲率场上施加截面曲率差量，保留浅导程与两端的轴向错位。按相同关节离散方式重建未变形的曲线，用于消去投影与离散化引入的共同偏差。
- 仿真外环读角 200 ms、发目标 20 ms，动力学步长 1 ms。新增顺应只消费实测角和时间戳；几何和接触用于场景初始放置、既有人工事件模拟及验收。开发采用 110 mm 竖直等径管、110→140 mm 竖直圆管和 86.39 mm 边长的等周长方管。渐扩起止高度 1.90 和 2.05 m，摩擦参数均为 `.5 .005 .002`。
- 2023 TRO 组复用 `Snake_Robot_Gait/snake_control/pole_compliance.py` 的目标生成器，输入由本仓库论文式轴估计与截面构造产生。使用论文的管外端部/中部增益 .04/.08、螺距角 .25 rad。与 ARC 组从相同的 MuJoCo 积分状态分支，驱动条件一致。110→140 mm 是开发场景。论文原 150→250 mm 工况仍待研究模型复核。
- 原始轨迹、目标与实测角、使能掩码、接触、曲线参数、控制源码快照以及成功和失败视频保存在 `artifacts/unified_section_20261111/`。这里没有实机数据。仿真机械功与实际电流分别评价。

## 已完成的验证

| 工况 | 实际结果 | 验收 |
| --- | --- | --- |
| 110 mm 等径，0.40 rad | `straight_verified/arc`，52.75–65.35 s 连续合格，净上升 135.6 mm，成环后最大回滑 0.8 mm | 通过等径单周期。全程无自碰或触地，J18→J10 逐节释放。成环初期仍有包角偏低。 |
| 110 mm 等径，1.15 rad | `straight_phase/arc`，50.85–63.45 s 连续合格，净上升 136.2 mm，成环后最大回滑 1.0 mm | 通过等径单周期。全程无自碰或触地。 |
| 110→140 mm 渐扩，固定拼接 | `variable_fixed/arc` 在渐扩内停滞，最大回滑 125.6 mm | 未通过。 |
| 110→140 mm 渐扩，截面顺应 | `variable/arc` 在渐扩内停滞，最大回滑 115.2 mm。`variable_spare/arc` 在原使能区内增加约 18 mm 有效弧长，最大回滑降到 40.5 mm。`variable_bearing/arc` 重分配主体导程与端部错位，最大回滑降到 23.1 mm，环中心最高约 2.000 m | 三者均未完整通过，包角在过渡内曾低于 325°。 |
| 110→140 mm 渐扩，450 mm 长过渡 | `variable_long/arc` 保持其余条件，回滑 18.2 mm。头端最高 2.3555 m，高于 2.350 m 出口。环中心最高 2.3371 m，最后一个外壳最高只到 1.5879 m | 头端越过，整身未通过。全程有接触、无自碰和触地。 |
| 110→140 mm 渐扩，2023 Helix 顺应 | `variable_bearing/tro` 在 94.50 s 整身通过，随后 12.60 s 净上升 194.3 mm，无自碰、触地或重新进入过渡区 | 通过。 |
| 140 mm 等径，先在粗管贴合 | `straight_wide_seated/arc` 成环后净上升 397.7 mm，最大回滑 0.9 mm，承载区用 10 个关节，曲线弧长约 632→672 mm | 能持续承载上行。部分帧包角或轴向跨度超限，未得到一个连续合格周期。过渡全程出现 23 帧自接触。此例仅用于区分稳态和渐扩交接。 |
| 等周长方管，2023 Helix 顺应 | `square/tro` 第一个完整周期净上升 158.2 mm，四个角部均有实际接触，全程无自碰或触地 | 通过。 |
| 等周长方管，ARC | `square/arc` 在逐节退尾期间失载触地。`square_early/arc` 提前接入截面顺应仍失载。`square_hold_turn/arc` 保留承载环的净闭合转角，仍在退尾时触地 | 未形成可持续承载的单圈。 |

## 复盘与后续验证

1. **渐扩的停滞和粗管稳态是两个问题。** 原九关节承载环只有约 582–600 mm 曲线。粗管预贴合试验中使用 10 关节和更长的曲线，能维持上行。用相位平衡的实测半径直接扩窗曾让 J10 提前参与，最大回滑达 151 mm，并有短时脱离。增大反馈权重和约束净闭合转角也没有通过。`variable_coverage/arc` 分别根据半径变化和包角亏损给同一环扩径、追加材料，在约 83 s 失载，最大回滑 866 mm。失载前实际包角仍接近 346°。环绕形状和简易目标角偏差都不足以可靠识别夹持。后续先查新增材料接管期间，端部过渡如何改变承载接触和力矩，再从较小变径开始验证连续交接。
2. **方管失效发生在完成退尾之前。** 提前打开强顺应进一步提前了失载。净闭合转角相近时也可能缺少足够的承载接触，因此下轮应检查从实测非圆轮廓连续生成初始承载环的做法，以及退尾与承载交接的时序。
3. **水平数值回归已通过。** 三个窗口、18 个相位、非均匀连杆长度共 54 组的水平目标与原积分一致。相关离线测试通过。旧实机适配测试有三项因现有配置与测试夹具不一致报错，用修改前的水平内核复核得到同样报错。
4. **完整研究目标仍待验证。** 竖直渐扩后再做渐缩、局部凸起与单管转双管，方管通过后验证圆转方。连续急弯用两起态，以最后一个实体连杆完全出弯后整圈净前进 72 mm 为准。完成竖直验证后，再用同一截面核心验证水平急弯、变径急弯和变径直管。现有日常入口和实机入口仍对应原版本。

## 复现

```powershell
conda activate mujoco
$ring = @('--manual-takeover', '--reference-radius', '.13', '--ring-turns', '1.12',
    '--ring-radius-delta', '-0.003', '--section-shell-diameter', '.058', '--section-keep-curvature',
    '--progressive-tail-release', '--section-before-bend', '--outlet-use-spare-length',
    '--section-feedback-weight', '.35', '--vertical-use-spare-length')
python validation/acute_arc_experiment.py @ring --scene straight --modes arc --phase .4 --duration 76 --output artifacts/my_ring_straight
python validation/acute_arc_experiment.py @ring --scene variable --modes arc tro --section-bearing-advance .044 --duration 130 --output artifacts/my_ring_taper
python validation/acute_arc_experiment.py --review artifacts/my_ring_taper/arc artifacts/my_ring_taper/tro
python validation/render_vertical_arc.py artifacts/my_ring_taper/arc --speed 4
```

`--modes tro` 读取相邻 `Snake_Robot_Gait` 仓库的对照控制器。每次运行会保存用到的原始文件和当时的场景模型。
