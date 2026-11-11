# 单圈 ARC 竖直仿真复测

18 关节模型使用 72 mm 相邻轴距，控制周期 20 ms，实测角更新周期 200 ms，滚动角速度保持 0.5 rad/s。连续控制读取角反馈、时间戳、模型连杆长度和自身的滚动相位。MuJoCo 的接触、根位姿、管道几何用于评价或模拟操作者在入弯处发出一次指令。水平与竖直沿用同一套截面平滑和材料转角积分。以下均为仿真结果。

## 已复核的通过范围

| 工况 | 起始相位 | 完整通过与续行 | 续行及越弯形态 |
| --- | ---: | --- | --- |
| 110 mm 等径连续短弯，中心线半径 105 mm | 0.40 rad | 87.501 s 全身清空。91.951 s 净前进 72.11 mm | 入弯稳定段至 72 mm 门槛，最低包角 325.74°，最大轴向跨度 96.61 mm |
| 同一短弯 | 1.15 rad | 82.601 s 全身清空。87.001 s 净前进 72.45 mm | 同一口径，最低包角 326.05°，最大跨度 97.59 mm |
| 110→120→110 mm 竖直管，两段渐变各 150 mm，中间粗管 300 mm | 0.40 rad | 207.901 s 全身清空。完整滚动周期净上升 135.95 mm | 最低包角 326.72°，最大跨度 97.86 mm |
| 同一细→粗→细管道 | 1.15 rad | 首次过线 205.051 s，尾端随后回摆。从 209.001 s 持续清空开始，完整周期净上升 135.82 mm | 最低包角 326.64°，最大跨度 97.84 mm |

两种短弯起态的全程均无自接触或触地，J18 至 J10 逐节释放，弯中维持 J1–J9。出弯由碰撞外壳、实体出口面和水平管承载共同判定，随后使用固定的同一批实体连杆计程。两组的实际进弯接触均在稳定单圈检查窗口内。该短弯已有全身使能的 Helix 通过证据，当前结果证实 ARC 可完整通过该弯。

细→粗→细两组均无自接触或触地。增径过程中加入 J10，回到细管后退出，整段最大回滑分别为 5.1 和 5.3 mm。通过后的单周期满足包角、轴向跨度、接触和清空要求。穿越变径过渡期间曾出现短时包角或跨度超限，故通过后的验收与变径全过程形态分别报告。两种起态在几乎所有记录帧中均有至少一关节力矩饱和，实机力矩余量仍未验证。

## 复盘

- **短弯。**连续保留直段中已使用的约 18 mm 环内弧长，减小入弯时的开口。全弯压缩轴向错位曾导致首尾自碰。恢复首尾净空后，仅在需要压紧的窄区重分配弧长，并只在入弯的编码器间距允许时适当压缩轴向错位，两个起态才同时通过。对出口同时增加压缩的版本使 0.40 rad 起态退化到 321.4°，因此撤去该项。
- **双向变径。**渐扩时给同一承载环增加有效弧长，新关节目标从实测角平滑过渡。渐缩时，等较窄截面的证据覆盖一个滚动周期，再退回追加材料，同时减小环主体的浅导程。没有延迟回收的版本停在缩径出口以下。保留材料但不调整导程的版本完成整身通过，续行峰值跨度仍约 101.6 mm。当前方案将两组续行的峰值降到 98 mm 以内。
- **110→140 mm。**增大初始半径导致机器人在细管提前失载。保留细管起态并逐步加入材料后能保持接触。将可追加材料上限从 60 提至 105 mm 并保留最低包绕目标，实际最多追加 93.1 mm，最大回滑降到 22.3 mm。承载环中心最高约 2.022 m，仍低于 2.050 m 渐扩出口。这个场景的原论文方法对照已通过。下一步需核对渐扩时轴向滚动和实际接触分布，追加弧长本身尚不足以解决停滞。
- **等周长方管。**保留仍承载的 Helix 导程，直到局部环完成接管，可逐节退尾并出现四角接触。之后用编码器估计实际包绕缺口来追加弧长，环的净上升由 44 秒约 62 mm 提高到 50 秒约 364 mm，最大回滑降至 6.2 mm。首圈之后轴向跨度峰值 128.1 mm、最低包角 317.1°，整次记录有 11 帧自接触。把主体导程直接挪回两端的试验进一步增加了首尾自接触，未纳入当前方案。

圆转方、竖直管外局部障碍和单管转双管仍待完整验证。水平三个目标工况仍待用共享控制器做动力学验证。两种短弯起态在本轮均只验证整圈续行，开成半圆后的自动归中另行验收。

## 回放与复现

- [短弯 0.40 rad](../../../artifacts/unified_section_followup/short_entry_gap_040/arc/process_4x.mp4) · [短弯 1.15 rad](../../../artifacts/unified_section_followup/short_entry_gap_115/arc/process_4x.mp4)
- [细→粗→细 0.40 rad](../../../artifacts/unified_section_followup/variable_return_compact/arc/process_4x.mp4) · [方管未通过](../../../artifacts/unified_section_followup/square_wrap_material/arc/process_4x.mp4)

上述本地目录保留 `hypothesis.json`、`run.json`、`controller.json`、`trace.npz`、控制源码快照及几何模型。使用既有 Conda 环境，以下两条命令各运行一个起始相位，将 `--phase .4` 改成 `--phase 1.15` 并另设输出目录即可复测第二相位。

```powershell
conda activate mujoco
$ring = @('--manual-takeover', '--reference-radius', '.13', '--ring-turns', '1.12',
    '--ring-radius-delta', '-.003', '--section-shell-diameter', '.058', '--section-keep-curvature',
    '--progressive-tail-release', '--section-before-bend', '--outlet-use-spare-length',
    '--outlet-end-correction', '--section-feedback-weight', '.35', '--vertical-use-spare-length')
python validation/acute_arc_experiment.py @ring --scene short --modes arc --phase .4 --duration 104 --bend-section-radius-trim .0016 --output artifacts/my_short
python validation/acute_arc_experiment.py @ring --scene variable_return --modes arc --phase .4 --duration 239 --upper-pipe-diameter .120 --vertical-adaptive-material --vertical-material-radius-gain 0 --vertical-material-handoff .036 --vertical-material-minimum-wrap 345 --vertical-material-return-phase 6.2831853 --vertical-return-advance .026 --section-bearing-advance .044 --output artifacts/my_return
python validation/acute_arc_experiment.py --review artifacts/my_short/arc artifacts/my_return/arc
```
