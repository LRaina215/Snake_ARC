# Snake_ARC

研究蛇形机器人在管道外的局部圆弧滚动：使能承载圆弧及交接所需关节，其余关节失能。项目提供基础ARC、局部顺应与自动选窗、两侧长度调整、ARC↔Helix切换及T管重力矩越障控制。单文件实机程序带有关节构型与使能预览，仿真手调入口共用Ctrl热键和控制逻辑。

**完整操作请看：[从上电、Ctrl+U到ARC→Helix→ARC](docs/上电到ARC_Helix往返切换操作说明.md)。**包含每一步按什么、观察什么，以及各阶段的步长和速度调节。

## 启动环境

沿用本机已有的`mujoco` Conda环境：

```powershell
conda activate mujoco
cd "D:\Desktop\SNAKE\Paper Code\Mujoco\Snake_ARC"
```

## 实机测试

```powershell
python .\codes\Rolling_Arc_Real.py
```

直接运行一份Python文件，自动识别CH340串口；需要指定端口时在文件顶部修改`PORT`。当前配置18关节，默认Ctrl+U形成J8–J13。机械尺寸按手量近似配置：头端25 mm、相邻转轴72 mm、尾端91 mm。参数和速度均在文件顶部修改，改完重启程序；日志保存到`codes/logs/`。

按同一份程序连续做完整个往返：**上电搭管 → 启动一次程序 → Ctrl+U形成ARC并等归零结束 → Ctrl+5、Ctrl+G自动准备前部 → 前部搭稳后Ctrl+L确认、Ctrl+G送尾 → Helix完成后直接Ctrl+7、Ctrl+G解缠并归中 → 最终归零结束，Ctrl+J／Ctrl+H正反ARC运动。**完整的逐步按键、观察现象和调速位置见上方操作说明。

**Ctrl+空格**暂停，**Ctrl+K**查看阶段与下一步，**Ctrl+X**全部失能，**Ctrl+Q**退出。启动时自动打开预览，关闭预览窗口不会停止控制。操作时只运行一个控制入口。

## 仿真手动测试

```powershell
python .\codes\Rolling_Arc_Debug_Mujoco.py
```

使用相同的Ctrl按键。在该文件顶部修改`SCENE / START_X_M / SIM_FIRST_JOINT / SIM_LAST_JOINT`；仿真幅值和频率为`SIM_AMPLITUDE_DEG / SIM_OMEGA_RAD_S`。仿真使用所加载模型的实际尺寸，实机使用实机文件顶部尺寸。

| 测试场景 | 配置 |
| --- | --- |
| 直管，手调ARC与往返切换 | `SCENE='straight'` |
| 110→210→110 mm变径直管 | `SCENE='length_taper_return'`，`START_X_M=-.35` |
| 110→210 mm连续90°急弯变径 | `SCENE='elbow90_taper110_210'`，`START_X_M=-.25` |
| T管，主管/支管110/60 mm | `SCENE='tee_up60'`，`START_X_M=.2` |

仿真中按住**Ctrl+A或Ctrl+D**侧拉悬垂端，松开停止；**Ctrl+[或Ctrl+]**调拉力，**Ctrl+R**重建初始预览。

更详细的专题操作：[两侧长度调整](docs/曲线积分调长实机验收.md)、[ARC→Helix与T管](docs/ARC_Helix与重力矩越障操作说明.md)、[Helix→ARC](docs/Helix_ARC实机与热键调试说明.md)、[仿真实机统一调试](docs/仿真实机统一调试说明.md)。
