# Snake_ARC

研究蛇形机器人在管道外的局部圆弧滚动：使能圆弧及交接所需关节，其余关节失能。仿真手调与单文件实机程序共用基础圆弧、固定截面顺应、仅角度反馈的自动移窗与数量自适应、主动送链调长，以及Ctrl热键。实机程序附带实测全身构型与使能预览，真实运动效果待用户验证。

## 启动环境

沿用参考项目已有的`mujoco` Conda环境。在Anaconda Prompt或已配置Conda的PowerShell中执行：

```powershell
conda activate mujoco
cd "D:\Desktop\SNAKE\Paper Code\Mujoco\Snake_ARC"
```

## 仿真手动调试

```powershell
python .\codes\Rolling_Arc_Debug_Mujoco.py
```

这是供人工操作的版本，与实机共用基本控制和Ctrl热键。场景、位置与搭管关节在该文件顶部改`SCENE / START_X_M / SIM_FIRST_JOINT / SIM_LAST_JOINT`，仿真幅值与频率改`SIM_AMPLITUDE_DEG / SIM_OMEGA_RAD_S`。普通字母/空格不控制步态，也不触发MuJoCo原生键盘菜单。

**按住Ctrl+A/D**拉动左/右悬垂端，松开停止；**Ctrl+[/]**调扰动力，**Ctrl+R**重新开始初始预览。文件顶部`SLIDING_FEEDBACK='contact'`可切回历史仿真接触反馈版本。

110→210→110 mm直管设`SCENE='length_taper_return'`、`START_X_M=-.35`；110→210 mm连续90°急弯设`SCENE='elbow90_taper110_210'`、`START_X_M=-.25`。

## 实机测试

直接运行一份Python文件控制舵机，无需填写启动参数或使用外部可视化程序。默认自动识别CH340，日志自动保存到`codes/logs/`：

```powershell
python .\codes\Rolling_Arc_Real.py
```

两端操作：**Ctrl+U**形成圆弧，等待形成与静置完成；**Ctrl+1/2/3**切换基础/固定顺应/自动滑窗，**Ctrl+W**开关自动选窗，**Ctrl+G**连续/暂停，**Ctrl+J/H**正/反单步，**Ctrl+E**顺应开关，**Ctrl+N**冻结目标，**Ctrl+X**全部失能，**Ctrl+Q**退出。默认J8–J13使能。默认开启DIY调参，可修改顶部幅值、频率、窗口与步态公式后直接运行，三模式不再因60°/0.6标定或开放形态门槛拒绝接入。两端使用全局Ctrl热键，操作时运行其中一个入口。

失能状态下，**Ctrl+3**按全身实测姿态选择初始圆弧并接管，形成后进入自动滑窗顺应；无法定位圆弧时保持失能并提示原因。**Ctrl+U**始终按配置窗口形成，X→U不会沿用上次扩大的窗口。**模式4→3会重新按实测选窗，不继承送链缓冲使能；成功后按Ctrl+G滚动。**其他已使能模式按Ctrl+3接入当前窗口的自动滑窗，Ctrl+W同时开关自动位置与数量选择。两端启动全20关节截面预览，蓝色使能、紫色失能，按估计重力朝下定向；关闭预览不停止控制，实机`SHOW_PREVIEW=False`、仿真`SHOW_SHAPE_PREVIEW=False`可关闭。

**主动送链调长**：形成完成后按**Ctrl+4**，每按一次**Ctrl+J/H**执行一段头侧增长/缩短，到位保持，不自动往返。默认请求64.5 mm，**Ctrl+-/=**减半/加倍；到可达边界停止同向推进，反向键可退出。**Ctrl+G**暂停/继续，**Ctrl+L**推进一帧，**Ctrl+Y**手动返回本步起态，**Ctrl+X**全部失能。仿真同键位；参数在实机文件顶部`FEED_*`修改。[完整操作说明](docs/主动送链实机验收说明.md)。

**ARC→Helix**：搭管圆弧稳定后按 **Ctrl+5**，此时保持现状。**Ctrl+J** 逐步推进，**Ctrl+G** 连续或暂停，**Ctrl+空格** 暂停，**Ctrl+K** 查看阶段和两侧长度估计。第一次推进时按最新关节角重新判断是否先调长。实机和仿真手调同键位。[三种起态的仿真记录](validation/reports/arc_helix_motor_feedback/README.md)

也可在使用`mujoco`解释器的IDE中直接运行该文件。需要固定端口时修改文件顶部的`PORT`；不连接机器人演练时将顶部`DRY_RUN`改为`True`，再运行同一文件。运行前关闭其他占用串口的程序。

## 自动实验

自动研究实验与手调入口分开保留。例如无图形窗口运行基础急弯：

```powershell
python .\codes\Rolling_Arc_Mujoco.py --headless --scene elbow90 --start-x -0.25 --duration 60 --output validation/results/elbow90.json
```

完整操作见[两端统一调试说明](docs/仿真实机统一调试说明.md)及[实机调试说明](docs/实机调试代码说明.md)；其他实验入口及记录见[开发与验证记录](docs/开发与验证记录.md)。
