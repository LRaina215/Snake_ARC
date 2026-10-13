# Snake_ARC

研究蛇形机器人在管道外的局部圆弧滚动：仅使能形成圆弧的关节，其余关节失能。仿真手调与单文件实机程序共用基础圆弧、固定截面顺应、仅角度反馈的6关节自动滑窗，以及Ctrl热键。另含[连续滚动交接](docs/连续滚动交接设计.md)和[相位均衡的自主窗口位置与数量选择](docs/相位均衡数量判断设计.md)研究原型，尚未接入实测入口。实机运动效果待用户验证。

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

## 实机测试

直接运行一份Python文件控制舵机，无需填写启动参数或使用外部可视化程序。默认自动识别CH340，日志自动保存到`codes/logs/`：

```powershell
python .\codes\Rolling_Arc_Real.py
```

两端操作：**Ctrl+U**形成圆弧，等待形成与静置完成；**Ctrl+1/2/3**切换基础/固定顺应/自动滑窗，**Ctrl+W**开关自动选窗，**Ctrl+G**连续/暂停，**Ctrl+J/H**正/反单步，**Ctrl+E**顺应开关，**Ctrl+N**冻结目标，**Ctrl+X**全部失能，**Ctrl+Q**退出。默认J8–J13使能。自动滑窗使用60°、0.6 rad/s正向基准，该模式不接受反向单步或调幅值/频率。两端使用全局Ctrl热键，操作时运行其中一个入口。

也可在使用`mujoco`解释器的IDE中直接运行该文件。需要固定端口时修改文件顶部的`PORT`；不连接机器人演练时将顶部`DRY_RUN`改为`True`，再运行同一文件。运行前关闭其他占用串口的程序。

## 自动实验

自动研究实验与手调入口分开保留。例如无图形窗口运行基础急弯：

```powershell
python .\codes\Rolling_Arc_Mujoco.py --headless --scene elbow90 --start-x -0.25 --duration 60 --output validation/results/elbow90.json
```

完整操作见[两端统一调试说明](docs/仿真实机统一调试说明.md)及[实机调试说明](docs/实机调试代码说明.md)；其他实验入口及记录见[开发与验证记录](docs/开发与验证记录.md)。
