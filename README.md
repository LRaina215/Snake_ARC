# Snake_ARC

研究蛇形机器人在管道外的局部圆弧滚动：仅使能形成圆弧的关节，其余关节失能。包含基础步态、固定截面顺应和滑动窗口的MuJoCo仿真，以及单文件实机控制程序。实机默认基础圆弧，可用热键开启固定截面顺应；滑动窗口目前用于仿真。

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

这是供人工操作的版本，与实机共用控制器和Ctrl热键。场景与初始位置在该文件顶部改`SCENE / START_X_M`，步态参数在`Rolling_Arc_Real.py`顶部修改。普通字母/空格不控制步态，也不触发MuJoCo原生键盘菜单。

仿真额外支持**Ctrl+1/2/3**切换基础圆弧/固定截面顺应/滑动窗口顺应；**按住Ctrl+A/D**拉动左/右悬垂端，松开即停止，用实际本体移动测试自动移窗；**Ctrl+[/]**调扰动力，**Ctrl+R**重新开始初始预览。

## 实机测试

直接运行一份Python文件控制舵机，无需填写启动参数或使用外部可视化程序。默认自动识别CH340，日志自动保存到`codes/logs/`：

```powershell
python .\codes\Rolling_Arc_Real.py
```

两端共用的基本操作：**Ctrl+U**形成基础圆弧，等待形成与静置完成；**Ctrl+J/H**正/反单步，**Ctrl+G**连续/暂停，**Ctrl+E**开/关顺应，**Ctrl+空格**暂停相位，**Ctrl+N**冻结目标，**Ctrl+X**全部失能，**Ctrl+Q**退出。默认使能J8–J13，其余14关节失能。两端使用全局热键，操作时运行其中一个入口。

也可在使用`mujoco`解释器的IDE中直接运行该文件。需要固定端口时修改文件顶部的`PORT`；不连接机器人演练时将顶部`DRY_RUN`改为`True`，再运行同一文件。运行前关闭其他占用串口的程序。

## 自动实验

自动研究实验与手调入口分开保留。例如无图形窗口运行基础急弯：

```powershell
python .\codes\Rolling_Arc_Mujoco.py --headless --scene elbow90 --start-x -0.25 --duration 60 --output validation/results/elbow90.json
```

完整操作见[两端统一调试说明](docs/仿真实机统一调试说明.md)及[实机调试说明](docs/实机调试代码说明.md)；其他实验入口及记录见[开发与验证记录](docs/开发与验证记录.md)。
