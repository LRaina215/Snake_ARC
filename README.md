# Snake_ARC

研究蛇形机器人在管道外的局部圆弧滚动：仅使能形成圆弧的关节，其余关节失能。包含基础步态、固定截面顺应和滑动窗口的MuJoCo仿真，以及单文件实机控制程序。实机默认基础圆弧，可用热键开启固定截面顺应；滑动窗口目前用于仿真。

## 启动环境

沿用参考项目已有的`mujoco` Conda环境。在Anaconda Prompt或已配置Conda的PowerShell中执行：

```powershell
conda activate mujoco
cd "D:\Desktop\SNAKE\Paper Code\Mujoco\Snake_ARC"
```

## 仿真测试

```powershell
# 基础圆弧：直管
python .\codes\Rolling_Arc_Mujoco.py --scene straight --duration 30

# 固定截面顺应：连续90°柔弯内变径
python .\codes\Rolling_Arc_Section_Mujoco.py --scene taper_true90 --start-x -0.35 --duration 75

# 滑动窗口：110→210 mm连续90°急弯变径
python .\codes\Rolling_Arc_Sliding_Section_Mujoco.py --scene elbow90_taper110_210 --start-x -0.25 --duration 150
```

基础入口还支持`--scene bend90`（柔弯）和`--scene elbow90`（连续急弯）。三个查看器均用空格暂停/恢复滚动相位；固定截面用E切换顺应增益、R反向，滑动窗口用A/D请求头/尾交接、U切换自动选窗、R反向。

无图形窗口测试基础步态：

```powershell
python .\codes\Rolling_Arc_Mujoco.py --headless --scene elbow90 --start-x -0.25 --duration 60 --output validation/results/elbow90.json
```

## 实机测试

一份Python文件直接控制舵机，无需外部可视化程序。按实际连接修改`--port`，并关闭其他占用串口的程序：

```powershell
python .\codes\Rolling_Arc_Real.py --port COM9 --log logs/rolling_arc_real.jsonl
```

启动后按**Ctrl+U**形成基础圆弧，等待形成与静置完成；**Ctrl+J/H**正/反单步，**Ctrl+G**连续/暂停，**Ctrl+E**开启/关闭固定截面顺应，**Ctrl+X**全部失能，**Ctrl+Q**退出。默认固定使能J8–J13，其余14关节失能。

不连接机器人时，可用内存串口演练同一操作流程：

```powershell
python .\codes\Rolling_Arc_Real.py --dry-run
```

完整热键及调参见[实机调试说明](docs/实机调试代码说明.md)；实现、环境和实验资料见[开发与验证记录](docs/开发与验证记录.md)。
