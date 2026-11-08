"""Run the separate curve-integral MuJoCo experiment with the existing keys."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.curve_feeding.backend import install_in_debug_process

# Modify experiment settings HERE; production entry files stay untouched.
SCENE = 'straight'
START_X_M = -.5
FIRST_JOINT, LAST_JOINT = 7, 12
AMPLITUDE_DEG = 60.
OMEGA_RAD_S = .6


if __name__ == '__main__':
    debug = install_in_debug_process()
    debug.SCENE, debug.START_X_M = SCENE, START_X_M
    debug.SIM_FIRST_JOINT, debug.SIM_LAST_JOINT = FIRST_JOINT, LAST_JOINT
    debug.SIM_AMPLITUDE_DEG, debug.SIM_OMEGA_RAD_S = AMPLITUDE_DEG, OMEGA_RAD_S
    print('实验入口：目标由拼接曲线积分生成；主仿真/主实机文件未修改。', flush=True)
    debug.main()
