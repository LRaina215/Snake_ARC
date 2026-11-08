"""Embed the verified curve feeder without regenerating other real cores.

Run with the existing mujoco Python. The resulting Real.py has NO runtime
repository or MuJoCo dependency. Only this marked block and its short status
label are updated; manual calibration/configuration and serial code survive.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_real_feeding_core import declaration

BEGIN, END = '# BEGIN CURVE_FEEDING_CORE', '# END CURVE_FEEDING_CORE'


def embed(source):
    shape = declaration(ROOT/'experiments/curve_feeding/curve_profile.py')
    shape = re.sub(r'\bSegment\b', 'HelixCurveSegment', shape)
    adapter = declaration(ROOT/'experiments/curve_feeding/backend.py',
                          {'_CurveTemplateHelper', 'curve_feeding_class'})
    block = (BEGIN+'\n'
        '# 模式4目标生成：拼接曲率子段 → 相邻连杆积分 → 目标角。\n'
        '# J/H任务、角反馈、长度估计、预置使能与退出逻辑继承前面的原控制器。\n'
        '# 改形状分配看FeedingCurveBuilder；改速度/步长仍看顶部FEED_*。\n'
        +shape+'\n\n\n'+adapter+'\n\n'
        '# 保存旧类仅用于离线对照；实机所有调长实例默认用曲线积分。\n'
        'LegacyContinuousFeeding = ContinuousFeeding\n'
        'ContinuousFeeding = curve_feeding_class(LegacyContinuousFeeding)\n'
        +END)
    if BEGIN in source:
        before, rest = source.split(BEGIN, 1)
        _, after = rest.split(END, 1)
        source = before+block+after
    else:
        anchor = '# END ARC_HELIX_CORE'
        if source.count(anchor) != 1:
            raise ValueError('Expected one ARC/Helix curve definition block')
        source = source.replace(anchor, anchor+'\n\n'+block, 1)
    source = source.replace(
        '# 主动送链：只用角度/时间，暂停滚动，首次按旧版模板建形；不是管径顺应。',
        '# 主动送链：只用角度/时间，暂停滚动；目标由拼接曲线积分生成，操作与交接不变。')
    source = source.replace(
        'Ctrl+4 主动送链；Ctrl+J/H 各执行一段头侧增长/缩短，Ctrl+Y 提前返回，Ctrl+L沿任务方向强制推进一步。',
        'Ctrl+4 曲线积分主动送链；Ctrl+J/H 各执行一段头侧增长/缩短，Ctrl+Y 提前返回，Ctrl+L沿任务方向强制推进一步。')
    anchor = '            core = self.feeding\n            o = {} if core is None else core.observation or {}'
    replacement = ('            core = self.feeding\n'
        "            print('目标生成=拼接曲线积分 | J/H手动相对调长，G暂停/继续，L强制一帧', flush=True)\n"
        '            o = {} if core is None else core.observation or {}')
    if anchor in source:
        source = source.replace(anchor, replacement, 1)
    return source


def main():
    target = ROOT/'codes/Rolling_Arc_Real.py'
    source = target.read_text(encoding='utf-8')
    target.write_text(embed(source), encoding='utf-8')
    print('Curve feeding embedded. Real serial entry, hardware parameters and hotkeys preserved.')


if __name__ == '__main__':
    main()
