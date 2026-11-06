"""Copy the verified ARC→Helix motor-only core into the standalone entry."""
from pathlib import Path
import re

from build_real_feeding_core import declaration


ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = '# BEGIN ARC_HELIX_CORE', '# END ARC_HELIX_CORE'


def main():
    src = ROOT/'snake_control'
    curve = declaration(src/'curve_gait.py', {'Segment', 'line', 'arc', 'helix', 'CurveGait'})
    curve = re.sub(r'\bSegment\b', 'HelixCurveSegment', curve)
    parts = [curve, declaration(src/'gravity_global_shift.py', {'ArcHelixShift'}),
             declaration(src/'active_arc_transfer.py'), declaration(src/'arc_helix_transition.py')]
    target = ROOT/'codes/Rolling_Arc_Real.py'
    source = target.read_text(encoding='utf-8')
    if BEGIN in source:
        before, rest = source.split(BEGIN, 1)
        _, after = rest.split(END, 1)
    else:
        anchor = '# ==================== 4. 飞特串口协议'
        before, after = source.split(anchor, 1)
        after = '\n\n'+anchor+after
    target.write_text(before+BEGIN+'\n'+'\n\n\n'.join(parts)+'\n'+END+after, encoding='utf-8')


if __name__ == '__main__':
    main()
