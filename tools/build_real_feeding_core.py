"""Explicitly copy active feeding math into the existing standalone file.

Only the new marked region is replaced. Serial calibration, manual edits,
hotkeys and other gait cores are preserved. No runtime repository imports.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = '# BEGIN ACTIVE_FEEDING_CORE', '# END ACTIVE_FEEDING_CORE'


def declaration(path, names=None):
    source = path.read_text(encoding='utf8')
    lines = source.splitlines()
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and (names is None or n.name in names)]
    return '\n\n\n'.join('\n'.join(lines[min([n.lineno]+[d.lineno for d in n.decorator_list])-1:n.end_lineno])
                            for n in nodes)


def main():
    source = (ROOT/'snake_control/active_arc_transfer.py').read_text(encoding='utf8')
    lines = source.splitlines()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name=='ActiveArcTransfer')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('mask', 'template')]
    geometry = ('class FeedingArcGeometry:\n'
        '    """Original transfer mask/template; no unused observer or constructor gates."""\n'
        '    def __init__(self, lengths):\n'
        '        self.n = len(lengths)-1\n\n'+
        '\n\n'.join('\n'.join(lines[n.lineno-1:n.end_lineno]) for n in methods))
    controller = declaration(ROOT/'snake_control/continuous_feeding.py')
    old = ("self.helper = ActiveArcTransfer(self.lengths, 'transfer', TransferOptions(\n"
        '            first_joint=options.first_joint, last_joint=options.last_joint,\n'
        '            amplitude_deg=options.amplitude_deg, hold_phase_rad=0.))')
    if controller.count(old)!=1:
        raise ValueError('Research helper changed; inspect this explicit sync before updating')
    controller = controller.replace(old, 'self.helper = FeedingArcGeometry(self.lengths)')
    body = geometry+'\n\n\n'+declaration(ROOT/'snake_control/whole_feeding_observer.py')+'\n\n\n'+controller
    target = ROOT/'codes/Rolling_Arc_Real.py'
    source = target.read_text(encoding='utf8')
    if BEGIN in source:
        before, rest = source.split(BEGIN, 1)
        _, after = rest.split(END, 1)
    else:
        anchor = '# ==================== 4. 飞特串口协议'
        before, after = source.split(anchor, 1)
        before += '# ==================== 3C. 主动送链核心：角度/时间输入，可直接调参 ====================\n'
        after = '\n\n'+anchor+after
    target.write_text(before+BEGIN+'\n'+body+'\n'+END+after, encoding='utf8')
    print('Active feeding math embedded; surrounding manual/serial code preserved.')


if __name__ == '__main__':
    main()
