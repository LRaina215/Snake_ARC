"""Copy the verified angle-only math into the existing standalone real file.

This is a development-time operation, never a real-runtime dependency.
Run explicitly when a verified research revision is authorized for sync.
--preview only writes a draft.
The surrounding serial driver, calibrated settings and editable controller
are preserved; no MuJoCo/simulation declarations enter the embedded region.
"""
import argparse
import ast
import json
import pprint
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = '# BEGIN MOTOR_SLIDING_CORE', '# END MOTOR_SLIDING_CORE'


def declarations(path, names=None):
    source = path.read_text(encoding='utf8')
    lines, blocks = source.splitlines(), []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (names is None or node.name in names):
            start = min([node.lineno]+[d.lineno for d in node.decorator_list])-1
            blocks.append('\n'.join(lines[start:node.end_lineno]))
    return '\n\n\n'.join(blocks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    target = ROOT/'codes/Rolling_Arc_Real.py'
    body = declarations(ROOT/'snake_control/rolling_arc.py', {'ArcParameters'})
    for name in ('angle_shape_observer.py', 'motor_shape_observer.py', 'motor_feedback_sliding.py'):
        body += '\n\n\n'+declarations(ROOT/'snake_control'/name)
    body += '\n\n\n'+declarations(ROOT/'snake_control/adaptive_arc_length.py', {'HalfArcParameters', 'circle_shape'})
    for name in ('feature_arc.py', 'rolling_shape_observer.py', 'phase_balanced_count.py', 'posture_window.py'):
        body += '\n\n\n'+declarations(ROOT/'snake_control'/name)
    reference = json.loads((ROOT/'configs/motor_shape_reference.json').read_text(encoding='utf8'))
    body += '\n\nMOTOR_PHASE_REFERENCE = '+pprint.pformat(reference, width=100, sort_dicts=False)+'\n'
    reference8 = json.loads((ROOT/'configs/motor_shape_reference8.json').read_text(encoding='utf8'))
    body += '\nMOTOR_PHASE_REFERENCE8 = '+pprint.pformat(reference8, width=100, sort_dicts=False)+'\n'
    source = target.read_text(encoding='utf8')
    if BEGIN in source:
        before, rest = source.split(BEGIN, 1)
        _, after = rest.split(END, 1)
    else:
        anchor = '# ==================== 4. 飞特串口协议'
        before, after = source.split(anchor, 1)
        after = '\n\n'+anchor+after
    output = before+BEGIN+'\n'+body+'\n'+END+after
    if args.preview:
        destination = ROOT/'tmp/Rolling_Arc_Real_motor_core_preview.py'
        destination.parent.mkdir(parents=True, exist_ok=True)
    else:
        destination = target
    destination.write_text(output, encoding='utf8')
    print(destination)


if __name__ == '__main__':
    main()
