"""Explicitly refresh the standalone geometry copy from the verified simulator.

Only the marked 3B region is replaced. Real controller/bus/hotkeys stay editable.
Not imported at runtime; no MuJoCo dependency enters the standalone real file.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def declarations(path):
    source = path.read_text(encoding='utf-8')
    lines = source.splitlines()
    result = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = min([node.lineno]+[d.lineno for d in node.decorator_list])-1
            result.append('\n'.join(lines[start:node.end_lineno]))
    return '\n\n\n'.join(result)


def main():
    target = ROOT/'codes/Rolling_Arc_Real.py'
    text = target.read_text(encoding='utf-8')
    begin, end = '# BEGIN FIXED_SECTION_CORE', '# END FIXED_SECTION_CORE'
    before, rest = text.split(begin, 1)
    _, after = rest.split(end, 1)
    body = '\n\n'.join(declarations(ROOT/'snake_control'/name)
                       for name in ('section_geometry.py', 'fixed_section_compliance.py'))
    target.write_text(before+begin+'\n'+body+'\n'+end+after, encoding='utf-8')
    print('Standalone fixed-section geometry copied; runtime remains one file.')


if __name__ == '__main__':
    main()
