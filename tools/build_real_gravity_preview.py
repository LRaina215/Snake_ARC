"""Explicitly embed display-only angle math in the standalone real entry."""
from pathlib import Path
from build_real_motor_core import declarations

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    path = ROOT/'codes/Rolling_Arc_Real.py'
    source = path.read_text(encoding='utf8')
    begin, end = '# BEGIN GRAVITY_PREVIEW_CORE', '# END GRAVITY_PREVIEW_CORE'
    body = declarations(ROOT/'snake_control/gravity_preview.py')
    if begin in source:
        before, rest = source.split(begin, 1)
        _, after = rest.split(end, 1)
    else:
        anchor = 'def preview_payload(controller):'
        before, after = source.split(anchor, 1)
        after = '\n\n'+anchor+after
    path.write_text(before+begin+'\n'+body+'\n'+end+after, encoding='utf8')
