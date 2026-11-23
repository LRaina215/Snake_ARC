"""Embed the shared vertical-ring controller in the standalone real entry."""
import ast
from pathlib import Path

from build_real_feeding_core import declaration


ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = '# BEGIN VERTICAL_RING_CORE', '# END VERTICAL_RING_CORE'


def method(path, class_name, method_name):
    source = path.read_text(encoding='utf-8')
    owner = next(node for node in ast.parse(source).body
                 if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in owner.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name)
    return '\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno])


def main():
    source = ROOT / 'snake_control'
    # Keep the bundled Ctrl+7 implementation intact. The vertical mode needs
    # two shared methods newer than its standalone base classes.
    adapter = ('class VerticalSampledCurveGait(CurveGait):\n'
               + method(source / 'curve_gait.py', 'CurveGait', 'sampled_targets')
               + '\n\n\nclass VerticalRingReference(HelixArcTransition):\n'
               '    def __init__(self, angles_rad, lengths, options):\n'
               '        super().__init__(angles_rad, lengths, options)\n'
               '        self.curve = VerticalSampledCurveGait(self.n, self.l, lengths=self.lengths)\n\n'
               + method(source / 'helix_arc_transition.py', 'HelixArcTransition', 'apply_reference'))
    core = (declaration(source / 'acute_arc_transition.py')
            .replace('class AcuteArcTransition(HelixArcTransition):',
                     'class AcuteArcTransition(VerticalRingReference):')
            .replace('local_section(', 'vertical_local_section(')
            .replace('        from .section_reach import fit_section_turn\n', ''))
    # The horizontal entry retains its current section API. The vertical fit
    # also accepts a restricted material interval and a projection direction.
    section_geometry = declaration(source / 'section_geometry.py', {
        'local_section', 'section_circle', 'section_curvature_field', 'section_integral', 'section_outline',
    }).replace('def local_section(', 'def vertical_local_section(')
    parts = [declaration(source / 'curve_gait.py', {'helix_turn_length'}),
             declaration(source / 'bent_helix_curve.py'),
             section_geometry,
             declaration(source / 'section_effort.py'),
             declaration(source / 'section_reach.py'),
             declaration(source / 'section_ring_curve.py'),
             adapter, core,
             declaration(source / 'vertical_ring_manual.py')]
    target = ROOT / 'codes/Rolling_Arc_Real.py'
    body = target.read_text(encoding='utf-8')
    if BEGIN in body:
        before, block = body.split(BEGIN, 1)
        _, after = block.split(END, 1)
    else:
        before, after = body.split('# END ARC_HELIX_CORE', 1)
        before += '# END ARC_HELIX_CORE\n\n'
    target.write_text(before + BEGIN + '\n' + '\n\n\n'.join(parts) + '\n' + END + after,
                      encoding='utf-8')


if __name__ == '__main__':
    main()
