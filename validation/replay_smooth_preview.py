"""Replay ONE existing archive through real/manual adapters and their shared preview.

No MuJoCo stepping or serial connection. Check actual cached controller curves,
target/mask parity, JSON transport, display isolation; render the whole record.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import gzip
import io
import json
import time
from contextlib import redirect_stdout
from types import SimpleNamespace
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from codes import Rolling_Arc_Real as real
from codes import Rolling_Arc_Debug_Mujoco as debug
from validation.test_repeated_motor_port import AngleBus
from validation.test_posture_port import references

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'validation/reports/reconstructed_section_preview'


def main(path):
    status = json.loads((path/'status.json').read_text(encoding='utf8'))
    p = real.HalfArcParameters(**status['parameters'])
    session = SimpleNamespace(sim=SimpleNamespace(p=p), elapsed=0., record_event=lambda *a, **k: None)
    buses = [AngleBus(), AngleBus()]
    controllers = [real.RollingArcController(buses[0]), debug.DebugArcController(buses[1], session)]
    for c, bus in zip(controllers, buses):
        c.sliding = real.PostureWindow(p, real.LINK_LENGTHS_M, references(), feedback_period_s=.2)
        c.compliance = c.sliding.core
        c.first, c.last = p.first_joint, p.last_joint
        c.enabled = p.enabled.tolist()
        c.targets = np.degrees(p.targets(p.initial_phase_rad)).tolist()
        c.mode, c.length_normalized, c.touched_hardware = 'hold', True, True
        bus.enabled = p.enabled.copy()
    REPORT.mkdir(parents=True, exist_ok=True)
    output = ROOT/'validation/results/reconstructed_section_preview'; output.mkdir(parents=True, exist_ok=True)
    figure, seen, writer = real.JointPreviewFigure(), set(), None
    summary = dict(source=path.name, control_frames=0, display_frames=0, cached_curve_frames=0,
                   maximum_target_error_rad=0., hardware_connections=0, new_dynamics=0, passed=False)
    try:
        with gzip.open(path/'actual_trace.jsonl.gz', 'rt', encoding='utf8') as stream:
            for row in map(json.loads, stream):
                f = row['motor_input']
                if f is None: continue
                t, ft = row['controller']['control_time_s'], f['feedback_time_s']
                session.elapsed = t
                with redirect_stdout(io.StringIO()):
                    for i, (c, bus) in enumerate(zip(controllers, buses)):
                        bus.angles = np.array(f['angles_deg'])
                        if i == 0:
                            c.states = bus.read_states()
                            c.feedback_phase_rad = f['phase_rad']
                            c.motor_sample_clock = lambda: (t, ft)
                        c.command_phase = f['phase_rad']; c.phase = row['controller']['phase_rad']
                        c.tick_motor_sliding(.02)
                        assert not c.sliding_fault, (t, c.section_block_reason)
                        assert c.enabled == row['enabled']
                        error = float(abs(np.radians(c.targets)-row['target_rad']).max())
                        summary['maximum_target_error_rad'] = max(error, summary['maximum_target_error_rad'])
                        assert error < 1e-9, (t, error)
                summary['control_frames'] += 1
                if ft in seen: continue
                seen.add(ft)
                for c in controllers: c.feedback_wall_s = time.monotonic()
                payloads = [real.preview_payload(c) for c in controllers]
                assert payloads[0]['smooth_section'] == payloads[1]['smooth_section']
                payload = json.loads(json.dumps(payloads[0]))  # Actual process-safe data only.
                # Replay injects archived phase while adapters are in hold. Label the
                # recorded motion, not the offline harness's phase-driving mechanism.
                payload['mode'] = '回放·静置' if row['stage'] == 'settle' else '回放·暂停' if row['controller']['phase_hold'] else '回放·滚动'
                payload['gait'] = '已有仿真记录'
                core = controllers[0].compliance
                if core.geometry is not None:
                    summary['cached_curve_frames'] += 1
                    # The sampled measured curve must equal the archive's actual control snapshot.
                    np.testing.assert_allclose(real.sample_section(core.geometry['curve']),
                        row['controller']['compliance']['section_curve_m'], atol=1e-9)
                    for name, geometry in (('measured', core.geometry), ('candidate', core.target_geometry)):
                        expected = real.sample_section(geometry['curve'])@geometry['axis']['basis'][:, :2].T+geometry['axis']['point']
                        np.testing.assert_allclose(real.preview_smooth_points(payload['smooth_section'][name]), expected, atol=1e-12)
                before = copy.deepcopy((controllers[0].targets, controllers[0].enabled, controllers[0].phase,
                                        real.preview_payload(controllers[0])['smooth_section'], buses[0].target.tolist()))
                figure.update(payload); figure.figure.canvas.draw()
                after = (controllers[0].targets, controllers[0].enabled, controllers[0].phase,
                         real.preview_payload(controllers[0])['smooth_section'], buses[0].target.tolist())
                assert before == after
                if core.geometry is not None:
                    assert set(figure.smooth_display) == {'measured', 'candidate'}
                    np.testing.assert_allclose(figure.section_basis, core.geometry['axis']['basis'][:, :2], atol=1e-12)
                    np.testing.assert_allclose(figure.section_origin, core.geometry['axis']['point'], atol=1e-12)
                    # Crucial: the drawn measured curve IS the controller's 2D section,
                    # not another plane chosen using gravity or global Z.
                    np.testing.assert_allclose(figure.smooth_display['measured'], real.sample_section(core.geometry['curve']), atol=1e-12)
                    candidate_in_measured_plane = (real.preview_smooth_points(payload['smooth_section']['candidate'])-figure.section_origin)@figure.section_basis
                    np.testing.assert_allclose(figure.smooth_display['candidate'], candidate_in_measured_plane, atol=1e-12)
                pixels = np.asarray(figure.figure.canvas.buffer_rgba())[:, :, :3]
                if writer is None:
                    writer = cv2.VideoWriter(str(output/'smooth_section_preview.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                             5., (pixels.shape[1], pixels.shape[0]))
                    assert writer.isOpened()
                writer.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
                summary['display_frames'] += 1
                if round(ft, 6) in (8., 11.8, 14., 24.8):
                    figure.figure.savefig(REPORT/f'preview_{ft:.1f}s.png', dpi=110)
        # Display fallbacks: missing cache is explicit, ordinary gait is marked reconstructed.
        empty = copy.deepcopy(payload)
        empty['smooth_section']['measured'] = empty['smooth_section']['candidate'] = None
        figure.update(empty); assert '等待控制器更新' in figure.note.get_text()
        empty['smooth_section'] = None
        figure.update(empty); assert '当前未参与顺应控制' in figure.note.get_text()
        assert set(figure.smooth_display) == {'measured', 'candidate'}
        assert not hasattr(figure, 'gravity') and not hasattr(figure, 'view_button')
        summary.update(passed=True, adapter_curve_parity=True, control_curves_match_archive=True,
                       display_does_not_mutate_control=True, fallback_labels_checked=True,
                       actual_controller_plane_used=True, displayed_measured_curve_equals_controller_2d=True)
    finally:
        if writer is not None: writer.release()
        figure.plt.close(figure.figure)
        (REPORT/'review.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
