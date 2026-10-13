"""Reproduce reported parameter/mode failures with an in-memory bus; no hardware/physics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
from contextlib import redirect_stdout
import io
import json
import time
import numpy as np
from codes import Rolling_Arc_Real as real
from codes import Rolling_Arc_Debug_Mujoco as debug
from validation.test_repeated_motor_port import AngleBus


class Bus(AngleBus):
    def prepare_enabled(self, angles, enabled, configure_mask=None):
        if configure_mask is None:
            configure_mask = np.asarray(enabled) & ~self.enabled
        super().prepare_enabled(angles, enabled, configure_mask)


def check(adapter, amplitude, omega, window):
    bus = Bus()
    p = real.ManualArcReference(first_joint=window[0], last_joint=window[1], amplitude_deg=amplitude, omega_rad_s=omega)
    session = SimpleNamespace(sim=SimpleNamespace(p=p), elapsed=0., stop_push=lambda: None,
                              record_event=lambda *a, **k: None)
    c = real.RollingArcController(bus) if adapter == 'real' else debug.DebugArcController(bus, session)
    clock = [0.]
    c.motor_sample_clock = lambda: (clock[0], clock[0])
    c.assign_parameters(f'window={window[0]}:{window[1]} amplitude={amplitude} omega={omega}')
    c.command('apply')
    for _ in range(1200):
        c.tick(.02)
        bus.angles = np.array(c.targets)
        if not c.forming and c.settle_remaining <= 0: break
    assert not c.forming and c.settle_remaining <= 0
    np.testing.assert_allclose(c.targets, np.degrees(p.targets(c.phase)), atol=1e-8)
    c.command('gait_fixed')
    assert c.compliance is not None and c.compliance.p.manual_tuning
    c.refresh_compliance_feedback(); c.command('run')
    for _ in range(20):
        clock[0] += .02; session.elapsed = clock[0]
        c.refresh_compliance_feedback(); c.tick(.02)
        bus.angles = np.array(c.targets)
        assert c.section_block_reason is None
    c.command('gait_sliding')
    assert isinstance(c.sliding.p, real.ManualArcReference)
    assert (c.sliding.p.amplitude_deg, c.sliding.p.omega_rad_s) == (amplitude, omega)
    np.testing.assert_allclose(c.sliding.p.targets(.3), p.targets(.3))
    for _ in range(30):
        clock[0] += .02; session.elapsed = clock[0]
        c.refresh_compliance_feedback(); c.tick(.02)
        bus.angles = np.array(c.targets)
        assert not c.sliding_fault, c.section_block_reason
        assert c.sliding.core.status == 'running'
    c.command('window_auto'); assert not c.sliding.automatic
    c.command('window_auto'); assert c.sliding.automatic
    c.change_parameters(amplitude=amplitude+2, omega=omega+.1)
    assert c.sliding.p.amplitude_deg == amplitude+2
    assert c.sliding.p.omega_rad_s == omega+.1
    c.command('reverse'); c.command('back')
    clock[0] += .02; session.elapsed = clock[0]; c.refresh_compliance_feedback(); c.tick(.02)
    assert not c.sliding_fault, c.section_block_reason
    c.command('release'); assert not any(bus.enabled)
    return dict(adapter=adapter, amplitude_deg=amplitude, omega_rad_s=omega, window=window,
                establish_fixed_sliding_and_hotkeys=True)


if __name__ == '__main__':
    real.validate_setup()
    records = []
    with redirect_stdout(io.StringIO()):
        for case in ((70., .6, (8, 13)), (70., .6, (5, 9)), (80., 1.1, (7, 14))):
            for adapter in ('real', 'manual_mujoco'):
                records.append(check(adapter, *case))
    # Nonfinite and unencodable values still fail instead of becoming malformed packets.
    for angle in (float('nan'), 500.):
        try: real.angle_to_raw(0, angle)
        except ValueError: pass
        else: raise AssertionError('Invalid protocol target was accepted')
    assert real.angle_to_raw(0, 100.) >= 0  # The old ±90 software clamp is disabled.
    result = dict(passed=True, cases=records, new_dynamics=0, hardware_connections=0,
                  scope='parameter startup, formation, mode acceptance and short command updates only')
    path = Path(__file__).resolve().parent/'reports/manual_diy'
    path.mkdir(parents=True, exist_ok=True)
    (path/'software_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))
