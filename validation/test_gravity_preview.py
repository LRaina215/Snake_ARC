"""Two focused checks: embedded estimator parity and display-only state isolation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import gzip
import json
import time
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from snake_control.gravity_preview import DualEndGravityPreview
from codes import Rolling_Arc_Real as real
from validation.test_repeated_motor_port import AngleBus

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'validation/reports/gravity_preview'


class GravityPreviewTests(unittest.TestCase):
    def test_full_archive_embedded_parity_new_sample_rule_and_rotation(self):
        path = next((ROOT/'validation/reports/phase_balanced_window').glob('return_repair_*/actual_trace.jsonl.gz'))
        module, embedded = DualEndGravityPreview([.0645]*21), real.DualEndGravityPreview([.0645]*21)
        seen, states = set(), set()
        with gzip.open(path,'rt',encoding='utf8') as stream:
            for row in map(json.loads,stream):
                f = row['motor_input']
                if not f or f['feedback_time_s'] in seen: continue
                seen.add(f['feedback_time_s'])
                args=(f['angles_deg'],row['controller']['window'],f['feedback_time_s'])
                a,b=module.update(*args),embedded.update(*args)
                self.assertEqual(a,b)
                self.assertEqual(copy.deepcopy(a),module.update(*args))
                states.add(a['state'])
                if a['basis'] is not None:
                    basis=np.array(a['basis'])
                    np.testing.assert_allclose(basis@basis.T,np.eye(3),atol=1e-10)
                    np.testing.assert_allclose(basis@a['down_head'],[0,0,-1],atol=1e-10)
        self.assertTrue({'dual','single_degraded','conflict_hold'} <= states)
        # No valid ends: explicitly retain rather than accepting a fabricated vertical.
        held=module.update([0.]*20,(8,13),module.last_time+.2)
        self.assertEqual(held['state'],'unobservable_hold')
        self.assertFalse(held['reliable'])

    def test_display_toggle_and_incomplete_feedback_do_not_modify_control(self):
        path=next((REPORT/'physics').glob('*/actual_trace.jsonl.gz'))
        with gzip.open(path,'rt',encoding='utf8') as f:
            r=next(r for r in map(json.loads,f) if r['time_s']>=8.)
        bus=AngleBus();bus.angles=np.array(r['motor_input']['angles_deg'])
        c=real.RollingArcController(bus)
        c.enabled=r['enabled'];c.targets=np.degrees(r['target_rad']).tolist()
        c.touched_hardware=True;c.refresh_compliance_feedback()
        before=copy.deepcopy((c.enabled,c.targets,c.phase,c.mode))
        row=real.preview_payload(c)
        figure=real.JointPreviewFigure()
        try:
            figure.update(row)
            self.assertIsNotNone(figure.gravity.result['basis'])
            figure.toggle_view(None)
            self.assertFalse(figure.gravity_align)
            figure.toggle_view(None)
            self.assertTrue(figure.gravity_align)
            stamp=figure.gravity.last_time
            figure.update(row)
            self.assertEqual(figure.gravity.last_time,stamp)
            self.assertEqual((c.enabled,c.targets,c.phase,c.mode),before)
            self.assertFalse(hasattr(bus,'target'))
            # A stale display frame must not masquerade as a new gravity observation.
            row['feedback_wall_s']=time.monotonic()-2.
            row['gravity_time_s']+=1.
            figure.update(row)
            self.assertEqual(figure.gravity.last_time,stamp)
        finally:
            figure.plt.close(figure.figure)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GravityPreviewTests))
    (REPORT/'software_checks.json').write_text(json.dumps(dict(passed=result.wasSuccessful(), tests=result.testsRun,
        hardware_connections=0, additional_dynamics=0),indent=2),encoding='utf8')
    sys.exit(0 if result.wasSuccessful() else 1)
