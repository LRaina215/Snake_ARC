"""Historical gravity estimator and current display-only state isolation."""
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
    def test_research_archive_new_sample_rule_and_rotation(self):
        path = next((ROOT/'validation/reports/phase_balanced_window').glob('return_repair_*/actual_trace.jsonl.gz'))
        module = DualEndGravityPreview([.0645]*21)
        self.assertTrue(hasattr(real, 'DualEndGravityPreview'))
        embedded = real.DualEndGravityPreview([.0645]*21)
        seen, states = set(), set()
        with gzip.open(path,'rt',encoding='utf8') as stream:
            for row in map(json.loads,stream):
                f = row['motor_input']
                if not f or f['feedback_time_s'] in seen: continue
                seen.add(f['feedback_time_s'])
                args=(f['angles_deg'],row['controller']['window'],f['feedback_time_s'])
                a=module.update(*args)
                self.assertEqual(a, embedded.update(*args))
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

    def test_reconstructed_plane_display_does_not_modify_control(self):
        path=next((REPORT/'physics').glob('*/actual_trace.jsonl.gz'))
        with gzip.open(path,'rt',encoding='utf8') as f:
            r=next(r for r in map(json.loads,f) if r['time_s']>=8.)
        bus=AngleBus();bus.angles=np.array(r['motor_input']['angles_deg'])
        c=real.RollingArcController(bus)
        c.first,c.last=r['controller']['window']
        c.enabled=r['enabled'];c.targets=np.degrees(r['target_rad']).tolist()
        c.touched_hardware=True;c.refresh_compliance_feedback()
        before=copy.deepcopy((c.enabled,c.targets,c.phase,c.mode))
        row=real.preview_payload(c)
        figure=real.JointPreviewFigure()
        try:
            figure.update(row)
            self.assertIsNotNone(figure.section_basis)
            basis = figure.section_basis.copy()
            figure.update(row)
            np.testing.assert_allclose(figure.section_basis, basis)
            self.assertEqual((c.enabled,c.targets,c.phase,c.mode),before)
            self.assertFalse(hasattr(bus,'target'))
            self.assertEqual(figure.body_points.shape, (22, 2))
            labels = {item.get_text() for item in figure.body.texts}
            self.assertTrue({f'J{i}' for i in range(1, 21)} <= labels)
            self.assertTrue({'头端', '尾端'} <= labels)
            from matplotlib.colors import to_rgba
            expected=[to_rgba('#147bb5' if e else '#9259b5') for e in c.enabled]
            np.testing.assert_allclose(figure.body.collections[0].get_facecolors(),expected)
            # A repaint cannot advance time, nor can stale feedback change gravity.
            gravity_before = copy.deepcopy(figure.gravity.result)
            row['feedback_wall_s']=time.monotonic()-2.
            row['gravity_time_s']+=1.
            figure.update(row)
            self.assertIn('陈旧', figure.header.get_text())
            self.assertEqual(figure.gravity.result, gravity_before)
            np.testing.assert_allclose(figure.section_basis, basis, atol=1e-12)
            self.assertIn('方向保持', figure.gravity_status)
        finally:
            figure.plt.close(figure.figure)

    def test_fit_basis_roll_and_sign_do_not_rotate_or_mirror_display(self):
        samples=json.loads((ROOT/'validation/reports/real_feeding_distance_20261021/task_19_fresh_samples.json').read_text(encoding='utf8'))
        sample=samples[0]
        window=sample['window']
        angles=sample['angles_deg']
        geometry=real.local_section(np.radians(angles), real.LINK_LENGTHS_M, *window)
        section=dict(source='control', window=window, updates=1,
                     measured=real.preview_section_geometry(geometry),
                     candidate=real.preview_section_geometry(geometry))
        row=dict(angles_deg=angles, target_deg=sample['target_deg'],
                 enabled=[window[0]<=i<=window[1] for i in range(1,21)],
                 smooth_section=section, gravity_time_s=sample['time_s'],
                 gravity_hint=dict(down_head=sample['down_head'],source=sample['proxy_source']),
                 enable_known=True, feedback_wall_s=time.monotonic(), window=window,
                 phase_rad=0.,mode='已有实机角度回放',gait='预览检查',feeding=None,
                 automatic=True,selection={},pending=None,observer_state='',reasons=[],
                 block_reason=None,lengths=list(real.LINK_LENGTHS_M))
        before=copy.deepcopy(row)
        figure=real.JointPreviewFigure()
        try:
            row['feedback_wall_s']=time.monotonic()
            before=copy.deepcopy(row)
            figure.update(row)
            expected=figure.body_points.copy()
            original_basis=np.array(section['measured']['basis'])
            np.testing.assert_allclose(figure.section_basis@figure.section_basis.T,
                                       original_basis@original_basis.T,atol=1e-12)
            down=np.array(figure.gravity.result['down_head'])@figure.section_basis
            self.assertLess(down[1],0.)
            self.assertAlmostEqual(down[0],0.,places=12)
            for change in (np.array([[0.,-1.],[1.,0.]]),np.diag([1.,-1.])):
                # Re-express the exact same control curve in another fit basis.
                changed=copy.deepcopy(row)
                g=changed['smooth_section']['measured']
                g['basis']=(original_basis@change).tolist()
                for segment in g['segments']:
                    segment['start']=(np.array(segment['start'])@change).tolist()
                    segment['tangent']=(np.array(segment['tangent'])@change).tolist()
                    if np.linalg.det(change)<0:
                        segment['turn']=-segment['turn']
                        segment['radius']=-segment['radius']
                figure.update(changed)
                np.testing.assert_allclose(figure.body_points,expected,atol=1e-12)
                displayed=(real.preview_smooth_points(g)-figure.section_origin)@figure.section_basis
                np.testing.assert_allclose(figure.smooth_display['measured'],displayed,atol=1e-12)
            self.assertEqual(row,before)
        finally:
            figure.plt.close(figure.figure)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GravityPreviewTests))
    (REPORT/'software_checks.json').write_text(json.dumps(dict(passed=result.wasSuccessful(), tests=result.testsRun,
        hardware_connections=0, additional_dynamics=0),indent=2),encoding='utf8')
    sys.exit(0 if result.wasSuccessful() else 1)
