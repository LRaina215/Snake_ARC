"""Read archived feedback/sections only; no controller steps or physical experiments."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import gzip
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
from codes import Rolling_Arc_Real as real

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'validation/reports/full_body_section_preview_20261021'


def payload(q, target, window, stamp, hint=None, section=None):
    enabled = [window[0] <= i <= window[1] for i in range(1, 21)]
    return dict(angles_deg=q, target_deg=[t if e else a for t,e,a in zip(target,enabled,q)],
        enabled=enabled, smooth_section=section, gravity_hint=hint, gravity_time_s=stamp,
        enable_known=True, feedback_wall_s=time.monotonic(), window=window,
        phase_rad=0.,mode='已有记录回放',gait='全身截面检查',feeding=None,
        automatic=True,selection={},pending=None,observer_state='',reasons=[],
        block_reason=None,lengths=list(real.LINK_LENGTHS_M))


def stored_section(row):
    c = row['controller']['compliance']
    if c.get('projected_m') is None:
        return None
    window = c['window']
    points = real.forward(np.radians(row['motor_input']['angles_deg']), real.LINK_LENGTHS_M)
    origin = points[window[0]:window[1]+2].mean(axis=0)
    local = points[window[0]-1:window[1]+3]-origin
    projected = np.array(c['projected_m'])
    # Recover the archived basis, including its sign/roll, from saved coordinates.
    basis = np.linalg.lstsq(np.vstack((local,c['axis'])),
                            np.vstack((projected,[0.,0.])),rcond=None)[0]
    np.testing.assert_allclose(local@basis, projected, atol=1e-10)
    np.testing.assert_allclose(basis.T@basis,np.eye(2),atol=1e-8)
    curve = real.continuous_section(projected)
    np.testing.assert_allclose(real.sample_section(curve),c['section_curve_m'],atol=1e-10)
    geometry = dict(axis=dict(point=origin,basis=basis),curve=curve)
    return dict(source='control', window=window, updates=c['geometry_updates'],
                measured=real.preview_section_geometry(geometry),candidate=None)


def check(figure, row, stats):
    row['feedback_wall_s']=time.monotonic()
    before=copy.deepcopy(row)
    figure.update(row)
    assert row==before
    stats['frames']+=1
    if figure.body_points is None or not figure.smooth_display:
        stats['unformed_or_waiting']+=1
        return
    assert figure.body_points.shape==(22,2)
    labels={t.get_text() for t in figure.body.texts}
    assert {f'J{i}' for i in range(1,21)} <= labels
    down=figure.gravity.result.get('down_head')
    if down is not None:
        projected=np.array(down)@figure.section_basis
        assert projected[1]<0 and abs(projected[0])<1e-10
    state=figure.gravity.result['state']
    stats['gravity_states'][state]=stats['gravity_states'].get(state,0)+1
    if row['smooth_section'] is not None:
        g=row['smooth_section']['measured']
        control_basis=np.array(g['basis'])
        np.testing.assert_allclose(figure.section_basis@figure.section_basis.T,
                                   control_basis@control_basis.T,atol=1e-8)
        expected=(real.preview_smooth_points(g)-figure.section_origin)@figure.section_basis
        np.testing.assert_allclose(figure.smooth_display['measured'],expected,atol=1e-12)
    stats['full_body_frames']+=1


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    summary=dict(hardware_connections=0,new_dynamics=0,controller_steps=0,passed=False)
    source=ROOT/'validation/reports/gravity_preview/physics/display_pull_20261013_165050_222158/actual_trace.jsonl.gz'
    figure=real.JointPreviewFigure()
    figure.figure.canvas.draw_idle=lambda: None  # Render selected evidence, check every sample numerically.
    try:
        stats=dict(frames=0,full_body_frames=0,unformed_or_waiting=0,gravity_states={})
        summary['simulation_archive']=stats
        seen=set()
        cached_key=cached_section=None
        with gzip.open(source,'rt',encoding='utf8') as stream:
            for archived in map(json.loads,stream):
                f=archived['motor_input']
                if not f: continue
                c=archived['controller']['compliance']
                key=(tuple(c.get('window',[])),c.get('geometry_updates'))
                # Geometry runs on its own clock. Recover its frame exactly
                # when the cache changes, not from a later encoder sample.
                if c.get('projected_m') is not None and key!=cached_key:
                    cached_section=stored_section(archived)
                    cached_key=key
                if f['feedback_time_s'] in seen: continue
                seen.add(f['feedback_time_s'])
                o=archived['controller'].get('observation') or {}
                hint=(dict(down_head=o['down_head'],source=o['end_quality']['source'])
                      if o.get('valid') and o.get('down_head') is not None and
                      o.get('feedback_time_s')==f['feedback_time_s'] else None)
                row=payload(f['angles_deg'],np.degrees(archived['target_rad']).tolist(),
                            archived['controller']['window'],f['feedback_time_s'],hint,cached_section)
                row['phase_rad']=archived['controller']['phase_rad']
                row['mode']='已有记录·暂停' if archived['controller']['phase_hold'] else '已有记录·滚动'
                check(figure,row,stats)
                if round(f['feedback_time_s'],6) in (8.,11.8,14.,24.8):
                    figure.figure.savefig(REPORT/f"simulation_{f['feedback_time_s']:.1f}s.png",dpi=110)
        figure.plt.close(figure.figure)
        figure=real.JointPreviewFigure()
        figure.figure.canvas.draw_idle=lambda: None
        stats=dict(frames=0,full_body_frames=0,unformed_or_waiting=0,gravity_states={})
        summary['real_task19_archive']=stats
        path=ROOT/'validation/reports/real_feeding_distance_20261021/task_19_fresh_samples.json'
        for i,s in enumerate(json.loads(path.read_text(encoding='utf8'))):
            row=payload(s['angles_deg'],s['target_deg'],s['window'],s['time_s'],
                        dict(down_head=s['down_head'],source=s['proxy_source']) if s['down_head'] is not None else None)
            check(figure,row,stats)
            if i==0: figure.figure.savefig(REPORT/'real_pose.png',dpi=110)
        summary['passed']=True
    finally:
        figure.plt.close(figure.figure)
        (REPORT/'display_replay.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
