"""Render the entire observed preview from saved angles; no physics or serial."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gzip
import json
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
from codes import Rolling_Arc_Real as real

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    folder = Path(sys.argv[1])
    name = sys.argv[2] if len(sys.argv) > 2 else 'gravity_preview'
    if name not in ('gravity_preview', 'section_preview'):
        raise ValueError('Expected gravity_preview or section_preview output name')
    output = ROOT/'validation/results'/name
    output.mkdir(parents=True, exist_ok=True)
    report = ROOT/'validation/reports'/name
    report.mkdir(parents=True, exist_ok=True)
    figure, seen = real.JointPreviewFigure(), set()
    movie = output/'gravity_aligned_preview.mp4'
    writer = None
    try:
        with gzip.open(folder/'actual_trace.jsonl.gz', 'rt', encoding='utf8') as f:
            for r in map(json.loads, f):
                sample = r['motor_input']
                if not sample or sample['feedback_time_s'] in seen: continue
                seen.add(sample['feedback_time_s'])
                c = r['controller']; q = sample['angles_deg']
                row = dict(angles_deg=q, target_deg=[np.degrees(r['target_rad'][i]) if e else q[i] for i,e in enumerate(r['enabled'])],
                    enabled=r['enabled'], enable_known=True, feedback_wall_s=time.monotonic(), gravity_time_s=sample['feedback_time_s'],
                    window=c['window'], phase_rad=c['phase_rad'], mode='run', gait='仿真记录重放', automatic=True,
                    selection=c.get('selection') or {}, pending=c['pending'], observer_state=(c['observation'] or {}).get('state','waiting'),
                    reasons=[], block_reason=None, lengths=[.0645]*21)
                figure.update(row); figure.figure.canvas.draw()
                pixels = np.asarray(figure.figure.canvas.buffer_rgba())[:, :, :3]
                if writer is None:
                    writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*'mp4v'), 5., (pixels.shape[1], pixels.shape[0]))
                    if not writer.isOpened(): raise RuntimeError('Could not open preview movie writer')
                writer.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
                t = round(sample['feedback_time_s'], 6)
                if t in (8., 11.8, 14., 24.8):
                    figure.figure.savefig(report/f'preview_{t:g}s.png', dpi=110)
    finally:
        if writer is not None: writer.release()
        figure.plt.close(figure.figure)
    print(json.dumps(dict(movie=str(movie), frames=len(seen), fps=5, new_dynamics=0)))
