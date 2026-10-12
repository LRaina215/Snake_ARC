"""Reproduce the single bounded endpoint-pull case in the mujoco environment.

No serial, GUI, or parameter search. Every attempt has a new output directory;
actual prefixes are saved even if the controller or simulator raises an error.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from datetime import datetime
from contextlib import redirect_stdout
import gzip
import io
import json
import numpy as np
from codes import Rolling_Arc_Real as real
from codes.Rolling_Arc_Debug_Mujoco import DebugSession
from snake_control.sliding_section_compliance import open_geometry

OUTPUT = Path(__file__).resolve().parents[2] / 'results' / ('debug_modes_reproduction_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    session = DebugSession()
    console, error, records = io.StringIO(), None, []
    try:
        with redirect_stdout(console):
            session.commands.put('apply')
            for _ in range(110):
                session.advance()
                row = session.snapshot()
                row['stage'] = 'formation'
                records.append(row)
            session.commands.put('gait_sliding')
            session.commands.put('run')
            for frame in range(1700):
                if frame == 600:
                    session.commands.put('push_right')
                if frame == 1300:
                    session.commands.put('push_right_stop')
                session.advance()
                row = session.snapshot()
                row['stage'] = 'no_force' if frame < 600 else 'right_force' if frame < 1300 else 'recovery'
                q = np.asarray(row['actual_qpos'])[session.sim.drive.qadr]
                row['actual_body_geometry'] = open_geometry(q, real.LINK_LENGTHS_M, row['window'])
                records.append(row)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        with gzip.open(OUTPUT / 'actual_trace.jsonl.gz', 'wt', encoding='utf8') as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        status = dict(status='complete' if error is None else 'interrupted', error=error,
                      samples=len(records), final_simulation_time_s=session.sim.data.time,
                      scene=session.sim.scene, parameters=vars(session.sim.p),
                      window_options=(None if session.controller.sliding is None else
                                      vars(session.controller.sliding.options)),
                      events=session.snapshot()['sliding_events'])
        (OUTPUT / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf8')
        (OUTPUT / 'console.txt').write_text(console.getvalue(), encoding='utf8')
        session.close()
        print(f'Actual-state archive: {OUTPUT}')


if __name__ == '__main__':
    main()
