"""Open/close the preview from a copied standalone file; never open real serial."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = '''
import json
from pathlib import Path
import Rolling_Arc_Real as r
if __name__ == '__main__':
    serial = r.DryRunSerial()
    controller = r.RollingArcController(r.FeetechBus(serial))
    controller.preview_feedback = True
    r.poll_control_feedback(controller, 0., 0.)
    window = r.JointPreviewWindow()
    try:
        window.publish(controller)
        assert window.ready.wait(15), 'Preview did not render its first frame'
        assert window.active
        assert not controller.touched_hardware
        assert not any(serial.registers[s][0x28] for s in r.SERVO_IDS)
    finally:
        window.close()
    assert not window.active
    Path('result.json').write_text(json.dumps(dict(standalone_preview_rendered=True,
        child_closed=True, hardware_connections=0, motor_enable_writes=0)), encoding='utf8')
'''

if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='rolling_arc_preview_') as folder:
        path = Path(folder)
        shutil.copy2(ROOT/'codes/Rolling_Arc_Real.py', path/'Rolling_Arc_Real.py')
        (path/'check.py').write_text(SCRIPT, encoding='utf8')
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        environment.pop('MPLBACKEND', None)
        subprocess.run([sys.executable, '-X', 'utf8', str(path/'check.py')], cwd=path,
                       env=environment, check=True, timeout=25)
        result = json.loads((path/'result.json').read_text(encoding='utf8'))
    output = ROOT/'validation/reports/posture_port/preview_process.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))
