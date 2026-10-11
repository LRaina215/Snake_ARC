"""导出实机兼容的局部使能/目标帧；本入口只做离线预览，不打开串口。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from snake_control.rolling_arc import ArcParameters
from snake_control.hardware_preview import preview_frame
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/step1.json')
    parser.add_argument('--profile', type=Path, default=ROOT/'configs/hardware_20j_reference.json')
    parser.add_argument('--phase', type=float, default=0.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    frame = preview_frame(ArcParameters.load(args.config), args.phase, args.profile)
    payload = json.dumps(frame, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
        print(f'Offline frame saved: {args.output}')
    else:
        print(payload)


if __name__ == '__main__':
    main()
