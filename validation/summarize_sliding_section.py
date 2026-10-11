"""Summarize all declared runs from saved evidence; does not run dynamics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from snake_control.simulation import ROOT

LABELS = [
    's1_head', 's1_head_repair', 's1_head_contract', 's1_head_phase', 's1_tail',
    's2_front', 's2_middle_payload', 's2_rear', 's3_taper', 's3_taper90',
    's3_elbow90', 's3_taper90_cycles', 's4_baseline', 's4_fixed', 's4_sliding',
    's4_sliding_cyclemean', 's2_middle_cyclemean',
]


def main():
    path = ROOT/'validation/reports/sliding_section'
    results = []
    lines = ['# 全部17次动力学运行：保存证据总表', '',
             '由`summarize_sliding_section.py`读取原报告和全采样审核生成。'
             '“完整”同时要求运动、协议、身体形态和管轴包绕；轴不可观测保留未验收。'
             '失败和修复均计数，不能用此表推算随机通过率。', '',
             '| 保存记录 | 时长 s | 身体转角范围 ° | 管轴包绕范围 °（可观测） | 管轴越界/不可观测采样 | 完成交接 | 未满足的项目 |',
             '| --- | ---: | --- | --- | --- | ---: | --- |']
    for label in LABELS:
        report = json.loads((path/f'{label}.json').read_text(encoding='utf-8'))
        audit = json.loads((path/f'{label}_audit.json').read_text(encoding='utf-8'))
        incomplete = [name for name, passed in {**report['checks'], **report['pipe_checks']}.items() if not passed]
        results.append(dict(label=label, scene=report['scene'], mode=report['mode'],
                            complete_verdict=report['passed'], failed_checks=incomplete,
                            duration_s=report['duration_s'], exit_time_s=report['exit_time_s'],
                            post_exit_phase_cycles=report.get('post_exit_phase_cycles'),
                            body_turn_range_deg=[report['minimum_body_turn_deg'], report['maximum_body_turn_deg']],
                            pipe_wrap_range_deg=[report['minimum_wrap_deg'], report['maximum_wrap_deg']],
                            body_failed_samples=audit['body_failed_samples'], pipe_failed_samples=audit['pipe_failed_samples'],
                            pipe_unobservable_samples=audit['pipe_unobservable_samples'],
                            inactive_force_max_nm=audit['inactive_force_max_nm'],
                            handovers=sum(event['kind'] == 'complete' for event in report['events'])))
        row = results[-1]
        verdict = '；'.join(incomplete) if incomplete else '完整通过'
        lines.append(f"| [{label}]({label}.json) | {row['duration_s']:.1f} | "
                     f"{row['body_turn_range_deg'][0]:.2f}–{row['body_turn_range_deg'][1]:.2f} | "
                     f"{row['pipe_wrap_range_deg'][0]:.2f}–{row['pipe_wrap_range_deg'][1]:.2f} | "
                     f"{row['pipe_failed_samples']} / {row['pipe_unobservable_samples']} | "
                     f"{row['handovers']} | {verdict} |")
    (path/'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (path/'结果总表.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'{len(results)} archived dynamics runs summarized; no new experiment.')


if __name__ == '__main__':
    main()
