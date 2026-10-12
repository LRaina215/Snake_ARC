"""Separate the requested repeated-motion result from unexcited return probes."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'validation/reports/repeated_motor_sliding'


def main():
    primary = json.loads((REPORT/'natural_review.json').read_text(encoding='utf8'))
    probes = []
    for path in sorted(REPORT.glob('return_*')):
        if not path.is_dir():
            continue
        review = json.loads((path/'review.json').read_text(encoding='utf8'))
        cause = json.loads((path/'cause_audit.json').read_text(encoding='utf8'))
        physical_reverse_samples = sum(stage['tail_crest_and_bending_samples'] for stage in cause['stages'].values())
        probes.append(dict(source=path.name, return_passed=review['actual_return_pass'],
            physical_reverse_samples=physical_reverse_samples,
            classification='not covered: no corroborated reverse arc migration' if physical_reverse_samples == 0 else
                           'reverse input present: inspect tracking failure',
            morphology_pass=review['morphology_pass'], invariants_pass=review['six_and_disabled_invariant_pass'],
            mechanics_pass=review['mechanics_pass']))
    allow = (primary['passed'] and len(primary['transitions']) >= 2 and
             all(p['physical_reverse_samples'] == 0 and p['morphology_pass'] and p['invariants_pass'] and
                 p['mechanics_pass'] for p in probes))
    result = dict(allow_sync=bool(allow), requested_objective='verify multiple consecutive automatic window transfers',
        verified_scope='headward J6-J11 -> J5-J10 -> J4-J9, 200ms motor feedback / 20ms control',
        primary_source=primary['source'], primary_passed=primary['passed'], extra_probes=probes,
        unverified=['tailward handover and return', 'short passive ends', 'real device timing and mechanics'],
        scope_decision='Do not turn an unexcited extra return case into failure of the passed primary objective; '
                       'sync this experimental method without claiming return or real-world validation')
    (REPORT/'sync_review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
