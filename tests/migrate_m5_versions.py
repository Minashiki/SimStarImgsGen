"""D77: prove default-path equality before migrating version provenance."""
import copy
import json

from starsim import algorithm_versions
from starsim.io import write_json
from starsim.runtime import runtime_environment
from tests import freeze_m3_baseline as baseline
from tests.migrate_m3_covered_modules import sequence_hashes
from tests.runtime_env import require_genstarimg


def main():
    require_genstarimg()
    manifest = json.loads(baseline.OUT.read_text())
    if any(g['algorithm_versions']['motion'] != 2 for g in manifest['groups'].values()):
        raise ValueError('motion migration requires version 2; refusing repeat')
    before = copy.deepcopy(manifest)
    hashes = sequence_hashes(manifest)
    current = baseline.algorithm_versions
    # Only the preflight expected version is adapted. Generation executes
    # the actual v3 motion with all new optional arguments left at None.
    baseline.algorithm_versions = lambda: dict(current(), motion=2)
    try:
        problems = {name: baseline.baseline_problems(name) for name in manifest['groups']}
    finally:
        baseline.algorithm_versions = current
    if any(problems.values()):
        raise ValueError(f'default path changed: {problems}')
    for group in manifest['groups'].values():
        group['algorithm_versions']['motion'] = 3
        group.setdefault('migrations', []).append('motion 2→3, default path bitwise identical')
    assert sequence_hashes(manifest) == hashes
    for name in manifest['groups']:
        assert manifest['groups'][name]['sequences'] == before['groups'][name]['sequences']
    baseline._atomic_write(manifest)
    report = dict(environment=runtime_environment(), default_path_problems=problems,
                  sequence_hashes_before=hashes, sequence_hashes_after=sequence_hashes(manifest),
                  migration='motion 2→3, default path bitwise identical',
                  rng_argument='No new RNG calls; None skips both new deterministic checks. Full frame, population, sensor hashes and effective configs match.')
    path = baseline.ROOT/'artifacts/m2_report.json'
    m2 = json.loads(path.read_text())
    for section in m2['sections'].values():
        section['algorithm_versions'] = algorithm_versions()
        section.setdefault('migrations', []).append('D77: motion 2→3 default path identical; split 1 unused by M2')
    write_json(path, m2)
    write_json(baseline.ROOT/'artifacts/m5_migration.json', report)


if __name__ == '__main__':
    main()
