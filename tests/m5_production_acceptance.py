"""Fresh three-split determinism, production identity and source acceptance."""
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from starsim import algorithm_versions
from starsim.config import load_config
from starsim.generate import generate_sequence, verify_completed, _signature
from starsim.io import _sha256, write_json
from starsim.metadata import git_provenance
from starsim.runtime import runtime_environment
from tests.runtime_env import require_genstarimg
from scripts.m5_test_evidence import validate

ROOT = Path(__file__).resolve().parents[1]


def hashes(directory):
    return {p.name: _sha256(p) for p in sorted(Path(directory).iterdir())
            if p.name.endswith('.fits.fz') or p.name.startswith(('objects_', 'labels_')) and p.suffix == '.npz'}


def git(*args):
    return subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', *args],
                          cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def production_config(manifest):
    source = manifest['source']
    return load_config(overrides=dict(camera=dict(sensor=source['camera']['sensor']), generator=source['generator']))


def verified_records(production, manifest, config, regions, signature):
    n = config.generator['dataset']['n_sequences']
    if set(manifest['sequences']) != {str(i) for i in range(n)}:
        raise ValueError('sequence index set differs')
    records = {}
    for i in range(n):
        record = verify_completed(production, i, config.generator['dataset']['seed'], regions, signature)
        old = dict(manifest['sequences'][str(i)])
        timing = old.pop('commit_timing', None)
        if old != record:
            raise ValueError(f'manifest sequence {i} differs from verified record')
        if timing is not None:
            record['commit_timing'] = timing
        records[str(i)] = record
    for key, field in (('total_sequence_seconds', 'seconds'), ('sequence_bytes', 'bytes')):
        if manifest[key] != sum(r[field] for r in records.values()):
            raise ValueError(f'manifest {key} differs')
    return records


def commit_summary(records):
    measured = [r['commit_timing'] for r in records.values() if 'commit_timing' in r]
    return dict(measured_sequences=len(measured), missing_sequences=len(records)-len(measured),
                measured_seconds={key: sum(t[key] for t in measured) for key in
                    ('hash_seconds', 'publish_seconds', 'other_seconds', 'total_seconds')})


def main():
    require_genstarimg()
    if '--child' in sys.argv:
        production, index, out = sys.argv[-3:]
        manifest = json.loads((Path(production)/'dataset.json').read_text(encoding='utf-8'))
        config = production_config(manifest)
        generate_sequence(config, config.generator['dataset']['seed'], int(index), out)
        return
    provenance = git_provenance()
    if provenance['git_commit'] is None or provenance['git_dirty'] is not False:
        raise ValueError('acceptance requires clean starsim scripts configs tests and a git commit')
    acceptance_commit = provenance['git_commit']
    quick = json.loads((ROOT/'artifacts/m5_quick_tests.json').read_text(encoding='utf-8'))
    validate(quick)
    production = Path(sys.argv[1]).resolve()
    manifest = json.loads((production/'dataset.json').read_text(encoding='utf-8'))
    if manifest['schema_version'] != 2 or not manifest['complete'] or len(manifest['sequences']) != 10:
        raise ValueError('require complete schema 2 ten-sequence production dataset')
    config = production_config(manifest)
    signature = _signature(config)
    if signature != manifest['source']:
        raise ValueError('current source signature differs')
    from starsim.split import make_splits, audit
    regions = make_splits(config.generator['dataset']['split'], config.generator['dataset']['seed'], 10)
    if regions != manifest['regions']:
        raise ValueError('production regions differ')
    sensor = config.camera['sensor']
    if sensor['resolution'] != dict(width=4096, height=4096) or sensor['focal_length_mm'] != 300:
        raise ValueError('production camera required')
    sources = set()
    verified = verified_records(production, manifest, config, regions, signature)
    records = list(verified.values())
    for index in range(10):
        sequence = json.loads((production/f'seq_{index:06d}'/'sequence.json').read_text(encoding='utf-8'))
        source = sequence['git_commit']
        if source is None or sequence['git_dirty'] is not False:
            raise ValueError('generation source must be committed and clean')
        git('merge-base', '--is-ancestor', source, acceptance_commit)
        git('diff', '--quiet', source, acceptance_commit, '--', 'starsim', 'configs')
        sources.add(source)
    distributions = audit(regions, records, config.generator['dataset']['split']['min_seqs_for_assert'])
    concurrency = manifest.get('concurrent_memory', {})
    if concurrency.get('max_active_workers', 0) < 2:
        raise ValueError('production must measure at least two simultaneously active workers')
    if any(r['peak_rss_bytes'] >= 4*2**30 for r in records):
        raise ValueError('worker peak exceeds 4 GiB empirical budget')
    root = ROOT/'.tmp'/f'm5_acceptance_{uuid.uuid4().hex}'
    root.mkdir(parents=True, exist_ok=False)
    comparisons = {}
    for split in ('train', 'val', 'test'):
        index = regions['assignments'].index(split)
        results = []
        for seed in ('1', '2'):
            out = root/f'{split}_hashseed_{seed}'
            if out.exists():
                raise ValueError('reproduction directory already exists')
            subprocess.run([sys.executable, '-m', 'tests.m5_production_acceptance', '--child',
                            str(production), str(index), str(out)],
                           env=dict(os.environ, PYTHONHASHSEED=seed), cwd=ROOT, check=True)
            results.append(hashes(out))
        if not results[0] or results[0] != results[1] or results[0] != hashes(production/f'seq_{index:06d}'):
            raise ValueError(f'{split} reproduction differs')
        comparisons[split] = dict(index=index, hashes=results[0], matches_batch=True)
    timings = [json.loads((production/f'seq_{i:06d}'/'timing.json').read_text(encoding='utf-8')) for i in range(10)]
    generation = {}
    for name in timings[0]['seconds']:
        generation[name] = sum(sum(t['seconds'][name]) if isinstance(t['seconds'][name], list)
                               else t['seconds'][name] for t in timings)
    if git_provenance() != provenance:
        raise ValueError('acceptance source changed during execution')
    report = dict(environment=runtime_environment(), algorithm_versions=algorithm_versions(),
        acceptance_source=provenance, generation_commits=sorted(sources), source=signature, regions=regions,
        sequences=verified, star_counts=distributions, runs=manifest['runs'],
        active_seconds=manifest['active_seconds'], active_seconds_is_lower_bound=manifest['active_seconds_is_lower_bound'],
        elapsed_seconds=manifest['elapsed_seconds'], total_sequence_seconds=sum(r['seconds'] for r in records),
        legacy_run_timing='unrecoverable for runs without segment records; no inferred durations',
        generation_timing=dict(segments=generation, total_seconds=sum(t['total_seconds'] for t in timings)),
        dense_storage={key: sum(v[key] for t in timings for v in t['dense_storage'])
                       for key in ('compression_seconds', 'write_seconds')},
        commit_timing=commit_summary(verified),
        task_timing=dict(initialization_seconds=sum(r['initialization_seconds'] for r in records),
                         construction_seconds=sum(r['construction_seconds'] for r in records)),
        timing_ranges=dict(initialization='_worker entry to _write entry; repeated per sequence',
            construction='_write entry to write_sequence call; includes sampling and prepare_sequence',
            spawn_import='unmeasured: task deserialization imports precede _worker entry',
            numba_cache='first-call loading belongs to construction or write_sequence; not pure compute',
            dense_storage='compression includes archive serialization; writes include flush, use OS cache, no fsync'),
        concurrent_memory=concurrency, sequence_bytes=sum(r['bytes'] for r in records),
        disk_bytes_including_manifests=sum(p.stat().st_size for p in production.rglob('*') if p.is_file()),
        determinism=dict(interpreters_per_split=2, pythonhashseed=[1, 2], splits=comparisons, fresh_directory=str(root)))
    write_json(ROOT/'artifacts/m5_report.json', report)
    write_json(ROOT/'artifacts/m5_verification.json', dict(acceptance_source=provenance,
        generation_commits=sorted(sources), production_identity_and_hashes_verified=10,
        determinism_splits=comparisons, concurrent_memory=concurrency,
        quick_tests=quick))
    print(json.dumps(dict(production_sequences=10, reproduced_splits=list(comparisons))))


if __name__ == '__main__':
    main()
