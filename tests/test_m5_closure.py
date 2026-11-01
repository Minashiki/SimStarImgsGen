"""D82–D86 negative cases and cancellation behavior."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from tests.test_m5_generation import completed_dataset, production_record, small_config
from starsim.generate import _signature, generate_dataset, replay_sequence
from starsim.split import make_splits
from tests.m5_production_acceptance import verified_records, commit_summary


@pytest.mark.parametrize('damage', ['missing', 'extra', 'index', 'record', 'seconds', 'bytes', 'disk_bytes'])
def test_acceptance_binds_manifest(completed_dataset, tmp_path, damage):
    import shutil
    out = tmp_path/'dataset'
    shutil.copytree(completed_dataset, out)
    manifest = json.loads((out/'dataset.json').read_text())
    if damage == 'missing':
        manifest['sequences'].pop('0')
    elif damage == 'extra':
        manifest['sequences']['3'] = manifest['sequences']['0']
    elif damage == 'index':
        manifest['sequences']['03'] = manifest['sequences'].pop('0')
    elif damage == 'record':
        manifest['sequences']['0']['seconds'] += 1
    elif damage in ('seconds', 'bytes'):
        manifest['total_sequence_seconds' if damage == 'seconds' else 'sequence_bytes'] += 1
    else:
        path = out/'seq_000000/complete.json'
        record = json.loads(path.read_text())
        record['bytes'] += 1
        path.write_text(json.dumps(record))
    config = small_config(out)
    regions = make_splits(config.generator['dataset']['split'], config.generator['dataset']['seed'], 3)
    with pytest.raises(ValueError):
        verified_records(out, manifest, config, regions, _signature(config))


def test_missing_commit_timing_and_four_phases(completed_dataset):
    manifest = json.loads((completed_dataset/'dataset.json').read_text())
    config = small_config(completed_dataset)
    manifest['sequences']['0'].pop('commit_timing')
    records = verified_records(completed_dataset, manifest, config, manifest['regions'], _signature(config))
    summary = commit_summary(records)
    assert summary['measured_sequences'] == 2 and summary['missing_sequences'] == 1
    assert summary['measured_seconds']['total_seconds'] == sum(records[str(i)]['commit_timing']['total_seconds'] for i in (1, 2))
    assert all(r['initialization_seconds'] > 0 and r['construction_seconds'] > 0 for r in records.values())
    assert all('started_monotonic_seconds' not in run for run in manifest['runs'])
    for i in range(3):
        timing = json.loads((completed_dataset/f'seq_{i:06d}/timing.json').read_text())
        assert timing['total_seconds'] > 0
        assert all(t['compression_seconds'] >= 0 and t['write_seconds'] >= 0 for t in timing['dense_storage'])


def test_replay_seed_before_stream(production_record, tmp_path, monkeypatch):
    record = deepcopy(production_record)
    record['config_unsampled']['dataset']['seed'] += 1
    path = tmp_path/'sequence.json'
    path.write_text(json.dumps(record))
    monkeypatch.setattr('starsim.rng.SequenceRng.get', lambda *a: pytest.fail('random stream issued'))
    with pytest.raises(ValueError, match='seed'):
        replay_sequence(path, tmp_path/'absent')
    assert not (tmp_path/'absent').exists()


def test_executor_failure_marks_run(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('executor failure')
    monkeypatch.setattr('starsim.generate._execute_tasks', fail)
    out = tmp_path/'failed'
    with pytest.raises(RuntimeError, match='executor failure'):
        generate_dataset(small_config(out))
    assert json.loads((out/'dataset.json').read_text())['runs'][-1]['status'] == 'failed'


class SentinelTimeout(Exception):
    pass


def cancellation_task(index, directory):
    directory = Path(directory)
    (directory/f'started_{index}').touch()
    if index == 0:
        deadline = time.monotonic()+20
        while not (directory/'submitted').exists():
            if time.monotonic() > deadline:
                raise SentinelTimeout('submission sentinel timeout')
            time.sleep(.02)
        raise RuntimeError('expected worker failure')
    time.sleep(.5)


def cancellation_child(directory):
    from starsim.generate import _execute_tasks
    try:
        list(_execute_tasks(cancellation_task, [(i, directory) for i in range(10)], workers=1,
                            after_submit=lambda: (Path(directory)/'submitted').touch()))
    except SentinelTimeout:
        raise
    except RuntimeError as exc:
        assert str(exc) == 'expected worker failure'
    else:
        raise AssertionError('worker failure not propagated')
    assert len(list(Path(directory).glob('started_*'))) < 10


def test_cancels_queued_tasks_with_outer_timeout(tmp_path):
    script = 'from tests.test_m5_closure import cancellation_child; import sys; cancellation_child(sys.argv[1])'
    with (tmp_path/'child.log').open('w') as log:
        child = subprocess.Popen([sys.executable, '-c', script, str(tmp_path)], stdout=log, stderr=log)
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            processes = psutil.Process(child.pid).children(recursive=True)
            for process in [*reversed(processes), psutil.Process(child.pid)]:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            child.wait(timeout=10)
            pytest.fail('executor process tree exceeded timeout')
    assert child.returncode == 0, (tmp_path/'child.log').read_text()
    assert 'Exception in thread' not in (tmp_path/'child.log').read_text()


@pytest.mark.parametrize('setup_failure', [False, True])
def test_node_properties_survive_setup_and_parameters(tmp_path, setup_failure):
    from scripts.m5_test_evidence import ROOT, environment
    path = tmp_path/'test_probe.py'
    path.write_text('import pytest\n@pytest.fixture\ndef setup():\n    '+
                    ('raise RuntimeError("setup")' if setup_failure else 'return 1')+
                    '\n@pytest.mark.parametrize("value", [1, 2])\ndef test_probe(setup, value):\n    assert value > 0\n')
    junit = tmp_path/'probe.xml'
    result = subprocess.run([sys.executable, '-m', 'pytest', '-c', str(ROOT/'pytest.ini'),
                             '--rootdir', str(ROOT), '--confcutdir', str(tmp_path),
                             '-p', 'tests.evidence_plugin', str(path), f'--junitxml={junit}'],
                            cwd=ROOT, env=environment(), capture_output=True, text=True)
    assert result.returncode == (1 if setup_failure else 0), result.stdout+result.stderr
    import xml.etree.ElementTree as ET
    cases = ET.parse(junit).findall('.//testcase')
    assert len(cases) == 2
    ids = [c.findall('./properties/property[@name="node_id"]') for c in cases]
    assert all(len(values) == 1 for values in ids)
    assert len({values[0].get('value') for values in ids}) == 2


@pytest.mark.parametrize('damage', ['skip', 'zero', 'missing', 'duplicate_property', 'duplicate_id', 'counts'])
def test_reject_invalid_junit(tmp_path, damage):
    from scripts.m5_test_evidence import parse_junit
    properties = '<properties><property name="node_id" value="tests/test_a.py::test_a"/></properties>'
    if damage == 'missing':
        properties = ''
    if damage == 'duplicate_property':
        properties = properties.replace('</properties>', '<property name="node_id" value="x"/></properties>')
    case = '<testcase>'+properties+('<skipped/>' if damage == 'skip' else '')+'</testcase>'
    cases = '' if damage == 'zero' else case* (2 if damage == 'duplicate_id' else 1)
    path = tmp_path/'junit.xml'
    path.write_text(f'<testsuites><testsuite tests="{99 if damage == "counts" else 1}" failures="0" errors="0" skipped="0">{cases}</testsuite></testsuites>')
    with pytest.raises(ValueError):
        parse_junit(path)


@pytest.mark.parametrize('failure', ['pytest', 'partial', 'filter', 'changed'])
def test_failed_evidence_removes_old_summary(tmp_path, monkeypatch, failure):
    from scripts import m5_test_evidence as evidence
    summary = tmp_path/'summary.json'
    summary.write_text('old usable evidence')
    monkeypatch.setattr(evidence, 'SUMMARY', summary)
    monkeypatch.setattr(evidence, 'require_committed_sources', lambda: None)
    collection = dict(node_ids=['a', 'b'])
    monkeypatch.setattr(evidence, 'collect', lambda: collection)
    monkeypatch.setattr(evidence, 'invoke', lambda *a: (1 if failure == 'pytest' else 0,
                        dict(node_ids=['a']) if failure == 'filter' else collection))
    monkeypatch.setattr(evidence, 'parse_junit', lambda *a: dict(node_ids=['a'] if failure == 'partial' else ['a', 'b']))
    digests = iter(['before', 'after' if failure == 'changed' else 'before'])
    monkeypatch.setattr(evidence, 'code_digest', lambda: next(digests))
    if failure == 'pytest':
        assert evidence.main() == 1
    else:
        with pytest.raises(ValueError):
            evidence.main()
    assert not summary.exists()


@pytest.mark.parametrize('damage', ['hash', 'collection', 'source'])
def test_acceptance_rechecks_evidence(tmp_path, monkeypatch, damage):
    from scripts import m5_test_evidence as evidence
    junit = tmp_path/'junit.xml'
    junit.write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
                     '<testcase><properties><property name="node_id" value="a"/></properties>'
                     '</testcase></testsuite></testsuites>')
    collection = dict(node_ids=['a'])
    quick = dict(source_sha256='source', junit_sha256=evidence.sha256(junit),
                 collection=collection, junit=evidence.parse_junit(junit))
    monkeypatch.setattr(evidence, 'JUNIT', junit)
    monkeypatch.setattr(evidence, 'code_digest', lambda: 'changed' if damage == 'source' else 'source')
    monkeypatch.setattr(evidence, 'collect', lambda: dict(node_ids=['b']) if damage == 'collection' else collection)
    if damage == 'hash':
        junit.write_text(junit.read_text()+' ')
    with pytest.raises(ValueError):
        evidence.validate(quick)


def test_uncommitted_root_config_is_dirty(tmp_path, monkeypatch):
    import starsim
    from starsim.metadata import git_provenance
    (tmp_path/'starsim').mkdir()
    monkeypatch.setattr(starsim, '__file__', str(tmp_path/'starsim/__init__.py'))
    def git(*args):
        subprocess.run(['git', '-c', f'safe.directory={tmp_path.as_posix()}', *args],
                       cwd=tmp_path, check=True, capture_output=True)
    git('init')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '--allow-empty', '-m', 'initial')
    assert git_provenance()['git_dirty'] is False
    (tmp_path/'pytest.ini').write_text('[pytest]\njunit_family=xunit2\n')
    assert git_provenance()['git_dirty'] is True
