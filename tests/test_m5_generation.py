"""Production provenance, seeded caps, isolated replay and resume contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from starsim.config import load_config
from starsim.generate import generate_sequence, generate_dataset, replay_sequence
from starsim.io import _sha256
from starsim.split import make_splits, accepts, vector, separation, audit
from tests.test_m4_labels import disk_sequence


def small_config(out, seed=20261027):
    return load_config(overrides=dict(camera=dict(sensor=dict(
        resolution=dict(width=128, height=128), focal_length_mm=9.375)),
        generator=dict(dataset=dict(n_sequences=3, n_frames=2, seed=seed, workers=2, out_dir=str(out)),
            targets=dict(min_visible_frames=2), pointing=dict(min_labeled_stars=1),
            camera_motion=dict(speed_px_per_frame=[1, 2], accel_px_per_frame2=[0, 0]))))


def hashes(path):
    return {p.name: _sha256(p) for p in Path(path).iterdir()
            if p.suffix == '.npz' or p.name.endswith('.fits.fz')}


@pytest.mark.parametrize('damage', ['sensor', 'derived', 'copy'])
def test_camera_rejects_before_stream_or_output(damage, monkeypatch, tmp_path):
    config = small_config(tmp_path/'out')
    if damage == 'sensor':
        config.camera['sensor']['focal_length_mm'] *= 2
    elif damage == 'derived':
        config.derived.K[0, 0] += 1
    else:
        config.camera['derived']['K'][0][0] += 1
    def forbidden(*args, **kwargs):
        pytest.fail('stream issued before camera check')
    monkeypatch.setattr('starsim.rng.SequenceRng.get', forbidden)
    with pytest.raises(ValueError, match='camera'):
        generate_sequence(config, config.generator['dataset']['seed'], 0, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_caps_reproduce_stratify_and_do_not_overlap():
    cfg = load_config().generator['dataset']['split']
    a = make_splits(cfg, 42, 100)
    assert a == make_splits(cfg, 42, 100)
    assert a != make_splits(cfg, 43, 100)
    assert {name: a['assignments'].count(name) for name in ('train', 'val', 'test')} == dict(train=80, val=10, test=10)
    for i, c in enumerate(a['caps']):
        center = vector(c['ra_deg'], c['dec_deg'])
        assert accepts(a, c['split'], center)
        assert not accepts(a, 'train', center)
        for other in a['caps'][i+1:]:
            assert separation(center, vector(other['ra_deg'], other['dec_deg'])) > c['radius_deg']+other['radius_deg']
    cfg['mode'] = 'random'
    random = make_splits(cfg, 42, 10)
    assert random['caps'] == [] and random['guard_loss_fraction'] == 0
    assert random['area_fraction'] == dict(train=1., val=1., test=1.)


def test_star_count_hard_assert_and_small_warning():
    regions = dict(mode='random')
    records = [dict(split=s, star_counts=[n]) for s, n in [('train', 100), ('val', 101), ('test', 200)]]
    with pytest.raises(AssertionError, match='30%'):
        audit(regions, records, 1)
    with pytest.warns(UserWarning, match='too few'):
        audit(regions, records, 2)


def test_frozen_m4_camera_record_is_not_replayable(disk_sequence, tmp_path):
    with pytest.raises(ValueError, match='camera'):
        replay_sequence(disk_sequence.directory/'sequence.json', tmp_path/'replay')
    assert not (tmp_path/'replay').exists()


def test_one_cross_frame_status_call_and_cached_canvas(tmp_path, monkeypatch):
    import starsim.labels as labels
    import starsim.render_labels as renderer
    original, geometry = labels.assign_status, renderer.canvas_geometry
    calls, canvases = [], []
    def spy(objects, cfg):
        calls.append(len(objects))
        return original(objects, cfg)
    def canvas(objects, cfg):
        canvases.append(len(objects))
        return geometry(objects, cfg)
    monkeypatch.setattr(labels, 'assign_status', spy)
    monkeypatch.setattr(renderer, 'canvas_geometry', canvas)
    config = small_config(tmp_path/'out')
    config.generator['labels']['label_frame'] = 'reference'
    generate_sequence(config, config.generator['dataset']['seed'], 0, tmp_path/'out')
    assert calls == [2]
    assert canvases == [2]


def test_independent_hash_seeds_replay_and_resume(tmp_path):
    outputs = [tmp_path/'one', tmp_path/'two']
    script = ('from tests.test_m5_generation import small_config; '
              'from starsim.generate import generate_dataset; '
              'import sys; generate_dataset(small_config(sys.argv[1]))')
    for seed, out in zip(('1', '2'), outputs):
        subprocess.run([sys.executable, '-c', script, str(out)],
                       env=dict(os.environ, PYTHONHASHSEED=seed), check=True, capture_output=True, text=True)
    for index in range(3):
        assert hashes(outputs[0]/f'seq_{index:06d}') == hashes(outputs[1]/f'seq_{index:06d}')
    replay_sequence(outputs[0]/'seq_000000/sequence.json', tmp_path/'replayed')
    assert hashes(outputs[0]/'seq_000000') == hashes(tmp_path/'replayed')
    # Simulate interruption after sequence 0, before sequence 1 commits.
    import shutil
    out = outputs[1]
    shutil.rmtree(out/'seq_000001')
    (out/'.seq_000001.partial').mkdir()
    (out/'.seq_000001.partial'/'incomplete').write_text('partial')
    manifest = json.loads((out/'dataset.json').read_text())
    manifest['sequences'].pop('1')
    manifest['complete'] = False
    (out/'dataset.json').write_text(json.dumps(manifest))
    config = small_config(out)
    generate_dataset(config, resume=True)
    assert hashes(outputs[0]/'seq_000001') == hashes(out/'seq_000001')
    config.generator['dataset']['seed'] += 1
    with pytest.raises(ValueError, match='resume'):
        generate_dataset(config, resume=True)
    file = out/'seq_000000/objects_000.npz'
    file.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='corrupt'):
        generate_dataset(small_config(out), resume=True)


def test_grid_boundary_and_equal_contribution_matches_exhaustive():
    from starsim.labels import pixel_groups
    from tests.test_m4_labels import scene
    frame, result, sensor, psf, *_ = scene([31., 32., 38.], side=64, flux=[1000, 1000, 20])
    assert len(frame.sources) == 3
    # Construct exact inclusive endpoint overlap across a grid cell boundary.
    sources = [dict(class_id=0, has_flux=True, raw_group_id=i, peak_row=32,
                    peak_col=32, s_self=1., box=[32, 32, 32, 32], flux_e=1., x=32., y=32.)
               for i in range(2)]
    stamps = [np.ones((1, 1)), np.ones((1, 1))]
    light = np.zeros((64, 64)); light[32, 32] = 2
    groups, dominated = pixel_groups(sources, stamps, light, psf)
    assert groups == [(0, 1)] and dominated == {}


@pytest.mark.parametrize('failure', ['sky_split', 'min_labeled_stars'])
def test_motion_new_checks_share_rejection_loop(failure, monkeypatch, tmp_path):
    import starsim.motion as motion
    from starsim.config import sample_sequence_params
    from starsim.rng import SequenceRng, root_seed, sequence_seed
    config = small_config(tmp_path)
    rngs = SequenceRng(sequence_seed(root_seed(18), 0))
    effective = sample_sequence_params(config, rngs)
    effective['camera_motion']['jitter_px'] = 0.
    monkeypatch.setattr(motion, '_MAX_ATTEMPTS', 3)
    monkeypatch.setattr('starsim.split.accepts', lambda *a: failure != 'sky_split')
    monkeypatch.setattr(motion, '_minimum_stars', lambda *a: False)
    with pytest.raises(RuntimeError, match=f'{failure}=3'):
        motion.simulate_sequence(effective, rngs, config.derived,
                                 sky_regions={}, split='train', catalog=object())
    assert rngs.issued.count('motion') == 1


def test_motion_version_migration_preserves_records_and_refuses_failure(tmp_path, monkeypatch):
    import copy
    from tests import migrate_m5_versions as migration
    baseline = migration.baseline
    manifest = json.loads(baseline.OUT.read_text())
    for group in manifest['groups'].values():
        group['algorithm_versions']['motion'] = 2
    before = copy.deepcopy(manifest)
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(baseline, 'OUT', path)
    monkeypatch.setattr(baseline, 'ROOT', tmp_path)
    (tmp_path/'artifacts').mkdir()
    (tmp_path/'artifacts/m2_report.json').write_text('{"sections": {}}')
    monkeypatch.setattr(baseline, 'baseline_problems', lambda name: ['changed'])
    contents = path.read_bytes()
    with pytest.raises(ValueError, match='default path changed'):
        migration.main()
    assert path.read_bytes() == contents
    monkeypatch.setattr(baseline, 'baseline_problems', lambda name: [])
    migration.main()
    after = json.loads(path.read_text())
    assert migration.sequence_hashes(after) == migration.sequence_hashes(before)
    assert all(g['algorithm_versions']['motion'] == 3 for g in after['groups'].values())
    contents = path.read_bytes()
    with pytest.raises(ValueError, match='refusing repeat'):
        migration.main()
    assert path.read_bytes() == contents


def test_real_interruption_then_resume(tmp_path):
    import time
    import psutil
    out = tmp_path/'interrupted'
    script = ('from tests.test_m5_generation import small_config; '
              'from starsim.generate import generate_dataset; '
              'import sys; generate_dataset(small_config(sys.argv[1]))')
    with (tmp_path/'child.log').open('w') as log:
        child = subprocess.Popen([sys.executable, '-c', script, str(out)], stdout=log, stderr=log)
        try:
            deadline = time.monotonic()+90
            while time.monotonic() < deadline:
                if list(out.glob('.seq_*.partial/frame_*.fits.fz')):
                    break
                if child.poll() is not None:
                    pytest.fail('generation exited before interruption point')
                time.sleep(.05)
            else:
                pytest.fail('no partial frame reached')
        finally:
            descendants = psutil.Process(child.pid).children(recursive=True) if child.poll() is None else []
            if child.poll() is None:
                child.terminate()
            for process in descendants:
                try:
                    process.terminate()
                except psutil.NoSuchProcess:
                    pass
            child.wait(timeout=30)
            psutil.wait_procs(descendants, timeout=30)
    assert not json.loads((out/'dataset.json').read_text())['complete']
    resumed = generate_dataset(small_config(out), resume=True)
    assert resumed['complete'] and len(resumed['sequences']) == 3
    assert resumed['runs'][0]['status'] == 'interrupted'
    assert resumed['active_seconds_is_lower_bound']
    replay_sequence(out/'seq_000000/sequence.json', tmp_path/'expected')
    assert hashes(out/'seq_000000') == hashes(tmp_path/'expected')


@pytest.fixture(scope='module')
def production_record(tmp_path_factory):
    out = tmp_path_factory.mktemp('production_record')/'sequence'
    generate_sequence(small_config(out, seed=23), 23, 0, out)
    return json.loads((out/'sequence.json').read_text())


@pytest.mark.parametrize('damage', ['sensor', 'derived', 'copy', 'split', 'minimum'])
def test_replay_rejects_changed_camera_or_conditions(production_record, damage, tmp_path, monkeypatch):
    import copy
    record = copy.deepcopy(production_record)
    if damage == 'sensor':
        record['camera_config']['sensor']['focal_length_mm'] *= 2
    elif damage == 'derived':
        record['derived_camera']['K'][0][0] += 1
    elif damage == 'copy':
        record['camera_config']['derived']['K'][0][0] += 1
    elif damage == 'split':
        record['config_effective']['pointing']['split_constraints']['guard_deg'] += 1
    else:
        record['config_effective']['pointing']['min_labeled_stars'] += 1
    if damage in ('sensor', 'derived', 'copy'):
        def forbidden(*a, **kw):
            pytest.fail('camera mismatch issued a random stream')
        monkeypatch.setattr('starsim.rng.SequenceRng.get', forbidden)
    path = tmp_path/'sequence.json'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='camera|constraint'):
        replay_sequence(path, tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('count', [10, 11, 100, 1000, 5000])
def test_split_caps_and_prefix_stable(count):
    from copy import deepcopy
    cfg = load_config().generator['dataset']['split']
    reference = make_splits(cfg, 42, 5000)
    result = make_splits(cfg, 42, count)
    assert result['caps'] == reference['caps']
    assert result['assignments'] == reference['assignments'][:count]
    changed = deepcopy(cfg)
    changed['fractions'] = dict(train=.7, val=.2, test=.1)
    assert make_splits(changed, 42, count)['caps'] == reference['caps']


@pytest.mark.parametrize('fractions', [(.8, .1, .1), (.7, .2, .1), (1/3, 1/3, 1/3)])
def test_all_prefix_counts_within_one(fractions):
    from starsim.split import assign
    names = ('train', 'val', 'test')
    cfg = dict(fractions=dict(zip(names, fractions)))
    assignments = assign(cfg, 42, 10000)
    counts = np.zeros(3, dtype=int)
    for i, name in enumerate(assignments, 1):
        counts[names.index(name)] += 1
        assert np.all(np.abs(counts-np.array(fractions)*i) < 1)


@pytest.mark.parametrize('schema', [1, None])
def test_legacy_schema_rejected_without_writes(tmp_path, schema):
    out = tmp_path/'legacy'
    out.mkdir()
    manifest = dict(source={}, sequences={})
    if schema is not None:
        manifest['schema_version'] = schema
    (out/'dataset.json').write_text(json.dumps(manifest))
    (out/'seq_000000').mkdir()
    (out/'seq_000000'/'sentinel').write_bytes(b'unchanged')
    before = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='schema.*new empty directory'):
        generate_dataset(small_config(out), resume=True)
    assert before == {str(p.relative_to(out)): p.read_bytes() for p in out.rglob('*') if p.is_file()}


def test_seed_mismatch_rejected_before_output(tmp_path):
    with pytest.raises(ValueError, match='seed'):
        generate_sequence(small_config(tmp_path), 18, 0, tmp_path/'absent')
    assert not (tmp_path/'absent').exists()


@pytest.fixture(scope='module')
def completed_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp('completed')/'dataset'
    generate_dataset(small_config(out))
    return out


@pytest.mark.parametrize('damage', ['swap', 'split', 'boresight', 'star_counts', 'missing', 'entropy', 'spawn_key', 'constraints', 'source'])
def test_completed_identity_and_summaries_rejected(completed_dataset, tmp_path, damage):
    import shutil
    from starsim.generate import verify_completed, _signature
    out = tmp_path/'copy'
    shutil.copytree(completed_dataset, out)
    if damage == 'swap':
        (out/'seq_000000').rename(out/'temporary')
        (out/'seq_000001').rename(out/'seq_000000')
        (out/'temporary').rename(out/'seq_000001')
    else:
        path = out/'seq_000000'/'complete.json'
        if damage in ('entropy', 'spawn_key', 'constraints', 'source'):
            source_path = out/'seq_000000'/'sequence.json'
            sequence = json.loads(source_path.read_text())
            if damage == 'entropy':
                sequence['rng']['entropy'] = str(int(sequence['rng']['entropy'])+1)
            elif damage == 'spawn_key':
                sequence['rng']['spawn_key'][1] = 1
            elif damage == 'constraints':
                sequence['config_effective']['pointing']['split_constraints']['guard_deg'] += 1
            else:
                sequence['environment']['cpu'] = 'different'
            source_path.write_text(json.dumps(sequence))
            record = json.loads(path.read_text())
            record['files']['sequence.json'] = _sha256(source_path)
            path.write_text(json.dumps(record))
        elif damage == 'missing':
            path.unlink()
        else:
            record = json.loads(path.read_text())
            record[damage] = 'tampered' if damage == 'split' else [0]
            path.write_text(json.dumps(record))
    config = small_config(out)
    regions = make_splits(config.generator['dataset']['split'], config.generator['dataset']['seed'], 3)
    with pytest.raises(ValueError, match='corrupt'):
        verify_completed(out, 0, config.generator['dataset']['seed'], regions, _signature(config))


def test_resume_execution_parameters_and_timing(completed_dataset, tmp_path):
    import shutil
    out = tmp_path/'copy'
    shutil.copytree(completed_dataset, out)
    config = small_config(out)
    config.generator['dataset']['workers'] = 1
    config.generator['dataset']['out_dir'] = str(out/'..'/'copy')
    manifest = generate_dataset(config, resume=True)
    assert manifest['complete']
    for index, record in manifest['sequences'].items():
        timing = json.loads((out/f'seq_{int(index):06d}'/'timing.json').read_text())
        total = sum(sum(v) if isinstance(v, list) else v for v in timing['seconds'].values())
        assert total == pytest.approx(timing['total_seconds'], abs=1e-9)
        commit = record['commit_timing']
        assert sum(commit[k] for k in ('hash_seconds', 'publish_seconds', 'other_seconds')) == pytest.approx(commit['total_seconds'])
    config.generator['dataset']['n_sequences'] = 4
    with pytest.raises(ValueError, match='resume'):
        generate_dataset(config, resume=True)



def test_insufficient_memory_records_failed_run(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('starsim.generate.psutil.virtual_memory', lambda: SimpleNamespace(available=6*2**30))
    out = tmp_path/'failed'
    with pytest.raises(MemoryError, match='4 GiB'):
        generate_dataset(small_config(out))
    manifest = json.loads((out/'dataset.json').read_text())
    assert manifest['runs'][0]['status'] == 'failed'
    assert manifest['runs'][0]['active_seconds'] > 0
    assert manifest['active_seconds'] == manifest['runs'][0]['active_seconds']
    assert not manifest['complete']



def test_acceptance_reconstructs_config_from_source(completed_dataset):
    from tests.m5_production_acceptance import production_config
    from starsim.generate import _signature
    manifest = json.loads((completed_dataset/'dataset.json').read_text())
    reconstructed = production_config(manifest)
    assert _signature(reconstructed) == manifest['source']
