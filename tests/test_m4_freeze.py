"""D67–D71 frozen-round regressions."""
import copy
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from starsim.config import load_config, validate_label_cfg
from starsim.io import (CANDIDATE_COMMON, CANDIDATE_STAR, CANDIDATE_TARGET,
    load_objects, save_objects, save_dense, validate_labels_manifest, write_sequence)
from starsim.metadata import git_provenance
from starsim.render_labels import render_dense, decode_dense
from tests.test_m4_labels import CFG, candidate, sparse, scene, disk_sequence, verifier


def test_ignore_pole_and_retained_endpoint_crossing():
    H = np.array([[1., 0, 0], [0, 1., 0], [.01, 0, 1.]])
    ignored = sparse([candidate(0, x=-100., has_flux=False)])
    ignored.sources[0]['box'] = [-1, -1, -1, -1]
    render_dense([ignored], 0, H, dict(CFG, label_frame='reference'))
    retained = sparse([candidate(0, cls=2, x1=-101.)])
    with pytest.raises(ValueError, match='homography mapping'):
        render_dense([retained], 0, H, dict(CFG, label_frame='reference'))
    # A loser does not need endpoint geometry.
    loser = sparse([candidate(0, cls=2, flux=200.), candidate(1, cls=2, x1=-101.)])
    render_dense([loser], 0, H, dict(CFG, label_frame='reference'))


def test_prev_transform_validated_only_when_needed():
    previous = sparse([candidate(0)], H=np.diag([1., 0., 1.]))
    current = sparse([candidate(0)], k=1)
    cfg = dict(CFG, label_frame='reference', canvas='reference', motion=True)
    with pytest.raises(ValueError):
        render_dense([previous, current], 1, np.eye(3), cfg)
    previous.candidates[0]['in_image'] = False
    render_dense([previous, current], 1, np.eye(3), cfg)


def test_relative_ulp_crop_and_unencodable_edge():
    frame = sparse([candidate(0, x=np.nextafter(.5, 0), y=0.)], width=4, height=1)
    dense = render_dense([frame], 0, np.eye(3), dict(CFG, label_frame='reference', origin=(-3, 0), size=(4, 1)))
    assert dense['cropped'].tolist() == [0]
    frame.width = 1
    with pytest.raises(ValueError, match='encoded'):
        render_dense([frame], 0, np.eye(3), CFG)


@pytest.mark.parametrize('H', [np.eye(3, dtype=bool), np.eye(3).astype(str)])
def test_native_rejects_non_numeric_identity(H):
    with pytest.raises(ValueError, match='native'):
        render_dense([sparse([candidate(0)])], 0, H, CFG)


def test_native_rejects_outside_labeled_center():
    with pytest.raises(ValueError, match='outside'):
        render_dense([sparse([candidate(0, x=40.)])], 0, np.eye(3), CFG)


def test_offset_float32_large_stride_and_clipping(tmp_path):
    frame = sparse([candidate(0, x=.03, y=.03)], width=256, height=256)
    dense = render_dense([frame], 0, np.eye(3), dict(CFG, stride=256))
    path = tmp_path/'offset.npz'
    save_dense(path, dense)
    with np.load(path) as stored:
        assert stored['offset'].dtype == np.float32
        assert abs(decode_dense(stored)[0]['x']-.03) < 1e-3
    dense['offset'].flat[0] = .5
    save_dense(path, dense)
    with np.load(path) as stored:
        assert stored['offset'].flat[0] == np.nextafter(np.float32(.5), np.float32(0))


@pytest.mark.parametrize('key,value', [('label_snr_eff_min', float('nan')), ('label_snr_min', -1),
    ('stride', 1.2), ('snr_mode', 'bad'), ('motion', 1), ('per_class_offset', False)])
def test_label_override_validation(key, value):
    cfg = copy.deepcopy(load_config().generator['labels']); cfg[key] = value
    with pytest.raises(ValueError):
        validate_label_cfg(cfg)


def test_nan_cli_writes_nothing(disk_sequence, tmp_path):
    out = tmp_path/'derived'
    child = subprocess.run([sys.executable, '-m', 'starsim.render_labels', str(disk_sequence.directory),
        str(out), '--snr-eff-min', 'nan'], capture_output=True, text=True)
    assert child.returncode != 0
    assert not out.exists()


@pytest.mark.parametrize('damage', ['stride', 'origin', 'missing'])
def test_manifest_consistency(disk_sequence, tmp_path, damage):
    out = tmp_path/'sequence'; shutil.copytree(disk_sequence.directory, out)
    path = out/'labels_manifest.json'; manifest = json.loads(path.read_text())
    if damage == 'stride': manifest['config']['stride'] += 1
    elif damage == 'origin': manifest['origin'][0] += 1
    else: del manifest['dense']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): validate_labels_manifest(out, out)


@pytest.mark.parametrize('damage', ['frame', 'size', 'class', 'missing', 'H', 'empty', 'schema', 'extra', 'removed'])
def test_objects_reader_and_independent_verifier_contract(disk_sequence, verifier, tmp_path, damage):
    out = tmp_path/'sequence'; shutil.copytree(disk_sequence.directory, out)
    path = out/'objects_000.npz'
    with np.load(path) as data: arrays = {k: data[k].copy() for k in data.files}
    if damage == 'frame': arrays['frame_index'] = np.array(.9)
    elif damage == 'size': arrays['size'] = np.array([32.9, 32.9])
    elif damage == 'class': arrays['stars']['class_id'][:] = 9
    elif damage == 'missing': del arrays['points']
    elif damage == 'H': arrays['H_to_ref'] = arrays['H_to_ref'].astype(np.float32)
    elif damage == 'empty': arrays['points'] = np.empty(0, dtype=[('id', 'U2')])
    elif damage == 'schema': arrays['schema_version'] = np.array(1)
    elif damage == 'extra':
        original = arrays['stars']; replacement = np.empty(original.shape, dtype=original.dtype.descr+[('extra', 'f8')])
        for name in original.dtype.names: replacement[name] = original[name]
        arrays['stars'] = replacement
    else: arrays['stars'] = arrays['stars'][[n for n in arrays['stars'].dtype.names if n != 'snr']]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError): load_objects(path)
    module, catalog = verifier
    with pytest.raises(module.VerificationError): module.verify(out, catalog)


@pytest.mark.parametrize('cls', [0, 1, 2])
@pytest.mark.parametrize('extra', [False, True])
def test_writer_whitelist(disk_sequence, tmp_path, cls, extra):
    frame = copy.deepcopy(disk_sequence.objects[0])
    c = next(c for c in frame.candidates if c['class_id'] == cls)
    if extra: c['extra'] = 1
    else: del c['snr']
    with pytest.raises(ValueError): save_objects(tmp_path/'bad.npz', frame)


def test_all_target_fields_are_copied_and_checked(disk_sequence, tmp_path):
    for frame in disk_sequence.objects:
        for c in frame.candidates:
            assert set(c) == CANDIDATE_COMMON | (CANDIDATE_STAR if c['class_id'] == 0 else CANDIDATE_TARGET)
            if c['class_id']:
                member = frame.sources[c['members'][0]]
                assert all(c[k] == member[k] for k in CANDIDATE_TARGET | set('x y x_ref y_ref flux_e ra dec'.split()))
    frame = copy.deepcopy(disk_sequence.objects[0])
    c = next(c for c in frame.candidates if c['class_id'])
    c['x_ref'] = np.nextafter(c['x_ref'], float('inf')).item()
    with pytest.raises(ValueError, match='differs from member'): save_objects(tmp_path/'bad.npz', frame)


@pytest.mark.parametrize('damage', ['derived', 'background_e', 'vignetting', 'prnu', 'dark_e',
    'row_offset_e', 'col_offset_e', 'hot_state', 'hot_row', 'dead_col', 'cx'])
def test_camera_sensor_preflight_before_output(disk_sequence, tmp_path, damage):
    from dataclasses import replace
    args = list(disk_sequence.writer_args)
    sensor = copy.deepcopy(args[2]); args[2] = sensor
    derived = disk_sequence.derived
    if damage == 'derived': derived = replace(derived, width=derived.width+1)
    elif damage == 'cx': sensor.cx += 1
    elif damage == 'hot_row': sensor.hot_row = np.array([.5])
    elif damage == 'dead_col': sensor.dead_col = np.array([args[4].width])
    else: setattr(sensor, damage, np.empty((1,)))
    out = tmp_path/'never'
    with pytest.raises(ValueError): write_sequence(out, *args, derived_camera=derived)
    assert not out.exists()


def test_git_failure_and_untracked(monkeypatch):
    def failure(*a, **kw): raise OSError('git unavailable')
    monkeypatch.setattr(subprocess, 'run', failure)
    assert git_provenance() == dict(git_commit=None, git_dirty=None)


def test_git_uses_package_directory_and_untracked(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        output = 'D:/StarProcess/GenStarImgs' if '--show-toplevel' in args else ('?? tests/new.py' if 'status' in args else 'a'*40)
        return type('Result', (), dict(stdout=output))()
    monkeypatch.setattr(subprocess, 'run', run)
    assert git_provenance() == dict(git_commit='a'*40, git_dirty=True)
    assert 'starsim' in str(calls[0][1]['cwd'])
    assert '--untracked-files=all' in calls[-1][0]
