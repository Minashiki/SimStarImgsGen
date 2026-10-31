"""M4 review regressions: raw H, crop partitions, schema and provenance."""

import copy
from dataclasses import fields, make_dataclass, replace
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from starsim.config import load_config, validate_generator
from starsim.io import (TABLE_NAMES, _TARGET_FIELDS, _TARGET_FRAME_FIELDS, _MOTION_FRAMES,
                        _MOTION_STEPS, load_dense, load_objects, preflight_sequence,
                        save_objects, save_fits, validate_labels_manifest, write_sequence)
from starsim.labels import build_frame_labels
from starsim.render_labels import render_dense, validate_homography
from tests.test_m4_labels import CFG, candidate, sparse, scene, disk_sequence, verifier


@pytest.mark.parametrize("H", [np.eye(3, dtype=bool), np.eye(3).astype(str),
    np.zeros((3, 3)), np.ones((2, 3)), np.diag([1., 0., 1.]),
    [[1., 0, 0], [0, 1., 0], [1., 0, 0]],
    [[1., 0, 0], [0, 1., 0], [0, 0, np.inf]],
    [[1e308, 0, 0], [0, 1e308, 0], [0, 0, 1e308]],
    [[1., 0, 0], [0, 1., 0], [1e308, 0, 1e308]]])
def test_raw_h_rejections(H):
    with pytest.raises(ValueError):
        validate_homography(H, 32, 32)


@pytest.mark.parametrize("H", [-np.eye(3), np.diag([-1., 1., 1.]), np.eye(3)*2])
def test_reference_accepts_negative_denominators_and_mirrors(H):
    assert np.array_equal(validate_homography(H, 32, 32), H)
    render_dense([sparse([candidate(0)])], 0, H, dict(CFG, label_frame="reference"))


@pytest.mark.parametrize("H,extra", [(np.eye(3)*2, {}), (-np.eye(3), {}),
    (np.eye(3), {"origin": (0, 0)}), (np.eye(3), {"size": (32, 32)})])
def test_native_api_is_exact(H, extra):
    with pytest.raises(ValueError, match="native"):
        render_dense([sparse([candidate(0)])], 0, H, dict(CFG, **extra))


@pytest.mark.parametrize("stride", [1, 4])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_continuous_crop_before_grid_collision(stride, axis):
    frame = sparse([candidate(0), candidate(1, **{axis: 33.}),
                    candidate(2, x=5.1, flux=50.), candidate(3, has_flux=False)], width=33, height=33)
    dense = render_dense([frame], 0, np.eye(3), dict(CFG, label_frame="reference", stride=stride))
    assert dense["cropped"].dtype == np.int64
    assert dense["cropped"].tolist() == [1]
    assert dense["collision"].tolist() == [[2, 0]]
    assert dense["retained"].tolist() == [0]
    assert frame.candidates[1]["status_default"] == "labeled"
    assert dense["reg_mask"].sum() == 1
    assert dense["ignore"].any()


@pytest.mark.parametrize("stride", [1, 4])
def test_ignore_box_wholly_left_does_not_wrap_negative_stop(stride):
    frame = sparse([candidate(0, has_flux=False)])
    frame.sources[0]["box"] = [0, 5, 0, 5]
    H = np.array([[1., 0, -10], [0, 1., 0], [0, 0, 1.]])
    dense = render_dense([frame], 0, H, dict(CFG, label_frame="reference", canvas="reference", stride=stride))
    assert not dense["ignore"].any()
    assert dense["cropped"].size == 0


def test_cropped_support_and_previous_outside_canvas_motion():
    previous = sparse([candidate(0, x=25., in_defect_sat=True)])
    assert previous.candidates[0]["status_default"] == "ignore"
    current = sparse([candidate(0, x=5.), candidate(1, x=25.)], k=1)
    cfg = dict(CFG, label_frame="reference", size=(10, 10), motion=True)
    dense = render_dense([previous, current], 1, np.eye(3), cfg)
    assert dense["cropped"].tolist() == [1]
    assert dense["motion_mask"].sum() == 1
    cls, row, col = np.argwhere(dense["reg_mask"])[0]
    assert dense["motion"][cls, 0, row, col] == 20
    assert dense["ignore"].any()
    current.sources[1]["box"] = [20, 31, 20, 31]
    current.sources[0]["box"] = [0, 9, 0, 9]
    dense = render_dense([previous, current], 1, np.eye(3), cfg)
    assert not dense["ignore"].any()


def test_candidate_sigma_is_local_centroid_psf():
    labels, frame, sensor, psf, camera, motion, catalog = scene([12., 13.], raw=[0, 0])
    psf = replace(psf, field_growth=2.)
    population = SimpleNamespace(track_id=[])
    labels = build_frame_labels(frame, sensor, psf, camera, motion, population, catalog)
    c = labels.candidates[0]
    assert len(c["members"]) == 2
    assert c["sigma_psf"] == psf.sigma_iso(c["x"], c["y"])


def test_objects_schema_mixed_roundtrip_and_registry(tmp_path, disk_sequence):
    registry = {}
    for k, frame in enumerate(disk_sequence.objects):
        path = tmp_path/f"objects_{k}.npz"
        save_objects(path, frame, registry)
        loaded = load_objects(path)
        assert loaded.sources == frame.sources and loaded.candidates == frame.candidates
        with np.load(path, allow_pickle=False) as data:
            assert all(name in data for name in TABLE_NAMES)
            assert "sources" not in data and data["schema_version"] == 2
    frame = copy.deepcopy(disk_sequence.objects[0])
    frame.sources[0]["extra"] = 3
    with pytest.raises(ValueError, match="field"):
        save_objects(tmp_path/"drift.npz", frame, registry)
    frame = copy.deepcopy(disk_sequence.objects[0])
    for source in frame.sources:
        if source["class_id"] == 0:
            source["flux_e"] = int(source["flux_e"])
    with pytest.raises(ValueError, match="types"):
        save_objects(tmp_path/"types.npz", frame, registry)


@pytest.mark.parametrize("bad", [None, {"a": [float("nan")]}, [float("inf")]])
def test_writer_rejects_recursive_invalid_values(tmp_path, bad):
    frame, *_ = scene([12.])
    frame.candidates[0]["extra"] = bad
    with pytest.raises(ValueError, match="None|non-finite"):
        save_objects(tmp_path/"bad.npz", frame)


def test_json_width_empty_table_and_legacy(tmp_path):
    registry = {}
    frame, *_ = scene([12.])
    for text in ("a", "a"*1000):
        frame.sources[0]["extra"] = [text]
        save_objects(tmp_path/"new.npz", frame, registry)
    assert load_objects(tmp_path/"new.npz").candidates == frame.candidates
    np.savez(tmp_path/"old.npz", sources=np.empty(0))
    with pytest.raises(ValueError, match="missing"):
        load_objects(tmp_path/"old.npz")


def _preflight_inputs(n=2, t=0):
    motion = SimpleNamespace(**{name: np.zeros((n,)) for name in _MOTION_FRAMES},
                             **{name: np.zeros((n-1,)) for name in _MOTION_STEPS})
    declarations = [(name, np.ndarray) for name in sorted(_TARGET_FIELDS | _TARGET_FRAME_FIELDS)]
    pop = make_dataclass("Population", declarations)(**{
        name: np.zeros((t,) if name in _TARGET_FIELDS else (t, n))
        for name, annotation in declarations})
    pop.track_id = np.arange(t)
    plan = SimpleNamespace(stars=[None]*n, population=pop)
    from tests.freeze_m3_baseline import small_derived
    from starsim.camera import Camera
    from tests.test_m3_targets import _sensor
    derived = small_derived(load_config().derived)
    camera = Camera.from_derived(derived)
    sensor = _sensor(camera, n_frames=n)
    return plan, sensor, motion, {"dataset": {"n_frames": n}}, camera, derived


def test_preflight_zero_targets_and_rejects_before_output(tmp_path):
    args = _preflight_inputs()
    assert preflight_sequence(*args) == 2
    args[1].n_frames = 3
    out = tmp_path/"never_created"
    with pytest.raises(ValueError, match="frame counts"):
        write_sequence(out, args[0], None, args[1], None, None, args[2], None, None, None, args[3], derived_camera=args[5])
    assert not out.exists()


@pytest.mark.parametrize("damage", ["one_frame", "duplicate_track", "shape", "unregistered", "motion"])
def test_preflight_contract(damage):
    args = _preflight_inputs(n=1 if damage == "one_frame" else 2, t=2)
    plan, sensor, motion, cfg, camera, derived = args
    if damage == "duplicate_track":
        plan.population.track_id[:] = 0
    if damage == "shape":
        plan.population.x_ref = np.empty((2, 3))
    if damage == "unregistered":
        declarations = [(f.name, f.type) for f in fields(plan.population)]+[("new_array", np.ndarray)]
        plan.population = make_dataclass("NewPopulation", declarations)(**vars(plan.population), new_array=np.zeros(2))
    if damage == "motion":
        motion.speed_px_per_frame = np.zeros(2)
    with pytest.raises(ValueError):
        preflight_sequence(*args)


def test_manifest_cli_separate_readonly_input_and_stride3(disk_sequence, tmp_path):
    source = disk_sequence.directory
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    out = tmp_path/"derived"
    child = subprocess.run([sys.executable, "-m", "starsim.render_labels", str(source), str(out),
                            "--stride", "3", "--label-frame", "reference"], capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    manifest = validate_labels_manifest(out, source)
    assert manifest["config"]["stride"] == 3
    for k, frame in enumerate(disk_sequence.objects):
        dense = load_dense(out, source, k)
        assert np.array_equal(dense["H"], frame.H_to_ref)
        assert not dense["cropped"].size
    validate_labels_manifest(source, source)


@pytest.mark.parametrize("damage", ["input", "output", "missing", "version", "name", "duplicate", "entry_missing"])
def test_manifest_rejects_corruption(disk_sequence, tmp_path, damage):
    source, out = tmp_path/"input", tmp_path/"derived"
    shutil.copytree(disk_sequence.directory, source)
    shutil.copytree(disk_sequence.directory, out)
    path = out/"labels_manifest.json"
    manifest = json.loads(path.read_text())
    if damage == "input":
        with (source/"objects_000.npz").open("ab") as stream:
            stream.write(b"changed")
    elif damage == "output":
        with (out/"labels_000.npz").open("ab") as stream:
            stream.write(b"changed")
    elif damage == "missing":
        (out/"labels_000.npz").unlink()
    elif damage == "version":
        manifest["algorithm_versions"]["labels"] += 1
    elif damage == "name":
        manifest["dense"][0]["name"] = "labels_0000.npz"
    elif damage == "duplicate":
        manifest["objects"].append(manifest["objects"][0])
    else:
        manifest["dense"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_labels_manifest(out, source)


@pytest.mark.parametrize("key", ["per_class_offset", "streak_geom"])
def test_unsupported_false_config(key):
    cfg = copy.deepcopy(load_config().generator)
    cfg["labels"][key] = False
    with pytest.raises(ValueError, match=key):
        validate_generator(cfg)


def test_identity_and_partition_survive_optimized_python():
    script = '''
from starsim.labels import assign_status, assert_partition
from tests.test_m4_labels import candidate, sparse, CFG
f = sparse([candidate(0), candidate(1)])
f.candidates[1]['identity'] = f.candidates[0]['identity']
try:
    assign_status([f], CFG)
except ValueError:
    pass
else:
    raise RuntimeError('identity check disabled')
f.candidates[1]['identity'] = 'different'
f.candidates[1]['members'] = [0]
try:
    assert_partition(f)
except ValueError:
    pass
else:
    raise RuntimeError('partition check disabled')
'''
    child = subprocess.run([sys.executable, "-O", "-c", script], capture_output=True, text=True)
    assert child.returncode == 0, child.stderr


def test_fits_lossless_unsigned_pixels(tmp_path):
    pixels = np.array([[0, 32767, 32768, 65535]], dtype=np.uint16)
    path = tmp_path/"pixels.fits.fz"
    save_fits(path, pixels, {}, frame_index=0, exposure_s=1.)
    with fits.open(path) as hdus:
        assert hdus[1].data.dtype.kind == "u"
        assert np.array_equal(hdus[1].data, pixels)


def test_no_dense_has_no_manifest_and_nonempty_output_is_rejected(disk_sequence, tmp_path):
    args = list(disk_sequence.writer_args)
    from starsim.rng import SequenceRng
    args[1] = SequenceRng.from_metadata(args[1].metadata())
    args[-1] = copy.deepcopy(args[-1])
    args[-1]["labels"]["save_dense"] = False
    out = tmp_path/"without_dense"
    write_sequence(out, *args, derived_camera=disk_sequence.derived)
    assert (out/"sequence.json").is_file()
    assert not (out/"labels_manifest.json").exists()
    assert not list(out.glob("labels_*.npz"))
    with pytest.raises(ValueError, match="new or empty"):
        write_sequence(out, *args, derived_camera=disk_sequence.derived)


@pytest.mark.parametrize("versions", [None, [], {"labels": True}, {"labels": -1}, {"labels": 1.0}])
def test_verifier_version_structure(disk_sequence, verifier, tmp_path, versions):
    module, catalog = verifier
    directory = tmp_path/"sequence"
    shutil.copytree(disk_sequence.directory, directory)
    path = directory/"sequence.json"
    sequence = json.loads(path.read_text())
    sequence["algorithm_versions"] = versions
    path.write_text(json.dumps(sequence))
    with pytest.raises(module.VerificationError, match="algorithm_versions"):
        module.verify(directory, catalog)


def test_native_saved_h_is_exact_identity(tmp_path):
    from starsim.io import save_dense
    frame = sparse([candidate(0)])
    save_dense(tmp_path/"dense.npz", render_dense([frame], 0, np.eye(3), CFG))
    with np.load(tmp_path/"dense.npz") as dense:
        assert dense["H"].dtype == np.float64
        assert np.array_equal(dense["H"], np.eye(3))
        assert dense["cropped"].size == 0


@pytest.mark.parametrize("position", ["center", "endpoint", "previous"])
def test_every_render_mapping_rejects_invalid_intermediates(position):
    # Footprint stays finite and strictly convex, but outlying geometry hits a pole.
    H = np.array([[1., 0, 0], [0, 1., 0], [.01, 0, 1.]])
    current = sparse([candidate(0, cls=2)])
    previous = sparse([candidate(0, cls=2)], k=0)
    current.frame_index = 1
    current.H_to_ref = H
    previous.H_to_ref = H
    if position == "center":
        current.candidates[0]["x"] = -100.
    elif position == "endpoint":
        current.candidates[0]["x1"] = -100.
    else:
        previous.candidates[0]["x"] = -100.
    with pytest.raises(ValueError, match="homography mapping"):
        render_dense([previous, current], 1, H, dict(CFG, label_frame="reference", motion=True))


@pytest.mark.parametrize("array", ["H_to_ref", "width", "frame_index"])
def test_writer_rejects_nonfinite_outside_tables(tmp_path, array):
    frame, *_ = scene([12.])
    if array == "H_to_ref":
        frame.H_to_ref[0, 0] = np.nan
    else:
        setattr(frame, array, np.inf)
    with pytest.raises(ValueError, match="non-finite"):
        save_objects(tmp_path/"bad.npz", frame)


def test_reader_rejects_duplicate_source_indices(tmp_path):
    frame, *_ = scene([12., 20.])
    path = tmp_path/"duplicate.npz"
    save_objects(path, frame)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    arrays["sources_stars"]["source_index"][:] = 0
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="source_index"):
        load_objects(path)


def test_independent_verifier_rejects_legacy_and_principal_point(disk_sequence, verifier, tmp_path):
    module, catalog = verifier
    directory = tmp_path/"seq"
    shutil.copytree(disk_sequence.directory, directory)
    path = directory/"sequence.json"
    sequence = json.loads(path.read_text())
    sequence["camera"]["cx"] += 1
    path.write_text(json.dumps(sequence))
    with pytest.raises(module.VerificationError, match="unsupported principal point"):
        module.verify(directory, catalog)
    shutil.copyfile(disk_sequence.directory/"sequence.json", path)
    path = directory/"objects_000.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files if key != "schema_version"}
    np.savez_compressed(path, **arrays)
    with pytest.raises(module.VerificationError, match="missing"):
        module.verify(directory, catalog)


def test_disk_truth_regenerates_dense_bit_for_bit(disk_sequence):
    sequence = json.loads((disk_sequence.directory/"sequence.json").read_text())
    cfg = sequence["config_effective"]["labels"]
    disk_objects = [load_objects(disk_sequence.directory/f"objects_{k:03d}.npz")
                    for k in range(len(disk_sequence.objects))]
    for k, frame in enumerate(disk_objects):
        a = render_dense(disk_objects, k, frame.H_to_ref, cfg)
        b = render_dense(disk_sequence.objects, k, frame.H_to_ref, cfg)
        assert a.keys() == b.keys()
        assert all(np.array_equal(a[key], b[key]) for key in a)


def test_crop_uses_recomputed_status_and_same_frame_identity(tmp_path):
    frame = sparse([candidate(0, x=33.)], width=33)
    cfg = dict(CFG, label_frame="reference", label_snr_eff_min=100.)
    dense = render_dense([frame], 0, np.eye(3), cfg)
    assert dense["cropped"].size == dense["retained"].size == 0
    assert frame.candidates[0]["status_default"] == "labeled"
    frame, *_ = scene([12., 20.])
    frame.candidates[1]["identity"] = frame.candidates[0]["identity"]
    with pytest.raises(ValueError, match="identity"):
        save_objects(tmp_path/"duplicate.npz", frame)


def test_complete_reference_canvas_sequence(disk_sequence, tmp_path):
    from starsim.rng import SequenceRng
    args = list(disk_sequence.writer_args)
    args[1] = SequenceRng.from_metadata(args[1].metadata())
    args[-1] = copy.deepcopy(args[-1])
    args[-1]["labels"].update(label_frame="reference", canvas="reference")
    out = tmp_path/"reference_canvas"
    objects = write_sequence(out, *args, derived_camera=disk_sequence.derived)
    validate_labels_manifest(out, out)
    for k, frame in enumerate(objects):
        dense = load_dense(out, out, k)
        assert dense["size"].tolist() == [frame.width, frame.height]
        assert np.array_equal(dense["H"], frame.H_to_ref)


@pytest.mark.parametrize("array", ["frame_index", "size"])
def test_independent_verifier_nonfinite_objects_arrays(disk_sequence, verifier, tmp_path, array):
    module, catalog = verifier
    directory = tmp_path/"sequence"
    shutil.copytree(disk_sequence.directory, directory)
    path = directory/"objects_000.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    arrays[array] = arrays[array].astype(float)
    arrays[array].flat[0] = np.nan
    np.savez_compressed(path, **arrays)
    with pytest.raises(module.VerificationError, match="frame_index|size|non-finite"):
        module.verify(directory, catalog)
