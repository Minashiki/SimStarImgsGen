"""M4 constructive rules, dense round trips, disk-only independent audit."""

import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from starsim.attitude import pointing_roll_to_matrix
from starsim.blend import blend_stars
from starsim.camera import apply_homography, unproject_icrs
from starsim.catalog import StarCatalog, default_cache_path
from starsim.config import load_config, sample_sequence_params
from starsim.io import load_objects, load_sequence, save_dense, save_objects, write_json, write_sequence
from starsim.labels import (FrameLabels, assign_status, build_frame_labels, climb_peak,
                            group_peak, pixel_groups)
from starsim.metadata import validate_versions
from starsim.motion import simulate_sequence
from starsim.noise import expose_frame
from starsim.photometry import anchor_photometry
from starsim.render import (RenderedSources, _empty_targets, centers_in_image, peak_energy_fraction,
                            render_points)
from starsim.render_labels import (canvas_geometry, decode_coordinate, decode_dense,
                                  encode_coordinate, footprint, render_dense, valid_mask)
from starsim.rng import SequenceRng, root_seed, sequence_seed
from starsim.runtime import runtime_environment
from starsim.sequence import FrameResult, prepare_sequence
from starsim.psf import PsfModel
from tests.test_m3_targets import _camera, _psf, _sensor, _pcg

ROOT = Path(__file__).resolve().parents[1]
CFG = dict(snr_mode="effective", label_snr_eff_min=0., label_snr_min=0., snr_window="all",
           never_full_policy="label", label_frame="native", canvas="union", stride=1,
           sigma_hm=1., motion=False, save_dense=True)


def scene(x, y=None, flux=None, *, side=32, sigma=1.0, raw=None, permutation=None):
    camera = _camera(side)
    psf = _psf(camera, sigma0=sigma, field_growth=0., max_ellipticity=0.)
    x = np.array(x, dtype=float)
    y = np.full(len(x), 16.) if y is None else np.array(y, dtype=float)
    flux = np.full(len(x), 1000.) if flux is None else np.array(flux, dtype=float)
    rad = np.full(len(x), 6, dtype=np.int64)
    pr, pc, ss, image, r0, r1, c0, c1 = render_points((side, side), x, y, flux, psf, rad)
    ids = np.arange(1, len(x)+1, dtype=np.int32)
    sources = RenderedSources(image, x, y, flux, np.full(len(x), 8.), rad, r0, r1, c0, c1,
                              pr, pc, ss, r0 >= 0, centers_in_image(x, y, side, side), ids, ids)
    R = pointing_roll_to_matrix(.8, .4, .2)
    xyz = unproject_icrs(camera, R, x, y)
    catalog = SimpleNamespace(tyc1=ids.copy(), tyc2=ids.copy(),
        ra_deg=np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360,
        dec_deg=np.degrees(np.arcsin(xyz[:, 2])))
    if permutation is not None:
        for key, arr in vars(sources).items():
            if key != "image":
                setattr(sources, key, arr[permutation])
    sensor = _sensor(camera, n_frames=2)
    exposure = expose_frame(sources, sensor, 0, _pcg(91), psf)
    blend = blend_stars(sources.x, sources.y, sources.flux_e, psf, sources.radius,
                        box_row0=sources.box_row0, box_row1=sources.box_row1,
                        box_col0=sources.box_col0, box_col1=sources.box_col1)
    if raw is not None:
        object.__setattr__(blend, "group_id", np.array(raw, dtype=int))
    frame = FrameResult(0, sources, _empty_targets(side, side), exposure, blend)
    motion = SimpleNamespace(H_to_ref=np.array([np.eye(3)]), matrix=np.array([R]))
    labels = build_frame_labels(frame, sensor, psf, camera, motion,
                               SimpleNamespace(track_id=[]), catalog)
    assign_status([labels], CFG)
    return labels, frame, sensor, psf, camera, motion, catalog


def test_plateaus_strict_equal_row_major_and_higher_far_end():
    image = np.zeros((7, 8))
    image[3, 2:6] = 5
    assert climb_peak(image, 3, 5) == (3, 2)
    image[2, 6] = 6
    assert climb_peak(image, 3, 2) == (2, 6)
    image[3, 3] = np.nextafter(5., 6.)
    assert climb_peak(image, 3, 2) == (3, 3)


def test_pixel_merge_wing_ignore_no_flux_and_determinism():
    merged, *_ = scene([15., 16.], flux=[1000, 800], raw=[0, 1])
    assert len(merged.candidates) == 1
    assert merged.candidates[0]["members"] == [0, 1]
    wing, frame, *_ = scene([12., 15.], flux=[100000, 20], raw=[0, 1])
    start = (int(frame.stars.peak_row[1]), int(frame.stars.peak_col[1]))
    end = climb_peak(frame.stars.image, *start)
    assert start != end and end == (16, 12)
    assert wing.candidates[1]["ignore_reason_default"] == "pixel_dominated"
    empty, *_ = scene([100., 12.])
    assert len(empty.sources) == 2
    assert empty.candidates[0]["ignore_reason_default"] == "no_flux"
    first, *_ = scene([12., 15., 25.], flux=[100000, 20, 1000], raw=[0, 0, 1])
    second, *_ = scene([12., 15., 25.], flux=[100000, 20, 1000], permutation=[2, 0, 1], raw=[1, 0, 0])
    assert first.sources == second.sources
    assert first.candidates == second.candidates


def test_equal_star_plateau_and_outside_group_center():
    labels, frame, *_ = scene([12., 13.], flux=[1000, 1000], raw=[0, 1])
    assert frame.stars.image[16, 12] == frame.stars.image[16, 13]
    assert climb_peak(frame.stars.image, 16, 13) == (16, 12)
    assert len(labels.candidates) == 1
    labels, *_ = scene([-.7, .6], flux=[100000, 20], raw=[0, 1])
    assert len(labels.candidates) == 1
    assert labels.candidates[0]["ignore_reason_default"] == "center_outside"


def test_clipped_flags_require_clipping_and_final_group_ownership():
    labels, frame, sensor, psf, camera, motion, catalog = scene([15., 16.], raw=[0, 1])
    assert not labels.candidates[0]["in_defect_sat"]
    frame.exposure.peak_clipped[:] = True
    frame.exposure.peak_owner[:] = [1, 0]
    labels = build_frame_labels(frame, sensor, psf, camera, motion, SimpleNamespace(track_id=[]), catalog)
    assign_status([labels], CFG)
    assert labels.candidates[0]["saturated"]
    assert labels.candidates[0]["status_default"] == "labeled"
    frame.exposure.peak_owner[1] = -1
    labels = build_frame_labels(frame, sensor, psf, camera, motion, SimpleNamespace(track_id=[]), catalog)
    assign_status([labels], CFG)
    assert labels.candidates[0]["saturated"] and labels.candidates[0]["in_defect_sat"]
    assert labels.candidates[0]["ignore_reason_default"] == "clipped_by_other"


def moving_scene(x=16., length=0., flux=30000., star_flux=20.):
    from starsim.render import render_moving_targets
    from starsim.targets import _project_track

    _, frame, sensor, psf, camera, motion, catalog = scene([16.], flux=[star_flux])
    shape = _project_track(np.full(2, x), np.full(2, 16.), np.full(2, length),
                           np.ones(2), np.zeros(2), np.zeros(2), np.array([np.eye(3)]*2),
                           camera, length >= 5)
    values = {key: np.array([value]) for key, value in shape.items()}
    values.update(track_id=np.array([7]), kind=np.array([int(length >= 5)]),
                  x_ref=np.full((1, 2), x), y_ref=np.full((1, 2), 16.),
                  L_true=np.full((1, 2), length), mag_eq=np.full((1, 2), 8.),
                  tier_design=np.array(["bright"]), duty=np.array([.5]), t_exp_eff=np.array([.5]),
                  direction_x=np.array([1.]), direction_y=np.array([0.]), speed=np.array([10.]))
    pop = SimpleNamespace(**values)
    targets = render_moving_targets(pop.track_id, pop.kind, pop.x_c[:, 0], pop.y_c[:, 0],
        pop.x1[:, 0], pop.y1[:, 0], pop.x2[:, 0], pop.y2[:, 0], np.array([flux]), psf,
        sensor.reference_sigma())
    frame.targets = targets
    frame.exposure = expose_frame(frame.stars, sensor, 0, _pcg(15), psf, targets)
    return frame, sensor, psf, camera, motion, pop, catalog


def test_moving_light_dominates_star_and_center_not_peak_controls_clipping():
    args = moving_scene()
    labels = build_frame_labels(*args)
    assign_status([labels], CFG)
    assert labels.candidates[0]["ignore_reason_default"] == "pixel_dominated"
    assert labels.candidates[0]["dominant_source"] == 1
    assert labels.candidates[1]["status_default"] == "labeled"
    frame = args[0]
    frame.exposure.peak_clipped[1] = True
    frame.exposure.peak_owner[1] = 0
    frame.exposure.center_clipped[0] = False
    labels = build_frame_labels(*args)
    assign_status([labels], CFG)
    assert labels.candidates[1]["status_default"] == "labeled"
    assert not labels.candidates[1]["in_foreign_sat"]
    frame.exposure.center_clipped[0] = True
    frame.exposure.center_owner[0] = 0
    labels = build_frame_labels(*args)
    assign_status([labels], CFG)
    assert labels.candidates[1]["ignore_reason_default"] == "clipped_by_other"
    # Start already is a local maximum, but its dominant source is the target.
    light = frame.stars.image+frame.targets.image
    start = (frame.stars.peak_row[0], frame.stars.peak_col[0])
    assert climb_peak(light, *start) == start


def test_three_clipping_flags_and_reason_priority():
    labels, frame, sensor, psf, camera, motion, catalog = scene([12., 13., 14., 28.], raw=[0, 0, 0, 1])
    frame.exposure.peak_clipped[:3] = True
    frame.exposure.peak_owner[:3] = [0, 3, -1]
    labels = build_frame_labels(frame, sensor, psf, camera, motion, SimpleNamespace(track_id=[]), catalog)
    assign_status([labels], CFG)
    c = labels.candidates[0]
    assert c["saturated"] and c["in_foreign_sat"] and c["in_defect_sat"]
    assert c["ignore_reason_default"] == "clipped_by_other"
    c["dominant_source"] = 3
    assign_status([labels], CFG)
    assert c["ignore_reason_default"] == "pixel_dominated"


def test_truncated_streak_true_center_outside_visible_ignore_and_never_full():
    inside = build_frame_labels(*moving_scene(x=.1, length=18.))
    assign_status([inside], CFG)
    c = inside.candidates[1]
    assert c["truncated"] and c["in_image"] and c["never_full"]
    dense = render_dense([inside], 0, np.eye(3), CFG)
    actual = next(p for p in decode_dense(dense) if p["class_id"] == 2)
    assert abs(actual["x"]-.1) < .01 and abs(actual["length"]-18) < .01
    assign_status([inside], dict(CFG, never_full_policy="ignore"))
    assert c["ignore_reason_default"] == "never_full"
    outside = build_frame_labels(*moving_scene(x=-2., length=18.))
    assign_status([outside], CFG)
    assert outside.candidates[1]["ignore_reason_default"] == "center_outside"
    dense = render_dense([outside], 0, np.eye(3), CFG)
    assert not dense["reg_mask"][2].any() and dense["ignore"][2].any()


def test_snr_rules_all_frames_same_decision_and_motion_just_entered():
    a = sparse([candidate(0, snr=2., in_image=False)], k=0)
    b = sparse([candidate(0, snr=2.)], k=1)
    c = sparse([candidate(0, snr=2.)], k=2)
    cfg = dict(CFG, label_snr_eff_min=3., motion=True)
    assign_status([a, b, c], cfg)
    assert all(f.candidates[0]["snr_eff"] == math.sqrt(8) for f in (a, b, c))
    assert [f.candidates[0]["ignore_reason_default"] for f in (a, b, c)] == ["center_outside", "low_snr", "low_snr"]
    dense = render_dense([a, b], 1, np.eye(3), dict(CFG, motion=True))
    assert not dense["motion_mask"].any()


def test_perturbed_homography_uses_image_transform_and_valid_footprint():
    frame = sparse([candidate(0, x=9.15, y=8.25)])
    H = np.array([[1., .1, 2.35], [-.07, 1., -1.2], [.001, .002, 1.]])
    dense = render_dense([frame], 0, H, dict(CFG, label_frame="reference", origin=(-5, -5), size=(50, 50)))
    point = decode_dense(dense)[0]
    expected = H @ np.array([9.15, 8.25, 1.])
    assert np.max(abs(expected[:2]/expected[2]-[point["x"], point["y"]])) < .01


def test_long_streak_half_precision_heatmap_has_one_nms_peak(tmp_path):
    frame = sparse([candidate(0, cls=2, x=256., y=256., x1=6., y1=256., x2=506., y2=256.)],
                   width=512, height=512)
    dense = render_dense([frame], 0, np.eye(3), CFG)
    save_dense(tmp_path/"long.npz", dense)
    with np.load(tmp_path/"long.npz") as disk:
        decoded = decode_dense(disk)
    assert len(decoded) == 1 and decoded[0]["length"] == 500.


def test_group_peak_not_member_peak():
    # Two broad overlapping sources with peaks at columns 1 and 3; sum peaks at 2.
    a = np.array([[1., 4., 3., 1., 0.]])
    b = a[:, ::-1].copy()
    assert np.argmax(a) == 1 and np.argmax(b) == 3
    sources = [dict(box=[0, 0, 0, 4]), dict(box=[0, 0, 0, 4])]
    assert group_peak((0, 1), sources, [a, b]) == (6., 0, 2)


def candidate(gid, cls=0, x=5., y=5., flux=100., **kwargs):
    value = dict(label_group_id=gid, source_index=gid, members=[gid], components=[str(gid)],
        identity=f"{cls}:{gid}", stable_key=[gid], class_id=cls, x=x, y=y, flux_e=flux,
        in_image=True, has_flux=True, dominant_source=-1, in_foreign_sat=False,
        in_defect_sat=False, saturated=False, snr=10., never_full=False, sigma_psf=1.,
        x1=x-8, y1=y-2, x2=x+8, y2=y+2)
    value.update(kwargs)
    return value


def sparse(candidates, width=32, height=32, k=0, H=None):
    sources = [dict(candidate_id=c["label_group_id"], box=[0, height-1, 0, width-1]) for c in candidates]
    result = FrameLabels(k, width, height, sources, candidates, np.eye(3) if H is None else H)
    assign_status([result], CFG)
    return result


def test_identity_windows_ignore_does_not_remove_snr_and_split():
    a = sparse([candidate(0, snr=2.)], k=0)
    b = sparse([candidate(0, snr=2., in_defect_sat=True)], k=1)
    c = sparse([candidate(0, snr=20., identity="different")], k=2)
    cfg = dict(CFG, label_snr_eff_min=3.)
    assign_status([a, b, c], cfg)
    assert a.candidates[0]["snr_eff"] == math.sqrt(8)
    assert a.candidates[0]["identity_frame_count"] == 2
    assert a.candidates[0]["ignore_reason_default"] == "low_snr"
    assert b.candidates[0]["ignore_reason_default"] == "clipped_by_other"
    assign_status([a, b, c], dict(cfg, snr_window=2))
    assert b.candidates[0]["identity_frame_count"] == 1


@pytest.mark.parametrize("stride", [1, 4])
@pytest.mark.parametrize("reference", [False, True])
def test_dense_disk_roundtrip_nms_offset_geometry_motion(tmp_path, stride, reference):
    cs = [candidate(0, x=5.123, y=8.321), candidate(1, cls=1, x=13.23, y=16.32),
          candidate(2, cls=2, x=23.12, y=24.41)]
    frame = sparse(cs)
    prev = sparse([dict(c, x=c["x"]-.7, y=c["y"]+.3) for c in cs], k=0)
    frame.frame_index = 1
    H = np.array([[1., .03, -2], [-.02, 1., 1], [.0002, 0, 1.]]) if reference else np.eye(3)
    frame.H_to_ref = H
    prev.H_to_ref = H
    cfg = dict(CFG, stride=stride, label_frame="reference" if reference else "native", motion=True)
    seq = [prev, frame]
    dense = render_dense(seq, 1, H, cfg)
    save_dense(tmp_path/"dense.npz", dense)
    with np.load(tmp_path/"dense.npz", allow_pickle=False) as disk:
        decoded = decode_dense(disk)
    assert len(decoded) == 3
    for actual, c in zip(decoded, cs):
        expected = apply_homography(H, c["x"], c["y"])
        assert np.max(abs(expected-[actual["x"], actual["y"]])) < .01
        if c["class_id"] == 2:
            p = apply_homography(H, c["x1"], c["y1"])
            q = apply_homography(H, c["x2"], c["y2"])
            assert abs(actual["length"]-np.linalg.norm(q-p)) < .01
            assert abs(math.sin(actual["angle"]-math.atan2(q[1]-p[1], q[0]-p[0]))) < .01
    for cls, row, col in np.argwhere(dense["reg_mask"]):
        assert dense["motion_mask"][cls, row, col]
        p = apply_homography(H, cs[cls]["x"]-.7, cs[cls]["y"]+.3)
        q = apply_homography(H, cs[cls]["x"], cs[cls]["y"])
        assert np.max(abs(dense["motion"][cls, :, row, col]-(p-q))) < .01
    assert not render_dense(seq, 0, H, cfg)["motion_mask"].any()


def test_collision_priority_equal_flux_stable_id_and_class_ignore():
    frame = sparse([candidate(0, cls=2, x=5, flux=10), candidate(1, cls=2, x=5.1, flux=20),
                    candidate(2, cls=2, x=5.2, flux=20),
                    candidate(3, cls=0, x=5, has_flux=False)])
    dense = render_dense([frame], 0, np.eye(3), dict(CFG, stride=4))
    assert dense["collision"].tolist() == [[0, 1], [2, 1]]
    assert dense["retained"].tolist() == [1]
    ix, ox = encode_coordinate(5.1, 4)
    iy, oy = encode_coordinate(5., 4)
    assert dense["heatmap"][2, iy, ix] == 1
    assert dense["ignore"][2, iy, ix] == 0
    assert dense["ignore"][0, iy, ix] == 1
    assert dense["offset"][2, 0, iy, ix] == pytest.approx(ox)


@pytest.mark.parametrize("stride", [1, 4])
def test_grid_boundaries_partial_blocks_and_tilted_valid(stride):
    for x in (-100.5, -.5, -.499, .5, 1.5, 9.499, 9.5):
        cell, offset = encode_coordinate(x, stride)
        assert -.5 <= offset < .5
        assert decode_coordinate(cell, offset, stride) == pytest.approx(x)
    frame = sparse([candidate(0, x=9.499, y=6.499)], width=10, height=7)
    native = render_dense([frame], 0, np.eye(3), dict(CFG, stride=stride))
    assert native["heatmap"].shape == (3, math.ceil(7/stride), math.ceil(10/stride))
    H = np.array([[1., 0., .4], [.2, 1., .4], [0., 0., 1.]])
    frame = sparse([candidate(0, x=-.49, y=-.49)], width=10, height=7, H=H)
    cfg = dict(CFG, stride=stride, label_frame="reference")
    dense = render_dense([frame], 0, H, cfg)
    for cls, row, col in np.argwhere(dense["reg_mask"]):
        assert dense["valid"][row, col]
    origin, size = canvas_geometry([frame], cfg)
    fine = valid_mask(footprint(frame, H), origin, size, 1)
    expected = np.zeros_like(dense["valid"])
    for r in range(expected.shape[0]):
        for c in range(expected.shape[1]):
            expected[r, c] = fine[r*stride:(r+1)*stride, c*stride:(c+1)*stride].any()
    assert np.array_equal(expected, dense["valid"])


def test_exact_native_edges_and_reference_pixel_center_outside_footprint():
    labels, *_ = scene([-.5, 9.5], y=[5., 5.], side=10, raw=[0, 1])
    assert labels.candidates[0]["status_default"] == "labeled"
    assert labels.candidates[1]["ignore_reason_default"] == "center_outside"
    H = np.array([[1., .5, .1], [0., 1., .1], [0., 0., 1.]])
    frame = sparse([candidate(0, x=-.49, y=.49)], width=10, height=7, H=H)
    dense = render_dense([frame], 0, H, dict(CFG, label_frame="reference"))
    cls, row, col = np.argwhere(dense["reg_mask"])[0]
    target_center = np.array([col+dense["origin"][0], row+dense["origin"][1]])
    native_center = apply_homography(np.linalg.inv(H), *target_center)
    assert native_center[0] < -.5
    assert dense["valid"][row, col] == 1


def test_group_signal_and_variance_use_group_at_its_own_peak():
    labels, frame, sensor, *_ = scene([12., 13.], flux=[1300., 900.], raw=[0, 0])
    c = labels.candidates[0]
    r, col = c["group_peak_row"], c["group_peak_col"]
    expected = frame.stars.image[r, col]*sensor.vignetting[r, col]*sensor.prnu[r, col]
    assert c["s_group"] == expected
    variance = frame.exposure.expectation_e[r, col]-expected+sensor.dark_e[r, col]+sensor.read_e**2
    assert c["sigma_local"]**2 == pytest.approx(variance)
    assert c["snr"] == expected/c["sigma_local"]


def test_sparse_pickle_free_roundtrip_and_versions(tmp_path):
    frame, *_ = scene([12., 13., 80.])
    save_objects(tmp_path/"objects.npz", frame)
    loaded = load_objects(tmp_path/"objects.npz")
    assert loaded.sources == frame.sources
    assert loaded.candidates == frame.candidates
    from starsim import algorithm_versions
    versions = algorithm_versions()
    validate_versions(versions)
    for bad in (dict(versions, labels=True), dict(versions, unexpected=1),
                {k: v for k, v in versions.items() if k != "labels"}):
        with pytest.raises(ValueError):
            validate_versions(bad)


VERIFY = ROOT/"scripts/verify_sequence.py"


def _verifier():
    # In-process copy for negative cases only; isolation is checked in the child run.
    spec = importlib.util.spec_from_file_location("verify_sequence_under_test", VERIFY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def disk_sequence(tmp_path_factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = load_config()
    from tests.freeze_m3_baseline import small_derived
    from starsim.camera import Camera
    from starsim.noise import build_sensor

    rngs = SequenceRng(sequence_seed(root_seed(20261031), 0))
    effective = sample_sequence_params(loaded, rngs)
    effective["dataset"]["n_frames"] = 4
    effective["targets"]["min_visible_frames"] = 2
    effective["labels"].update(stride=4, motion=True, label_frame="reference")
    derived = small_derived(loaded.derived)
    camera = Camera.from_derived(derived)
    motion = simulate_sequence(effective, rngs, derived)
    sensor = build_sensor(effective, camera, rngs)
    psf = PsfModel.from_effective(effective["psf"], camera)
    photo = anchor_photometry(effective["photometry"]["mag_limit"],
        effective["photometry"]["snr_at_limit"], effective["photometry"]["render_mag_margin"],
        sensor.reference_sigma(), peak_energy_fraction(psf))
    catalog = StarCatalog.from_cache(default_cache_path())
    plan = prepare_sequence(catalog, camera, motion, effective, rngs, sensor, psf, photo)
    directory = tmp_path_factory.mktemp("m4_sequence")
    objects = write_sequence(directory, plan, rngs, sensor, psf, camera, motion, photo,
                             catalog, loaded, effective, derived_camera=derived)
    return SimpleNamespace(directory=directory, objects=objects, camera=camera, derived=derived,
        writer_args=(plan, rngs, sensor, psf, camera, motion, photo, catalog, loaded, effective))


@pytest.fixture(scope="module")
def verifier():
    module = _verifier()
    return module, module.load_catalog(default_cache_path())


def test_complete_disk_sequence_independent_verifier(disk_sequence, tmp_path):
    directory = tmp_path/"seq"
    shutil.copytree(disk_sequence.directory, directory)
    seq = load_sequence(directory/"sequence.json")
    assert seq["environment"] == runtime_environment()
    assert len(disk_sequence.objects) == 4
    shutil.copy(default_cache_path(), directory/"catalog.npz")
    # Fresh child, cwd contains disk files only. The script has no repo path injection.
    child = subprocess.run([sys.executable, str(VERIFY), str(directory), "--catalog", str(directory/"catalog.npz")],
                           cwd=directory, capture_output=True, text=True, check=True)
    report = json.loads(child.stdout)
    # The verifier accepts empty fields; acceptance requires real coverage.
    assert report["modules_clean"] and report["checked"] > 0 and report["star_sources"] > 0
    assert report["no_flux"] > 0 and all(n > 0 for n in report["labeled"])
    from starsim import algorithm_versions
    assert report["algorithm_versions"] == algorithm_versions()
    assert report["environment"] == runtime_environment()
    report["stride4_collisions"] = seq["stride4_collisions"]
    report["root_seed"] = 20261031
    report["sequence_index"] = 0
    report["size"] = [disk_sequence.camera.width, disk_sequence.camera.height]
    report["negative_checks"] = sorted(DAMAGES)+["unsupported_model"]
    report["source_records"] = sum(len(f.sources) for f in disk_sequence.objects)
    report["candidate_records"] = sum(len(f.candidates) for f in disk_sequence.objects)
    report["no_flux_records"] = sum(not s["has_flux"] for f in disk_sequence.objects for s in f.sources)
    write_json(tmp_path/"m4_report.json", report)


def test_verifier_north_angle_is_plus_pi_at_zero_roll(verifier):
    from starsim.attitude import north_position_angle

    module, _ = verifier
    for ra, dec in ((0., 0.), (1.2, -.4), (5.9, 1.3)):
        R = pointing_roll_to_matrix(ra, dec, 0.)
        assert north_position_angle(R) == math.pi
        assert module.north_angle(R, ra, dec) == math.pi


def _edit_json(path, change):
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    # allow_nan: the damage itself may be a non-finite JSON literal.
    path.write_text(json.dumps(value), encoding="utf-8")


def _edit_objects(path, change):
    frame = load_objects(path)
    change(frame)
    # Writer intentionally refuses non-finite records; corrupt raw columns instead.
    if any(not math.isfinite(r.get(key, 0.)) for rows, key in
           ((frame.candidates, "x"), (frame.candidates, "y_ref"), (frame.sources, "flux_e")) for r in rows):
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        for prefix, rows in (("sources_", frame.sources), ("", frame.candidates)):
            for cls, name in enumerate(("stars", "points", "streaks")):
                records = [r for r in rows if r["class_id"] == cls]
                for i, record in enumerate(records):
                    for key, value in record.items():
                        if isinstance(value, float) and not math.isfinite(value):
                            arrays[prefix+name][key][i] = value
        np.savez_compressed(path, **arrays)
    else:
        from starsim.io import TABLE_NAMES, _table
        tables = {name: _table([r for r in rows if r["class_id"] == cls])
                  for prefix, rows in (("sources_", frame.sources), ("", frame.candidates))
                  for cls, name in enumerate((prefix+"stars", prefix+"points", prefix+"streaks"))}
        np.savez_compressed(path, **tables, schema_version=np.array(2),
                            frame_index=np.array(frame.frame_index), size=np.array([frame.width, frame.height]),
                            H_to_ref=frame.H_to_ref)


def _labeled_star(frame):
    return next(c for c in frame.candidates if c["class_id"] == 0 and c["status_default"] == "labeled")


def _set(key, value):
    return lambda record: record.__setitem__(key, value)


def _add(key, delta):
    return lambda record: record.__setitem__(key, record[key]+delta)


def _shift_h(H):
    H[0][2] += 5.


def _shift_both_h(d):
    _edit_json(d/"meta_000.json", lambda m: _shift_h(m["H_to_ref"]))
    _edit_json(d/"sequence.json", lambda s: _shift_h(s["motion"]["H_to_ref"][0]))


def _shift_both_quaternion(d):
    _edit_json(d/"meta_000.json", lambda m: _index("quaternion", 0, .1)(m))
    _edit_json(d/"sequence.json", lambda s: _index(0, 0, .1)(s["motion"]["quaternion"]))


def _move_origin(s):
    s["canvas"]["origin"][0] -= 1
    s["canvas"]["translation"][0] += 1


def _duplicate_member(frame):
    stars = [c for c in frame.candidates if c["class_id"] == 0]
    stars[1]["members"].append(stars[0]["members"][0])


def _no_flux_relabelled(frame):
    next(c for c in frame.candidates if c["ignore_reason_default"] == "no_flux")["ignore_reason_default"] = "center_outside"


def _meta(change):
    return lambda d: _edit_json(d/"meta_000.json", change)


def _objects(change):
    return lambda d: _edit_objects(d/"objects_000.npz", change)


def _star(change):
    return _objects(lambda f: change(_labeled_star(f)))


def _star_source(change):
    return _objects(lambda f: change(f.sources[_labeled_star(f)["members"][0]]))


def _index(key, i, delta=None, value=None):
    def change(m):
        m[key][i] = m[key][i]+delta if value is None else value
    return change


# name -> (damage(directory), expected message). Frame 0 is not the reference frame.
DAMAGES = {
    "quaternion_meta_only": (_meta(_index("quaternion", 0, .1)), "differs from meta"),
    "quaternion": (_shift_both_quaternion, "quaternion/matrix"),
    "WCS": (_meta(lambda m: _add("CRPIX1", 1)(m["WCS"])), "WCS"),
    "pointing_roll": (_meta(_index("pointing_roll_rad", 0, .1)), "pointing/roll"),
    "euler313": (_meta(_index("euler313_rad", 0, .1)), "Euler"),
    "north_plus_pi": (_meta(lambda m: m.__setitem__(
        "north_angle_rad", math.remainder(m["north_angle_rad"]+math.pi, 2*math.pi))), "north position angle"),
    "north_minus_pi": (_meta(_set("north_angle_rad", -math.pi)), r"outside \(-pi, pi\]"),
    "meta_H_only": (_meta(lambda m: _shift_h(m["H_to_ref"])), "differs from meta"),
    "motion_H_only": (lambda d: _edit_json(d/"sequence.json", lambda s: _shift_h(s["motion"]["H_to_ref"][0])),
                      "differs from meta"),
    "meta_and_motion_H": (_shift_both_h, "meta_000 H_to_ref mismatch"),
    "objects_H": (_objects(lambda f: f.H_to_ref.__setitem__((0, 2), f.H_to_ref[0, 2]+5)), "objects_000 H_to_ref mismatch"),
    "canvas_H": (lambda d: _edit_json(d/"sequence.json", lambda s: _shift_h(s["canvas"]["H_to_canvas"][0])),
                 "H_to_canvas"),
    "canvas_origin": (lambda d: _edit_json(d/"sequence.json", _move_origin), "tightly cover"),
    "candidate_x": (_star(_add("x", .1)), "catalog reprojection"),
    "candidate_x_ref": (_star(_add("x_ref", .01)), "candidate reference coordinate"),
    "source_x_ref": (_star_source(_add("x_ref", .01)), "reference reprojection"),
    "nan_candidate_x": (_star(_set("x", math.nan)), "non-finite"),
    "inf_candidate_y_ref": (_star(_set("y_ref", math.inf)), "non-finite"),
    "inf_source_flux": (_star_source(_set("flux_e", math.inf)), "non-finite"),
    "nan_quaternion": (_meta(_index("quaternion", 1, value=math.nan)), "non-finite"),
    "nan_north": (_meta(_set("north_angle_rad", math.nan)), "non-finite"),
    "nan_WCS": (_meta(lambda m: _set("CRVAL1", math.nan)(m["WCS"])), "non-finite"),
    "deleted_last_candidate": (_objects(lambda f: f.candidates.pop()), "without a candidate"),
    "deleted_first_candidate": (_objects(lambda f: f.candidates.pop(0)), "label_group_id"),
    "empty_members": (_objects(lambda f: _set("members", [])(f.candidates[-1])), "no members"),
    "duplicate_member": (_objects(_duplicate_member), "belongs to candidates"),
    "out_of_range_member": (_objects(lambda f: f.candidates[0]["members"].append(len(f.sources))), "out of range"),
    "reverse_link": (_objects(lambda f: _add("candidate_id", 1)(f.sources[0])), "points to candidate"),
    "no_flux_relabelled": (_objects(_no_flux_relabelled), "no-flux source"),
    "status_reason": (_star(_set("ignore_reason_default", "low_snr")), "status"),
}


@pytest.mark.parametrize("name", sorted(DAMAGES))
def test_verifier_rejects_damaged_disk_truth(disk_sequence, verifier, tmp_path, name):
    module, catalog = verifier
    directory = tmp_path/"seq"
    shutil.copytree(disk_sequence.directory, directory)
    damage, message = DAMAGES[name]
    damage(directory)
    with pytest.raises(module.VerificationError, match=message):
        module.verify(directory, catalog)


def test_verifier_accepts_undamaged_copy_in_process(disk_sequence, verifier, tmp_path):
    module, catalog = verifier
    directory = tmp_path/"seq"
    shutil.copytree(disk_sequence.directory, directory)
    # A load/save round trip of the truth tables alone must not trip any check.
    _objects(lambda f: None)(directory)
    assert module.verify(directory, catalog)["checked"] > 0


def test_verifier_rejects_unsupported_camera_model(disk_sequence, verifier, tmp_path):
    module, catalog = verifier
    directory = tmp_path/"seq"
    shutil.copytree(disk_sequence.directory, directory)
    _edit_json(directory/"sequence.json", _set("camera_model", "unsupported"))
    with pytest.raises(module.VerificationError, match="unsupported camera model"):
        module.verify(directory, catalog)
