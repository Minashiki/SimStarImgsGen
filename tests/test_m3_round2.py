"""M3 round 2: point evaluation, baseline, deviation cap, and dual coordinates."""

from __future__ import annotations

import copy
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from starsim.attitude import pointing_roll_to_matrix
from starsim.camera import Camera, apply_homography, homography
from starsim.catalog import StarCatalog, default_cache_path
from starsim.config import load_config, validate_generator
from starsim.photometry import anchor_photometry
from starsim.psf import PsfModel
from starsim.render import (
    StaticStarFrame,
    clipped_stamp_boxes,
    paint_static_stars,
    project_static_stars,
    segment_envelope_brightness,
    star_value_at,
)
from starsim.targets import _capped_uniform, _project_track, sample_targets
from tests.freeze_m3_baseline import FREEZE_REASON, OUT, baseline_problems
from tests.m3_checks import reference_visible_segment
from tests.runtime_env import runtime_environment

ROOT = Path(__file__).resolve().parents[1]


def _camera(side: int, f_over_p: float) -> Camera:
    cx = (side - 1) / 2.0
    matrix = np.array([[f_over_p, 0.0, cx], [0.0, f_over_p, cx], [0.0, 0.0, 1.0]])
    return Camera(side, side, f_over_p, cx, cx, matrix, np.linalg.inv(matrix))


def _generator():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = load_config()
    return copy.deepcopy(loaded.generator)


def _assert_rejected(mutator, field: str) -> None:
    gen = _generator()
    mutator(gen)
    with pytest.raises(ValueError) as caught:
        validate_generator(gen)
    text = str(caught.value)
    assert field in text
    assert "下界" in text
    assert "cap" in text


def test_deviation_floor_rejects_cross_and_along():
    _assert_rejected(
        lambda gen: gen["targets"]["track_deviation"].__setitem__("rand_cross_px", [0.5, 1.0]),
        "targets.point.rand_cross_px",
    )
    _assert_rejected(
        lambda gen: gen["targets"]["track_deviation"].__setitem__("rand_along_px", [0.5, 1.0]),
        "targets.point.rand_along_px",
    )


def test_deviation_floor_accepts_exact_cap_and_the_two_exceptions():
    exact = _generator()
    exact["targets"]["track_deviation"]["rand_cross_px"] = [0.45, 1.0]
    validate_generator(exact)

    streaks_off = _generator()
    streaks_off["targets"]["streak_fraction"] = 0.0
    streaks_off["targets"]["streak"]["speed_px_per_frame"] = [0.5, 100.0]
    streaks_off["targets"]["track_deviation"]["rand_cross_px"] = [0.2, 1.0]
    validate_generator(streaks_off)

    points_off = _generator()
    points_off["targets"]["streak_fraction"] = 1
    points_off["targets"]["point"]["speed_px_per_frame"] = [1.0, 50.0]
    points_off["targets"]["track_deviation"]["rand_cross_px"] = [0.2, 1.0]
    validate_generator(points_off)

    strict = _generator()
    strict["targets"]["track_deviation"]["strict_fraction"] = 1
    strict["targets"]["point"]["speed_px_per_frame"] = [1.0, 50.0]
    strict["targets"]["streak"]["speed_px_per_frame"] = [1.0, 50.0]
    strict["targets"]["track_deviation"]["rand_cross_px"] = [0.5, 1.0]
    strict["targets"]["track_deviation"]["rand_along_px"] = [0.5, 1.0]
    validate_generator(strict)


def test_capped_uniform_rejects_an_empty_intersection_before_drawing():
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match="does not intersect the cap"):
        _capped_uniform(rng, [0.5, 1.0], 0.45)
    assert rng.bit_generator.state == state


def test_capped_uniform_returns_the_floor_inside_tolerance_without_drawing():
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    cap = 0.15 * 3.0
    assert cap < 0.45
    got = _capped_uniform(rng, [0.45, 1.0], cap)
    assert got == 0.45
    assert rng.bit_generator.state == state


def test_sample_targets_draws_the_cap_floor(monkeypatch):
    """The config check and the real cap draw agree on the float-dust boundary."""
    from starsim.photometry import anchor_photometry
    from starsim.render import _empty_star_frame
    from starsim.rng import SequenceRng
    from tests.test_m3_targets import _camera, _effective, _psf, _sensor, _still_motion

    monkeypatch.setattr("starsim.targets._trunc_normal", lambda rng, sigma, clip: 0.0)
    gen = _generator()
    point = gen["targets"]["point"]
    deviation = gen["targets"]["track_deviation"]
    point["speed_px_per_frame"] = [3.0, 3.0]
    gen["targets"]["streak_fraction"] = 0.0
    deviation["strict_fraction"] = 0.0
    deviation["curve_amp_px"] = [0.0, 0.0]
    deviation["rand_cross_px"] = [0.45, 1.0]
    validate_generator(gen)

    n_frames = 4
    camera = _camera(256)
    effective = _effective(2)
    targets = effective["targets"]
    targets["per_frame"] = 2
    targets["streak_fraction"] = 0.0
    targets["min_visible_frames"] = 3
    targets["point"]["speed_px_per_frame"] = [3.0, 3.0]
    targets["track_deviation"]["strict_fraction"] = 0.0
    targets["track_deviation"]["curve_amp_px"] = [0.0, 0.0]
    targets["track_deviation"]["rand_cross_px"] = [0.45, 1.0]
    psf = _psf(camera, sigma0=1.0, field_growth=0.0, max_ellipticity=0.0, halo_fraction=0.0)
    sensor = _sensor(camera, n_frames=n_frames, sky_e=80.0, dark_e=1.0, read_e=2.0)
    photo = anchor_photometry(11.0, 1.5, 1.0, sensor.reference_sigma(), 0.2)
    stars = tuple(_empty_star_frame(camera, psf) for _ in range(n_frames))
    rngs = SequenceRng(np.random.SeedSequence(7))
    population = sample_targets(
        effective,
        rngs,
        camera,
        _still_motion(n_frames),
        psf,
        photo,
        sensor,
        stars,
    )
    assert population.track_id.shape[0] >= 1
    assert np.all(population.sigma_cross == 0.45)


def _frame(psf: PsfModel, x, y, flux, radius) -> StaticStarFrame:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.int64)
    boxes = clipped_stamp_boxes(x, y, radius, psf.height, psf.width)
    return StaticStarFrame(
        x=x,
        y=y,
        flux_e=flux,
        mag=np.zeros(x.shape[0], dtype=np.float64),
        radius=radius,
        box_row0=boxes[0],
        box_row1=boxes[1],
        box_col0=boxes[2],
        box_col1=boxes[3],
        tyc1=np.zeros(x.shape[0], dtype=np.int32),
        tyc2=np.zeros(x.shape[0], dtype=np.int32),
        psf=psf,
        height=psf.height,
        width=psf.width,
    )


def _covered(frame: StaticStarFrame, row: int, col: int) -> bool:
    return bool(
        np.any(
            (frame.box_row0 >= 0)
            & (row >= frame.box_row0)
            & (row <= frame.box_row1)
            & (col >= frame.box_col0)
            & (col <= frame.box_col1)
        )
    )


def test_star_value_at_matches_the_painted_image_on_hard_pixels():
    """Cores, overlaps, a 10^4 contrast, the border, and an off-axis PSF."""
    side = 48
    psf = PsfModel(
        sigma0=1.5,
        field_growth=0.3,
        max_ellipticity=0.2,
        halo_fraction=0.03,
        width=side,
        height=side,
        cx=23.5,
        cy=23.5,
    )
    frame = _frame(
        psf,
        x=[10.2, 12.7, 0.2, 47.4, 0.1, 47.2, -2.0, 24.0, 46.0, 1.0],
        y=[11.4, 13.1, 0.3, 46.8, 47.1, 0.2, 24.0, -3.0, 1.5, 46.5],
        flux=[1.0e6, 50.0, 2.0e4, 2.0e4, 2.0e4, 2.0e4, 8.0e4, 8.0e4, 3.0e4, 3.0e4],
        radius=[8, 8, 6, 6, 6, 6, 10, 10, 8, 8],
    )
    painted = paint_static_stars(frame)
    assert np.array_equal(painted.box_row0, frame.box_row0)
    assert np.array_equal(painted.box_col1, frame.box_col1)
    pixels = [
        (11, 10),
        (13, 12),
        (0, 0),
        (47, 47),
        (0, 47),
        (47, 0),
        (24, 0),
        (0, 24),
        (2, 46),
        (46, 2),
    ]
    for row, col in pixels:
        assert _covered(frame, row, col), (row, col)
        assert star_value_at(frame, row, col) == float(painted.image[row, col])
    overlap = float(painted.image[12, 11])
    assert overlap > 0.0
    assert float(frame.flux_e[0]) / float(frame.flux_e[1]) >= 1.0e4


def test_star_value_at_matches_a_real_star_field():
    path = default_cache_path()
    if not path.exists():
        pytest.skip("catalog cache is not built")
    catalog = StarCatalog.from_cache(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = load_config()
    side = 512
    camera = _camera(side, float(loaded.derived.f_over_p))
    psf = PsfModel(
        sigma0=1.2,
        field_growth=0.2,
        max_ellipticity=0.15,
        halo_fraction=0.03,
        width=side,
        height=side,
        cx=camera.cx,
        cy=camera.cy,
    )
    photo = anchor_photometry(12.0, 1.5, 1.0, 20.0, 0.2)
    attitude = pointing_roll_to_matrix(1.2, 0.4, 0.1)
    frame = project_static_stars(catalog, camera, attitude, photo, psf, 20.0)
    painted = paint_static_stars(frame)
    mask = np.zeros((side, side), dtype=bool)
    for index in range(int(frame.x.shape[0])):
        row0 = int(frame.box_row0[index])
        if row0 < 0:
            continue
        mask[
            row0 : int(frame.box_row1[index]) + 1,
            int(frame.box_col0[index]) : int(frame.box_col1[index]) + 1,
        ] = True
    rows, cols = np.nonzero(mask)
    assert rows.size >= 1000
    pick = np.random.default_rng(0).choice(rows.size, size=1000, replace=False)
    for index in pick:
        row = int(rows[index])
        col = int(cols[index])
        assert star_value_at(frame, row, col) == float(painted.image[row, col])
    assert float(np.max(painted.image[rows[pick], cols[pick]])) > 0.0


def test_star_value_at_rejects_a_painted_source_image():
    from starsim.render import _empty_sources

    sources = _empty_sources(_camera(16, 1000.0))
    with pytest.raises(TypeError, match="StaticStarFrame"):
        star_value_at(sources, 0, 0)


def test_envelope_bounds_psf_values_on_and_off_the_axis():
    """B(r) is an upper bound, including a segment whose near point is interior."""
    side = 180
    psf = PsfModel(
        sigma0=1.5,
        field_growth=0.3,
        max_ellipticity=0.2,
        halo_fraction=0.03,
        width=side,
        height=side,
        cx=89.5,
        cy=89.5,
    )
    segments = (
        (40.0, 89.5, 140.0, 89.5),
        (2.0, 4.0, 30.0, 18.0),
        (8.0, 170.0, 40.0, 150.0),
    )
    rng = np.random.default_rng(4)
    for index, (x1, y1, x2, y2) in enumerate(segments):
        dx, dy = x2 - x1, y2 - y1
        length2 = dx * dx + dy * dy
        fraction = ((psf.cx - x1) * dx + (psf.cy - y1) * dy) / length2
        if index == 0:
            assert 0.0 < fraction < 1.0
        for _draw in range(80):
            along = float(rng.random())
            source_x = x1 + along * dx
            source_y = y1 + along * dy
            distance = float(rng.uniform(0.0, 25.0))
            theta = float(rng.uniform(0.0, 2.0 * math.pi))
            sample_x = source_x + distance * math.cos(theta)
            sample_y = source_y + distance * math.sin(theta)
            actual = float(psf.evaluate(sample_x, sample_y, source_x, source_y))
            bound = segment_envelope_brightness(x1, y1, x2, y2, psf, distance)
            assert actual <= bound * (1.0 + 1e-9) + 1e-15


def test_exposure_radius_pixel_matrix_stays_inside_the_noise_bound():
    """Every design in the D48 matrix is within 0.055 σ_ref of the untruncated streak."""
    from starsim.render import paint_segment, render_segment_stamp, segment_exposure_radius, segment_sample_count

    side = 280
    noise = 20.0
    limit = 0.055 * noise
    rng = np.random.default_rng(11)
    worst = 0.0
    worst_case = None
    grouped: dict[tuple, list[int]] = {}
    for sigma0 in (0.7, 1.5):
        for halo in (0.0, 0.03):
            for length in (0.5, 5.0, 20.0, 90.0):
                for snr in (2.0, 30.0, 300.0):
                    for place in ("center", "corner"):
                        for degrees in (0.0, 45.0, 90.0):
                            psf = PsfModel(
                                sigma0=sigma0,
                                field_growth=0.3,
                                max_ellipticity=0.2,
                                halo_fraction=halo,
                                width=side,
                                height=side,
                                cx=(side - 1) / 2.0,
                                cy=(side - 1) / 2.0,
                            )
                            if place == "center":
                                radial_x, radial_y = 1.0, 0.0
                                origin_x, origin_y = psf.cx, psf.cy
                            else:
                                origin_x, origin_y = 36.0, 28.0
                                radial_x = origin_x - psf.cx
                                radial_y = origin_y - psf.cy
                                scale = math.hypot(radial_x, radial_y)
                                radial_x /= scale
                                radial_y /= scale
                            theta = math.radians(degrees)
                            direction_x = radial_x * math.cos(theta) - radial_y * math.sin(theta)
                            direction_y = radial_x * math.sin(theta) + radial_y * math.cos(theta)
                            for _phase in range(3):
                                shift_x = float(rng.uniform(-0.5, 0.5))
                                shift_y = float(rng.uniform(-0.5, 0.5))
                                center_x = origin_x + shift_x
                                center_y = origin_y + shift_y
                                half = 0.5 * length
                                x1 = center_x - half * direction_x
                                y1 = center_y - half * direction_y
                                x2 = center_x + half * direction_x
                                y2 = center_y + half * direction_y
                                _stamp, _r0, _c0, _r1, _c1, _pr, _pc, peak = render_segment_stamp(
                                    side, side, x1, y1, x2, y2, 1.0, psf
                                )
                                assert peak > 0.0
                                flux = snr * noise / float(peak)
                                count = segment_sample_count(x1, y1, x2, y2)
                                radius = segment_exposure_radius(flux, x1, y1, x2, y2, count, psf, noise)
                                full = np.zeros((side, side))
                                shared = np.zeros_like(full)
                                paint_segment(full, x1, y1, x2, y2, flux, psf)
                                paint_segment(shared, x1, y1, x2, y2, flux, psf, shared_radius=int(radius))
                                diff = float(np.max(np.abs(full - shared)))
                                key = (sigma0, halo, length, snr, place)
                                grouped.setdefault(key, []).append(int(radius))
                                case = {
                                    "sigma0": sigma0,
                                    "halo": halo,
                                    "length": length,
                                    "snr": snr,
                                    "place": place,
                                    "degrees": degrees,
                                    "radius": int(radius),
                                    "diff": diff,
                                }
                                if diff > worst:
                                    worst = diff
                                    worst_case = case
                                assert diff <= limit, case
    radii = [radius for values in grouped.values() for radius in values]
    report = {
        "environment": runtime_environment(),
        "limit": limit,
        "max_diff": worst,
        "max_case": worst_case,
        "radius_min": int(min(radii)),
        "radius_max": int(max(radii)),
        "n_cases": len(radii),
        "groups": [
            {
                "sigma0": sigma0,
                "halo": halo,
                "length": length,
                "snr": snr,
                "place": place,
                "radii": values,
            }
            for (sigma0, halo, length, snr, place), values in grouped.items()
        ],
    }
    path = ROOT / "artifacts" / "m3_radius_matrix.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert worst <= limit


def test_per_sample_radius_exceeds_the_noise_bound():
    """A radius taken from one sample's flux, F/n, cuts a long streak too early."""
    from starsim.render import paint_segment, render_segment_stamp

    side = 280
    noise = 20.0
    sigma0 = 1.5
    length = 90.0
    snr = 300.0
    psf = PsfModel(
        sigma0=sigma0,
        field_growth=0.3,
        max_ellipticity=0.2,
        halo_fraction=0.03,
        width=side,
        height=side,
        cx=(side - 1) / 2.0,
        cy=(side - 1) / 2.0,
    )
    half = 0.5 * length
    x1, y1 = psf.cx - half, psf.cy
    x2, y2 = psf.cx + half, psf.cy
    _stamp, _r0, _c0, _r1, _c1, _pr, _pc, peak = render_segment_stamp(side, side, x1, y1, x2, y2, 1.0, psf)
    assert peak > 0.0
    flux = snr * noise / float(peak)
    full = np.zeros((side, side))
    cut = np.zeros_like(full)
    paint_segment(full, x1, y1, x2, y2, flux, psf)
    paint_segment(cut, x1, y1, x2, y2, flux, psf, per_sample_noise=noise)
    assert float(np.max(np.abs(full - cut))) > 0.055 * noise


def test_reference_visible_segment_crosses_two_edges():
    got = reference_visible_segment(-10.0, 8.0, 30.0, 8.0, 16, 16)
    assert got == pytest.approx((-0.5, 8.0, 15.5, 8.0))
    assert reference_visible_segment(-10.0, -10.0, -8.0, -8.0, 16, 16) is None


def test_boundary_streak_flux_tracks_vis_frac():
    """In-image flux of a narrow streak tracks vis_frac to order sigma/L."""
    from starsim.render import paint_segment

    side = 160
    sigma = 0.7
    length = 100.0
    camera = _camera(side, 3000.0)
    psf = PsfModel(
        sigma0=sigma,
        field_growth=0.0,
        max_ellipticity=0.0,
        halo_fraction=0.0,
        width=side,
        height=side,
        cx=(side - 1) / 2.0,
        cy=(side - 1) / 2.0,
    )
    mapped = _project_track(
        np.array([10.0]),
        np.array([80.0]),
        np.array([length]),
        np.array([1.0]),
        np.array([0.0]),
        np.zeros(1),
        [np.eye(3)],
        camera,
        True,
    )
    vis = float(mapped["vis_frac"][0])
    assert 0.0 < vis < 1.0
    flux = 1.0e6
    image = np.zeros((side, side))
    paint_segment(
        image,
        float(mapped["x1"][0]),
        float(mapped["y1"][0]),
        float(mapped["x2"][0]),
        float(mapped["y2"][0]),
        flux,
        psf,
    )
    fraction = float(image.sum()) / flux
    assert abs(fraction - vis) <= 5.0 * sigma / length


def test_reference_endpoints_map_onto_native_endpoints():
    side = 400
    camera = _camera(side, 3000.0)
    reference = pointing_roll_to_matrix(0.2, 0.05, 0.0)
    native = pointing_roll_to_matrix(0.22, 0.07, 0.03)
    homographies = [homography(camera, reference, reference), homography(camera, reference, native)]
    mapped = _project_track(
        np.array([180.0, 210.0]),
        np.array([160.0, 190.0]),
        np.array([40.0, 40.0]),
        np.array([1.0, 0.2]),
        np.array([0.0, 0.98]),
        np.zeros(2),
        homographies,
        camera,
        True,
    )
    gap = 0.0
    for frame in range(2):
        matrix = homographies[frame]
        for ref_x, ref_y, got_x, got_y in (
            (mapped["x1_ref"], mapped["y1_ref"], mapped["x1"], mapped["y1"]),
            (mapped["x2_ref"], mapped["y2_ref"], mapped["x2"], mapped["y2"]),
        ):
            image = apply_homography(matrix, float(ref_x[frame]), float(ref_y[frame]))
            assert float(image[0]) == float(got_x[frame])
            assert float(image[1]) == float(got_y[frame])
        native_length = math.hypot(
            float(mapped["x2"][frame] - mapped["x1"][frame]),
            float(mapped["y2"][frame] - mapped["y1"][frame]),
        )
        assert float(mapped["L_native"][frame]) == native_length
        assert float(mapped["angle_native"][frame]) == float(mapped["angle_rad"][frame])
        midpoint_x = 0.5 * (float(mapped["x1"][frame]) + float(mapped["x2"][frame]))
        midpoint_y = 0.5 * (float(mapped["y1"][frame]) + float(mapped["y2"][frame]))
        gap = max(
            gap,
            math.hypot(float(mapped["x_c"][frame]) - midpoint_x, float(mapped["y_c"][frame]) - midpoint_y),
        )
    assert math.isfinite(gap)
    assert gap > 0.0


def test_small_baseline_matches():
    assert baseline_problems("small") == []


@pytest.mark.slow
def test_default_baseline_matches():
    assert baseline_problems("default") == []


def test_static_star_images_match_the_pre_envelope_baseline():
    """Exposure stamps changed. The static-star images did not."""
    previous = json.loads((ROOT / "artifacts" / "m3_baseline" / "manifest_before_render3.json").read_text(encoding="utf-8"))
    current = json.loads(OUT.read_text(encoding="utf-8"))
    expected = {"small": [0, 1], "default": [0, 1, 2]}
    n_frames = 10
    for name, indices in expected.items():
        assert current["groups"][name]["freeze_reason"] == FREEZE_REASON
        old_seqs = previous["groups"][name]["sequences"]
        new_seqs = current["groups"][name]["sequences"]
        assert [item["index"] for item in old_seqs] == indices
        assert [item["index"] for item in new_seqs] == indices
        old_by_index = {item["index"]: item for item in old_seqs}
        new_by_index = {item["index"]: item for item in new_seqs}
        for index in indices:
            old_frames = old_by_index[index]["frames"]
            new_frames = new_by_index[index]["frames"]
            assert len(old_frames) == n_frames
            assert len(new_frames) == n_frames
            for frame in range(n_frames):
                assert old_frames[frame]["stars_image"] == new_frames[frame]["stars_image"]


def test_iter_frames_blends_static_stars_only(monkeypatch):
    """A bright target passing the pair does not enter the blend inputs."""
    from starsim.blend import blend_stars
    from starsim.render import render_segment_stamp
    from starsim.rng import SequenceRng
    from starsim.sequence import SequencePlan, iter_frames
    from starsim.targets import TargetPopulation
    from tests.test_m3_targets import _camera, _psf, _sensor

    camera = _camera(64)
    psf = _psf(camera, sigma0=1.2, field_growth=0.0, max_ellipticity=0.0, halo_fraction=0.0)
    sensor = _sensor(camera, n_frames=2, sky_e=80.0, dark_e=1.0, read_e=2.0)
    sigma = 1.2
    x = np.array([24.0, 24.0 + 1.5 * sigma])
    y = np.array([32.0, 32.0])
    flux = np.array([120.0, 90.0])
    radius = np.full(2, 16, dtype=np.int64)
    boxes = clipped_stamp_boxes(x, y, radius, camera.height, camera.width)
    star = StaticStarFrame(
        x=x,
        y=y,
        flux_e=flux,
        mag=np.zeros(2),
        radius=radius,
        box_row0=boxes[0],
        box_row1=boxes[1],
        box_col0=boxes[2],
        box_col1=boxes[3],
        tyc1=np.zeros(2, dtype=np.int32),
        tyc2=np.zeros(2, dtype=np.int32),
        psf=psf,
        height=camera.height,
        width=camera.width,
    )
    _stamp, _r0, _c0, _r1, _c1, _pr, _pc, peak = render_segment_stamp(
        camera.height, camera.width, 24.9, 32.0, 24.9, 32.0, 1.0, psf
    )
    bright = 300.0 * sensor.reference_sigma() / float(peak)
    population = _one_target(camera, bright, (20.0, 24.9), 32.0)
    plan = SequencePlan(population=population, stars=(star, star), noise_sigma=sensor.reference_sigma(), search_sigma=4.0)
    calls = []
    real = blend_stars

    def recorder(*args, **kwargs):
        calls.append(kwargs if kwargs else {})
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr("starsim.sequence.blend_stars", recorder)
    rngs = SequenceRng(np.random.SeedSequence(5))
    frames = list(iter_frames(plan, rngs, sensor, psf))
    assert len(calls) == 4
    for frame_index in range(2):
        args = calls[frame_index * 2 + 1]
        assert np.array_equal(args[0], star.x)
        assert np.array_equal(args[1], star.y)
        assert np.array_equal(args[2], star.flux_e)
        assert np.array_equal(args[4], star.radius)
        n_static = int(frames[frame_index].exposure.n_static)
        assert args[0].shape[0] == n_static
    for frame_index in range(2):
        kwargs = calls[frame_index * 2]
        n_static = int(frames[frame_index].exposure.n_static)
        assert np.array_equal(kwargs["box_row0"], star.box_row0)
        assert np.array_equal(kwargs["box_col0"], star.box_col0)
        assert np.array_equal(kwargs["box_row1"], star.box_row1)
        assert np.array_equal(kwargs["box_col1"], star.box_col1)
        assert kwargs["box_row0"].shape[0] == n_static
        assert kwargs["search_sigma"] == plan.search_sigma
    empty = _one_target(camera, bright, (20.0, 24.9), 32.0, count=0)
    empty_rng = SequenceRng(np.random.SeedSequence(5))
    bare = list(iter_frames(plan, empty_rng, sensor, psf, population=empty))
    for frame_index in range(2):
        assert np.array_equal(frames[frame_index].blend.group_id, bare[frame_index].blend.group_id)
    with_target = blend_stars(
        np.array([x[0], x[1], 24.9]),
        np.array([y[0], y[1], 32.0]),
        np.array([flux[0], flux[1], bright]),
        psf,
        np.full(3, 16),
        search_sigma=4.0,
    )
    assert max(int(group.size) for group in with_target.groups) > max(
        int(group.size) for group in frames[1].blend.groups
    )


def _one_target(camera, flux: float, xs: tuple[float, float], y: float, count: int = 1):
    from starsim.targets import TargetPopulation

    n_frames = 2
    def column(value, dtype=np.float64):
        array = np.zeros((count, n_frames), dtype=dtype)
        if count:
            array[0, :] = value
        return array

    def scalar(value, dtype=np.float64):
        return np.full(count, value, dtype=dtype)

    x = np.asarray(xs, dtype=np.float64)
    return TargetPopulation(
        track_id=np.arange(count, dtype=np.int64),
        kind=scalar(0, np.int8),
        strict=scalar(False, bool),
        p0_x=scalar(0.0),
        p0_y=scalar(0.0),
        speed=scalar(4.0),
        direction_x=scalar(1.0),
        direction_y=scalar(0.0),
        accel_along=scalar(0.0),
        curve_amp=scalar(0.0),
        curve_shape=scalar(0, np.int8),
        sigma_cross=scalar(0.0),
        sigma_along=scalar(0.0),
        delta_cross=column(0.0),
        delta_along=column(0.0),
        duty=scalar(1.0),
        t_exp_eff=scalar(1.0),
        angle_sigma_rad=scalar(0.0),
        angle_jitter_rad=column(0.0),
        tier_design=np.array(["bright"] * count),
        snr_design=scalar(300.0),
        flux_design=scalar(flux),
        jitter=column(1.0),
        flux_e=column(flux),
        mag_eq=column(0.0),
        x_ref=column(x),
        y_ref=column(y),
        x_c=column(x),
        y_c=column(y),
        x1=column(x),
        y1=column(y),
        x2=column(x),
        y2=column(y),
        x1_ref=column(x),
        y1_ref=column(y),
        x2_ref=column(x),
        y2_ref=column(y),
        L_true=column(0.0),
        L_native=column(0.0),
        angle_rad=column(0.0),
        angle_native=column(0.0),
        angle_ref=column(0.0),
        in_image=column(True, bool),
        x_vis=column(x),
        y_vis=column(y),
        vis_x1=column(x),
        vis_y1=column(y),
        vis_x2=column(x),
        vis_y2=column(y),
        L_vis=column(0.0),
        vis_frac=column(1.0),
        truncated=column(False, bool),
        entering=scalar(False, bool),
        exiting=scalar(False, bool),
        never_full=scalar(False, bool),
        attempts=1,
    )

