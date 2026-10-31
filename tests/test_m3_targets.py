"""M3 acceptance: moving targets, shared exposure, and the frame loop.

Population statistics use 200 sequences. That test is marked slow and still
runs by default. The SNR check is the expectation at the peak, the same
quantity ``expose_frame`` records, not an average of noisy ratios.
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from starsim import algorithm_versions
from starsim.blend import blend_stars
from starsim.camera import Camera
from starsim.catalog import StarCatalog, default_cache_path
from starsim.config import load_config
from starsim.motion import SequenceMotion
from starsim.noise import NOISE_VERSION, expose_frame, uniform_sensor
from starsim.photometry import anchor_photometry, local_sigma, peak_snr
from starsim.psf import MOFFAT_BETA, PsfModel
from starsim.render import (
    KIND_POINT,
    KIND_STREAK,
    RENDER_VERSION,
    _empty_sources,
    _empty_targets,
    _tail_radii_many,
    centers_in_image,
    line_equivalent_flux,
    noise_capture_radii,
    paint_segment,
    peak_energy_fraction,
    render_moving_targets,
    render_points,
    tail_capture_radius,
)
from starsim.rng import SequenceRng, root_seed, sequence_seed
from starsim.sequence import SEQUENCE_VERSION, frame_noise_name, render_sequence
from starsim.targets import (
    TARGETS_VERSION,
    classify_measured_snr,
    clip_segment_to_image,
    max_line_distance,
    sample_targets,
)
from tests.m3_checks import population_constraint_problems

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts" / "m3_report.json"
N_SEQUENCES = 200
_STEP_TOL = 1e-6
_TIERS = {
    "bright": {"snr": [10.0, 300.0], "frac": 0.50},
    "faint": {"snr": [4.0, 10.0], "frac": 0.25},
    "ultra": {"snr": [1.5, 3.0], "frac": 0.25},
}


def _pcg(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def _camera(side: int, f_over_p: float = 3000.0) -> Camera:
    cx = (side - 1) / 2.0
    matrix = np.array([[f_over_p, 0.0, cx], [0.0, f_over_p, cx], [0.0, 0.0, 1.0]])
    return Camera(side, side, f_over_p, cx, cx, matrix, np.linalg.inv(matrix))


def _psf(camera: Camera, **kwargs) -> PsfModel:
    values = dict(
        sigma0=1.15,
        field_growth=0.1,
        max_ellipticity=0.15,
        halo_fraction=0.0,
        width=camera.width,
        height=camera.height,
        cx=camera.cx,
        cy=camera.cy,
    )
    values.update(kwargs)
    return PsfModel(**values)


def _sensor(camera: Camera, **kwargs):
    values = dict(
        sky_e=200.0,
        dark_e=4.0,
        read_e=3.0,
        gain_e_per_adu=1.2,
        bias_adu=100.0,
        full_well_e=1.0e6,
        adc_bits=16,
        n_frames=6,
        blooming=False,
        cosmic_rate=0.0,
    )
    values.update(kwargs)
    sensor = uniform_sensor(camera.height, camera.width, **values)
    sensor.cx = camera.cx
    sensor.cy = camera.cy
    return sensor


def _still_motion(n_frames: int, interval_s: float = 1.0) -> SequenceMotion:
    eye = np.eye(3)
    step = n_frames - 1
    return SequenceMotion(
        quaternion=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, 1)),
        matrix=np.repeat(eye[None], n_frames, axis=0),
        ra_rad=np.zeros(n_frames),
        dec_rad=np.zeros(n_frames),
        roll_rad=np.zeros(n_frames),
        boresight_ra_rad=np.zeros(n_frames),
        boresight_dec_rad=np.zeros(n_frames),
        euler_phi_rad=np.zeros(n_frames),
        euler_theta_rad=np.zeros(n_frames),
        euler_psi_rad=np.zeros(n_frames),
        north_angle_rad=np.zeros(n_frames),
        H_to_ref=np.repeat(eye[None], n_frames, axis=0),
        cumulative_px=np.zeros(n_frames),
        center_shift_px=np.zeros((step, 2)),
        omega_cam_rad=np.zeros((step, 3)),
        step_angle_rad=np.zeros(step),
        speed_px_per_frame=np.zeros(step),
        speed_deg_per_s=np.zeros(step),
        direction_rad=np.zeros(step),
        reference_index=n_frames // 2,
        axis_cam=np.array([1.0, 0.0, 0.0]),
        beta_rad=0.0,
        speed0_px=0.0,
        accel_px_per_frame2=0.0,
        jitter_px=0.0,
        ra0_rad=0.0,
        dec0_rad=0.0,
        roll0_rad=0.0,
        frame_interval_s=interval_s,
        attempts=1,
    )


def _effective(per_frame: int, *, jitter: float = 0.1, interval_s: float = 1.0) -> dict:
    return {
        "timing": {"frame_interval_s": interval_s},
        "labels": {"blend_search_sigma": 4.0},
        "targets": {
            "per_frame": int(per_frame),
            "streak_fraction": 0.5,
            "min_visible_frames": 3,
            "point": {"speed_px_per_frame": [3.0, 50.0], "smear_px": [0.0, 1.0]},
            "streak": {
                "speed_px_per_frame": [20.0, 100.0],
                "duty": [0.2, 0.9],
                "min_length_px": 5.0,
                "angle_jitter_deg": [0.0, 2.0],
            },
            "track_deviation": {
                "strict_fraction": 0.2,
                "curve_amp_px": [0.0, 2.0],
                "rand_cross_px": [0.0, 1.0],
                "rand_along_px": [0.0, 1.0],
                "rand_rel_cap": 0.15,
                "clip_sigma": 2.5,
                "max_track_dev_px": 3.0,
            },
            "tiers": {
                "bright": {"snr": [10.0, 300.0], "frac": 0.50},
                "faint": {"snr": [4.0, 10.0], "frac": 0.25},
                "ultra": {"snr": [1.5, 3.0], "frac": 0.25},
            },
            "flux_design_noise": "trajectory_mean",
            "brightness_jitter_sigma": jitter,
        },
    }


def _paint_stars(sensor, psf, x, y, flux):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    radii = noise_capture_radii(x, y, flux, psf, sensor.reference_sigma())
    from starsim.render import RenderedSources

    peak_row, peak_col, s_self, image, box_row0, box_row1, box_col0, box_col1 = render_points(
        (sensor.height, sensor.width), x, y, flux, psf, radii
    )
    return RenderedSources(
        image=image,
        x=x,
        y=y,
        flux_e=flux,
        mag=np.zeros(x.shape[0], dtype=np.float64),
        radius=radii,
        box_row0=box_row0,
        box_row1=box_row1,
        box_col0=box_col0,
        box_col1=box_col1,
        peak_row=peak_row,
        peak_col=peak_col,
        s_self_linear=s_self,
        flux_in_image=s_self > 0.0,
        in_image=centers_in_image(x, y, sensor.width, sensor.height),
        tyc1=np.zeros(x.shape[0], dtype=np.int32),
        tyc2=np.zeros(x.shape[0], dtype=np.int32),
    )


def _merge_report(updates: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if REPORT_PATH.exists():
        current = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    current.update(updates)
    REPORT_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")


def _inside(x: float, y: float, width: int, height: int) -> bool:
    return bool(x >= -0.5 and x < width - 0.5 and y >= -0.5 and y < height - 0.5)


def _population_record(spec: tuple) -> dict:
    """One sequence: geometry, design tiers, and expectation SNR."""
    index, per_frame, side, n_frames = spec
    if per_frame is None:
        per_frame = int(np.random.Generator(np.random.PCG64(10_000 + index)).integers(30, 51))
    camera = _camera(side)
    motion = _still_motion(n_frames)
    psf = _psf(camera, halo_fraction=0.0, field_growth=0.1, max_ellipticity=0.15)
    sensor = _sensor(camera, n_frames=n_frames, sky_e=200.0, dark_e=4.0, read_e=3.0)
    photo = anchor_photometry(11.0, 1.5, 1.0, sensor.reference_sigma(), 0.2)
    stars = [_empty_sources(camera) for _ in range(n_frames)]
    effective = _effective(per_frame)
    rngs = SequenceRng(sequence_seed(root_seed(20261029), index))
    pop = sample_targets(effective, rngs, camera, motion, psf, photo, sensor, stars)
    checked = population_constraint_problems(
        pop,
        width=camera.width,
        height=camera.height,
        n_frames=n_frames,
        frame_dt=1.0,
        photometry=photo,
    )
    problems = checked["problems"]
    distances = checked["distances"]
    in_bin = 0
    gap = 0
    measured = 0
    sigma_ref = sensor.reference_sigma()
    for frame in range(n_frames):
        rendered = render_moving_targets(
            pop.track_id,
            pop.kind,
            pop.x_c[:, frame],
            pop.y_c[:, frame],
            pop.x1[:, frame],
            pop.y1[:, frame],
            pop.x2[:, frame],
            pop.y2[:, frame],
            pop.flux_e[:, frame],
            psf,
            sigma_ref,
        )
        for i in range(pop.track_id.shape[0]):
            if not bool(pop.in_image[i, frame]):
                continue
            row = int(rendered.peak_row[i])
            col = int(rendered.peak_col[i])
            if row < 0:
                continue
            signal = float(rendered.s_self_linear[i])
            expectation = float(sensor.background_e[row, col] + rendered.image[row, col])
            sigma = local_sigma(expectation, signal, float(sensor.dark_e[row, col]), sensor.read_e)
            snr = peak_snr(signal, sigma)
            measured += 1
            label = classify_measured_snr(snr, effective["targets"]["tiers"])
            if label == "gap":
                gap += 1
            elif label == pop.tier_design[i]:
                in_bin += 1
    mean_count = float(pop.in_image.sum()) / n_frames
    return {
        "index": index,
        "per_frame": int(per_frame),
        "mean_count": mean_count,
        "n": int(pop.track_id.shape[0]),
        "attempts": int(pop.attempts),
        "distances": distances,
        "problems": problems[:8],
        "problem_count": len(problems),
        "tiers": [str(name) for name in pop.tier_design],
        "in_bin": in_bin,
        "gap": gap,
        "measured": measured,
        "n_entering": int(np.count_nonzero(pop.entering)),
        "n_exiting": int(np.count_nonzero(pop.exiting)),
        "n_truncated_frames": int(np.count_nonzero(pop.truncated)),
        "n_never_full": int(np.count_nonzero(pop.never_full)),
        "issued": list(rngs.issued),
    }


def test_versions_cover_targets_and_the_frame_loop():
    assert TARGETS_VERSION == 2
    assert SEQUENCE_VERSION == 1
    assert RENDER_VERSION == 3
    assert NOISE_VERSION == 4
    found = algorithm_versions()
    assert found["targets"] == 2
    assert found["render"] == 3
    assert found["sequence"] == 1
    assert frame_noise_name(0) == "frame/0/noise"
    assert frame_noise_name(12) == "frame/12/noise"
    with pytest.raises(ValueError):
        frame_noise_name(-1)
    with pytest.raises(ValueError):
        frame_noise_name(True)


def test_measured_tier_edges():
    tiers = _TIERS
    assert classify_measured_snr(1.5, tiers) == "ultra"
    assert classify_measured_snr(2.999, tiers) == "ultra"
    assert classify_measured_snr(3.0, tiers) == "gap"
    assert classify_measured_snr(3.5, tiers) == "gap"
    assert classify_measured_snr(4.0, tiers) == "faint"
    assert classify_measured_snr(9.999, tiers) == "faint"
    assert classify_measured_snr(10.0, tiers) == "bright"
    assert classify_measured_snr(300.0, tiers) == "bright"
    assert classify_measured_snr(300.1, tiers) == "gap"


def test_line_equivalent_flux_collapses_a_long_segment():
    assert line_equivalent_flux(40.0, 1.2, 0.0) == 40.0
    length = 90.0
    sigma = 1.2
    expected = 40.0 * math.sqrt(2.0 * math.pi) * sigma / length
    assert line_equivalent_flux(40.0, sigma, length) == pytest.approx(expected)


def test_empty_targets_interface_matches_no_targets():
    camera = _camera(96)
    psf = _psf(camera, field_growth=0.0, max_ellipticity=0.0)
    sensor = _sensor(
        camera,
        n_frames=2,
        sky_e=80.0,
        dark_e=2.0,
        read_e=3.0,
        full_well_e=400.0,
        blooming=True,
    )
    sources = _paint_stars(
        sensor,
        psf,
        [20.0, 60.0, 70.0],
        [30.0, 40.0, 55.0],
        [2.0e5, 40.0, 25.0],
    )
    bare = expose_frame(sources, sensor, 0, _pcg(11), psf)
    empty = expose_frame(sources, sensor, 0, _pcg(11), psf, _empty_targets(camera.height, camera.width))
    assert bare.n_static == 3
    assert empty.n_static == 3
    assert empty.center_clipped.shape == (0,)
    assert empty.center_owner.shape == (0,)
    for left, right in zip(_exposure_bytes(bare), _exposure_bytes(empty)):
        assert left == right
    assert bare.peak_clipped[0]


def _exposure_bytes(exposure) -> tuple:
    return (
        exposure.mean_e.tobytes(),
        exposure.adu.tobytes(),
        exposure.saturation_mask.tobytes(),
        exposure.saturation_owner.tobytes(),
        exposure.peak_clipped.tobytes(),
        exposure.peak_owner.tobytes(),
        exposure.s_self.tobytes(),
        exposure.sigma_local.tobytes(),
        exposure.snr.tobytes(),
        exposure.center_clipped.tobytes(),
        exposure.center_owner.tobytes(),
    )


def test_local_noise_includes_the_other_population():
    camera = _camera(128)
    psf = _psf(camera, sigma0=1.2, field_growth=0.0, max_ellipticity=0.0, halo_fraction=0.0)
    sensor = _sensor(
        camera,
        n_frames=1,
        sky_e=150.0,
        dark_e=5.0,
        read_e=4.0,
        full_well_e=1.0e9,
    )
    angles = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
    faint_x = 70.0 + 4.0 * np.cos(angles)
    faint_y = 70.0 + 4.0 * np.sin(angles)
    stars = _paint_stars(sensor, psf, faint_x, faint_y, np.full(8, 80.0))
    bright = render_moving_targets(
        np.array([0]),
        np.array([KIND_POINT], dtype=np.int8),
        np.array([70.0]),
        np.array([70.0]),
        np.array([70.0]),
        np.array([70.0]),
        np.array([70.3]),
        np.array([70.0]),
        np.array([8.0e6]),
        psf,
        sensor.reference_sigma(),
    )
    probe = expose_frame(stars, sensor, 0, _pcg(2), psf, bright)
    alone = expose_frame(stars, sensor, 0, _pcg(2), psf)
    assert np.all(probe.sigma_local[:8] > alone.sigma_local)
    stack = np.empty((200, 8))
    for draw in range(200):
        exposure = expose_frame(stars, sensor, 0, _pcg(4000 + draw), psf, bright)
        for star in range(8):
            stack[draw, star] = exposure.electrons[stars.peak_row[star], stars.peak_col[star]]
    ratios = stack.std(axis=0, ddof=1) / probe.sigma_local[:8]
    assert abs(float(ratios.mean()) - 1.0) < 0.05
    sky_only = math.sqrt(150.0 + 5.0 + 16.0)
    assert float(probe.sigma_local[0]) > 2.0 * sky_only

    bright_star = _paint_stars(sensor, psf, [64.0], [64.0], [8.0e6])
    faint_targets = render_moving_targets(
        np.arange(8),
        np.full(8, KIND_POINT, dtype=np.int8),
        64.0 + 4.0 * np.cos(angles),
        64.0 + 4.0 * np.sin(angles),
        64.0 + 4.0 * np.cos(angles),
        64.0 + 4.0 * np.sin(angles),
        64.0 + 4.0 * np.cos(angles) + 0.2,
        64.0 + 4.0 * np.sin(angles),
        np.full(8, 80.0),
        psf,
        sensor.reference_sigma(),
    )
    wing = expose_frame(bright_star, sensor, 0, _pcg(5), psf, faint_targets)
    wing_stack = np.empty((200, 8))
    for draw in range(200):
        exposure = expose_frame(bright_star, sensor, 0, _pcg(5000 + draw), psf, faint_targets)
        for target in range(8):
            row = int(faint_targets.peak_row[target])
            col = int(faint_targets.peak_col[target])
            wing_stack[draw, target] = exposure.electrons[row, col]
    wing_ratios = wing_stack.std(axis=0, ddof=1) / wing.sigma_local[1:]
    assert abs(float(wing_ratios.mean()) - 1.0) < 0.05
    assert float(wing.sigma_local[1]) > 2.0 * sky_only


def test_saturation_owners_read_the_stamp_that_was_added():
    camera = _camera(96)
    psf = _psf(camera, sigma0=1.2, field_growth=0.0, max_ellipticity=0.0, halo_fraction=0.0)
    sensor = _sensor(
        camera,
        n_frames=1,
        sky_e=0.0,
        dark_e=0.0,
        read_e=2.0,
        full_well_e=60.0,
        blooming=False,
    )
    star = _paint_stars(sensor, psf, [28.0], [48.0], [6.0e3])
    streak = render_moving_targets(
        np.array([7]),
        np.array([KIND_STREAK], dtype=np.int8),
        np.array([48.0]),
        np.array([48.0]),
        np.array([22.0]),
        np.array([48.0]),
        np.array([78.0]),
        np.array([48.0]),
        np.array([4.0e4]),
        psf,
        sensor.reference_sigma(),
    )
    window = streak.image[
        streak.box_row0[0] : streak.box_row1[0] + 1,
        streak.box_col0[0] : streak.box_col1[0] + 1,
    ]
    assert window.tobytes() == streak.stamps[0].tobytes()
    far = np.argwhere(streak.image > sensor.full_well_e)
    assert far.size
    # A column well past the star, still on the streak.
    candidates = [(int(row), int(col)) for row, col in far if col >= 70 and star.image[row, col] == 0.0]
    assert candidates
    row, col = candidates[0]
    sensor.hot_row = np.array([row], dtype=np.int64)
    sensor.hot_col = np.array([col], dtype=np.int64)
    sensor.hot_kind = np.zeros(1, dtype=np.int64)
    sensor.hot_level_a = np.array([streak.image[row, col]])
    sensor.hot_level_b = np.array([streak.image[row, col]])
    sensor.hot_state = np.zeros((1, 1), dtype=np.uint8)
    sensor.hot_p_flip = np.zeros(1, dtype=np.float64)
    exposure = expose_frame(star, sensor, 0, _pcg(9), psf, streak)
    assert exposure.n_static == 1
    owned = {"star": 0, "target": 0, "defect": 0}
    mask = exposure.saturation_mask
    for r, c in np.argwhere(mask):
        star_value = float(star.image[r, c])
        target_value = float(streak.image[r, c])
        defect = float(sensor.hot_electron_image(0)[r, c])
        best = defect
        expect = -1
        if star_value > best:
            best = star_value
            expect = 0
        if target_value > best:
            expect = 1
        assert int(exposure.saturation_owner[r, c]) == expect
        owned[{"-1": "defect", "0": "star", "1": "target"}[str(expect)]] += 1
    assert owned["star"] > 0
    assert owned["target"] > 0
    assert owned["defect"] > 0
    assert int(exposure.saturation_owner[row, col]) == -1


def test_streak_s_self_and_center_pixel():
    camera = _camera(64)
    psf = _psf(camera, field_growth=0.0, max_ellipticity=0.0, halo_fraction=0.0)
    sensor = _sensor(camera, n_frames=1, sky_e=20.0, dark_e=1.0, read_e=1.0, full_well_e=1.0e9)
    targets = render_moving_targets(
        np.array([3, 4]),
        np.array([KIND_STREAK, KIND_STREAK], dtype=np.int8),
        np.array([-4.0, 20.3]),
        np.array([32.0, 32.7]),
        np.array([-20.0, 5.0]),
        np.array([32.0, 32.0]),
        np.array([30.0, 40.0]),
        np.array([32.0, 33.0]),
        np.array([5000.0, 4000.0]),
        psf,
        sensor.reference_sigma(),
    )
    assert int(targets.center_row[0]) == -1 and int(targets.center_col[0]) == -1
    assert int(targets.peak_row[0]) >= 0
    assert targets.s_self_linear[0] == targets.stamps[0].max()
    assert targets.s_self_linear[0] == _first_maximum(targets.stamps[0])
    assert (int(targets.center_row[1]), int(targets.center_col[1])) == (33, 20)
    exposure = expose_frame(_empty_sources(camera), sensor, 0, _pcg(3), psf, targets)
    assert exposure.n_static == 0
    assert exposure.s_self[0] == pytest.approx(targets.s_self_linear[0])
    assert exposure.center_clipped.shape == (2,)
    assert bool(exposure.center_clipped[0]) is False
    assert int(exposure.center_owner[0]) == -1
    assert int(targets.center_row[1]) == 33


def _first_maximum(stamp: np.ndarray) -> float:
    best = float(stamp[0, 0])
    for value in stamp.ravel():
        if float(value) > best:
            best = float(value)
    return best


def test_tail_radii_are_independent_of_the_exposure_envelope():
    camera = _camera(64)
    _assert_tail_radii(_psf(camera))
    _assert_tail_radii(_psf(camera, halo_fraction=0.03, sigma0=1.5))


def _assert_tail_radii(psf: PsfModel) -> None:
    xs = np.array([20.0, 80.0, 110.0, 160.0])
    ys = np.array([30.0, 40.0, 110.0, 180.0])
    got = _tail_radii_many(psf, xs, ys)
    for index in range(xs.size):
        sigma_r, sigma_t, sigma_iso, _ux, _uy = psf.axes(float(xs[index]), float(ys[index]))
        expect = tail_capture_radius(
            sigma_r,
            sigma_t,
            psf.halo_fraction,
            psf.moffat_alpha(sigma_iso),
            MOFFAT_BETA,
        )
        assert int(got[index]) == expect


def test_exposure_time_does_not_scale_electrons():
    camera = _camera(360)
    motion = _still_motion(4)
    psf = _psf(camera, halo_fraction=0.0, field_growth=0.0, max_ellipticity=0.0)
    sensor = _sensor(camera, n_frames=4)
    photo = anchor_photometry(11.0, 1.5, 1.0, sensor.reference_sigma(), peak_energy_fraction(psf))
    stars = [_empty_sources(camera) for _ in range(4)]
    pop = sample_targets(
        _effective(4, interval_s=0.2),
        SequenceRng(sequence_seed(root_seed(8), 3)),
        camera,
        motion,
        psf,
        photo,
        sensor,
        stars,
    )
    expect = pop.flux_design[:, None] * pop.jitter
    assert pop.flux_e.tobytes() == expect.tobytes()
    assert np.array_equal(pop.mag_eq, photo.zero_point - 2.5 * np.log10(pop.flux_e))
    assert np.allclose(pop.t_exp_eff, pop.duty * 0.2)
    flux = 2500.0
    short = np.zeros((160, 160))
    long = np.zeros((160, 160))
    local = _psf(_camera(160), field_growth=0.0, max_ellipticity=0.0, halo_fraction=0.0)
    paint_segment(short, 40.0, 80.0, 50.0, 80.0, flux, local)
    paint_segment(long, 30.0, 80.0, 120.0, 80.0, flux, local)
    assert short.sum() == pytest.approx(flux, rel=1e-3)
    assert long.sum() == pytest.approx(flux, rel=1e-3)
    assert abs(float(short.sum() - long.sum())) / flux < 1e-3


def test_a_bright_target_does_not_change_star_groups():
    sigma = 1.2
    camera = _camera(64)
    psf = _psf(camera, sigma0=sigma, field_growth=0.0, max_ellipticity=0.0, halo_fraction=0.0)
    x = np.array([24.0, 24.0 + 1.5 * sigma])
    y = np.array([32.0, 32.0])
    flux = np.array([120.0, 90.0])
    radii = np.full(2, 16)
    stars = blend_stars(x, y, flux, psf, radii, search_sigma=4.0)
    again = blend_stars(x, y, flux, psf, radii, search_sigma=4.0)
    assert stars.blended.any()
    assert np.array_equal(stars.group_id, again.group_id)
    assert stars.group_id.shape == (2,)
    for group in stars.groups:
        assert np.all(group < 2)
    mid = 0.5 * (x[0] + x[1])
    with_target = blend_stars(
        np.array([x[0], x[1], mid]),
        np.array([y[0], y[1], y[0]]),
        np.array([flux[0], flux[1], 1.0e6]),
        psf,
        np.full(3, 16),
        search_sigma=4.0,
    )
    assert with_target.group_id.shape == (3,)
    assert max(int(group.size) for group in with_target.groups) == 3
    assert max(int(group.size) for group in stars.groups) == 2


def test_one_sequence_obeys_track_flux_and_visibility_rules():
    record = _population_record((0, 6, 400, 5))
    assert record["problems"] == []
    assert record["per_frame"] <= record["mean_count"] < record["per_frame"] + 1
    assert record["measured"] > 0
    assert record["in_bin"] / record["measured"] >= 0.7
    assert record["issued"] == ["targets/population", "targets/brightness_jitter"]


def test_frame_streams_are_independent_and_replay(catalog):
    side = 400
    n_frames = 4
    camera = _camera(side, f_over_p=catalog_plate_scale())
    motion = _still_motion(n_frames)
    psf = _psf(camera, field_growth=0.05, max_ellipticity=0.05, halo_fraction=0.0)
    sensor = _sensor(camera, n_frames=n_frames, sky_e=120.0, dark_e=2.0, read_e=3.0, cosmic_rate=40.0)
    photo = anchor_photometry(11.0, 1.5, 1.0, sensor.reference_sigma(), 0.2)
    effective = _effective(4)
    seed = sequence_seed(root_seed(20261029), 19)

    def run(rngs, override):
        sensor.cosmic_rate = 40.0
        return render_sequence(
            catalog,
            camera,
            motion,
            effective,
            rngs,
            sensor,
            psf,
            photo,
            cosmic_rate_override=override,
        )

    first_rng = SequenceRng(seed)
    first = run(first_rng, None)
    assert first_rng.issued == (
        "targets/population",
        "targets/brightness_jitter",
        "frame/0/noise",
        "frame/1/noise",
        "frame/2/noise",
        "frame/3/noise",
    )
    second = run(SequenceRng(seed), None)
    for frame in first.frames:
        assert frame.exposure.adu.tobytes() == second.frames[frame.frame_index].exposure.adu.tobytes()
        assert frame.blend.group_id.shape == frame.stars.x.shape
        assert frame.exposure.n_static == frame.stars.x.shape[0]
    changed = run(SequenceRng(seed), {1: 0.0})
    for index in (0, 2, 3):
        assert changed.frames[index].exposure.adu.tobytes() == first.frames[index].exposure.adu.tobytes()
    assert changed.frames[1].exposure.adu.tobytes() != first.frames[1].exposure.adu.tobytes()
    assert sensor.cosmic_rate == 40.0


def catalog_plate_scale() -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = load_config()
    return float(loaded.derived.f_over_p)


@pytest.fixture(scope="module")
def catalog():
    path = default_cache_path()
    if not path.exists():
        pytest.skip("catalog cache is not built")
    return StarCatalog.from_cache(path)


@pytest.fixture(scope="module")
def camera():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = load_config()
    return Camera.from_derived(loaded.derived)


def test_frame_of_5000_stars_and_50_targets_is_under_three_seconds(camera):
    """One frame of 5000 stars and 50 targets stays under 3 s.

    Twenty-five of the targets are 90 px streaks. Their paint, timed on its
    own after the stars exist, stays within 1 s.
    """
    psf = PsfModel(
        sigma0=1.15,
        field_growth=0.1,
        max_ellipticity=0.15,
        halo_fraction=0.02,
        width=camera.width,
        height=camera.height,
        cx=camera.cx,
        cy=camera.cy,
    )
    sensor = uniform_sensor(
        camera.height,
        camera.width,
        sky_e=300.0,
        dark_e=6.0,
        read_e=5.0,
        gain_e_per_adu=1.5,
        bias_adu=200.0,
        full_well_e=90000.0,
        adc_bits=16,
        n_frames=1,
        blooming=True,
    )
    sensor.cx = camera.cx
    sensor.cy = camera.cy
    photo = anchor_photometry(11.0, 1.5, 1.0, sensor.reference_sigma(), peak_energy_fraction(psf))
    rng = _pcg(21)
    count = 5000
    x = rng.uniform(8.0, camera.width - 8.0, size=count)
    y = rng.uniform(8.0, camera.height - 8.0, size=count)
    mag = np.full(count, 10.0)
    mag[0] = 0.0
    mag[1:80] = 6.0
    flux = np.asarray(photo.flux(mag), dtype=np.float64)
    noise = sensor.reference_sigma()
    render_moving_targets(
        np.array([0]),
        np.array([KIND_STREAK], dtype=np.int8),
        np.array([200.0]),
        np.array([200.0]),
        np.array([160.0]),
        np.array([200.0]),
        np.array([250.0]),
        np.array([200.0]),
        np.array([2.0e4]),
        psf,
        noise,
    )
    warmup = _paint_stars(sensor, psf, x[:20], y[:20], flux[:20])
    expose_frame(warmup, sensor, 0, _pcg(1), psf)
    ids = np.arange(50, dtype=np.int64)
    kinds = np.array([KIND_STREAK] * 25 + [KIND_POINT] * 25, dtype=np.int8)
    y_line = np.linspace(200.0, camera.height - 200.0, 50)
    x1 = np.full(50, 400.0)
    x2 = x1 + np.where(np.arange(50) < 25, 90.0, 0.4)
    y1 = y_line.copy()
    y2 = y_line.copy()
    target_flux = np.full(50, 2.0e4)
    started = time.perf_counter()
    sources = _paint_stars(sensor, psf, x, y, flux)
    star_done = time.perf_counter()
    targets = render_moving_targets(
        ids, kinds, 0.5 * (x1 + x2), y1, x1, y1, x2, y2, target_flux, psf, noise
    )
    target_done = time.perf_counter()
    exposure = expose_frame(sources, sensor, 0, _pcg(22), psf, targets)
    blend_stars(
        sources.x,
        sources.y,
        sources.flux_e,
        psf,
        sources.radius,
        search_sigma=4.0,
        box_row0=sources.box_row0,
        box_row1=sources.box_row1,
        box_col0=sources.box_col0,
        box_col1=sources.box_col1,
    )
    finished = time.perf_counter()
    assert exposure.n_static == count
    assert targets.kind.shape == (50,)
    assert target_done - star_done <= 1.0
    assert finished - started < 3.0


@pytest.mark.slow
def test_m3_population_regression():
    """Simplified regression. Not the production acceptance."""
    workers = min(8, os.cpu_count() or 1)
    specs = [(index, None, 512, 6) for index in range(N_SEQUENCES)]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(_population_record, specs, chunksize=2))
    elapsed = time.perf_counter() - started
    problems = [item for item in records if item["problem_count"]]
    distances = np.concatenate([np.asarray(item["distances"], dtype=np.float64) for item in records])
    counts, edges = np.histogram(distances, bins=np.linspace(0.0, 3.0, 13))
    tiers = [name for item in records for name in item["tiers"]]
    total_targets = len(tiers)
    fractions = {name: tiers.count(name) / total_targets for name in ("bright", "faint", "ultra")}
    measured = sum(item["measured"] for item in records)
    in_bin = sum(item["in_bin"] for item in records)
    gap = sum(item["gap"] for item in records)
    means = np.array([item["mean_count"] for item in records])
    report = {
        "note": "简化设置：静止相机、无星、均匀传感器，不作为生产验收",
        "n_sequences": N_SEQUENCES,
        "n_targets": total_targets,
        "mean_visible_per_frame": float(means.mean()),
        "design_fractions": fractions,
        "measured_in_bin_fraction": in_bin / measured,
        "gap_fraction": gap / measured,
        "entering": int(sum(item["n_entering"] for item in records)),
        "exiting": int(sum(item["n_exiting"] for item in records)),
        "truncated_frames": int(sum(item["n_truncated_frames"] for item in records)),
        "never_full": int(sum(item["n_never_full"] for item in records)),
        "attempts": int(sum(item["attempts"] for item in records)),
        "line_distance_max": float(distances.max()),
        "line_distance_histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
        "seconds": elapsed,
        "workers": workers,
        "problem_sequences": len(problems),
    }
    path = ROOT / "artifacts" / "m3_population_regression.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert problems == []
    for item in records:
        assert item["per_frame"] <= item["mean_count"] < item["per_frame"] + 1
    assert 30.0 <= float(means.mean()) <= 50.0
    for name, expected in (("bright", 0.50), ("faint", 0.25), ("ultra", 0.25)):
        assert abs(fractions[name] - expected) <= 0.05
    assert in_bin / measured >= 0.85
    assert report["entering"] > 0
    assert report["exiting"] > 0
    assert report["truncated_frames"] > 0
    assert float(distances.max()) <= 3.0
