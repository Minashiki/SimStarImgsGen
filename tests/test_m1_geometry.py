"""M1 acceptance tests: attitude, projection, homography, WCS, and camera motion."""

from __future__ import annotations

import copy
import math
import re
import subprocess
import sys
import warnings

import numpy as np
import pytest
from scipy import stats

from starsim.attitude import (
    GIMBAL_SIN_THETA,
    boresight_radec,
    boresight_vector,
    direction_to_radec,
    euler313_to_matrix,
    icrs_direction,
    matrix_to_euler313,
    matrix_to_pointing_roll,
    matrix_to_quaternion,
    north_position_angle,
    pointing_roll_to_matrix,
    quaternion_to_matrix,
    rotate_about_axis,
    rotation_vector,
    tilt_from_pixels,
    wrap_pi,
    wrap_two_pi,
)
from starsim.camera import (
    Camera,
    apply_homography,
    homography,
    project,
    project_icrs,
    tangent_plane_cd,
    unproject,
    unproject_icrs,
    wcs_from_attitude,
    wcs_keywords,
)
from starsim.catalog import icrs_unit_vector
from starsim import algorithm_versions
from starsim.config import (
    CONFIG_SAMPLING_VERSION,
    DerivedCamera,
    equal_speed_endpoint_message,
    load_config,
    sample_sequence_params,
    sample_uniform,
)
from starsim.motion import (
    MOTION_ALGORITHM_VERSION,
    _MAX_ATTEMPTS,
    _candidate_attitudes,
    _first_rejection,
    _nominal_steps_accepted,
    galactic_latitude_rad,
    latitude_frame_index,
    simulate_sequence,
)
from starsim.rng import RNG_SCHEME_VERSION, SequenceRng, root_seed, sequence_seed

_TWO_PI = 2.0 * math.pi


@pytest.fixture(scope="module")
def loaded():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return load_config()


@pytest.fixture(scope="module")
def camera(loaded):
    return Camera.from_derived(loaded.derived)


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def _angdiff(a, b) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64) + math.pi) % _TWO_PI - math.pi


def _matrix_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _quat_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def _pointing(seed: int, dec_rad: float | None = None):
    rng = _rng(seed)
    ra = float(rng.uniform(0.0, _TWO_PI))
    if dec_rad is None:
        dec = math.asin(float(rng.uniform(-1.0, 1.0)))
    else:
        dec = dec_rad
    roll = float(rng.uniform(0.0, _TWO_PI))
    return ra, dec, roll


def _effective(loaded, index: int, **motion_overrides):
    rngs = SequenceRng(sequence_seed(root_seed(20261027), index))
    effective = sample_sequence_params(loaded, rngs)
    effective["camera_motion"].update(motion_overrides)
    return effective, rngs


def _image_corners(camera) -> np.ndarray:
    return np.array(
        [
            [-0.5, -0.5],
            [camera.width - 0.5, -0.5],
            [-0.5, camera.height - 0.5],
            [camera.width - 0.5, camera.height - 0.5],
        ],
        dtype=np.float64,
    )


def _rotation_of_similarity(src: np.ndarray, dst: np.ndarray) -> float:
    """Rotation angle of the similarity that carries row-vectors ``src`` onto ``dst``."""
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    covariance = dst_c.T @ src_c / src.shape[0]
    u, _, vt = np.linalg.svd(covariance)
    if np.linalg.det(u @ vt) < 0.0:
        u = u.copy()
        u[:, -1] *= -1.0
    rotation = u @ vt
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def test_roll_zero_puts_plus_x_on_east_and_plus_y_on_north():
    for seed, dec in ((1, 0.4), (2, -0.7), (3, 1.2), (4, -1.1)):
        ra, _, _ = _pointing(seed, dec)
        matrix = pointing_roll_to_matrix(ra, dec, 0.0)
        east = np.array([-math.sin(ra), math.cos(ra), 0.0])
        north = np.array(
            [-math.sin(dec) * math.cos(ra), -math.sin(dec) * math.sin(ra), math.cos(dec)]
        )
        np.testing.assert_allclose(matrix[0], east, atol=1e-12)
        np.testing.assert_allclose(matrix[1], north, atol=1e-12)
        np.testing.assert_allclose(matrix[2], icrs_direction(ra, dec), atol=1e-12)
        assert abs(float(np.linalg.det(matrix)) - 1.0) < 1e-12


def test_icrs_direction_matches_catalog_formula():
    ra, dec, _ = _pointing(5)
    np.testing.assert_allclose(
        icrs_direction(ra, dec),
        icrs_unit_vector(math.degrees(ra), math.degrees(dec)).reshape(3),
        atol=1e-15,
    )


def test_attitude_round_trip_away_from_the_poles():
    checked = 0
    for seed in range(40):
        ra, dec, roll = _pointing(100 + seed)
        if abs(dec) > math.radians(89.0):
            continue
        checked += 1
        matrix = pointing_roll_to_matrix(ra, dec, roll)
        quat = matrix_to_quaternion(matrix)
        restored = quaternion_to_matrix(quat)
        assert _matrix_error(restored, matrix) < 1e-12
        assert _quat_error(matrix_to_quaternion(restored), quat) < 1e-12

        phi, theta, psi = matrix_to_euler313(matrix)
        assert _matrix_error(euler313_to_matrix(phi, theta, psi), matrix) < 1e-12
        phi2, theta2, psi2 = matrix_to_euler313(euler313_to_matrix(phi, theta, psi))
        assert abs(_angdiff(phi2, phi)) < 1e-12
        assert abs(theta2 - theta) < 1e-12
        assert abs(_angdiff(psi2, psi)) < 1e-12

        ra2, dec2, roll2 = matrix_to_pointing_roll(matrix)
        assert abs(_angdiff(ra2, ra)) < 1e-12
        assert abs(dec2 - dec) < 1e-12
        assert abs(_angdiff(roll2, roll)) < 1e-12
        assert _matrix_error(pointing_roll_to_matrix(ra2, dec2, roll2), matrix) < 1e-12

        chain = pointing_roll_to_matrix(*matrix_to_pointing_roll(quaternion_to_matrix(quat)))
        assert _matrix_error(chain, matrix) < 1e-12
    assert checked >= 30


def test_gimbal_lock_round_trip_at_and_near_the_poles():
    declinations = [
        math.pi / 2.0,
        -math.pi / 2.0,
        math.pi / 2.0 - 1e-10,
        -math.pi / 2.0 + 1e-10,
        math.pi / 2.0 - 1e-6,
        -math.pi / 2.0 + 1e-6,
        math.pi / 2.0 - 1e-3,
        -math.pi / 2.0 + 1e-3,
    ]
    for dec in declinations:
        for seed in range(8):
            ra, _, roll = _pointing(200 + seed, dec)
            matrix = pointing_roll_to_matrix(ra, dec, roll)
            quat = matrix_to_quaternion(matrix)
            assert _matrix_error(quaternion_to_matrix(quat), matrix) < 1e-12
            assert _quat_error(matrix_to_quaternion(quaternion_to_matrix(quat)), quat) < 1e-12

            phi, theta, psi = matrix_to_euler313(matrix)
            rebuilt = euler313_to_matrix(phi, theta, psi)
            assert _matrix_error(rebuilt, matrix) < 1e-12
            # The canonical angles are stable: extracting them again changes nothing.
            phi2, theta2, psi2 = matrix_to_euler313(rebuilt)
            assert abs(_angdiff(phi2, phi)) < 1e-12
            assert abs(theta2 - theta) < 1e-12
            assert abs(_angdiff(psi2, psi)) < 1e-12

            ra2, dec2, roll2 = matrix_to_pointing_roll(matrix)
            assert _matrix_error(pointing_roll_to_matrix(ra2, dec2, roll2), matrix) < 1e-12
            if abs(math.cos(dec)) <= GIMBAL_SIN_THETA:
                assert psi == 0.0
                assert abs(_angdiff(ra2, 1.5 * math.pi)) < 1e-12


def test_noncanonical_pole_still_rebuilds_the_same_rotation():
    # At δ = ±90°, α and roll are not unique. The canonical extraction folds
    # them into roll with ψ = 0, and that pair rebuilds the same matrix.
    matrix = pointing_roll_to_matrix(math.radians(40.0), math.pi / 2.0, math.radians(15.0))
    ra, dec, roll = matrix_to_pointing_roll(matrix)
    assert dec == pytest.approx(math.pi / 2.0)
    assert abs(_angdiff(ra, 1.5 * math.pi)) < 1e-12
    assert _matrix_error(pointing_roll_to_matrix(ra, dec, roll), matrix) < 1e-12

    south = pointing_roll_to_matrix(math.radians(10.0), -math.pi / 2.0, math.radians(80.0))
    ra_s, dec_s, roll_s = matrix_to_pointing_roll(south)
    assert dec_s == pytest.approx(-math.pi / 2.0)
    assert _matrix_error(pointing_roll_to_matrix(ra_s, dec_s, roll_s), south) < 1e-12


def test_optical_axis_is_the_third_row_and_the_principal_point(camera):
    for seed in range(25):
        ra, dec, roll = _pointing(300 + seed)
        matrix = pointing_roll_to_matrix(ra, dec, roll)
        axis_ra, axis_dec = boresight_radec(matrix)
        np.testing.assert_allclose(icrs_direction(axis_ra, axis_dec), matrix[2], atol=1e-12)
        np.testing.assert_allclose(boresight_vector(matrix), matrix[2], atol=1e-12)
        from_center = unproject_icrs(camera, matrix, camera.cx, camera.cy)
        np.testing.assert_allclose(from_center, matrix[2], atol=1e-12)
        # Roll does not move the optical axis.
        rolled = pointing_roll_to_matrix(ra, dec, roll + 0.3)
        np.testing.assert_allclose(rolled[2], matrix[2], atol=1e-12)


def test_north_position_angle_is_pi_minus_roll():
    for seed in list(range(20)) + [900, 901]:
        dec = math.pi / 2.0 if seed == 900 else -math.pi / 2.0 if seed == 901 else None
        ra, dec, roll = _pointing(400 + seed, dec)
        matrix = pointing_roll_to_matrix(ra, dec, roll)
        # At the pole the gimbal convention folds α into roll, so the angle
        # matches the roll read back from the matrix, which is π − roll away
        # from the pole and the canonical roll at the pole.
        _, _, roll_back = matrix_to_pointing_roll(matrix)
        angle = north_position_angle(matrix)
        # Same half-open interval as wrap_pi. Comparing the values, not the
        # angle modulo 2π, so +π and −π do not count as equal.
        assert -math.pi < angle <= math.pi
        assert angle == pytest.approx(wrap_pi(math.pi - roll_back), abs=1e-12)


def test_projection_round_trip_is_below_a_micro_pixel(camera):
    rng = _rng(7)
    x = rng.uniform(-0.5, camera.width - 0.5, size=200)
    y = rng.uniform(-0.5, camera.height - 0.5, size=200)
    directions = unproject(camera, x, y)
    back = project(camera, directions)
    assert np.max(np.abs(back - np.column_stack((x, y)))) < 1e-6

    center = unproject(camera, camera.cx, camera.cy)
    np.testing.assert_allclose(center, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(project(camera, np.array([0.0, 0.0, 1.0])), [camera.cx, camera.cy])


def test_homography_matches_direct_projection_on_the_middle_frame(loaded, camera):
    effective, rngs = _effective(loaded, 11, jitter_px=0.0)
    motion = simulate_sequence(effective, rngs, loaded.derived)
    assert motion.reference_index == effective["dataset"]["n_frames"] // 2
    ref = motion.reference_index
    np.testing.assert_allclose(motion.H_to_ref[ref], np.eye(3), atol=1e-12)

    rng = _rng(8)
    x = camera.cx + rng.uniform(-1000.0, 1000.0, size=40)
    y = camera.cy + rng.uniform(-1000.0, 1000.0, size=40)
    corners = _image_corners(camera)
    x = np.concatenate((x, corners[:, 0]))
    y = np.concatenate((y, corners[:, 1]))
    icrs = unproject_icrs(camera, motion.matrix[ref], x, y)
    worst = 0.0
    corner_worst = 0.0
    for index in range(motion.matrix.shape[0]):
        pix_k = project_icrs(camera, motion.matrix[index], icrs)
        mapped = apply_homography(motion.H_to_ref[index], pix_k[:, 0], pix_k[:, 1])
        pix_ref = project_icrs(camera, motion.matrix[ref], icrs)
        error = np.abs(mapped - pix_ref)
        worst = max(worst, float(np.max(error)))
        corner_worst = max(corner_worst, float(np.max(error[-4:])))
    assert worst < 1e-4
    assert corner_worst < 1e-8


def test_astropy_wcs_matches_projection_and_is_not_mirrored(camera):
    cases = [
        (math.radians(30.0), math.radians(20.0), 0.0),
        (math.radians(359.0), math.radians(-35.0), math.radians(40.0)),
        (math.radians(10.0), math.radians(80.0), math.radians(15.0)),
        (math.radians(200.0), math.radians(89.5), math.radians(70.0)),
        (math.radians(270.0), math.pi / 2.0, math.radians(25.0)),
        (math.radians(270.0), -math.pi / 2.0, math.radians(25.0)),
        (math.radians(1.0), math.radians(5.0), math.radians(170.0)),
    ]
    rng = _rng(9)
    x = camera.cx + rng.uniform(-1800.0, 1800.0, size=30)
    y = camera.cy + rng.uniform(-1800.0, 1800.0, size=30)
    for ra, dec, roll in cases:
        matrix = pointing_roll_to_matrix(ra, dec, roll)
        assert float(np.linalg.det(matrix)) > 0.0
        cd, ra_deg, dec_deg = tangent_plane_cd(matrix, camera.f_over_p)
        # det(CD) > 0 catches a sign flip of one axis. A single-axis flip is a
        # mirror; flipping both axes is roll + π and keeps the determinant positive.
        assert float(np.linalg.det(cd)) > 0.0
        cards = wcs_keywords(camera, matrix)
        assert cards["CTYPE1"] == "RA---TAN"
        assert cards["CTYPE2"] == "DEC--TAN"
        assert cards["CRPIX1"] == pytest.approx(camera.cx + 1.0)
        assert cards["CRPIX2"] == pytest.approx(camera.cy + 1.0)
        assert cards["LONPOLE"] == 180.0
        assert cards["RADESYS"] == "ICRS"
        assert cards["CRVAL1"] == pytest.approx(ra_deg)
        assert cards["CRVAL2"] == pytest.approx(dec_deg)

        wcs = wcs_from_attitude(camera, matrix)
        directions = unproject_icrs(camera, matrix, x, y)
        ra_s, dec_s = _directions_to_radec(directions)
        predicted = wcs.wcs_world2pix(np.column_stack((ra_s, dec_s)), 0)
        assert np.max(np.abs(predicted - np.column_stack((x, y)))) < 1e-3


def _directions_to_radec(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dec = np.degrees(np.arctan2(directions[:, 2], np.hypot(directions[:, 0], directions[:, 1])))
    ra = np.degrees(np.arctan2(directions[:, 1], directions[:, 0])) % 360.0
    return ra, dec


def test_zero_jitter_slew_has_no_roll_and_the_locked_sign(loaded, camera):
    effective, rngs = _effective(
        loaded,
        3,
        jitter_px=0.0,
        accel_px_per_frame2=[0.0, 0.0],
        speed_px_per_frame=[20.0, 80.0],
    )
    motion = simulate_sequence(effective, rngs, loaded.derived)
    assert motion.axis_cam[2] == 0.0
    assert motion.jitter_px == 0.0
    np.testing.assert_allclose(motion.omega_cam_rad[:, 2], 0.0, atol=1e-12)
    # Stars at the center move along +β; the camera rate is the opposite tilt.
    assert np.max(np.abs(_angdiff(motion.direction_rad, motion.beta_rad))) < 1e-12
    step = (motion.speed0_px + motion.accel_px_per_frame2 * (np.arange(motion.center_shift_px.shape[0]) + 0.5))
    delta = step / camera.f_over_p
    expected_shift = camera.f_over_p * np.tan(delta)
    np.testing.assert_allclose(motion.speed_px_per_frame, expected_shift, rtol=0.0, atol=1e-9)
    expected_omega = -delta[:, None] * motion.axis_cam
    np.testing.assert_allclose(motion.omega_cam_rad, expected_omega, atol=1e-12)
    np.testing.assert_allclose(motion.matrix[0], pointing_roll_to_matrix(motion.ra0_rad, motion.dec0_rad, motion.roll0_rad), atol=1e-12)
    assert motion.cumulative_px[0] == 0.0
    np.testing.assert_allclose(motion.cumulative_px[-1], np.sum(motion.speed_px_per_frame))
    assert effective["camera_motion"]["speed_px_per_frame"] == pytest.approx(motion.speed0_px)
    assert effective["camera_motion"]["accel_px_per_frame2"] == pytest.approx(0.0)
    assert 20.0 <= motion.speed0_px <= 80.0


def test_speed_and_accel_are_written_back_and_the_stream_is_issued_once(loaded):
    effective, rngs = _effective(loaded, 4)
    original_speed = tuple(effective["camera_motion"]["speed_px_per_frame"])
    original_accel = tuple(effective["camera_motion"]["accel_px_per_frame2"])
    motion = simulate_sequence(effective, rngs, loaded.derived)
    written_speed = effective["camera_motion"]["speed_px_per_frame"]
    written_accel = effective["camera_motion"]["accel_px_per_frame2"]
    assert type(written_speed) is float
    assert type(written_accel) is float
    assert original_speed[0] <= written_speed <= original_speed[1]
    assert original_accel[0] <= written_accel <= original_accel[1]
    assert _nominal_steps_accepted(
        motion.speed0_px,
        motion.accel_px_per_frame2,
        motion.matrix.shape[0],
        loaded.derived.f_over_p,
        original_speed[0],
        original_speed[1],
    )
    assert "motion" in rngs.issued
    with pytest.raises(ValueError, match="motion"):
        simulate_sequence(effective, rngs, loaded.derived)
    assert motion.reference_index == motion.matrix.shape[0] // 2


def test_same_seed_replays_bit_exactly(loaded):
    def once(index: int):
        effective, rngs = _effective(loaded, index)
        motion = simulate_sequence(effective, rngs, loaded.derived)
        return motion.quaternion, motion.center_shift_px, effective["camera_motion"]["speed_px_per_frame"]

    quat_a, shift_a, speed_a = once(15)
    quat_b, shift_b, speed_b = once(15)
    np.testing.assert_array_equal(quat_a, quat_b)
    np.testing.assert_array_equal(shift_a, shift_b)
    assert speed_a == speed_b


def test_galactic_latitude_of_the_poles_and_the_optional_cut(loaded):
    north = galactic_latitude_rad(math.radians(192.85948), math.radians(27.12825))
    south = galactic_latitude_rad(math.radians(12.85948), math.radians(-27.12825))
    assert north == pytest.approx(math.pi / 2.0, abs=1e-7)
    assert south == pytest.approx(-math.pi / 2.0, abs=1e-7)

    for index in range(12):
        effective, rngs = _effective(loaded, 100 + index, jitter_px=0.0, reference_frame=0)
        effective["dataset"]["n_frames"] = 10
        effective["pointing"]["galactic_lat_min_deg"] = 50.0
        motion = simulate_sequence(effective, rngs, loaded.derived)
        # Horizon uses the configured reference (frame 0). Latitude uses N // 2.
        assert motion.reference_index == 0
        assert latitude_frame_index(motion.matrix.shape[0]) == 5
        assert motion.reference_index != latitude_frame_index(motion.matrix.shape[0])
        ra = motion.boresight_ra_rad[5]
        dec = motion.boresight_dec_rad[5]
        latitude = abs(math.degrees(galactic_latitude_rad(ra, dec)))
        assert latitude >= 50.0


def test_pointing_is_uniform_on_the_sphere(loaded):
    # Latitude floor and the split constraints are off. The sample is the
    # middle-frame optical axis, jitter included, not the pre-slew draw.
    ra = []
    sin_dec = []
    for index in range(4000):
        effective, rngs = _effective(loaded, 10_000 + index)
        effective["pointing"]["galactic_lat_min_deg"] = None
        motion = simulate_sequence(effective, rngs, loaded.derived)
        mid = latitude_frame_index(motion.matrix.shape[0])
        ra.append(motion.boresight_ra_rad[mid])
        sin_dec.append(math.sin(motion.boresight_dec_rad[mid]))
    ra_hist, _ = np.histogram(ra, bins=12, range=(0.0, _TWO_PI))
    sin_hist, _ = np.histogram(sin_dec, bins=10, range=(-1.0, 1.0))
    # cos δ weighting: isotropic pointing is flat in sin δ, and flat in α.
    assert stats.chisquare(ra_hist).pvalue > 0.01
    assert stats.chisquare(sin_hist).pvalue > 0.01


def test_one_thousand_sequences_stay_in_the_speed_box_without_roll(loaded, camera):
    offsets = np.array([-1200.0, -600.0, 0.0, 600.0, 1200.0])
    grid_x, grid_y = np.meshgrid(camera.cx + offsets, camera.cy + offsets)
    grid_x = grid_x.ravel()
    grid_y = grid_y.ravel()
    max_omega_z = 0.0
    max_rotation = 0.0
    for index in range(1000):
        effective, rngs = _effective(loaded, index)
        lo, hi = effective["camera_motion"]["speed_px_per_frame"]
        motion = simulate_sequence(effective, rngs, loaded.derived)
        assert motion.axis_cam[2] == 0.0
        assert np.all(motion.speed_px_per_frame >= lo)
        assert np.all(motion.speed_px_per_frame <= hi)
        assert _nominal_steps_accepted(
            motion.speed0_px,
            motion.accel_px_per_frame2,
            motion.matrix.shape[0],
            camera.f_over_p,
            lo,
            hi,
        )
        max_omega_z = max(max_omega_z, float(np.max(np.abs(motion.omega_cam_rad[:, 2]))))
        for frame in range(motion.matrix.shape[0] - 1):
            src = np.column_stack((grid_x, grid_y))
            icrs = unproject_icrs(camera, motion.matrix[frame], grid_x, grid_y)
            dst = project_icrs(camera, motion.matrix[frame + 1], icrs)
            max_rotation = max(max_rotation, abs(_rotation_of_similarity(src, dst)))
    assert max_omega_z < 1e-6
    assert max_rotation < 1e-5


def _matrix_gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _reason_counts(text: str) -> dict[str, int]:
    counts = {}
    for name in ("nominal_step", "projection", "speed", "horizon", "galactic_latitude"):
        match = re.search(rf"^{name}=(\d+)$", text, re.MULTILINE)
        assert match is not None, text
        counts[name] = int(match.group(1))
    return counts


def _assert_rejection(text: str, reason: str) -> None:
    assert "不是所有失败条件的完整统计" in text
    assert f"总次数 {_MAX_ATTEMPTS}" in text
    counts = _reason_counts(text)
    assert sum(counts.values()) == _MAX_ATTEMPTS
    assert counts[reason] == _MAX_ATTEMPTS
    for name, count in counts.items():
        if name != reason:
            assert count == 0


def _midpoint_uniform(_rng, spec):
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return 0.5 * (float(spec[0]) + float(spec[1]))
    return float(spec)


def _wide_derived() -> DerivedCamera:
    """Short focal length so a few pixels of slew carry a corner past the horizon."""
    width = height = 64
    focal = 8.0
    return DerivedCamera(
        width=width,
        height=height,
        pixel_size_mm=0.009,
        focal_length_mm=focal * 0.009,
        aperture_mm=1.0,
        fov_width_deg=1.0,
        fov_height_deg=1.0,
        pixel_scale_arcsec=1.0,
        half_diagonal_deg=1.0,
        f_over_p=focal,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        K=np.eye(3),
    )


def _galactic_latitude_deg(matrix: np.ndarray) -> float:
    ra, dec = boresight_radec(matrix)
    return abs(math.degrees(galactic_latitude_rad(ra, dec)))


def _attitude_at_galactic_latitude(latitude_deg: float, roll: float) -> np.ndarray:
    ra_p = math.radians(192.85948)
    dec_p = math.radians(27.12825)
    pole = icrs_direction(ra_p, dec_p)
    east = np.array([-math.sin(ra_p), math.cos(ra_p), 0.0])
    rotated = rotate_about_axis(east, math.radians(90.0 - latitude_deg)) @ pole
    ra, dec = direction_to_radec(rotated)
    return pointing_roll_to_matrix(ra, dec, roll)


def test_wrap_endpoints_stay_inside_the_documented_intervals():
    assert wrap_two_pi(0.0) == 0.0
    assert wrap_two_pi(-0.0) == 0.0
    assert wrap_two_pi(-1e-17) == 0.0
    assert wrap_two_pi(-1e-16) == 0.0
    assert wrap_two_pi(_TWO_PI) == 0.0
    assert wrap_two_pi(-_TWO_PI) == 0.0
    assert wrap_two_pi(-math.pi) == pytest.approx(math.pi)
    assert 0.0 <= wrap_two_pi(-math.pi) < _TWO_PI
    for turn in range(-4, 5):
        for offset in (0.0, 1e-12, 0.4, math.pi, _TWO_PI - 1e-8):
            wrapped = wrap_two_pi(offset + turn * _TWO_PI)
            assert 0.0 <= wrapped < _TWO_PI

    assert wrap_pi(0.0) == 0.0
    assert wrap_pi(math.pi) == math.pi
    assert wrap_pi(-math.pi) == math.pi
    assert wrap_pi(3.0 * math.pi) == math.pi
    assert wrap_pi(-3.0 * math.pi) == math.pi
    just_below = math.nextafter(math.pi, 0.0)
    assert wrap_pi(just_below) == pytest.approx(just_below)
    assert -math.pi < wrap_pi(just_below) <= math.pi
    for turn in range(-3, 4):
        for offset in (-math.pi, math.nextafter(-math.pi, 0.0), -1.0, 0.0, 1.0, just_below, math.pi):
            wrapped = wrap_pi(offset + turn * _TWO_PI)
            assert -math.pi < wrapped <= math.pi

    matrix = pointing_roll_to_matrix(0.4, -0.2, 0.0)
    assert north_position_angle(matrix) == pytest.approx(math.pi)
    assert north_position_angle(matrix) != -math.pi


def test_gimbal_threshold_both_sides_poles_and_angle_pairs():
    # The locked branch snaps a tilt of size sin θ. Just inside the current
    # threshold that error is about 9.45e-13, which is inside 1e-12, so the
    # threshold stays at 1e-12.
    alphas = (0.0, 0.4, 1.7, 2.6, 3.8, 5.5)
    rolls = (0.0, 0.25, 1.1, 2.4, 3.9, 6.0)
    just_inside = float(np.nextafter(GIMBAL_SIN_THETA, 0.0))
    specs = (
        ("pole", 0.0, True),
        ("inside", 0.5 * GIMBAL_SIN_THETA, True),
        ("just_inside", just_inside, None),
        ("threshold", GIMBAL_SIN_THETA, None),
        ("outside", 2.0 * GIMBAL_SIN_THETA, False),
    )
    worst = 0.0
    for _name, offset, locked in specs:
        for sign in (1.0, -1.0):
            dec = sign * (math.pi / 2.0 - offset)
            for ra in alphas:
                for roll in rolls:
                    matrix = pointing_roll_to_matrix(ra, dec, roll)
                    phi, theta, psi = matrix_to_euler313(matrix)
                    gap = _matrix_gap(euler313_to_matrix(phi, theta, psi), matrix)
                    worst = max(worst, gap)
                    assert gap < 1e-12
                    if locked is True:
                        assert psi == 0.0
                        if offset == 0.0:
                            assert abs(_angdiff(matrix_to_pointing_roll(matrix)[0], 1.5 * math.pi)) < 1e-12
                    if locked is False:
                        sin_theta = math.hypot(float(matrix[2, 0]), float(matrix[2, 1]))
                        assert sin_theta >= GIMBAL_SIN_THETA
    assert worst < 1e-12


def test_near_pole_outputs_come_from_the_quaternion_matrix():
    generator = np.random.default_rng(0)
    direct_gaps = []
    for dec_off in (1e-11, 1e-8, 1e-4):
        for sign in (1.0, -1.0):
            for _ in range(12):
                ra = float(generator.uniform(0.0, _TWO_PI))
                roll = float(generator.uniform(0.0, _TWO_PI))
                dec = sign * (math.pi / 2.0 - dec_off)
                base = pointing_roll_to_matrix(ra, dec, roll)
                axis = generator.normal(size=3)
                axis = axis / np.linalg.norm(axis)
                extra = rotate_about_axis(axis, float(generator.uniform(0.2, 2.5)))
                product = (base @ extra.T) @ extra
                quat = matrix_to_quaternion(product)
                rebuilt = quaternion_to_matrix(quat)
                euler_back = euler313_to_matrix(*matrix_to_euler313(rebuilt))
                pointing_back = pointing_roll_to_matrix(*matrix_to_pointing_roll(rebuilt))
                assert _matrix_gap(euler_back, rebuilt) < 1e-12
                assert _matrix_gap(pointing_back, rebuilt) < 1e-12
                assert _matrix_gap(boresight_vector(rebuilt), rebuilt[2]) < 1e-15
                north = north_position_angle(rebuilt)
                assert -math.pi < north <= math.pi
                direct = euler313_to_matrix(*matrix_to_euler313(product))
                direct_gaps.append(_matrix_gap(direct, product))
    # Cancelled products at both poles. The stable extraction rebuilds that
    # rotation to machine precision.
    assert max(direct_gaps) < 1e-14


def test_rotation_vector_round_trip_including_exact_half_turn():
    assert rotation_vector(np.eye(3)) == pytest.approx(np.zeros(3))
    diagonals = (
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    )
    for matrix in diagonals:
        vector = rotation_vector(matrix)
        angle = float(np.linalg.norm(vector))
        assert angle == pytest.approx(math.pi)
        rebuilt = rotate_about_axis(vector, angle)
        assert _matrix_gap(rebuilt, matrix) < 1e-12
        # At exactly π the opposite axis is the same rotation. Do not require a sign.
        assert _matrix_gap(rotate_about_axis(-vector, angle), matrix) < 1e-12

    axes = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.2, -0.5, 0.8]),
    )
    for axis in axes:
        axis = axis / np.linalg.norm(axis)
        for angle in (0.2, math.pi / 2.0, math.pi - 1e-4, math.pi, math.pi + 0.2):
            matrix = rotate_about_axis(axis, angle)
            vector = rotation_vector(matrix)
            magnitude = float(np.linalg.norm(vector))
            assert 0.0 <= magnitude <= math.pi + 1e-12
            rebuilt = np.eye(3) if magnitude == 0.0 else rotate_about_axis(vector, magnitude)
            assert _matrix_gap(rebuilt, matrix) < 1e-12
            if angle < math.pi:
                assert magnitude == pytest.approx(angle, abs=1e-10)
                np.testing.assert_allclose(vector / magnitude, axis, atol=1e-10)
            elif angle > math.pi:
                assert magnitude == pytest.approx(_TWO_PI - angle, abs=1e-10)

    composed = rotate_about_axis(np.array([0.3, 0.4, 0.5]), 0.7) @ rotate_about_axis(
        np.array([-0.2, 0.8, 0.1]), 1.3
    )
    vector = rotation_vector(composed)
    magnitude = float(np.linalg.norm(vector))
    assert _matrix_gap(rotate_about_axis(vector, magnitude), composed) < 1e-12

    with pytest.raises(ValueError, match="proper rotation|orthonormal"):
        rotation_vector(np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError):
        rotation_vector(np.ones((3, 3)))


def test_homography_zero_pivot_is_not_called_degenerate():
    focal = 2.0
    center = 1.5
    intrinsic = np.array([[focal, 0.0, center], [0.0, focal, center], [0.0, 0.0, 1.0]])
    inverse = np.array([[1.0 / focal, 0.0, -center / focal], [0.0, 1.0 / focal, -center / focal], [0.0, 0.0, 1.0]])
    small = Camera(4, 4, focal, center, center, intrinsic, inverse)
    z_row = np.array([1.0, -1.0, 0.0])
    z_row /= np.linalg.norm(z_row)
    x_row = np.array([1.0, 1.0, 0.0])
    x_row /= np.linalg.norm(x_row)
    y_row = np.cross(z_row, x_row)
    target = np.vstack((x_row, y_row, z_row))
    assert float(np.linalg.det(target)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="归一化方式不适用") as caught:
        homography(small, np.eye(3), target)
    text = str(caught.value)
    assert "H[2, 2]=0.0" in text
    assert "退化" not in text
    assert "degenerate" not in text.lower()


def test_homography_full_frame_corners_for_an_interval_of_n_frames(loaded, camera):
    generator = copy.deepcopy(loaded.generator)
    generator["dataset"]["n_frames"] = [8, 16]
    rngs = SequenceRng(sequence_seed(root_seed(20261027), 4242))
    effective = sample_sequence_params(generator, rngs)
    effective["camera_motion"]["jitter_px"] = 0.0
    motion = simulate_sequence(effective, rngs, loaded.derived)
    count = motion.matrix.shape[0]
    assert isinstance(effective["dataset"]["n_frames"], int)
    assert 8 <= count <= 16
    assert motion.reference_index == count // 2
    ref = motion.reference_index
    corners = _image_corners(camera)
    icrs = unproject_icrs(camera, motion.matrix[ref], corners[:, 0], corners[:, 1])
    worst = 0.0
    for index in range(count):
        pix_k = project_icrs(camera, motion.matrix[index], icrs)
        mapped = apply_homography(motion.H_to_ref[index], pix_k[:, 0], pix_k[:, 1])
        pix_ref = project_icrs(camera, motion.matrix[ref], icrs)
        worst = max(worst, float(np.max(np.abs(mapped - pix_ref))))
    assert worst < 1e-8


def test_accepted_outputs_share_the_quaternion_matrix(loaded, camera):
    effective, rngs = _effective(loaded, 33, jitter_px=0.0)
    motion = simulate_sequence(effective, rngs, loaded.derived)
    reference = motion.matrix[motion.reference_index]
    for index in range(motion.matrix.shape[0]):
        np.testing.assert_array_equal(motion.matrix[index], quaternion_to_matrix(motion.quaternion[index]))
        phi, theta, psi = matrix_to_euler313(motion.matrix[index])
        assert motion.euler_phi_rad[index] == phi
        assert motion.euler_theta_rad[index] == theta
        assert motion.euler_psi_rad[index] == psi
        ra, dec, roll = matrix_to_pointing_roll(motion.matrix[index])
        assert motion.ra_rad[index] == ra
        assert motion.dec_rad[index] == dec
        assert motion.roll_rad[index] == roll
        bore_ra, bore_dec = boresight_radec(motion.matrix[index])
        assert motion.boresight_ra_rad[index] == bore_ra
        assert motion.boresight_dec_rad[index] == bore_dec
        assert motion.north_angle_rad[index] == north_position_angle(motion.matrix[index])
        np.testing.assert_array_equal(motion.H_to_ref[index], homography(camera, motion.matrix[index], reference))
    relative = motion.matrix[1] @ motion.matrix[0].T
    np.testing.assert_allclose(motion.omega_cam_rad[0], -rotation_vector(relative), atol=0.0, rtol=0.0)


def test_horizon_uses_frame_zero_and_latitude_uses_frame_five(camera):
    assert latitude_frame_index(10) == 5
    low = _attitude_at_galactic_latitude(15.0, 0.2)
    high = _attitude_at_galactic_latitude(60.0, 0.2)
    assert _galactic_latitude_deg(low) < 40.0
    assert _galactic_latitude_deg(high) > 40.0
    # Frames 0–4 share the low-latitude attitude; frames 5–9 share the high one.
    # Adjacent boresights are at most 45° apart, so projection and both horizons pass.
    matrices = np.stack([low] * 5 + [high] * 5)
    assert _first_rejection(
        matrices, camera, ref_index=0, speed_lo=0.0, speed_hi=1.0e9, galactic_min_deg=40.0
    ) is None
    assert _first_rejection(
        matrices, camera, ref_index=0, speed_lo=0.0, speed_hi=1.0e9, galactic_min_deg=80.0
    ) == "galactic_latitude"

    base = pointing_roll_to_matrix(0.4, -0.3, 0.2)
    axis = np.array([0.0, 1.0, 0.0])
    step = math.radians(100.0) / 9.0
    slewed = []
    for index in range(10):
        slew = np.eye(3) if index == 0 else rotate_about_axis(axis, step * index)
        slewed.append(quaternion_to_matrix(matrix_to_quaternion(slew @ base)))
    slewed = np.stack(slewed)
    # 100° from frame 0 puts a corner behind frame 0. Frame 5 sits in the middle
    # of that slew, and the same corners stay in front of frame 5.
    assert _first_rejection(
        slewed, camera, ref_index=0, speed_lo=0.0, speed_hi=1.0e9, galactic_min_deg=None
    ) == "horizon"
    assert _first_rejection(
        slewed, camera, ref_index=5, speed_lo=0.0, speed_hi=1.0e9, galactic_min_deg=None
    ) is None


def test_rejection_reasons_are_counted_by_first_failure(loaded, camera, monkeypatch):
    alias = 4.0 * math.pi * camera.f_over_p
    effective, rngs = _effective(loaded, 70, jitter_px=0.0, accel_px_per_frame2=alias)
    with pytest.raises(RuntimeError) as caught:
        simulate_sequence(effective, rngs, loaded.derived)
    text = str(caught.value)
    _assert_rejection(text, "nominal_step")
    assert "ValueError" not in text

    effective, rngs = _effective(
        loaded,
        71,
        jitter_px=0.0,
        speed_px_per_frame=[50.0, 50.0 + 1e-8],
        accel_px_per_frame2=[0.0, 0.0],
    )
    effective["pointing"]["galactic_lat_min_deg"] = None
    with pytest.raises(RuntimeError) as caught:
        simulate_sequence(effective, rngs, loaded.derived)
    text = str(caught.value)
    _assert_rejection(text, "speed")
    assert "[50.0, 50.00000001]" in text or "50.0" in text

    monkeypatch.setattr("starsim.motion.sample_uniform", _midpoint_uniform)
    effective, rngs = _effective(
        loaded,
        72,
        jitter_px=0.0,
        reference_frame=0,
        speed_px_per_frame=[1.0, 4.0],
        accel_px_per_frame2=[0.0, 0.0],
    )
    effective["dataset"]["n_frames"] = 10
    effective["pointing"]["galactic_lat_min_deg"] = None
    with pytest.raises(RuntimeError) as caught:
        simulate_sequence(effective, rngs, _wide_derived())
    text = str(caught.value)
    _assert_rejection(text, "horizon")
    assert "reference_index=0" in text
    assert "latitude_frame=5" in text

    effective, rngs = _effective(
        loaded,
        73,
        jitter_px=0.0,
        speed_px_per_frame=[20.0, 40.0],
        accel_px_per_frame2=[0.0, 0.0],
    )
    effective["pointing"]["galactic_lat_min_deg"] = 90.0
    with pytest.raises(RuntimeError) as caught:
        simulate_sequence(effective, rngs, loaded.derived)
    text = str(caught.value)
    _assert_rejection(text, "galactic_latitude")
    assert "galactic_lat_min_deg=90.0" in text


def test_equal_speed_endpoints_fail_before_sampling(loaded):
    effective, rngs = _effective(loaded, 74, speed_px_per_frame=[50.0, 50.0], accel_px_per_frame2=[0.0, 0.0])
    with pytest.raises(ValueError, match="lo < hi") as caught:
        simulate_sequence(effective, rngs, loaded.derived)
    text = str(caught.value)
    assert text == equal_speed_endpoint_message(50.0, 50.0)
    assert "不适合作为拒绝采样的接受区间" in text
    assert "f·tan" not in text
    assert "概率" not in text
    assert "不可能" not in text
    assert "motion" not in rngs.issued


def test_nonfinite_motion_span_is_rejected_before_the_stream(loaded):
    for field in ("speed_px_per_frame", "accel_px_per_frame2"):
        effective, rngs = _effective(loaded, 90, **{field: [-1e308, 1e308]})
        issued = rngs.issued
        with pytest.raises(ValueError) as caught:
            simulate_sequence(effective, rngs, loaded.derived)
        text = str(caught.value)
        assert field in text
        assert "-1e+308" in text
        assert "motion" not in rngs.issued
        assert rngs.issued == issued


def test_wide_finite_accel_interval_reaches_the_rejection_diagnosis(loaded):
    effective, rngs = _effective(
        loaded,
        91,
        jitter_px=0.0,
        accel_px_per_frame2=[-1e300, 1e300],
    )
    effective["dataset"]["n_frames"] = 2
    effective["pointing"]["galactic_lat_min_deg"] = None
    with pytest.raises(RuntimeError) as caught:
        simulate_sequence(effective, rngs, loaded.derived)
    _assert_rejection(str(caught.value), "nominal_step")
    assert "motion" in rngs.issued


def test_algorithm_versions_returns_a_fresh_mapping():
    from starsim import (
        BLEND_VERSION,
        LABELS_VERSION,
        NOISE_VERSION,
        PHOTOMETRY_VERSION,
        PSF_VERSION,
        RENDER_VERSION,
        SEQUENCE_VERSION,
        TARGETS_VERSION,
        SPLIT_VERSION,
    )

    first = algorithm_versions()
    assert set(first) == {
        "config_sampling",
        "motion",
        "photometry",
        "psf",
        "render",
        "noise",
        "blend",
        "targets",
        "sequence",
        "labels",
        "split",
    }
    assert "rng_scheme" not in first
    assert first == {
        "config_sampling": CONFIG_SAMPLING_VERSION,
        "motion": MOTION_ALGORITHM_VERSION,
        "photometry": PHOTOMETRY_VERSION,
        "psf": PSF_VERSION,
        "render": RENDER_VERSION,
        "noise": NOISE_VERSION,
        "blend": BLEND_VERSION,
        "targets": TARGETS_VERSION,
        "sequence": SEQUENCE_VERSION,
        "labels": LABELS_VERSION,
        "split": SPLIT_VERSION,
    }
    assert CONFIG_SAMPLING_VERSION == 1
    assert all(type(value) is int for value in first.values())
    first["config_sampling"] = 0
    first["motion"] = 0
    first["rng_scheme"] = 1
    second = algorithm_versions()
    assert second == {
        "config_sampling": CONFIG_SAMPLING_VERSION,
        "motion": MOTION_ALGORITHM_VERSION,
        "photometry": PHOTOMETRY_VERSION,
        "psf": PSF_VERSION,
        "render": RENDER_VERSION,
        "noise": NOISE_VERSION,
        "blend": BLEND_VERSION,
        "targets": TARGETS_VERSION,
        "sequence": SEQUENCE_VERSION,
        "labels": LABELS_VERSION,
        "split": SPLIT_VERSION,
    }
    assert "rng_scheme" not in second
    assert second is not first


def test_motion_covered_float_inputs_fail_before_the_stream(loaded):
    huge = 10**400
    edits = (
        lambda effective: effective["camera_motion"].__setitem__("jitter_px", huge),
        lambda effective: effective["timing"].__setitem__("frame_interval_s", huge),
        lambda effective: effective["pointing"].__setitem__("galactic_lat_min_deg", huge),
        lambda effective: effective["camera_motion"].__setitem__("accel_px_per_frame2", huge),
        lambda effective: effective["camera_motion"].__setitem__("accel_px_per_frame2", float("nan")),
        lambda effective: effective["camera_motion"].__setitem__("accel_px_per_frame2", float("inf")),
    )
    for index, edit in enumerate(edits):
        effective, rngs = _effective(loaded, 300 + index)
        edit(effective)
        with pytest.raises(ValueError):
            simulate_sequence(effective, rngs, loaded.derived)
        assert "motion" not in rngs.issued


def test_motion_algorithm_version_stays_out_of_rng_metadata():
    assert type(MOTION_ALGORITHM_VERSION) is int
    assert MOTION_ALGORITHM_VERSION == 3
    assert RNG_SCHEME_VERSION == 1
    rngs = SequenceRng(sequence_seed(root_seed(1), 0))
    meta = rngs.metadata()
    assert set(meta) == {"rng_scheme", "entropy", "spawn_key", "pool_size", "issued"}
    assert "motion_algorithm" not in meta


def test_nominal_step_rejects_alias_reversal_and_the_quarter_turn_cap(camera):
    f_over_p = camera.f_over_p
    alias = 4.0 * math.pi * f_over_p
    first = 29.0 + 0.5 * alias
    assert first / f_over_p > _TWO_PI
    assert not _nominal_steps_accepted(29.0, alias, 2, f_over_p, 5.0, 100.0)

    # Four frames, s0 = 20.70, a = -29.5. The first step is still inside
    # [5, 100]; the later steps reverse.
    steps = [20.70 + (-29.5) * (k + 0.5) for k in range(3)]
    assert 5.0 <= steps[0] <= 100.0
    assert steps[1] < 0.0 and steps[2] < 0.0
    assert not _nominal_steps_accepted(20.70, -29.5, 4, f_over_p, 5.0, 100.0)

    # The signed interval contains the aliased step. Only the π/2 cap rejects it.
    aliased = _TWO_PI * f_over_p + 29.0
    assert 5.0 <= aliased <= aliased + 1.0
    assert not (0.0 < aliased / f_over_p < math.pi / 2.0)
    assert not _nominal_steps_accepted(aliased, 0.0, 3, f_over_p, 5.0, aliased + 1.0)

    # A non-positive step is rejected even when the caller bypasses config and
    # hands _bounds an interval that contains it.
    assert not _nominal_steps_accepted(-5.0, 0.0, 2, f_over_p, -10.0, 100.0)
    assert _nominal_steps_accepted(30.0, 0.0, 4, f_over_p, -10.0, 100.0)
    assert not _nominal_steps_accepted(float("nan"), 0.0, 2, f_over_p, 5.0, 100.0)
    assert not _nominal_steps_accepted(30.0, float("nan"), 2, f_over_p, 5.0, 100.0)
    assert not _nominal_steps_accepted(1e308, 1e308, 3, f_over_p, 5.0, 1e308)
    assert _nominal_steps_accepted(30.0, -1.0, 4, f_over_p, 5.0, 100.0)


def test_quarter_turn_cap_is_counted_as_nominal_step(loaded, camera):
    f_over_p = camera.f_over_p
    aliased = _TWO_PI * f_over_p + 29.0
    effective, rngs = _effective(
        loaded,
        81,
        jitter_px=0.0,
        speed_px_per_frame=[aliased, aliased + 1.0],
        accel_px_per_frame2=[0.0, 0.0],
    )
    with pytest.raises(RuntimeError) as caught:
        simulate_sequence(effective, rngs, loaded.derived)
    _assert_rejection(str(caught.value), "nominal_step")


def test_a_reversing_acceleration_is_counted_as_nominal_step(loaded):
    effective, rngs = _effective(
        loaded,
        82,
        jitter_px=0.0,
        speed_px_per_frame=[5.0, 100.0],
        accel_px_per_frame2=-100.0,
    )
    effective["dataset"]["n_frames"] = 4
    with pytest.raises(RuntimeError) as caught:
        simulate_sequence(effective, rngs, loaded.derived)
    _assert_rejection(str(caught.value), "nominal_step")


def test_projection_rejects_a_jittered_attitude_past_the_horizon(camera):
    base = pointing_roll_to_matrix(0.4, -0.2, 0.3)
    slew = rotate_about_axis(np.array([0.0, 1.0, 0.0]), 0.3)
    tilt = tilt_from_pixels(camera.f_over_p * math.tan(1.4), 0.0, camera.f_over_p)
    first = quaternion_to_matrix(matrix_to_quaternion(base))
    second = quaternion_to_matrix(matrix_to_quaternion(tilt @ slew @ base))
    matrices = np.stack((first, second))
    assert float((second @ first[2])[2]) <= 0.0
    assert (
        _first_rejection(
            matrices,
            camera,
            ref_index=0,
            speed_lo=0.0,
            speed_hi=1.0e9,
            galactic_min_deg=None,
        )
        == "projection"
    )


def test_nominal_rejection_still_consumes_the_full_draw(loaded, camera):
    index = 82
    speed_spec = [20.0, 40.0]
    accel_spec = [-60.0, 60.0]
    sigma = 0.15
    n_frames = 4
    seed = sequence_seed(root_seed(20261027), index)
    rngs = SequenceRng(seed)
    effective = sample_sequence_params(loaded, rngs)
    effective["camera_motion"].update(
        {
            "jitter_px": sigma,
            "speed_px_per_frame": speed_spec,
            "accel_px_per_frame2": accel_spec,
        }
    )
    effective["dataset"]["n_frames"] = n_frames
    effective["pointing"]["galactic_lat_min_deg"] = None
    motion = simulate_sequence(effective, rngs, loaded.derived)
    assert motion.attempts >= 3

    replay = SequenceRng(seed).get("motion")
    assert isinstance(replay, np.random.Generator)
    early_nominal_failures = 0
    for attempt in range(1, motion.attempts + 1):
        ra = float(replay.uniform(0.0, _TWO_PI))
        dec = math.asin(float(replay.uniform(-1.0, 1.0)))
        roll = float(replay.uniform(0.0, _TWO_PI))
        beta = float(replay.uniform(0.0, _TWO_PI))
        speed0 = sample_uniform(replay, speed_spec)
        accel = sample_uniform(replay, accel_spec)
        jitter = replay.normal(0.0, sigma, size=(n_frames, 2))
        accepted = _nominal_steps_accepted(
            speed0,
            accel,
            n_frames,
            camera.f_over_p,
            speed_spec[0],
            speed_spec[1],
        )
        if attempt < motion.attempts:
            if not accepted:
                early_nominal_failures += 1
            continue
        assert accepted
        assert ra == motion.ra0_rad
        assert dec == motion.dec0_rad
        assert roll == motion.roll0_rad
        assert beta == motion.beta_rad
        assert speed0 == motion.speed0_px
        assert accel == motion.accel_px_per_frame2
        # Jitter is not stored. Attitudes built from this draw match the
        # stored quaternions and matrices bit for bit, so the consumed
        # jitter is the same array.
        _, quaternions, matrices = _candidate_attitudes(
            camera=camera,
            n_frames=n_frames,
            ra0=ra,
            dec0=dec,
            roll0=roll,
            beta=beta,
            speed0=speed0,
            accel=accel,
            jitter=jitter,
        )
        np.testing.assert_array_equal(quaternions, motion.quaternion)
        np.testing.assert_array_equal(matrices, motion.matrix)
    assert early_nominal_failures >= 1


def test_importing_geometry_does_not_load_astropy():
    script = (
        "import sys\n"
        "import starsim.attitude\n"
        "import starsim.camera\n"
        "import starsim.motion\n"
        "raise SystemExit(1 if 'astropy' in sys.modules else 0)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
