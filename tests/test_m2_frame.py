"""M2 acceptance tests: PSF, photometry, rendering, noise, and blending.

The SNR statistic is ``mean(peak − background) / std(background)`` pooled
over realizations. Per-realization ratios are not averaged.
"""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image
from scipy import ndimage
from tests.report_sections import section_problems, write_section

from starsim.attitude import pointing_roll_to_matrix
from starsim.camera import Camera, project_icrs, wcs_keywords
from starsim.catalog import StarCatalog, default_cache_path
from starsim.config import load_config, sample_sequence_params
from starsim.blend import (
    PROFILE_SAMPLES,
    _segment_profile,
    _sources_on_segment,
    _unclipped_boxes,
    blend_stars,
    has_profile_valley,
)
from starsim.noise import (
    KIND_CLUSTER,
    KIND_RTS,
    KIND_STABLE,
    STREAM_DEAD,
    STREAM_DSNU,
    STREAM_FPN,
    STREAM_GRADIENT,
    STREAM_HOT,
    STREAM_PRNU,
    STREAM_SMOOTH_FIELD,
    STREAM_SMOOTH_SCALE,
    Exposure,
    _SMOOTH_DOWNSAMPLE,
    _smooth_coarse_canvas,
    _smooth_field_with_length,
    _source_kernel,
    adu_is_high_clipped,
    build_sensor,
    electron_to_adu,
    expose_frame,
    uniform_sensor,
)
from starsim.photometry import (
    anchor_photometry,
    effective_snr,
    local_sigma,
    peak_snr,
)
from starsim.psf import MOFFAT_BETA, PsfModel
from starsim.render import (
    add_streak,
    centers_in_image,
    clipped_stamp_boxes,
    gaussian_pixel_integral,
    integrate_pixel,
    noise_capture_radii,
    peak_energy_fraction,
    peak_fraction_at,
    render_points,
    render_static_stars,
    render_unit_stamp,
)
from starsim.rng import SequenceRng, root_seed, sequence_seed

ROOT = Path(__file__).resolve().parents[1]
PREVIEW_DIR = ROOT / "artifacts" / "m2_fields"
REPORT_PATH = ROOT / "artifacts" / "m2_report.json"

# J2000 positions. Tycho is J1991.25; a 1-arcmin match still finds the star.
ALCYONE = (56.87115, 24.10514)
ALNITAK = (85.18958, -1.94278)
ALNILAM = (84.05333, -1.20192)
MINTAKA = (83.00167, -0.29908)
DUBHE = (165.93196, 61.75083)
MERAK = (165.46042, 56.38242)
PLEIADES = (56.75, 24.12)


@pytest.fixture(scope="module")
def loaded():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return load_config()


@pytest.fixture(scope="module")
def camera(loaded):
    return Camera.from_derived(loaded.derived)


@pytest.fixture(scope="module")
def catalog():
    path = default_cache_path()
    if not path.exists():
        pytest.skip("catalog cache is not built")
    return StarCatalog.from_cache(path)


def _pcg(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def _psf(**kwargs) -> PsfModel:
    values = dict(
        sigma0=1.2,
        field_growth=0.0,
        max_ellipticity=0.0,
        halo_fraction=0.0,
        width=64,
        height=64,
        cx=31.5,
        cy=31.5,
    )
    values.update(kwargs)
    return PsfModel(**values)


def _flat_sensor(**kwargs) -> tuple:
    values = dict(
        sky_e=400.0,
        dark_e=10.0,
        read_e=5.0,
        gain_e_per_adu=1.5,
        bias_adu=200.0,
        full_well_e=1.0e7,
        adc_bits=16,
        n_frames=4,
        blooming=False,
        cosmic_rate=0.0,
    )
    height = int(kwargs.pop("height", 96))
    width = int(kwargs.pop("width", 96))
    values.update(kwargs)
    sensor = uniform_sensor(height, width, **values)
    return sensor


def _paint_stars(sensor, psf, x, y, flux, mag):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    mag = np.asarray(mag, dtype=np.float64)
    radii = noise_capture_radii(x, y, flux, psf, sensor.reference_sigma())
    from starsim.render import RenderedSources

    peak_row, peak_col, s_self, image, box_row0, box_row1, box_col0, box_col1 = render_points(
        (sensor.height, sensor.width), x, y, flux, psf, radii
    )
    sources = RenderedSources(
        image=image,
        x=x,
        y=y,
        flux_e=flux,
        mag=mag,
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
    return sources


def _noise_effective(**overrides) -> dict:
    config = {
        "dataset": {"n_frames": 30},
        "timing": {"exposure_s_nominal": 0.2},
        "background": {
            "sky_e": 400.0,
            "gradient_frac": 0.05,
            "smooth_field_frac": 0.02,
            "vignetting_corner": 0.1,
        },
        "sensor_noise": {
            "read_e": 6.0,
            "dark_e_per_s": 20.0,
            "prnu": 0.01,
            "gain_e_per_adu": 1.2,
            "bias_adu": 300.0,
            "full_well_e": 60000.0,
            "row_col_fpn_e": 0.4,
            "adc_bits": 16,
        },
        "artifacts": {
            "hot_pixel_frac": 1.5e-4,
            "hot_cluster_frac": 0.10,
            "hot_cluster_size_px": [2, 3],
            "rts_frac": 0.05,
            "rts_p_flip": [0.1, 0.3],
            "dead_pixel_frac": 2.0e-6,
            "cosmic_rays_per_frame": 4.0,
            "blooming": True,
        },
    }
    for key, value in overrides.items():
        section, leaf = key.split(".")
        config[section][leaf] = value
    return config


def test_unit_point_stamps_sum_to_one():
    for sigma in (0.7, 1.0, 1.5):
        for halo in (0.0, 0.03):
            psf = _psf(sigma0=sigma, halo_fraction=halo)
            for offset_x, offset_y in ((0.0, 0.0), (0.25, -0.4), (0.5, 0.5)):
                stamp = render_unit_stamp(psf, offset_x, offset_y)
                assert stamp.sum() == pytest.approx(1.0, abs=1e-3)


def test_streak_stamps_conserve_flux_across_angles_and_lengths():
    psf = _psf(sigma0=1.0, halo_fraction=0.02, width=512, height=512, cx=255.5, cy=255.5)
    for angle_deg in (0.0, 20.0, 45.0, 90.0, 130.0, 180.0):
        angle = math.radians(angle_deg)
        for length in (5.0, 20.0, 45.0, 90.0):
            image = np.zeros((512, 512), dtype=np.float64)
            x1, y1 = 255.3, 250.7
            add_streak(
                image,
                x1,
                y1,
                x1 + length * math.cos(angle),
                y1 + length * math.sin(angle),
                1.0,
                psf,
            )
            assert image.sum() == pytest.approx(1.0, abs=1e-3)


def test_supersampled_core_matches_the_erf_integral():
    sigma = 1.1
    offset_x, offset_y = 0.2, -0.15
    psf = _psf(sigma0=sigma, halo_fraction=0.0)
    stamp = render_unit_stamp(psf, offset_x, offset_y)
    center = (stamp.shape[0] - 1) / 2.0
    worst = 0.0
    for row in range(stamp.shape[0]):
        for col in range(stamp.shape[1]):
            reference = gaussian_pixel_integral(center + offset_x, center + offset_y, sigma, row, col)
            worst = max(worst, abs(float(stamp[row, col]) - reference))
    # 5× supersampling tracks the analytic pixel integral. The residual is
    # the quadrature error, which stays well under the 1e-3 flux budget.
    assert worst < 5e-4


def test_vector_psf_matches_the_scalar_evaluation():
    psf = _psf(
        sigma0=1.1,
        field_growth=0.2,
        max_ellipticity=0.2,
        halo_fraction=0.02,
        width=80,
        height=60,
        cx=30.0,
        cy=20.0,
    )
    source_x = np.array([10.0, 30.0, 55.5])
    source_y = np.array([8.0, 20.0, 40.2])
    sample_x = np.linspace(0.0, 20.0, 7)
    sample_y = np.linspace(5.0, 18.0, 7)
    batch = psf.evaluate_at_sources(sample_x, sample_y, source_x, source_y)
    for index, (x0, y0) in enumerate(zip(source_x, source_y)):
        scalar = psf.evaluate(sample_x, sample_y, float(x0), float(y0))
        assert batch[index] == pytest.approx(scalar, rel=1e-12, abs=1e-12)


def test_stamp_box_covers_painted_pixels_and_a_miss_is_empty():
    psf = _psf(sigma0=1.0, halo_fraction=0.01)
    sensor = _flat_sensor(height=48, width=48)
    sources = _paint_stars(
        sensor,
        psf,
        np.array([10.2, -30.0]),
        np.array([12.4, 10.0]),
        np.array([500.0, 500.0]),
        np.array([8.0, 8.0]),
    )
    rows, cols = np.nonzero(sources.image > 0.0)
    assert rows.min() >= sources.box_row0[0]
    assert rows.max() <= sources.box_row1[0]
    assert cols.min() >= sources.box_col0[0]
    assert cols.max() <= sources.box_col1[0]
    assert sources.box_row0[0] <= sources.peak_row[0] <= sources.box_row1[0]
    assert sources.box_col0[0] <= sources.peak_col[0] <= sources.box_col1[0]
    assert sources.flux_in_image[0]
    assert sources.box_row0[1] == -1
    assert sources.box_col0[1] == -1
    assert sources.peak_row[1] == -1
    assert not sources.flux_in_image[1]


def test_sigma_along_follows_the_radial_stretch():
    psf = _psf(sigma0=1.0, field_growth=0.2, max_ellipticity=0.2, width=201, height=201, cx=100.0, cy=100.0)
    assert psf.sigma_along(100.0, 100.0, 0.3) == pytest.approx(1.0)
    sigma_r, sigma_t, sigma_iso, radial_x, radial_y = psf.axes(180.0, 100.0)
    assert sigma_r * sigma_t == pytest.approx(sigma_iso ** 2)
    assert sigma_r > sigma_t
    assert psf.sigma_along(180.0, 100.0, (radial_x, radial_y)) == pytest.approx(sigma_r)
    tangent = (-radial_y, radial_x)
    assert psf.sigma_along(180.0, 100.0, tangent) == pytest.approx(sigma_t)
    diagonal = psf.sigma_along(180.0, 100.0, math.pi / 4.0)
    cos_phi = math.cos(math.pi / 4.0) * radial_x + math.sin(math.pi / 4.0) * radial_y
    expected = math.sqrt(sigma_r ** 2 * cos_phi ** 2 + sigma_t ** 2 * (1.0 - cos_phi ** 2))
    assert diagonal == pytest.approx(expected)


def test_anchor_makes_the_limit_star_snr_1_5():
    psf = _psf(sigma0=1.2, halo_fraction=0.0)
    sensor = _flat_sensor(height=64, width=64, sky_e=250.0, dark_e=8.0, read_e=4.0)
    sigma_ref = sensor.reference_sigma()
    peak_fraction = peak_energy_fraction(psf, n_side=5)
    photo = anchor_photometry(11.0, 1.5, 1.0, sigma_ref, peak_fraction)
    assert photo.flux(photo.mag_limit) * peak_fraction / sigma_ref == pytest.approx(1.5, rel=1e-12)
    assert photo.magnitude(photo.flux_at_limit) == pytest.approx(11.0)
    offsets = (np.arange(5) + 0.5) / 5.0 - 0.5
    best_gap = None
    chosen = (0.0, 0.0, peak_fraction)
    for offset_y in offsets:
        for offset_x in offsets:
            fraction = peak_fraction_at(psf, float(offset_x), float(offset_y))
            gap = abs(fraction - peak_fraction)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                chosen = (float(offset_x), float(offset_y), fraction)
    offset_x, offset_y, fraction = chosen
    flux = np.array([photo.flux(11.0)])
    sources = _paint_stars(
        sensor,
        psf,
        np.array([32.0 + offset_x]),
        np.array([32.0 + offset_y]),
        flux,
        np.array([11.0]),
    )
    exposure = expose_frame(sources, sensor, 0, _pcg(1), psf)
    assert exposure.snr[0] == pytest.approx(1.5 * fraction / peak_fraction, rel=1e-6)
    assert exposure.snr[0] == pytest.approx(1.5, rel=0.06)


def test_pooled_realizations_match_the_predicted_snr():
    """200 realizations. The ratio is pooled, not averaged per frame."""
    psf = _psf(sigma0=1.2)
    sensor = _flat_sensor(sky_e=400.0, dark_e=12.0, read_e=6.0, full_well_e=1.0e8)
    sources = _paint_stars(
        sensor,
        psf,
        np.array([48.0]),
        np.array([48.0]),
        np.array([2.0e4]),
        np.array([6.0]),
    )
    row = int(sources.peak_row[0])
    col = int(sources.peak_col[0])
    peaks = np.empty(200)
    background = np.empty((200, 16, 16))
    for index in range(200):
        exposure = expose_frame(sources, sensor, 0, _pcg(1000 + index), psf)
        peaks[index] = exposure.electrons[row, col]
        background[index] = exposure.electrons[8:24, 8:24]
    signal = float(peaks.mean() - background.mean())
    noise = float(background.std(ddof=1))
    measured = signal / noise
    # Uniform PRNU and vignetting are 1, so the exposed s_self matches the linear one.
    assert exposure.s_self[0] == pytest.approx(sources.s_self_linear[0])
    assert measured == pytest.approx(float(exposure.snr[0]), rel=0.05)


def test_local_noise_on_a_bright_wing_matches_200_realizations():
    psf = _psf(sigma0=1.2)
    sensor = _flat_sensor(height=128, width=128, sky_e=150.0, dark_e=5.0, read_e=4.0, full_well_e=1.0e9)
    faint_x = [70.0 + 4.0 * math.cos(angle) for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)]
    faint_y = [70.0 + 4.0 * math.sin(angle) for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)]
    x = np.array([70.0, *faint_x])
    y = np.array([70.0, *faint_y])
    flux = np.array([8.0e6, *([80.0] * 8)])
    mag = np.array([2.0, *([14.0] * 8)])
    sources = _paint_stars(sensor, psf, x, y, flux, mag)
    probe = expose_frame(sources, sensor, 0, _pcg(2), psf)
    stack = np.empty((200, 8))
    for frame in range(200):
        exposure = expose_frame(sources, sensor, 0, _pcg(3000 + frame), psf)
        for star in range(8):
            stack[frame, star] = exposure.electrons[sources.peak_row[star + 1], sources.peak_col[star + 1]]
    ratios = stack.std(axis=0, ddof=1) / probe.sigma_local[1:]
    # The formula excludes the faint star's own shot noise, so the ratio sits
    # slightly above 1. The wing, not the sky, dominates the denominator.
    # The check uses the mean of 8 stars: over 200 realizations the standard
    # error of one star's sample std is about 5% (σ / √(2N)), so a single
    # star would not resolve a 5% bias.
    assert abs(float(ratios.mean()) - 1.0) < 0.05
    sky_only = math.sqrt(150.0 + 5.0 + 16.0)
    assert float(probe.sigma_local[1]) > 2.0 * sky_only


def test_effective_snr_is_the_root_sum_square():
    assert effective_snr([3.0, 4.0, 0.0]) == pytest.approx(5.0)
    assert effective_snr([]) == 0.0
    assert peak_snr(0.0, 0.0) == 0.0
    assert local_sigma(10.0, 4.0, 1.0, 2.0) == pytest.approx(math.sqrt(10.0 - 4.0 + 1.0 + 4.0))


def test_equal_pairs_merge_at_2_00_sigma_and_split_at_2_02():
    sigma = 1.0
    psf = _psf(sigma0=sigma)
    radii = np.full(2, 32)

    def merged(separation: float, ratio: float) -> bool:
        result = blend_stars(
            np.array([0.0, separation]),
            np.array([0.0, 0.0]),
            np.array([1.0, ratio]),
            psf,
            radii,
            search_sigma=40.0,
        )
        return bool(result.blended[0])

    assert merged(2.00 * sigma, 1.0)
    assert not merged(2.02 * sigma, 1.0)
    for ratio in (0.1, 0.01):
        assert merged(2.0 * sigma, ratio)
        assert not merged(12.0 * sigma, ratio)
        lo, hi = 2.0 * sigma, 12.0 * sigma
        for _step in range(24):
            mid = 0.5 * (lo + hi)
            if merged(mid, ratio):
                lo = mid
            else:
                hi = mid
        assert lo > 2.0 * sigma


def test_three_star_chain_merges_through_the_union_find():
    sigma = 1.0
    psf = _psf(sigma0=sigma)
    radii = np.full(3, 32)
    step = 1.5 * sigma
    forward = blend_stars(
        np.array([0.0, step, 2.0 * step]),
        np.zeros(3),
        np.ones(3),
        psf,
        radii,
        search_sigma=4.0,
    )
    backward = blend_stars(
        np.array([2.0 * step, step, 0.0]),
        np.zeros(3),
        np.ones(3),
        psf,
        radii,
        search_sigma=4.0,
    )
    assert forward.blended.all()
    assert len(forward.groups) == 1
    assert set(forward.groups[0].tolist()) == {0, 1, 2}
    assert backward.blended.all()
    assert len(backward.groups) == 1
    assert forward.centroid_x[0] == pytest.approx(step)
    separated = blend_stars(
        np.array([0.0, 3.0 * sigma, 6.0 * sigma]),
        np.zeros(3),
        np.ones(3),
        psf,
        radii,
        search_sigma=4.0,
    )
    assert not separated.blended.any()
    assert len(separated.groups) == 3


def test_third_star_beyond_the_pair_stays_alone():
    """A and B at 1.5σ merge. C, 2.2–3.0σ outside B, stays its own group."""
    sigma = 1.0
    psf = _psf(sigma0=sigma)
    radii = np.full(3, 32)
    for gap in (2.2, 2.6, 3.0):
        c_at = 1.5 * sigma + gap * sigma
        result = blend_stars(
            np.array([0.0, 1.5 * sigma, c_at]),
            np.zeros(3),
            np.ones(3),
            psf,
            radii,
            search_sigma=8.0,
        )
        groups = _partition(result)
        assert groups == {frozenset({0, 1}), frozenset({2})}, gap


def test_a_star_outside_the_search_radius_still_enters_the_profile():
    """The profile is the whole field of original members, not the pair alone.

    C sits outside the search radius of both A and B, so the first round
    cannot pair C with either end. Alone, A and B at 2.3σ have a valley.
    C's light removes that valley, A and B merge, and the merged centroid
    then falls inside C's search radius.
    """
    psf = _psf(sigma0=1.0)
    alone = blend_stars(
        np.array([0.0, 2.3]),
        np.zeros(2),
        np.ones(2),
        psf,
        np.full(2, 32),
        search_sigma=2.5,
    )
    assert not alone.blended.any()
    field = blend_stars(
        np.array([0.0, 2.3, 1.15]),
        np.array([0.0, 0.0, 2.40]),
        np.array([1.0, 1.0, 30.0]),
        psf,
        np.full(3, 32),
        search_sigma=2.5,
    )
    assert field.blended.all()
    assert len(field.groups) == 1


def test_shuffling_sources_does_not_change_the_blend_partition():
    psf = _psf(sigma0=1.0)
    x = np.array([0.0, 1.5, 4.2, 12.0, 12.4])
    y = np.array([0.0, 0.2, 0.0, 3.0, 3.1])
    flux = np.array([1.0, 1.2, 0.4, 2.0, 0.5])
    radii = np.full(5, 32)
    original = blend_stars(x, y, flux, psf, radii, search_sigma=6.0)
    order = _pcg(5).permutation(5)
    shuffled = blend_stars(x[order], y[order], flux[order], psf, radii, search_sigma=6.0)
    remapped = [{int(order[index]) for index in group.tolist()} for group in shuffled.groups]
    assert {frozenset(group) for group in remapped} == _partition(original)


def test_box_edge_crossing_the_rising_segment_still_merges_the_pair():
    """A third star's stamp edge cuts the rising half of an equal pair.

    The pair is 400 e- each. The third star is bright enough that a hard
    cut at that edge would open a valley; the continuous sum does not.
    """
    sigma = 1.15
    psf = _psf(sigma0=sigma, halo_fraction=0.02, width=256, height=256, cx=127.5, cy=127.5)
    for separation in (1.5 * sigma, 1.8 * sigma):
        x0 = 80.0
        y0 = 80.0
        third_flux = 2000.0
        radius = int(
            noise_capture_radii(np.array([128.0]), np.array([128.0]), np.array([third_flux]), psf, 15.0)[0]
        )
        # col1 = x0 puts the box's right edge at x0 + 0.5, on the rising half.
        third_x = (int(x0) - 1) - radius
        x = np.array([x0, x0 + separation, float(third_x)])
        y = np.array([y0, y0, y0])
        flux = np.array([400.0, 400.0, third_flux])
        radii = noise_capture_radii(x, y, flux, psf, 15.0)
        row0, row1, col0, col1 = _unclipped_boxes(x, y, radii)
        samples = np.linspace(0.0, separation, PROFILE_SAMPLES)
        sample_x = x0 + samples
        sample_row = np.floor(y0 + 0.5)
        sample_col = np.floor(sample_x + 0.5)
        inside = (
            (sample_row >= row0[2])
            & (sample_row <= row1[2])
            & (sample_col >= col0[2])
            & (sample_col <= col1[2])
        )
        assert bool(inside[0]) and not bool(inside[-1])
        crossed = np.flatnonzero(inside[:-1] & ~inside[1:])
        assert crossed.size >= 1
        assert float(sample_x[int(crossed[0])]) < x0 + 0.5 * separation
        result = blend_stars(x, y, flux, psf, radii, search_sigma=12.0)
        assert result.group_id[0] == result.group_id[1]


def test_halo_scans_match_the_continuous_valley_at_both_radii():
    """Four hundred placements, real capture radii and a radius of 200.

    The valley on the A-B segment has to match a continuous reference both
    ways. These are the halo fractions and separations where a stamp cut
    used to split pairs the continuous sum keeps together.
    """
    sigma = 1.15
    cases = ((0.0, 1.8), (0.02, 1.5), (0.02, 1.8), (0.03, 1.95))
    for case_index, (halo, separation_sigma) in enumerate(cases):
        psf = _psf(sigma0=sigma, halo_fraction=halo, width=256, height=256, cx=127.5, cy=127.5)
        separation = separation_sigma * sigma
        rng = _pcg(5000 + case_index)
        mismatches = {True: 0, False: 0}
        crossings = 0
        for _trial in range(400):
            third_x = float(rng.uniform(-6.0, separation + 6.0))
            third_y = float(rng.uniform(-6.0, 6.0))
            x = np.array([0.0, separation, third_x])
            y = np.array([0.0, 0.0, third_y])
            flux = np.array([400.0, 400.0, 400.0])
            real = noise_capture_radii(x, y, flux, psf, 15.0)
            for use_real, radii in ((True, real), (False, np.full(3, 200, dtype=np.int64))):
                row0, row1, col0, col1 = _unclipped_boxes(x, y, radii)
                produced = _segment_profile(
                    x[0], y[0], x[1], y[1], x, y, flux, row0, row1, col0, col1, psf, PROFILE_SAMPLES, True,
                )
                reference = _reference_profile(
                    x[0], y[0], x[1], y[1], x, y, flux, row0, row1, col0, col1, psf, PROFILE_SAMPLES, True,
                )
                produced_keep = _sources_on_segment(
                    x[0], y[0], x[1], y[1], row0, row1, col0, col1, True,
                )
                reference_keep = _reference_sources_on_segment(
                    x[0], y[0], x[1], y[1], row0, row1, col0, col1, True,
                )
                assert np.array_equal(produced_keep, reference_keep)
                _assert_profiles_close(produced, reference)
                if has_profile_valley(produced) != has_profile_valley(reference):
                    mismatches[use_real] += 1
            row0, row1, col0, col1 = _unclipped_boxes(x, y, real)
            samples = np.linspace(0.0, separation, PROFILE_SAMPLES)
            sample_x = samples
            sample_y = np.zeros(PROFILE_SAMPLES)
            sample_col = np.floor(sample_x + 0.5)
            sample_row = np.floor(sample_y + 0.5)
            inside = (
                (sample_row >= row0[2]) & (sample_row <= row1[2])
                & (sample_col >= col0[2]) & (sample_col <= col1[2])
            )
            if bool(inside[0]) != bool(inside[-1]) or np.any(inside[:-1] != inside[1:]):
                crossings += 1
        assert mismatches[True] == 0
        assert mismatches[False] == 0
        if halo > 0.0:
            assert crossings > 0


def test_screened_profile_matches_the_continuous_reference():
    """Box intersection, then a continuous sum, checked against an independent reference."""
    rng = _pcg(4242)
    count = 15
    psf = _psf(sigma0=1.15, halo_fraction=0.02, width=128, height=128, cx=63.5, cy=63.5)
    x = rng.uniform(10.0, 110.0, size=count)
    y = rng.uniform(10.0, 110.0, size=count)
    flux = rng.uniform(80.0, 8000.0, size=count)
    radii = noise_capture_radii(x, y, flux, psf, 15.0)
    row0, row1, col0, col1 = _unclipped_boxes(x, y, radii)
    compared = 0
    for _draw in range(40):
        left, right = (int(value) for value in rng.integers(0, count, size=2))
        if left == right:
            continue
        if math.hypot(float(x[left] - x[right]), float(y[left] - y[right])) == 0.0:
            continue
        x1, y1 = float(x[left]), float(y[left])
        x2, y2 = float(x[right]), float(y[right])
        produced = _segment_profile(
            x1, y1, x2, y2, x, y, flux, row0, row1, col0, col1, psf, PROFILE_SAMPLES, True,
        )
        reference = _reference_profile(
            x1, y1, x2, y2, x, y, flux, row0, row1, col0, col1, psf, PROFILE_SAMPLES, True,
        )
        produced_keep = _sources_on_segment(x1, y1, x2, y2, row0, row1, col0, col1, True)
        reference_keep = _reference_sources_on_segment(x1, y1, x2, y2, row0, row1, col0, col1, True)
        assert np.array_equal(produced_keep, reference_keep)
        _assert_profiles_close(produced, reference)
        compared += 1
        if compared == 20:
            break
    assert compared == 20


def test_raw_endpoint_on_the_left_edge_is_kept():
    """A rebuilt sample end falls short of the original endpoint by 1 ulp.

    The segment runs from (0.1, 0) to (2.5, 3.2613443344334434). Rebuilding
    the far x the old way yields 2.4999999999999996, which misses a box whose
    left edge is exactly 2.5. The raw endpoint still touches that edge.
    """
    x1, y1 = 0.1, 0.0
    x2 = 2.5
    y2 = 3.2613443344334434
    distance = math.hypot(x2 - x1, y2 - y1)
    samples = np.linspace(0.0, distance, PROFILE_SAMPLES)
    rebuilt_x = float(x1 + samples[-1] * ((x2 - x1) / distance))
    assert rebuilt_x < 2.5
    row0 = np.array([0], dtype=np.int64)
    row1 = np.array([6], dtype=np.int64)
    col0 = np.array([3], dtype=np.int64)
    col1 = np.array([8], dtype=np.int64)
    assert min(y1, y2) >= float(row0[0]) - 0.5
    assert max(y1, y2) < float(row1[0]) + 0.5
    produced = _sources_on_segment(x1, y1, x2, y2, row0, row1, col0, col1, True)
    reference = _reference_sources_on_segment(x1, y1, x2, y2, row0, row1, col0, col1, True)
    assert bool(produced[0])
    assert bool(reference[0])
    reversed_produced = _sources_on_segment(x2, y2, x1, y1, row0, row1, col0, col1, True)
    reversed_reference = _reference_sources_on_segment(x2, y2, x1, y1, row0, row1, col0, col1, True)
    assert bool(reversed_produced[0])
    assert bool(reversed_reference[0])


def test_half_open_screen_follows_each_edge_and_its_neighbors():
    """Left and top edges include a touch. Right and bottom edges exclude one."""
    row0 = np.array([10], dtype=np.int64)
    row1 = np.array([20], dtype=np.int64)
    col0 = np.array([10], dtype=np.int64)
    col1 = np.array([20], dtype=np.int64)
    left = float(col0[0]) - 0.5
    right = float(col1[0]) + 0.5
    top = float(row0[0]) - 0.5
    bottom = float(row1[0]) + 0.5

    def kept(x1, y1, x2, y2, boxes=(row0, row1, col0, col1)):
        box_row0, box_row1, box_col0, box_col1 = boxes
        produced = _sources_on_segment(x1, y1, x2, y2, box_row0, box_row1, box_col0, box_col1, True)
        reference = _reference_sources_on_segment(
            x1, y1, x2, y2, box_row0, box_row1, box_col0, box_col1, True,
        )
        assert np.array_equal(produced, reference)
        return bool(produced[0])

    assert kept(left - 4.0, 15.0, left, 15.0)
    assert kept(left - 4.0, 15.0, float(np.nextafter(left, np.inf)), 15.0)
    assert not kept(left - 4.0, 15.0, float(np.nextafter(left, -np.inf)), 15.0)
    assert not kept(right, 15.0, right + 4.0, 15.0)
    assert kept(float(np.nextafter(right, -np.inf)), 15.0, right + 4.0, 15.0)
    assert not kept(float(np.nextafter(right, np.inf)), 15.0, right + 4.0, 15.0)
    assert kept(15.0, top - 4.0, 15.0, top)
    assert kept(15.0, top - 4.0, 15.0, float(np.nextafter(top, np.inf)))
    assert not kept(15.0, top - 4.0, 15.0, float(np.nextafter(top, -np.inf)))
    assert not kept(15.0, bottom, 15.0, bottom + 4.0)
    assert kept(15.0, float(np.nextafter(bottom, -np.inf)), 15.0, bottom + 4.0)
    assert not kept(15.0, float(np.nextafter(bottom, np.inf)), 15.0, bottom + 4.0)

    forward = _sources_on_segment(left - 4.0, 12.0, 15.0, 16.0, row0, row1, col0, col1, True)
    backward = _sources_on_segment(15.0, 16.0, left - 4.0, 12.0, row0, row1, col0, col1, True)
    backward_reference = _reference_sources_on_segment(
        15.0, 16.0, left - 4.0, 12.0, row0, row1, col0, col1, True,
    )
    assert np.array_equal(forward, backward)
    assert np.array_equal(backward, backward_reference)
    on_edge = _sources_on_segment(left - 4.0, 15.0, left, 15.0, row0, row1, col0, col1, True)
    flipped_edge = _sources_on_segment(left, 15.0, left - 4.0, 15.0, row0, row1, col0, col1, True)
    assert np.array_equal(on_edge, flipped_edge)

    assert kept(12.0, 15.0, 18.0, 15.0)
    assert kept(15.0, 12.0, 15.0, 18.0)
    assert kept(12.0, top, 18.0, top)
    assert not kept(12.0, bottom, 18.0, bottom)
    assert kept(left, 12.0, left, 18.0)
    assert not kept(right, 12.0, right, 18.0)
    assert 11.0 > left and 16.0 < right and 12.0 > top and 17.0 < bottom
    assert kept(11.0, 12.0, 16.0, 17.0)

    sentinel = (
        np.array([-1], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        np.array([-1], dtype=np.int64),
    )
    for ends in ((0.0, 0.0, 3.0, 4.0), (100.0, -50.0, 130.0, -40.0), (12.0, 15.0, 18.0, 16.0)):
        assert not kept(*ends, sentinel)
    mixed_row0 = np.array([-1, 10], dtype=np.int64)
    mixed_row1 = np.array([-1, 20], dtype=np.int64)
    mixed_col0 = np.array([-1, 10], dtype=np.int64)
    mixed_col1 = np.array([-1, 20], dtype=np.int64)
    produced = _sources_on_segment(
        12.0, 15.0, 18.0, 16.0, mixed_row0, mixed_row1, mixed_col0, mixed_col1, True,
    )
    reference = _reference_sources_on_segment(
        12.0, 15.0, 18.0, 16.0, mixed_row0, mixed_row1, mixed_col0, mixed_col1, True,
    )
    assert np.array_equal(produced, reference)
    assert produced.tolist() == [False, True]
    open_keep = _sources_on_segment(
        12.0, 15.0, 18.0, 16.0, mixed_row0, mixed_row1, mixed_col0, mixed_col1, False,
    )
    open_reference = _reference_sources_on_segment(
        12.0, 15.0, 18.0, 16.0, mixed_row0, mixed_row1, mixed_col0, mixed_col1, False,
    )
    assert np.array_equal(open_keep, open_reference)
    assert open_keep.tolist() == [False, True]


def _blend_rate_field_psf(camera, sigma, halo):
    return PsfModel(
        sigma0=sigma,
        field_growth=0.0,
        max_ellipticity=0.0,
        halo_fraction=halo,
        width=camera.width,
        height=camera.height,
        cx=camera.cx,
        cy=camera.cy,
    )


def test_blend_rate_is_the_same_order_as_the_tycho_pair_counts(catalog, camera):
    """Neighbor counts use σ = 1 and no halo. The wider PSF is a separate test."""
    narrow = _blend_rate_field_psf(camera, 1.0, 0.0)
    half_width = camera.width / (2.0 * camera.f_over_p)
    half_height = camera.height / (2.0 * camera.f_over_p)
    radius_deg = math.degrees(math.atan(math.hypot(half_width, half_height))) + 0.2
    photo = anchor_photometry(12.0, 1.5, 1.0, 20.0, 0.1)
    rng = _pcg(11)
    neighbor = {1.0: [], 2.0: [], 4.0: []}
    blended_fractions = []
    group_disagreements = 0
    for _index in range(40):
        ra = float(rng.uniform(0.0, 360.0))
        dec = math.degrees(math.asin(float(rng.uniform(-1.0, 1.0))))
        attitude = pointing_roll_to_matrix(math.radians(ra), math.radians(dec), 0.0)
        found = catalog.cone_query(ra, dec, radius_deg)
        if found.size < 50:
            continue
        xy = np.asarray(project_icrs(camera, attitude, catalog.xyz[found]))
        inside = centers_in_image(xy[:, 0], xy[:, 1], camera.width, camera.height)
        bright = catalog.vmag[found] <= 12.0
        keep = inside & bright
        if int(keep.sum()) < 50:
            continue
        points = xy[keep]
        mag = catalog.vmag[found][keep]
        flux = np.asarray(photo.flux(mag), dtype=np.float64)
        tree_x = points[:, 0]
        tree_y = points[:, 1]
        from scipy.spatial import cKDTree

        distances, _indexes = cKDTree(points).query(points, k=2)
        nearest = distances[:, 1]
        for limit in neighbor:
            neighbor[limit].append(float(np.mean(nearest < limit)))

        def _groups(psf):
            radii = noise_capture_radii(tree_x, tree_y, flux, psf, 20.0)
            boxes = clipped_stamp_boxes(tree_x, tree_y, radii, camera.height, camera.width)
            screened = blend_stars(
                tree_x, tree_y, flux, psf, radii, search_sigma=4.0,
                box_row0=boxes[0], box_row1=boxes[1], box_col0=boxes[2], box_col1=boxes[3],
                screen=True,
            )
            reference = blend_stars(
                tree_x, tree_y, flux, psf, radii, search_sigma=4.0,
                box_row0=boxes[0], box_row1=boxes[1], box_col0=boxes[2], box_col1=boxes[3],
                screen=False,
            )
            return _partition(screened) != _partition(reference), float(np.mean(screened.blended))

        disagreed, fraction = _groups(narrow)
        if disagreed:
            group_disagreements += 1
        blended_fractions.append(fraction)
    report = {
        "n_fields": len(blended_fractions),
        "neighbor_fraction_lt1_median": float(np.median(neighbor[1.0])) if neighbor[1.0] else None,
        "neighbor_fraction_lt2_median": float(np.median(neighbor[2.0])) if neighbor[2.0] else None,
        "neighbor_fraction_lt4_median": float(np.median(neighbor[4.0])) if neighbor[4.0] else None,
        "blend_rate_median": float(np.median(blended_fractions)) if blended_fractions else None,
        "blend_rate_p95": float(np.percentile(blended_fractions, 95)) if blended_fractions else None,
        "historical_blend_rate_median_v2": 0.0066,
        "historical_blend_rate_p95_v2": 0.0145,
        "historical_blend_rate_note": (
            "BLEND_VERSION 2 实现在同一批 40 个天区上的结果，只供对照，不是独立基准，也不参与断言"
        ),
        "screen_psf_sigma": 1.0,
        "screen_psf_halo_fraction": 0.0,
        "screen_vs_full_group_disagreements": group_disagreements,
    }
    write_section(REPORT_PATH, "blend_rate", report)
    assert len(blended_fractions) >= 30
    median_4 = float(np.median(neighbor[4.0]))
    median_blend = float(np.median(blended_fractions))
    message = (
        f"median fraction with a neighbor <1/<2/<4 px = "
        f"{np.median(neighbor[1.0]):.4%}/{np.median(neighbor[2.0]):.4%}/{median_4:.4%}; "
        f"median blended fraction = {median_blend:.4%}, "
        f"95th = {report['blend_rate_p95']:.4%}"
    )
    # Plan section 0.9, m <= 12: about 0.17%, 0.46%, 1.22% (median frame).
    assert 0.0005 < float(np.median(neighbor[1.0])) < 0.01, message
    assert 0.001 < float(np.median(neighbor[2.0])) < 0.02, message
    assert 0.004 < median_4 < 0.04, message
    assert 0.0005 < median_blend < 0.03, message


@pytest.mark.slow
def test_wide_psf_blend_groups_are_reported(catalog, camera):
    """σ = 1.5 and halo 0.03, on the same 40 fields. Reported, not asserted."""
    wide = _blend_rate_field_psf(camera, 1.5, 0.03)
    half_width = camera.width / (2.0 * camera.f_over_p)
    half_height = camera.height / (2.0 * camera.f_over_p)
    radius_deg = math.degrees(math.atan(math.hypot(half_width, half_height))) + 0.2
    photo = anchor_photometry(12.0, 1.5, 1.0, 20.0, 0.1)
    rng = _pcg(11)
    wide_fractions = []
    wide_disagreements = 0
    for _index in range(40):
        ra = float(rng.uniform(0.0, 360.0))
        dec = math.degrees(math.asin(float(rng.uniform(-1.0, 1.0))))
        attitude = pointing_roll_to_matrix(math.radians(ra), math.radians(dec), 0.0)
        found = catalog.cone_query(ra, dec, radius_deg)
        if found.size < 50:
            continue
        xy = np.asarray(project_icrs(camera, attitude, catalog.xyz[found]))
        inside = centers_in_image(xy[:, 0], xy[:, 1], camera.width, camera.height)
        bright = catalog.vmag[found] <= 12.0
        keep = inside & bright
        if int(keep.sum()) < 50:
            continue
        points = xy[keep]
        mag = catalog.vmag[found][keep]
        flux = np.asarray(photo.flux(mag), dtype=np.float64)
        tree_x = points[:, 0]
        tree_y = points[:, 1]
        radii = noise_capture_radii(tree_x, tree_y, flux, wide, 20.0)
        boxes = clipped_stamp_boxes(tree_x, tree_y, radii, camera.height, camera.width)
        screened = blend_stars(
            tree_x, tree_y, flux, wide, radii, search_sigma=4.0,
            box_row0=boxes[0], box_row1=boxes[1], box_col0=boxes[2], box_col1=boxes[3],
            screen=True,
        )
        reference = blend_stars(
            tree_x, tree_y, flux, wide, radii, search_sigma=4.0,
            box_row0=boxes[0], box_row1=boxes[1], box_col0=boxes[2], box_col1=boxes[3],
            screen=False,
        )
        if _partition(screened) != _partition(reference):
            wide_disagreements += 1
        wide_fractions.append(float(np.mean(screened.blended)))
    extra = {
        "wide_psf_sigma": 1.5,
        "wide_psf_halo_fraction": 0.03,
        "wide_psf_screen_vs_full_group_disagreements": wide_disagreements,
        "wide_psf_blend_rate_median": float(np.median(wide_fractions)) if wide_fractions else None,
    }
    write_section(REPORT_PATH, "wide_psf", extra)
    assert section_problems(REPORT_PATH, ("blend_rate", "wide_psf")) == []
    assert wide_fractions


def test_hot_pixels_are_stable_rts_or_clusters(camera):
    rngs = SequenceRng(sequence_seed(root_seed(19), 2))
    effective = _noise_effective()
    effective["dataset"]["n_frames"] = 40
    sensor = build_sensor(effective, camera, rngs)
    for name in (
        STREAM_GRADIENT,
        STREAM_SMOOTH_SCALE,
        STREAM_SMOOTH_FIELD,
        STREAM_PRNU,
        STREAM_DSNU,
        STREAM_HOT,
        STREAM_DEAD,
        STREAM_FPN,
    ):
        assert name in rngs.issued
    with pytest.raises(ValueError):
        build_sensor(effective, camera, rngs)

    kind = sensor.hot_kind
    assert kind.size > 200
    cluster = kind == KIND_CLUSTER
    rts = kind == KIND_RTS
    stable = kind == KIND_STABLE
    assert abs(float(cluster.mean()) - 0.10) < 0.02
    assert abs(float(rts.mean()) - 0.05) < 0.02
    assert not np.any(cluster & rts)

    mask = np.zeros((sensor.height, sensor.width), dtype=bool)
    mask[sensor.hot_row, sensor.hot_col] = True
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int32))
    assert count > 0
    component = labels[sensor.hot_row, sensor.hot_col]
    assert np.all(component > 0)
    sizes = np.bincount(labels.ravel())
    order = np.argsort(component, kind="mergesort")
    component_sorted = component[order]
    kind_sorted = kind[order]
    breaks = np.flatnonzero(np.diff(component_sorted)) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), breaks))
    ends = np.concatenate((breaks, np.array([component_sorted.shape[0]], dtype=np.int64)))
    for start, end in zip(starts.tolist(), ends.tolist()):
        comp_id = int(component_sorted[start])
        kinds = kind_sorted[start:end]
        size = int(sizes[comp_id])
        assert size == end - start
        if np.any(kinds == KIND_CLUSTER):
            assert np.all(kinds == KIND_CLUSTER)
            assert 2 <= size <= 3
        else:
            assert size == 1
            assert int(kinds[0]) in (KIND_STABLE, KIND_RTS)

    level0 = sensor.hot_level(0)
    for frame in range(1, sensor.n_frames):
        level = sensor.hot_level(frame)
        assert np.array_equal(level[stable | cluster], level0[stable | cluster])
    rts_levels = np.stack([sensor.hot_level(frame)[rts] for frame in range(sensor.n_frames)], axis=1)
    for index in range(rts_levels.shape[0]):
        values = set(np.round(rts_levels[index], decimals=6).tolist())
        assert values == {
            round(float(sensor.hot_level_a[rts][index]), 6),
            round(float(sensor.hot_level_b[rts][index]), 6),
        }
    states = sensor.hot_state[rts]
    flips = np.sum(states[:, 1:] != states[:, :-1], axis=1) / (sensor.n_frames - 1)
    assert abs(float(flips.mean()) - float(sensor.hot_p_flip[rts].mean())) < 0.03

    dead = set(zip(sensor.dead_row.tolist(), sensor.dead_col.tolist()))
    hot = set(zip(sensor.hot_row.tolist(), sensor.hot_col.tolist()))
    assert dead.isdisjoint(hot)


def test_non_rts_hot_pixels_fluctuate_only_with_poisson_and_read_noise():
    sensor = _flat_sensor(height=48, width=48, sky_e=80.0, dark_e=4.0, read_e=3.0, n_frames=8)
    rows = np.arange(8, 40, 4)
    cols = np.arange(8, 40, 4)
    grid_row, grid_col = np.meshgrid(rows, cols, indexing="ij")
    count = int(grid_row.size)
    sensor.hot_row = grid_row.ravel().astype(np.int64)
    sensor.hot_col = grid_col.ravel().astype(np.int64)
    sensor.hot_kind = np.full(count, KIND_STABLE, dtype=np.int64)
    sensor.hot_level_a = np.full(count, 600.0)
    sensor.hot_level_b = np.full(count, 600.0)
    sensor.hot_state = np.zeros((count, sensor.n_frames), dtype=np.uint8)
    sensor.hot_p_flip = np.zeros(count)
    psf = _psf(width=48, height=48, cx=23.5, cy=23.5)
    sources = _paint_stars(sensor, psf, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
    samples = np.empty((30, count))
    for frame in range(30):
        exposure = expose_frame(sources, sensor, 0, _pcg(5000 + frame), psf)
        samples[frame] = exposure.electrons[sensor.hot_row, sensor.hot_col]
    mean = 80.0 + 4.0 + 600.0
    expected = mean + 9.0
    ratio = float(samples.var(axis=0, ddof=1).mean() / expected)
    assert ratio == pytest.approx(1.0, rel=0.1)
    assert np.allclose(sensor.hot_level(0), sensor.hot_level(sensor.n_frames - 1))


def test_blooming_walks_excess_of_any_overfull_star():
    """A star fainter than magnitude 4 still blooms once it is over full well."""
    psf = _psf(sigma0=1.0, width=64, height=96, cx=31.5, cy=47.5)
    sensor = _flat_sensor(
        height=96,
        width=64,
        sky_e=20.0,
        dark_e=0.0,
        read_e=0.0,
        full_well_e=800.0,
        blooming=True,
        n_frames=1,
    )
    quiet = _flat_sensor(
        height=96,
        width=64,
        sky_e=20.0,
        dark_e=0.0,
        read_e=0.0,
        full_well_e=800.0,
        blooming=False,
        n_frames=1,
    )
    sources = _paint_stars(sensor, psf, np.array([32.0]), np.array([30.0]), np.array([2.0e5]), np.array([6.0]))
    bloomed = expose_frame(sources, sensor, 0, _pcg(3), psf)
    clipped = expose_frame(sources, quiet, 0, _pcg(3), psf)
    row = int(sources.peak_row[0])
    col = int(sources.peak_col[0])
    assert bloomed.peak_clipped[0]
    assert clipped.peak_clipped[0]
    assert bloomed.peak_owner[0] == 0
    assert clipped.peak_owner[0] == 0
    assert bloomed.mean_e[row, col] == pytest.approx(800.0)
    assert not np.any(bloomed.mean_e > sensor.full_well_e)
    overflow = bloomed.mean_e[:, col] - clipped.mean_e[:, col]
    assert float(overflow.sum()) > 100.0
    assert np.allclose(bloomed.mean_e[:, col + 8], clipped.mean_e[:, col + 8])
    spilled = np.flatnonzero(overflow > 1.0)
    assert spilled.size > 0
    assert np.all(bloomed.saturation_owner[spilled, col] == 0)
    _assert_step2(bloomed, sensor, sources, 0)
    _assert_step4(bloomed)


def test_step2_conserves_charge_for_bright_stars_and_a_sky_push():
    psf = _psf(sigma0=1.1, width=160, height=256, cx=79.5, cy=127.5)
    photo = _photo(psf)
    for magnitude in (0.0, -1.4):
        sensor = _flat_sensor(
            height=256,
            width=160,
            sky_e=80.0,
            dark_e=2.0,
            read_e=3.0,
            full_well_e=9.0e4,
            blooming=True,
            n_frames=1,
        )
        flux = np.array([photo.flux(magnitude)])
        mag = np.array([magnitude])
        assert photo.magnitude(float(flux[0])) == pytest.approx(magnitude)
        sources = _paint_stars(sensor, psf, np.array([80.0]), np.array([128.0]), flux, mag)
        exposure = expose_frame(sources, sensor, 0, _pcg(40 + int(magnitude * 10)), psf)
        assert exposure.peak_clipped[0]
        assert exposure.peak_owner[0] == 0
        _assert_step2(exposure, sensor, sources, 0)
        _assert_step4(exposure)
        assert not np.any(exposure.mean_e > sensor.full_well_e)

    sensor = _flat_sensor(
        height=48,
        width=48,
        sky_e=700.0,
        dark_e=0.0,
        read_e=0.0,
        full_well_e=1000.0,
        blooming=True,
        n_frames=1,
    )
    psf_small = _psf(sigma0=1.2, width=48, height=48, cx=23.5, cy=23.5)
    sources = _paint_stars(
        sensor, psf_small, np.array([24.0]), np.array([24.0]), np.array([6000.0]), np.array([8.0])
    )
    row = int(sources.peak_row[0])
    col = int(sources.peak_col[0])
    star_only = float(sources.image[row, col])
    assert star_only < sensor.full_well_e
    assert star_only + sensor.background_e[row, col] > sensor.full_well_e
    exposure = expose_frame(sources, sensor, 0, _pcg(41), psf_small)
    assert exposure.peak_clipped[0]
    assert exposure.peak_owner[0] == 0
    _assert_step2(exposure, sensor, sources, 0)
    _assert_step4(exposure)


def test_shuffling_nonoverlapping_sources_keeps_m2_and_owners():
    psf = _psf(sigma0=1.0, width=256, height=256, cx=127.5, cy=127.5)
    sensor = _flat_sensor(
        height=256, width=256, sky_e=30.0, dark_e=0.0, read_e=0.0,
        full_well_e=2000.0, blooming=True, n_frames=1,
    )
    x = np.array([40.0, 210.0])
    y = np.array([40.0, 200.0])
    flux = np.array([8.0e4, 6.0e4])
    mag = np.array([4.0, 4.5])
    sources = _paint_stars(sensor, psf, x, y, flux, mag)
    original = expose_frame(sources, sensor, 0, _pcg(7), psf)
    order = np.array([1, 0])
    shuffled_sources = _paint_stars(sensor, psf, x[order], y[order], flux[order], mag[order])
    shuffled = expose_frame(shuffled_sources, sensor, 0, _pcg(7), psf)
    assert np.array_equal(shuffled.mean_e, original.mean_e)
    assert np.array_equal(shuffled.fullwell_mask, original.fullwell_mask)
    assert np.array_equal(shuffled.saturation_mask, original.saturation_mask)
    mapped = shuffled.saturation_owner.copy()
    known = mapped >= 0
    mapped[known] = order[mapped[known]]
    assert np.array_equal(mapped, original.saturation_owner)


def test_shuffling_overlapping_sources_far_from_a_boundary_keeps_masks():
    """Forty overlapping stamps, none of them sitting on a decision boundary.

    Summation order can move a pixel by ~1e-13. That is not allowed to
    change a mask or an owner once every pixel is clearly off the full-well
    and ADC thresholds and every saturated pixel has a separated winner.
    """
    count = 40
    psf = _psf(sigma0=1.2, halo_fraction=0.02, width=220, height=220, cx=109.5, cy=109.5)
    sensor = _flat_sensor(
        height=220, width=220, sky_e=40.0, dark_e=1.0, read_e=3.0,
        gain_e_per_adu=2.0, bias_adu=1000.0, full_well_e=5.0e4, blooming=True, n_frames=1,
    )
    angle = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    radius = 0.08 * np.arange(count)
    x = 110.0 + radius * np.cos(angle)
    y = 110.0 + radius * np.sin(angle)
    flux = 2500.0 + 600.0 * np.arange(count, dtype=np.float64)
    flux[-1] = 6.0e5
    mag = np.linspace(9.0, 2.0, count)
    sources = _paint_stars(sensor, psf, x, y, flux, mag)
    m1 = (sensor.background_e + sources.image) * sensor.vignetting * sensor.prnu + sensor.dark_e
    assert float(np.min(np.abs(m1 - sensor.full_well_e))) > 1e-6 * sensor.full_well_e
    original = expose_frame(sources, sensor, 0, _pcg(90), psf)
    ceiling = (1 << sensor.adc_bits) - 1
    expected = electron_to_adu(
        original.mean_e + sensor.row_offset_e[:, None] + sensor.col_offset_e[None, :],
        sensor,
    )
    assert float(np.min(np.abs(expected - (ceiling - 0.5)))) > 1e-6
    assert np.any(original.saturation_mask)
    saturated = np.argwhere(original.saturation_mask)
    for row, col in saturated.tolist():
        contribution = _pixel_contributions(sources, psf, sensor, int(row), int(col))
        ordered = np.sort(contribution)
        assert ordered[-1] > 0.0
        assert (ordered[-1] - ordered[-2]) / ordered[-1] > 1e-9
    order = _pcg(91).permutation(count)
    shuffled_sources = _paint_stars(sensor, psf, x[order], y[order], flux[order], mag[order])
    shuffled = expose_frame(shuffled_sources, sensor, 0, _pcg(90), psf)
    scale = max(float(np.max(np.abs(original.mean_e))), 1.0)
    assert float(np.max(np.abs(shuffled.mean_e - original.mean_e))) <= 1e-12 * scale
    assert np.array_equal(shuffled.fullwell_mask, original.fullwell_mask)
    assert np.array_equal(shuffled.saturation_mask, original.saturation_mask)
    assert np.array_equal(shuffled.adc_mask, original.adc_mask)
    assert np.array_equal(_remap_owners(shuffled.saturation_owner, order), original.saturation_owner)
    assert np.array_equal(_remap_owners(shuffled.peak_owner, order), original.peak_owner[order])


def test_two_stars_in_one_column_keep_separate_trails():
    psf = _psf(sigma0=1.0, width=64, height=220, cx=31.5, cy=109.5)
    sensor = _flat_sensor(
        height=220, width=64, sky_e=10.0, dark_e=0.0, read_e=0.0,
        full_well_e=1500.0, blooming=True, n_frames=1,
    )
    sources = _paint_stars(
        sensor, psf,
        np.array([32.0, 32.0]),
        np.array([40.0, 180.0]),
        np.array([1.2e5, 9.0e4]),
        np.array([3.0, 3.4]),
    )
    exposure = expose_frame(sources, sensor, 0, _pcg(11), psf)
    assert exposure.peak_owner[0] == 0
    assert exposure.peak_owner[1] == 1
    col = int(sources.peak_col[0])
    above = int(sources.peak_row[0]) - 6
    below = int(sources.peak_row[1]) + 6
    assert exposure.saturation_owner[above, col] == 0
    assert exposure.saturation_owner[below, col] == 1
    _assert_step2(exposure, sensor, sources, 0)


def test_a_dead_pixel_in_the_trail_stays_empty():
    psf = _psf(sigma0=1.0, width=64, height=96, cx=31.5, cy=47.5)
    sensor = _flat_sensor(
        height=96, width=64, sky_e=20.0, dark_e=0.0, read_e=0.0,
        full_well_e=800.0, blooming=True, n_frames=1,
    )
    sensor.dead_row = np.array([48], dtype=np.int64)
    sensor.dead_col = np.array([32], dtype=np.int64)
    sources = _paint_stars(sensor, psf, np.array([32.0]), np.array([30.0]), np.array([2.0e5]), np.array([2.0]))
    exposure = expose_frame(sources, sensor, 0, _pcg(12), psf)
    assert exposure.mean_e[48, 32] == 0.0
    assert exposure.electrons[48, 32] == 0.0
    assert exposure.mean_e[47, 32] == pytest.approx(800.0)
    assert exposure.mean_e[49, 32] == pytest.approx(800.0)
    _assert_step2(exposure, sensor, sources, 0)


def test_gain_probe_puts_the_peak_in_the_adc_mask():
    psf = _psf(sigma0=1.2, width=64, height=64, cx=31.5, cy=31.5)
    sensor = _flat_sensor(
        height=64, width=64, sky_e=40.0, dark_e=0.0, read_e=4.0,
        gain_e_per_adu=0.6, bias_adu=500.0, full_well_e=9.0e4, blooming=True, n_frames=1,
    )
    sources = _paint_stars(sensor, psf, np.array([32.2]), np.array([32.4]), np.array([5.0e5]), np.array([5.0]))
    exposure = expose_frame(sources, sensor, 0, _pcg(13), psf)
    row = int(sources.peak_row[0])
    col = int(sources.peak_col[0])
    assert exposure.mean_e[row, col] < sensor.full_well_e
    assert exposure.saturation_mask[row, col]
    assert exposure.adc_mask[row, col]
    assert exposure.peak_clipped[0]
    assert exposure.peak_owner[0] == 0
    assert not np.any(exposure.adc_clipped_low)
    _assert_adc_realized(exposure, sensor)


def test_faint_star_keeps_its_peak_inside_a_bright_neighbors_box():
    """A's peak lies in B's stamp, and A still owns it.

    B is the bright star. A's own contribution at its peak is over full
    well, and B's contribution there is far smaller. Blooming on and off
    must agree.
    """
    psf = _psf(sigma0=1.0, halo_fraction=0.02, width=128, height=128, cx=63.5, cy=63.5)
    x = np.array([60.0, 40.0])
    y = np.array([64.0, 64.0])
    flux = np.array([1.0e6, 3.0e7])
    mag = np.array([6.0, 1.0])
    for blooming in (True, False):
        sensor = _flat_sensor(
            height=128, width=128, sky_e=100.0, dark_e=0.0, read_e=3.0,
            full_well_e=5.0e4, blooming=blooming, n_frames=1,
        )
        sources = _paint_stars(sensor, psf, x, y, flux, mag)
        row = int(sources.peak_row[0])
        col = int(sources.peak_col[0])
        assert sources.box_row0[1] <= row <= sources.box_row1[1]
        assert sources.box_col0[1] <= col <= sources.box_col1[1]
        contribution = _pixel_contributions(sources, psf, sensor, row, col)
        assert contribution[0] > sensor.full_well_e
        assert contribution[1] < 0.01 * contribution[0]
        exposure = expose_frame(sources, sensor, 0, _pcg(30 + int(blooming)), psf)
        assert exposure.peak_clipped[0]
        assert exposure.peak_owner[0] == 0
        assert exposure.peak_owner[1] == 1


def test_zero_read_noise_can_set_the_realized_adc_flag():
    """Poisson alone can cross the ADC threshold when the expectation does not."""
    psf = _psf(sigma0=1.0, width=64, height=64, cx=31.5, cy=31.5)
    sensor = _flat_sensor(
        height=64, width=64, sky_e=0.0, dark_e=0.0, read_e=0.0,
        gain_e_per_adu=1.0, bias_adu=0.0, full_well_e=1.0e6, blooming=False, n_frames=1,
    )
    unit = _paint_stars(sensor, psf, np.array([32.0]), np.array([32.0]), np.array([1.0]), np.array([10.0]))
    row = int(unit.peak_row[0])
    col = int(unit.peak_col[0])
    flux = np.array([65450.0 / float(unit.image[row, col])])
    saw_realized_above_expectation = False
    for seed in range(24):
        sources = _paint_stars(sensor, psf, np.array([32.0]), np.array([32.0]), flux, np.array([8.0]))
        exposure = expose_frame(sources, sensor, 0, _pcg(800 + seed), psf)
        assert exposure.mean_e[row, col] == pytest.approx(65450.0, abs=1e-4)
        assert not exposure.adc_mask[row, col]
        assert not np.any(exposure.fullwell_mask)
        _assert_adc_realized(exposure, sensor)
        if np.any(exposure.adc_clipped_realized & ~exposure.adc_mask):
            saw_realized_above_expectation = True
    assert saw_realized_above_expectation


def test_adc_stays_clear_when_full_well_cannot_reach_the_ceiling():
    """Q' cannot pass the ADC threshold when full well plus six read-noise sigma is short of it."""
    psf = _psf(sigma0=1.0, width=64, height=64, cx=31.5, cy=31.5)
    sensor = _flat_sensor(
        height=64, width=64, sky_e=20.0, dark_e=0.0, read_e=7.0,
        gain_e_per_adu=1.25, bias_adu=250.0, full_well_e=3.2e4, blooming=True, n_frames=1,
    )
    ceiling = (1 << sensor.adc_bits) - 1
    offset = float(sensor.row_offset_e.max() + sensor.col_offset_e.max())
    bound = (
        (sensor.full_well_e + offset) / sensor.gain_e_per_adu
        + sensor.bias_adu
        + 6.0 * sensor.read_e / sensor.gain_e_per_adu
    )
    assert bound <= ceiling - 0.5
    sources = _paint_stars(sensor, psf, np.array([32.0]), np.array([32.0]), np.array([4.0e5]), np.array([4.0]))
    exposure = expose_frame(sources, sensor, 0, _pcg(81), psf)
    assert np.any(exposure.fullwell_mask)
    assert not np.any(exposure.adc_clipped_realized)
    _assert_adc_realized(exposure, sensor)


def test_adc_rounds_half_to_even_at_the_ceiling():
    assert float(np.rint(65534.5)) == 65534.0
    assert not bool(adu_is_high_clipped(65534.5, 16))
    nudged = float(np.nextafter(65534.5, 1.0e9))
    assert bool(adu_is_high_clipped(nudged, 16))
    assert not bool(adu_is_high_clipped(-0.5, 16))


def test_ab_probe_and_a_cosmic_ray_own_their_pixels():
    psf = _psf(sigma0=1.1, width=64, height=64, cx=31.5, cy=31.5)
    sensor = _flat_sensor(
        height=64, width=64, sky_e=15.0, dark_e=0.0, read_e=0.0,
        full_well_e=4000.0, blooming=True, cosmic_rate=0.0, n_frames=1,
    )
    sources = _paint_stars(
        sensor, psf,
        np.array([24.0, 40.0]),
        np.array([32.0, 33.0]),
        np.array([8.0e4, 2.0e4]),
        np.array([3.0, 5.0]),
    )
    exposure = expose_frame(sources, sensor, 0, _pcg(14), psf)
    assert exposure.peak_owner[0] == 0
    assert exposure.peak_clipped[0]

    height, width = 48, 48
    cosmic = _flat_sensor(
        height=height, width=width, sky_e=5.0, dark_e=0.0, read_e=0.0,
        full_well_e=400.0, blooming=False, cosmic_rate=40.0, n_frames=1,
    )
    hit = _first_pointlike_cosmic(40.0, height, width, 15)
    assert hit is not None
    row, col = hit
    faint = _paint_stars(
        cosmic, _psf(width=width, height=height, cx=23.5, cy=23.5),
        np.array([float(col)]), np.array([float(row)]), np.array([30.0]), np.array([12.0]),
    )
    exposed = expose_frame(faint, cosmic, 0, _pcg(15), _psf(width=width, height=height, cx=23.5, cy=23.5))
    assert exposed.fullwell_mask[row, col]
    assert exposed.saturation_owner[row, col] == -1


def test_prnu_and_vignetting_scale_s_self():
    psf = _psf(sigma0=1.2)
    sensor = _flat_sensor(sky_e=200.0, dark_e=4.0, read_e=3.0, full_well_e=1.0e8)
    sources = _paint_stars(sensor, psf, np.array([48.2]), np.array([47.6]), np.array([8000.0]), np.array([8.0]))
    row = int(sources.peak_row[0])
    col = int(sources.peak_col[0])
    sensor.prnu[row, col] = 1.15
    sensor.vignetting[row, col] = 0.82
    exposure = expose_frame(sources, sensor, 0, _pcg(16), psf)
    assert exposure.s_self[0] == pytest.approx(sources.s_self_linear[0] * 1.15 * 0.82)


def test_smooth_field_length_is_the_padded_canvas_correlation():
    """L is the generation length, measured on the padded coarse canvas.

    A single demeaned 4096 image does not recover L. The linear
    autocorrelation (zero-padded, normalized by the overlap count) of the
    canvas before the crop does. Each length uses six seeds; the average
    has to sit within 10% of L.
    """
    for length in (500.0, 1000.0, 2000.0):
        measured = []
        for seed in range(6):
            canvas = _smooth_coarse_canvas(4096, 4096, length, _pcg(100 + int(length) + seed))
            measured.append(_autocorrelation_length(canvas, pixel_scale=_SMOOTH_DOWNSAMPLE))
        assert float(np.mean(measured)) == pytest.approx(length, rel=0.10)


def test_production_smooth_field_is_normalized_and_not_periodic():
    """The 4096 image is the cropped, bicubic field, not a circular period."""
    length = 2000.0
    amplitude = 5.0
    correlations = []
    for seed in range(16):
        field = _smooth_field_with_length(4096, 4096, amplitude, length, _pcg(4000 + seed))
        assert field.shape == (4096, 4096)
        assert float(field.std()) == pytest.approx(amplitude, abs=1e-12)
        left = field[:, 0]
        right = field[:, -1]
        left = left - float(left.mean())
        right = right - float(right.mean())
        correlations.append(
            float(np.dot(left, right) / math.sqrt(float(np.dot(left, left) * np.dot(right, right))))
        )
        second = np.diff(field, n=2, axis=1)
        spectrum = np.mean(np.abs(np.fft.rfft(second, axis=1)) ** 2, axis=0)
        body = spectrum[1:]
        log_spec = np.log(body + 1e-30)
        trend = np.convolve(log_spec, np.ones(31) / 31.0, mode="same")
        residual = log_spec[40:-40] - trend[40:-40]
        assert float(residual.max()) < 0.5
    # A circular filter correlates column 0 with column 4095 at nearly 1.
    # The padded generator sits near exp(-(4095/L)^2) ≈ 0.015.
    assert float(np.mean(correlations)) < 0.2


def test_dead_pixel_mean_is_zero_and_cosmic_rays_are_single_frame():
    psf = _psf(width=32, height=32, cx=15.5, cy=15.5)
    sensor = _flat_sensor(height=32, width=32, sky_e=50.0, dark_e=2.0, read_e=0.0, cosmic_rate=0.0, n_frames=2)
    sensor.dead_row = np.array([4], dtype=np.int64)
    sensor.dead_col = np.array([5], dtype=np.int64)
    sources = _paint_stars(sensor, psf, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
    exposure = expose_frame(sources, sensor, 0, _pcg(8), psf)
    assert exposure.mean_e[4, 5] == 0.0
    sensor.cosmic_rate = 12.0
    first = expose_frame(sources, sensor, 0, _pcg(9), psf)
    second = expose_frame(sources, sensor, 1, _pcg(9), psf)
    assert len(first.cosmic_rays) > 0
    assert np.array_equal(first.electrons, second.electrons)
    continued = _pcg(9)
    frame0 = expose_frame(sources, sensor, 0, continued, psf)
    frame1 = expose_frame(sources, sensor, 1, continued, psf)
    assert frame0.cosmic_rays and frame1.cosmic_rays
    assert not np.array_equal(frame0.mean_e, frame1.mean_e)
    assert _without_cosmic_rays(frame0) == pytest.approx(_without_cosmic_rays(frame1))
    for ray in (*frame0.cosmic_rays, *frame1.cosmic_rays):
        assert len(ray.cells) >= 1
        assert ray.n_steps == len(ray.cells)
        assert ray.n_unique_cells == len(set(ray.cells))
        if ray.pointlike:
            assert ray.cells == ((ray.row, ray.col),)
            assert ray.n_steps == 1
            assert ray.path_length_px == 0.0
            assert ray.endpoint_disp_px == 0.0
            assert ray.n_unique_cells == 1
        else:
            assert ray.path_length_px == pytest.approx(ray.n_steps - 1)
            assert ray.endpoint_disp_px <= ray.path_length_px + 1e-9


def test_frame_render_of_5000_sources_is_under_three_seconds(camera):
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
    sensor = _flat_sensor(
        height=camera.height,
        width=camera.width,
        sky_e=300.0,
        dark_e=6.0,
        read_e=5.0,
        full_well_e=90000.0,
        blooming=True,
        n_frames=1,
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
    assert photo.magnitude(float(flux[0])) == pytest.approx(0.0)
    warmup = _paint_stars(sensor, psf, x[:20], y[:20], flux[:20], mag[:20])
    expose_frame(warmup, sensor, 0, _pcg(1), psf)
    started = time.perf_counter()
    sources = _paint_stars(sensor, psf, x, y, flux, mag)
    exposure = expose_frame(sources, sensor, 0, _pcg(22), psf)
    exposed = time.perf_counter()
    blend_stars(
        sources.x, sources.y, sources.flux_e, psf, sources.radius, search_sigma=4.0,
        box_row0=sources.box_row0, box_row1=sources.box_row1,
        box_col0=sources.box_col0, box_col1=sources.box_col1,
    )
    finished = time.perf_counter()
    assert exposure.adu.shape == (camera.height, camera.width)
    assert exposure.adu.dtype == np.uint16
    assert exposure.peak_clipped[0]
    assert finished - started < 3.0
    assert finished - exposed <= 0.5


def test_one_very_bright_star_and_build_sensor_stay_inside_their_budgets(camera, loaded):
    """以默认配置为基础，将帧数临时覆盖为 16。

    ``generator_default.yaml`` 里的 ``n_frames`` 仍然是 10。
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
    sensor = _flat_sensor(
        height=camera.height,
        width=camera.width,
        sky_e=300.0,
        dark_e=6.0,
        read_e=5.0,
        full_well_e=90000.0,
        blooming=True,
        n_frames=1,
    )
    photo = anchor_photometry(11.0, 1.5, 1.0, sensor.reference_sigma(), peak_energy_fraction(psf))
    mag = np.array([-1.4])
    flux = np.array([photo.flux(-1.4)])
    assert photo.magnitude(float(flux[0])) == pytest.approx(-1.4)
    expose_frame(
        _paint_stars(sensor, psf, np.array([camera.cx]), np.array([camera.cy]), flux, mag),
        sensor, 0, _pcg(2), psf,
    )
    started = time.perf_counter()
    sources = _paint_stars(sensor, psf, np.array([camera.cx]), np.array([camera.cy]), flux, mag)
    exposure = expose_frame(sources, sensor, 0, _pcg(23), psf)
    assert time.perf_counter() - started < 2.0
    assert exposure.peak_clipped[0]
    assert not np.any(exposure.mean_e > sensor.full_well_e)

    assert loaded.generator["dataset"]["n_frames"] == 10
    rngs = SequenceRng(sequence_seed(root_seed(29), 3))
    effective = sample_sequence_params(loaded, rngs)
    effective["dataset"]["n_frames"] = 16
    started = time.perf_counter()
    built = build_sensor(effective, camera, rngs)
    assert time.perf_counter() - started < 5.0
    assert built.n_frames == 16


def test_shuffled_cone_query_paints_the_same_electrons(catalog, camera):
    """A shuffled cone is sorted again before painting, so the image matches."""
    psf = PsfModel(
        sigma0=1.15,
        field_growth=0.12,
        max_ellipticity=0.12,
        halo_fraction=0.015,
        width=camera.width,
        height=camera.height,
        cx=camera.cx,
        cy=camera.cy,
    )
    photo = anchor_photometry(11.3, 1.5, 1.0, 20.0, peak_energy_fraction(psf))
    attitude = pointing_roll_to_matrix(math.radians(ALNILAM[0]), math.radians(ALNILAM[1]), 0.0)
    original = render_static_stars(catalog, camera, attitude, photo, psf, 20.0)
    real_query = catalog.cone_query
    seen = {}

    def shuffled(ra_deg, dec_deg, radius_deg):
        found = np.asarray(real_query(ra_deg, dec_deg, radius_deg), dtype=np.int64)
        assert found.size > 1
        scrambled = found[_pcg(7).permutation(found.size)]
        assert not np.array_equal(scrambled, np.sort(found))
        seen["called"] = True
        return scrambled

    catalog.cone_query = shuffled
    try:
        again = render_static_stars(catalog, camera, attitude, photo, psf, 20.0)
    finally:
        del catalog.cone_query
    assert seen.get("called") is True
    assert again.image.tobytes() == original.image.tobytes()
    assert np.array_equal(again.tyc1, original.tyc1)
    assert np.array_equal(again.tyc2, original.tyc2)
    assert np.array_equal(again.x, original.x)
    assert np.array_equal(again.y, original.y)


def test_known_fields_match_catalog_brightness_and_position(catalog, loaded, camera):
    """Write three fields for a visual check against Stellarium or Aladin.

    The assertions check that the named stars land on their catalog positions
    and rank among the brightest peaks. The PNG is an asinh stretch; the FITS
    file carries the TAN header.
    """
    rngs = SequenceRng(sequence_seed(root_seed(20261027), 4))
    effective = _noise_effective()
    # A fixed, readable exposure. The sampled intervals are replaced here on
    # purpose so the preview does not depend on which quantile was drawn.
    effective["dataset"]["n_frames"] = 1
    effective["background"]["sky_e"] = 250.0
    effective["background"]["gradient_frac"] = 0.04
    effective["background"]["smooth_field_frac"] = 0.03
    effective["background"]["vignetting_corner"] = 0.12
    effective["sensor_noise"]["read_e"] = 5.0
    effective["sensor_noise"]["dark_e_per_s"] = 15.0
    effective["sensor_noise"]["prnu"] = 0.008
    effective["sensor_noise"]["full_well_e"] = 80000.0
    effective["artifacts"]["hot_pixel_frac"] = 8.0e-5
    effective["artifacts"]["cosmic_rays_per_frame"] = 6.0
    effective["psf"] = {
        "sigma_px": 1.15,
        "field_growth": 0.12,
        "max_ellipticity": 0.12,
        "halo_fraction": 0.015,
    }
    sensor = build_sensor(effective, camera, rngs)
    psf = PsfModel.from_effective(effective["psf"], camera)
    photo = anchor_photometry(11.3, 1.5, 1.0, sensor.reference_sigma(), peak_energy_fraction(psf))
    fields = (
        ("pleiades", PLEIADES, (("Alcyone", ALCYONE),)),
        ("orion_belt", ALNILAM, (("Alnitak", ALNITAK), ("Alnilam", ALNILAM), ("Mintaka", MINTAKA))),
        (
            "dipper_pointers",
            ((DUBHE[0] + MERAK[0]) / 2.0, (DUBHE[1] + MERAK[1]) / 2.0),
            (("Dubhe", DUBHE), ("Merak", MERAK)),
        ),
    )
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for name, center, stars in fields:
        attitude = pointing_roll_to_matrix(math.radians(center[0]), math.radians(center[1]), 0.0)
        sources = render_static_stars(catalog, camera, attitude, photo, psf, sensor.reference_sigma())
        exposure = expose_frame(sources, sensor, 0, rngs.get(f"noise/frame/{name}"), psf)
        crops = []
        for star_name, (ra_deg, dec_deg) in stars:
            index = _nearest_rendered(catalog, sources, ra_deg, dec_deg)
            assert index is not None
            crops.append((star_name, int(sources.peak_row[index]), int(sources.peak_col[index])))
        _write_preview(PREVIEW_DIR / name, exposure, camera, attitude, sources, crops)
        for star_name, (ra_deg, dec_deg) in stars:
            index = _nearest_rendered(catalog, sources, ra_deg, dec_deg)
            assert index is not None
            assert sources.in_image[index]
            offset = math.hypot(
                sources.x[index] - sources.peak_col[index],
                sources.y[index] - sources.peak_row[index],
            )
            assert offset < 1.0
            rank = int(np.sum(sources.s_self_linear > sources.s_self_linear[index]))
            assert rank < 12
    assert (PREVIEW_DIR / "pleiades.fits").exists()
    assert (PREVIEW_DIR / "orion_belt.png").exists()
    assert (PREVIEW_DIR / "dipper_pointers.fits").exists()


def _without_cosmic_rays(exposure: Exposure) -> np.ndarray:
    """Mean image with this frame's cosmic-ray deposits removed."""
    image = exposure.mean_e.copy()
    for ray in exposure.cosmic_rays:
        share = ray.energy_e / len(ray.cells)
        for row, col in ray.cells:
            image[row, col] -= share
    return image


def _nearest_rendered(catalog, sources, ra_deg: float, dec_deg: float) -> int | None:
    found = catalog.cone_query(ra_deg, dec_deg, 2.0 / 60.0)
    if found.size == 0:
        return None
    target = int(found[np.argmin(catalog.vmag[found])])
    match = np.flatnonzero((sources.tyc1 == catalog.tyc1[target]) & (sources.tyc2 == catalog.tyc2[target]))
    if match.size == 0:
        return None
    return int(match[0])


def _write_preview(path, exposure, camera, attitude, sources, crops) -> None:
    header = fits.Header()
    for key, value in wcs_keywords(camera, attitude).items():
        header[key] = value
    header["BUNIT"] = "ADU"
    fits.PrimaryHDU(exposure.adu, header=header).writeto(path.with_suffix(".fits"), overwrite=True)
    valid = (sources.peak_row >= 0) & (sources.peak_col >= 0)
    star_peak = float(exposure.adu[sources.peak_row[valid], sources.peak_col[valid]].max())
    sky = float(np.median(exposure.adu))
    white = float(np.arcsinh(max(star_peak - sky, 0.0) / 8.0))
    Image.fromarray(_stretch_adu(exposure.adu, sky, white)).save(path.with_suffix(".png"))
    half = 48
    for star_name, row, col in crops:
        row0 = max(0, row - half)
        row1 = min(exposure.adu.shape[0], row + half)
        col0 = max(0, col - half)
        col1 = min(exposure.adu.shape[1], col + half)
        crop = _stretch_adu(exposure.adu[row0:row1, col0:col1], sky, white)
        Image.fromarray(crop).save(path.parent / f"{path.name}_{star_name}.png")


def _stretch_adu(adu: np.ndarray, sky: float, white: float) -> np.ndarray:
    """Asinh stretch whose white point is the star peak, not a percentile."""
    stretched = np.arcsinh((adu.astype(np.float64) - sky) / 8.0)
    stretched = np.clip(stretched, 0.0, white)
    if white <= 0.0:
        return np.zeros(adu.shape, dtype=np.uint8)
    return (stretched / white * 255.0).astype(np.uint8)


def _photo(psf: PsfModel):
    sigma = math.sqrt(80.0 + 2.0 + 9.0)
    return anchor_photometry(11.0, 1.5, 1.0, sigma, peak_energy_fraction(psf))


def _partition(result) -> set[frozenset[int]]:
    return {frozenset(group.tolist()) for group in result.groups}


def _assert_step2(exposure: Exposure, sensor, sources, frame: int) -> None:
    m1 = exposure.expectation_e + sensor.dark_e + sensor.hot_electron_image(frame)
    for ray in exposure.cosmic_rays:
        share = ray.energy_e / len(ray.cells)
        for row, col in ray.cells:
            m1[row, col] += share
    accounted = (
        float(exposure.mean_e.sum())
        + exposure.bloom_edge_loss_e
        + exposure.dead_loss_e
        + exposure.fullwell_clip_loss_e
    )
    assert float(m1.sum()) == pytest.approx(accounted, rel=1e-9, abs=1e-4)
    assert not np.any(exposure.mean_e > sensor.full_well_e + 1e-6)


def _assert_step4(exposure: Exposure) -> None:
    assert exposure.poisson_sum_e == exposure.q_prime_sum_e + exposure.post_sample_clip_loss_e


def _first_pointlike_cosmic(rate: float, height: int, width: int, seed: int):
    """Replay the cosmic-ray generator and return the first point-like hit."""
    rng = _pcg(seed)
    count = int(rng.poisson(rate))
    for _index in range(count):
        rng.uniform(500.0, 20000.0)
        row = int(rng.integers(0, height))
        col = int(rng.integers(0, width))
        pointlike = float(rng.random()) < 0.5
        if pointlike:
            return row, col
        length = float(rng.uniform(2.0, 30.0))
        steps = max(2, int(round(length)))
        cursor_row, cursor_col = row, col
        for _step in range(steps - 1):
            options = []
            for next_row, next_col in (
                (cursor_row - 1, cursor_col),
                (cursor_row + 1, cursor_col),
                (cursor_row, cursor_col - 1),
                (cursor_row, cursor_col + 1),
            ):
                if 0 <= next_row < height and 0 <= next_col < width:
                    options.append((next_row, next_col))
            if not options:
                break
            cursor_row, cursor_col = options[int(rng.integers(0, len(options)))]
    return None


def _reference_sources_on_segment(x1, y1, x2, y2, row0, row1, col0, col1, screen):
    """Per-source half-open screen. This is not the production vector test.

    A stamp covers ``[col0 - 0.5, col1 + 0.5) x [row0 - 0.5, row1 + 0.5)``.
    The segment box is the min and max of the original endpoints. A sentinel
    whose four sides are all -1 is dropped wherever the segment sits.
    """
    count = int(np.asarray(row0).shape[0])
    xmin = min(float(x1), float(x2))
    xmax = max(float(x1), float(x2))
    ymin = min(float(y1), float(y2))
    ymax = max(float(y1), float(y2))
    keep = np.zeros(count, dtype=bool)
    for index in range(count):
        if (
            int(row0[index]) == -1
            and int(row1[index]) == -1
            and int(col0[index]) == -1
            and int(col1[index]) == -1
        ):
            continue
        if not screen:
            keep[index] = True
            continue
        left = float(col0[index]) - 0.5
        right = float(col1[index]) + 0.5
        top = float(row0[index]) - 0.5
        bottom = float(row1[index]) + 0.5
        if xmax >= left and xmin < right and ymax >= top and ymin < bottom:
            keep[index] = True
    return keep


def _reference_profile(
    x1, y1, x2, y2, src_x, src_y, src_flux, row0, row1, col0, col1, psf, n_profile, screen,
):
    """Continuous sum with a different sample formula and per-source evaluate.

    Samples are ``t = k / (n - 1)`` and ``sx = x1 + t * (x2 - x1)``, not the
    production ``linspace`` times a unit vector. Each kept source calls
    ``psf.evaluate`` once on the whole sample vector.
    """
    count = int(n_profile)
    fraction = np.arange(count, dtype=np.float64) / (count - 1)
    sample_x = float(x1) + fraction * (float(x2) - float(x1))
    sample_y = float(y1) + fraction * (float(y2) - float(y1))
    keep = _reference_sources_on_segment(x1, y1, x2, y2, row0, row1, col0, col1, screen)
    profile = np.zeros(count, dtype=np.float64)
    for index in np.flatnonzero(keep):
        brightness = psf.evaluate(sample_x, sample_y, float(src_x[index]), float(src_y[index]))
        profile += float(src_flux[index]) * np.asarray(brightness, dtype=np.float64)
    return profile


def _assert_profiles_close(produced: np.ndarray, reference: np.ndarray) -> None:
    peak = max(float(np.max(produced)), float(np.max(reference)))
    assert float(np.max(np.abs(produced - reference))) <= 1e-12 * peak


def _pixel_contributions(sources, psf, sensor, row: int, col: int) -> np.ndarray:
    """Scaled pixel integrals of every source at one pixel. Sky is not included."""
    kernel = _source_kernel(sources, psf)
    sigma_r, sigma_t, radial_x, radial_y, inv_core, inv_halo, inv_alpha2 = kernel
    scale = float(sensor.vignetting[row, col] * sensor.prnu[row, col])
    values = np.zeros(sources.x.shape[0], dtype=np.float64)
    for index in range(values.shape[0]):
        if sources.box_row0[index] < 0:
            continue
        if not (
            sources.box_row0[index] <= row <= sources.box_row1[index]
            and sources.box_col0[index] <= col <= sources.box_col1[index]
        ):
            continue
        values[index] = integrate_pixel(
            row,
            col,
            float(sources.x[index]),
            float(sources.y[index]),
            float(sources.flux_e[index]),
            float(sigma_r[index]),
            float(sigma_t[index]),
            float(radial_x[index]),
            float(radial_y[index]),
            float(inv_core[index]),
            float(inv_halo[index]),
            float(inv_alpha2[index]),
            float(MOFFAT_BETA),
        ) * scale
    return values


def _remap_owners(owner: np.ndarray, order: np.ndarray) -> np.ndarray:
    mapped = np.array(owner, copy=True)
    known = mapped >= 0
    mapped[known] = order[mapped[known]]
    return mapped


def _assert_adc_realized(exposure: Exposure, sensor) -> None:
    """Realized ADC flags are the noisy electrons, and disagreements stay inside 6 sigma."""
    ceiling = (1 << int(sensor.adc_bits)) - 1
    adu_float = electron_to_adu(exposure.electrons, sensor)
    assert np.array_equal(exposure.adc_clipped_realized, adu_float > (ceiling - 0.5))
    expected = electron_to_adu(
        exposure.mean_e + sensor.row_offset_e[:, None] + sensor.col_offset_e[None, :],
        sensor,
    )
    disagree = exposure.adc_clipped_realized != exposure.adc_mask
    if not np.any(disagree):
        return
    sigma_tot = np.sqrt(exposure.mean_e + sensor.read_e * sensor.read_e) / sensor.gain_e_per_adu
    distance = np.abs(expected - (ceiling - 0.5))
    assert np.all(distance[disagree] <= 6.0 * sigma_tot[disagree])


def _autocorrelation_length(field: np.ndarray, pixel_scale: float = 1.0) -> float:
    """1/e lag of the linear autocorrelation, in pixels.

    The transform is zero-padded, so lags do not wrap. Each lag is divided
    by how many samples overlap there, then by the zero-lag value.
    ``pixel_scale`` is the pixel width of one array step.
    """
    values = np.asarray(field, dtype=np.float64)
    height, width = values.shape
    fft_shape = (2 * height - 1, 2 * width - 1)
    spectrum = np.fft.rfft2(values, s=fft_shape)
    acf = np.fft.irfft2(spectrum * np.conjugate(spectrum), s=fft_shape)
    lag_y = np.arange(fft_shape[0])
    lag_x = np.arange(fft_shape[1])
    lag_y = np.where(lag_y <= height - 1, lag_y, lag_y - fft_shape[0]).astype(np.float64)
    lag_x = np.where(lag_x <= width - 1, lag_x, lag_x - fft_shape[1]).astype(np.float64)
    overlap = (height - np.abs(lag_y))[:, None] * (width - np.abs(lag_x))[None, :]
    acf = acf / overlap
    acf = acf / acf[0, 0]
    radius = np.hypot(lag_y[:, None], lag_x[None, :])
    max_lag = int(min(height, width) // 2)
    bins = np.rint(radius).astype(np.int32)
    keep = (bins >= 0) & (bins < max_lag)
    sums = np.bincount(bins[keep].ravel(), weights=acf[keep].ravel(), minlength=max_lag)
    counts = np.bincount(bins[keep].ravel(), minlength=max_lag)
    radial = np.full(max_lag, np.nan)
    good = counts > 0
    radial[good] = sums[good] / counts[good]
    target = math.exp(-1.0)
    below = np.flatnonzero(np.isfinite(radial) & (radial <= target))
    if below.size == 0:
        return float(max_lag) * pixel_scale
    hit = int(below[0])
    if hit == 0:
        return 0.0
    left = float(radial[hit - 1])
    right = float(radial[hit])
    if left == right:
        lag = float(hit)
    else:
        lag = (hit - 1) + (left - target) / (left - right)
    return lag * pixel_scale


def test_clipped_boxes_and_the_shared_integral_match_the_renderer():
    psf = _psf(sigma0=1.1, halo_fraction=0.02, width=48, height=40, cx=20.0, cy=15.0)
    sensor = _flat_sensor(height=40, width=48)
    x = np.array([12.3, 30.0])
    y = np.array([8.7, 22.2])
    flux = np.array([1000.0, 50.0])
    sources = _paint_stars(sensor, psf, x, y, flux, np.array([6.0, 9.0]))
    row0, row1, col0, col1 = clipped_stamp_boxes(x, y, sources.radius, 40, 48)
    assert np.array_equal(row0, sources.box_row0)
    assert np.array_equal(row1, sources.box_row1)
    assert np.array_equal(col0, sources.box_col0)
    assert np.array_equal(col1, sources.box_col1)
    from starsim.noise import _source_kernel
    from starsim.psf import MOFFAT_BETA

    kernel = _source_kernel(sources, psf)
    row = int(sources.peak_row[0])
    col = int(sources.peak_col[0])
    value = integrate_pixel(
        row,
        col,
        float(x[0]),
        float(y[0]),
        float(flux[0]),
        float(kernel[0][0]),
        float(kernel[1][0]),
        float(kernel[2][0]),
        float(kernel[3][0]),
        float(kernel[4][0]),
        float(kernel[5][0]),
        float(kernel[6][0]),
        float(MOFFAT_BETA),
    )
    assert value == sources.image[row, col]
