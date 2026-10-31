"""Background, sensor fingerprint, and the electron-to-ADU noise chain.

The chain follows plan section 4.5 (D36), in electron units until the last step:

1. Build the unclipped mean M₁: stars, sky (already multiplied by vignetting
   and PRNU), dark current, this frame's hot-pixel level, and cosmic rays.
   Dead pixels still hold their charge here.
2. Full-well clip and column blooming on M₁, producing M₂. Record the
   deterministic full-well mask, owners, edge loss, dead-pixel loss, and
   clip loss. Nothing in M₂ exceeds full well, and dead pixels are then zero.
3. One Poisson draw of M₂, called Q.
4. Hard-clip Q at full well, called Q′. This step does not bloom and does
   not change the label mask or the owners.
5. Add read noise and the fixed row/column offsets, divide by gain, add
   bias, round, and clip to the ADC.

Given M₁, step 2 does not depend on the Poisson draw or the read noise.
Hot pixels and cosmic rays join M₁, so their shot noise is that one Poisson
draw and not a second sample.

Hot pixels, dead pixels, PRNU, DSNU, and the row/column pattern are drawn
once per sequence from named streams on ``SequenceRng``. RTS pixels are a
two-level Markov chain: the two levels are fixed for the sequence, and each
later frame flips with that pixel's ``p_flip``. They are not redrawn
independently every frame. Non-RTS hot pixels keep one level; only Poisson
statistics move the observed value.

``σ_ref`` is the noise at the principal point of the optical background
(after vignetting and PRNU) plus dark current and read noise. Photometry
anchors on that number.

DSNU is a zero-mean Gaussian relative perturbation of the mean dark with
standard deviation ``DSNU_RELATIVE_SIGMA``. The plan names DSNU but does not
give it a configuration field, so the relative width is a fixed model
constant. Hot-pixel amplitudes are lognormal with median
``HOT_LOG_MEDIAN_E`` electrons; the plan specifies the family and not the
parameters. Cosmic-ray tracks are half point-like and half random-walk
worms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates

from numba import njit

from starsim.camera import Camera
from starsim.photometry import local_sigma, peak_snr
from starsim.psf import MOFFAT_ALPHA_PER_SIGMA, MOFFAT_BETA, PsfModel
from starsim.render import RenderedSources, RenderedTargets, integrate_pixel, render_static_stars
from starsim.rng import SequenceRng

# Behavior version for the noise chain. Increment when a legal sequence
# starts producing different electron images from the same source image.
# 1 allowed hot-pixel clusters to touch and form a larger component.
# 2 keeps a one-pixel gap, including diagonally, so each cluster stays 2–3 px.
# 3 blooms every over-full pixel on the unclipped mean, clips once after the
#   Poisson draw, draws the smooth field as filtered white noise, and places
#   hot pixels clusters-first so stable and RTS pixels stay isolated.
# 4 draws the smooth field on a padded coarse grid and bicubic-upsamples it,
#   so the image is not a circular period of the filter.
NOISE_VERSION = 4

# Relative std of dark current. Not a YAML field; see the module docstring.
DSNU_RELATIVE_SIGMA = 0.1

# Lognormal hot-pixel amplitude: median 1000 e−, natural-log std 0.6.
HOT_LOG_MEDIAN_E = 1000.0
HOT_LOG_SIGMA = 0.6

KIND_STABLE = 0
KIND_CLUSTER = 1
KIND_RTS = 2

STREAM_GRADIENT = "noise/gradient_angle"
STREAM_SMOOTH_SCALE = "noise/smooth_scale"
STREAM_SMOOTH_FIELD = "noise/smooth_field"
STREAM_PRNU = "noise/prnu"
STREAM_DSNU = "noise/dsnu"
STREAM_HOT = "noise/hot_pixels"
STREAM_DEAD = "noise/dead_pixels"
STREAM_FPN = "noise/row_col_fpn"

_COSMIC_ENERGY = (500.0, 20000.0)
_COSMIC_LENGTH = (2.0, 30.0)

# Coarse cell size for the smooth field. Valid while L >= 500, since the
# generator requires this spacing to be at most L/20.
_SMOOTH_DOWNSAMPLE = 8
_SMOOTH_CUBIC_MARGIN = 3


@dataclass
class CosmicRay:
    """One cosmic-ray deposit.

    ``cells`` lists every ``(row, col)`` the track touched, in walk order.
    A revisit is stored again, because each visit received an equal share
    of ``energy_e``. ``n_steps`` counts those visits, including the start.
    ``path_length_px`` is the sum of the steps. ``endpoint_disp_px`` is the
    straight-line distance from the first cell to the last. ``n_unique_cells``
    collapses revisits.
    """

    energy_e: float
    row: int
    col: int
    pointlike: bool
    cells: tuple[tuple[int, int], ...]
    n_steps: int
    path_length_px: float
    endpoint_disp_px: float
    n_unique_cells: int


@dataclass
class SensorFingerprint:
    """Per-sequence sensor and background. Arrays are read by ``expose_frame``."""

    height: int
    width: int
    cx: float
    cy: float
    background_e: np.ndarray
    vignetting: np.ndarray
    prnu: np.ndarray
    dark_e: np.ndarray
    read_e: float
    gain_e_per_adu: float
    bias_adu: float
    adc_bits: int
    full_well_e: float
    blooming: bool
    row_offset_e: np.ndarray
    col_offset_e: np.ndarray
    hot_row: np.ndarray
    hot_col: np.ndarray
    hot_kind: np.ndarray
    hot_level_a: np.ndarray
    hot_level_b: np.ndarray
    hot_state: np.ndarray
    hot_p_flip: np.ndarray
    dead_row: np.ndarray
    dead_col: np.ndarray
    cosmic_rate: float
    sky_e: float
    n_frames: int
    exposure_s: float

    def hot_level(self, frame_index: int) -> np.ndarray:
        """Electron level of every hot pixel in this frame, before Poisson."""
        self._check_frame(frame_index)
        if self.hot_row.size == 0:
            return np.zeros(0, dtype=np.float64)
        state = self.hot_state[:, frame_index]
        return np.where(state == 0, self.hot_level_a, self.hot_level_b).astype(np.float64, copy=False)

    def hot_electron_image(self, frame_index: int) -> np.ndarray:
        image = np.zeros((self.height, self.width), dtype=np.float64)
        if self.hot_row.size == 0:
            return image
        image[self.hot_row, self.hot_col] = self.hot_level(frame_index)
        return image

    def reference_sigma(self) -> float:
        """Background noise at the principal point, in electrons.

        The optical term is bilinear in ``background · vignetting · PRNU``.
        Dark current at the same point and ``read²`` are added under the square root.
        """
        optical = self.background_e * self.vignetting * self.prnu
        center = _bilinear(optical, self.cx, self.cy) + _bilinear(self.dark_e, self.cx, self.cy)
        return math.sqrt(max(center, 0.0) + self.read_e * self.read_e)

    def _check_frame(self, frame_index: int) -> None:
        if type(frame_index) is not int or isinstance(frame_index, bool):
            raise TypeError(f"frame_index must be int, got {frame_index!r}")
        if frame_index < 0 or frame_index >= self.n_frames:
            raise ValueError(
                f"frame_index must satisfy 0 <= index < {self.n_frames}, got {frame_index}"
            )


@dataclass
class Exposure:
    """One noisy frame.

    ``expectation_e`` is the plan's ``E``: sky, structure, sources, vignetting,
    and PRNU. It does not include dark, hot pixels, or cosmic rays.
    ``sigma_local`` uses that ``E`` together with ``s_self``, dark, and read noise.
    ``mean_e`` is M₂, the post-bloom image Poisson sampling sees. No pixel
    of M₂ exceeds full well.
    ``electrons`` is Q′ plus read noise and the fixed row/column offsets,
    before gain. ``adu`` is the quantized image.
    ``saturation_mask`` is the label mask: full-well pixels or the expected
    ADC mask. ``saturation_owner`` is the source index of each of those
    pixels, or −1 when the defect candidate wins.     ``peak_clipped`` and
    ``peak_owner`` are the same facts at each source's peak pixel.
    Source-level arrays follow the frame's global index: static stars first,
    in catalog-index order, then moving targets in track-id order.
    ``n_static`` is the star count, so target ``i`` has index ``n_static + i``.
    ``center_clipped`` and ``center_owner`` are the same facts at each
    target's true-center pixel, and are empty when the frame has no targets.
    A center outside the image stays unclipped with owner −1.
    ``poisson_sum_e`` and ``q_prime_sum_e`` are ΣQ and ΣQ′, so
    ``poisson_sum_e == q_prime_sum_e + post_sample_clip_loss_e``.
    """

    expectation_e: np.ndarray
    mean_e: np.ndarray
    electrons: np.ndarray
    adu: np.ndarray
    s_self: np.ndarray
    sigma_local: np.ndarray
    snr: np.ndarray
    peak_clipped: np.ndarray
    peak_owner: np.ndarray
    saturation_mask: np.ndarray
    saturation_owner: np.ndarray
    fullwell_mask: np.ndarray
    adc_mask: np.ndarray
    adc_clipped_realized: np.ndarray
    adc_clipped_low: np.ndarray
    post_sample_over: np.ndarray
    bloom_edge_loss_e: float
    dead_loss_e: float
    fullwell_clip_loss_e: float
    post_sample_clip_loss_e: float
    poisson_sum_e: float
    q_prime_sum_e: float
    n_static: int
    center_clipped: np.ndarray
    center_owner: np.ndarray
    cosmic_rays: tuple[CosmicRay, ...] = field(default_factory=tuple)


def uniform_sensor(
    height: int,
    width: int,
    *,
    sky_e: float,
    dark_e: float,
    read_e: float,
    gain_e_per_adu: float,
    bias_adu: float,
    full_well_e: float,
    adc_bits: int,
    n_frames: int,
    blooming: bool = False,
    cosmic_rate: float = 0.0,
    exposure_s: float = 1.0,
) -> SensorFingerprint:
    """Spatially flat sensor with no defects. Used to test the noise formulas.

    PRNU and vignetting are 1, the background is ``sky_e`` everywhere, and
    dark current is ``dark_e`` everywhere. There are no hot pixels, dead
    pixels, or row/column offsets.
    """
    if type(height) is not int or type(width) is not int or height < 1 or width < 1:
        raise ValueError("height and width must be integers >= 1")
    if type(n_frames) is not int or n_frames < 1:
        raise ValueError(f"n_frames must be an int >= 1, got {n_frames!r}")
    if type(adc_bits) is not int or adc_bits not in (12, 14, 16):
        raise ValueError(f"adc_bits must be 12, 14, or 16, got {adc_bits!r}")
    if type(blooming) is not bool:
        raise TypeError("blooming must be bool")
    shape = (height, width)
    empty_i = np.zeros(0, dtype=np.int64)
    empty_f = np.zeros(0, dtype=np.float64)
    return SensorFingerprint(
        height=height,
        width=width,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        background_e=np.full(shape, float(sky_e), dtype=np.float64),
        vignetting=np.ones(shape, dtype=np.float64),
        prnu=np.ones(shape, dtype=np.float64),
        dark_e=np.full(shape, float(dark_e), dtype=np.float64),
        read_e=float(read_e),
        gain_e_per_adu=float(gain_e_per_adu),
        bias_adu=float(bias_adu),
        adc_bits=adc_bits,
        full_well_e=float(full_well_e),
        blooming=blooming,
        row_offset_e=np.zeros(height, dtype=np.float64),
        col_offset_e=np.zeros(width, dtype=np.float64),
        hot_row=empty_i,
        hot_col=empty_i.copy(),
        hot_kind=empty_i.copy(),
        hot_level_a=empty_f,
        hot_level_b=empty_f.copy(),
        hot_state=np.zeros((0, n_frames), dtype=np.uint8),
        hot_p_flip=empty_f.copy(),
        dead_row=empty_i.copy(),
        dead_col=empty_i.copy(),
        cosmic_rate=float(cosmic_rate),
        sky_e=float(sky_e),
        n_frames=n_frames,
        exposure_s=float(exposure_s),
    )


def build_sensor(effective: dict, camera: Camera, rngs: SequenceRng) -> SensorFingerprint:
    """Draw one sequence's fingerprint. Each noise stream is issued once.

    ``effective`` is the sampled generator config: interval fields that
    ``sample_sequence_params`` draws must already be scalars. Hot-pixel
    positions, RTS levels, and the smooth field still use their own streams.
    """
    if not isinstance(camera, Camera):
        raise TypeError("camera must be starsim.camera.Camera")
    if not isinstance(rngs, SequenceRng):
        raise TypeError("rngs must be starsim.rng.SequenceRng")
    if not isinstance(effective, dict):
        raise TypeError("effective config must be a mapping")
    height = int(camera.height)
    width = int(camera.width)
    sky = _scalar(effective, "background", "sky_e")
    gradient_frac = _scalar(effective, "background", "gradient_frac")
    smooth_frac = _scalar(effective, "background", "smooth_field_frac")
    vignetting_corner = _scalar(effective, "background", "vignetting_corner")
    read_e = _scalar(effective, "sensor_noise", "read_e")
    dark_per_s = _scalar(effective, "sensor_noise", "dark_e_per_s")
    prnu_sigma = _scalar(effective, "sensor_noise", "prnu")
    gain = _scalar(effective, "sensor_noise", "gain_e_per_adu")
    bias = _scalar(effective, "sensor_noise", "bias_adu")
    full_well = _scalar(effective, "sensor_noise", "full_well_e")
    fpn = _scalar(effective, "sensor_noise", "row_col_fpn_e")
    exposure_s = _scalar(effective, "timing", "exposure_s_nominal")
    hot_frac = _scalar(effective, "artifacts", "hot_pixel_frac")
    dead_frac = _scalar(effective, "artifacts", "dead_pixel_frac")
    cosmic_rate = _scalar(effective, "artifacts", "cosmic_rays_per_frame")
    cluster_frac = _scalar(effective, "artifacts", "hot_cluster_frac")
    rts_frac = _scalar(effective, "artifacts", "rts_frac")
    adc_bits = effective["sensor_noise"]["adc_bits"]
    blooming = effective["artifacts"]["blooming"]
    n_frames = effective["dataset"]["n_frames"]
    if type(adc_bits) is not int or adc_bits not in (12, 14, 16):
        raise ValueError(f"sensor_noise.adc_bits must be 12, 14, or 16, got {adc_bits!r}")
    if type(blooming) is not bool:
        raise TypeError("artifacts.blooming must be bool")
    if type(n_frames) is not int or n_frames < 1:
        raise ValueError(f"dataset.n_frames must be an int >= 1, got {n_frames!r}")
    if gain <= 0.0 or full_well <= 0.0 or exposure_s <= 0.0:
        raise ValueError("gain, full well, and exposure must be > 0")

    background, vignetting = _background(
        height,
        width,
        camera.cx,
        camera.cy,
        sky,
        gradient_frac,
        smooth_frac,
        vignetting_corner,
        rngs.get(STREAM_GRADIENT),
        rngs.get(STREAM_SMOOTH_SCALE),
        rngs.get(STREAM_SMOOTH_FIELD),
    )
    prnu = 1.0 + rngs.get(STREAM_PRNU).normal(0.0, prnu_sigma, size=(height, width))
    np.maximum(prnu, 0.05, out=prnu)
    dark_mean = dark_per_s * exposure_s
    dsnu = rngs.get(STREAM_DSNU).normal(0.0, DSNU_RELATIVE_SIGMA, size=(height, width))
    dark = np.maximum(dark_mean * (1.0 + dsnu), 0.0)
    fpn_rng = rngs.get(STREAM_FPN)
    row_offset = fpn_rng.normal(0.0, fpn, size=height)
    col_offset = fpn_rng.normal(0.0, fpn, size=width)
    dead_row, dead_col = _dead_pixels(height, width, dead_frac, rngs.get(STREAM_DEAD))
    hot = _hot_pixels(
        height,
        width,
        hot_frac,
        cluster_frac,
        rts_frac,
        _int_interval(effective, "artifacts", "hot_cluster_size_px"),
        _float_interval(effective, "artifacts", "rts_p_flip"),
        n_frames,
        dead_row,
        dead_col,
        rngs.get(STREAM_HOT),
    )
    return SensorFingerprint(
        height=height,
        width=width,
        cx=float(camera.cx),
        cy=float(camera.cy),
        background_e=background,
        vignetting=vignetting,
        prnu=prnu,
        dark_e=dark,
        read_e=read_e,
        gain_e_per_adu=gain,
        bias_adu=bias,
        adc_bits=adc_bits,
        full_well_e=full_well,
        blooming=blooming,
        row_offset_e=row_offset,
        col_offset_e=col_offset,
        hot_row=hot[0],
        hot_col=hot[1],
        hot_kind=hot[2],
        hot_level_a=hot[3],
        hot_level_b=hot[4],
        hot_state=hot[5],
        hot_p_flip=hot[6],
        dead_row=dead_row,
        dead_col=dead_col,
        cosmic_rate=cosmic_rate,
        sky_e=sky,
        n_frames=n_frames,
        exposure_s=exposure_s,
    )


def expose_frame(
    sources: RenderedSources,
    sensor: SensorFingerprint,
    frame_index: int,
    rng: np.random.Generator,
    psf: PsfModel | None = None,
    targets: RenderedTargets | None = None,
    cosmic_rate: float | None = None,
) -> Exposure:
    """Turn a linear source image into one noisy ADU frame.

    ``rng`` supplies cosmic rays, the one Poisson draw, and read noise, in
    that order. The sensor fingerprint is not redrawn. Pass a PSF whenever
    the frame can saturate and still has sources: star owners are the shared
    pixel integral, and target owners are read from each target's stamp.
    With no moving targets the electrons, mask, owners, and ADU match the
    static-star path bit for bit.

    ``cosmic_rate`` replaces ``sensor.cosmic_rate`` for this call only. The
    sensor object is not modified. ``None`` uses the rate on the sensor.
    The draw order is unchanged.
    """
    if not isinstance(sources, RenderedSources):
        raise TypeError("sources must be starsim.render.RenderedSources")
    if targets is not None and not isinstance(targets, RenderedTargets):
        raise TypeError("targets must be starsim.render.RenderedTargets or None")
    if not isinstance(sensor, SensorFingerprint):
        raise TypeError("sensor must be starsim.noise.SensorFingerprint")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    sensor._check_frame(frame_index)
    if sources.image.shape != (sensor.height, sensor.width):
        raise ValueError(
            f"source image shape {sources.image.shape} does not match the sensor "
            f"{(sensor.height, sensor.width)}"
        )
    moving = targets if targets is not None and targets.track_id.size else None
    if moving is not None:
        if moving.image.shape != sources.image.shape:
            raise ValueError(
                f"target image shape {moving.image.shape} does not match the stars "
                f"{sources.image.shape}"
            )
        linear = sources.image + moving.image
    else:
        linear = sources.image
    expectation = (sensor.background_e + linear) * sensor.vignetting * sensor.prnu
    cosmic_image, cosmic_rays = _cosmic_rays(sensor, rng, cosmic_rate)
    hot_image = sensor.hot_electron_image(frame_index)
    # Hot pixels and cosmic rays join M₁. Their shot noise is the single
    # Poisson draw below, not a second sample.
    m1 = expectation + sensor.dark_e + hot_image + cosmic_image
    defect = hot_image + cosmic_image
    m2, fullwell_mask, owner, edge_loss, dead_loss, clip_loss = _fullwell_step(
        m1, sensor, sources, psf, defect, moving
    )
    sampled = rng.poisson(m2).astype(np.float64)
    clipped = np.minimum(sampled, sensor.full_well_e)
    poisson_sum = float(np.sum(sampled))
    prime_sum = float(np.sum(clipped))
    post_loss = poisson_sum - prime_sum
    post_over = (sampled > sensor.full_well_e) & ~fullwell_mask
    electrons = clipped + rng.normal(0.0, sensor.read_e, size=m2.shape)
    electrons += sensor.row_offset_e[:, None]
    electrons += sensor.col_offset_e[None, :]
    adu_float = electron_to_adu(electrons, sensor)
    ceiling = (1 << sensor.adc_bits) - 1
    realized = adu_float > (ceiling - 0.5)
    low = adu_float < -0.5
    adu = np.clip(np.rint(adu_float), 0, ceiling).astype(np.uint16)
    expected = electron_to_adu(
        m2 + sensor.row_offset_e[:, None] + sensor.col_offset_e[None, :],
        sensor,
    )
    adc_mask = expected > (ceiling - 0.5)
    adc_only = adc_mask & ~fullwell_mask
    if np.any(adc_only):
        _attribute_mask(owner, adc_only, sources, psf, sensor, defect, moving)
    sat_mask = fullwell_mask | adc_mask
    s_self, sigma, snr, peak_clipped, peak_owner = _source_metrics(
        sources, sensor, expectation, sat_mask, owner
    )
    n_static = int(sources.x.shape[0])
    if moving is None:
        center_clipped = np.zeros(0, dtype=bool)
        center_owner = np.zeros(0, dtype=np.int32)
    else:
        target_s, target_sigma, target_snr, target_clipped, target_owner = _metrics_at(
            moving.peak_row,
            moving.peak_col,
            moving.s_self_linear,
            sensor,
            expectation,
            sat_mask,
            owner,
        )
        s_self = np.concatenate((s_self, target_s))
        sigma = np.concatenate((sigma, target_sigma))
        snr = np.concatenate((snr, target_snr))
        peak_clipped = np.concatenate((peak_clipped, target_clipped))
        peak_owner = np.concatenate((peak_owner, target_owner))
        center_clipped, center_owner = _metrics_flags(
            moving.center_row, moving.center_col, sat_mask, owner
        )
    return Exposure(
        expectation_e=expectation,
        mean_e=m2,
        electrons=electrons,
        adu=adu,
        s_self=s_self,
        sigma_local=sigma,
        snr=snr,
        peak_clipped=peak_clipped,
        peak_owner=peak_owner,
        saturation_mask=sat_mask,
        saturation_owner=owner,
        fullwell_mask=fullwell_mask,
        adc_mask=adc_mask,
        adc_clipped_realized=realized,
        adc_clipped_low=low,
        post_sample_over=post_over,
        bloom_edge_loss_e=edge_loss,
        dead_loss_e=dead_loss,
        fullwell_clip_loss_e=clip_loss,
        post_sample_clip_loss_e=post_loss,
        poisson_sum_e=poisson_sum,
        q_prime_sum_e=prime_sum,
        n_static=n_static,
        center_clipped=center_clipped,
        center_owner=center_owner,
        cosmic_rays=tuple(cosmic_rays),
    )


def expose_catalog_frame(
    catalog,
    camera: Camera,
    attitude: np.ndarray,
    psf: PsfModel,
    photometry,
    sensor: SensorFingerprint,
    frame_index: int,
    rng: np.random.Generator,
    margin_px: float = 64.0,
) -> tuple[RenderedSources, Exposure]:
    """Project one catalog field and expose it with this sensor."""
    sources = render_static_stars(
        catalog,
        camera,
        attitude,
        photometry,
        psf,
        sensor.reference_sigma(),
        margin_px=margin_px,
    )
    return sources, expose_frame(sources, sensor, frame_index, rng, psf)


def electron_to_adu(electrons: np.ndarray, sensor: SensorFingerprint) -> np.ndarray:
    """Electrons to floating-point ADU, before rounding and the ADC clip."""
    return np.asarray(electrons, dtype=np.float64) / sensor.gain_e_per_adu + sensor.bias_adu


def adu_is_high_clipped(adu_float: np.ndarray | float, bits: int) -> np.ndarray:
    """True when rounding would land on the ADC ceiling.

    ``np.rint`` is half-to-even, so a value of ``ceiling - 0.5`` rounds down
    and is not clipped. The next representable value rounds up.
    """
    if type(bits) is not int or bits not in (12, 14, 16):
        raise ValueError(f"bits must be 12, 14, or 16, got {bits!r}")
    ceiling = (1 << bits) - 1
    return np.asarray(adu_float, dtype=np.float64) > (ceiling - 0.5)


def quantize(electrons: np.ndarray, sensor: SensorFingerprint) -> np.ndarray:
    """Electrons to clipped integer ADU."""
    ceiling = (1 << sensor.adc_bits) - 1
    return np.clip(np.rint(electron_to_adu(electrons, sensor)), 0, ceiling).astype(np.uint16)


def _source_metrics(sources, sensor, expectation, sat_mask, owner):
    return _metrics_at(
        sources.peak_row,
        sources.peak_col,
        sources.s_self_linear,
        sensor,
        expectation,
        sat_mask,
        owner,
    )


def _metrics_at(row, col, linear, sensor, expectation, sat_mask, owner):
    count = int(np.asarray(row).shape[0])
    s_self = np.zeros(count, dtype=np.float64)
    sigma = np.zeros(count, dtype=np.float64)
    snr = np.zeros(count, dtype=np.float64)
    clipped = np.zeros(count, dtype=bool)
    who = np.full(count, -1, dtype=np.int32)
    for index in range(count):
        peak_row = int(row[index])
        peak_col = int(col[index])
        if peak_row < 0 or peak_col < 0:
            continue
        scale = float(sensor.vignetting[peak_row, peak_col] * sensor.prnu[peak_row, peak_col])
        signal = float(linear[index]) * scale
        s_self[index] = signal
        sigma[index] = local_sigma(
            float(expectation[peak_row, peak_col]),
            signal,
            float(sensor.dark_e[peak_row, peak_col]),
            sensor.read_e,
        )
        snr[index] = peak_snr(signal, sigma[index])
        clipped[index] = bool(sat_mask[peak_row, peak_col])
        who[index] = np.int32(owner[peak_row, peak_col])
    return s_self, sigma, snr, clipped, who


def _metrics_flags(row, col, sat_mask, owner):
    count = int(np.asarray(row).shape[0])
    clipped = np.zeros(count, dtype=bool)
    who = np.full(count, -1, dtype=np.int32)
    for index in range(count):
        peak_row = int(row[index])
        peak_col = int(col[index])
        if peak_row < 0 or peak_col < 0:
            continue
        clipped[index] = bool(sat_mask[peak_row, peak_col])
        who[index] = np.int32(owner[peak_row, peak_col])
    return clipped, who


def _fullwell_step(m1, sensor, sources, psf, defect, targets=None):
    """Clip and, when enabled, bloom M₁. Dead pixels are zeroed at the end.

    Returns M₂, the deterministic full-well mask, the owner image, and the
    three step-2 losses. Owners of over-full pixels are assigned before the
    walk, because a segment's owner is the one with the most excess inside it.
    """
    full_well = float(sensor.full_well_e)
    over = m1 > full_well
    owner = np.full(m1.shape, -1, dtype=np.int32)
    best = np.array(defect, dtype=np.float64, copy=True)
    if np.any(over):
        _attribute_mask(owner, over, sources, psf, sensor, best, targets)
    if sensor.blooming:
        m2, fullwell_mask, edge_loss = _bloom_columns(m1, over, owner, sensor, full_well)
        clip_loss = 0.0
    else:
        m2 = np.minimum(m1, full_well)
        fullwell_mask = over
        edge_loss = 0.0
        clip_loss = float(np.sum(m1[over] - full_well)) if np.any(over) else 0.0
    dead_loss = 0.0
    if sensor.dead_row.size:
        dead_loss = float(np.sum(m2[sensor.dead_row, sensor.dead_col]))
        m2[sensor.dead_row, sensor.dead_col] = 0.0
    return m2, fullwell_mask, owner, edge_loss, dead_loss, clip_loss


def _attribute_mask(owner, mask, sources, psf, sensor, best, targets=None) -> None:
    """Write the winning candidate into ``owner`` where ``mask`` is set.

    ``best`` holds the running contribution and starts as the defect image.
    A source replaces it only when its scaled contribution is strictly
    larger, so an equal contribution keeps the smaller index. The defect
    index is −1. Stars are judged first, from the shared pixel integral.
    Targets are judged after that, from the stamp that was added to the
    image, and their index starts at the star count.
    """
    if not np.any(mask):
        return
    if sources.x.size:
        if not isinstance(psf, PsfModel):
            raise TypeError("a PsfModel is required to attribute saturated pixels")
        kernel = _source_kernel(sources, psf)
        _assign_owners(
            np.ascontiguousarray(mask),
            owner,
            best,
            np.ascontiguousarray(sources.box_row0),
            np.ascontiguousarray(sources.box_row1),
            np.ascontiguousarray(sources.box_col0),
            np.ascontiguousarray(sources.box_col1),
            np.ascontiguousarray(sources.x),
            np.ascontiguousarray(sources.y),
            np.ascontiguousarray(sources.flux_e),
            *kernel,
            float(MOFFAT_BETA),
            np.ascontiguousarray(sensor.vignetting),
            np.ascontiguousarray(sensor.prnu),
        )
    if targets is None or targets.track_id.size == 0:
        return
    offset = int(sources.x.shape[0])
    vig = np.ascontiguousarray(sensor.vignetting)
    prnu = np.ascontiguousarray(sensor.prnu)
    mask_c = np.ascontiguousarray(mask)
    for index in range(int(targets.track_id.size)):
        row0 = int(targets.box_row0[index])
        if row0 < 0:
            continue
        _assign_one_stamp(
            mask_c,
            owner,
            best,
            np.ascontiguousarray(targets.stamps[index]),
            row0,
            int(targets.box_col0[index]),
            np.int32(offset + index),
            vig,
            prnu,
        )


def _source_kernel(sources, psf: PsfModel):
    count = int(sources.x.shape[0])
    sigma_r = np.empty(count, dtype=np.float64)
    sigma_t = np.empty(count, dtype=np.float64)
    radial_x = np.empty(count, dtype=np.float64)
    radial_y = np.empty(count, dtype=np.float64)
    inv_core = np.empty(count, dtype=np.float64)
    inv_halo = np.empty(count, dtype=np.float64)
    inv_alpha2 = np.empty(count, dtype=np.float64)
    halo = float(psf.halo_fraction)
    beta = float(MOFFAT_BETA)
    for index in range(count):
        src_r, src_t, sigma_iso, rad_x, rad_y = psf.axes(float(sources.x[index]), float(sources.y[index]))
        alpha = float(MOFFAT_ALPHA_PER_SIGMA) * float(sigma_iso)
        sigma_r[index] = src_r
        sigma_t[index] = src_t
        radial_x[index] = rad_x
        radial_y[index] = rad_y
        inv_core[index] = (1.0 - halo) / (2.0 * math.pi * src_r * src_t)
        inv_halo[index] = halo * (beta - 1.0) / (math.pi * alpha * alpha)
        inv_alpha2[index] = 1.0 / (alpha * alpha)
    return sigma_r, sigma_t, radial_x, radial_y, inv_core, inv_halo, inv_alpha2


def _bloom_columns(m1, over, owner, sensor, full_well):
    """Walk each column's over-full runs. Charge that leaves the sensor is counted."""
    m2 = np.array(m1, dtype=np.float64, copy=True)
    fullwell_mask = np.array(over, dtype=bool, copy=True)
    dead = np.zeros(m1.shape, dtype=bool)
    if sensor.dead_row.size:
        dead[sensor.dead_row, sensor.dead_col] = True
    edge_loss = 0.0
    columns = np.flatnonzero(np.any(over, axis=0))
    for col in columns.tolist():
        column = m2[:, col]
        dead_col = dead[:, col]
        received_amt = np.zeros(column.shape[0], dtype=np.float64)
        received_owner = np.full(column.shape[0], -1, dtype=np.int32)
        for start, end in _true_runs(over[:, col]):
            excess = 0.0
            tally: dict[int, float] = {}
            for row in range(start, end + 1):
                extra = float(column[row] - full_well)
                excess += extra
                who = int(owner[row, col])
                tally[who] = tally.get(who, 0.0) + extra
                column[row] = full_well
            seg_owner = _segment_owner(tally)
            upward = excess * 0.5
            downward = excess - upward
            edge_loss += _push(column, dead_col, start - 1, -1, upward, full_well, seg_owner, received_amt, received_owner)
            edge_loss += _push(column, dead_col, end + 1, 1, downward, full_well, seg_owner, received_amt, received_owner)
        got = received_amt > 0.0
        if np.any(got):
            fullwell_mask[got, col] = True
            owner[got, col] = received_owner[got]
    return m2, fullwell_mask, edge_loss


def _segment_owner(tally: dict[int, float]) -> int:
    """Owner with the most excess. An equal total keeps the smaller index."""
    chosen = -1
    best = -1.0
    for who, total in tally.items():
        if total > best or (total == best and who < chosen):
            best = total
            chosen = who
    return chosen


def _true_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    index = np.flatnonzero(flags)
    if index.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(index) > 1) + 1
    groups = np.split(index, cuts)
    return [(int(group[0]), int(group[-1])) for group in groups]


def _push(column, dead_col, row, step, charge, full_well, seg_owner, received_amt, received_owner) -> float:
    """Fill outward from ``row``. Charge that walks off the sensor is returned.

    Dead pixels and pixels already at full well are skipped. The segment that
    deposits the most into a pixel owns it; an equal deposit keeps the smaller
    source index.
    """
    height = int(column.shape[0])
    while charge > 0.0 and 0 <= row < height:
        if dead_col[row] or column[row] >= full_well:
            row += step
            continue
        room = full_well - float(column[row])
        if charge >= room:
            column[row] = full_well
            charge -= room
            take = room
        else:
            column[row] = float(column[row]) + charge
            take = charge
            charge = 0.0
        if take > received_amt[row] or (take == received_amt[row] and seg_owner < int(received_owner[row])):
            received_amt[row] = take
            received_owner[row] = seg_owner
        if charge > 0.0:
            row += step
    if row < 0 or row >= height:
        return float(charge)
    return 0.0


@njit(cache=True)
def _assign_owners(
    mask,
    owner,
    best,
    row0,
    row1,
    col0,
    col1,
    x,
    y,
    flux,
    sigma_r,
    sigma_t,
    radial_x,
    radial_y,
    inv_core,
    inv_halo,
    inv_alpha2,
    beta,
    vig,
    prnu,
):
    count = x.shape[0]
    for index in range(count):
        if row0[index] < 0:
            continue
        for row in range(row0[index], row1[index] + 1):
            for col in range(col0[index], col1[index] + 1):
                if not mask[row, col]:
                    continue
                value = integrate_pixel(
                    row,
                    col,
                    x[index],
                    y[index],
                    flux[index],
                    sigma_r[index],
                    sigma_t[index],
                    radial_x[index],
                    radial_y[index],
                    inv_core[index],
                    inv_halo[index],
                    inv_alpha2[index],
                    beta,
                )
                value *= vig[row, col] * prnu[row, col]
                if value > best[row, col]:
                    best[row, col] = value
                    owner[row, col] = index


@njit(cache=True)
def _assign_one_stamp(mask, owner, best, stamp, row0, col0, global_index, vig, prnu):
    """Compete one target stamp. A tie keeps the smaller index already stored."""
    height, width = stamp.shape
    for local_row in range(height):
        row = row0 + local_row
        for local_col in range(width):
            col = col0 + local_col
            if not mask[row, col]:
                continue
            value = stamp[local_row, local_col] * vig[row, col] * prnu[row, col]
            if value > best[row, col]:
                best[row, col] = value
                owner[row, col] = global_index


def _background(height, width, cx, cy, sky, gradient_frac, smooth_frac, vignetting_corner, rng_angle, rng_scale, rng_field):
    yy, xx = np.indices((height, width), dtype=np.float64)
    dx = xx - float(cx)
    dy = yy - float(cy)
    r_max = 0.5 * math.hypot(width, height)
    angle = float(rng_angle.uniform(0.0, 2.0 * math.pi))
    projection = (dx * math.cos(angle) + dy * math.sin(angle)) / (2.0 * r_max)
    gradient = sky * gradient_frac * projection
    smooth = _smooth_field(height, width, sky * smooth_frac, rng_scale, rng_field)
    background = np.maximum(sky + gradient + smooth, 0.0)
    vig = 1.0 - vignetting_corner * (dx * dx + dy * dy) / (r_max * r_max)
    np.maximum(vig, 0.0, out=vig)
    return background, vig


def _smooth_field(height, width, amplitude, rng_scale, rng_field) -> np.ndarray:
    """Smooth background of standard deviation ``amplitude``.

    The length L is uniform on [500, 2000] px. On an infinite plane the
    autocorrelation would be ``exp(−r²/L²)``; one finite image, after the
    mean is removed, does not measure L. See ``_smooth_field_with_length``.
    """
    length = float(rng_scale.uniform(500.0, 2000.0))
    return _smooth_field_with_length(height, width, amplitude, length, rng_field)


def _smooth_pad_cells(length: float) -> int:
    """Coarse cells of padding on each side, at least ``4 σ_k / f``."""
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError(f"smooth-field length must be finite and > 0, got {length}")
    if _SMOOTH_DOWNSAMPLE > length / 20.0:
        raise ValueError(
            f"smooth-field downsample {_SMOOTH_DOWNSAMPLE} px exceeds L/20 ({length / 20.0:g})"
        )
    sigma_k = length / 2.0
    cells = int(math.ceil(4.0 * sigma_k / _SMOOTH_DOWNSAMPLE))
    if cells < _SMOOTH_CUBIC_MARGIN:
        cells = _SMOOTH_CUBIC_MARGIN
    return cells


def _smooth_cover_count(n_fine: int) -> int:
    """Coarse samples whose spacing covers fine pixels ``0 .. n_fine − 1``."""
    if n_fine < 1:
        raise ValueError(f"image axis must be >= 1, got {n_fine}")
    return int(math.ceil((n_fine - 1) / _SMOOTH_DOWNSAMPLE)) + 1


def _smooth_coarse_canvas(height, width, length, rng) -> np.ndarray:
    """Filtered white noise on the padded coarse canvas, before the crop.

    Each cell is ``_SMOOTH_DOWNSAMPLE`` pixels. The Gaussian kernel has
    standard deviation ``(L/2) / f`` in these cells, so the autocorrelation
    on an infinite plane is ``exp(−r²/L²)`` with ``r`` in pixels. The
    production image crops the interior of this canvas.
    """
    pad = _smooth_pad_cells(float(length))
    cover_h = _smooth_cover_count(int(height))
    cover_w = _smooth_cover_count(int(width))
    shape = (pad + cover_h + pad, pad + cover_w + pad)
    white = rng.normal(0.0, 1.0, size=shape)
    sigma_cells = (float(length) / 2.0) / _SMOOTH_DOWNSAMPLE
    return _gaussian_filter_fft(white, sigma_cells)


def _smooth_field_with_length(height, width, amplitude, length, rng) -> np.ndarray:
    """One smooth image of correlation length ``length`` and std ``amplitude``.

    White noise is drawn on a coarse grid, padded by at least four kernel
    standard deviations, and filtered there. The crop that covers the image
    is bicubic-upsampled. The result is mean-subtracted, scaled to unit
    standard deviation, and multiplied by ``amplitude``.
    """
    canvas = _smooth_coarse_canvas(height, width, length, rng)
    if amplitude == 0.0:
        return np.zeros((int(height), int(width)), dtype=np.float64)
    return _resample_smooth_canvas(canvas, int(height), int(width), float(length), float(amplitude))


def _resample_smooth_canvas(canvas, height, width, length, amplitude) -> np.ndarray:
    """Crop ``canvas`` and bicubic-upsample it onto the ``height × width`` image."""
    pad = _smooth_pad_cells(length)
    margin = _SMOOTH_CUBIC_MARGIN
    cover_h = _smooth_cover_count(height)
    cover_w = _smooth_cover_count(width)
    cropped = canvas[pad - margin : pad + cover_h + margin, pad - margin : pad + cover_w + margin]
    row = np.arange(height, dtype=np.float64) / _SMOOTH_DOWNSAMPLE + margin
    col = np.arange(width, dtype=np.float64) / _SMOOTH_DOWNSAMPLE + margin
    grid_row, grid_col = np.meshgrid(row, col, indexing="ij")
    image = map_coordinates(cropped, np.stack((grid_row, grid_col)), order=3, prefilter=True)
    image = np.asarray(image, dtype=np.float64)
    image -= float(image.mean())
    std = float(image.std())
    if std > 0.0:
        image /= std
    return image * amplitude


def _gaussian_filter_fft(white: np.ndarray, sigma: float) -> np.ndarray:
    """Circular Gaussian filter. ``sigma`` is the kernel standard deviation."""
    if sigma <= 0.0:
        return np.array(white, dtype=np.float64, copy=True)
    height, width = white.shape
    freq_y = np.fft.fftfreq(height)[:, None]
    freq_x = np.fft.rfftfreq(width)[None, :]
    transfer = np.exp(-2.0 * (math.pi * sigma) ** 2 * (freq_y * freq_y + freq_x * freq_x))
    spectrum = np.fft.rfft2(white) * transfer
    return np.fft.irfft2(spectrum, s=(height, width))


def _dead_pixels(height, width, fraction, rng):
    n_pix = height * width
    count = int(rng.binomial(n_pix, fraction))
    if count == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    flat = rng.choice(n_pix, size=count, replace=False)
    return np.divmod(flat, width)[0].astype(np.int64), (flat % width).astype(np.int64)


def _hot_pixels(height, width, fraction, cluster_frac, rts_frac, cluster_size, p_flip, n_frames, dead_row, dead_col, rng):
    n_pix = height * width
    n_hot = int(rng.binomial(n_pix, fraction))
    empty = _empty_hot(n_frames)
    if n_hot == 0:
        return empty
    n_rts = int(round(rts_frac * n_hot))
    n_cluster = int(round(cluster_frac * n_hot))
    if n_rts + n_cluster > n_hot:
        n_cluster = max(0, n_hot - n_rts)
    n_stable = n_hot - n_rts - n_cluster
    occupied = np.zeros(n_pix, dtype=bool)
    if dead_row.size:
        occupied[dead_row * width + dead_col] = True
    # Clusters first, then stable and RTS. The later pixels stay off the
    # 8-neighborhood of every defect already placed, including dead pixels.
    clusters = _place_clusters(occupied, n_cluster, cluster_size, height, width, rng)
    if dead_row.size:
        for flat in (dead_row * width + dead_col).tolist():
            _block_halo(occupied, int(flat), height, width)
    stable = _take_isolated(occupied, n_stable, height, width, rng)
    rts = _take_isolated(occupied, n_rts, height, width, rng)
    # A cluster that could not be grown is not left as a hole in the count:
    # those pixels were never taken. The realized cluster share is then short
    # by at most one cluster.
    rows = []
    cols = []
    kinds = []
    level_a = []
    level_b = []
    flips = []
    states = []
    _append_simple(stable, width, KIND_STABLE, n_frames, rng, rows, cols, kinds, level_a, level_b, flips, states)
    _append_rts(rts, width, n_frames, p_flip, rng, rows, cols, kinds, level_a, level_b, flips, states)
    _append_simple(clusters, width, KIND_CLUSTER, n_frames, rng, rows, cols, kinds, level_a, level_b, flips, states)
    if not rows:
        return empty
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(kinds, dtype=np.int64),
        np.asarray(level_a, dtype=np.float64),
        np.asarray(level_b, dtype=np.float64),
        np.asarray(states, dtype=np.uint8),
        np.asarray(flips, dtype=np.float64),
    )


def _empty_hot(n_frames: int):
    empty_i = np.zeros(0, dtype=np.int64)
    empty_f = np.zeros(0, dtype=np.float64)
    return (
        empty_i,
        empty_i.copy(),
        empty_i.copy(),
        empty_f,
        empty_f.copy(),
        np.zeros((0, n_frames), dtype=np.uint8),
        empty_f.copy(),
    )


def _reject_flat(occupied: np.ndarray, rng, n_pix: int) -> int:
    """Rejection sample one free pixel. Occupancy is tiny, so this does not scan the sensor."""
    for _try in range(64):
        flat = int(rng.integers(0, n_pix))
        if not occupied[flat]:
            return flat
    raise ValueError("not enough free pixels for hot-pixel placement")


def _block_halo(occupied: np.ndarray, flat: int, height: int, width: int) -> None:
    row, col = divmod(int(flat), width)
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            n_row = row + d_row
            n_col = col + d_col
            if 0 <= n_row < height and 0 <= n_col < width:
                occupied[n_row * width + n_col] = True


def _take_isolated(occupied: np.ndarray, count: int, height: int, width: int, rng) -> np.ndarray:
    """Place ``count`` pixels, blocking each one's 8-neighborhood afterwards."""
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    chosen = np.empty(count, dtype=np.int64)
    n_pix = int(occupied.shape[0])
    for index in range(count):
        flat = _reject_flat(occupied, rng, n_pix)
        occupied[flat] = True
        chosen[index] = flat
        _block_halo(occupied, flat, height, width)
    return chosen


def _place_clusters(occupied, n_cluster, cluster_size, height, width, rng) -> np.ndarray:
    lo, hi = cluster_size
    placed: list[int] = []
    left = int(n_cluster)
    while left >= lo:
        size = int(rng.integers(lo, hi + 1))
        if size > left:
            size = left
        if size < lo:
            break
        if 0 < left - size < lo:
            size = left if left <= hi else lo
            if size < lo or size > left:
                break
        cells = _grow_cluster(occupied, height, width, size, rng)
        if not cells:
            break
        placed.extend(cells)
        _separate_cluster(occupied, cells, height, width)
        left -= len(cells)
    if not placed:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(placed, dtype=np.int64)


def _separate_cluster(occupied: np.ndarray, cells: list[int], height: int, width: int) -> None:
    """Stop the next cluster from touching this one, including on a diagonal.

    The blocked pixels are not hot. They only keep two 2–3 px clusters from
    becoming one larger connected component.
    """
    member = set(cells)
    for flat in cells:
        row, col = divmod(int(flat), width)
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                n_row = row + d_row
                n_col = col + d_col
                if not (0 <= n_row < height and 0 <= n_col < width):
                    continue
                neighbor = n_row * width + n_col
                if neighbor not in member:
                    occupied[neighbor] = True


def _grow_cluster(occupied, height, width, size, rng) -> list[int]:
    n_pix = height * width
    for _attempt in range(16):
        try:
            seed = _reject_flat(occupied, rng, n_pix)
        except ValueError:
            return []
        cells = [seed]
        occupied[seed] = True
        failed = False
        while len(cells) < size:
            candidates = []
            for flat in cells:
                row, col = divmod(int(flat), width)
                for neighbor_row, neighbor_col in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ):
                    if 0 <= neighbor_row < height and 0 <= neighbor_col < width:
                        neighbor = neighbor_row * width + neighbor_col
                        if not occupied[neighbor]:
                            candidates.append(neighbor)
            if not candidates:
                failed = True
                break
            unique = np.unique(np.asarray(candidates, dtype=np.int64))
            pick = int(unique[int(rng.integers(0, unique.size))])
            occupied[pick] = True
            cells.append(pick)
        if not failed and len(cells) == size:
            return cells
        for flat in cells:
            occupied[flat] = False
    return []


def _append_simple(flat, width, kind, n_frames, rng, rows, cols, kinds, level_a, level_b, flips, states):
    for pixel in flat:
        level = _lognormal_level(rng)
        rows.append(int(pixel) // width)
        cols.append(int(pixel) % width)
        kinds.append(kind)
        level_a.append(level)
        level_b.append(level)
        flips.append(0.0)
        states.append([0] * n_frames)


def _append_rts(flat, width, n_frames, p_flip, rng, rows, cols, kinds, level_a, level_b, flips, states):
    lo, hi = p_flip
    for pixel in flat:
        first = _lognormal_level(rng)
        second = _lognormal_level(rng)
        if abs(first - second) <= 0.05 * max(first, second, 1.0):
            second = first * 1.5
        probability = float(rng.uniform(lo, hi))
        state = int(rng.integers(0, 2))
        sequence = []
        for frame in range(n_frames):
            sequence.append(state)
            if frame + 1 < n_frames and float(rng.random()) < probability:
                state = 1 - state
        rows.append(int(pixel) // width)
        cols.append(int(pixel) % width)
        kinds.append(KIND_RTS)
        level_a.append(first)
        level_b.append(second)
        flips.append(probability)
        states.append(sequence)


def _lognormal_level(rng) -> float:
    return float(rng.lognormal(math.log(HOT_LOG_MEDIAN_E), HOT_LOG_SIGMA))


def _cosmic_geometry(cells: list[tuple[int, int]]) -> tuple[int, float, float, int]:
    """``n_steps``, path length, end-to-end displacement, and unique-cell count."""
    n_steps = len(cells)
    path = 0.0
    for (row0, col0), (row1, col1) in zip(cells, cells[1:]):
        path += math.hypot(row1 - row0, col1 - col0)
    last_row, last_col = cells[-1]
    first_row, first_col = cells[0]
    disp = math.hypot(last_row - first_row, last_col - first_col)
    return n_steps, path, disp, len(set(cells))


def _cosmic_rays(sensor: SensorFingerprint, rng, rate=None):
    if rate is None:
        rate = sensor.cosmic_rate
    if rate < 0.0:
        raise ValueError(f"cosmic_rate must be >= 0, got {rate}")
    image = np.zeros((sensor.height, sensor.width), dtype=np.float64)
    if rate == 0.0:
        return image, []
    count = int(rng.poisson(rate))
    records = []
    for _index in range(count):
        energy = float(rng.uniform(*_COSMIC_ENERGY))
        row = int(rng.integers(0, sensor.height))
        col = int(rng.integers(0, sensor.width))
        pointlike = float(rng.random()) < 0.5
        if pointlike:
            image[row, col] += energy
            records.append(CosmicRay(energy, row, col, True, ((row, col),), 1, 0.0, 0.0, 1))
            continue
        length = float(rng.uniform(*_COSMIC_LENGTH))
        steps = max(2, int(round(length)))
        cells = [(row, col)]
        cursor_row, cursor_col = row, col
        for _step in range(steps - 1):
            options = []
            for next_row, next_col in (
                (cursor_row - 1, cursor_col),
                (cursor_row + 1, cursor_col),
                (cursor_row, cursor_col - 1),
                (cursor_row, cursor_col + 1),
            ):
                if 0 <= next_row < sensor.height and 0 <= next_col < sensor.width:
                    options.append((next_row, next_col))
            if not options:
                break
            cursor_row, cursor_col = options[int(rng.integers(0, len(options)))]
            cells.append((cursor_row, cursor_col))
        share = energy / len(cells)
        for cell_row, cell_col in cells:
            image[cell_row, cell_col] += share
        records.append(CosmicRay(energy, row, col, False, tuple(cells), *_cosmic_geometry(cells)))
    return image, records


def _bilinear(image: np.ndarray, x: float, y: float) -> float:
    height, width = image.shape
    if x < 0.0 or y < 0.0 or x > width - 1 or y > height - 1:
        raise ValueError(f"bilinear sample ({x}, {y}) is outside the image")
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    tx = x - x0
    ty = y - y0
    return float(
        (1.0 - tx) * (1.0 - ty) * image[y0, x0]
        + tx * (1.0 - ty) * image[y0, x1]
        + (1.0 - tx) * ty * image[y1, x0]
        + tx * ty * image[y1, x1]
    )


def _scalar(mapping: dict, *path: str) -> float:
    node: object = mapping
    walked: list[str] = []
    for key in path:
        walked.append(key)
        if not isinstance(node, dict) or key not in node:
            raise KeyError(".".join(walked))
        node = node[key]
    dotted = ".".join(path)
    if isinstance(node, (list, tuple)):
        raise TypeError(f"{dotted} must be a sampled scalar, got {node!r}")
    if isinstance(node, bool) or not isinstance(node, (int, float, np.floating)):
        raise TypeError(f"{dotted} must be a real number, got {node!r}")
    value = float(node)
    if not math.isfinite(value):
        raise ValueError(f"{dotted} must be finite, got {node!r}")
    return value


def _float_interval(mapping: dict, *path: str) -> tuple[float, float]:
    node = _lookup(mapping, path)
    if not isinstance(node, (list, tuple)) or len(node) != 2:
        raise TypeError(f"{'.'.join(path)} must be an interval, got {node!r}")
    lo = float(node[0])
    hi = float(node[1])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
        raise ValueError(f"{'.'.join(path)} must be a finite ordered interval, got {node!r}")
    return lo, hi


def _int_interval(mapping: dict, *path: str) -> tuple[int, int]:
    node = _lookup(mapping, path)
    if not isinstance(node, (list, tuple)) or len(node) != 2:
        raise TypeError(f"{'.'.join(path)} must be an interval, got {node!r}")
    if type(node[0]) is not int or type(node[1]) is not int:
        raise TypeError(f"{'.'.join(path)} endpoints must be int, got {node!r}")
    if node[0] > node[1]:
        raise ValueError(f"{'.'.join(path)} lower bound exceeds upper bound, got {node!r}")
    return int(node[0]), int(node[1])


def _lookup(mapping: dict, path: tuple[str, ...]) -> object:
    node: object = mapping
    walked: list[str] = []
    for key in path:
        walked.append(key)
        if not isinstance(node, dict) or key not in node:
            raise KeyError(".".join(walked))
        node = node[key]
    return node
