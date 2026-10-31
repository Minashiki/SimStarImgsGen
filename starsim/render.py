"""Pixel-integrated stamps for point sources and streaks.

Every stamp is a 5× supersampled evaluation of the PSF, block-averaged back
to pixels. That is the production integrator for both the Gaussian core and
the Moffat halo, and for a streak, which is a chain of point sources spaced
by at most 0.2 px. A circular Gaussian can also be integrated with ``erf``;
that function is the reference for the supersampled core, not a second
renderer.

The stamp radius used while exposing a frame is the radius at which the
wing falls below ``0.05 · σ_noise``, and it is capped at 64 px. An
untruncated stamp uses :func:`tail_capture_radius` instead, which keeps the
analytic tail under a requested fraction of the flux and is what the
flux-conservation tests call.

Pixel ``(row, col)`` is centered at continuous ``(x=col, y=row)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from starsim.attitude import boresight_radec
from starsim.camera import Camera, project_icrs
from starsim.catalog import StarCatalog
from starsim.photometry import Photometry
from starsim.psf import MOFFAT_ALPHA_PER_SIGMA, MOFFAT_BETA, PsfModel

# Behavior version for the pixel integrator. Increment when the same sources
# start producing different electron images or s_self values, whether or not
# a record of the old output has been written.
# 1 is the shared 5× supersampled integral.
# 2 draws static stars in catalog-index order (D44).
# 3：动目标曝光半径改用覆盖整条线段 PSF 的径向包络，取点源界与线源界中较小的一个。
RENDER_VERSION = 3

SUBSAMPLE = 5
MAX_STAMP_RADIUS = 64
WING_NOISE_FRACTION = 0.05
STREAK_STEP_PX = 0.2
_TAIL_LOSS = 1e-4


@dataclass
class RenderedSources:
    """Linear electron image of the sources, before vignetting and PRNU.

    ``s_self_linear`` is this source's own signal in its peak pixel.
    ``peak_row`` and ``peak_col`` are absolute pixel indices, or −1 when the
    stamp missed the image. The box fields are the inclusive absolute pixel
    window that was integrated, clipped to the image, and −1 when that
    window is empty. ``in_image`` is whether the true center lies in
    ``[−0.5, W−0.5) × [−0.5, H−0.5)``.
    """

    image: np.ndarray
    x: np.ndarray
    y: np.ndarray
    flux_e: np.ndarray
    mag: np.ndarray
    radius: np.ndarray
    box_row0: np.ndarray
    box_row1: np.ndarray
    box_col0: np.ndarray
    box_col1: np.ndarray
    peak_row: np.ndarray
    peak_col: np.ndarray
    s_self_linear: np.ndarray
    flux_in_image: np.ndarray
    in_image: np.ndarray
    tyc1: np.ndarray
    tyc2: np.ndarray


@dataclass(frozen=True)
class StaticStarFrame:
    """One frame of static stars as source arrays, with no electron image.

    :func:`project_static_stars` builds this. :func:`paint_static_stars`
    paints it. :func:`star_value_at` evaluates one pixel from it.
    """

    x: np.ndarray
    y: np.ndarray
    flux_e: np.ndarray
    mag: np.ndarray
    radius: np.ndarray
    box_row0: np.ndarray
    box_row1: np.ndarray
    box_col0: np.ndarray
    box_col1: np.ndarray
    tyc1: np.ndarray
    tyc2: np.ndarray
    psf: PsfModel
    height: int
    width: int


# Point-like moving targets and streaks share one code. Track ids increase.
KIND_POINT = 0
KIND_STREAK = 1


@dataclass
class RenderedTargets:
    """One frame of moving targets, sorted by track id.

    The linear ``image`` is the sum of the per-target stamps and does not
    include vignetting or PRNU. ``stamps[i]`` is the window
    ``image[row0:row1+1, col0:col1+1]`` of that target before later targets
    were added, so a pixel of the stamp is the value that was added.
    ``s_self_linear`` is the brightest pixel of that stamp. ``center_row``
    and ``center_col`` are the pixel that contains the true center, or −1
    when the center is outside the image. An empty window is −1 on all four
    sides and the stamp has shape ``(0, 0)``.
    """

    image: np.ndarray
    track_id: np.ndarray
    kind: np.ndarray
    x_c: np.ndarray
    y_c: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    flux_e: np.ndarray
    radius: np.ndarray
    box_row0: np.ndarray
    box_row1: np.ndarray
    box_col0: np.ndarray
    box_col1: np.ndarray
    peak_row: np.ndarray
    peak_col: np.ndarray
    s_self_linear: np.ndarray
    center_row: np.ndarray
    center_col: np.ndarray
    in_image: np.ndarray
    flux_in_image: np.ndarray
    stamps: tuple[np.ndarray, ...]


def noise_capture_radius(
    flux: float,
    sigma_r: float,
    sigma_t: float,
    halo_fraction: float,
    alpha: float,
    beta: float,
    noise_sigma: float,
    cap: int = MAX_STAMP_RADIUS,
) -> int:
    """Radius at which the PSF wing drops below ``0.05 · σ_noise``, capped."""
    if noise_sigma < 0.0 or not math.isfinite(noise_sigma):
        raise ValueError(f"noise_sigma must be finite and >= 0, got {noise_sigma}")
    threshold = WING_NOISE_FRACTION * float(noise_sigma)
    radius = _wing_radius(flux, sigma_r, sigma_t, halo_fraction, alpha, beta, threshold)
    if radius < 1.0:
        radius = 1.0
    return int(min(cap, max(1, math.ceil(radius))))


def tail_capture_radius(
    sigma_r: float,
    sigma_t: float,
    halo_fraction: float,
    alpha: float,
    beta: float,
    loss: float = _TAIL_LOSS,
) -> int:
    """Radius that leaves less than ``loss`` of a unit-flux PSF outside.

    No 64 px cap: this is the untruncated stamp, not the exposure stamp.
    """
    if not 0.0 < loss < 1.0:
        raise ValueError(f"loss must be in (0, 1), got {loss}")
    sigma = max(float(sigma_r), float(sigma_t), 1e-6)
    # 2D Gaussian tail beyond R is exp(−R² / 2σ²).
    gauss_r = sigma * math.sqrt(-2.0 * math.log(loss))
    if halo_fraction <= 0.0:
        moffat_r = 0.0
    else:
        # Enclosed Moffat fraction is 1 − [1 + (R/α)²]^(1−β).
        remain = loss / float(halo_fraction)
        if remain >= 1.0:
            moffat_r = 0.0
        else:
            power = 1.0 / (1.0 - beta)
            moffat_r = alpha * math.sqrt(max(remain ** power - 1.0, 0.0))
    return int(max(1, math.ceil(max(gauss_r, moffat_r))))


def line_equivalent_flux(flux: float, sigma_max: float, length: float) -> float:
    """Point-source flux whose core matches a line source of length ``length``.

    ``F_eq = F · min(1, √(2π) · σ_max / L)``. A zero length is a point, so
    the equivalent flux is ``F`` itself. ``σ_max`` is the larger core axis
    at the segment midpoint.
    """
    flux = float(flux)
    sigma_max = float(sigma_max)
    length = float(length)
    if flux < 0.0 or not math.isfinite(flux):
        raise ValueError(f"flux must be finite and >= 0, got {flux}")
    if sigma_max <= 0.0 or not math.isfinite(sigma_max):
        raise ValueError(f"sigma_max must be finite and > 0, got {sigma_max}")
    if not math.isfinite(length) or length < 0.0:
        raise ValueError(f"length must be finite and >= 0, got {length}")
    if length == 0.0:
        return flux
    factor = math.sqrt(2.0 * math.pi) * sigma_max / length
    if factor > 1.0:
        factor = 1.0
    return flux * factor


def segment_sample_count(x1: float, y1: float, x2: float, y2: float) -> int:
    """How many samples :func:`streak_samples` places on this segment."""
    length = math.hypot(float(x2) - float(x1), float(y2) - float(y1))
    if length == 0.0:
        return 1
    return max(1, int(math.ceil(length / STREAK_STEP_PX)))


def segment_envelope_brightness(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    psf: PsfModel,
    radius: float,
) -> float:
    """Radial upper bound B(r) on the unit-flux PSF of any source on the segment.

    σ_iso is smallest at the optical axis's projection onto the segment
    (clamped to the endpoints) and largest at the farther endpoint. The
    Gaussian core uses the smallest σ_iso for its amplitude and the largest
    σ_r for its decay. The Moffat halo uses the smallest α for its amplitude
    and the largest α for its decay.
    """
    _require_psf(psf)
    coefficients = _envelope_coefficients(float(x1), float(y1), float(x2), float(y2), psf)
    return _envelope_b(coefficients, float(radius))


def segment_exposure_radius(
    flux: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    n_samples: int,
    psf: PsfModel,
    noise_sigma: float,
) -> int:
    """Shared stamp radius from the point-source and line-source envelopes.

    The radius is the smallest integer R ≥ 1, at most 64, where
    min(P(R), W(R)) ≤ 0.05 σ_ref. P is the point-source bound. W is the
    line-source bound, including the one-sample discrete correction, and is
    used only when the segment has positive length and more than one sample.
    The value is judged at R, the same place a static star's noise radius
    is judged.
    """
    _require_psf(psf)
    if type(n_samples) is not int or isinstance(n_samples, bool) or n_samples < 1:
        raise ValueError(f"n_samples must be an int >= 1, got {n_samples!r}")
    flux = float(flux)
    if flux < 0.0 or not math.isfinite(flux):
        raise ValueError(f"flux must be finite and >= 0, got {flux}")
    if noise_sigma < 0.0 or not math.isfinite(noise_sigma):
        raise ValueError(f"noise_sigma must be finite and >= 0, got {noise_sigma}")
    for name, value in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value}")
    length = math.hypot(float(x2) - float(x1), float(y2) - float(y1))
    coefficients = _envelope_coefficients(float(x1), float(y1), float(x2), float(y2), psf)
    use_line = length > 0.0 and n_samples > 1
    threshold = WING_NOISE_FRACTION * float(noise_sigma)
    for radius in range(1, MAX_STAMP_RADIUS + 1):
        point = flux * _envelope_b(coefficients, float(radius))
        if use_line:
            bound = min(point, _line_bound(coefficients, flux, length, n_samples, float(radius)))
        else:
            bound = point
        if bound <= threshold:
            return radius
    return MAX_STAMP_RADIUS


def _envelope_coefficients(x1: float, y1: float, x2: float, y2: float, psf: PsfModel):
    dx = x2 - x1
    dy = y2 - y1
    length2 = dx * dx + dy * dy
    if length2 == 0.0:
        near_x, near_y = x1, y1
    else:
        fraction = ((psf.cx - x1) * dx + (psf.cy - y1) * dy) / length2
        if fraction < 0.0:
            fraction = 0.0
        elif fraction > 1.0:
            fraction = 1.0
        near_x = x1 + fraction * dx
        near_y = y1 + fraction * dy
    far_x, far_y = x1, y1
    if (x2 - psf.cx) ** 2 + (y2 - psf.cy) ** 2 > (x1 - psf.cx) ** 2 + (y1 - psf.cy) ** 2:
        far_x, far_y = x2, y2
    _sr_near, _st_near, sigma_min, _ux, _uy = psf.axes(near_x, near_y)
    sigma_r_far, _st_far, sigma_max, _ux, _uy = psf.axes(far_x, far_y)
    halo = float(psf.halo_fraction)
    beta = float(MOFFAT_BETA)
    alpha_min = float(MOFFAT_ALPHA_PER_SIGMA) * float(sigma_min)
    alpha_max = float(MOFFAT_ALPHA_PER_SIGMA) * float(sigma_max)
    amplitude_g = (1.0 - halo) / (2.0 * math.pi * float(sigma_min) * float(sigma_min))
    if halo == 0.0:
        amplitude_m = 0.0
    else:
        amplitude_m = halo * (beta - 1.0) / (math.pi * alpha_min * alpha_min)
    return amplitude_g, amplitude_m, float(sigma_r_far), alpha_max, beta


def _envelope_b(coefficients, radius: float) -> float:
    amplitude_g, amplitude_m, sigma_r, alpha_max, beta = coefficients
    gauss = math.exp(-0.5 * (radius * radius) / (sigma_r * sigma_r))
    if amplitude_m == 0.0:
        return amplitude_g * gauss
    moffat = (1.0 + (radius * radius) / (alpha_max * alpha_max)) ** (-beta)
    return amplitude_g * gauss + amplitude_m * moffat


def _line_bound(coefficients, flux: float, length: float, n_samples: int, radius: float) -> float:
    amplitude_g, amplitude_m, sigma_r, alpha_max, beta = coefficients
    gauss = amplitude_g * math.sqrt(2.0 * math.pi) * sigma_r * math.exp(-0.5 * (radius * radius) / (sigma_r * sigma_r))
    if amplitude_m == 0.0:
        halo = 0.0
    else:
        halo = (
            amplitude_m
            * alpha_max
            * math.sqrt(math.pi)
            * math.gamma(beta - 0.5)
            / math.gamma(beta)
            * (1.0 + (radius * radius) / (alpha_max * alpha_max)) ** (0.5 - beta)
        )
    line = (flux / length) * (gauss + halo)
    samples = 2.0 * (flux / float(n_samples)) * _envelope_b(coefficients, radius)
    return line + samples


def noise_capture_radii(
    x: np.ndarray,
    y: np.ndarray,
    flux: np.ndarray,
    psf: PsfModel,
    noise_sigma: float,
) -> np.ndarray:
    """Per-source exposure radii. Sources share ``noise_sigma``."""
    _require_psf(psf)
    radii = np.empty(flux.shape[0], dtype=np.int64)
    for index in range(flux.shape[0]):
        sigma_r, sigma_t, sigma_iso, _ux, _uy = psf.axes(float(x[index]), float(y[index]))
        alpha = psf.moffat_alpha(sigma_iso)
        radii[index] = noise_capture_radius(
            float(flux[index]),
            sigma_r,
            sigma_t,
            psf.halo_fraction,
            alpha,
            MOFFAT_BETA,
            noise_sigma,
        )
    return radii


def add_points(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    flux: np.ndarray,
    psf: PsfModel,
    radii: np.ndarray,
    origin_row: int = 0,
    origin_col: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add point-source stamps.

    Returns absolute ``peak_row, peak_col, s_self`` and the inclusive stamp
    boxes ``box_row0, box_row1, box_col0, box_col1``.

    ``(x, y)`` are absolute pixel coordinates. ``image[0, 0]`` is absolute
    pixel ``(origin_row, origin_col)``. ``s_self`` is this source's own
    signal in its peak pixel, before vignetting and PRNU.
    """
    _require_psf(psf)
    destination = np.asarray(image, dtype=np.float64)
    if destination.ndim != 2 or not destination.flags.c_contiguous:
        raise ValueError("image must be a C-contiguous 2D float array")
    coords_x = np.ascontiguousarray(x, dtype=np.float64)
    coords_y = np.ascontiguousarray(y, dtype=np.float64)
    weights = np.ascontiguousarray(flux, dtype=np.float64)
    stamp_r = np.ascontiguousarray(radii, dtype=np.int64)
    if coords_x.shape != coords_y.shape or coords_x.shape != weights.shape or coords_x.shape != stamp_r.shape:
        raise ValueError("x, y, flux, and radii must have the same shape")
    if coords_x.ndim != 1:
        raise ValueError("x, y, flux, and radii must be 1D")
    count = int(coords_x.shape[0])
    peak_row = np.empty(count, dtype=np.int64)
    peak_col = np.empty(count, dtype=np.int64)
    s_self = np.empty(count, dtype=np.float64)
    box_row0 = np.empty(count, dtype=np.int64)
    box_row1 = np.empty(count, dtype=np.int64)
    box_col0 = np.empty(count, dtype=np.int64)
    box_col1 = np.empty(count, dtype=np.int64)
    if count == 0:
        return peak_row, peak_col, s_self, box_row0, box_row1, box_col0, box_col1
    _paint(
        destination,
        int(origin_row),
        int(origin_col),
        coords_x,
        coords_y,
        weights,
        stamp_r,
        float(psf.cx),
        float(psf.cy),
        float(psf.r_max),
        float(psf.sigma0),
        float(psf.field_growth),
        float(psf.max_ellipticity),
        float(psf.halo_fraction),
        float(MOFFAT_ALPHA_PER_SIGMA),
        float(MOFFAT_BETA),
        peak_row,
        peak_col,
        s_self,
        box_row0,
        box_row1,
        box_col0,
        box_col1,
    )
    return peak_row, peak_col, s_self, box_row0, box_row1, box_col0, box_col1


def render_points(
    shape: tuple[int, int],
    x: np.ndarray,
    y: np.ndarray,
    flux: np.ndarray,
    psf: PsfModel,
    radii: np.ndarray,
) -> tuple:
    """Render points into a fresh image.

    Returns ``peak_row, peak_col, s_self, image`` and the inclusive stamp
    boxes ``box_row0, box_row1, box_col0, box_col1``.
    """
    image = np.zeros(shape, dtype=np.float64)
    peak_row, peak_col, s_self, box_row0, box_row1, box_col0, box_col1 = add_points(
        image, x, y, flux, psf, radii
    )
    return peak_row, peak_col, s_self, image, box_row0, box_row1, box_col0, box_col1


def streak_samples(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    step: float = STREAK_STEP_PX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a uniform segment with spacing at most ``step``.

    Samples sit at the centers of equal pieces, so the weights sum to 1 and
    a zero-length segment is a single sample.
    """
    if step <= 0.0 or not math.isfinite(step):
        raise ValueError(f"step must be > 0, got {step}")
    for name, value in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0.0:
        return (
            np.array([x1], dtype=np.float64),
            np.array([y1], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )
    count = max(1, int(math.ceil(length / step)))
    fraction = (np.arange(count, dtype=np.float64) + 0.5) / count
    return (
        x1 + fraction * (x2 - x1),
        y1 + fraction * (y2 - y1),
        np.full(count, 1.0 / count, dtype=np.float64),
    )


def add_streak(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    flux: float,
    psf: PsfModel,
    radii_for_unit: np.ndarray | None = None,
    step: float = STREAK_STEP_PX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Paint a streak as point sources spaced by at most 0.2 px.

    Each sample carries ``flux / n`` and uses the local PSF at that sample.
    Pass per-sample radii, or omit them to use :func:`tail_capture_radius`
    so an untruncated streak keeps its flux. Returns the sample peak records;
    the streak's own peak is the maximum of the image region it painted only
    when the image was otherwise empty.
    """
    _require_psf(psf)
    if not math.isfinite(flux) or flux < 0.0:
        raise ValueError(f"flux must be finite and >= 0, got {flux}")
    xs, ys, weights = streak_samples(x1, y1, x2, y2, step)
    if radii_for_unit is None:
        radii = np.empty(xs.shape[0], dtype=np.int64)
        for index in range(xs.shape[0]):
            sigma_r, sigma_t, sigma_iso, _ux, _uy = psf.axes(float(xs[index]), float(ys[index]))
            radii[index] = tail_capture_radius(
                sigma_r,
                sigma_t,
                psf.halo_fraction,
                psf.moffat_alpha(sigma_iso),
                MOFFAT_BETA,
            )
    else:
        radii = np.asarray(radii_for_unit, dtype=np.int64)
        if radii.shape != xs.shape:
            raise ValueError("radii must have one entry per streak sample")
    return add_points(image, xs, ys, flux * weights, psf, radii)


def center_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """Pixel containing ``(x, y)``, or ``(−1, −1)`` when it is outside.

    Pixel ``(row, col)`` covers ``[col−0.5, col+0.5) × [row−0.5, row+0.5)``.
    """
    if not (x >= -0.5 and x < width - 0.5 and y >= -0.5 and y < height - 0.5):
        return -1, -1
    return int(math.floor(y + 0.5)), int(math.floor(x + 0.5))


def _axes_many(psf: PsfModel, x: np.ndarray, y: np.ndarray):
    """Vector form of :meth:`PsfModel.axes` without the radial unit vector."""
    dx = x - psf.cx
    dy = y - psf.cy
    r2 = dx * dx + dy * dy
    rho2 = r2 / (psf.r_max ** 2)
    sigma_iso = psf.sigma0 * (1.0 + psf.field_growth * rho2)
    ellipticity = np.minimum(psf.max_ellipticity * rho2, psf.max_ellipticity)
    sigma_r = sigma_iso / np.sqrt(1.0 - ellipticity)
    sigma_t = sigma_iso * np.sqrt(1.0 - ellipticity)
    return sigma_r, sigma_t, sigma_iso


def _tail_radii_many(psf: PsfModel, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-sample :func:`tail_capture_radius`. Matches the scalar function."""
    sigma_r, sigma_t, sigma_iso = _axes_many(psf, x, y)
    sigma = np.maximum(np.maximum(sigma_r, sigma_t), 1e-6)
    gauss_r = sigma * math.sqrt(-2.0 * math.log(_TAIL_LOSS))
    alpha = MOFFAT_ALPHA_PER_SIGMA * sigma_iso
    if psf.halo_fraction <= 0.0:
        moffat_r = np.zeros(x.shape, dtype=np.float64)
    else:
        remain = _TAIL_LOSS / float(psf.halo_fraction)
        if remain >= 1.0:
            moffat_r = np.zeros(x.shape, dtype=np.float64)
        else:
            power = 1.0 / (1.0 - MOFFAT_BETA)
            moffat_r = alpha * np.sqrt(np.maximum(remain ** power - 1.0, 0.0))
    return np.maximum(1, np.ceil(np.maximum(gauss_r, moffat_r))).astype(np.int64)


def _sample_radii(
    xs: np.ndarray,
    ys: np.ndarray,
    sample_flux: np.ndarray,
    psf: PsfModel,
    shared_radius: int | None,
    per_sample_noise: float | None,
) -> np.ndarray:
    if shared_radius is not None and per_sample_noise is not None:
        raise ValueError("pass either a shared radius or a per-sample noise, not both")
    if shared_radius is not None:
        if type(shared_radius) is not int or shared_radius < 1:
            raise ValueError(f"shared radius must be an int >= 1, got {shared_radius!r}")
        return np.full(xs.shape[0], shared_radius, dtype=np.int64)
    if per_sample_noise is None:
        return _tail_radii_many(psf, xs, ys)
    radii = np.empty(xs.shape[0], dtype=np.int64)
    for index in range(xs.shape[0]):
        sigma_r, sigma_t, sigma_iso, _ux, _uy = psf.axes(float(xs[index]), float(ys[index]))
        radii[index] = noise_capture_radius(
            float(sample_flux[index]),
            sigma_r,
            sigma_t,
            psf.halo_fraction,
            psf.moffat_alpha(sigma_iso),
            MOFFAT_BETA,
            per_sample_noise,
        )
    return radii


def _stamp_peak(stamp: np.ndarray, row0: int, col0: int) -> tuple[int, int, float]:
    """Brightest pixel, row-major, replacing only on a strict increase.

    ``np.argmax`` returns the first maximum, which is the same rule ``_paint``
    uses. An all-zero stamp still records its first pixel and a signal of 0.
    """
    flat = int(np.argmax(stamp))
    width = stamp.shape[1]
    local_row, local_col = divmod(flat, width)
    return row0 + local_row, col0 + local_col, float(stamp[local_row, local_col])


def render_segment_stamp(
    height: int,
    width: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    flux: float,
    psf: PsfModel,
    *,
    shared_radius: int | None = None,
    per_sample_noise: float | None = None,
) -> tuple[np.ndarray, int, int, int, int, int, int, float]:
    """Paint one segment into its own stamp.

    Returns ``stamp, row0, col0, row1, col1, peak_row, peak_col, s_self``.
    The stamp is empty and the box is −1 when every sample misses the image.
    Omit both radius arguments to use the untruncated tail radius on each
    sample. ``shared_radius`` is the exposure radius (D48). ``per_sample_noise``
    gives each sample ``noise_capture_radius(F/n)``; that cuts a long streak
    too early and exists so the radius test can show it.
    """
    _require_psf(psf)
    if not math.isfinite(flux) or flux < 0.0:
        raise ValueError(f"flux must be finite and >= 0, got {flux}")
    xs, ys, weights = streak_samples(x1, y1, x2, y2)
    sample_flux = flux * weights
    radii = _sample_radii(xs, ys, sample_flux, psf, shared_radius, per_sample_noise)
    row0, row1, col0, col1 = clipped_stamp_boxes(xs, ys, radii, height, width)
    valid = row0 >= 0
    if not np.any(valid):
        return np.zeros((0, 0), dtype=np.float64), -1, -1, -1, -1, -1, -1, 0.0
    r0 = int(row0[valid].min())
    r1 = int(row1[valid].max())
    c0 = int(col0[valid].min())
    c1 = int(col1[valid].max())
    stamp = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.float64)
    if flux > 0.0:
        add_points(stamp, xs, ys, sample_flux, psf, radii, origin_row=r0, origin_col=c0)
        peak_row, peak_col, s_self = _stamp_peak(stamp, r0, c0)
    else:
        peak_row, peak_col, s_self = -1, -1, 0.0
    return stamp, r0, c0, r1, c1, peak_row, peak_col, s_self


def paint_segment(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    flux: float,
    psf: PsfModel,
    *,
    shared_radius: int | None = None,
    per_sample_noise: float | None = None,
) -> np.ndarray:
    """Add one segment to ``image`` and return that image.

    The radius rules match :func:`render_segment_stamp`. Samples are painted
    straight into ``image``, not via a cropped stamp, so this is the reference
    for a radius comparison on a canvas that already holds the whole segment.
    """
    _require_psf(psf)
    destination = np.asarray(image, dtype=np.float64)
    if not math.isfinite(flux) or flux < 0.0:
        raise ValueError(f"flux must be finite and >= 0, got {flux}")
    xs, ys, weights = streak_samples(x1, y1, x2, y2)
    sample_flux = flux * weights
    radii = _sample_radii(xs, ys, sample_flux, psf, shared_radius, per_sample_noise)
    if flux > 0.0:
        add_points(destination, xs, ys, sample_flux, psf, radii)
    return destination


def render_moving_targets(
    track_id: np.ndarray,
    kind: np.ndarray,
    x_c: np.ndarray,
    y_c: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    flux_e: np.ndarray,
    psf: PsfModel,
    noise_sigma: float,
) -> RenderedTargets:
    """Paint each target into a private stamp, then add the stamps in track order.

    Every sample of one target uses :func:`segment_exposure_radius`. The
    peak and ``s_self_linear`` come from the finished stamp. Targets are
    sorted by track id before anything is drawn.
    """
    _require_psf(psf)
    ids = np.asarray(track_id)
    kinds = np.asarray(kind)
    xc = np.asarray(x_c, dtype=np.float64)
    yc = np.asarray(y_c, dtype=np.float64)
    xa = np.asarray(x1, dtype=np.float64)
    ya = np.asarray(y1, dtype=np.float64)
    xb = np.asarray(x2, dtype=np.float64)
    yb = np.asarray(y2, dtype=np.float64)
    flux = np.asarray(flux_e, dtype=np.float64)
    count = int(ids.shape[0])
    if not (
        kinds.shape == (count,)
        and xc.shape == (count,)
        and yc.shape == (count,)
        and xa.shape == (count,)
        and ya.shape == (count,)
        and xb.shape == (count,)
        and yb.shape == (count,)
        and flux.shape == (count,)
    ):
        raise ValueError("target arrays must be 1D and the same length")
    image = np.zeros((psf.height, psf.width), dtype=np.float64)
    if count == 0:
        return _empty_targets(psf.height, psf.width)
    order = np.argsort(ids, kind="mergesort")
    ids = np.ascontiguousarray(ids[order])
    kinds = np.ascontiguousarray(kinds[order])
    xc = np.ascontiguousarray(xc[order])
    yc = np.ascontiguousarray(yc[order])
    xa = np.ascontiguousarray(xa[order])
    ya = np.ascontiguousarray(ya[order])
    xb = np.ascontiguousarray(xb[order])
    yb = np.ascontiguousarray(yb[order])
    flux = np.ascontiguousarray(flux[order])
    radius = np.empty(count, dtype=np.int64)
    box_row0 = np.empty(count, dtype=np.int64)
    box_row1 = np.empty(count, dtype=np.int64)
    box_col0 = np.empty(count, dtype=np.int64)
    box_col1 = np.empty(count, dtype=np.int64)
    peak_row = np.empty(count, dtype=np.int64)
    peak_col = np.empty(count, dtype=np.int64)
    s_self = np.empty(count, dtype=np.float64)
    center_row = np.empty(count, dtype=np.int64)
    center_col = np.empty(count, dtype=np.int64)
    stamps: list[np.ndarray] = []
    for index in range(count):
        n_samples = segment_sample_count(float(xa[index]), float(ya[index]), float(xb[index]), float(yb[index]))
        radius[index] = segment_exposure_radius(
            float(flux[index]),
            float(xa[index]),
            float(ya[index]),
            float(xb[index]),
            float(yb[index]),
            n_samples,
            psf,
            noise_sigma,
        )
        stamp, r0, c0, r1, c1, prow, pcol, signal = render_segment_stamp(
            psf.height,
            psf.width,
            float(xa[index]),
            float(ya[index]),
            float(xb[index]),
            float(yb[index]),
            float(flux[index]),
            psf,
            shared_radius=int(radius[index]),
        )
        box_row0[index] = r0
        box_col0[index] = c0
        box_row1[index] = r1
        box_col1[index] = c1
        peak_row[index] = prow
        peak_col[index] = pcol
        s_self[index] = signal
        center_row[index], center_col[index] = center_pixel(
            float(xc[index]), float(yc[index]), psf.width, psf.height
        )
        stamps.append(np.ascontiguousarray(stamp))
        if r0 >= 0:
            image[r0 : r1 + 1, c0 : c1 + 1] += stamp
    return RenderedTargets(
        image=image,
        track_id=np.ascontiguousarray(ids),
        kind=np.ascontiguousarray(kinds),
        x_c=xc,
        y_c=yc,
        x1=xa,
        y1=ya,
        x2=xb,
        y2=yb,
        flux_e=flux,
        radius=radius,
        box_row0=box_row0,
        box_row1=box_row1,
        box_col0=box_col0,
        box_col1=box_col1,
        peak_row=peak_row,
        peak_col=peak_col,
        s_self_linear=s_self,
        center_row=center_row,
        center_col=center_col,
        in_image=centers_in_image(xc, yc, psf.width, psf.height),
        flux_in_image=s_self > 0.0,
        stamps=tuple(stamps),
    )


def _empty_targets(height: int, width: int) -> RenderedTargets:
    none = np.zeros(0, dtype=np.float64)
    empty_i = np.zeros(0, dtype=np.int64)
    return RenderedTargets(
        image=np.zeros((height, width), dtype=np.float64),
        track_id=empty_i.copy(),
        kind=np.zeros(0, dtype=np.int8),
        x_c=none,
        y_c=none.copy(),
        x1=none.copy(),
        y1=none.copy(),
        x2=none.copy(),
        y2=none.copy(),
        flux_e=none.copy(),
        radius=empty_i.copy(),
        box_row0=empty_i.copy(),
        box_row1=empty_i.copy(),
        box_col0=empty_i.copy(),
        box_col1=empty_i.copy(),
        peak_row=empty_i.copy(),
        peak_col=empty_i.copy(),
        s_self_linear=none.copy(),
        center_row=empty_i.copy(),
        center_col=empty_i.copy(),
        in_image=np.zeros(0, dtype=bool),
        flux_in_image=np.zeros(0, dtype=bool),
        stamps=(),
    )


def render_unit_stamp(
    psf: PsfModel,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    radius: int | None = None,
) -> np.ndarray:
    """Unit-flux stamp of the field-center PSF, source at a pixel center plus ``offset``.

    The stamp is large enough for ``radius`` pixels around the source. When
    ``radius`` is omitted the analytic tail radius is used, so the stamp is
    the untruncated one.
    """
    _require_psf(psf)
    sigma_r, sigma_t, sigma_iso, _ux, _uy = psf.axes(psf.cx, psf.cy)
    if radius is None:
        radius = tail_capture_radius(
            sigma_r,
            sigma_t,
            psf.halo_fraction,
            psf.moffat_alpha(sigma_iso),
            MOFFAT_BETA,
        )
    if type(radius) is not int or radius < 1:
        raise ValueError(f"radius must be an int >= 1, got {radius!r}")
    side = 2 * radius + 3
    # A private camera-sized model would make the source miss the field center.
    # The stamp's own principal point is the stamp center, so σ = σ₀.
    local = PsfModel(
        sigma0=psf.sigma0,
        field_growth=0.0,
        max_ellipticity=0.0,
        halo_fraction=psf.halo_fraction,
        width=side,
        height=side,
        cx=(side - 1) / 2.0,
        cy=(side - 1) / 2.0,
    )
    center = (side - 1) / 2.0
    _peak_row, _peak_col, _s_self, image, _br0, _br1, _bc0, _bc1 = render_points(
        (side, side),
        np.array([center + offset_x]),
        np.array([center + offset_y]),
        np.array([1.0]),
        local,
        np.array([radius], dtype=np.int64),
    )
    return image


def peak_energy_fraction(psf: PsfModel, n_side: int = 5) -> float:
    """Mean, over a uniform subpixel grid, of the brightest pixel's flux share.

    The grid covers ``[−0.5, 0.5)`` with ``n_side²`` phases. The PSF is the
    field-center PSF, including the halo.
    """
    if type(n_side) is not int or n_side < 1:
        raise ValueError(f"n_side must be an int >= 1, got {n_side!r}")
    offsets = (np.arange(n_side, dtype=np.float64) + 0.5) / n_side - 0.5
    total = 0.0
    for offset_y in offsets:
        for offset_x in offsets:
            stamp = render_unit_stamp(psf, float(offset_x), float(offset_y))
            total += float(stamp.max())
    return total / float(n_side * n_side)


def peak_fraction_at(psf: PsfModel, offset_x: float, offset_y: float) -> float:
    """Brightest-pixel share for one subpixel phase of the field-center PSF."""
    stamp = render_unit_stamp(psf, float(offset_x), float(offset_y))
    return float(stamp.max())


def gaussian_pixel_integral(x0: float, y0: float, sigma: float, row: int, col: int) -> float:
    """Analytic integral of a circular unit Gaussian over one pixel.

    This is the reference for the supersampled core. The pixel covers
    ``[col−0.5, col+0.5) × [row−0.5, row+0.5)``.
    """
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if type(row) is not int or type(col) is not int:
        raise TypeError("row and col must be int")
    along_x = _erf_span(x0, sigma, col - 0.5, col + 0.5)
    along_y = _erf_span(y0, sigma, row - 0.5, row + 0.5)
    return along_x * along_y


def project_static_stars(
    catalog: StarCatalog,
    camera: Camera,
    attitude: np.ndarray,
    photometry: Photometry,
    psf: PsfModel,
    noise_sigma: float,
    margin_px: float = float(MAX_STAMP_RADIUS),
) -> StaticStarFrame:
    """Project one frame into source arrays. This does not allocate an image.

    The cone, magnitude cut, catalog order, radii, and stamp boxes are the
    same ones :func:`render_static_stars` paints.
    """
    if not isinstance(catalog, StarCatalog):
        raise TypeError("catalog must be starsim.catalog.StarCatalog")
    if not isinstance(camera, Camera):
        raise TypeError("camera must be starsim.camera.Camera")
    if not isinstance(photometry, Photometry):
        raise TypeError("photometry must be starsim.photometry.Photometry")
    _require_psf(psf)
    if psf.width != camera.width or psf.height != camera.height:
        raise ValueError("psf width and height must match the camera")
    if margin_px < 0.0 or not math.isfinite(margin_px):
        raise ValueError(f"margin_px must be finite and >= 0, got {margin_px}")
    ra_rad, dec_rad = boresight_radec(np.asarray(attitude, dtype=np.float64))
    half_width = camera.width / (2.0 * camera.f_over_p)
    half_height = camera.height / (2.0 * camera.f_over_p)
    half_diag_deg = math.degrees(math.atan(math.hypot(half_width, half_height)))
    margin_deg = math.degrees(margin_px / camera.f_over_p)
    found = catalog.cone_query(math.degrees(ra_rad), math.degrees(dec_rad), half_diag_deg + margin_deg)
    if found.size == 0:
        return _empty_star_frame(camera, psf)
    bright_enough = catalog.vmag[found] <= photometry.render_mag_max
    found = np.sort(found[bright_enough])
    if found.size == 0:
        return _empty_star_frame(camera, psf)
    vectors = catalog.xyz[found]
    xy = project_icrs(camera, attitude, vectors)
    x = np.ascontiguousarray(xy[:, 0], dtype=np.float64)
    y = np.ascontiguousarray(xy[:, 1], dtype=np.float64)
    mag = np.ascontiguousarray(catalog.vmag[found], dtype=np.float64)
    flux = np.ascontiguousarray(photometry.flux(mag), dtype=np.float64)
    radii = noise_capture_radii(x, y, flux, psf, noise_sigma)
    box_row0, box_row1, box_col0, box_col1 = clipped_stamp_boxes(
        x, y, radii, camera.height, camera.width
    )
    return StaticStarFrame(
        x=x,
        y=y,
        flux_e=flux,
        mag=mag,
        radius=radii,
        box_row0=box_row0,
        box_row1=box_row1,
        box_col0=box_col0,
        box_col1=box_col1,
        tyc1=np.ascontiguousarray(catalog.tyc1[found], dtype=np.int32),
        tyc2=np.ascontiguousarray(catalog.tyc2[found], dtype=np.int32),
        psf=psf,
        height=int(camera.height),
        width=int(camera.width),
    )


def paint_static_stars(frame: StaticStarFrame) -> RenderedSources:
    """Paint ``frame`` in catalog order. The boxes are the ones ``_paint`` wrote."""
    if int(frame.x.shape[0]) == 0:
        return _empty_sources_shape(frame.height, frame.width)
    peak_row, peak_col, s_self, image, box_row0, box_row1, box_col0, box_col1 = render_points(
        (frame.height, frame.width),
        frame.x,
        frame.y,
        frame.flux_e,
        frame.psf,
        frame.radius,
    )
    return RenderedSources(
        image=image,
        x=frame.x,
        y=frame.y,
        flux_e=frame.flux_e,
        mag=frame.mag,
        radius=frame.radius,
        box_row0=box_row0,
        box_row1=box_row1,
        box_col0=box_col0,
        box_col1=box_col1,
        peak_row=peak_row,
        peak_col=peak_col,
        s_self_linear=s_self,
        flux_in_image=s_self > 0.0,
        in_image=centers_in_image(frame.x, frame.y, frame.width, frame.height),
        tyc1=frame.tyc1,
        tyc2=frame.tyc2,
    )


def render_static_stars(
    catalog: StarCatalog,
    camera: Camera,
    attitude: np.ndarray,
    photometry: Photometry,
    psf: PsfModel,
    noise_sigma: float,
    margin_px: float = float(MAX_STAMP_RADIUS),
) -> RenderedSources:
    """Project catalog stars with ``m ≤ m_lim + margin`` and paint them.

    The cone is the field half-diagonal plus ``margin_px``, so wings of stars
    just outside the frame can still land in the image. Stars whose stamp
    misses the image are kept in the records with ``flux_in_image`` false.

    Order is catalog-index order (D44). ``cone_query`` already returns
    indices sorted that way; they are sorted again after the magnitude cut
    so a replay paints the same sequence.
    """
    return paint_static_stars(
        project_static_stars(
            catalog,
            camera,
            attitude,
            photometry,
            psf,
            noise_sigma,
            margin_px,
        )
    )


def star_value_at(frame_sources, row: int, col: int) -> float:
    """Star light at one pixel, summed in catalog order from 0.

    Each term is the product ``_paint`` adds into that pixel, including the
    ``flux × integral`` order inside :func:`integrate_pixel`. Background is
    not added. Stars whose centers lie outside the image still contribute
    when the clipped stamp covers the pixel.
    """
    if int(np.asarray(frame_sources.x).shape[0]) == 0:
        return 0.0
    psf = frame_sources.psf
    if isinstance(frame_sources, StaticStarFrame):
        height = int(frame_sources.height)
        width = int(frame_sources.width)
    else:
        height = int(frame_sources.image.shape[0])
        width = int(frame_sources.image.shape[1])
    return float(
        _value_at_pixel(
            int(row),
            int(col),
            height,
            width,
            np.ascontiguousarray(frame_sources.x, dtype=np.float64),
            np.ascontiguousarray(frame_sources.y, dtype=np.float64),
            np.ascontiguousarray(frame_sources.flux_e, dtype=np.float64),
            np.ascontiguousarray(frame_sources.radius, dtype=np.int64),
            float(psf.cx),
            float(psf.cy),
            float(psf.r_max),
            float(psf.sigma0),
            float(psf.field_growth),
            float(psf.max_ellipticity),
            float(psf.halo_fraction),
            float(MOFFAT_ALPHA_PER_SIGMA),
            float(MOFFAT_BETA),
        )
    )


def _empty_star_frame(camera: Camera, psf: PsfModel) -> StaticStarFrame:
    none = np.zeros(0, dtype=np.float64)
    return StaticStarFrame(
        x=none,
        y=none.copy(),
        flux_e=none.copy(),
        mag=none.copy(),
        radius=np.zeros(0, dtype=np.int64),
        box_row0=np.zeros(0, dtype=np.int64),
        box_row1=np.zeros(0, dtype=np.int64),
        box_col0=np.zeros(0, dtype=np.int64),
        box_col1=np.zeros(0, dtype=np.int64),
        tyc1=np.zeros(0, dtype=np.int32),
        tyc2=np.zeros(0, dtype=np.int32),
        psf=psf,
        height=int(camera.height),
        width=int(camera.width),
    )


def _empty_sources_shape(height: int, width: int) -> RenderedSources:
    none = np.zeros(0, dtype=np.float64)
    return RenderedSources(
        image=np.zeros((height, width), dtype=np.float64),
        x=none,
        y=none.copy(),
        flux_e=none.copy(),
        mag=none.copy(),
        radius=np.zeros(0, dtype=np.int64),
        box_row0=np.zeros(0, dtype=np.int64),
        box_row1=np.zeros(0, dtype=np.int64),
        box_col0=np.zeros(0, dtype=np.int64),
        box_col1=np.zeros(0, dtype=np.int64),
        peak_row=np.zeros(0, dtype=np.int64),
        peak_col=np.zeros(0, dtype=np.int64),
        s_self_linear=none.copy(),
        flux_in_image=np.zeros(0, dtype=bool),
        in_image=np.zeros(0, dtype=bool),
        tyc1=np.zeros(0, dtype=np.int32),
        tyc2=np.zeros(0, dtype=np.int32),
    )


def centers_in_image(x: np.ndarray, y: np.ndarray, width: int, height: int) -> np.ndarray:
    """True when the center lies in ``[−0.5, W−0.5) × [−0.5, H−0.5)``."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    return (xx >= -0.5) & (xx < width - 0.5) & (yy >= -0.5) & (yy < height - 0.5)


def _empty_sources(camera: Camera) -> RenderedSources:
    none = np.zeros(0, dtype=np.float64)
    return RenderedSources(
        image=np.zeros((camera.height, camera.width), dtype=np.float64),
        x=none,
        y=none.copy(),
        flux_e=none.copy(),
        mag=none.copy(),
        radius=np.zeros(0, dtype=np.int64),
        box_row0=np.zeros(0, dtype=np.int64),
        box_row1=np.zeros(0, dtype=np.int64),
        box_col0=np.zeros(0, dtype=np.int64),
        box_col1=np.zeros(0, dtype=np.int64),
        peak_row=np.zeros(0, dtype=np.int64),
        peak_col=np.zeros(0, dtype=np.int64),
        s_self_linear=none.copy(),
        flux_in_image=np.zeros(0, dtype=bool),
        in_image=np.zeros(0, dtype=bool),
        tyc1=np.zeros(0, dtype=np.int32),
        tyc2=np.zeros(0, dtype=np.int32),
    )


def clipped_stamp_boxes(
    x: np.ndarray,
    y: np.ndarray,
    radii: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inclusive stamp windows, clipped the same way ``_paint`` clips them.

    A window that misses the image is ``−1`` on all four sides. Coordinates
    are absolute pixels with the image origin at ``(0, 0)``.
    """
    coords_x = np.asarray(x, dtype=np.float64)
    coords_y = np.asarray(y, dtype=np.float64)
    stamp_r = np.asarray(radii)
    count = int(coords_x.shape[0])
    row0 = np.full(count, -1, dtype=np.int64)
    row1 = np.full(count, -1, dtype=np.int64)
    col0 = np.full(count, -1, dtype=np.int64)
    col1 = np.full(count, -1, dtype=np.int64)
    for index in range(count):
        radius = int(stamp_r[index])
        if radius <= 0:
            continue
        r_lo = int(math.floor(float(coords_y[index]) - radius)) - 1
        r_hi = int(math.ceil(float(coords_y[index]) + radius)) + 1
        c_lo = int(math.floor(float(coords_x[index]) - radius)) - 1
        c_hi = int(math.ceil(float(coords_x[index]) + radius)) + 1
        if c_lo < 0:
            c_lo = 0
        if r_lo < 0:
            r_lo = 0
        if c_hi >= width:
            c_hi = width - 1
        if r_hi >= height:
            r_hi = height - 1
        if r_lo > r_hi or c_lo > c_hi:
            continue
        row0[index] = r_lo
        row1[index] = r_hi
        col0[index] = c_lo
        col1[index] = c_hi
    return row0, row1, col0, col1


def _wing_radius(flux, sigma_r, sigma_t, halo_fraction, alpha, beta, threshold) -> float:
    sigma_r = max(float(sigma_r), 1e-8)
    sigma_t = max(float(sigma_t), 1e-8)
    sigma_max = max(sigma_r, sigma_t)
    if threshold <= 0.0:
        # No noise floor: keep a wide core so the stamp is not empty.
        return 8.0 * sigma_max
    amp_g = (1.0 - halo_fraction) * flux / (2.0 * math.pi * sigma_r * sigma_t)
    if amp_g <= threshold:
        gauss_r = 0.0
    else:
        gauss_r = sigma_max * math.sqrt(2.0 * math.log(amp_g / threshold))
    amp_m = halo_fraction * flux * (beta - 1.0) / (math.pi * alpha * alpha)
    if halo_fraction <= 0.0 or amp_m <= threshold:
        moffat_r = 0.0
    else:
        ratio = (amp_m / threshold) ** (1.0 / beta)
        moffat_r = alpha * math.sqrt(max(ratio - 1.0, 0.0))
    return max(gauss_r, moffat_r)


def _erf_span(center: float, sigma: float, left: float, right: float) -> float:
    scale = sigma * math.sqrt(2.0)
    return 0.5 * (math.erf((right - center) / scale) - math.erf((left - center) / scale))


def _require_psf(psf: PsfModel) -> None:
    if not isinstance(psf, PsfModel):
        raise TypeError("psf must be starsim.psf.PsfModel")


@njit(cache=True)
def _axes_nb(x, y, cx, cy, r_max, sigma0, field_growth, e_max):
    dx = x - cx
    dy = y - cy
    r2 = dx * dx + dy * dy
    r = math.sqrt(r2)
    rho2 = r2 / (r_max * r_max)
    sigma_iso = sigma0 * (1.0 + field_growth * rho2)
    ellipticity = e_max * rho2
    if ellipticity > e_max:
        ellipticity = e_max
    sigma_r = sigma_iso / math.sqrt(1.0 - ellipticity)
    sigma_t = sigma_iso * math.sqrt(1.0 - ellipticity)
    if r < 1e-12:
        return sigma_r, sigma_t, sigma_iso, 1.0, 0.0
    return sigma_r, sigma_t, sigma_iso, dx / r, dy / r


@njit(cache=True)
def integrate_pixel(
    abs_row,
    abs_col,
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
):
    """5× supersampled integral of one source over one pixel.

    ``_paint`` and the saturation-owner pass both call this, so the two
    electron contributions match bit for bit. The pixel is centered at
    ``(abs_col, abs_row)``.
    """
    nsub = 5
    area = 25.0
    acc = 0.0
    for iy in range(nsub):
        sy = abs_row + (iy + 0.5) / nsub - 0.5
        dy = sy - y
        for ix in range(nsub):
            sx = abs_col + (ix + 0.5) / nsub - 0.5
            dx = sx - x
            parallel = dx * radial_x + dy * radial_y
            perpendicular = -dx * radial_y + dy * radial_x
            core = math.exp(
                -0.5
                * (
                    (parallel / sigma_r) * (parallel / sigma_r)
                    + (perpendicular / sigma_t) * (perpendicular / sigma_t)
                )
            )
            radius2 = dx * dx + dy * dy
            wing = (1.0 + radius2 * inv_alpha2) ** (-beta)
            acc += inv_core * core + inv_halo * wing
    return flux * acc / area


@njit(cache=True)
def _paint(
    image,
    origin_row,
    origin_col,
    x,
    y,
    flux,
    radius,
    cx,
    cy,
    r_max,
    sigma0,
    field_growth,
    e_max,
    halo,
    alpha_per_sigma,
    beta,
    peak_row,
    peak_col,
    s_self,
    box_row0,
    box_row1,
    box_col0,
    box_col1,
):
    height, width = image.shape
    two_pi = 2.0 * math.pi
    for i in range(x.shape[0]):
        peak_row[i] = -1
        peak_col[i] = -1
        s_self[i] = 0.0
        box_row0[i] = -1
        box_row1[i] = -1
        box_col0[i] = -1
        box_col1[i] = -1
        if not math.isfinite(flux[i]) or flux[i] <= 0.0 or radius[i] <= 0:
            continue
        sigma_r, sigma_t, sigma_iso, radial_x, radial_y = _axes_nb(
            x[i], y[i], cx, cy, r_max, sigma0, field_growth, e_max
        )
        alpha = alpha_per_sigma * sigma_iso
        inv_core = (1.0 - halo) / (two_pi * sigma_r * sigma_t)
        inv_halo = halo * (beta - 1.0) / (math.pi * alpha * alpha)
        inv_alpha2 = 1.0 / (alpha * alpha)
        c0 = int(math.floor(x[i] - radius[i])) - 1 - origin_col
        c1 = int(math.ceil(x[i] + radius[i])) + 1 - origin_col
        r0 = int(math.floor(y[i] - radius[i])) - 1 - origin_row
        r1 = int(math.ceil(y[i] + radius[i])) + 1 - origin_row
        if c0 < 0:
            c0 = 0
        if r0 < 0:
            r0 = 0
        if c1 >= width:
            c1 = width - 1
        if r1 >= height:
            r1 = height - 1
        if r0 > r1 or c0 > c1:
            continue
        box_row0[i] = r0 + origin_row
        box_row1[i] = r1 + origin_row
        box_col0[i] = c0 + origin_col
        box_col1[i] = c1 + origin_col
        best = -1.0
        best_r = -1
        best_c = -1
        for row in range(r0, r1 + 1):
            abs_row = row + origin_row
            for col in range(c0, c1 + 1):
                abs_col = col + origin_col
                value = integrate_pixel(
                    abs_row,
                    abs_col,
                    x[i],
                    y[i],
                    flux[i],
                    sigma_r,
                    sigma_t,
                    radial_x,
                    radial_y,
                    inv_core,
                    inv_halo,
                    inv_alpha2,
                    beta,
                )
                image[row, col] += value
                if value > best:
                    best = value
                    best_r = abs_row
                    best_c = abs_col
        if best >= 0.0:
            peak_row[i] = best_r
            peak_col[i] = best_c
            s_self[i] = best


@njit(cache=True)
def _value_at_pixel(
    row,
    col,
    height,
    width,
    x,
    y,
    flux,
    radius,
    cx,
    cy,
    r_max,
    sigma0,
    field_growth,
    e_max,
    halo,
    alpha_per_sigma,
    beta,
):
    """Replay ``_paint``'s additions at one pixel. Origin is (0, 0)."""
    if row < 0 or col < 0 or row >= height or col >= width:
        return 0.0
    total = 0.0
    two_pi = 2.0 * math.pi
    for i in range(x.shape[0]):
        if not math.isfinite(flux[i]) or flux[i] <= 0.0 or radius[i] <= 0:
            continue
        sigma_r, sigma_t, sigma_iso, radial_x, radial_y = _axes_nb(
            x[i], y[i], cx, cy, r_max, sigma0, field_growth, e_max
        )
        alpha = alpha_per_sigma * sigma_iso
        inv_core = (1.0 - halo) / (two_pi * sigma_r * sigma_t)
        inv_halo = halo * (beta - 1.0) / (math.pi * alpha * alpha)
        inv_alpha2 = 1.0 / (alpha * alpha)
        c0 = int(math.floor(x[i] - radius[i])) - 1
        c1 = int(math.ceil(x[i] + radius[i])) + 1
        r0 = int(math.floor(y[i] - radius[i])) - 1
        r1 = int(math.ceil(y[i] + radius[i])) + 1
        if c0 < 0:
            c0 = 0
        if r0 < 0:
            r0 = 0
        if c1 >= width:
            c1 = width - 1
        if r1 >= height:
            r1 = height - 1
        if r0 > r1 or c0 > c1:
            continue
        if row < r0 or row > r1 or col < c0 or col > c1:
            continue
        total += integrate_pixel(
            row,
            col,
            x[i],
            y[i],
            flux[i],
            sigma_r,
            sigma_t,
            radial_x,
            radial_y,
            inv_core,
            inv_halo,
            inv_alpha2,
            beta,
        )
    return total
