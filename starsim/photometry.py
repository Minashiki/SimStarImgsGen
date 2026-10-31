"""SNR-anchored photometry and the local-noise definition from plan section 4.3.

Each sequence samples its noise parameters first. The reference background
noise ``σ_ref`` is the electron noise at the image center. The field-center
PSF peak fraction ``pf`` is the brightest pixel's share of a unit-flux star,
averaged over subpixel phases. Fluxes then follow

    F(m_lim) = snr_at_limit · σ_ref / pf
    F(m)     = F(m_lim) · 10^(−0.4 · (m − m_lim))

The equivalent zero point is ``ZP = m_lim + 2.5·log10(F(m_lim))``. Catalog V
magnitudes are used as instrument magnitudes: the catalog has no color, and
this module does not apply a color term.

Local noise at a source's peak pixel, in electrons, is

    σ²_local = E(peak) − s_self + dark + read²

``E`` already contains sky, vignetting, PRNU, and every source. Subtracting
``s_self`` removes that source's own shot noise. Dark current and read noise
are added here because they are not part of ``E``. The single-frame peak SNR
is ``s_self / σ_local``. The effective SNR over a window is
``√(Σ SNR_k²)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Behavior version for the anchoring formulas and the SNR definitions.
# Increment when a legal sequence starts assigning different fluxes or SNRs.
PHOTOMETRY_VERSION = 1


@dataclass(frozen=True)
class Photometry:
    """Per-sequence flux scale. Magnitudes are catalog V used as instrument mag."""

    mag_limit: float
    snr_at_limit: float
    render_mag_margin: float
    sigma_ref: float
    peak_fraction: float
    flux_at_limit: float
    zero_point: float

    def flux(self, mag: np.ndarray | float) -> np.ndarray | float:
        """Electrons in one nominal exposure for a source of magnitude ``mag``."""
        scalar = np.isscalar(mag)
        values = np.asarray(mag, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("magnitude must be finite")
        scaled = self.flux_at_limit * np.power(10.0, -0.4 * (values - self.mag_limit))
        if scalar:
            return float(scaled)
        return scaled

    def magnitude(self, flux_e: np.ndarray | float) -> np.ndarray | float:
        """Inverse of :meth:`flux`. ``flux_e`` must be positive."""
        scalar = np.isscalar(flux_e)
        values = np.asarray(flux_e, dtype=np.float64)
        if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
            raise ValueError("flux must be finite and positive")
        mag = self.zero_point - 2.5 * np.log10(values)
        if scalar:
            return float(mag)
        return mag

    @property
    def render_mag_max(self) -> float:
        """Stars at or brighter than this magnitude are rendered."""
        return self.mag_limit + self.render_mag_margin


def anchor_photometry(
    mag_limit: float,
    snr_at_limit: float,
    render_mag_margin: float,
    sigma_ref: float,
    peak_fraction: float,
) -> Photometry:
    """Build the flux scale from the reference noise and the PSF peak fraction."""
    mag_limit = _positive_finite("mag_limit", mag_limit, allow_zero=True)
    snr_at_limit = _positive_finite("snr_at_limit", snr_at_limit, allow_zero=False)
    render_mag_margin = _positive_finite("render_mag_margin", render_mag_margin, allow_zero=True)
    sigma_ref = _positive_finite("sigma_ref", sigma_ref, allow_zero=False)
    peak_fraction = _positive_finite("peak_fraction", peak_fraction, allow_zero=False)
    if peak_fraction > 1.0:
        raise ValueError(f"peak_fraction must be <= 1, got {peak_fraction}")
    flux_at_limit = snr_at_limit * sigma_ref / peak_fraction
    zero_point = mag_limit + 2.5 * math.log10(flux_at_limit)
    return Photometry(
        mag_limit=mag_limit,
        snr_at_limit=snr_at_limit,
        render_mag_margin=render_mag_margin,
        sigma_ref=sigma_ref,
        peak_fraction=peak_fraction,
        flux_at_limit=flux_at_limit,
        zero_point=zero_point,
    )


def local_variance(expectation_e: float, s_self: float, dark_e: float, read_e: float) -> float:
    """``E − s_self + dark + read²``. A tiny negative from roundoff becomes 0."""
    expectation_e = _finite("expectation_e", expectation_e)
    s_self = _finite("s_self", s_self)
    dark_e = _finite("dark_e", dark_e)
    read_e = _finite("read_e", read_e)
    if s_self < 0.0 or dark_e < 0.0 or read_e < 0.0 or expectation_e < 0.0:
        raise ValueError("expectation, s_self, dark, and read noise must be >= 0")
    variance = expectation_e - s_self + dark_e + read_e * read_e
    if variance < 0.0:
        if variance > -1e-6 * max(expectation_e, 1.0):
            return 0.0
        raise ValueError(
            "local variance is negative: E - s_self + dark + read^2 = "
            f"{variance}"
        )
    return float(variance)


def local_sigma(expectation_e: float, s_self: float, dark_e: float, read_e: float) -> float:
    """Square root of :func:`local_variance`."""
    return math.sqrt(local_variance(expectation_e, s_self, dark_e, read_e))


def peak_snr(s_self: float, sigma_local: float) -> float:
    """Single-frame peak SNR. Zero noise with a positive signal is infinite."""
    s_self = _finite("s_self", s_self)
    sigma_local = _finite("sigma_local", sigma_local)
    if s_self < 0.0 or sigma_local < 0.0:
        raise ValueError("s_self and sigma_local must be >= 0")
    if sigma_local == 0.0:
        if s_self == 0.0:
            return 0.0
        return math.inf
    return s_self / sigma_local


def effective_snr(snr_k: np.ndarray | float) -> float:
    """``√(Σ SNR_k²)`` over the frames passed in. An empty window is 0.

    The caller selects the window. The default window is the whole sequence;
    a shorter window is a slice of the per-frame SNRs.
    """
    values = np.atleast_1d(np.asarray(snr_k, dtype=np.float64))
    if values.size == 0:
        return 0.0
    if not np.all(np.isfinite(values)):
        raise ValueError("snr values must be finite")
    if np.any(values < 0.0):
        raise ValueError("snr values must be >= 0")
    return float(np.sqrt(np.sum(values * values)))


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _positive_finite(name: str, value: object, *, allow_zero: bool) -> float:
    number = _finite(name, value)
    if allow_zero:
        if number < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value!r}")
    elif number <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return number
