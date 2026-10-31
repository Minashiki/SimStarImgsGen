"""Field-dependent PSF: elliptical Gaussian core plus an optional Moffat halo.

The core at the field center is circular with width ``sigma0``. Across the
field

    σ(r) = σ₀ · (1 + k_r · (r / r_max)²)

and the core is stretched along the radial direction. Ellipticity grows as
``e(r) = e_max · (r / r_max)²`` and is capped at ``e_max``, so the corner
value is the configured maximum (at most 0.2 in the default camera).
``e = 1 − σ_tangential / σ_radial`` and the product ``σ_r · σ_t`` stays
``σ(r)²``, which keeps the core normalized.

``r_max`` is the distance from the principal point to a corner of the image
coverage rectangle ``[−0.5, W−0.5) × [−0.5, H−0.5)``.

The halo is an isotropic Moffat profile with ``β = 3`` and
``α = 4 · σ(r)``. ``halo_fraction`` of the flux sits in the halo and the
rest in the core. Both pieces are normalized to unit integral, so the sum
is a unit-flux PSF.

``sigma_along`` returns the local core width in a requested direction:

    σ_dir² = σ_r² cos²φ + σ_t² sin²φ

``φ`` is the angle between that direction and the outward radial vector.
At the principal point the radial direction is undefined and every direction
returns ``σ₀``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from starsim.camera import Camera

# Behavior version for the PSF formulas. Increment when a legal PSF starts
# producing different pixels, peak fractions, or sigma_along values.
PSF_VERSION = 1

MOFFAT_BETA = 3.0
MOFFAT_ALPHA_PER_SIGMA = 4.0


@dataclass(frozen=True)
class PsfModel:
    """PSF parameters shared by every source in one sequence."""

    sigma0: float
    field_growth: float
    max_ellipticity: float
    halo_fraction: float
    width: int
    height: int
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if type(self.width) is not int or type(self.height) is not int:
            raise TypeError("width and height must be int")
        if self.width < 1 or self.height < 1:
            raise ValueError("width and height must be >= 1")
        for name, value in (
            ("sigma0", self.sigma0),
            ("field_growth", self.field_growth),
            ("max_ellipticity", self.max_ellipticity),
            ("halo_fraction", self.halo_fraction),
            ("cx", self.cx),
            ("cy", self.cy),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
                raise TypeError(f"{name} must be a real number, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.sigma0 <= 0.0:
            raise ValueError(f"sigma0 must be > 0, got {self.sigma0}")
        if self.field_growth < 0.0:
            raise ValueError(f"field_growth must be >= 0, got {self.field_growth}")
        if not 0.0 <= self.max_ellipticity < 1.0:
            raise ValueError(f"max_ellipticity must be in [0, 1), got {self.max_ellipticity}")
        if not 0.0 <= self.halo_fraction <= 1.0:
            raise ValueError(f"halo_fraction must be in [0, 1], got {self.halo_fraction}")

    @classmethod
    def from_effective(cls, psf_cfg: dict, camera: Camera) -> PsfModel:
        """Build a model from an already-sampled ``psf`` block and a camera."""
        if not isinstance(camera, Camera):
            raise TypeError("camera must be starsim.camera.Camera")
        if not isinstance(psf_cfg, dict):
            raise TypeError("psf_cfg must be a mapping")
        return cls(
            sigma0=float(psf_cfg["sigma_px"]),
            field_growth=float(psf_cfg["field_growth"]),
            max_ellipticity=float(psf_cfg["max_ellipticity"]),
            halo_fraction=float(psf_cfg["halo_fraction"]),
            width=int(camera.width),
            height=int(camera.height),
            cx=float(camera.cx),
            cy=float(camera.cy),
        )

    @property
    def r_max(self) -> float:
        """Distance from the principal point to a coverage corner, in pixels."""
        return 0.5 * math.hypot(self.width, self.height)

    def sigma_iso(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
        """Geometric-mean core width ``σ(r)`` at one or many positions."""
        scalar = np.isscalar(x) and np.isscalar(y)
        xx = np.asarray(x, dtype=np.float64)
        yy = np.asarray(y, dtype=np.float64)
        rho2 = ((xx - self.cx) ** 2 + (yy - self.cy) ** 2) / (self.r_max ** 2)
        sigma = self.sigma0 * (1.0 + self.field_growth * rho2)
        if scalar:
            return float(sigma)
        return sigma

    def axes(self, x: float, y: float) -> tuple[float, float, float, float, float]:
        """``(σ_r, σ_t, σ_iso, radial_x, radial_y)`` at one source position.

        The radial unit vector points away from the principal point. At the
        principal point it is ``(1, 0)`` and ``σ_r = σ_t = σ₀``.
        """
        dx = float(x) - self.cx
        dy = float(y) - self.cy
        r2 = dx * dx + dy * dy
        r = math.sqrt(r2)
        rho2 = r2 / (self.r_max ** 2)
        sigma_iso = self.sigma0 * (1.0 + self.field_growth * rho2)
        ellipticity = self.max_ellipticity * rho2
        if ellipticity > self.max_ellipticity:
            ellipticity = self.max_ellipticity
        # e_max < 1, so the denominator stays positive even at the cap.
        sigma_r = sigma_iso / math.sqrt(1.0 - ellipticity)
        sigma_t = sigma_iso * math.sqrt(1.0 - ellipticity)
        if r < 1e-12:
            return sigma_r, sigma_t, sigma_iso, 1.0, 0.0
        return sigma_r, sigma_t, sigma_iso, dx / r, dy / r

    def sigma_along(self, x: float, y: float, direction: np.ndarray | float) -> float:
        """Core σ in ``direction`` at pixel ``(x, y)``.

        ``direction`` is either an angle in radians, measured from +x toward
        +y, or a 2-vector. The vector need not be a unit vector.
        """
        sigma_r, sigma_t, _sigma_iso, radial_x, radial_y = self.axes(x, y)
        dir_x, dir_y = _unit_direction(direction)
        cos_phi = dir_x * radial_x + dir_y * radial_y
        sin2 = max(0.0, 1.0 - cos_phi * cos_phi)
        return math.sqrt(sigma_r * sigma_r * cos_phi * cos_phi + sigma_t * sigma_t * sin2)

    def moffat_alpha(self, sigma_iso: float) -> float:
        return MOFFAT_ALPHA_PER_SIGMA * float(sigma_iso)

    def evaluate(
        self,
        x: np.ndarray | float,
        y: np.ndarray | float,
        x0: float,
        y0: float,
    ) -> np.ndarray | float:
        """Unit-flux surface brightness at ``(x, y)`` for a source at ``(x0, y0)``.

        Ellipticity and the Moffat scale are taken at the source, and held
        fixed across its stamp. The return value is electrons per pixel for
        a unit-flux source (the pixel area is 1).
        """
        scalar = np.isscalar(x) and np.isscalar(y)
        xx = np.asarray(x, dtype=np.float64)
        yy = np.asarray(y, dtype=np.float64)
        sigma_r, sigma_t, sigma_iso, radial_x, radial_y = self.axes(x0, y0)
        dx = xx - float(x0)
        dy = yy - float(y0)
        parallel = dx * radial_x + dy * radial_y
        perpendicular = -dx * radial_y + dy * radial_x
        gauss = np.exp(-0.5 * ((parallel / sigma_r) ** 2 + (perpendicular / sigma_t) ** 2))
        gauss = gauss / (2.0 * math.pi * sigma_r * sigma_t)
        alpha = self.moffat_alpha(sigma_iso)
        radius2 = dx * dx + dy * dy
        moffat = (MOFFAT_BETA - 1.0) / (math.pi * alpha * alpha)
        moffat = moffat * (1.0 + radius2 / (alpha * alpha)) ** (-MOFFAT_BETA)
        brightness = (1.0 - self.halo_fraction) * gauss + self.halo_fraction * moffat
        if scalar:
            return float(brightness)
        return brightness

    def evaluate_at_sources(
        self,
        sample_x: np.ndarray,
        sample_y: np.ndarray,
        source_x: np.ndarray,
        source_y: np.ndarray,
    ) -> np.ndarray:
        """Unit-flux brightness of every source at every sample.

        ``sample_*`` and ``source_*`` are 1D. The result has shape
        ``(n_source, n_sample)`` and matches :meth:`evaluate` for each source.
        Ellipticity and the Moffat scale are taken at the source.
        """
        sx = np.asarray(sample_x, dtype=np.float64)
        sy = np.asarray(sample_y, dtype=np.float64)
        x0 = np.asarray(source_x, dtype=np.float64)
        y0 = np.asarray(source_y, dtype=np.float64)
        if sx.ndim != 1 or sy.shape != sx.shape:
            raise ValueError("sample_x and sample_y must be 1D arrays of the same length")
        if x0.ndim != 1 or y0.shape != x0.shape:
            raise ValueError("source_x and source_y must be 1D arrays of the same length")
        if x0.size == 0 or sx.size == 0:
            return np.zeros((x0.size, sx.size), dtype=np.float64)
        dx0 = x0 - self.cx
        dy0 = y0 - self.cy
        r2 = dx0 * dx0 + dy0 * dy0
        r = np.sqrt(r2)
        rho2 = r2 / (self.r_max ** 2)
        sigma_iso = self.sigma0 * (1.0 + self.field_growth * rho2)
        ellipticity = np.minimum(self.max_ellipticity * rho2, self.max_ellipticity)
        sigma_r = sigma_iso / np.sqrt(1.0 - ellipticity)
        sigma_t = sigma_iso * np.sqrt(1.0 - ellipticity)
        radial_x = np.ones(x0.shape, dtype=np.float64)
        radial_y = np.zeros(x0.shape, dtype=np.float64)
        near = r >= 1e-12
        np.divide(dx0, r, out=radial_x, where=near)
        np.divide(dy0, r, out=radial_y, where=near)
        dx = sx[None, :] - x0[:, None]
        dy = sy[None, :] - y0[:, None]
        parallel = dx * radial_x[:, None] + dy * radial_y[:, None]
        perpendicular = -dx * radial_y[:, None] + dy * radial_x[:, None]
        gauss = np.exp(-0.5 * (
            (parallel / sigma_r[:, None]) ** 2 + (perpendicular / sigma_t[:, None]) ** 2
        ))
        gauss = gauss / (2.0 * math.pi * sigma_r[:, None] * sigma_t[:, None])
        alpha = MOFFAT_ALPHA_PER_SIGMA * sigma_iso
        radius2 = dx * dx + dy * dy
        moffat = (MOFFAT_BETA - 1.0) / (math.pi * alpha[:, None] * alpha[:, None])
        moffat = moffat * (1.0 + radius2 / (alpha[:, None] ** 2)) ** (-MOFFAT_BETA)
        return (1.0 - self.halo_fraction) * gauss + self.halo_fraction * moffat


def _unit_direction(direction: np.ndarray | float) -> tuple[float, float]:
    if isinstance(direction, bool) or isinstance(direction, (int, float, np.floating)):
        angle = float(direction)
        if not math.isfinite(angle):
            raise ValueError(f"direction angle must be finite, got {direction!r}")
        return math.cos(angle), math.sin(angle)
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"direction must be an angle or a finite 2-vector, got {direction!r}")
    norm = float(math.hypot(float(vector[0]), float(vector[1])))
    if norm == 0.0:
        raise ValueError("direction vector must be non-zero")
    return float(vector[0] / norm), float(vector[1] / norm)
