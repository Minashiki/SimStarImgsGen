"""Pinhole camera: gnomonic projection, inter-frame homography, TAN WCS.

Pixels follow plan section 2. The center of pixel ``(row=r, col=c)`` is the
continuous coordinate ``(x=c, y=r)``. The image covers
``[−0.5, W−0.5) × [−0.5, H−0.5)``.

Projection, with the principal point at the image center::

    x = cx + (f/p) · X/Z
    y = cy + (f/p) · Y/Z

The homography from frame ``k`` to the reference frame is
``H = K · R_ref · R_kᵀ · K⁻¹``. ``Z`` of a camera-frame direction is an affine
function of pixel coordinates, so ``Z > 0`` at all four image corners means
the whole rectangle is in front of that camera. ``H[2, 2]`` is proportional
to ``Z`` of pixel ``(0, 0)``, which then lies inside the rectangle, and
dividing by ``H[2, 2]`` is legal. That test does not bound the union-canvas
size; size and memory limits belong with the canvas implementation.

The TAN header uses the same gnomonic map. ``CRPIX`` is 1-based
``(cx+1, cy+1)``. ``CD`` takes a pixel offset to the tangent-plane
coordinates ``(ξ, η)`` in degrees, with ``ξ`` toward east and ``η`` toward
north, and ``LONPOLE`` is fixed at 180° so the north celestial pole does not
flip the native frame. For this camera ``det(CD) > 0``. The opposite sign is
a mirror, and astropy would then disagree with :func:`project` by a field-sized
error.

Two different up-down flips (plan section 2):

- Changing only the display orientation leaves pixel data and WCS unchanged.
- Replacing the array by ``y' = H_img−1−y`` sends ``Y`` to ``−Y``.
  ``H_img`` is the image height, not the homography ``H``. The frame
  becomes left-handed, and no rotation ``R`` represents it. ``CRPIX2`` is
  unchanged because ``cy = (H_img−1)/2``. The second column of ``CD`` changes
  sign, so ``det(CD) < 0``. The homography is
  ``H' = F_ref H F_source⁻¹``; when both frames have the same size and the
  same flip, ``F⁻¹ = F`` and this is ``F H F``. Labels use the same map.
- Flipping up-down and left-right together is ``roll + π``, a real rotation,
  not a mirror.

Viewer defaults are not stated here. M4 checks them against real FITS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from starsim.attitude import boresight_radec

if TYPE_CHECKING:
    from astropy.wcs import WCS
from starsim.config import DerivedCamera


@dataclass
class Camera:
    """Intrinsic model shared by every frame of a sequence."""

    width: int
    height: int
    f_over_p: float
    cx: float
    cy: float
    K: np.ndarray
    K_inv: np.ndarray

    @classmethod
    def from_derived(cls, derived: DerivedCamera) -> Camera:
        if not isinstance(derived, DerivedCamera):
            raise TypeError("derived must be starsim.config.DerivedCamera")
        f_over_p = float(derived.f_over_p)
        cx = float(derived.cx)
        cy = float(derived.cy)
        intrinsic = np.array(
            [
                [f_over_p, 0.0, cx],
                [0.0, f_over_p, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        inverse = np.array(
            [
                [1.0 / f_over_p, 0.0, -cx / f_over_p],
                [0.0, 1.0 / f_over_p, -cy / f_over_p],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return cls(
            width=int(derived.width),
            height=int(derived.height),
            f_over_p=f_over_p,
            cx=cx,
            cy=cy,
            K=intrinsic,
            K_inv=inverse,
        )


def project(camera: Camera, v_cam: np.ndarray) -> np.ndarray:
    """Gnomonic projection of camera-frame vectors.

    ``v_cam`` has shape ``(3,)`` or ``(N, 3)`` and need not be a unit vector.
    The return shape is ``(2,)`` or ``(N, 2)`` in ``(x, y)`` = ``(col, row)``.
    Directions on or behind the focal plane (``Z ≤ 0``) raise ``ValueError``.
    """
    _require_camera(camera)
    vectors, single = _row_vectors(v_cam, "v_cam")
    z = vectors[:, 2]
    if np.any(z <= 0.0):
        raise ValueError("projection requires Z > 0")
    xy = np.column_stack(
        (
            camera.cx + camera.f_over_p * vectors[:, 0] / z,
            camera.cy + camera.f_over_p * vectors[:, 1] / z,
        )
    )
    return xy[0] if single else xy


def unproject(camera: Camera, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    """Pixel ``(x, y)`` → unit vector in the camera frame.

    Scalars return shape ``(3,)``. Arrays of the same shape return
    ``(..., 3)``.
    """
    _require_camera(camera)
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"x and y shapes differ: {x_arr.shape} vs {y_arr.shape}")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("pixel coordinates must be finite")
    vx = (x_arr - camera.cx) / camera.f_over_p
    vy = (y_arr - camera.cy) / camera.f_over_p
    ones = np.ones_like(vx)
    stacked = np.stack((vx, vy, ones), axis=-1)
    stacked /= np.linalg.norm(stacked, axis=-1, keepdims=True)
    return stacked


def project_icrs(camera: Camera, R: np.ndarray, v_icrs: np.ndarray) -> np.ndarray:
    """Project ICRS directions through attitude ``R`` (ICRS → camera)."""
    matrix = _rotation_like(R)
    vectors, single = _row_vectors(v_icrs, "v_icrs")
    if single:
        return project(camera, matrix @ vectors[0])
    return project(camera, vectors @ matrix.T)


def unproject_icrs(
    camera: Camera,
    R: np.ndarray,
    x: np.ndarray | float,
    y: np.ndarray | float,
) -> np.ndarray:
    """Pixel → unit ICRS direction. ``R`` maps ICRS to the camera."""
    matrix = _rotation_like(R)
    v_cam = unproject(camera, x, y)
    if v_cam.ndim == 1:
        return matrix.T @ v_cam
    return v_cam @ matrix


def homography(camera: Camera, R_from: np.ndarray, R_to: np.ndarray) -> np.ndarray:
    """Map pixels of the ``from`` frame onto pixels of the ``to`` frame.

    ``H_k→ref`` is ``homography(camera, R_k, R_ref)``. The matrix is scaled so
    that ``H[2, 2] == 1`` when that element is non-zero. A rotation-induced
    homography is always invertible. ``H[2, 2] == 0`` only means this
    normalization does not apply; it is not a singular map.
    """
    _require_camera(camera)
    source = _rotation_like(R_from)
    target = _rotation_like(R_to)
    # Identical attitudes are exactly the identity. Building K R Rᵀ K⁻¹ leaves
    # a 1e-12 px translation because R Rᵀ is not bitwise I.
    if np.array_equal(source, target):
        return np.eye(3, dtype=np.float64)
    mapped = camera.K @ target @ source.T @ camera.K_inv
    pivot = float(mapped[2, 2])
    if not math.isfinite(pivot):
        raise ValueError(f"homography normalization is not finite, H[2, 2]={pivot}")
    if pivot == 0.0:
        raise ValueError(f"归一化方式不适用，H[2, 2]={pivot}")
    return mapped / pivot


def apply_homography(H: np.ndarray, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
    """Apply a 3×3 homography. Scalars return ``(x, y)``; arrays return ``(..., 2)``."""
    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("H must be a finite 3x3 matrix")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"x and y shapes differ: {x_arr.shape} vs {y_arr.shape}")
    ones = np.ones_like(x_arr)
    packed = np.stack((x_arr, y_arr, ones), axis=-1)
    mapped = packed @ matrix.T
    z = mapped[..., 2]
    if np.any(z == 0.0) or not np.all(np.isfinite(z)):
        raise ValueError("homography sent a point to infinity")
    xy = mapped[..., :2] / z[..., None]
    return xy


def tangent_plane_cd(R: np.ndarray, f_over_p: float) -> tuple[np.ndarray, float, float]:
    """``CD`` in degrees per pixel, plus ``CRVAL`` ``(α, δ)`` in degrees.

    ``CD @ [Δx, Δy]`` is the tangent-plane offset ``(ξ, η)``, with ``ξ``
    toward east and ``η`` toward north. ``det(CD) > 0`` for this camera.
    """
    if f_over_p <= 0.0 or not math.isfinite(f_over_p):
        raise ValueError(f"f_over_p must be positive, got {f_over_p}")
    matrix = _rotation_like(R)
    ra_rad, dec_rad = boresight_radec(matrix)
    east = np.array([-math.sin(ra_rad), math.cos(ra_rad), 0.0], dtype=np.float64)
    north = np.array(
        [
            -math.sin(dec_rad) * math.cos(ra_rad),
            -math.sin(dec_rad) * math.sin(ra_rad),
            math.cos(dec_rad),
        ],
        dtype=np.float64,
    )
    scale = math.degrees(1.0 / f_over_p)
    cam_x = matrix[0]
    cam_y = matrix[1]
    cd = scale * np.array(
        [
            [float(np.dot(cam_x, east)), float(np.dot(cam_y, east))],
            [float(np.dot(cam_x, north)), float(np.dot(cam_y, north))],
        ],
        dtype=np.float64,
    )
    ra_deg = math.degrees(ra_rad) % 360.0
    dec_deg = math.degrees(dec_rad)
    return cd, ra_deg, dec_deg


def wcs_keywords(camera: Camera, R: np.ndarray) -> dict:
    """FITS WCS cards for one frame. ``CRPIX`` is 1-based."""
    _require_camera(camera)
    cd, ra_deg, dec_deg = tangent_plane_cd(R, camera.f_over_p)
    return {
        "WCSAXES": 2,
        "CRPIX1": camera.cx + 1.0,
        "CRPIX2": camera.cy + 1.0,
        "CRVAL1": ra_deg,
        "CRVAL2": dec_deg,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CD1_1": float(cd[0, 0]),
        "CD1_2": float(cd[0, 1]),
        "CD2_1": float(cd[1, 0]),
        "CD2_2": float(cd[1, 1]),
        "LONPOLE": 180.0,
        "RADESYS": "ICRS",
    }


def wcs_from_attitude(camera: Camera, R: np.ndarray) -> WCS:
    """astropy TAN WCS equivalent to :func:`project` for this attitude.

    astropy is imported here, so projection and homography do not load the WCS
    dependency. This does not remove an environment-level DLL failure once
    this function is called. ``wcs_world2pix(..., 0)`` returns this project's
    0-based ``(x, y)``.
    """
    from astropy.wcs import WCS

    cards = wcs_keywords(camera, R)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [cards["CRPIX1"], cards["CRPIX2"]]
    wcs.wcs.crval = [cards["CRVAL1"], cards["CRVAL2"]]
    wcs.wcs.ctype = [cards["CTYPE1"], cards["CTYPE2"]]
    wcs.wcs.cunit = [cards["CUNIT1"], cards["CUNIT2"]]
    wcs.wcs.cd = np.array(
        [
            [cards["CD1_1"], cards["CD1_2"]],
            [cards["CD2_1"], cards["CD2_2"]],
        ],
        dtype=np.float64,
    )
    wcs.wcs.lonpole = cards["LONPOLE"]
    wcs.wcs.radesys = cards["RADESYS"]
    return wcs


def reference_index(n_frames: int, spec: object) -> int:
    """Frame index of the reference. ``"middle"`` is ``⌊N/2⌋``."""
    if isinstance(n_frames, bool) or not isinstance(n_frames, (int, np.integer)):
        raise TypeError(f"n_frames must be an integer >= 2, got {n_frames!r}")
    count = int(n_frames)
    if count < 2:
        raise ValueError(f"n_frames must be >= 2, got {count}")
    if spec == "middle":
        return count // 2
    if type(spec) is not int:
        raise TypeError(f"reference_frame must be 'middle' or int, got {spec!r}")
    if spec < 0 or spec >= count:
        raise ValueError(f"reference_frame must satisfy 0 <= index < {count}, got {spec}")
    return spec


def _require_camera(camera: Camera) -> None:
    if not isinstance(camera, Camera):
        raise TypeError("camera must be starsim.camera.Camera")


def _rotation_like(R: np.ndarray) -> np.ndarray:
    matrix = np.asarray(R, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("R must be a finite 3x3 matrix")
    return matrix


def _row_vectors(values: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1 and array.shape == (3,):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        return array.reshape(1, 3), True
    if array.ndim == 2 and array.shape[1] == 3 and array.shape[0] >= 1:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
        return array, False
    raise ValueError(f"{name} must have shape (3,) or (N, 3), got {array.shape}")
