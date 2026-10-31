"""Camera attitude: quaternion, rotation matrix, 3-1-3 Euler, pointing.

Conventions (plan section 2). The quaternion is the primary representation.
Pipeline outputs are read from the rotation matrix rebuilt from that
quaternion (D25). :func:`matrix_to_euler313` itself is stable on a proper
rotation, including one formed as a product of rotations.

Verifiable definition, in this order:

- ``R`` maps ICRS to the camera, ``v_cam = R @ v_icrs``.
- The camera frame is right-handed. Columns increase to the right (+X) and
  rows increase downward (+Y). +Z is the optical axis toward the sky.
- ``roll = 0`` puts image +X on east and image +Y on north.
- Quaternion ``[w, x, y, z]`` is Hamilton, scalar first, and the same map:
  ``v_cam = q * v_icrs * q_conj``.

Looking outward from inside the celestial sphere is a geometric reading of
this same frame, not a second convention.

- 3-1-3 Euler angles use the passive rotations below, so
  ``R = R3(φ) @ R1(θ) @ R3(ψ)`` with ``ψ = α + π/2``, ``θ = π/2 − δ``,
  ``φ = roll``.
- Gimbal lock: when ``sin θ`` is below ``GIMBAL_SIN_THETA``, ``ψ`` is set to
  0 and the whole turn about the optical axis is carried by ``φ``.
"""

from __future__ import annotations

import math

import numpy as np

# Below this, φ and ψ are the same rotation and cannot be separated.
# The matrix rebuilt from the canonical angles (ψ = 0) still matches the
# original rotation to well under 1e-12.
GIMBAL_SIN_THETA = 1e-12

_TWO_PI = 2.0 * math.pi


def wrap_two_pi(angle: float) -> float:
    """Wrap radians into ``[0, 2π)``.

    A floating-point remainder of exactly ``2π`` (``(-1e-17) % 2π`` is the
    usual case) is mapped to 0 so the result stays inside the half-open
    interval.
    """
    wrapped = float(angle) % _TWO_PI
    if wrapped >= _TWO_PI:
        return 0.0
    return wrapped


def wrap_pi(angle: float) -> float:
    """Wrap radians into ``(−π, π]``.

    ``π`` and ``−π`` are the same angle. The representative on that cut is
    ``+π``, so ``wrap_pi(π)`` and ``wrap_pi(−π)`` are both ``π``.
    """
    wrapped = (float(angle) + math.pi) % _TWO_PI - math.pi
    if wrapped <= -math.pi:
        return math.pi
    return wrapped


def icrs_direction(ra_rad: float, dec_rad: float) -> np.ndarray:
    """Unit ICRS vector ``[cosδ cosα, cosδ sinα, sinδ]``."""
    cos_dec = math.cos(dec_rad)
    return np.array(
        [
            cos_dec * math.cos(ra_rad),
            cos_dec * math.sin(ra_rad),
            math.sin(dec_rad),
        ],
        dtype=np.float64,
    )


def direction_to_radec(vector: np.ndarray) -> tuple[float, float]:
    """``(α, δ)`` in radians. At the pole ``α`` is 0 because it is undefined."""
    v = _unit_vector(vector, "vector")
    dec = math.atan2(v[2], math.hypot(v[0], v[1]))
    ra = wrap_two_pi(math.atan2(v[1], v[0]))
    return ra, dec


def R3(angle: float) -> np.ndarray:
    """Passive rotation about axis 3 (the matrix of an active rotation by ``−angle``)."""
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def R1(angle: float) -> np.ndarray:
    """Passive rotation about axis 1."""
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]], dtype=np.float64)


def rotate_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Active Rodrigues rotation: ``v_new = Rot @ v_old``, right-handed about ``axis``."""
    axis = _unit_vector(axis, "axis")
    x, y, z = (float(axis[0]), float(axis[1]), float(axis[2]))
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def rotation_vector(R: np.ndarray) -> np.ndarray:
    """Principal rotation vector of an active rotation, magnitude in ``[0, π]``.

    ``v_new = Rot @ v_old`` rotates ``v_old`` by this vector (right-hand rule).
    The input must pass :func:`matrix_to_quaternion`'s rotation check
    (proper rotation: determinant and orthonormality), which is stricter than
    a shape check. The vector is ``2·atan2(‖v‖, w)·v/‖v‖`` from the Hamilton
    quaternion ``[w, x, y, z]``. Normalization keeps ``w ≥ 0``, so the angle
    lies in ``[0, π]``. ``‖v‖ == 0`` returns the zero vector. An exact half
    turn has magnitude ``π``; the quaternion's sign convention fixes the axis,
    and no separate axis extraction is required.
    """
    quat = matrix_to_quaternion(R)
    vector = np.array(quat[1:], dtype=np.float64, copy=True)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(norm, float(quat[0]))
    return vector * (angle / norm)


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Hamilton ``[w, x, y, z]`` → ``R`` with ``v_cam = R @ v_icrs``.

    The input is normalized. ``q`` and ``−q`` are the same rotation.
    """
    quat = np.asarray(q, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quat.shape}")
    if not np.all(np.isfinite(quat)):
        raise ValueError("quaternion must be finite")
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        raise ValueError("quaternion must be non-zero")
    w, x, y, z = (float(item) / norm for item in quat)
    return np.array(
        [
            [w * w + x * x - y * y - z * z, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), w * w - x * x + y * y - z * z, 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """``R`` → unit Hamilton quaternion ``[w, x, y, z]`` with ``w > 0``.

    ``w == 0`` is broken by the first non-zero vector component, so the sign
    is unique. Shepperd's branch keeps the result stable for every 180° turn.
    """
    matrix = _require_rotation(R)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0 or (
        quat[0] == 0.0
        and (
            quat[1] < 0.0
            or (quat[1] == 0.0 and (quat[2] < 0.0 or (quat[2] == 0.0 and quat[3] < 0.0)))
        )
    ):
        quat = -quat
    return quat


def euler313_to_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """``R = R3(φ) @ R1(θ) @ R3(ψ)``. Angles are radians."""
    _require_finite_angle(phi, "phi")
    _require_finite_angle(theta, "theta")
    _require_finite_angle(psi, "psi")
    return R3(phi) @ R1(theta) @ R3(psi)


def matrix_to_euler313(R: np.ndarray) -> tuple[float, float, float]:
    """Return ``(φ, θ, ψ)`` in radians.

    ``θ`` is in ``[0, π]`` and ``sin θ ≥ 0``. ``φ`` and ``ψ`` are in
    ``[0, 2π)``. When ``sin θ < GIMBAL_SIN_THETA``, ``ψ`` is 0 and ``φ``
    holds ``φ+ψ`` at the north pole or ``φ−ψ`` at the south pole.

    ``θ`` and ``ψ`` come from the third row. ``φ`` comes from the upper-left
    block of the residual ``R · (R1(θ) R3(ψ))ᵀ``, which is ``R3(φ)`` for a
    proper rotation. A matrix that only clears the 1e-6 orthogonality
    tolerance is not guaranteed to rebuild a true rotation to 1e-12.
    """
    matrix = _require_rotation(R)
    sin_theta = math.hypot(float(matrix[2, 0]), float(matrix[2, 1]))
    cos_theta = float(matrix[2, 2])
    if sin_theta < GIMBAL_SIN_THETA:
        psi = 0.0
        theta = 0.0 if cos_theta >= 0.0 else math.pi
    else:
        theta = math.atan2(sin_theta, cos_theta)
        psi = math.atan2(float(matrix[2, 0]), -float(matrix[2, 1]))
    phi = _phi_from_upper_left(matrix, theta, psi)
    return wrap_two_pi(phi), float(theta), wrap_two_pi(psi)


def _phi_from_upper_left(matrix: np.ndarray, theta: float, psi: float) -> float:
    """``φ`` from the upper-left block of ``R @ (R1(θ) @ R3(ψ)).T``.

    For a proper rotation that residual is ``R3(φ)``, whose upper-left block
    is ``[[cos φ, sin φ], [−sin φ, cos φ]]``.
    """
    residual = matrix @ (R1(theta) @ R3(psi)).T
    sine = 0.5 * (float(residual[0, 1]) - float(residual[1, 0]))
    cosine = 0.5 * (float(residual[0, 0]) + float(residual[1, 1]))
    return math.atan2(sine, cosine)


def pointing_roll_to_matrix(ra_rad: float, dec_rad: float, roll_rad: float) -> np.ndarray:
    """Full attitude from ``(α, δ, roll)``.

    ``ψ = α + π/2``, ``θ = π/2 − δ``, ``φ = roll``. ``δ`` must lie in
    ``[−π/2, π/2]``. At ``δ = ±π/2`` the same matrix can be written with
    other ``(α, roll)`` pairs; :func:`matrix_to_pointing_roll` returns the
    pair with ``ψ = 0``.
    """
    _require_finite_angle(ra_rad, "ra_rad")
    _require_finite_angle(dec_rad, "dec_rad")
    _require_finite_angle(roll_rad, "roll_rad")
    if dec_rad < -math.pi / 2.0 - 1e-12 or dec_rad > math.pi / 2.0 + 1e-12:
        raise ValueError(f"dec_rad must be in [-π/2, π/2], got {dec_rad}")
    dec = max(-math.pi / 2.0, min(math.pi / 2.0, float(dec_rad)))
    return euler313_to_matrix(roll_rad, math.pi / 2.0 - dec, ra_rad + math.pi / 2.0)


def matrix_to_pointing_roll(R: np.ndarray) -> tuple[float, float, float]:
    """Return ``(α, δ, roll)`` in radians, using the gimbal-lock convention.

    ``α`` and ``roll`` are in ``[0, 2π)``, ``δ`` is in ``[−π/2, π/2]``.
    Near either pole ``α`` is the value that makes ``ψ = 0``, namely
    ``α = 3π/2``, and ``roll`` absorbs the remaining turn.
    """
    phi, theta, psi = matrix_to_euler313(R)
    dec = math.pi / 2.0 - theta
    ra = wrap_two_pi(psi - math.pi / 2.0)
    return ra, dec, phi


def boresight_radec(R: np.ndarray) -> tuple[float, float]:
    """Optical axis ``(α, δ)`` from the third row of ``R``. Roll is not included.

    This is the direction of the camera +Z axis in ICRS. It cannot by itself
    determine the attitude. At exactly ``δ = ±π/2``, ``α`` is reported as 0.
    """
    matrix = _require_rotation(R)
    return direction_to_radec(matrix[2])


def boresight_vector(R: np.ndarray) -> np.ndarray:
    """Unit optical axis in ICRS: the third row of ``R``."""
    matrix = _require_rotation(R)
    return np.array(matrix[2], dtype=np.float64, copy=True)


def north_position_angle(R: np.ndarray) -> float:
    """Angle from celestial north to image up (−Y), positive toward east.

    Radians in ``(−π, π]``, the same half-open interval as :func:`wrap_pi`.
    With ``roll = 0``, image +X points east and image +Y points north, so
    image up points south and the angle is ``+π``, not ``−π``.
    The local north and east are built from the attitude ``(α, δ)``, which
    stays defined at the pole through the gimbal convention. Away from the
    pole this is ``π − roll``.
    """
    matrix = _require_rotation(R)
    ra, dec, _roll = matrix_to_pointing_roll(matrix)
    east, north = _east_north(ra, dec)
    image_up = -matrix[1]
    return wrap_pi(math.atan2(float(np.dot(image_up, east)), float(np.dot(image_up, north))))


def tilt_from_pixels(dx_px: float, dy_px: float, f_over_p: float) -> np.ndarray:
    """Camera-frame tilt that moves a centered star by ``(dx, dy)`` pixels.

    The axis lies in the camera XY plane, so the tilt adds no roll. A zero
    offset returns ``I``. ``+x`` is a column, ``+y`` is a row.
    """
    if f_over_p <= 0.0 or not math.isfinite(f_over_p):
        raise ValueError(f"f_over_p must be positive, got {f_over_p}")
    if not math.isfinite(dx_px) or not math.isfinite(dy_px):
        raise ValueError(f"pixel offset must be finite, got {(dx_px, dy_px)}")
    radius = math.hypot(dx_px, dy_px)
    if radius == 0.0:
        return np.eye(3, dtype=np.float64)
    beta = math.atan2(dy_px, dx_px)
    theta = math.atan(radius / f_over_p)
    axis = np.array([-math.sin(beta), math.cos(beta), 0.0], dtype=np.float64)
    return rotate_about_axis(axis, theta)


def _east_north(ra: float, dec: float) -> tuple[np.ndarray, np.ndarray]:
    """Local east and north at ``(α, δ)``. Both are defined at the poles."""
    east = np.array([-math.sin(ra), math.cos(ra), 0.0], dtype=np.float64)
    north = np.array(
        [
            -math.sin(dec) * math.cos(ra),
            -math.sin(dec) * math.sin(ra),
            math.cos(dec),
        ],
        dtype=np.float64,
    )
    return east, north


def _as_matrix(R: np.ndarray) -> np.ndarray:
    matrix = np.asarray(R, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rotation matrix must be finite")
    return matrix


def _require_rotation(R: np.ndarray) -> np.ndarray:
    matrix = _as_matrix(R)
    if abs(float(np.linalg.det(matrix)) - 1.0) > 1e-6:
        raise ValueError(f"matrix is not a proper rotation, det={np.linalg.det(matrix)}")
    gram = matrix @ matrix.T
    if float(np.max(np.abs(gram - np.eye(3)))) > 1e-6:
        raise ValueError("matrix is not orthonormal")
    return matrix


def _require_finite_angle(angle: float, name: str) -> None:
    if isinstance(angle, bool) or not isinstance(angle, (int, float, np.floating, np.integer)):
        raise TypeError(f"{name} must be a real angle in radians, got {angle!r}")
    if not math.isfinite(float(angle)):
        raise ValueError(f"{name} must be finite, got {angle!r}")


def _unit_vector(vector: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return array / norm
