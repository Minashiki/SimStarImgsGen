"""Pointing sample and no-roll great-circle camera motion.

``simulate_sequence`` draws one sequence from the ``motion`` stream and writes
the accepted speed and acceleration back onto the effective config. Those
scalars meet the nominal-step constraints below, so a whole-turn alias
(an acceleration near ``4π(f/p)``) is not written back. Optional sky regions
and catalog enable the production split and minimum-star acceptance checks.
Leaving both at None retains the v2 default path exactly (D77).

Draw order on every attempt, from the single ``motion`` generator. Changing
this order increments the motion-algorithm version:

1. ``α ~ U[0, 2π)``, ``u ~ U[−1, 1]``, ``δ = arcsin(u)``, ``roll ~ U[0, 2π)``
2. image-plane direction ``β ~ U[0, 2π)``
3. ``s0`` and ``a`` from the configured intervals
4. per-frame pitch/yaw jitter, only when ``jitter_px > 0``

Every attempt draws that full set before any check. A nominal-step rejection
may skip the matrix construction. It does not skip the jitter draw. Skipping
the draw would shift the random sequence of every later attempt.

Checks run in this order. The first failure is the only one counted:

``nominal_step``, ``projection``, ``speed``, ``horizon``, ``galactic_latitude``.

``nominal_step`` is decided before any matrix is built. For each
``k = 0 … N−2``, ``s_k = s0 + a(k + ½)`` must be finite, lie in ``[lo, hi]``,
and satisfy ``0 < s_k / (f/p) < π/2``. All three have to hold. A NaN compares
false with every bound, so a test written as "reject if some bound fails"
would accept it. The positive bound is enforced here: :func:`_bounds` does
not require ``lo > 0``, and a direct ``simulate_sequence`` call would
otherwise drop the config-level check. The three conditions pin each
inter-frame increment to the unique forward branch that does not cross the
projection horizon, which rejects a whole-turn alias and a mid-sequence
reversal. They do not constrain the instantaneous speed of a continuous-time
interpolation.

The other four checks use the candidate attitudes.

1. Generate, only after the nominal step passes. Build a candidate attitude
   for every frame, extract its quaternion, and rebuild the matrix with
   :func:`quaternion_to_matrix`. Every later quantity uses that rebuilt set:
   inter-frame displacement, inter-frame rotation, the reference matrix, and
   the derived outputs. The product ``tilt @ slew @ base`` is not used again
   (D25). Euler extraction itself is stable on a proper rotation; stored
   outputs are still read from the rebuilt matrix.

2. Judge the remaining checks, in the order above. The first failure is the
   only one counted, and the attempt is redrawn. Acceptance requires every
   check to pass, so the order does not change which attempt is accepted.
   Reordering the checks does not increment the motion-algorithm version
   when three things stay the same: the random-number consumption, the
   accept result, and the status of the rejection counts. Those counts are
   not a protected persisted output.

   - Adjacent frames are projectable (``Z > 0``) on the jittered attitudes.
     The nominal condition does not replace this. A noticeable chance of
     triggering usually needs a nominal step near ``π/2``; an unbounded
     Gaussian draw can still cross from a smaller step.
   - The actual adjacent-frame displacement, including jitter, lies in the
     speed interval.
   - The field of view does not touch or cross the reference camera's
     horizon. Each frame's four image corners have ``Z_ref > 0`` in the
     reference camera. The reference frame is the configured ``ref_index``.
     ``Z_ref`` is affine in pixel coordinates, so four positive corners put
     the whole rectangle in front of the reference camera. ``H[2, 2]`` of
     the unnormalized homography is proportional to ``Z_ref`` of pixel
     ``(0, 0)`` and is therefore positive, and normalizing by it is legal.
     The test does not limit the union-canvas size.
   - Galactic latitude of the middle frame's optical axis, frame ``N // 2``,
     including pointing jitter. This frame is not the registration reference.
     The cut does not claim that every frame, or the whole field of view,
     satisfies the floor.

3. Derive. Only an accepted candidate gets Euler angles, the north angle,
   homographies, and the other stored outputs.

``β`` is the direction in which a star at the image center moves,
``atan2(Δy, Δx)``. The camera-frame axis is ``(−sin β, cos β, 0)``, so the
axis has no optical-axis component and the nominal ``ω_z`` is 0. Jitter is a
pure tilt in the slewed camera frame and does not add roll either.

The slew angle of frame ``k`` is ``θ_k = (s0·k + ½·a·k²) / (f/p)``.
Pointing is uniform on the sphere. An optional ``|b|`` floor uses
``pointing.galactic_lat_min_deg`` on frame ``N // 2`` only.

An equal-endpoint speed interval is rejected before sampling. The interval
must satisfy ``lo < hi``. A zero-width interval constrains both the nominal
step and the jittered displacement, and is not an acceptance interval for
rejection sampling. Config validation raises the same text. The
continuous-model reason (no jitter, and ``tan x > x`` on ``0 < x < π/2``,
so ``f·tan(s/f) > s``) is in plan section 4.2. The error does not claim that
a floating-point draw is impossible.

Before ``rngs.get("motion")``, the float inputs covered here are converted,
checked for finiteness, and, when they are intervals, checked for order,
span, and the value constraints that already exist. The covered inputs are
speed and acceleration, each as a scalar or an interval, plus ``jitter_px``,
``frame_interval_s``, and ``galactic_lat_min_deg``. An acceleration scalar
that is NaN, Inf, or an integer outside the finite float range fails here.
It does not claim the stream, and a NaN does not spend the 10000 attempts.
This does not promise that every invalid input fails before the stream is
issued.

A float speed or acceleration interval whose span ``hi - lo`` is not finite
is part of that precheck. ``_bounds`` is not enough: finite, ordered
endpoints can still overflow the subtraction. The same predicate as config
validation makes the decision. The original interval is left unchanged.
This is an unsupported sampling input. It is not a rejection reason and it
does not spend the 10000 attempts. The uniform formula is not rewritten.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from starsim.attitude import (
    boresight_radec,
    matrix_to_euler313,
    matrix_to_pointing_roll,
    matrix_to_quaternion,
    north_position_angle,
    pointing_roll_to_matrix,
    quaternion_to_matrix,
    rotate_about_axis,
    rotation_vector,
    tilt_from_pixels,
    wrap_two_pi,
)
from starsim.camera import Camera, homography, project, reference_index
from starsim.config import (
    DerivedCamera,
    _FLOAT_OVERFLOW,
    _to_float_or_overflow,
    equal_speed_endpoint_message,
    is_interval,
    sample_uniform,
    uniform_span_is_finite,
)
from starsim.rng import SequenceRng

# 运动算法版本。属于序列生成契约。M4 写入 algorithm_versions() 返回的 motion。
# 该映射只有 config_sampling 和 motion。rng_scheme 只在 SequenceRng.metadata()。
# 版本不匹配时默认拒绝。只记录版本号并不能让新版算法重放出旧结果。
# 必须加一：改变接受条件、抽样顺序或运动模型；改变受重放契约保护的数值输出，
# 包括这些输出依赖的 attitude 和 camera 公式。依据是 4.10 的逐比特复现要求。
# 还没有正式数据集也不能悄悄覆盖已写下的版本：旧实现可能已经被外部实验用过。
# 可以不加一：重排检查顺序，且同时满足三点——不改变随机数消耗、不改变接受结果、
# 拒绝计数不属于受保护的持久化输出。
# 版本号保护合法配置的配置抽样结果，以及合法序列记录中受重放契约保护的输出
# 和相关随机数消耗。仅拒绝既有配置契约已禁止、且不可能参与任何合法记录的输入，
# 不递增版本。包级辅助函数的直接调用入口收紧，须单独记载。
# 跨度预检查、浮点超大整数的异常规范化，以及加速度标量 NaN/Inf 在领流前失败，
# 都属于这一类：成功样本的数值和随机数消耗都不变。NaN 标量原先会领流并跑满
# 10000 次，那是直接调用入口收紧，单独记载，不递增版本。
# derive_camera 的公式一旦改变，会通过 K 和 f_over_p 影响运动输出，本版本要加一。
# 本轮只收紧相机配置支持范围，公式没变，所以不加一。
# 1 → 2：新增名义步进检查；φ 改用稳定的左上块公式。
# 2 → 3：直接接入划分、最低星数检查；None 默认路径逐位不变（D77）。
MOTION_ALGORITHM_VERSION = 3

_MAX_ATTEMPTS = 10000
_RA_NGP = math.radians(192.85948)
_DEC_NGP = math.radians(27.12825)
_TWO_PI = 2.0 * math.pi

# First-failure labels, in check order. One attempt increments exactly one of
# these. The counts are not a complete tally of every condition that failed.
_REJECTION_REASONS = (
    "nominal_step",
    "projection",
    "speed",
    "horizon",
    "galactic_latitude",
)


@dataclass
class SequenceMotion:
    """Attitudes and inter-frame motion for one sequence.

    Per-frame arrays have length ``N``. Inter-frame arrays have length
    ``N−1`` and index ``k`` describes the step from frame ``k`` to ``k+1``.

    ``matrix`` is the rotation rebuilt from ``quaternion``. Euler angles,
    pointing, the optical axis, the north angle, and ``H_to_ref`` are read
    from that matrix, not from the product that produced the quaternion.

    ``ra_rad``, ``dec_rad`` and ``roll_rad`` are the full attitude, including
    the gimbal-lock convention. ``boresight_ra_rad`` and ``boresight_dec_rad``
    are the optical axis alone (the third row of ``R``). ``ra0_rad`` is the
    pointing drawn before slew and jitter. The galactic-latitude cut does not
    use ``ra0_rad``; it uses the optical axis of frame ``N // 2``.

    ``center_shift_px`` is where frame ``k``'s optical axis lands in frame
    ``k+1``, relative to ``(cx, cy)``. ``omega_cam_rad`` is the camera's
    rotation vector over that step, in radians per frame, expressed in frame
    ``k``. Its nominal z component is 0; jitter changes it only at second order.
    """

    quaternion: np.ndarray
    matrix: np.ndarray
    ra_rad: np.ndarray
    dec_rad: np.ndarray
    roll_rad: np.ndarray
    boresight_ra_rad: np.ndarray
    boresight_dec_rad: np.ndarray
    euler_phi_rad: np.ndarray
    euler_theta_rad: np.ndarray
    euler_psi_rad: np.ndarray
    north_angle_rad: np.ndarray
    H_to_ref: np.ndarray
    cumulative_px: np.ndarray
    center_shift_px: np.ndarray
    omega_cam_rad: np.ndarray
    step_angle_rad: np.ndarray
    speed_px_per_frame: np.ndarray
    speed_deg_per_s: np.ndarray
    direction_rad: np.ndarray
    reference_index: int
    axis_cam: np.ndarray
    beta_rad: float
    speed0_px: float
    accel_px_per_frame2: float
    jitter_px: float
    ra0_rad: float
    dec0_rad: float
    roll0_rad: float
    frame_interval_s: float
    attempts: int


def galactic_latitude_rad(ra_rad: float, dec_rad: float) -> float:
    """IAU galactic latitude of an ICRS direction, in radians."""
    sin_b = math.sin(dec_rad) * math.sin(_DEC_NGP) + math.cos(dec_rad) * math.cos(
        _DEC_NGP
    ) * math.cos(ra_rad - _RA_NGP)
    return math.asin(max(-1.0, min(1.0, sin_b)))


def latitude_frame_index(n_frames: int) -> int:
    """Frame whose optical axis is compared with the galactic-latitude floor.

    The index is ``N // 2``. It does not follow ``reference_frame``. The cut
    includes pointing jitter and does not apply to every frame or to the
    whole field of view.
    """
    return n_frames // 2


def simulate_sequence(
    effective: dict,
    rngs: SequenceRng,
    derived: DerivedCamera,
    *, sky_regions=None, split=None, catalog=None,
) -> SequenceMotion:
    """Sample a pointing and a no-roll slew. Mutates ``effective`` in place.

    ``effective`` is the dict returned by ``sample_sequence_params``. On
    success, ``camera_motion.speed_px_per_frame`` and
    ``camera_motion.accel_px_per_frame2`` become the accepted scalars. The
    same ``SequenceRng`` cannot be used twice: ``motion`` is issued once.

    See the module docstring for the three steps and for which changes
    increment :data:`MOTION_ALGORITHM_VERSION`.
    """
    if not isinstance(effective, dict):
        raise TypeError(
            "effective must be the dict returned by sample_sequence_params, "
            f"got {type(effective).__name__}"
        )
    if not isinstance(rngs, SequenceRng):
        raise TypeError("rngs must be starsim.rng.SequenceRng")
    camera = Camera.from_derived(derived)
    n_frames = _require_n_frames(effective)
    motion_cfg = _require_mapping(effective, "camera_motion")
    speed_spec = motion_cfg["speed_px_per_frame"]
    accel_spec = motion_cfg["accel_px_per_frame2"]
    # Conversion, finiteness, order, span, and the existing value checks for
    # these inputs all happen before the motion stream is issued. Other
    # invalid inputs are outside this promise.
    speed_lo, speed_hi = _bounds(speed_spec, "camera_motion.speed_px_per_frame")
    _reject_equal_speed_endpoints(speed_lo, speed_hi)
    _reject_nonfinite_span(speed_lo, speed_hi, "camera_motion.speed_px_per_frame", speed_spec)
    # Calling _bounds is not the span check. Finite, ordered endpoints can
    # still make hi - lo overflow. A scalar acceleration is checked here too:
    # NaN, Inf, and an integer outside the finite float range fail before
    # the stream is issued.
    accel_lo, accel_hi = _bounds(accel_spec, "camera_motion.accel_px_per_frame2")
    if is_interval(accel_spec):
        _reject_nonfinite_span(
            accel_lo,
            accel_hi,
            "camera_motion.accel_px_per_frame2",
            accel_spec,
        )
    sigma_px = _motion_float(motion_cfg["jitter_px"], "camera_motion.jitter_px")
    if not math.isfinite(sigma_px) or sigma_px < 0.0:
        raise ValueError(f"camera_motion.jitter_px must be >= 0, got {sigma_px}")
    frame_interval_s = _motion_float(
        _require_mapping(effective, "timing")["frame_interval_s"],
        "timing.frame_interval_s",
    )
    if not math.isfinite(frame_interval_s) or frame_interval_s <= 0.0:
        raise ValueError(f"timing.frame_interval_s must be positive, got {frame_interval_s}")
    galactic_min = _require_mapping(effective, "pointing").get("galactic_lat_min_deg")
    if galactic_min is not None:
        galactic_min = _motion_float(galactic_min, "pointing.galactic_lat_min_deg")
        if not math.isfinite(galactic_min) or not 0.0 <= galactic_min <= 90.0:
            raise ValueError(f"pointing.galactic_lat_min_deg must be in [0, 90], got {galactic_min}")
    reference_spec = motion_cfg["reference_frame"]
    ref_index = reference_index(n_frames, reference_spec)
    rng = rngs.get("motion")

    counts = {name: 0 for name in _REJECTION_REASONS}
    if sky_regions is not None:
        counts['sky_split'] = 0
    if catalog is not None:
        counts['min_labeled_stars'] = 0
    accepted: SequenceMotion | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        ra0, dec0, roll0 = _draw_pointing(rng)
        beta = float(rng.uniform(0.0, _TWO_PI))
        speed0 = sample_uniform(rng, speed_spec)
        accel = sample_uniform(rng, accel_spec)
        jitter = _draw_jitter(rng, n_frames, sigma_px)
        if not _nominal_steps_accepted(
            speed0,
            accel,
            n_frames,
            camera.f_over_p,
            speed_lo,
            speed_hi,
        ):
            counts["nominal_step"] += 1
            continue
        axis, quaternions, matrices = _candidate_attitudes(
            camera=camera,
            n_frames=n_frames,
            ra0=ra0,
            dec0=dec0,
            roll0=roll0,
            beta=beta,
            speed0=speed0,
            accel=accel,
            jitter=jitter,
        )
        reason = _first_rejection(
            matrices,
            camera,
            ref_index=ref_index,
            speed_lo=speed_lo,
            speed_hi=speed_hi,
            galactic_min_deg=galactic_min,
        )
        if reason is None and sky_regions is not None:
            from starsim.split import accepts
            if not accepts(sky_regions, split, matrices[n_frames//2, 2]):
                reason = 'sky_split'
        if reason is None and catalog is not None:
            if not _minimum_stars(matrices[n_frames//2], camera, catalog, effective):
                reason = 'min_labeled_stars'
        if reason is None:
            accepted = _derive_motion(
                camera=camera,
                quaternions=quaternions,
                matrices=matrices,
                axis=axis,
                beta=beta,
                speed0=speed0,
                accel=accel,
                sigma_px=sigma_px,
                ref_index=ref_index,
                frame_interval_s=frame_interval_s,
                ra0=ra0,
                dec0=dec0,
                roll0=roll0,
                attempt=attempt,
            )
            break
        counts[reason] += 1
    if accepted is None:
        raise RuntimeError(
            _rejection_message(
                counts=counts,
                speed_spec=speed_spec,
                accel_spec=accel_spec,
                n_frames=n_frames,
                sigma_px=sigma_px,
                galactic_min=galactic_min,
                reference_spec=reference_spec,
                ref_index=ref_index,
            )
        )
    motion_cfg["speed_px_per_frame"] = float(accepted.speed0_px)
    motion_cfg["accel_px_per_frame2"] = float(accepted.accel_px_per_frame2)
    return accepted


def _minimum_stars(matrix, camera, catalog, effective):
    """Count magnitude-limited centers in the middle frame, before exposure."""
    from starsim.camera import project_icrs
    ra = float(np.degrees(np.arctan2(matrix[2, 1], matrix[2, 0])) % 360)
    dec = float(np.degrees(np.arcsin(matrix[2, 2])))
    radius = math.degrees(math.atan(math.hypot(camera.width, camera.height)/(2*camera.f_over_p)))
    indices = catalog.cone_query(ra, dec, radius)
    indices = indices[catalog.vmag[indices] <= effective['photometry']['mag_limit']]
    if not len(indices):
        return effective['pointing']['min_labeled_stars'] == 0
    xy = project_icrs(camera, matrix, catalog.xyz[indices])
    inside = ((xy[:, 0] >= -.5) & (xy[:, 0] < camera.width-.5)
              & (xy[:, 1] >= -.5) & (xy[:, 1] < camera.height-.5))
    return int(inside.sum()) >= effective['pointing']['min_labeled_stars']


def _candidate_attitudes(
    *,
    camera: Camera,
    n_frames: int,
    ra0: float,
    dec0: float,
    roll0: float,
    beta: float,
    speed0: float,
    accel: float,
    jitter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 1. Return ``(axis, quaternions, matrices)`` rebuilt from quaternions."""
    axis = np.array([-math.sin(beta), math.cos(beta), 0.0], dtype=np.float64)
    base = pointing_roll_to_matrix(ra0, dec0, roll0)
    quaternions = np.empty((n_frames, 4), dtype=np.float64)
    matrices = np.empty((n_frames, 3, 3), dtype=np.float64)
    for index in range(n_frames):
        theta = (speed0 * index + 0.5 * accel * index * index) / camera.f_over_p
        slew = np.eye(3, dtype=np.float64) if theta == 0.0 else rotate_about_axis(axis, theta)
        tilt = tilt_from_pixels(float(jitter[index, 0]), float(jitter[index, 1]), camera.f_over_p)
        quaternions[index] = matrix_to_quaternion(tilt @ slew @ base)
        matrices[index] = quaternion_to_matrix(quaternions[index])
    return axis, quaternions, matrices


def _nominal_steps_accepted(
    speed0: float,
    accel: float,
    n_frames: int,
    f_over_p: float,
    speed_lo: float,
    speed_hi: float,
) -> bool:
    """True when every inter-frame nominal step is on the forward branch.

    ``s_k = s0 + a(k + 1/2)`` for ``k = 0 … N−2``. Each step must be finite,
    lie in ``[lo, hi]``, and satisfy ``0 < s_k / (f/p) < π/2``. The positive
    bound is written here, not taken from config: :func:`_bounds` does not
    require ``lo > 0``. This constrains the nominal inter-frame increment
    only, not the instantaneous speed of a continuous interpolation.

    The predicate is a conjunction. ``s_k > hi`` is false for NaN, so a
    test written as "reject if a bound fails" would accept NaN. Steps are
    Python floats: a numpy expression overflows with a ``RuntimeWarning``,
    and an overflowing step becomes ``inf`` and fails the finite check.
    """
    # π/2 is a quarter turn. It is also the projection horizon of the forward branch.
    quarter_turn = math.pi / 2.0
    for k in range(n_frames - 1):
        step = float(speed0) + float(accel) * (k + 0.5)
        angle = step / f_over_p
        finite = math.isfinite(step)
        in_interval = speed_lo <= step <= speed_hi
        # Separate from the configured lower bound. ``_bounds`` accepts lo <= 0.
        positive = step > 0.0
        on_forward_branch = math.isfinite(angle) and 0.0 < angle < quarter_turn
        if not (finite and in_interval and positive and on_forward_branch):
            return False
    return True


def _first_rejection(
    matrices: np.ndarray,
    camera: Camera,
    *,
    ref_index: int,
    speed_lo: float,
    speed_hi: float,
    galactic_min_deg: float | None,
) -> str | None:
    """First failing check after ``nominal_step``, or ``None`` if all pass.

    ``nominal_step`` is decided before the matrices exist. The order here is
    ``projection``, ``speed``, ``horizon``, ``galactic_latitude``.
    Production adds sky split and minimum-star checks immediately after
    this result in the same rejection loop. There is no callback list.
    """
    steps = matrices.shape[0] - 1
    center = np.array([camera.cx, camera.cy], dtype=np.float64)
    for index in range(steps):
        vector = matrices[index + 1] @ matrices[index, 2]
        if float(vector[2]) <= 0.0:
            return "projection"
    for index in range(steps):
        vector = matrices[index + 1] @ matrices[index, 2]
        landed = project(camera, vector)
        shift = landed - center
        speed = float(math.hypot(float(shift[0]), float(shift[1])))
        if speed < speed_lo or speed > speed_hi:
            return "speed"
    if not _horizon_clear(camera, matrices, ref_index):
        return "horizon"
    latitude_frame = latitude_frame_index(matrices.shape[0])
    if not _latitude_accepted(matrices[latitude_frame], galactic_min_deg):
        return "galactic_latitude"
    return None


def _derive_motion(
    *,
    camera: Camera,
    quaternions: np.ndarray,
    matrices: np.ndarray,
    axis: np.ndarray,
    beta: float,
    speed0: float,
    accel: float,
    sigma_px: float,
    ref_index: int,
    frame_interval_s: float,
    ra0: float,
    dec0: float,
    roll0: float,
    attempt: int,
) -> SequenceMotion:
    """Step 3. Derived outputs, read only from the rebuilt matrices."""
    n_frames = matrices.shape[0]
    ra = np.empty(n_frames, dtype=np.float64)
    dec = np.empty(n_frames, dtype=np.float64)
    roll = np.empty(n_frames, dtype=np.float64)
    bore_ra = np.empty(n_frames, dtype=np.float64)
    bore_dec = np.empty(n_frames, dtype=np.float64)
    phi = np.empty(n_frames, dtype=np.float64)
    theta_e = np.empty(n_frames, dtype=np.float64)
    psi = np.empty(n_frames, dtype=np.float64)
    north = np.empty(n_frames, dtype=np.float64)
    homographies = np.empty((n_frames, 3, 3), dtype=np.float64)
    reference = matrices[ref_index]
    for index in range(n_frames):
        matrix = matrices[index]
        ra[index], dec[index], roll[index] = matrix_to_pointing_roll(matrix)
        bore_ra[index], bore_dec[index] = boresight_radec(matrix)
        phi[index], theta_e[index], psi[index] = matrix_to_euler313(matrix)
        north[index] = north_position_angle(matrix)
        homographies[index] = homography(camera, matrix, reference)

    steps = n_frames - 1
    shift = np.empty((steps, 2), dtype=np.float64)
    omega = np.empty((steps, 3), dtype=np.float64)
    step_angle = np.empty(steps, dtype=np.float64)
    center = np.array([camera.cx, camera.cy], dtype=np.float64)
    for index in range(steps):
        landed = project(camera, matrices[index + 1] @ matrices[index, 2])
        shift[index] = landed - center
        relative = matrices[index + 1] @ matrices[index].T
        omega[index] = -rotation_vector(relative)
        step_angle[index] = _angle_between(matrices[index, 2], matrices[index + 1, 2])
    speed = np.linalg.norm(shift, axis=1)
    direction = np.array(
        [wrap_two_pi(float(angle)) for angle in np.arctan2(shift[:, 1], shift[:, 0])],
        dtype=np.float64,
    )
    cumulative = np.zeros(n_frames, dtype=np.float64)
    if steps:
        cumulative[1:] = np.cumsum(speed)
    return SequenceMotion(
        quaternion=quaternions,
        matrix=matrices,
        ra_rad=ra,
        dec_rad=dec,
        roll_rad=roll,
        boresight_ra_rad=bore_ra,
        boresight_dec_rad=bore_dec,
        euler_phi_rad=phi,
        euler_theta_rad=theta_e,
        euler_psi_rad=psi,
        north_angle_rad=north,
        H_to_ref=homographies,
        cumulative_px=cumulative,
        center_shift_px=shift,
        omega_cam_rad=omega,
        step_angle_rad=step_angle,
        speed_px_per_frame=speed,
        speed_deg_per_s=np.degrees(step_angle) / frame_interval_s,
        direction_rad=direction,
        reference_index=ref_index,
        axis_cam=axis,
        beta_rad=float(beta),
        speed0_px=float(speed0),
        accel_px_per_frame2=float(accel),
        jitter_px=float(sigma_px),
        ra0_rad=float(ra0),
        dec0_rad=float(dec0),
        roll0_rad=float(roll0),
        frame_interval_s=float(frame_interval_s),
        attempts=attempt,
    )


def _corner_vectors(camera: Camera) -> np.ndarray:
    """Unnormalized camera vectors of the four image-rectangle corners."""
    xs = (-0.5, float(camera.width) - 0.5)
    ys = (-0.5, float(camera.height) - 0.5)
    rows = [
        ((x - camera.cx) / camera.f_over_p, (y - camera.cy) / camera.f_over_p, 1.0)
        for y in ys
        for x in xs
    ]
    return np.asarray(rows, dtype=np.float64)


def _horizon_clear(camera: Camera, matrices: np.ndarray, ref_index: int) -> bool:
    """True when every frame's image rectangle is strictly in front of ``ref_index``."""
    corners = _corner_vectors(camera)
    reference = matrices[ref_index]
    for index in range(matrices.shape[0]):
        relative = reference @ matrices[index].T
        z_ref = corners @ relative[2]
        if bool(np.any(z_ref <= 0.0)):
            return False
    return True


def _latitude_accepted(matrix: np.ndarray, minimum_deg: float | None) -> bool:
    if minimum_deg is None:
        return True
    ra, dec = boresight_radec(matrix)
    latitude_deg = abs(math.degrees(galactic_latitude_rad(ra, dec)))
    return latitude_deg >= minimum_deg


def _rejection_message(
    *,
    counts: dict[str, int],
    speed_spec: object,
    accel_spec: object,
    n_frames: int,
    sigma_px: float,
    galactic_min: float | None,
    reference_spec: object,
    ref_index: int,
) -> str:
    lines = [
        f"相机运动拒绝采样失败：总次数 {_MAX_ATTEMPTS}。",
        "每次尝试只按首次失败原因计入一类，各类之和等于拒绝次数；"
        "这不是所有失败条件的完整统计。",
    ]
    lines.extend(f"{name}={counts[name]}" for name in counts)
    lines.extend(
        [
            f"speed_px_per_frame={speed_spec!r}",
            f"accel_px_per_frame2={accel_spec!r}",
            f"n_frames={n_frames}",
            f"jitter_px={sigma_px!r}",
            f"galactic_lat_min_deg={galactic_min!r}",
            f"reference_frame={reference_spec!r}",
            f"reference_index={ref_index}",
            f"latitude_frame={latitude_frame_index(n_frames)}",
        ]
    )
    return "\n".join(lines)


def _draw_pointing(rng: np.random.Generator) -> tuple[float, float, float]:
    ra = float(rng.uniform(0.0, _TWO_PI))
    dec = math.asin(float(rng.uniform(-1.0, 1.0)))
    roll = float(rng.uniform(0.0, _TWO_PI))
    return ra, dec, roll


def _draw_jitter(rng: np.random.Generator, n_frames: int, sigma_px: float) -> np.ndarray:
    if sigma_px == 0.0:
        return np.zeros((n_frames, 2), dtype=np.float64)
    return rng.normal(0.0, sigma_px, size=(n_frames, 2))


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    cross = float(np.linalg.norm(np.cross(a, b)))
    dot = float(np.dot(a, b))
    return math.atan2(cross, dot)


def _require_n_frames(effective: dict) -> int:
    value = _require_mapping(effective, "dataset")["n_frames"]
    if type(value) is not int:
        raise TypeError(
            "dataset.n_frames must already be an int (sample_sequence_params draws it), "
            f"got {value!r}"
        )
    if value < 2:
        raise ValueError(f"dataset.n_frames must be >= 2, got {value}")
    return value


def _require_mapping(effective: dict, key: str) -> dict:
    value = effective.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"effective[{key!r}] must be a mapping")
    return value


def _motion_float(value: object, path: str) -> float:
    """One motion input as a float. Overflow becomes ``ValueError``.

    A numeric string or a boolean still converts, as ``float`` did before
    this precheck. That direct-call behavior is known and deferred: the
    configuration entry already rejects those types. Tightening it later
    is a public-entry behavior change and does not increment a version.
    """
    number = _to_float_or_overflow(value)
    if number is _FLOAT_OVERFLOW:
        raise ValueError(f"{path} must be finite, got {value!r}")
    return number


def _bounds(spec: object, path: str) -> tuple[float, float]:
    if is_interval(spec):
        lo = _to_float_or_overflow(spec[0])
        hi = _to_float_or_overflow(spec[1])
    elif isinstance(spec, (bool, np.bool_)) or not isinstance(spec, (int, float, np.integer, np.floating)):
        raise TypeError(f"{path} must be a number or [lo, hi], got {spec!r}")
    else:
        lo = hi = _to_float_or_overflow(spec)
    if (
        lo is _FLOAT_OVERFLOW
        or hi is _FLOAT_OVERFLOW
        or not math.isfinite(lo)
        or not math.isfinite(hi)
        or hi < lo
    ):
        raise ValueError(f"{path} bounds must be finite and ordered, got {spec!r}")
    return lo, hi


def _reject_nonfinite_span(lo: float, hi: float, path: str, spec: object) -> None:
    """Reject an unsupported float interval before any stream is issued.

    The predicate is :func:`uniform_span_is_finite`. This function does not
    sample and does not replace ``spec``.
    """
    if uniform_span_is_finite(lo, hi):
        return
    raise ValueError(f"{path} span must be finite, got {spec!r}")


def _reject_equal_speed_endpoints(lo: float, hi: float) -> None:
    if lo != hi:
        return
    raise ValueError(equal_speed_endpoint_message(lo, hi))
