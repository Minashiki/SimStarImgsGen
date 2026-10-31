"""Moving-target population for one sequence.

Trajectories live in the reference frame and are mapped into each frame
with ``H_ref→k``. A target is kept only when its true centers form one
contiguous run of at least ``min_visible_frames``, adjacent reference-frame
steps stay inside that class's speed interval, the orthogonal distance to
the best-fit line stays inside ``max_track_dev_px``, and centers of the
same class stay at least 2 px apart in any frame where both are inside.

The smooth cross-track term is ``A_c · g(u)`` with
``u = (t − t_mid) / (N − 1)`` and ``t_mid = (N − 1) / 2``. ``g`` is either
the quadratic ``1 − 4u²`` or the half-period cosine ``cos(πu)``. Both are
1 at mid-track and 0 at the first and last frame. Along-track acceleration
is 0: the generator has no target-acceleration field.

Electrons in a frame are the design flux times that frame's brightness
jitter. ``t_exp_eff = duty · Δt`` changes the segment length only.

Random draws, including rejections, go through ``targets/population`` in
track-id order. The jitter matrix is one ``lognormal`` draw on
``targets/brightness_jitter``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from starsim.camera import Camera, apply_homography, homography
from starsim.config import exceeds_cap, is_interval, sample_log_uniform, sample_uniform
from starsim.motion import SequenceMotion
from starsim.noise import SensorFingerprint
from starsim.photometry import Photometry
from starsim.psf import PsfModel
from starsim.render import (
    KIND_POINT,
    KIND_STREAK,
    centers_in_image,
    render_segment_stamp,
    star_value_at,
)
from starsim.rng import SequenceRng, assert_distinct

# Behavior version for the target population. Increment when the same seed
# starts producing different tracks, duties, or design fluxes.
# 1 places tracks in the reference frame, designs flux from the mean local
# noise and the untruncated unit-stamp peak, and keeps electrons independent
# of t_exp_eff. Along-track acceleration stays 0.
# 2 rejects a random-deviation floor above 0.15·v. Boundary comparison unified
# with config validation (rel_tol=1e-9, abs_tol=0): a floor that exceeds the
# cap only inside that tolerance returns the floor and draws no random number.
# A true empty intersection still raises ValueError. The version stays 2
# because only inputs that used to crash change.
TARGETS_VERSION = 2

STREAM_POPULATION = "targets/population"
STREAM_BRIGHTNESS = "targets/brightness_jitter"
assert_distinct([STREAM_POPULATION, STREAM_BRIGHTNESS])

SHAPE_QUADRATIC = 0
SHAPE_SINE = 1
MIN_SAME_CLASS_PX = 2.0
_MAX_ATTEMPTS = 200000
# Float dust on an exact step of length v. A real miss of the speed
# interval is many orders larger.
_STEP_ABS_TOL = 1e-8


@dataclass
class TargetPopulation:
    """Accepted targets in track-id order, plus per-frame geometry.

    Per-frame arrays have shape ``(n_targets, n_frames)``. ``x_ref`` and
    ``y_ref`` are the reference-frame centers. ``x_c`` and ``y_c`` are
    ``H_ref→k`` of those centers. That mapped center is not, in general, the
    midpoint of the mapped endpoints. ``L_true`` keeps its name and is the
    reference-frame length. ``angle_rad`` keeps its name and is the native
    direction, the same values as ``angle_native``. ``x_vis``, ``y_vis``,
    ``L_vis``, ``vis_frac`` and the visible endpoints are clipped in the
    native frame, because the sensor boundary is native.
    ``flux_e`` is ``flux_design · jitter`` and does not depend on duty.
    """

    track_id: np.ndarray
    kind: np.ndarray
    strict: np.ndarray
    p0_x: np.ndarray
    p0_y: np.ndarray
    speed: np.ndarray
    direction_x: np.ndarray
    direction_y: np.ndarray
    accel_along: np.ndarray
    curve_amp: np.ndarray
    curve_shape: np.ndarray
    sigma_cross: np.ndarray
    sigma_along: np.ndarray
    delta_cross: np.ndarray
    delta_along: np.ndarray
    duty: np.ndarray
    t_exp_eff: np.ndarray
    angle_sigma_rad: np.ndarray
    angle_jitter_rad: np.ndarray
    tier_design: np.ndarray
    snr_design: np.ndarray
    flux_design: np.ndarray
    jitter: np.ndarray
    flux_e: np.ndarray
    mag_eq: np.ndarray
    x_ref: np.ndarray
    y_ref: np.ndarray
    x_c: np.ndarray
    y_c: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    x1_ref: np.ndarray
    y1_ref: np.ndarray
    x2_ref: np.ndarray
    y2_ref: np.ndarray
    L_true: np.ndarray
    L_native: np.ndarray
    angle_rad: np.ndarray
    angle_native: np.ndarray
    angle_ref: np.ndarray
    in_image: np.ndarray
    x_vis: np.ndarray
    y_vis: np.ndarray
    vis_x1: np.ndarray
    vis_y1: np.ndarray
    vis_x2: np.ndarray
    vis_y2: np.ndarray
    L_vis: np.ndarray
    vis_frac: np.ndarray
    truncated: np.ndarray
    entering: np.ndarray
    exiting: np.ndarray
    never_full: np.ndarray
    attempts: int


def classify_measured_snr(snr: float, tiers: dict) -> str:
    """Tier name for a measured SNR, or ``gap`` when it sits in none.

    Intervals are half-open on the right. The tier with the highest upper
    endpoint keeps that endpoint, so a configured ``[4, 10]`` next to
    ``[10, 300]`` gives 10 to the upper tier, and 3 falls between ultra and
    faint.
    """
    if not math.isfinite(snr):
        raise ValueError(f"snr must be finite, got {snr}")
    parsed: list[tuple[float, float, str]] = []
    for name, tier in tiers.items():
        bounds = tier["snr"]
        parsed.append((float(bounds[0]), float(bounds[1]), str(name)))
    if not parsed:
        raise ValueError("tiers must contain at least one entry")
    parsed.sort(key=lambda item: (item[0], item[1], item[2]))
    top = max(item[1] for item in parsed)
    for lo, hi, name in parsed:
        if lo <= snr < hi or (snr == hi and hi == top):
            return name
    return "gap"


def max_line_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Max orthogonal distance from the samples to their best-fit line."""
    points = np.column_stack((np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)))
    if points.shape[0] < 2:
        return 0.0
    centered = points - points.mean(axis=0)
    _singular, _values, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    return float(np.max(np.abs(centered @ normal)))


def clip_segment_to_image(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    """Clip a segment to the closed coverage rectangle, or None if it misses.

    The rectangle is ``[−0.5, W−0.5] × [−0.5, H−0.5]``. A segment that only
    touches the exclusive edge of the half-open image still counts as
    touching this closed rectangle.
    """
    return _clip_rect(x1, y1, x2, y2, -0.5, -0.5, width - 0.5, height - 0.5)


def sample_targets(
    effective: dict,
    rngs: SequenceRng,
    camera: Camera,
    motion: SequenceMotion,
    psf: PsfModel,
    photometry: Photometry,
    sensor: SensorFingerprint,
    static_stars: Sequence,
) -> TargetPopulation:
    """Draw targets until the mean per-frame visible count reaches ``per_frame``.

    Design noise reads sky plus :func:`star_value_at` on each frame's static
    stars. Moving targets are not part of that expectation. The two target
    streams are issued here, population first.
    """
    if not isinstance(rngs, SequenceRng):
        raise TypeError("rngs must be starsim.rng.SequenceRng")
    if not isinstance(camera, Camera):
        raise TypeError("camera must be starsim.camera.Camera")
    if not isinstance(motion, SequenceMotion):
        raise TypeError("motion must be starsim.motion.SequenceMotion")
    if not isinstance(psf, PsfModel):
        raise TypeError("psf must be starsim.psf.PsfModel")
    if not isinstance(photometry, Photometry):
        raise TypeError("photometry must be starsim.photometry.Photometry")
    if not isinstance(sensor, SensorFingerprint):
        raise TypeError("sensor must be starsim.noise.SensorFingerprint")
    n_frames = int(motion.matrix.shape[0])
    if len(static_stars) != n_frames:
        raise ValueError(
            f"static_stars has {len(static_stars)} frames, motion has {n_frames}"
        )
    if sensor.n_frames != n_frames:
        raise ValueError(
            f"sensor.n_frames is {sensor.n_frames}, motion has {n_frames}"
        )
    cfg = effective["targets"]
    if cfg.get("flux_design_noise") != "trajectory_mean":
        raise ValueError("targets.flux_design_noise must be 'trajectory_mean'")
    per_frame = int(cfg["per_frame"])
    if per_frame < 1:
        raise ValueError(f"targets.per_frame must be >= 1, got {per_frame}")
    frame_dt = float(effective["timing"]["frame_interval_s"])
    jitter_sigma = float(cfg["brightness_jitter_sigma"])
    if jitter_sigma < 0.0 or not math.isfinite(jitter_sigma):
        raise ValueError(f"brightness_jitter_sigma must be >= 0, got {jitter_sigma}")
    point_lo, point_hi = _bounds(cfg["point"]["speed_px_per_frame"])
    streak_lo, streak_hi = _bounds(cfg["streak"]["speed_px_per_frame"])
    margin = max(point_hi, streak_hi) * max(n_frames - 1, 1)
    xmin, xmax, ymin, ymax = _footprint_box(camera, motion)
    ref = motion.matrix[int(motion.reference_index)]
    H_from_ref = [homography(camera, ref, motion.matrix[k]) for k in range(n_frames)]
    times, parameter, span = _time_basis(n_frames)
    rng = rngs.get(STREAM_POPULATION)
    kept: list[_Track] = []
    visible = np.zeros(n_frames, dtype=np.int64)
    reasons = {
        "visible": 0,
        "speed": 0,
        "line": 0,
        "length": 0,
        "separation": 0,
        "flux": 0,
    }
    attempts = 0
    while float(visible.mean()) < per_frame:
        attempts += 1
        if attempts > _MAX_ATTEMPTS:
            raise RuntimeError(
                "target placement stopped after "
                f"{_MAX_ATTEMPTS} attempts with mean visible count "
                f"{float(visible.mean()):.2f} < {per_frame}; rejections: {reasons}"
            )
        track = _draw_track(
            rng,
            cfg,
            times,
            parameter,
            span,
            xmin - margin,
            xmax + margin,
            ymin - margin,
            ymax + margin,
            point_lo,
            point_hi,
            streak_lo,
            streak_hi,
            frame_dt,
            H_from_ref,
            camera,
            kept,
        )
        if isinstance(track, str):
            reasons[track] += 1
            continue
        flux = _design_flux(
            track,
            cfg,
            rng,
            psf,
            photometry,
            sensor,
            static_stars,
            camera,
        )
        if flux is None:
            reasons["flux"] += 1
            continue
        track.snr_design = flux[0]
        track.tier_design = flux[1]
        track.flux_design = flux[2]
        track.track_id = len(kept)
        kept.append(track)
        visible += track.in_image.astype(np.int64)
    jitter_rng = rngs.get(STREAM_BRIGHTNESS)
    count = len(kept)
    jitter = np.asarray(
        jitter_rng.lognormal(0.0, jitter_sigma, size=(count, n_frames)),
        dtype=np.float64,
    )
    return _pack(kept, jitter, photometry, attempts)


def _draw_track(
    rng,
    cfg,
    times,
    parameter,
    span,
    xmin,
    xmax,
    ymin,
    ymax,
    point_lo,
    point_hi,
    streak_lo,
    streak_hi,
    frame_dt,
    H_from_ref,
    camera: Camera,
    kept: list,
):
    """Draw one candidate. Return a rejection reason, or a track with no flux yet."""
    dev = cfg["track_deviation"]
    streak = float(rng.random()) < float(cfg["streak_fraction"])
    strict = float(rng.random()) < float(dev["strict_fraction"])
    if streak:
        speed = sample_uniform(rng, cfg["streak"]["speed_px_per_frame"])
        speed_lo, speed_hi = streak_lo, streak_hi
    else:
        speed = sample_uniform(rng, cfg["point"]["speed_px_per_frame"])
        speed_lo, speed_hi = point_lo, point_hi
    heading = float(rng.uniform(0.0, 2.0 * math.pi))
    ux, uy = math.cos(heading), math.sin(heading)
    nx, ny = -uy, ux
    n_frames = int(times.shape[0])
    if strict:
        amp = 0.0
        shape = SHAPE_QUADRATIC
        sigma_cross = 0.0
        sigma_along = 0.0
        delta_cross = np.zeros(n_frames, dtype=np.float64)
        delta_along = np.zeros(n_frames, dtype=np.float64)
    else:
        amp = sample_uniform(rng, dev["curve_amp_px"])
        shape = SHAPE_SINE if float(rng.random()) < 0.5 else SHAPE_QUADRATIC
        sigma_cross = _capped_uniform(rng, dev["rand_cross_px"], float(dev["rand_rel_cap"]) * speed)
        sigma_along = _capped_uniform(rng, dev["rand_along_px"], float(dev["rand_rel_cap"]) * speed)
        clip = float(dev["clip_sigma"])
        delta_cross = np.array(
            [_trunc_normal(rng, sigma_cross, clip) for _ in range(n_frames)],
            dtype=np.float64,
        )
        delta_along = np.array(
            [_trunc_normal(rng, sigma_along, clip) for _ in range(n_frames)],
            dtype=np.float64,
        )
    if streak:
        duty = sample_uniform(rng, cfg["streak"]["duty"])
        angle_sigma = math.radians(sample_uniform(rng, cfg["streak"]["angle_jitter_deg"]))
        angle_jitter = np.array(
            [float(rng.normal(0.0, angle_sigma)) for _ in range(n_frames)],
            dtype=np.float64,
        )
        smear = 0.0
    else:
        smear = sample_uniform(rng, cfg["point"]["smear_px"])
        duty = smear / speed
        angle_sigma = 0.0
        angle_jitter = np.zeros(n_frames, dtype=np.float64)
    p0_x = float(rng.uniform(xmin, xmax))
    p0_y = float(rng.uniform(ymin, ymax))
    along, cross, speed_k, tangent_x, tangent_y = _smooth_track(
        times, parameter, span, speed, amp, shape, ux, uy, nx, ny
    )
    x_ref = p0_x + along * ux + cross * nx + delta_along * ux + delta_cross * nx
    y_ref = p0_y + along * uy + cross * ny + delta_along * uy + delta_cross * ny
    steps = np.hypot(np.diff(x_ref), np.diff(y_ref))
    if np.any(steps < speed_lo - _STEP_ABS_TOL) or np.any(steps > speed_hi + _STEP_ABS_TOL):
        return "speed"
    if max_line_distance(x_ref, y_ref) > float(dev["max_track_dev_px"]):
        return "line"
    length = np.empty(n_frames, dtype=np.float64)
    if streak:
        length[:] = duty * speed_k
        min_length = float(cfg["streak"]["min_length_px"])
        if np.any(length < min_length):
            return "length"
        for index in range(n_frames):
            neighbors = []
            if index > 0:
                neighbors.append(float(steps[index - 1]))
            if index < n_frames - 1:
                neighbors.append(float(steps[index]))
            if any(float(length[index]) >= neighbor for neighbor in neighbors):
                return "length"
    else:
        length[:] = smear
    native = _project_track(
        x_ref,
        y_ref,
        length,
        tangent_x,
        tangent_y,
        angle_jitter,
        H_from_ref,
        camera,
        streak,
    )
    run = np.flatnonzero(native["in_image"])
    minimum = int(cfg["min_visible_frames"])
    if run.size < minimum or int(run[-1] - run[0] + 1) != int(run.size):
        return "visible"
    if not _separated(native["x_c"], native["y_c"], native["in_image"], int(streak), kept):
        return "separation"
    track = _Track()
    track.kind = KIND_STREAK if streak else KIND_POINT
    track.strict = strict
    track.p0_x = p0_x
    track.p0_y = p0_y
    track.speed = speed
    track.direction_x = ux
    track.direction_y = uy
    track.accel_along = 0.0
    track.curve_amp = amp
    track.curve_shape = shape
    track.sigma_cross = sigma_cross
    track.sigma_along = sigma_along
    track.delta_cross = delta_cross
    track.delta_along = delta_along
    track.duty = duty
    track.t_exp_eff = duty * frame_dt
    track.angle_sigma_rad = angle_sigma
    track.angle_jitter_rad = angle_jitter
    track.x_ref = x_ref
    track.y_ref = y_ref
    track.L_true = length
    for name, value in native.items():
        setattr(track, name, value)
    return track


def _design_flux(track, cfg, rng, psf, photometry, sensor, static_stars, camera):
    """Sample the tier, then invert the trajectory-mean noise and peak fraction."""
    tier_name, snr = _draw_tier(rng, cfg["tiers"])
    weights = []
    sigmas = []
    for frame in range(track.in_image.shape[0]):
        if not bool(track.in_image[frame]):
            continue
        stamp, row0, col0, _row1, _col1, peak_row, peak_col, peak = render_segment_stamp(
            camera.height,
            camera.width,
            float(track.x1[frame]),
            float(track.y1[frame]),
            float(track.x2[frame]),
            float(track.y2[frame]),
            1.0,
            psf,
        )
        if peak_row < 0 or peak <= 0.0 or stamp.size == 0:
            continue
        scale = float(sensor.vignetting[peak_row, peak_col] * sensor.prnu[peak_row, peak_col])
        expectation = (
            float(sensor.background_e[peak_row, peak_col])
            + star_value_at(static_stars[frame], peak_row, peak_col)
        ) * scale
        variance = expectation + float(sensor.dark_e[peak_row, peak_col]) + sensor.read_e * sensor.read_e
        if variance < 0.0:
            variance = 0.0
        sigmas.append(math.sqrt(variance))
        weights.append(peak * scale)
    if not weights or float(np.mean(weights)) <= 0.0:
        return None
    flux = snr * float(np.mean(sigmas)) / float(np.mean(weights))
    if not math.isfinite(flux) or flux <= 0.0:
        return None
    return snr, tier_name, flux


def _pack(kept: list, jitter: np.ndarray, photometry: Photometry, attempts: int) -> TargetPopulation:
    count = len(kept)
    n_frames = jitter.shape[1]
    flux_design = np.array([track.flux_design for track in kept], dtype=np.float64)
    flux_e = flux_design[:, None] * jitter
    mag_eq = np.asarray(photometry.magnitude(flux_e), dtype=np.float64)

    def column(name, dtype=np.float64):
        return np.stack([np.asarray(getattr(track, name), dtype=dtype) for track in kept])

    def scalar(name, dtype=np.float64):
        return np.array([getattr(track, name) for track in kept], dtype=dtype)

    return TargetPopulation(
        track_id=np.arange(count, dtype=np.int64),
        kind=scalar("kind", np.int8),
        strict=scalar("strict", bool),
        p0_x=scalar("p0_x"),
        p0_y=scalar("p0_y"),
        speed=scalar("speed"),
        direction_x=scalar("direction_x"),
        direction_y=scalar("direction_y"),
        accel_along=scalar("accel_along"),
        curve_amp=scalar("curve_amp"),
        curve_shape=scalar("curve_shape", np.int8),
        sigma_cross=scalar("sigma_cross"),
        sigma_along=scalar("sigma_along"),
        delta_cross=column("delta_cross"),
        delta_along=column("delta_along"),
        duty=scalar("duty"),
        t_exp_eff=scalar("t_exp_eff"),
        angle_sigma_rad=scalar("angle_sigma_rad"),
        angle_jitter_rad=column("angle_jitter_rad"),
        tier_design=np.array([track.tier_design for track in kept]),
        snr_design=scalar("snr_design"),
        flux_design=flux_design,
        jitter=jitter,
        flux_e=flux_e,
        mag_eq=mag_eq,
        x_ref=column("x_ref"),
        y_ref=column("y_ref"),
        x_c=column("x_c"),
        y_c=column("y_c"),
        x1=column("x1"),
        y1=column("y1"),
        x2=column("x2"),
        y2=column("y2"),
        x1_ref=column("x1_ref"),
        y1_ref=column("y1_ref"),
        x2_ref=column("x2_ref"),
        y2_ref=column("y2_ref"),
        L_true=column("L_true"),
        L_native=column("L_native"),
        angle_rad=column("angle_rad"),
        angle_native=column("angle_native"),
        angle_ref=column("angle_ref"),
        in_image=column("in_image", bool),
        x_vis=column("x_vis"),
        y_vis=column("y_vis"),
        vis_x1=column("vis_x1"),
        vis_y1=column("vis_y1"),
        vis_x2=column("vis_x2"),
        vis_y2=column("vis_y2"),
        L_vis=column("L_vis"),
        vis_frac=column("vis_frac"),
        truncated=column("truncated", bool),
        entering=scalar("entering", bool),
        exiting=scalar("exiting", bool),
        never_full=scalar("never_full", bool),
        attempts=attempts,
    )


class _Track:
    """One accepted target before the arrays are stacked."""


def _smooth_track(times, parameter, span, speed, amp, shape, ux, uy, nx, ny):
    along = speed * times
    cross = np.empty(times.shape[0], dtype=np.float64)
    cross_rate = np.empty(times.shape[0], dtype=np.float64)
    for index, u in enumerate(parameter):
        value, derivative = _curve(float(u), shape)
        cross[index] = amp * value
        cross_rate[index] = amp * derivative / span
    speed_k = np.hypot(speed, cross_rate)
    tangent_x = np.empty(times.shape[0], dtype=np.float64)
    tangent_y = np.empty(times.shape[0], dtype=np.float64)
    for index in range(times.shape[0]):
        vx = speed * ux + cross_rate[index] * nx
        vy = speed * uy + cross_rate[index] * ny
        norm = float(speed_k[index])
        if norm == 0.0:
            tangent_x[index] = ux
            tangent_y[index] = uy
        else:
            tangent_x[index] = vx / norm
            tangent_y[index] = vy / norm
    return along, cross, speed_k, tangent_x, tangent_y


def _curve(u: float, shape: int) -> tuple[float, float]:
    """Normalized bow ``g(u)`` and ``dg/du``. Peak 1 at ``u = 0``."""
    if shape == SHAPE_SINE:
        return math.cos(math.pi * u), -math.pi * math.sin(math.pi * u)
    return 1.0 - 4.0 * u * u, -8.0 * u


def _project_track(x_ref, y_ref, length, tangent_x, tangent_y, angle_jitter, H_from_ref, camera, streak):
    n_frames = x_ref.shape[0]
    x_c = np.empty(n_frames, dtype=np.float64)
    y_c = np.empty(n_frames, dtype=np.float64)
    x1 = np.empty(n_frames, dtype=np.float64)
    y1 = np.empty(n_frames, dtype=np.float64)
    x2 = np.empty(n_frames, dtype=np.float64)
    y2 = np.empty(n_frames, dtype=np.float64)
    x1_ref = np.empty(n_frames, dtype=np.float64)
    y1_ref = np.empty(n_frames, dtype=np.float64)
    x2_ref = np.empty(n_frames, dtype=np.float64)
    y2_ref = np.empty(n_frames, dtype=np.float64)
    angle = np.empty(n_frames, dtype=np.float64)
    angle_ref = np.empty(n_frames, dtype=np.float64)
    l_native = np.empty(n_frames, dtype=np.float64)
    x_vis = np.full(n_frames, -1.0, dtype=np.float64)
    y_vis = np.full(n_frames, -1.0, dtype=np.float64)
    vis_x1 = np.full(n_frames, -1.0, dtype=np.float64)
    vis_y1 = np.full(n_frames, -1.0, dtype=np.float64)
    vis_x2 = np.full(n_frames, -1.0, dtype=np.float64)
    vis_y2 = np.full(n_frames, -1.0, dtype=np.float64)
    l_vis = np.zeros(n_frames, dtype=np.float64)
    vis_frac = np.zeros(n_frames, dtype=np.float64)
    truncated = np.zeros(n_frames, dtype=bool)
    for frame in range(n_frames):
        center = apply_homography(H_from_ref[frame], float(x_ref[frame]), float(y_ref[frame]))
        x_c[frame] = float(center[0])
        y_c[frame] = float(center[1])
        direction = _rotate(float(tangent_x[frame]), float(tangent_y[frame]), float(angle_jitter[frame]))
        half = 0.5 * float(length[frame])
        # Endpoints in the reference frame, then the same homography. The
        # mapped center is the label center, not the midpoint of the mapped ends.
        ref_x1 = float(x_ref[frame] - half * direction[0])
        ref_y1 = float(y_ref[frame] - half * direction[1])
        ref_x2 = float(x_ref[frame] + half * direction[0])
        ref_y2 = float(y_ref[frame] + half * direction[1])
        x1_ref[frame] = ref_x1
        y1_ref[frame] = ref_y1
        x2_ref[frame] = ref_x2
        y2_ref[frame] = ref_y2
        angle_ref[frame] = math.atan2(ref_y2 - ref_y1, ref_x2 - ref_x1)
        end_a = apply_homography(H_from_ref[frame], ref_x1, ref_y1)
        end_b = apply_homography(H_from_ref[frame], ref_x2, ref_y2)
        x1[frame] = float(end_a[0])
        y1[frame] = float(end_a[1])
        x2[frame] = float(end_b[0])
        y2[frame] = float(end_b[1])
        angle[frame] = math.atan2(y2[frame] - y1[frame], x2[frame] - x1[frame])
        native_length = float(math.hypot(x2[frame] - x1[frame], y2[frame] - y1[frame]))
        l_native[frame] = native_length
        end_in = (
            _inside(x1[frame], y1[frame], camera.width, camera.height)
            and _inside(x2[frame], y2[frame], camera.width, camera.height)
        )
        clipped = clip_segment_to_image(
            x1[frame], y1[frame], x2[frame], y2[frame], camera.width, camera.height
        )
        if end_in:
            vx1, vy1, vx2, vy2 = x1[frame], y1[frame], x2[frame], y2[frame]
        elif clipped is None:
            vx1 = vy1 = vx2 = vy2 = None
        else:
            vx1, vy1, vx2, vy2 = clipped
        if vx1 is not None:
            vis_x1[frame], vis_y1[frame], vis_x2[frame], vis_y2[frame] = vx1, vy1, vx2, vy2
            x_vis[frame] = 0.5 * (vx1 + vx2)
            y_vis[frame] = 0.5 * (vy1 + vy2)
            l_vis[frame] = float(math.hypot(vx2 - vx1, vy2 - vy1))
            if native_length > 0.0:
                vis_frac[frame] = l_vis[frame] / native_length
            else:
                vis_frac[frame] = 1.0
        if streak and not end_in and clipped is not None:
            truncated[frame] = True
    in_image = centers_in_image(x_c, y_c, camera.width, camera.height)
    present = np.flatnonzero(in_image)
    entering = bool(present.size and int(present[0]) > 0)
    exiting = bool(present.size and int(present[-1]) < n_frames - 1)
    fully_inside = np.zeros(n_frames, dtype=bool)
    for frame in range(n_frames):
        fully_inside[frame] = _inside(x1[frame], y1[frame], camera.width, camera.height) and _inside(
            x2[frame], y2[frame], camera.width, camera.height
        )
    never_full = bool(streak and not np.any(fully_inside))
    return {
        "x_c": x_c,
        "y_c": y_c,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "x1_ref": x1_ref,
        "y1_ref": y1_ref,
        "x2_ref": x2_ref,
        "y2_ref": y2_ref,
        "angle_rad": angle,
        "angle_native": angle,
        "angle_ref": angle_ref,
        "L_native": l_native,
        "in_image": in_image,
        "x_vis": x_vis,
        "y_vis": y_vis,
        "vis_x1": vis_x1,
        "vis_y1": vis_y1,
        "vis_x2": vis_x2,
        "vis_y2": vis_y2,
        "L_vis": l_vis,
        "vis_frac": vis_frac,
        "truncated": truncated,
        "entering": entering,
        "exiting": exiting,
        "never_full": never_full,
    }


def _separated(x_c, y_c, in_image, streak: int, kept: list) -> bool:
    kind = KIND_STREAK if streak else KIND_POINT
    for other in kept:
        if int(other.kind) != kind:
            continue
        both = in_image & other.in_image
        if not np.any(both):
            continue
        distance = np.hypot(x_c[both] - other.x_c[both], y_c[both] - other.y_c[both])
        if np.any(distance < MIN_SAME_CLASS_PX):
            return False
    return True


def _draw_tier(rng, tiers: dict) -> tuple[str, float]:
    names = sorted(tiers)
    pick = float(rng.random())
    cumulative = 0.0
    chosen = names[-1]
    for name in names:
        cumulative += float(tiers[name]["frac"])
        if pick < cumulative:
            chosen = name
            break
    return chosen, sample_log_uniform(rng, tiers[chosen]["snr"])


def _capped_uniform(rng, spec, cap: float) -> float:
    """Draw U[lo, min(hi, cap)], or the floor when it meets the cap on dust.

    A floor above the cap only inside the shared tolerance is the degenerate
    interval ``[lo, lo]``: return ``lo`` and do not draw. A floor above the
    cap by more than that tolerance is a true empty intersection and raises
    before any random number is consumed. The lower bound is not used as a
    stand-in draw for that empty case.
    """
    lo, hi = _bounds(spec)
    if exceeds_cap(lo, cap, rel_tol=1e-9, abs_tol=0.0):
        raise ValueError(
            "random-deviation interval does not intersect the cap: "
            f"lower bound {lo}, upper bound {hi}, cap {cap}"
        )
    if lo > cap:
        return lo
    upper = min(hi, cap)
    return sample_uniform(rng, [lo, upper])


def _trunc_normal(rng, sigma: float, clip: float) -> float:
    if sigma == 0.0:
        return 0.0
    limit = clip * sigma
    while True:
        value = float(rng.normal(0.0, sigma))
        if -limit <= value <= limit:
            return value


def _time_basis(n_frames: int):
    if n_frames < 2:
        raise ValueError(f"n_frames must be >= 2, got {n_frames}")
    times = np.arange(n_frames, dtype=np.float64)
    span = float(n_frames - 1)
    t_mid = 0.5 * span
    return times, (times - t_mid) / span, span


def _footprint_box(camera: Camera, motion: SequenceMotion):
    corners_x = np.array([-0.5, camera.width - 0.5, camera.width - 0.5, -0.5], dtype=np.float64)
    corners_y = np.array([-0.5, -0.5, camera.height - 0.5, camera.height - 0.5], dtype=np.float64)
    xs = []
    ys = []
    for frame in range(motion.matrix.shape[0]):
        mapped = apply_homography(motion.H_to_ref[frame], corners_x, corners_y)
        xs.append(mapped[:, 0])
        ys.append(mapped[:, 1])
    return float(np.min(xs)), float(np.max(xs)), float(np.min(ys)), float(np.max(ys))


def _bounds(spec) -> tuple[float, float]:
    if is_interval(spec):
        return float(spec[0]), float(spec[1])
    value = float(spec)
    return value, value


def _rotate(dx: float, dy: float, angle: float) -> tuple[float, float]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return cosine * dx - sine * dy, sine * dx + cosine * dy


def _inside(x: float, y: float, width: int, height: int) -> bool:
    return bool(x >= -0.5 and x < width - 0.5 and y >= -0.5 and y < height - 0.5)


def _clip_rect(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    dx = x2 - x1
    dy = y2 - y1
    t0 = 0.0
    t1 = 1.0
    for p, q in ((-dx, x1 - xmin), (dx, xmax - x1), (-dy, y1 - ymin), (dy, ymax - y1)):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        ratio = q / p
        if p < 0.0:
            if ratio > t1:
                return None
            if ratio > t0:
                t0 = ratio
        else:
            if ratio < t0:
                return None
            if ratio < t1:
                t1 = ratio
    if t0 > t1:
        return None
    return x1 + t0 * dx, y1 + t0 * dy, x1 + t1 * dx, y1 + t1 * dy
