"""Shared population checks for the regression and the production statistics."""

from __future__ import annotations

import math

import numpy as np

from starsim.render import KIND_POINT, KIND_STREAK
from starsim.targets import max_line_distance

_STEP_TOL = 1e-6


def reference_visible_segment(x1: float, y1: float, x2: float, y2: float, width: int, height: int):
    """Visible piece of a segment, from the overlap of the two axis intervals.

    This is not ``clip_segment_to_image``. Each axis contributes the parameter
    values that stay inside the closed box ``[-0.5, W-0.5] × [-0.5, H-0.5]``;
    the overlap of those intervals is the visible piece. ``None`` means the
    segment misses the image.
    """
    dx = x2 - x1
    dy = y2 - y1
    t_enter = 0.0
    t_exit = 1.0
    for origin, delta, low, high in (
        (x1, dx, -0.5, width - 0.5),
        (y1, dy, -0.5, height - 0.5),
    ):
        if delta == 0.0:
            if origin < low or origin > high:
                return None
            continue
        t_low = (low - origin) / delta
        t_high = (high - origin) / delta
        if delta < 0.0:
            t_low, t_high = t_high, t_low
        if t_low > t_enter:
            t_enter = t_low
        if t_high < t_exit:
            t_exit = t_high
        if t_enter > t_exit:
            return None
    return (
        x1 + t_enter * dx,
        y1 + t_enter * dy,
        x1 + t_exit * dx,
        y1 + t_exit * dy,
    )


def population_constraint_problems(
    pop,
    *,
    width: int,
    height: int,
    n_frames: int,
    frame_dt: float,
    photometry,
    point_speed: tuple[float, float] = (3.0, 50.0),
    streak_speed: tuple[float, float] = (20.0, 100.0),
    min_visible: int = 3,
    min_length: float = 5.0,
    rand_rel_cap: float = 0.15,
    clip_sigma: float = 2.5,
    max_line: float = 3.0,
) -> dict:
    """Geometry, speed, and flux constraints. Measured SNR is not included."""
    problems: list[str] = []
    distances: list[float] = []
    step_violations = 0
    strict_residual = 0.0
    for i in range(pop.track_id.shape[0]):
        distance = max_line_distance(pop.x_ref[i], pop.y_ref[i])
        distances.append(distance)
        limit = 1e-6 if bool(pop.strict[i]) else max_line
        if bool(pop.strict[i]):
            strict_residual = max(strict_residual, distance)
        if distance > limit:
            problems.append(f"track {i} line distance {distance} > {limit}")
        steps = np.hypot(np.diff(pop.x_ref[i]), np.diff(pop.y_ref[i]))
        if int(pop.kind[i]) == KIND_STREAK:
            lo, hi = streak_speed
        else:
            lo, hi = point_speed
        if steps.size and (float(steps.min()) < lo - _STEP_TOL or float(steps.max()) > hi + _STEP_TOL):
            problems.append(f"track {i} step outside [{lo}, {hi}]")
            step_violations += 1
        cap = rand_rel_cap * float(pop.speed[i])
        if float(pop.sigma_cross[i]) > cap + 1e-8 or float(pop.sigma_along[i]) > cap + 1e-8:
            problems.append(f"track {i} random sigma above {rand_rel_cap} v")
        if np.any(np.abs(pop.delta_cross[i]) > clip_sigma * float(pop.sigma_cross[i]) + 1e-8):
            problems.append(f"track {i} cross deviation outside the clip")
        if np.any(np.abs(pop.delta_along[i]) > clip_sigma * float(pop.sigma_along[i]) + 1e-8):
            problems.append(f"track {i} along deviation outside the clip")
        if bool(pop.strict[i]):
            if float(pop.curve_amp[i]) != 0.0 or float(pop.sigma_cross[i]) != 0.0:
                problems.append(f"strict track {i} has a deviation amplitude")
            if np.any(pop.delta_cross[i] != 0.0) or np.any(pop.delta_along[i] != 0.0):
                problems.append(f"strict track {i} has a nonzero delta")
        run = np.flatnonzero(pop.in_image[i])
        if run.size < min_visible or int(run[-1] - run[0] + 1) != int(run.size):
            problems.append(f"track {i} visible run is not one block of length >= {min_visible}")
        present = np.flatnonzero(pop.in_image[i])
        entering = bool(present.size and int(present[0]) > 0)
        exiting = bool(present.size and int(present[-1]) < n_frames - 1)
        if bool(pop.entering[i]) != entering or bool(pop.exiting[i]) != exiting:
            problems.append(f"track {i} enter/exit flags disagree with the visible run")
        if int(pop.kind[i]) == KIND_STREAK:
            length = pop.L_true[i]
            if float(length.min()) < min_length - 1e-9:
                problems.append(f"streak {i} shorter than {min_length} px")
            for frame in range(n_frames):
                neighbors = []
                if frame > 0:
                    neighbors.append(float(steps[frame - 1]))
                if frame < n_frames - 1:
                    neighbors.append(float(steps[frame]))
                if any(float(length[frame]) >= neighbor for neighbor in neighbors):
                    problems.append(f"streak {i} frame {frame} is not shorter than the step")
                    break
        for frame in range(n_frames):
            x1 = float(pop.x1[i, frame])
            y1 = float(pop.y1[i, frame])
            x2 = float(pop.x2[i, frame])
            y2 = float(pop.y2[i, frame])
            contained = _inside(x1, y1, width, height) and _inside(x2, y2, width, height)
            length = math.hypot(x2 - x1, y2 - y1)
            endpoint_tol = 1e-6 * length
            clipped = reference_visible_segment(x1, y1, x2, y2, width, height)
            expect = int(pop.kind[i]) == KIND_STREAK and not contained and clipped is not None
            if bool(pop.truncated[i, frame]) != expect:
                problems.append(f"track {i} frame {frame} truncation flag")
                break
            if contained:
                ends = (pop.vis_x1[i, frame], pop.vis_y1[i, frame], pop.vis_x2[i, frame], pop.vis_y2[i, frame])
                if not np.allclose(ends, (x1, y1, x2, y2)):
                    problems.append(f"track {i} frame {frame} kept a clipped copy of an interior segment")
                    break
            elif clipped is None:
                if float(pop.x_vis[i, frame]) != -1.0 or float(pop.L_vis[i, frame]) != 0.0:
                    problems.append(f"track {i} frame {frame} invented a visible segment")
                    break
            else:
                if not np.allclose(
                    (pop.vis_x1[i, frame], pop.vis_y1[i, frame], pop.vis_x2[i, frame], pop.vis_y2[i, frame]),
                    clipped,
                    rtol=0.0,
                    atol=endpoint_tol,
                ):
                    problems.append(f"track {i} frame {frame} visible segment disagrees with the edge intersection")
                    break
            native = float(pop.L_native[i, frame])
            if native > 0.0:
                expect_frac = float(pop.L_vis[i, frame]) / native
            elif contained or clipped is not None:
                expect_frac = 1.0
            else:
                expect_frac = 0.0
            if not np.isclose(float(pop.vis_frac[i, frame]), expect_frac):
                problems.append(f"track {i} frame {frame} vis_frac is not L_vis / L_native")
                break
            if clipped is not None or contained:
                vx1, vy1 = float(pop.vis_x1[i, frame]), float(pop.vis_y1[i, frame])
                vx2, vy2 = float(pop.vis_x2[i, frame]), float(pop.vis_y2[i, frame])
                if not np.isclose(pop.x_vis[i, frame], 0.5 * (vx1 + vx2)) or not np.isclose(
                    pop.y_vis[i, frame], 0.5 * (vy1 + vy2)
                ):
                    problems.append(f"track {i} frame {frame} visible center is not the clip midpoint")
                    break
                if not np.isclose(pop.L_vis[i, frame], math.hypot(vx2 - vx1, vy2 - vy1)):
                    problems.append(f"track {i} frame {frame} L_vis disagrees with the endpoints")
                    break
        expect_never = int(pop.kind[i]) == KIND_STREAK and not any(
            _inside(float(pop.x1[i, frame]), float(pop.y1[i, frame]), width, height)
            and _inside(float(pop.x2[i, frame]), float(pop.y2[i, frame]), width, height)
            for frame in range(n_frames)
        )
        if bool(pop.never_full[i]) != expect_never:
            problems.append(f"track {i} never_full")
        if float(pop.accel_along[i]) != 0.0:
            problems.append(f"track {i} has along-track acceleration")
    for frame in range(n_frames):
        for kind in (KIND_POINT, KIND_STREAK):
            selected = np.flatnonzero((pop.kind == kind) & pop.in_image[:, frame])
            for left in range(selected.size):
                for right in range(left + 1, selected.size):
                    a = int(selected[left])
                    b = int(selected[right])
                    gap = math.hypot(pop.x_c[a, frame] - pop.x_c[b, frame], pop.y_c[a, frame] - pop.y_c[b, frame])
                    if gap < 2.0:
                        problems.append(f"same-class centers {gap} px apart in frame {frame}")
    if not np.array_equal(pop.flux_e, pop.flux_design[:, None] * pop.jitter):
        problems.append("electrons are not design flux times jitter")
    if not np.allclose(pop.t_exp_eff, pop.duty * frame_dt):
        problems.append("t_exp_eff is not duty times the frame interval")
    if not np.array_equal(pop.mag_eq, np.asarray(photometry.magnitude(pop.flux_e))):
        problems.append("mag_eq disagrees with the zero point")
    return {
        "problems": problems,
        "distances": distances,
        "step_violations": step_violations,
        "strict_residual": strict_residual,
    }


def _inside(x: float, y: float, width: int, height: int) -> bool:
    return bool(x >= -0.5 and x < width - 0.5 and y >= -0.5 and y < height - 0.5)
