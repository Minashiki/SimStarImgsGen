"""Blend groups from a valley test on the line between group centroids.

Plan sections 4.7, D40 and D41. Group centroids pick candidate pairs and
the line that joins them. The light on that line is the box-screened local
continuous field of the original members, not a merged point source and not
the pixel-integrated expectation image. A source is included when its stamp
box meets the axis-aligned box of the original endpoints. That box is the
min and max of ``(x1, y1)`` and ``(x2, y2)``, not endpoints rebuilt from
the samples. The stamp occupies ``[col0 − 0.5, col1 + 0.5) × [row0 − 0.5,
row1 + 0.5)``: the left and top edges are included, and the right and
bottom edges are not. Those sources are then summed from the continuous
local PSF at every sample. Samples are not zeroed where they fall outside
a stamp. A box of −1 on all four sides (the stamp missed the image) is
left out. ``screen=False`` skips that box test and sums every valid source
continuously: the full-source continuous field, which is a different
quantity from the screened field.

The profile is the closed interval between the two centroids, about 200
samples, with no extension past either end. A pair merges unless that
profile has an interior local minimum whose depth, divided by the fainter
of the two side maxima, exceeds ``VALLEY_TOL``. Normalizing by the whole
profile's amplitude would hide a faint companion behind the bright peak.
The global minimum of the closed interval sits at an endpoint until the
dip is deep, so the test uses the local minimum.

Candidates are rebuilt from the centroids each round. The light field is
always the original members. Union-find runs until a round adds no merge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from starsim.psf import PsfModel

# Behavior version for blend grouping. Increment when the same star field
# starts producing different groups.
# 1 summed only the two candidates. 2 sums every current source on a
# profile that extends past the pair. 3 tests a valley on the closed
# segment and stamp-truncates the original members. 4 keeps the box
# screen but sums the selected sources continuously, with no per-sample cut.
# 5 screens candidates with the original endpoints.
BLEND_VERSION = 5

PROFILE_SAMPLES = 200
VALLEY_TOL = 1e-5
_MAX_ROUNDS = 64


@dataclass(frozen=True)
class BlendResult:
    """Partition of the input sources.

    ``group_id[i]`` is the group containing source ``i``. ``groups`` lists
    member indices, each sorted. ``blended`` is true for members of a group
    of two or more. Centroids and ``flux_sum`` are constant inside a group.
    A single star keeps its own position and flux.
    """

    group_id: np.ndarray
    groups: tuple[np.ndarray, ...]
    blended: np.ndarray
    centroid_x: np.ndarray
    centroid_y: np.ndarray
    flux_sum: np.ndarray


def has_profile_valley(profile: np.ndarray, tol: float = VALLEY_TOL) -> bool:
    """True when an interior local minimum is deep against the fainter side peak.

    A flat run that is strictly below both neighboring samples is one valley.
    The run has to sit strictly inside the profile. ``L`` and ``R`` are the
    maxima on either side of that run, and the depth is
    ``(min(L, R) − v) / min(L, R)``.
    """
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"profile must be 1D, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("profile must be finite")
    count = int(values.size)
    if count < 3:
        return False
    index = 1
    while index < count - 1:
        if not (values[index] < values[index - 1]):
            index += 1
            continue
        end = index
        while end + 1 < count and values[end + 1] == values[index]:
            end += 1
        if end >= count - 1:
            break
        if values[end + 1] > values[end]:
            floor = float(values[index])
            left = float(values[:index].max())
            right = float(values[end + 1 :].max())
            faint = left if left < right else right
            if faint > 0.0 and (faint - floor) / faint > tol:
                return True
        index = end + 1
    return False


def blend_stars(
    x: np.ndarray,
    y: np.ndarray,
    flux: np.ndarray,
    psf: PsfModel,
    radii: np.ndarray,
    search_sigma: float = 4.0,
    n_profile: int = PROFILE_SAMPLES,
    *,
    box_row0: np.ndarray | None = None,
    box_row1: np.ndarray | None = None,
    box_col0: np.ndarray | None = None,
    box_col1: np.ndarray | None = None,
    screen: bool = True,
) -> BlendResult:
    """Group sources whose connecting profile has no valley.

    ``radii`` is one stamp radius per source. Pass the renderer's clipped
    boxes when the sources were painted into an image; otherwise the boxes
    are the unclipped stamp windows. ``screen`` drops sources whose stamp
    misses the axis-aligned box of the original endpoints. The stamp is the
    half-open pixel rectangle. Sources that remain are summed continuously
    along the whole segment. ``screen=False`` sums every valid source the
    same way, which is the full-source continuous field.
    """
    if not isinstance(psf, PsfModel):
        raise TypeError("psf must be starsim.psf.PsfModel")
    coords_x = np.asarray(x, dtype=np.float64)
    coords_y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(flux, dtype=np.float64)
    stamp_r = np.asarray(radii)
    if coords_x.shape != coords_y.shape or coords_x.shape != weights.shape or coords_x.ndim != 1:
        raise ValueError("x, y, and flux must be 1D arrays of the same length")
    if stamp_r.shape != coords_x.shape:
        raise ValueError("radii must have one entry per source")
    if not np.all(np.isfinite(coords_x)) or not np.all(np.isfinite(coords_y)):
        raise ValueError("positions must be finite")
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("flux must be finite and positive")
    if isinstance(search_sigma, bool) or not isinstance(search_sigma, (int, float, np.floating)):
        raise TypeError(f"search_sigma must be a real number, got {search_sigma!r}")
    if not math.isfinite(float(search_sigma)) or float(search_sigma) <= 0.0:
        raise ValueError(f"search_sigma must be > 0, got {search_sigma!r}")
    if type(n_profile) is not int or n_profile < 16:
        raise ValueError(f"n_profile must be an int >= 16, got {n_profile!r}")
    if type(screen) is not bool:
        raise TypeError("screen must be bool")
    row0, row1, col0, col1 = _resolve_boxes(
        coords_x, coords_y, stamp_r, box_row0, box_row1, box_col0, box_col1
    )
    count = int(coords_x.shape[0])
    if count == 0:
        empty = np.zeros(0, dtype=np.int64)
        return BlendResult(
            empty,
            (),
            np.zeros(0, dtype=bool),
            empty.astype(np.float64),
            empty.astype(np.float64),
            empty.astype(np.float64),
        )

    parent = np.arange(count, dtype=np.int64)
    for _round in range(_MAX_ROUNDS):
        roots, members = _components(parent)
        node_x, node_y, node_sigma = _node_state(coords_x, coords_y, weights, psf, members)
        if node_x.shape[0] == 1:
            break
        merged = _merge_round(
            parent,
            roots,
            node_x,
            node_y,
            node_sigma,
            coords_x,
            coords_y,
            weights,
            row0,
            row1,
            col0,
            col1,
            psf,
            float(search_sigma),
            n_profile,
            screen,
        )
        if not merged:
            break
    else:
        raise RuntimeError(f"blend grouping did not converge in {_MAX_ROUNDS} rounds")
    return _pack(coords_x, coords_y, weights, parent)


def _resolve_boxes(x, y, radii, box_row0, box_row1, box_col0, box_col1):
    provided = (box_row0, box_row1, box_col0, box_col1)
    if all(box is None for box in provided):
        return _unclipped_boxes(x, y, radii)
    if any(box is None for box in provided):
        raise ValueError("stamp boxes must be passed together")
    boxes = tuple(np.asarray(box, dtype=np.int64) for box in provided)
    if any(box.shape != x.shape for box in boxes):
        raise ValueError("stamp boxes must have one entry per source")
    return boxes


def _unclipped_boxes(x, y, radii):
    count = int(x.shape[0])
    row0 = np.empty(count, dtype=np.int64)
    row1 = np.empty(count, dtype=np.int64)
    col0 = np.empty(count, dtype=np.int64)
    col1 = np.empty(count, dtype=np.int64)
    for index in range(count):
        radius = float(radii[index])
        row0[index] = int(math.floor(float(y[index]) - radius)) - 1
        row1[index] = int(math.ceil(float(y[index]) + radius)) + 1
        col0[index] = int(math.floor(float(x[index]) - radius)) - 1
        col1[index] = int(math.ceil(float(x[index]) + radius)) + 1
    return row0, row1, col0, col1


def _components(parent: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    roots = np.empty(parent.shape[0], dtype=np.int64)
    for index in range(parent.shape[0]):
        roots[index] = _find(parent, index)
    unique = np.unique(roots)
    members = [np.flatnonzero(roots == root) for root in unique]
    return unique, members


def _find(parent: np.ndarray, index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = int(parent[index])
    return index


def _union(parent: np.ndarray, left: int, right: int) -> bool:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root == right_root:
        return False
    if left_root < right_root:
        parent[right_root] = left_root
    else:
        parent[left_root] = right_root
    return True


def _node_state(x, y, flux, psf: PsfModel, members):
    node_x = np.empty(len(members), dtype=np.float64)
    node_y = np.empty(len(members), dtype=np.float64)
    for index, group in enumerate(members):
        weight = flux[group]
        total = float(weight.sum())
        node_x[index] = float(np.dot(weight, x[group]) / total)
        node_y[index] = float(np.dot(weight, y[group]) / total)
    node_sigma = np.asarray(psf.sigma_iso(node_x, node_y), dtype=np.float64)
    return node_x, node_y, node_sigma


def _merge_round(
    parent,
    roots,
    node_x,
    node_y,
    node_sigma,
    src_x,
    src_y,
    src_flux,
    row0,
    row1,
    col0,
    col1,
    psf,
    search_sigma,
    n_profile,
    screen,
) -> bool:
    limit = float(node_sigma.max()) * search_sigma
    if not math.isfinite(limit) or limit <= 0.0:
        return False
    tree = cKDTree(np.column_stack((node_x, node_y)))
    pairs = tree.query_pairs(r=limit, output_type="ndarray")
    if len(pairs) == 0:
        return False
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    merged = False
    for left, right in pairs[order]:
        left = int(left)
        right = int(right)
        distance = math.hypot(float(node_x[left] - node_x[right]), float(node_y[left] - node_y[right]))
        reach = search_sigma * max(float(node_sigma[left]), float(node_sigma[right]))
        if distance > reach:
            continue
        if distance == 0.0:
            if _union(parent, int(roots[left]), int(roots[right])):
                merged = True
            continue
        profile = _segment_profile(
            float(node_x[left]),
            float(node_y[left]),
            float(node_x[right]),
            float(node_y[right]),
            src_x,
            src_y,
            src_flux,
            row0,
            row1,
            col0,
            col1,
            psf,
            n_profile,
            screen,
        )
        if not has_profile_valley(profile) and _union(parent, int(roots[left]), int(roots[right])):
            merged = True
    return merged


def _segment_profile(
    x1,
    y1,
    x2,
    y2,
    src_x,
    src_y,
    src_flux,
    row0,
    row1,
    col0,
    col1,
    psf: PsfModel,
    n_profile: int,
    screen: bool,
) -> np.ndarray:
    dx = x2 - x1
    dy = y2 - y1
    distance = math.hypot(dx, dy)
    samples = np.linspace(0.0, distance, n_profile)
    unit_x = dx / distance
    unit_y = dy / distance
    sample_x = x1 + samples * unit_x
    sample_y = y1 + samples * unit_y
    keep = _sources_on_segment(x1, y1, x2, y2, row0, row1, col0, col1, screen)
    if not np.any(keep):
        return np.zeros(n_profile, dtype=np.float64)
    brightness = psf.evaluate_at_sources(sample_x, sample_y, src_x[keep], src_y[keep])
    return (src_flux[keep, None] * brightness).sum(axis=0)


def _sources_on_segment(x1, y1, x2, y2, row0, row1, col0, col1, screen: bool) -> np.ndarray:
    """Sources admitted to the continuous sum, in input order.

    The segment box is the min and max of the original endpoints. Rebuilding
    an endpoint from the samples can move it by 1 ulp, so that rebuilt point
    is not used here. A stamp occupies
    ``[col0 − 0.5, col1 + 0.5) × [row0 − 0.5, row1 + 0.5)``. The left and
    top edges are included; the right and bottom edges are not. The two
    rectangles meet when ``xmax ≥ col0 − 0.5``, ``xmin < col1 + 0.5``,
    ``ymax ≥ row0 − 0.5`` and ``ymin < row1 + 0.5``. ``col0 − 0.5`` and the
    other edges are exact for the integer coordinates used on an image.

    A rendered box of ``−1`` on all four sides missed the image and never
    contributes. Unclipped boxes may be negative and are still real windows,
    so the empty test is the sentinel on all four sides, which
    ``clipped_stamp_boxes`` writes that way. ``screen=False`` keeps every
    valid source.
    """
    # The renderer writes −1 on all four sides when the stamp misses the image.
    # An unclipped window may sit at negative coordinates and is still real.
    valid = ~((row0 == -1) & (row1 == -1) & (col0 == -1) & (col1 == -1))
    if not screen:
        return valid
    xmin, xmax = min(float(x1), float(x2)), max(float(x1), float(x2))
    ymin, ymax = min(float(y1), float(y2)), max(float(y1), float(y2))
    return (
        valid
        & (xmax >= col0 - 0.5)
        & (xmin < col1 + 0.5)
        & (ymax >= row0 - 0.5)
        & (ymin < row1 + 0.5)
    )


def _pack(x, y, flux, parent) -> BlendResult:
    count = x.shape[0]
    roots = np.empty(count, dtype=np.int64)
    for index in range(count):
        roots[index] = _find(parent, index)
    unique, inverse = np.unique(roots, return_inverse=True)
    group_id = inverse.astype(np.int64, copy=False)
    groups = tuple(np.flatnonzero(group_id == index) for index in range(unique.shape[0]))
    blended = np.zeros(count, dtype=bool)
    centroid_x = np.empty(count, dtype=np.float64)
    centroid_y = np.empty(count, dtype=np.float64)
    flux_sum = np.empty(count, dtype=np.float64)
    for group in groups:
        total = float(flux[group].sum())
        center_x = float(np.dot(flux[group], x[group]) / total)
        center_y = float(np.dot(flux[group], y[group]) / total)
        centroid_x[group] = center_x
        centroid_y[group] = center_y
        flux_sum[group] = total
        if group.shape[0] >= 2:
            blended[group] = True
    return BlendResult(group_id, groups, blended, centroid_x, centroid_y, flux_sum)
