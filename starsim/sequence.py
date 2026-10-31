"""Frame loop for one sequence: stars, moving targets, exposure, blend.

Order, and the reason :data:`SEQUENCE_VERSION` exists:

1. Project the static stars of every frame into source arrays. This uses no
   random stream and does not keep the electron images.
2. Sample the target population. That issues ``targets/population`` and then
   ``targets/brightness_jitter``. Design noise reads stars with
   ``star_value_at``.
3. For frame ``k = 0, 1, …`` in that order: paint that frame's stars, render
   that frame's targets, issue ``frame/{k}/noise`` (decimal, no zero
   padding), expose, then blend the static stars only.

Changing how many cosmic rays one frame draws does not move the other
frames' noise streams. The cosmic-ray rate is an argument of ``expose_frame``
and does not modify the shared sensor.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from starsim.blend import BlendResult, blend_stars
from starsim.camera import Camera
from starsim.catalog import StarCatalog
from starsim.motion import SequenceMotion
from starsim.noise import Exposure, SensorFingerprint, expose_frame
from starsim.photometry import Photometry
from starsim.psf import PsfModel
from starsim.render import (
    RenderedSources,
    RenderedTargets,
    StaticStarFrame,
    paint_static_stars,
    project_static_stars,
    render_moving_targets,
)
from starsim.rng import SequenceRng
from starsim.targets import TargetPopulation, sample_targets

# Behavior version for the frame loop and the per-frame noise stream names.
# 1 is the order in the module docstring.
SEQUENCE_VERSION = 1


def frame_noise_name(frame_index: int) -> str:
    """``frame/{k}/noise``. ``k`` is decimal and is not zero-padded."""
    if type(frame_index) is not int or isinstance(frame_index, bool) or frame_index < 0:
        raise ValueError(f"frame_index must be an int >= 0, got {frame_index!r}")
    return f"frame/{frame_index}/noise"


@dataclass
class FrameResult:
    """One frame after exposure. ``blend`` partitions the static stars only."""

    frame_index: int
    stars: RenderedSources
    targets: RenderedTargets
    exposure: Exposure
    blend: BlendResult


@dataclass
class SequenceResult:
    """Population plus one :class:`FrameResult` per frame, in frame order.

    Holding this object keeps every frame's image. Use it for small images
    in tests. Production and acceptance iterate :func:`iter_frames` instead.
    """

    population: TargetPopulation
    frames: tuple[FrameResult, ...]


@dataclass
class SequencePlan:
    """Source arrays and the population. No electron images."""

    population: TargetPopulation
    stars: tuple[StaticStarFrame, ...]
    noise_sigma: float
    search_sigma: float


def prepare_sequence(
    catalog: StarCatalog,
    camera: Camera,
    motion: SequenceMotion,
    effective: dict,
    rngs: SequenceRng,
    sensor: SensorFingerprint,
    psf: PsfModel,
    photometry: Photometry,
) -> SequencePlan:
    """Project every frame's stars, then sample the population."""
    if not isinstance(rngs, SequenceRng):
        raise TypeError("rngs must be starsim.rng.SequenceRng")
    n_frames = int(motion.matrix.shape[0])
    if sensor.n_frames != n_frames:
        raise ValueError(f"sensor.n_frames is {sensor.n_frames}, motion has {n_frames}")
    noise_sigma = sensor.reference_sigma()
    stars = tuple(
        project_static_stars(
            catalog,
            camera,
            motion.matrix[frame],
            photometry,
            psf,
            noise_sigma,
        )
        for frame in range(n_frames)
    )
    population = sample_targets(
        effective,
        rngs,
        camera,
        motion,
        psf,
        photometry,
        sensor,
        stars,
    )
    return SequencePlan(
        population=population,
        stars=stars,
        noise_sigma=noise_sigma,
        search_sigma=float(effective["labels"]["blend_search_sigma"]),
    )


def iter_frames(
    plan: SequencePlan,
    rngs: SequenceRng,
    sensor: SensorFingerprint,
    psf: PsfModel,
    *,
    stars: Sequence[StaticStarFrame] | None = None,
    population: TargetPopulation | None = None,
    cosmic_rates: dict[int, float] | None = None,
) -> Iterator[FrameResult]:
    """Yield one exposed frame at a time.

    ``stars`` and ``population`` replace the plan's arrays for this pass.
    Tests use that seam. Each frame is painted inside the loop.
    """
    if not isinstance(rngs, SequenceRng):
        raise TypeError("rngs must be starsim.rng.SequenceRng")
    star_frames = plan.stars if stars is None else tuple(stars)
    pop = plan.population if population is None else population
    n_frames = len(star_frames)
    if sensor.n_frames != n_frames:
        raise ValueError(f"sensor.n_frames is {sensor.n_frames}, stars cover {n_frames} frames")
    if int(pop.x_c.shape[1]) != n_frames:
        raise ValueError(f"population has {pop.x_c.shape[1]} frames, stars cover {n_frames}")
    for frame in range(n_frames):
        painted = paint_static_stars(star_frames[frame])
        rendered = render_moving_targets(
            pop.track_id,
            pop.kind,
            pop.x_c[:, frame],
            pop.y_c[:, frame],
            pop.x1[:, frame],
            pop.y1[:, frame],
            pop.x2[:, frame],
            pop.y2[:, frame],
            pop.flux_e[:, frame],
            psf,
            plan.noise_sigma,
        )
        rate = None
        if cosmic_rates is not None and frame in cosmic_rates:
            rate = float(cosmic_rates[frame])
        exposure = expose_frame(
            painted,
            sensor,
            frame,
            rngs.get(frame_noise_name(frame)),
            psf,
            rendered,
            cosmic_rate=rate,
        )
        groups = blend_stars(
            painted.x,
            painted.y,
            painted.flux_e,
            psf,
            painted.radius,
            search_sigma=plan.search_sigma,
            box_row0=painted.box_row0,
            box_row1=painted.box_row1,
            box_col0=painted.box_col0,
            box_col1=painted.box_col1,
        )
        result = FrameResult(
            frame_index=frame,
            stars=painted,
            targets=rendered,
            exposure=exposure,
            blend=groups,
        )
        del painted, rendered, exposure, groups
        yield result
        del result


def render_sequence(
    catalog: StarCatalog,
    camera: Camera,
    motion: SequenceMotion,
    effective: dict,
    rngs: SequenceRng,
    sensor: SensorFingerprint,
    psf: PsfModel,
    photometry: Photometry,
    *,
    cosmic_rate_override: dict[int, float] | None = None,
) -> SequenceResult:
    """Collect every frame. This holds the whole sequence in memory.

    Only for small images in tests. Production code and acceptance must
    call :func:`prepare_sequence` and :func:`iter_frames` and drop each
    frame after using it.

    ``cosmic_rate_override`` is passed to ``expose_frame`` as ``cosmic_rate``
    for the listed frames. The sensor object is not modified.
    """
    plan = prepare_sequence(
        catalog,
        camera,
        motion,
        effective,
        rngs,
        sensor,
        psf,
        photometry,
    )
    frames = tuple(
        iter_frames(
            plan,
            rngs,
            sensor,
            psf,
            cosmic_rates=cosmic_rate_override,
        )
    )
    return SequenceResult(population=plan.population, frames=frames)
