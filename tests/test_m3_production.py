"""Production statistics for 200 sequences through the real frame loop.

The simplified population regression stays in ``test_m3_targets.py``. This
test is the acceptance: real motion, the catalog, the sensor, and
``iter_frames``.
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import pytest

from starsim import algorithm_versions
from starsim.camera import Camera, apply_homography, homography
from starsim.catalog import StarCatalog, default_cache_path, sha256_file
from starsim.config import load_config, sample_sequence_params
from starsim.motion import simulate_sequence
from starsim.noise import build_sensor
from starsim.photometry import anchor_photometry
from starsim.psf import PsfModel
from starsim.render import peak_energy_fraction
from starsim.rng import SequenceRng, root_seed, sequence_seed
from starsim.sequence import iter_frames, prepare_sequence
from starsim.targets import classify_measured_snr
from tests.m3_checks import population_constraint_problems
from tests.runtime_env import runtime_environment

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts" / "m3_production.json"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
N_SEQUENCES = 200
ROOT_SEED = 20261029
GIB = 1024**3


def production_record(index: int) -> dict:
    """One full sequence. The returned record holds counts, not images."""
    process = psutil.Process(os.getpid())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = load_config()
    catalog = StarCatalog.from_cache(default_cache_path())
    rngs = SequenceRng(sequence_seed(root_seed(ROOT_SEED), index))
    effective = sample_sequence_params(loaded, rngs)
    camera = Camera.from_derived(loaded.derived)
    motion = simulate_sequence(effective, rngs, loaded.derived)
    sensor = build_sensor(effective, camera, rngs)
    psf = PsfModel.from_effective(effective["psf"], camera)
    photo = anchor_photometry(
        float(effective["photometry"]["mag_limit"]),
        float(effective["photometry"]["snr_at_limit"]),
        float(effective["photometry"]["render_mag_margin"]),
        sensor.reference_sigma(),
        peak_energy_fraction(psf),
    )
    plan = prepare_sequence(catalog, camera, motion, effective, rngs, sensor, psf, photo)
    pop = plan.population
    n_frames = int(pop.x_c.shape[1])
    targets = effective["targets"]
    checked = population_constraint_problems(
        pop,
        width=camera.width,
        height=camera.height,
        n_frames=n_frames,
        frame_dt=float(effective["timing"]["frame_interval_s"]),
        photometry=photo,
        point_speed=tuple(targets["point"]["speed_px_per_frame"]),
        streak_speed=tuple(targets["streak"]["speed_px_per_frame"]),
        min_visible=int(targets["min_visible_frames"]),
        min_length=float(targets["streak"]["min_length_px"]),
        rand_rel_cap=float(targets["track_deviation"]["rand_rel_cap"]),
        clip_sigma=float(targets["track_deviation"]["clip_sigma"]),
        max_line=float(targets["track_deviation"]["max_track_dev_px"]),
    )
    problems = list(checked["problems"])
    reference = motion.matrix[int(motion.reference_index)]
    for frame in range(n_frames):
        matrix = homography(camera, reference, motion.matrix[frame])
        for target in range(int(pop.track_id.shape[0])):
            mapped = apply_homography(matrix, float(pop.x_ref[target, frame]), float(pop.y_ref[target, frame]))
            if float(mapped[0]) != float(pop.x_c[target, frame]) or float(mapped[1]) != float(pop.y_c[target, frame]):
                if abs(float(mapped[0]) - float(pop.x_c[target, frame])) > 1e-9 or abs(
                    float(mapped[1]) - float(pop.y_c[target, frame])
                ) > 1e-9:
                    problems.append(f"track {target} frame {frame} center is not H(x_ref, y_ref)")
                    break
    midpoint_gap = 0.0
    if pop.track_id.shape[0]:
        gap = np.hypot(pop.x_c - 0.5 * (pop.x1 + pop.x2), pop.y_c - 0.5 * (pop.y1 + pop.y2))
        midpoint_gap = float(np.max(gap))
    in_bin = 0
    gap_count = 0
    measured = 0
    tiers = targets["tiers"]
    for frame in iter_frames(plan, rngs, sensor, psf):
        snr = frame.exposure.snr
        offset = int(frame.exposure.n_static)
        for target in range(int(pop.track_id.shape[0])):
            if not bool(pop.in_image[target, frame.frame_index]):
                continue
            value = float(snr[offset + target])
            measured += 1
            label = classify_measured_snr(value, tiers)
            if label == "gap":
                gap_count += 1
            elif label == str(pop.tier_design[target]):
                in_bin += 1
        del frame, snr
    mean_count = float(pop.in_image.sum()) / n_frames
    return {
        "index": index,
        "pid": int(os.getpid()),
        "peak_wset": int(process.memory_info().peak_wset),
        "per_frame": int(targets["per_frame"]),
        "mean_count": mean_count,
        "n": int(pop.track_id.shape[0]),
        "distances": checked["distances"],
        "problems": problems[:12],
        "problem_count": len(problems),
        "tiers": [str(name) for name in pop.tier_design],
        "in_bin": in_bin,
        "gap": gap_count,
        "measured": measured,
        "n_entering": int(np.count_nonzero(pop.entering)),
        "n_exiting": int(np.count_nonzero(pop.exiting)),
        "n_truncated_frames": int(np.count_nonzero(pop.truncated)),
        "n_never_full": int(np.count_nonzero(pop.never_full)),
        "step_violations": int(checked["step_violations"]),
        "strict_residual": float(checked["strict_residual"]),
        "midpoint_gap": midpoint_gap,
    }


def _worker_count(configured: int) -> int:
    available = psutil.virtual_memory().available
    room = math.floor((available - 3 * GIB) / (3 * GIB))
    return max(1, min(int(configured), int(room)))


@pytest.mark.slow
def test_m3_production_statistics():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = load_config()
    configured = int(loaded.generator["dataset"]["workers"])
    workers = _worker_count(configured)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(production_record, range(N_SEQUENCES), chunksize=1))
    elapsed = time.perf_counter() - started
    records.sort(key=lambda item: item["index"])
    peaks: dict[int, int] = {}
    for item in records:
        peaks[item["pid"]] = max(peaks.get(item["pid"], 0), item["peak_wset"])
    problems = [item for item in records if item["problem_count"]]
    distances = np.concatenate([np.asarray(item["distances"], dtype=np.float64) for item in records])
    counts, edges = np.histogram(distances, bins=np.linspace(0.0, 3.0, 13))
    tiers = [name for item in records for name in item["tiers"]]
    total_targets = len(tiers)
    fractions = {name: tiers.count(name) / total_targets for name in ("bright", "faint", "ultra")}
    measured = sum(item["measured"] for item in records)
    in_bin = sum(item["in_bin"] for item in records)
    gap = sum(item["gap"] for item in records)
    means = np.array([item["mean_count"] - item["per_frame"] for item in records])
    catalog_path = default_cache_path()
    catalog = StarCatalog.from_cache(catalog_path)
    report = {
        "root_seed": ROOT_SEED,
        "sequence_indices": list(range(N_SEQUENCES)),
        "camera_yaml": (ROOT / "configs" / "camera_default.yaml").read_text(encoding="utf-8"),
        "generator_yaml": (ROOT / "configs" / "generator_default.yaml").read_text(encoding="utf-8"),
        "catalog": {
            "cache_path": str(catalog_path),
            "cache_sha256": sha256_file(catalog_path),
            "meta": _jsonable(catalog.meta),
        },
        "sequences": [
            {"index": item["index"], "N": item["per_frame"], "mean_visible": item["mean_count"]}
            for item in records
        ],
        "algorithm_versions": algorithm_versions(),
        "environment": runtime_environment(),
        "seed_note": "覆盖 YAML 中的 dataset.seed",
        "workers": workers,
        "configured_workers": configured,
        "worker_peak_wset": {str(pid): peak for pid, peak in sorted(peaks.items())},
        "seconds": elapsed,
        "n_sequences": N_SEQUENCES,
        "n_targets": total_targets,
        "n_target_frames": measured,
        "mean_minus_n_min": float(means.min()),
        "mean_minus_n_max": float(means.max()),
        "design_fractions": fractions,
        "measured_in_bin_fraction": in_bin / measured,
        "gap_fraction": gap / measured,
        "entering": int(sum(item["n_entering"] for item in records)),
        "exiting": int(sum(item["n_exiting"] for item in records)),
        "truncated_frames": int(sum(item["n_truncated_frames"] for item in records)),
        "never_full": int(sum(item["n_never_full"] for item in records)),
        "line_distance_histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
        "line_distance_max": float(distances.max()),
        "strict_residual_max": float(max(item["strict_residual"] for item in records)),
        "step_violations": int(sum(item["step_violations"] for item in records)),
        "center_midpoint_max_px": float(max(item["midpoint_gap"] for item in records)),
        "problems": [item["problems"] for item in problems],
        "problem_sequences": len(problems),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert report["problem_sequences"] == 0
    for peak in peaks.values():
        assert peak <= 3 * GIB
    for item in records:
        assert item["per_frame"] <= item["mean_count"] < item["per_frame"] + 1
    for name, expected in (("bright", 0.50), ("faint", 0.25), ("ultra", 0.25)):
        assert abs(fractions[name] - expected) <= 0.05
    assert in_bin / measured >= 0.85
    assert float(distances.max()) <= 3.0
    assert report["strict_residual_max"] < 1e-6
    assert report["step_violations"] == 0
