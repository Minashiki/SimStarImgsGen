"""Freeze M3 bit baselines with the code that is current when this runs.

Default group: 3 sequences, the loaded 4096 camera, 10 frames, serial.
Small group: 2 sequences, a 512 px camera with the same f/p and real motion.

Frames are hashed one at a time through ``prepare_sequence`` and
``iter_frames``. ``render_sequence`` is not used: it keeps every exposure.

Each group stores its own provenance. There is no shared header. ``--group
all`` replaces the manifest only after both groups succeed. Freezing one
group first re-checks the other group and keeps that group's provenance.
``complete`` is recorded, and the comparator does not trust it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import warnings

import numpy as np

from starsim import algorithm_versions
from starsim.catalog import StarCatalog, default_cache_path, sha256_file
from starsim.config import DerivedCamera, load_config
from starsim.motion import simulate_sequence
from starsim.photometry import anchor_photometry
from starsim.psf import PsfModel
from starsim.render import peak_energy_fraction
from starsim.rng import SequenceRng, root_seed, sequence_seed
from starsim.sequence import iter_frames, prepare_sequence
from starsim.targets import TargetPopulation
from tests.runtime_env import require_genstarimg, runtime_environment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "m3_baseline" / "manifest.json"
BASELINE_ROOT = 20261029
FREEZE_REASON = "切换到 D12 环境"
SEED_NOTE = "覆盖 YAML 中的 dataset.seed"
GROUP_INDICES = {"small": (0, 1), "default": (0, 1, 2)}
FRAME_ARRAYS = (
    "adu",
    "saturation_mask",
    "saturation_owner",
    "fullwell_mask",
    "adc_mask",
    "s_self",
    "sigma_local",
    "snr",
    "peak_clipped",
    "peak_owner",
    "center_clipped",
    "center_owner",
)
FRAME_FIELDS = FRAME_ARRAYS + ("group_id", "stars_image")
SENSOR_FIELDS = ("background_e", "vignetting", "prnu", "dark_e")


def hash_named(name: str, value) -> str:
    """sha256 of one field, including its name, dtype, and shape."""
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(b"\0")
        if array.dtype.kind in ("U", "O", "S"):
            payload = "\n".join(str(item) for item in array.ravel()).encode("utf-8")
        else:
            payload = array.tobytes()
        digest.update(payload)
        return digest.hexdigest()
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        digest.update(b"int\0()\0")
        digest.update(str(int(value)).encode("utf-8"))
        return digest.hexdigest()
    raise TypeError(f"{name} is not an array or int, got {type(value).__name__}")


def hash_population(population: TargetPopulation) -> dict[str, str]:
    hashes = {}
    for field in dataclasses.fields(population):
        hashes[field.name] = hash_named(field.name, getattr(population, field.name))
    return hashes


def population_field_names() -> set[str]:
    return {field.name for field in dataclasses.fields(TargetPopulation)}


def hash_issued(names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"issued\0")
    digest.update("\n".join(names).encode("utf-8"))
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def small_derived(full: DerivedCamera) -> DerivedCamera:
    side = 512
    center = (side - 1) / 2.0
    scale = side / float(full.width)
    return DerivedCamera(
        width=side,
        height=side,
        pixel_size_mm=full.pixel_size_mm,
        focal_length_mm=full.focal_length_mm,
        aperture_mm=full.aperture_mm,
        fov_width_deg=full.fov_width_deg * scale,
        fov_height_deg=full.fov_height_deg * scale,
        pixel_scale_arcsec=full.pixel_scale_arcsec,
        half_diagonal_deg=full.half_diagonal_deg * scale,
        f_over_p=full.f_over_p,
        cx=center,
        cy=center,
        K=np.array(
            [[full.f_over_p, 0.0, center], [0.0, full.f_over_p, center], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )


def configured_n_frames(loaded) -> int:
    value = loaded.generator["dataset"]["n_frames"]
    if type(value) is not int:
        raise ValueError(f"dataset.n_frames must be an int for the baseline, got {value!r}")
    return value


def run_one(loaded, catalog, derived, index: int) -> dict:
    from starsim.camera import Camera
    from starsim.config import sample_sequence_params
    from starsim.noise import build_sensor

    rngs = SequenceRng(sequence_seed(root_seed(BASELINE_ROOT), index))
    effective = sample_sequence_params(loaded, rngs)
    camera = Camera.from_derived(derived)
    motion = simulate_sequence(effective, rngs, derived)
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
    frames = []
    for frame in iter_frames(plan, rngs, sensor, psf):
        exposure = frame.exposure
        arrays = {name: hash_named(name, getattr(exposure, name)) for name in FRAME_ARRAYS}
        arrays["group_id"] = hash_named("group_id", frame.blend.group_id)
        arrays["stars_image"] = hash_named("stars_image", frame.stars.image)
        frames.append(arrays)
        del frame, exposure, arrays
        gc.collect()
    record = {
        "index": index,
        "population": hash_population(plan.population),
        "sensor": {name: hash_named(name, getattr(sensor, name)) for name in SENSOR_FIELDS},
        "frames": frames,
        "issued": hash_issued(rngs.issued),
        "issued_names": list(rngs.issued),
        "effective": jsonable(effective),
    }
    del plan, sensor, psf
    gc.collect()
    return record


def catalog_identity(catalog: StarCatalog) -> dict:
    path = default_cache_path()
    meta = jsonable(catalog.meta)
    return {"cache_path": str(path), "cache_sha256": sha256_file(path), "meta": meta}


def _load_quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return load_config()


def _provenance(catalog: StarCatalog, camera_sha: str, generator_sha: str) -> dict:
    return {
        "freeze_reason": FREEZE_REASON,
        "environment": runtime_environment(),
        "algorithm_versions": algorithm_versions(),
        "camera_yaml_sha256": camera_sha,
        "generator_yaml_sha256": generator_sha,
        "catalog": catalog_identity(catalog),
        "root_seed": BASELINE_ROOT,
        "seed_note": SEED_NOTE,
    }


def _preflight(group: dict, catalog: StarCatalog, camera_sha: str, generator_sha: str) -> str | None:
    reasons = []
    if group.get("environment") != runtime_environment():
        reasons.append("environment")
    if group.get("algorithm_versions") != algorithm_versions():
        reasons.append("algorithm_versions")
    if group.get("camera_yaml_sha256") != camera_sha:
        reasons.append("camera_yaml")
    if group.get("generator_yaml_sha256") != generator_sha:
        reasons.append("generator_yaml")
    if group.get("catalog") != catalog_identity(catalog):
        reasons.append("catalog")
    if not reasons:
        return None
    return "preflight: " + ", ".join(reasons)


def baseline_problems(group_name: str) -> list[str]:
    """Compare one group. A provenance mismatch is a single preflight failure.

    Sequence indices, sequence count, frame count, frame fields, and the
    population field set are checked here. ``complete`` is ignored.
    Sensor hashes are compared before the other arrays so an environment
    drift names ``background_e`` first.
    """
    if group_name not in GROUP_INDICES:
        return [f"unknown baseline group {group_name}"]
    if not OUT.is_file():
        return ["preflight: baseline manifest is missing"]
    manifest = json.loads(OUT.read_text(encoding="utf-8"))
    groups = manifest.get("groups") if isinstance(manifest, dict) else None
    group = groups.get(group_name) if isinstance(groups, dict) else None
    if not isinstance(group, dict):
        return [f"preflight: group {group_name} is missing"]
    loaded = _load_quiet()
    catalog = StarCatalog.from_cache(default_cache_path())
    camera_sha = sha256_file(ROOT / "configs" / "camera_default.yaml")
    generator_sha = sha256_file(ROOT / "configs" / "generator_default.yaml")
    preflight = _preflight(group, catalog, camera_sha, generator_sha)
    if preflight is not None:
        return [preflight]
    n_frames = configured_n_frames(loaded)
    expected = list(GROUP_INDICES[group_name])
    sequences = group.get("sequences")
    if not isinstance(sequences, list):
        return [f"{group_name} sequences missing"]
    problems = []
    found = [item.get("index") if isinstance(item, dict) else None for item in sequences]
    if found != expected or len(sequences) != len(expected):
        problems.append(f"{group_name} indices {found} != {expected}")
    if group.get("indices") != expected:
        problems.append(f"{group_name} declared indices {group.get('indices')} != {expected}")
    if group.get("n_frames") != n_frames:
        problems.append(f"{group_name} n_frames {group.get('n_frames')} != {n_frames}")
    derived = loaded.derived if group_name == "default" else small_derived(loaded.derived)
    by_index = {}
    for item in sequences:
        if isinstance(item, dict):
            by_index.setdefault(item.get("index"), item)
    pop_fields = population_field_names()
    frame_fields = set(FRAME_FIELDS)
    for index in expected:
        record = by_index.get(index)
        if record is None:
            problems.append(f"seq {index} missing")
            continue
        got = run_one(loaded, catalog, derived, index)
        sensor = record.get("sensor")
        got_sensor = got.get("sensor")
        if not isinstance(sensor, dict) or set(sensor) != set(SENSOR_FIELDS) or set(got_sensor) != set(SENSOR_FIELDS):
            problems.append(f"seq {index} sensor fields")
        else:
            for name in SENSOR_FIELDS:
                if got_sensor.get(name) != sensor.get(name):
                    problems.append(f"seq {index} sensor {name}")
        got_pop = got["population"]
        want_pop = record.get("population")
        if not isinstance(want_pop, dict) or set(got_pop) != pop_fields or set(want_pop) != pop_fields:
            problems.append(f"seq {index} population fields")
        else:
            for key in sorted(pop_fields):
                if got_pop.get(key) != want_pop.get(key):
                    problems.append(f"seq {index} pop {key}")
        if got["issued"] != record.get("issued") or got["issued_names"] != record.get("issued_names"):
            problems.append(f"seq {index} issued")
        want_frames = record.get("frames")
        if not isinstance(want_frames, list) or len(want_frames) != n_frames or len(got["frames"]) != n_frames:
            problems.append(
                f"seq {index} frame count got {len(got['frames'])} "
                f"manifest {len(want_frames) if isinstance(want_frames, list) else None} != {n_frames}"
            )
            continue
        for frame_index in range(n_frames):
            got_frame = got["frames"][frame_index]
            want_frame = want_frames[frame_index]
            if not isinstance(want_frame, dict) or set(got_frame) != frame_fields or set(want_frame) != frame_fields:
                problems.append(f"seq {index} frame {frame_index} fields")
                continue
            for key in FRAME_FIELDS:
                if got_frame.get(key) != want_frame.get(key):
                    problems.append(f"seq {index} frame {frame_index} {key}")
    return problems


def _build_group(loaded, catalog, derived, indices: tuple[int, ...], side: int, provenance: dict) -> dict:
    sequences = []
    for index in indices:
        print(f"sequence {index}", flush=True)
        sequences.append(run_one(loaded, catalog, derived, index))
    group = dict(provenance)
    group.update(
        {
            "side": int(side),
            "n_frames": configured_n_frames(loaded),
            "indices": list(indices),
            "sequences": sequences,
        }
    )
    return group


def _atomic_write(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.name + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, OUT)


def main() -> None:
    require_genstarimg()
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("small", "default", "all"), default="all")
    args = parser.parse_args()
    loaded = _load_quiet()
    catalog = StarCatalog.from_cache(default_cache_path())
    camera_sha = sha256_file(ROOT / "configs" / "camera_default.yaml")
    generator_sha = sha256_file(ROOT / "configs" / "generator_default.yaml")
    provenance = _provenance(catalog, camera_sha, generator_sha)
    if args.group == "all":
        print("freezing small", flush=True)
        small = _build_group(loaded, catalog, small_derived(loaded.derived), GROUP_INDICES["small"], 512, provenance)
        print("freezing default", flush=True)
        default = _build_group(
            loaded, catalog, loaded.derived, GROUP_INDICES["default"], int(loaded.derived.width), provenance
        )
        _atomic_write({"complete": True, "groups": {"small": small, "default": default}})
        print("both groups written", flush=True)
        return
    other = "default" if args.group == "small" else "small"
    print(f"checking {other} before replacing {args.group}", flush=True)
    problems = baseline_problems(other)
    if problems:
        raise SystemExit("refusing to replace the manifest; other group failed:\n" + "\n".join(problems))
    manifest = json.loads(OUT.read_text(encoding="utf-8"))
    if args.group == "small":
        print("freezing small", flush=True)
        manifest["groups"]["small"] = _build_group(
            loaded, catalog, small_derived(loaded.derived), GROUP_INDICES["small"], 512, provenance
        )
    else:
        print("freezing default", flush=True)
        manifest["groups"]["default"] = _build_group(
            loaded, catalog, loaded.derived, GROUP_INDICES["default"], int(loaded.derived.width), provenance
        )
    manifest["complete"] = "small" in manifest["groups"] and "default" in manifest["groups"]
    _atomic_write(manifest)
    print(f"{args.group} written", flush=True)


if __name__ == "__main__":
    main()
