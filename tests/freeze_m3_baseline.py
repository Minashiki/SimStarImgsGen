"""Freeze M3 bit baselines with the code that is current when this runs.

Default group: 3 sequences, the loaded 4096 camera, 10 frames, serial.
Small group: 2 sequences, a 512 px camera with the same f/p and real motion.

Each array is hashed with its field name, dtype, and shape. The manifest
records algorithm versions, the YAML text, and the catalog cache identity.
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
import warnings
from pathlib import Path

import numpy as np

from starsim import algorithm_versions
from starsim.catalog import StarCatalog, default_cache_path, sha256_file
from starsim.config import DerivedCamera, load_config
from starsim.motion import simulate_sequence
from starsim.photometry import anchor_photometry
from starsim.psf import PsfModel
from starsim.render import peak_energy_fraction
from starsim.rng import SequenceRng, root_seed, sequence_seed
from starsim.sequence import render_sequence
from starsim.targets import TargetPopulation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "m3_baseline" / "manifest.json"
BASELINE_ROOT = 20261029
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


def run_one(loaded, catalog, derived, index: int) -> dict:
    from starsim.camera import Camera
    from starsim.noise import build_sensor

    from starsim.config import sample_sequence_params

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
    result = render_sequence(catalog, camera, motion, effective, rngs, sensor, psf, photo)
    frames = []
    for frame in result.frames:
        exposure = frame.exposure
        arrays = {name: hash_named(name, getattr(exposure, name)) for name in FRAME_ARRAYS}
        arrays["group_id"] = hash_named("group_id", frame.blend.group_id)
        arrays["stars_image"] = hash_named("stars_image", frame.stars.image)
        frames.append(arrays)
    record = {
        "index": index,
        "population": hash_population(result.population),
        "frames": frames,
        "issued": hash_issued(rngs.issued),
        "issued_names": list(rngs.issued),
        "effective": jsonable(effective),
    }
    del result, sensor, psf
    gc.collect()
    return record


def catalog_identity(catalog: StarCatalog) -> dict:
    path = default_cache_path()
    meta = jsonable(catalog.meta)
    return {"cache_path": str(path), "cache_sha256": sha256_file(path), "meta": meta}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("small", "default", "all"), default="all")
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = load_config()
    catalog = StarCatalog.from_cache(default_cache_path())
    camera_yaml = (ROOT / "configs" / "camera_default.yaml").read_text(encoding="utf-8")
    generator_yaml = (ROOT / "configs" / "generator_default.yaml").read_text(encoding="utf-8")
    manifest = {
        "freeze_reason": "RENDER_VERSION 3: moving-target exposure radius uses the radial envelope",
        "root_seed": BASELINE_ROOT,
        "algorithm_versions": algorithm_versions(),
        "camera_yaml": camera_yaml,
        "generator_yaml": generator_yaml,
        "catalog": catalog_identity(catalog),
        "groups": {},
    }
    if OUT.exists() and args.group != "all":
        manifest = json.loads(OUT.read_text(encoding="utf-8"))
    if args.group in ("small", "all"):
        print("freezing small", flush=True)
        derived = small_derived(loaded.derived)
        manifest["groups"]["small"] = {
            "side": 512,
            "indices": [0, 1],
            "sequences": [run_one(loaded, catalog, derived, index) for index in (0, 1)],
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(manifest), encoding="utf-8")
        print("small written", flush=True)
    if args.group in ("default", "all"):
        print("freezing default", flush=True)
        sequences = []
        for index in (0, 1, 2):
            print(f"default sequence {index}", flush=True)
            sequences.append(run_one(loaded, catalog, loaded.derived, index))
            manifest["groups"]["default"] = {
                "side": int(loaded.derived.width),
                "indices": [0, 1, 2],
                "sequences": sequences,
            }
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(manifest), encoding="utf-8")
        print("default written", flush=True)


if __name__ == "__main__":
    main()
