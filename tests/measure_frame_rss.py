"""Peak RSS of one worker that drops each frame after a scalar summary.

The slow test starts one child per length. Each child prints a single JSON
line. The parent asserts the 10-frame cap and the 5-versus-20 span, then
writes ``artifacts/m3_memory.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.runtime_env import require_genstarimg


def measure(n_frames: int) -> int:
    import gc
    import os
    import warnings
    from copy import deepcopy

    import psutil

    from starsim.camera import Camera
    from starsim.catalog import StarCatalog, default_cache_path
    from starsim.config import load_config, sample_sequence_params
    from starsim.motion import simulate_sequence
    from starsim.noise import build_sensor
    from starsim.photometry import anchor_photometry
    from starsim.psf import PsfModel
    from starsim.render import peak_energy_fraction
    from starsim.rng import SequenceRng, root_seed, sequence_seed
    from starsim.sequence import iter_frames, prepare_sequence

    require_genstarimg()
    process = psutil.Process(os.getpid())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = load_config()
    generator = deepcopy(loaded.generator)
    generator["dataset"]["n_frames"] = int(n_frames)
    catalog = StarCatalog.from_cache(default_cache_path())
    rngs = SequenceRng(sequence_seed(root_seed(20261029), 0))
    effective = sample_sequence_params(generator, rngs)
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
    total = 0.0
    count = 0
    for frame in iter_frames(plan, rngs, sensor, psf):
        snr = frame.exposure.snr
        total += float(snr.mean()) if snr.size else 0.0
        count += 1
        del frame, snr
        gc.collect()
    peak = int(process.memory_info().peak_wset)
    print(
        json.dumps(
            {
                "n_frames": int(n_frames),
                "peak_wset": peak,
                "frames": count,
                "snr_sum": total,
            }
        ),
        flush=True,
    )
    return peak


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tests/measure_frame_rss.py N_FRAMES")
    measure(int(sys.argv[1]))


if __name__ == "__main__":
    main()
