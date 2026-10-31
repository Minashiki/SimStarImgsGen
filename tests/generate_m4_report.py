"""Generate D73 acceptance from a real blended field and reference derivative."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from starsim.camera import Camera
from starsim.catalog import StarCatalog, default_cache_path
from starsim.config import derive_camera, load_config, sample_sequence_params
from starsim.io import (load_sequence, save_dense, validate_labels_manifest,
                        write_json, write_labels_manifest, write_sequence)
from starsim.labels import assign_status
from starsim.motion import simulate_sequence
from starsim.noise import build_sensor
from starsim.photometry import anchor_photometry
from starsim.psf import PsfModel
from starsim.render import peak_energy_fraction
from starsim.render_labels import _render_frame, render_dense
from starsim.rng import SequenceRng, root_seed, sequence_seed
from starsim.sequence import prepare_sequence
from tests.runtime_env import require_genstarimg
from tests.test_m4_labels import candidate, sparse, CFG

ROOT = Path(__file__).resolve().parents[1]


def generate():
    require_genstarimg()
    loaded = load_config()
    rngs = SequenceRng(sequence_seed(root_seed(20261031), 0))
    effective = sample_sequence_params(loaded, rngs)
    effective['dataset']['n_frames'] = 8
    # Same angular field as the 4096 camera, sampled on 512 pixels.
    sensor_cfg = copy.deepcopy(loaded.camera['sensor'])
    sensor_cfg['resolution'] = dict(width=512, height=512)
    sensor_cfg['focal_length_mm'] = 37.5
    derived = derive_camera(sensor_cfg)
    camera = Camera.from_derived(derived)
    motion = simulate_sequence(effective, rngs, derived)
    sensor = build_sensor(effective, camera, rngs)
    psf = PsfModel.from_effective(effective['psf'], camera)
    photo = anchor_photometry(effective['photometry']['mag_limit'],
        effective['photometry']['snr_at_limit'], effective['photometry']['render_mag_margin'],
        sensor.reference_sigma(), peak_energy_fraction(psf))
    catalog = StarCatalog.from_cache(default_cache_path())
    plan = prepare_sequence(catalog, camera, motion, effective, rngs, sensor, psf, photo)
    (ROOT/".tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'.tmp', prefix='m4_report_') as temporary:
        base = Path(temporary); source = base/'sequence'; reference = base/'reference'
        objects = write_sequence(source, plan, rngs, sensor, psf, camera, motion, photo,
            catalog, loaded, effective, derived_camera=derived)
        sequence = load_sequence(source/'sequence.json')
        child = subprocess.run([sys.executable, str(ROOT/'scripts/verify_sequence.py'), str(source),
            '--catalog', str(default_cache_path())], cwd=base, capture_output=True, text=True, check=True)
        report = json.loads(child.stdout)
        assert report['modules_clean'] and report['blended'] > 0
        reference.mkdir()
        cfg = dict(effective['labels'], label_frame='reference', canvas='reference')
        statused = copy.deepcopy(objects); assign_status(statused, cfg)
        cropped = 0
        for k, frame in enumerate(statused):
            dense = _render_frame(statused, k, frame.H_to_ref, cfg)
            cropped += len(dense['cropped'])
            save_dense(reference/f'labels_{k:03d}.npz', dense)
        write_labels_manifest(reference, source, cfg, statused)
        validate_labels_manifest(source, source)
        validate_labels_manifest(reference, source)
        assert cropped > 0
        assert sequence['git_commit'] is not None and sequence['git_dirty'] is False
        report.update(git_commit=sequence['git_commit'], git_dirty=sequence['git_dirty'],
            root_seed=20261031, sequence_index=0, size=[camera.width, camera.height],
            derived_camera=sequence['derived_camera'], stride4_collisions=sequence['stride4_collisions'],
            reference=dict(canvas='reference', cropped=cropped, manifest_valid=True),
            manifest_valid=True, source_records=sum(len(f.sources) for f in objects),
            candidate_records=sum(len(f.candidates) for f in objects))
    timings = {}
    for n in (8, 32, 128):
        frames = [sparse([candidate(i) for i in range(20)], k=k) for k in range(n)]
        before = copy.deepcopy(frames)
        start = time.perf_counter(); render_dense(frames, n-1, np.eye(3), CFG)
        timings[str(n)] = time.perf_counter()-start
        assert all(a.sources == b.sources and a.candidates == b.candidates for a,b in zip(frames,before))
    report['render_dense_performance'] = dict(seconds=timings, candidates_per_frame=20,
        measured=True, input_unchanged=True)
    write_json(ROOT/'artifacts/m4_report.json', report)
    print(json.dumps(dict(blended=report['blended'], cropped=cropped, git_commit=report['git_commit'])))


if __name__ == '__main__':
    generate()
