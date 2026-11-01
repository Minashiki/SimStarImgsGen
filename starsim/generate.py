"""Production sequence construction, audited replay and resumable batch CLI."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np
import psutil

from starsim import algorithm_versions
from starsim.camera import Camera
from starsim.catalog import StarCatalog, default_cache_path
from starsim.config import AppConfig, derive_camera, load_config, sample_sequence_params
from starsim.io import _sha256, load_objects, load_sequence, write_json, write_sequence
from starsim.metadata import jsonable, validate_versions
from starsim.motion import simulate_sequence
from starsim.noise import build_sensor
from starsim.photometry import anchor_photometry
from starsim.psf import PsfModel
from starsim.render import peak_energy_fraction
from starsim.rng import SequenceRng, root_seed, sequence_seed, _SEQUENCE_DOMAIN
from starsim.runtime import runtime_environment
from starsim.sequence import prepare_sequence
from starsim.split import audit, make_splits


def _equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return np.array_equal(a, b)


def check_camera(config):
    """D74: no stream or output may precede these exact comparisons."""
    recomputed = derive_camera(config.camera['sensor'])
    for field in fields(recomputed):
        if not _equal(getattr(recomputed, field.name), getattr(config.derived, field.name)):
            raise ValueError(f'inconsistent camera config.derived.{field.name}')
    if not _equal(config.camera.get('derived'), config.derived.as_dict()):
        raise ValueError('inconsistent camera camera_config.derived')


def _construct(config, rngs, catalog, regions, split):
    effective = sample_sequence_params(config, rngs)
    camera = Camera.from_derived(config.derived)
    effective['pointing']['split_constraints'] = dict(schema_version=1, split=split,
        **{key: regions[key] for key in ('mode', 'caps', 'guard_deg')})
    motion = simulate_sequence(effective, rngs, config.derived, sky_regions=regions,
                               split=split, catalog=catalog)
    sensor = build_sensor(effective, camera, rngs)
    psf = PsfModel.from_effective(effective['psf'], camera)
    photo = anchor_photometry(effective['photometry']['mag_limit'],
        effective['photometry']['snr_at_limit'], effective['photometry']['render_mag_margin'],
        sensor.reference_sigma(), peak_energy_fraction(psf))
    plan = prepare_sequence(catalog, camera, motion, effective, rngs, sensor, psf, photo)
    return plan, sensor, psf, camera, motion, photo, effective


def generate_sequence(config, seed, index, out_dir, *, catalog_path=None, _task_started=None):
    """Only AppConfig and seed/index determine the camera and simulation."""
    check_camera(config)
    if seed != config.generator['dataset']['seed']:
        raise ValueError('seed differs from configured dataset seed')
    regions = make_splits(config.generator['dataset']['split'], seed,
                          config.generator['dataset']['n_sequences'])
    split = regions['assignments'][index]
    catalog = StarCatalog.from_cache(catalog_path or default_cache_path())
    rngs = SequenceRng(sequence_seed(root_seed(seed), index))
    return _write(config, rngs, catalog, regions, split, out_dir, task_started=_task_started)


def _write(config, rngs, catalog, regions, split, out_dir, *, task_started=None):
    start = time.perf_counter()
    plan, sensor, psf, camera, motion, photo, effective = _construct(config, rngs, catalog, regions, split)
    construction_seconds = time.perf_counter()-start
    initialization_seconds = start-task_started if task_started is not None else None
    objects = write_sequence(out_dir, plan, rngs, sensor, psf, camera, motion, photo,
        catalog, config, effective, derived_camera=config.derived, split=split)
    return dict(initialization_seconds=initialization_seconds, construction_seconds=construction_seconds,
                split=split, boresight=motion.matrix[len(motion.matrix)//2, 2].tolist(),
                star_counts=[sum(c['class_id'] == 0 and c['status_default'] == 'labeled'
                                 for c in f.candidates) for f in objects],
                seconds=time.perf_counter()-start,
                generation_interval=[start, time.perf_counter()],
                peak_rss_bytes=psutil.Process().memory_info().peak_wset if os.name == 'nt'
                    else psutil.Process().memory_info().rss)


def replay_sequence(sequence_path, out_dir, *, catalog_path=None):
    record = load_sequence(sequence_path)
    from starsim.config import DerivedCamera
    raw = dict(record['derived_camera'])
    raw['K'] = np.asarray(raw['K'])
    config = AppConfig(record['camera_config'], record['config_unsampled'], DerivedCamera(**raw))
    check_camera(config)
    if int(record['rng']['entropy']) != config.generator['dataset']['seed']:
        raise ValueError('replay seed differs from configured dataset seed')
    validate_versions(record['algorithm_versions'])
    if record['environment'] != runtime_environment():
        raise ValueError('replay environment differs')
    catalog = StarCatalog.from_cache(catalog_path or default_cache_path())
    if catalog.meta != record['catalog']:
        raise ValueError('replay catalog differs')
    rngs = SequenceRng.from_metadata(record['rng'])
    seed = int(record['rng']['entropy'])
    regions = make_splits(config.generator['dataset']['split'], seed,
                          config.generator['dataset']['n_sequences'])
    index = record['rng']['spawn_key'][1]
    if regions['assignments'][index] != record['split']:
        raise ValueError('replay split differs')
    expected = dict(schema_version=1, split=record['split'],
                    **{key: regions[key] for key in ('mode', 'caps', 'guard_deg')})
    if record['config_effective']['pointing'].get('split_constraints') != expected:
        raise ValueError('replay split constraints differ or are missing')
    if record['config_effective']['pointing']['min_labeled_stars'] != config.generator['pointing']['min_labeled_stars']:
        raise ValueError('replay minimum-star constraint differs')
    return _write(config, rngs, catalog, regions, record['split'], out_dir)


def _atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    write_json(temporary, value)
    os.replace(temporary, path)


def _worker(config, seed, index, output, catalog_path):
    task_started = time.perf_counter()
    final = Path(output)/f'seq_{index:06d}'
    staging = Path(output)/f'.seq_{index:06d}.partial'
    # Only the explicitly owned per-sequence staging directory is removed.
    if staging.exists():
        if staging.resolve().parent != Path(output).resolve() or staging.is_symlink():
            raise ValueError('staging directory must stay inside the dataset directory')
        shutil.rmtree(staging)
    result = generate_sequence(config, seed, index, staging, catalog_path=catalog_path, _task_started=task_started)
    commit_start = time.perf_counter()
    result['files'] = {p.name: _sha256(p) for p in sorted(staging.iterdir()) if p.is_file()}
    hash_seconds = time.perf_counter()-commit_start
    result['bytes'] = sum(p.stat().st_size for p in staging.iterdir() if p.is_file())
    start = time.perf_counter()
    write_json(staging/'complete.json', result)
    staging.rename(final)
    publish_seconds = time.perf_counter()-start
    total = time.perf_counter()-commit_start
    result['commit_timing'] = dict(hash_seconds=hash_seconds, publish_seconds=publish_seconds,
        other_seconds=total-hash_seconds-publish_seconds, total_seconds=total)
    return index, result


def _signature(config, catalog=None):
    generator = deepcopy(config.generator)
    for key in ('workers', 'out_dir'):
        generator['dataset'].pop(key)
    return jsonable(dict(camera=config.camera, derived=config.derived, generator=generator,
        environment=runtime_environment(), algorithm_versions=algorithm_versions(),
        catalog=(catalog or StarCatalog.from_cache(default_cache_path())).meta))


def verify_completed(out, index, seed, regions, signature):
    final = Path(out)/f'seq_{index:06d}'
    try:
        record = json.loads((final/'complete.json').read_text(encoding='utf-8'))
        sequence = load_sequence(final/'sequence.json')
        n = len(sequence['motion']['matrix'])
        expected = {'sequence.json', 'timing.json'}
        expected.update(f'{prefix}_{k:03d}.{extension}' for k in range(n)
                        for prefix, extension in (('frame', 'fits.fz'), ('objects', 'npz'), ('meta', 'json')))
        if sequence['config_effective']['labels']['save_dense']:
            expected.add('labels_manifest.json')
            expected.update(f'labels_{k:03d}.npz' for k in range(n))
        if (set(record['files']) != expected or
                {p.name for p in final.iterdir()} != expected | {'complete.json'} or
                any(_sha256(final/name) != digest for name, digest in record['files'].items())):
            raise ValueError('file set or hashes differ')
        if record['bytes'] != sum((final/name).stat().st_size for name in record['files']):
            raise ValueError('bytes differ from sequence files')
        if int(sequence['rng']['entropy']) != seed or sequence['rng']['spawn_key'] != [_SEQUENCE_DOMAIN, index]:
            raise ValueError('sequence RNG identity differs')
        split = regions['assignments'][index]
        constraints = dict(schema_version=1, split=split,
                          **{key: regions[key] for key in ('mode', 'caps', 'guard_deg')})
        if sequence['split'] != split or sequence['config_effective']['pointing']['split_constraints'] != constraints:
            raise ValueError('split constraints differ')
        generator = deepcopy(sequence['config_unsampled'])
        for key in ('workers', 'out_dir'):
            generator['dataset'].pop(key)
        recorded_signature = dict(camera=sequence['camera_config'], derived=sequence['derived_camera'],
            generator=generator, environment=sequence['environment'],
            algorithm_versions=sequence['algorithm_versions'], catalog=sequence['catalog'])
        if recorded_signature != signature:
            raise ValueError('source signature differs')
        boresight = sequence['motion']['matrix'][n//2][2]
        counts = [sum(c['class_id'] == 0 and c['status_default'] == 'labeled'
                      for c in load_objects(final/f'objects_{k:03d}.npz').candidates) for k in range(n)]
        if record['split'] != split or record['boresight'] != boresight or record['star_counts'] != counts:
            raise ValueError('summary differs from sequence/objects')
        return dict(record, split=split, boresight=boresight, star_counts=counts)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'corrupt completed sequence {index}: {exc}') from exc


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _execute_tasks(task, tasks, *, workers, after_submit=None):
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             max_tasks_per_child=1) as pool:
        try:
            futures = [pool.submit(task, *args) for args in tasks]
            if after_submit is not None:
                after_submit()
            for future in as_completed(futures):
                yield future.result()
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise


def generate_dataset(config, *, resume=False, catalog_path=None):
    # Reject legacy manifests before signatures, run records or any writes.
    data = config.generator['dataset']
    path = Path(data['out_dir']).resolve()/'dataset.json'
    if resume:
        schema = json.loads(path.read_text(encoding='utf-8')).get('schema_version')
        if type(schema) is not int or schema != 2:
            raise ValueError(f'schema {schema} dataset (schema 1 used split 1) cannot resume; '
                             'regenerate in a new empty directory')
    check_camera(config)
    data = config.generator['dataset']
    if data['image_format'] != 'fits_rice' or data['export_registered']:
        raise ValueError('M5 supports fits_rice and export_registered=false')
    seed = data['seed']
    catalog_path = Path(catalog_path or default_cache_path()).resolve()
    catalog = StarCatalog.from_cache(catalog_path)
    out = Path(data['out_dir']).resolve()
    regions = make_splits(data['split'], seed, data['n_sequences'])
    signature = _signature(config, catalog)
    path = out/'dataset.json'
    if resume:
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if manifest['source'] != signature or manifest['regions'] != regions:
            raise ValueError('resume inputs/environment/versions/catalog differ')
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError('output must be empty; use --resume for an existing dataset')
        out.mkdir(parents=True, exist_ok=True)
        manifest = dict(schema_version=2, source=signature, regions=regions, sequences={}, runs=[], complete=False)
        _atomic_json(path, manifest)
    print(json.dumps(dict(area_fraction=regions['area_fraction'], guard_loss_fraction=regions['guard_loss_fraction'])), flush=True)
    pending = []
    recovered = {}
    previous_sequences = manifest['sequences']
    for old in manifest['runs']:
        if old['status'] == 'running':
            old.update(status='interrupted', active_seconds_is_lower_bound=True)
    run = dict(started_at=_utc_now(), updated_at=_utc_now(), active_seconds=0.,
               active_seconds_is_lower_bound=False, status='running')
    manifest['runs'].append(run)
    manifest['sequences'] = previous_sequences
    manifest['complete'] = False
    started = time.perf_counter()
    stop = threading.Event()
    minimum_available = [psutil.virtual_memory().available]
    concurrent_minimum = [None]
    maximum_active = [0]
    workers = min(data['workers'], max(0, (minimum_available[0]-3*2**30)//(4*2**30)))
    run['workers'] = int(workers)
    sampling_active = [False]
    def sample_memory():
        while not stop.wait(.1):
            available = psutil.virtual_memory().available
            minimum_available[0] = min(minimum_available[0], available)
            active = len(list(out.glob('.seq_*.partial'))) if sampling_active[0] else 0
            maximum_active[0] = max(maximum_active[0], active)
            if active >= 2:
                concurrent_minimum[0] = available if concurrent_minimum[0] is None else min(concurrent_minimum[0], available)
    def refresh():
        run.update(updated_at=_utc_now(), active_seconds=time.perf_counter()-started,
                   minimum_available_bytes=minimum_available[0])
        _atomic_json(path, manifest)
    refresh()
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        for index in range(data['n_sequences']):
            final = out/f'seq_{index:06d}'
            if final.exists():
                record = verify_completed(out, index, seed, regions, signature)
                old = previous_sequences.get(str(index), {})
                if 'commit_timing' in old:
                    record['commit_timing'] = old['commit_timing']
                recovered[str(index)] = record
            else:
                pending.append(index)
        manifest['sequences'] = recovered
        # Remove only owned stale staging directories before memory sampling starts.
        for index in pending:
            staging = out/f'.seq_{index:06d}.partial'
            if staging.exists():
                if staging.resolve().parent != out or staging.is_symlink():
                    raise ValueError('staging directory must stay inside the dataset directory')
                shutil.rmtree(staging)
        sampling_active[0] = True
        if pending and workers < 1:
            raise MemoryError('need 3 GiB reserve plus 4 GiB per worker')
        if pending:
            tasks = [(config, seed, i, str(out), str(catalog_path)) for i in pending]
            for index, result in _execute_tasks(_worker, tasks, workers=int(workers)):
                manifest['sequences'][str(index)] = result
                refresh()
                print(f'completed sequence {index}', flush=True)
        # All summaries have been reconstructed during recovery or produced by this run.
        distributions = audit(regions, list(manifest['sequences'].values()), data['split']['min_seqs_for_assert'])
        manifest.update(complete=True, star_counts=distributions, workers=int(workers),
                        total_sequence_seconds=sum(r['seconds'] for r in manifest['sequences'].values()),
                        sequence_bytes=sum(r['bytes'] for r in manifest['sequences'].values()))
        run['status'] = 'completed'
        manifest['completed_at'] = _utc_now()
    except BaseException:
        run['status'] = 'failed'
        raise
    finally:
        stop.set()
        sampler.join()
        refresh()
        run['max_active_workers'] = maximum_active[0]
        run['concurrent_minimum_available_bytes'] = concurrent_minimum[0]
        measured = [r for r in manifest['runs'] if r.get('concurrent_minimum_available_bytes') is not None]
        manifest['concurrent_memory'] = dict(max_active_workers=max(r.get('max_active_workers', 0) for r in manifest['runs']),
            minimum_available_bytes=min((r['concurrent_minimum_available_bytes'] for r in measured), default=None),
            sampling_interval_seconds=.1, active_definition='owned staging directories present during worker tasks')
        manifest['active_seconds'] = sum(r['active_seconds'] for r in manifest['runs'])
        manifest['active_seconds_is_lower_bound'] = any(r['active_seconds_is_lower_bound'] for r in manifest['runs'])
        if manifest['complete']:
            manifest['elapsed_seconds'] = (datetime.fromisoformat(manifest['completed_at'])-
                                          datetime.fromisoformat(manifest['runs'][0]['started_at'])).total_seconds()
        _atomic_json(path, manifest)
    print(json.dumps(dict(star_counts=distributions, bytes=manifest['sequence_bytes'])), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera')
    parser.add_argument('--config')
    parser.add_argument('--catalog')
    parser.add_argument('--out-dir')
    parser.add_argument('--n-sequences', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--replay', help='sequence.json to reproduce into --out-dir')
    args = parser.parse_args()
    if args.replay:
        if not args.out_dir:
            parser.error('--replay requires --out-dir')
        replay_sequence(args.replay, args.out_dir, catalog_path=args.catalog)
        return
    overrides = {key: value for key, value in vars(args).items()
                 if key in ('out_dir', 'n_sequences', 'seed', 'workers') and value is not None}
    config = load_config(args.camera, args.config, overrides=dict(generator=dict(dataset=overrides)))
    generate_dataset(config, resume=args.resume, catalog_path=args.catalog)


if __name__ == '__main__':
    main()
