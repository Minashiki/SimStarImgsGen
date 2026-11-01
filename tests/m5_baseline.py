"""Run against ad66eba before optimization, then compare the same inputs."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from starsim.io import save_objects, save_dense
from starsim.labels import assign_status
from starsim.render_labels import _render_frame
from starsim.runtime import runtime_environment
from tests.test_m4_labels import disk_sequence, scene, CFG


def report_sequence(base):
    from starsim.camera import Camera
    from starsim.catalog import StarCatalog, default_cache_path
    from starsim.config import derive_camera, load_config, sample_sequence_params
    from starsim.io import write_sequence
    from starsim.motion import simulate_sequence
    from starsim.noise import build_sensor
    from starsim.photometry import anchor_photometry
    from starsim.psf import PsfModel
    from starsim.render import peak_energy_fraction
    from starsim.rng import SequenceRng, root_seed, sequence_seed
    from starsim.sequence import prepare_sequence
    config = load_config()
    rngs = SequenceRng(sequence_seed(root_seed(20261031), 0))
    effective = sample_sequence_params(config, rngs)
    effective['dataset']['n_frames'] = 8
    sensor_cfg = copy.deepcopy(config.camera['sensor'])
    sensor_cfg.update(resolution=dict(width=512, height=512), focal_length_mm=37.5)
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
    return write_sequence(base/'report', plan, rngs, sensor, psf, camera, motion, photo,
                          catalog, config, effective, derived_camera=derived)


def snapshot():
    from starsim.catalog import default_cache_path
    from starsim.config import DEFAULT_CAMERA_PATH, DEFAULT_GENERATOR_PATH
    from tests.runtime_env import require_genstarimg
    require_genstarimg()
    result = {}
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        class Factory:
            def mktemp(self, name):
                path = base/name
                path.mkdir()
                return path
        disk = disk_sequence.__wrapped__(Factory())
        groups = dict(reference_union=disk.objects)
        groups['native'] = copy.deepcopy(disk.objects)
        report = report_sequence(base)
        groups['report_native'] = report
        groups['report_reference'] = copy.deepcopy(report)
        for name, x, flux in [('boundary', [12., 18.], [1000., 500.]),
                              ('tie', [12., 13.], [1000., 1000.])]:
            groups[name] = [scene(x, flux=flux)[0]]
        for name, objects in groups.items():
            cfg = dict(CFG, label_frame='reference' if name == 'reference_union' else 'native',
                       motion=name == 'reference_union')
            if name == 'report_reference':
                cfg.update(label_frame='reference', canvas='reference')
            assign_status(objects, cfg)
            for k, frame in enumerate(objects):
                objects_path = base/'objects.npz'
                save_objects(objects_path, frame)
                result[f'{name}/{k}/objects'] = hashlib.sha256(objects_path.read_bytes()).hexdigest()
                for stride in (1, 4):
                    import numpy as np
                    H = frame.H_to_ref if cfg['label_frame'] == 'reference' else np.eye(3)
                    dense = _render_frame(objects, k, H, dict(cfg, stride=stride))
                    path = base/'dense.npz'
                    save_dense(path, dense)
                    result[f'{name}/{k}/stride{stride}'] = hashlib.sha256(path.read_bytes()).hexdigest()
        result['statistics'] = [json.loads(p.read_text())['statistics']
                                for p in sorted(disk.directory.glob('meta_*.json'))]
        result['report_statistics'] = [json.loads(p.read_text())['statistics']
                                      for p in sorted((base/'report').glob('meta_*.json'))]
    return dict(source_commit='ad66eba', environment=runtime_environment(),
                # Archive and checkout can have different line endings.
                # Normalize YAML line endings, leaving binary NPZ untouched.
                files={p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')
                       if p.suffix == '.yaml' else p.read_bytes()).hexdigest()
                       for p in (DEFAULT_CAMERA_PATH, DEFAULT_GENERATOR_PATH, default_cache_path())},
                inputs=dict(root_seed=20261031, sequence_index=0, frames=4,
                            fixture='tests.test_m4_labels.disk_sequence', report_fixture='tests.generate_m4_report.generate',
                            report_frames=8, strides=[1, 4]), results=result)


if __name__ == '__main__':
    if '--freeze' in sys.argv:
        import io
        import shutil
        import subprocess
        import tarfile
        root = Path(__file__).resolve().parents[1]
        archive = subprocess.check_output(['git', '-c', f'safe.directory={root.as_posix()}',
                                           'archive', 'ad66eba'], cwd=root)
        (root/'.tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root/'.tmp', prefix='m5_frozen_') as directory:
            frozen = Path(directory)
            with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
                contents.extractall(frozen, filter='data')
            shutil.copy(__file__, frozen/'tests/m5_baseline.py')
            (frozen/'cache').mkdir(exist_ok=True)
            shutil.copy(root/'cache/tycho1.npz', frozen/'cache/tycho1.npz')
            subprocess.run([sys.executable, '-m', 'tests.m5_baseline', str(Path(sys.argv[1]).resolve())],
                           cwd=frozen, check=True)
        sys.exit(0)
    output = snapshot()
    path = Path(sys.argv[1])
    if '--compare' in sys.argv:
        expected = json.loads(path.read_text())
        differences = []
        def compare(a, b, name=''):
            if isinstance(a, dict) and isinstance(b, dict):
                for key in sorted(a.keys() | b.keys()):
                    compare(a.get(key), b.get(key), f'{name}/{key}')
            elif a != b:
                differences.append(name)
        compare(output, expected)
        if differences:
            Path('.tmp/m5_current_snapshot.json').write_text(json.dumps(output, indent=2, sort_keys=True))
            raise AssertionError(f'M5 frozen output changed: {differences}')
    else:
        path.write_text(json.dumps(output, indent=2, sort_keys=True)+'\n')
