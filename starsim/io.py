"""Lossless FITS and pickle-free sparse/dense NPZ sequence files."""

import hashlib
import json
import os
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
from astropy.io import fits

from starsim.labels import FrameLabels
from starsim.metadata import jsonable, validate_versions

OBJECTS_SCHEMA_VERSION = 2
CANDIDATE_COMMON = frozenset("label_group_id class_id members components blended identity stable_key representative_member x y x_ref y_ref flux_e sigma_psf in_image has_flux dominant_source saturated in_foreign_sat in_defect_sat group_peak_row group_peak_col s_group sigma_local snr snr_eff identity_frame_count status_default ignore_reason_default ra dec".split())
CANDIDATE_STAR = frozenset("vmag member_identity_counts".split())
CANDIDATE_TARGET = frozenset("track_id mag_eq tier_design tier_measured duty_eff t_exp_eff x1 y1 x2 y2 x1_ref y1_ref x2_ref y2_ref L_true L_native angle_rad angle_ref x_vis y_vis vis_x1 vis_y1 vis_x2 vis_y2 L_vis vis_frac vis_x1_ref vis_y1_ref vis_x2_ref vis_y2_ref x_vis_ref y_vis_ref L_vis_ref vx_ref vy_ref truncated entering exiting never_full".split())
EMPTY_TABLE_DTYPE = np.dtype([("id", "<U1")])
TABLE_NAMES = ("sources_stars", "sources_points", "sources_streaks", "stars", "points", "streaks")


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True,
                                     indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _validate_values(value):
    value = jsonable(value)
    if value is None:
        raise ValueError("None is not allowed in objects")
    if isinstance(value, dict):
        for child in value.values():
            _validate_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_values(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            finite = np.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("non-finite numeric value in objects")


def _table(records):
    """Structured columns; ragged components use JSON text, never pickle."""
    if not records:
        return np.empty(0, dtype=[("id", "U1")])
    _validate_values(records)
    keys = sorted(records[0])
    if any(set(r) != set(keys) for r in records):
        raise ValueError("table field set drift")
    columns = {}
    for key in keys:
        values = [r[key] for r in records]
        if all(type(v) is bool for v in values):
            columns[key] = np.array(values, dtype=bool)
        elif all(type(v) is int for v in values):
            columns[key] = np.array(values, dtype=np.int64)
        elif all(type(v) in (int, float) for v in values):
            columns[key] = np.array(values, dtype=np.float64)
        else:
            columns[key] = np.array([json.dumps(jsonable(v), sort_keys=True, separators=(",", ":"), allow_nan=False)
                                     for v in values])
    table = np.empty(len(records), dtype=[(key, arr.dtype) for key, arr in columns.items()])
    for key, arr in columns.items():
        table[key] = arr
    return table


def _records(table):
    result = []
    for row in table:
        result.append({key: json.loads(str(row[key])) if table.dtype[key].kind == "U"
                       else row[key].item() for key in table.dtype.names})
    return result


def _check_objects_arrays(data):
    if "schema_version" not in data:
        raise ValueError("unsupported legacy objects format (不支持旧格式): missing schema_version")
    required = set(TABLE_NAMES) | {"schema_version", "frame_index", "size", "H_to_ref"}
    if not required <= set(data):
        raise ValueError("missing objects tables or arrays: " + str(sorted(required-set(data))))
    schema, index, size, H = (data[key] for key in ("schema_version", "frame_index", "size", "H_to_ref"))
    if schema.shape != () or schema.dtype.kind not in "iu" or schema != OBJECTS_SCHEMA_VERSION:
        raise ValueError("unsupported objects schema_version")
    if index.shape != () or index.dtype.kind not in "iu" or index < 0:
        raise ValueError("frame_index must be a nonnegative scalar integer")
    if size.shape != (2,) or size.dtype.kind not in "iu" or np.any(size <= 0):
        raise ValueError("size must contain two positive integers")
    if H.dtype != np.dtype("float64") or H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise ValueError("H_to_ref must be a finite float64 3x3 array")
    for j, name in enumerate(TABLE_NAMES):
        table = data[name]
        if table.ndim != 1 or table.dtype.names is None:
            raise ValueError(f"{name}: expected one-dimensional structured table")
        if not len(table):
            if table.dtype != EMPTY_TABLE_DTYPE:
                raise ValueError(f"{name}: invalid empty table dtype")
            continue
        if "class_id" not in table.dtype.names or table.dtype["class_id"].kind not in "iu" or np.any(table["class_id"] != j % 3):
            raise ValueError(f"{name}: invalid class_id")
        if j >= 3 and set(table.dtype.names) != CANDIDATE_COMMON | (CANDIDATE_STAR if j == 3 else CANDIDATE_TARGET):
            raise ValueError(f"{name}: candidate field whitelist mismatch")


def _check_target_members(frame):
    for c in frame.candidates:
        if c["class_id"] not in (0, 1, 2):
            raise ValueError("invalid candidate class_id")
        if c["class_id"] == 0:
            continue
        members = c["members"]
        if len(members) != 1 or type(members[0]) is not int or not 0 <= members[0] < len(frame.sources):
            raise ValueError("target candidate must have one member")
        source = frame.sources[members[0]]
        for key in CANDIDATE_TARGET | set("x y x_ref y_ref flux_e ra dec".split()):
            if c.get(key) != source.get(key):
                raise ValueError(f"target candidate differs from member: {key}")


def save_objects(path, frame, type_registry=None):
    identities = [c["identity"] for c in frame.candidates]
    if len(identities) != len(set(identities)):
        raise ValueError("candidate identity must be unique within a frame")
    _check_target_members(frame)
    if any(r["class_id"] not in (0, 1, 2) for r in frame.sources):
        raise ValueError("invalid source class_id")
    tables = {}
    registry = {} if type_registry is None else type_registry
    for name, records in zip(TABLE_NAMES, [
            [r for r in frame.sources if r["class_id"] == cls] for cls in range(3)]+[
            [r for r in frame.candidates if r["class_id"] == cls] for cls in range(3)]):
        table = _table(records)
        if records:
            signature = {key: {"b": "bool", "i": "int64", "f": "float64", "U": "JSON"}[
                table.dtype[key].kind] for key in table.dtype.names}
            if name in registry and registry[name] != signature:
                raise ValueError(f"{name}: table fields or storage types drift across frames")
            registry[name] = signature
        tables[name] = table
    arrays = dict(schema_version=np.array(OBJECTS_SCHEMA_VERSION),
                  frame_index=np.array(frame.frame_index), size=np.array([frame.width, frame.height]),
                  H_to_ref=frame.H_to_ref)
    for value in arrays.values():
        _validate_values(value)
    _check_objects_arrays(dict(tables, **arrays))
    np.savez_compressed(path, **tables, **arrays)


def load_objects(path):
    with np.load(path, allow_pickle=False) as data:
        _check_objects_arrays(data)
        candidates = sum((_records(data[key]) for key in TABLE_NAMES[3:]), [])
        sources = sum((_records(data[key]) for key in TABLE_NAMES[:3]), [])
        sources.sort(key=lambda r: r["source_index"])
        if [r["source_index"] for r in sources] != list(range(len(sources))):
            raise ValueError("source_index must be continuous and unique")
        candidates.sort(key=lambda c: c["label_group_id"])
        _validate_values([sources, candidates, data["frame_index"], data["size"], data["H_to_ref"]])
        frame = FrameLabels(int(data["frame_index"]), int(data["size"][0]), int(data["size"][1]),
                           sources, candidates, data["H_to_ref"].copy())
        _check_target_members(frame)
        return frame


def save_dense(path, dense):
    # L uses original-pixel units: float16 cannot guarantee 0.01 px for long
    # streaks. Offsets, streak geometry and motion stay float32.
    arrays = {key: value.astype(np.float16) if value.dtype.kind == "f"
              and key not in ("offset", "streak_geom", "motion", "H") else value for key, value in dense.items()}
    arrays["offset"] = dense["offset"].astype(np.float32, copy=True)
    # Long ellipses can round a shoulder to 1 in half precision. Only the
    # true integer peak may stay exactly 1 after storage.
    shoulder = (arrays["heatmap"] == 1) & (dense["reg_mask"] == 0)
    arrays["heatmap"][shoulder] = np.nextafter(np.float16(1), np.float16(0))
    np.minimum(arrays["offset"], np.nextafter(np.float32(.5), np.float32(0)), out=arrays["offset"])
    np.savez_compressed(path, **arrays)


def save_fits(path, adu, wcs, *, frame_index, exposure_s):
    header = fits.Header(wcs)
    header["FRAME"] = frame_index
    header["EXPTIME"] = exposure_s
    header["ORIGIN"] = "GenStarImgs"
    hdu = fits.CompImageHDU(data=adu, header=header, compression_type="RICE_1")
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def load_sequence(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_versions(value["algorithm_versions"])
    from starsim.rng import SequenceRng
    SequenceRng.from_metadata(value["rng"])
    return value


_TARGET_FIELDS = set("track_id kind strict p0_x p0_y speed direction_x direction_y accel_along curve_amp curve_shape sigma_cross sigma_along duty t_exp_eff angle_sigma_rad tier_design snr_design flux_design entering exiting never_full".split())
_TARGET_FRAME_FIELDS = set("delta_cross delta_along angle_jitter_rad jitter flux_e mag_eq x_ref y_ref x_c y_c x1 y1 x2 y2 x1_ref y1_ref x2_ref y2_ref L_true L_native angle_rad angle_native angle_ref in_image x_vis y_vis vis_x1 vis_y1 vis_x2 vis_y2 L_vis vis_frac truncated".split())
_MOTION_STEPS = set("center_shift_px omega_cam_rad step_angle_rad speed_px_per_frame speed_deg_per_s direction_rad".split())
_MOTION_FRAMES = set("quaternion matrix ra_rad dec_rad roll_rad boresight_ra_rad boresight_dec_rad euler_phi_rad euler_theta_rad euler_psi_rad north_angle_rad H_to_ref cumulative_px".split())


def preflight_sequence(plan, sensor, motion, effective, camera, derived):
    n = len(motion.matrix)
    if n < 2 or any(v != n for v in (len(plan.stars), sensor.n_frames, effective["dataset"]["n_frames"])):
        raise ValueError("sequence frame counts must agree and N >= 2")
    for name in _MOTION_FRAMES | _MOTION_STEPS:
        value = getattr(motion, name)
        if value.ndim == 0 or len(value) != (n-1 if name in _MOTION_STEPS else n):
            raise ValueError(f"motion.{name}: inconsistent frame count")
    pop = plan.population
    registered = _TARGET_FIELDS | _TARGET_FRAME_FIELDS
    array_fields = {f.name for f in fields(pop)
                    if f.type in (np.ndarray, "np.ndarray")
                    or isinstance(getattr(pop, f.name), np.ndarray)}
    if array_fields != registered:
        raise ValueError("population array fields differ from registered contract")
    t = len(pop.track_id)
    for name in registered:
        value = getattr(pop, name)
        if not isinstance(value, np.ndarray) or value.shape != ((t,) if name in _TARGET_FIELDS else (t, n)):
            raise ValueError(f"population.{name}: invalid shape")
    if len(set(pop.track_id.tolist())) != t:
        raise ValueError("population track_id must be unique")
    from starsim.camera import Camera
    expected = Camera.from_derived(derived)
    if any(not np.array_equal(getattr(camera, f.name), getattr(expected, f.name)) for f in fields(expected)):
        raise ValueError("camera differs from derived_camera")
    for name in ("height", "width", "cx", "cy"):
        if getattr(sensor, name) != getattr(camera, name):
            raise ValueError(f"sensor.{name} differs from camera")
    h, w = camera.height, camera.width
    shapes = {name: (h, w) for name in ("background_e", "vignetting", "prnu", "dark_e")}
    shapes.update(row_offset_e=(h,), col_offset_e=(w,))
    nh, nd = len(sensor.hot_row), len(sensor.dead_row)
    shapes.update({name: (nh,) for name in ("hot_row", "hot_col", "hot_kind", "hot_level_a", "hot_level_b", "hot_p_flip")})
    shapes.update(hot_state=(nh, n), dead_row=(nd,), dead_col=(nd,))
    for name, shape in shapes.items():
        if np.shape(getattr(sensor, name)) != shape:
            raise ValueError(f"sensor.{name}: invalid shape")
    for name, limit in (("hot_row", h), ("dead_row", h), ("hot_col", w), ("dead_col", w)):
        value = np.asarray(getattr(sensor, name))
        if value.dtype.kind not in "iu" or np.any(value < 0) or np.any(value >= limit):
            raise ValueError(f"sensor.{name}: invalid integer coordinates")
    return n


def prepare_output_directory(directory):
    out = Path(directory)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("output directory must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_count(sequence):
    n = len(sequence["motion"]["matrix"])
    if n < 2 or sequence["config_effective"]["dataset"]["n_frames"] != n:
        raise ValueError("sequence frame counts must agree and N >= 2")
    for name in _MOTION_FRAMES | _MOTION_STEPS:
        if len(sequence["motion"][name]) != (n-1 if name in _MOTION_STEPS else n):
            raise ValueError(f"motion.{name}: inconsistent frame count")
    return n


def write_labels_manifest(derived_dir, sequence_dir, cfg, objects, *, canvas=None):
    from starsim import algorithm_versions
    from starsim.render_labels import canvas_geometry

    derived, source = Path(derived_dir), Path(sequence_dir)
    sequence = load_sequence(source/"sequence.json")
    n = _sequence_count(sequence)
    if len(objects) != n:
        raise ValueError("objects frame count differs from sequence")
    origin, size = canvas_geometry(objects, cfg) if canvas is None else canvas
    frames = []
    for k in range(n):
        name = f"labels_{k:03d}.npz"
        with np.load(derived/name, allow_pickle=False) as dense:
            frames.append(dict(name=name, sha256=_sha256(derived/name), H=dense["H"].tolist(),
                               origin=dense["origin"].tolist(), size=dense["size"].tolist(),
                               stride=int(dense["stride"])))
    manifest = dict(algorithm_versions=algorithm_versions(), objects_schema_version=OBJECTS_SCHEMA_VERSION,
                    config=cfg, sequence=dict(name="sequence.json", sha256=_sha256(source/"sequence.json")),
                    objects=[dict(name=f"objects_{k:03d}.npz", sha256=_sha256(source/f"objects_{k:03d}.npz"))
                             for k in range(n)], dense=frames,
                    origin=cfg.get("origin", origin), size=cfg.get("size", size), stride=cfg.get("stride", 1))
    _check_manifest_structure(jsonable(manifest))
    temporary = derived/"labels_manifest.json.tmp"
    write_json(temporary, manifest)
    os.replace(temporary, derived/"labels_manifest.json")


def _check_manifest_structure(manifest):
    def require(mapping, key, kind):
        if not isinstance(mapping, dict) or key not in mapping or type(mapping[key]) is not kind:
            raise ValueError(f"manifest missing or invalid {key}")
        return mapping[key]
    require(manifest, "algorithm_versions", dict)
    require(manifest, "objects_schema_version", int)
    cfg = require(manifest, "config", dict)
    from starsim.config import validate_label_cfg
    validate_label_cfg(cfg)
    def geometry(value):
        stride = require(value, "stride", int)
        if stride < 1:
            raise ValueError("manifest invalid stride")
        for key in ("origin", "size"):
            values = require(value, key, list)
            if len(values) != 2 or any(type(v) is not int for v in values) or (key == "size" and min(values) < 1):
                raise ValueError(f"manifest invalid {key}")
    geometry(manifest)
    for key in ("stride", "origin", "size"):
        if key in cfg and cfg[key] != manifest[key]:
            raise ValueError(f"manifest config/top-level mismatch: {key}")
    groups = [[require(manifest, "sequence", dict)], require(manifest, "objects", list),
              require(manifest, "dense", list)]
    for entries in groups:
        for entry in entries:
            require(entry, "name", str)
            require(entry, "sha256", str)
    for entry in groups[2]:
        geometry(entry)
        H = require(entry, "H", list)
        if np.asarray(H).shape != (3, 3) or not np.all(np.isfinite(np.asarray(H, dtype=float))):
            raise ValueError("manifest invalid H")
        for key in ("stride", "origin", "size"):
            if entry[key] != manifest[key]:
                raise ValueError(f"manifest frame/top-level mismatch: {key}")


def validate_labels_manifest(derived_dir, sequence_dir):
    derived, source = Path(derived_dir), Path(sequence_dir)
    manifest = json.loads((derived/"labels_manifest.json").read_text(encoding="utf-8"))
    _check_manifest_structure(manifest)
    sequence = load_sequence(source/"sequence.json")
    n = _sequence_count(sequence)
    validate_versions(manifest["algorithm_versions"])
    if type(manifest["objects_schema_version"]) is not int or manifest["objects_schema_version"] != OBJECTS_SCHEMA_VERSION:
        raise ValueError("unsupported objects schema_version in manifest")
    groups = [(source, [manifest["sequence"]], ["sequence.json"]),
              (source, manifest["objects"], [f"objects_{k:03d}.npz" for k in range(n)]),
              (derived, manifest["dense"], [f"labels_{k:03d}.npz" for k in range(n)])]
    for directory, entries, names in groups:
        if [entry["name"] for entry in entries] != names:
            raise ValueError("manifest filenames/count/order are not canonical")
        for entry in entries:
            path = directory/entry["name"]
            if not path.is_file() or _sha256(path) != entry["sha256"]:
                raise ValueError(f"manifest hash mismatch or missing file: {entry['name']}")
    for k, entry in enumerate(manifest["dense"]):
        frame = load_objects(source/f"objects_{k:03d}.npz")
        expected_H = frame.H_to_ref if manifest["config"]["label_frame"] == "reference" else np.eye(3)
        with np.load(derived/entry["name"], allow_pickle=False) as dense:
            for key in ("H", "origin", "size", "stride"):
                if key not in dense:
                    raise ValueError(f"dense missing {key}")
                if not np.array_equal(dense[key], entry[key]):
                    raise ValueError(f"manifest dense geometry mismatch: {key}")
            if not np.array_equal(dense["H"], expected_H):
                raise ValueError("dense H differs from actual frame transform")
    return manifest


def load_dense(derived_dir, sequence_dir, k):
    manifest = validate_labels_manifest(derived_dir, sequence_dir)
    if type(k) is not int or not 0 <= k < len(manifest["dense"]):
        raise ValueError("dense frame index out of range")
    with np.load(Path(derived_dir)/f"labels_{k:03d}.npz", allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def write_sequence(out_dir, plan, rngs, sensor, psf, camera, motion, photometry,
                   catalog, config, effective, *, derived_camera, split="train"):
    """Stream exposures to FITS, then finalize cross-frame sparse identities.

    Only source-level records survive the frame loop. Dense supervision is
    derived after the identity SNR windows have been computed.
    """
    from starsim.labels import assign_status, build_frame_labels
    from starsim.metadata import frame_metadata, sequence_metadata
    from starsim.render_labels import _render_frame, canvas_geometry
    from starsim.sequence import iter_frames

    generation_start = time.perf_counter()
    preflight_sequence(plan, sensor, motion, effective, camera, derived_camera)
    from starsim.metadata import git_provenance
    provenance = git_provenance()
    out = prepare_output_directory(out_dir)
    type_registry = {}
    objects, metas = [], []
    timings = dict(build_frame_labels=[], assign_status=0., dense=[], fits=[], objects_npz=[], dense_npz=[], frame_json=[])
    catalog_index = {(int(a), int(b)): i for i, (a, b) in
                     enumerate(zip(catalog.tyc1, catalog.tyc2))}
    for frame in iter_frames(plan, rngs, sensor, psf):
        start = time.perf_counter()
        labels = build_frame_labels(frame, sensor, psf, camera, motion, plan.population, catalog,
                                    tiers=effective["targets"]["tiers"], catalog_index=catalog_index)
        timings['build_frame_labels'].append(time.perf_counter()-start)
        meta = frame_metadata(frame, labels, camera, motion, sensor, photometry)
        start = time.perf_counter()
        save_fits(out/f"frame_{frame.frame_index:03d}.fits.fz", frame.exposure.adu, meta["WCS"],
                  frame_index=frame.frame_index, exposure_s=sensor.exposure_s)
        timings['fits'].append(time.perf_counter()-start)
        objects.append(labels)
        metas.append(meta)
        del frame
    start = time.perf_counter()
    assign_status(objects, effective["labels"])
    timings['assign_status'] = time.perf_counter()-start
    union = canvas_geometry(objects, dict(label_frame='reference', canvas='union'))
    cfg = effective['labels']
    canvas = (union if cfg['label_frame'] == 'reference' and cfg.get('canvas', 'union') == 'union'
              else ((0, 0), (camera.width, camera.height)))
    collisions, labeled, cropped = 0, 0, 0
    for labels, meta in zip(objects, metas):
        k = labels.frame_index
        H = labels.H_to_ref if cfg["label_frame"] == "reference" else np.eye(3)
        start = time.perf_counter()
        dense = _render_frame(objects, k, H, cfg, canvas=canvas)
        dense_time = time.perf_counter()-start
        for cls, name in enumerate(("stars", "points", "streaks")):
            cs = [c for c in labels.candidates if c["class_id"] == cls]
            meta["statistics"][name].update(labeled=sum(c["status_default"] == "labeled" for c in cs),
                                           ignore=sum(c["status_default"] == "ignore" for c in cs))
        meta["statistics"]["ignore_fraction"] = dense["ignore"].mean(axis=(1, 2))
        meta["statistics"]["collision_count"] = len(dense["collision"])
        start = time.perf_counter()
        save_objects(out/f"objects_{k:03d}.npz", labels, type_registry)
        timings['objects_npz'].append(time.perf_counter()-start)
        start = time.perf_counter()
        write_json(out/f"meta_{k:03d}.json", meta)
        timings['frame_json'].append(time.perf_counter()-start)
        start = time.perf_counter()
        if cfg["save_dense"]:
            save_dense(out/f"labels_{k:03d}.npz", dense)
        timings['dense_npz'].append(time.perf_counter()-start)
        del dense
        start = time.perf_counter()
        stride4 = _render_frame(objects, k, H, dict(cfg, stride=4), canvas=canvas)
        timings['dense'].append(dense_time+time.perf_counter()-start)
        collisions += len(stride4["collision"])
        labeled += len(stride4["collision"])+len(stride4["retained"])+len(stride4["cropped"])
        cropped += len(stride4["cropped"])
        del stride4
    meta = sequence_metadata(config, effective, rngs, camera, motion, sensor, psf,
                             photometry, plan.population, catalog, objects, derived_camera=derived_camera,
                             provenance=provenance, split=split, canvas=union)
    meta["stride4_collisions"] = dict(count=collisions, labeled=labeled, cropped=cropped,
                                       rate=collisions/labeled if labeled else 0.0)
    write_json(out/"sequence.json", meta)
    if effective["labels"]["save_dense"]:
        write_labels_manifest(out, out, effective["labels"], objects, canvas=canvas)
    total = time.perf_counter()-generation_start
    accounted = sum(sum(v) if isinstance(v, list) else v for v in timings.values())
    timings['other'] = total-accounted
    write_json(out/'timing.json', dict(seconds=timings, total_seconds=total,
        clock='perf_counter', phase='write_sequence',
        ranges=dict(other='phase total minus named operations; includes rendering exposures and manifests',
                    fits='FITS writing call including compression',
                    dense='configured stride plus stride 4; excludes writes'),
        excludes='timing.json serialization itself'))
    return objects
