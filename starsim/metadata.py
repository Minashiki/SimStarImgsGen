"""JSON provenance and pinhole geometry for independently verifiable files."""

from dataclasses import asdict, is_dataclass
import subprocess
from pathlib import Path

import numpy as np

from starsim.camera import wcs_keywords
from starsim.render_labels import canvas_geometry, footprint
from starsim.runtime import runtime_environment


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def validate_versions(recorded):
    from starsim import algorithm_versions

    current = algorithm_versions()
    if not isinstance(recorded, dict):
        raise ValueError("algorithm_versions must be a dict")
    missing = sorted(repr(k) for k in current.keys()-recorded.keys())
    extra = sorted(repr(k) for k in recorded.keys()-current.keys())
    mismatched = sorted(repr(k) for k in current.keys() & recorded.keys()
                        if type(recorded[k]) is not int or recorded[k] != current[k])
    if missing or extra or mismatched:
        raise ValueError(f"algorithm_versions missing={missing}, extra={extra}, mismatched={mismatched}")


def git_provenance():
    """Capture package repository provenance before generation starts."""
    import starsim
    try:
        package = Path(starsim.__file__).resolve().parent
        def git(args, cwd):
            return subprocess.run(["git", "-c", f"safe.directory={package.parent.as_posix()}", *args],
                                  cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        root = git(["rev-parse", "--show-toplevel"], package)
        commit = git(["rev-parse", "HEAD"], root)
        dirty = bool(git(["status", "--porcelain", "--untracked-files=all", "--",
                          "starsim", "scripts", "configs", "tests"], root))
        return dict(git_commit=commit, git_dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        return dict(git_commit=None, git_dirty=None)


def sequence_metadata(config, effective, rngs, camera, motion, sensor, psf,
                      photometry, population, catalog, objects_seq, *, derived_camera, provenance, split="train", canvas=None):
    from starsim import algorithm_versions

    origin, size = (canvas_geometry(objects_seq, {"label_frame": "reference", "canvas": "union"})
                    if canvas is None else canvas)
    translation = np.array([[1, 0, -origin[0]], [0, 1, -origin[1]], [0, 0, 1.]])
    sensor_fields = ("hot_row", "hot_col", "hot_kind", "hot_level_a", "hot_level_b", "hot_state",
                     "hot_p_flip", "dead_row", "dead_col", "row_offset_e", "col_offset_e")
    return jsonable(dict(schema_version=2, **provenance,
        environment=runtime_environment(), algorithm_versions=algorithm_versions(),
        rng=rngs.metadata(), split=split, config_unsampled=config.generator, config_effective=effective,
        camera_config=config.camera, derived_camera=derived_camera,
        camera_model="pinhole", camera=camera, motion=motion, psf=psf, photometry=photometry,
        exposure_s_nominal=sensor.exposure_s,
        exposure_s_nominal_applies_to=["static_stars", "sky", "dark_current"],
        target_morphology_model="per_target_duty_nonphysical", catalog=catalog.meta,
        reference_index=motion.reference_index, population=population,
        sensor={name: getattr(sensor, name) for name in sensor_fields},
        noise={name: getattr(sensor, name) for name in ("read_e", "gain_e_per_adu", "bias_adu",
               "adc_bits", "full_well_e", "blooming", "sky_e", "cosmic_rate")},
        canvas=dict(origin=origin, size=size, translation=(-origin[0], -origin[1]),
                    H_to_canvas=[translation @ H for H in motion.H_to_ref],
                    footprints=[footprint(f, f.H_to_ref) for f in objects_seq])))


def frame_metadata(frame, labels, camera, motion, sensor, photometry, dense=None):
    k = frame.frame_index
    stats = {}
    for cls, name in enumerate(("stars", "points", "streaks")):
        cs = [c for c in labels.candidates if c["class_id"] == cls]
        stats[name] = dict(rendered=sum(s["class_id"] == cls for s in labels.sources),
                           candidates=len(cs),
                           blended=sum(c["blended"] for c in cs))
        if cls == 0:
            mags = [c["vmag"] for c in cs]
            hist, bins = np.histogram(mags, bins=np.arange(-2, 19))
            stats[name].update(magnitude_min=min(mags) if mags else None,
                               magnitude_max=max(mags) if mags else None,
                               magnitude_median=float(np.median(mags)) if mags else None,
                               magnitude_histogram=hist, magnitude_bins=bins)
        else:
            for field in ("tier_design", "tier_measured"):
                stats[name][field] = {t: sum(c[field] == t for c in cs)
                                     for t in sorted({c[field] for c in cs})}
            for field in ("entering", "exiting", "truncated", "never_full"):
                stats[name][field] = sum(c[field] for c in cs)
    stats.update(outside_with_flux=sum(not s["in_image"] and s["has_flux"] for s in labels.sources),
                 cosmic_rays=len(frame.exposure.cosmic_rays),
                 hot_kinds={str(i): int(np.count_nonzero(sensor.hot_kind == i)) for i in range(3)})
    if dense is not None:
        stats["ignore_fraction"] = dense["ignore"].mean(axis=(1, 2))
        stats["collision_count"] = len(dense["collision"])
    step = max(0, k-1)
    return jsonable(dict(schema_version=1, environment=runtime_environment(), frame_index=k,
        t_s=k*motion.frame_interval_s, frame_interval_s=motion.frame_interval_s,
        quaternion=motion.quaternion[k], matrix=motion.matrix[k],
        pointing_roll_rad=[motion.ra_rad[k], motion.dec_rad[k], motion.roll_rad[k]],
        boresight_rad=[motion.boresight_ra_rad[k], motion.boresight_dec_rad[k]],
        euler313_rad=[motion.euler_phi_rad[k], motion.euler_theta_rad[k], motion.euler_psi_rad[k]],
        north_angle_rad=motion.north_angle_rad[k], WCS=wcs_keywords(camera, motion.matrix[k]),
        H_to_ref=motion.H_to_ref[k], displacement_px=motion.center_shift_px[k-1] if k else [0., 0.],
        speed_px_per_frame=motion.speed_px_per_frame[step],
        speed_deg_per_s=motion.speed_deg_per_s[step], direction_rad=motion.direction_rad[step],
        omega_cam_rad=motion.omega_cam_rad[step], accel_px_per_frame2=motion.accel_px_per_frame2,
        statistics=stats, cosmic_rays=frame.exposure.cosmic_rays,
        background=dict(mean_e=float(sensor.background_e.mean()), sigma_ref=sensor.reference_sigma(),
                        mag_limit=photometry.mag_limit)))
