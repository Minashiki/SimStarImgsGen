"""Independent disk-only pinhole/Tycho/WCS/partition audit. Never imports starsim.

Every failure raises :class:`VerificationError`. Inputs and errors are
checked for finiteness before any tolerance comparison, so NaN can neither
pass a ``>=`` test nor vanish from a ``max``. An empty star field verifies;
acceptance tests must require non-zero counts themselves.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial.transform import Rotation

REASONS = {"", "no_flux", "pixel_dominated", "clipped_by_other", "center_outside", "low_snr", "never_full"}
POSITION_TOL = 1e-3   # px, catalog reprojection (plan M4)
H_TOL = 1e-6          # px, recorded vs independently rebuilt homography


class VerificationError(Exception):
    pass


def fail(message):
    raise VerificationError(message)


def finite(name, value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        fail(f"non-finite {name}")
    return array


def within(name, error, tolerance):
    error = float(error)
    if not math.isfinite(error) or error >= tolerance:
        fail(f"{name} mismatch: {error!r}")
    return error


def environment():
    import llvmlite
    import llvmlite.binding
    import numba
    import scipy
    return dict(python=".".join(map(str, sys.version_info[:3])), numpy=np.__version__,
                numba=numba.__version__, llvmlite=llvmlite.__version__, scipy=scipy.__version__,
                cpu=llvmlite.binding.get_host_cpu_name())


def records(table):
    return [{key: json.loads(str(row[key])) if table.dtype[key].kind == "U" else row[key].item()
             for key in table.dtype.names} for row in table]


def check_values(value):
    if value is None:
        fail("None is not allowed in objects")
    if isinstance(value, dict):
        for child in value.values():
            check_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            check_values(child)
    elif type(value) in (int, float) and not math.isfinite(value):
        fail("non-finite numeric value in objects")


OBJECTS_SCHEMA_VERSION = 2
CANDIDATE_COMMON = frozenset("label_group_id class_id members components blended identity stable_key representative_member x y x_ref y_ref flux_e sigma_psf in_image has_flux dominant_source saturated in_foreign_sat in_defect_sat group_peak_row group_peak_col s_group sigma_local snr snr_eff identity_frame_count status_default ignore_reason_default ra dec".split())
CANDIDATE_STAR = frozenset("vmag member_identity_counts".split())
CANDIDATE_TARGET = frozenset("track_id mag_eq tier_design tier_measured duty_eff t_exp_eff x1 y1 x2 y2 x1_ref y1_ref x2_ref y2_ref L_true L_native angle_rad angle_ref x_vis y_vis vis_x1 vis_y1 vis_x2 vis_y2 L_vis vis_frac vis_x1_ref vis_y1_ref vis_x2_ref vis_y2_ref x_vis_ref y_vis_ref L_vis_ref vx_ref vy_ref truncated entering exiting never_full".split())
EMPTY_TABLE_DTYPE = np.dtype([("id", "<U1")])
TABLE_NAMES = ("sources_stars", "sources_points", "sources_streaks", "stars", "points", "streaks")


def _check_objects_arrays(data):
    if "schema_version" not in data:
        raise VerificationError("unsupported legacy objects format (不支持旧格式): missing schema_version")
    required = set(TABLE_NAMES) | {"schema_version", "frame_index", "size", "H_to_ref"}
    if not required <= set(data):
        raise VerificationError("missing objects tables or arrays: " + str(sorted(required-set(data))))
    schema, index, size, H = (data[key] for key in ("schema_version", "frame_index", "size", "H_to_ref"))
    if schema.shape != () or schema.dtype.kind not in "iu" or schema != OBJECTS_SCHEMA_VERSION:
        raise VerificationError("unsupported objects schema_version")
    if index.shape != () or index.dtype.kind not in "iu" or index < 0:
        raise VerificationError("frame_index must be a nonnegative scalar integer")
    if size.shape != (2,) or size.dtype.kind not in "iu" or np.any(size <= 0):
        raise VerificationError("size must contain two positive integers")
    if H.dtype != np.dtype("float64") or H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise VerificationError("H_to_ref must be a finite float64 3x3 array")
    for j, name in enumerate(TABLE_NAMES):
        table = data[name]
        if table.ndim != 1 or table.dtype.names is None:
            raise VerificationError(f"{name}: expected one-dimensional structured table")
        if not len(table):
            if table.dtype != EMPTY_TABLE_DTYPE:
                raise VerificationError(f"{name}: invalid empty table dtype")
            continue
        if "class_id" not in table.dtype.names or table.dtype["class_id"].kind not in "iu" or np.any(table["class_id"] != j % 3):
            raise VerificationError(f"{name}: invalid class_id")
        if j >= 3 and set(table.dtype.names) != CANDIDATE_COMMON | (CANDIDATE_STAR if j == 3 else CANDIDATE_TARGET):
            raise VerificationError(f"{name}: candidate field whitelist mismatch")


def load_catalog(path):
    with np.load(path, allow_pickle=False) as cat:
        meta = json.loads(str(cat["meta"].item()))
        index = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(cat["tyc1"], cat["tyc2"]))}
        ra_deg, dec_deg = np.array(cat["ra_deg"]), np.array(cat["dec_deg"])
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    xyz = np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))
    return dict(meta=meta, index=index, ra_deg=ra_deg, dec_deg=dec_deg, xyz=xyz)


def apply(H, points):
    mapped = np.column_stack((points, np.ones(len(points)))) @ np.asarray(H).T
    if np.any(mapped[:, 2] == 0) or not np.all(np.isfinite(mapped)):
        fail("homography sent a point to infinity")
    return mapped[:, :2]/mapped[:, 2:]


def rebuilt_homography(K, R_from, R_to):
    """The project's normalization: exact identity, else scale H[2, 2] to 1."""
    if np.array_equal(R_from, R_to):
        return np.eye(3)
    mapped = K @ R_to @ R_from.T @ np.linalg.inv(K)
    pivot = float(mapped[2, 2])
    if not math.isfinite(pivot) or pivot == 0.0:
        fail(f"independent homography cannot be normalized, H[2, 2]={pivot!r}")
    return mapped/pivot


def check_homography(name, recorded, expected, probes):
    H = finite(name, recorded)
    if H.shape != (3, 3):
        fail(f"{name} is not 3x3")
    # Both sides use H[2, 2] = 1, so scale and sign agree before det is compared.
    if H[2, 2] != 1.0:
        fail(f"{name} is not normalized to H[2, 2] = 1")
    if np.sign(np.linalg.det(H)) != np.sign(np.linalg.det(expected)):
        fail(f"{name} determinant sign mismatch")
    return within(name, np.max(abs(apply(H, probes)-apply(expected, probes))), H_TOL)


def north_angle(R, ra, dec):
    """Angle from north to image up (−Y), east positive, in (−π, π]."""
    east = np.array([-math.sin(ra), math.cos(ra), 0.])
    north = np.array([-math.sin(dec)*math.cos(ra), -math.sin(dec)*math.sin(ra), math.cos(dec)])
    up = -np.asarray(R)[1]
    angle = math.atan2(float(up @ east), float(up @ north))
    return math.pi if angle <= -math.pi else angle


def check_partition(sources, candidates, width, height):
    """Both D56 layers on disk truth, for every class, not just labeled stars."""
    identities = [require(c, 'identity') for c in candidates]
    if len(identities) != len(set(identities)):
        fail("candidate identity must be unique within a frame")
    for i, s in enumerate(sources):
        if require(s, 'source_index') != i:
            fail(f"source_index {require(s, 'source_index')} at row {i}")
        if not require(s, 'has_flux'):
            inside = -0.5 <= finite("source x", require(s, 'x')) < width-0.5 and -0.5 <= finite("source y", require(s, 'y')) < height-0.5
            if inside:
                fail(f"source {i} has no flux but its center is inside the image")
    ids = sorted(require(c, 'label_group_id') for c in candidates)
    if ids != list(range(len(candidates))):
        fail("label_group_id values are not 0..n-1")
    owner = {}
    for c in candidates:
        gid, members = require(c, 'label_group_id'), require(c, 'members')
        if not isinstance(members, list) or not members:
            fail(f"candidate {gid} has no members")
        for i in members:
            if type(i) is not int or not 0 <= i < len(sources):
                fail(f"candidate {gid} member {i!r} out of range")
            if i in owner:
                fail(f"source {i} belongs to candidates {owner[i]} and {gid}")
            owner[i] = gid
            if sources[i]["candidate_id"] != gid:
                fail(f"source {i} points to candidate {sources[i]['candidate_id']}, listed by {gid}")
            if sources[i]["class_id"] != require(c, 'class_id'):
                fail(f"candidate {gid} mixes classes")
        member_sources = [sources[i] for i in members]
        if require(c, 'class_id') == 0:
            if require(c, 'components') != [require(s, 'id') for s in member_sources]:
                fail(f"candidate {gid} components differ from member TYC ids")
            for s in member_sources:
                if require(s, 'id') != f"{require(s, 'tyc1')}-{require(s, 'tyc2')}":
                    fail(f"source {require(s, 'source_index')} id differs from its TYC numbers")
            pairs = sorted((require(s, 'tyc1'), require(s, 'tyc2')) for s in member_sources)
            if require(c, 'identity') != "star:" + ";".join(f"{a}-{b}" for a, b in pairs):
                fail(f"candidate {gid} identity differs from member TYC set")
        else:
            if len(members) != 1:
                fail(f"moving candidate {gid} must have exactly one member")
            track = member_sources[0]["track_id"]
            if (type(track) is not int or require(c, 'track_id') != track or require(c, 'components') != [str(track)]
                    or require(c, 'identity') != f"target:{track}"):
                fail(f"candidate {gid} track_id/identity differs from its source")
        reason = require(c, 'ignore_reason_default')
        if reason not in REASONS or require(c, 'status_default') != ("ignore" if reason else "labeled"):
            fail(f"candidate {gid} status {require(c, 'status_default')!r} / reason {reason!r}")
        no_flux = [not require(s, 'has_flux') for s in member_sources]
        # D56 rule 0 is one-way: no flux implies its own no_flux candidate.
        # A center outside the image does not imply no_flux (visible streaks).
        if any(no_flux) and (reason != "no_flux" or len(members) != 1):
            fail(f"candidate {gid} holds a no-flux source without being its own no_flux candidate")
        if reason == "no_flux" and not all(no_flux):
            fail(f"candidate {gid} is no_flux but a member has flux")
    if len(owner) != len(sources):
        missing = sorted(set(range(len(sources)))-owner.keys())
        fail(f"sources without a candidate: {missing[:10]}")


def require(mapping, key):
    if not isinstance(mapping, dict) or key not in mapping:
        fail(f"missing required key: {key}")
    return mapping[key]


def verify(directory, cache):
    try:
        return _verify(directory, cache)
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        fail(f"invalid disk structure: {exc}")


def _verify(directory, cache):
    directory = Path(directory)
    catalog = cache if isinstance(cache, dict) else load_catalog(cache)
    seq = json.loads((directory/"sequence.json").read_text(encoding="utf-8"))
    if type(require(seq, "schema_version")) is not int or require(seq, 'schema_version') != 2:
        fail("unsupported sequence schema_version")
    versions = seq.get("algorithm_versions")
    if not isinstance(versions, dict) or any(
            not isinstance(key, str) or type(value) is not int or value < 0
            for key, value in versions.items()):
        fail("invalid algorithm_versions structure")
    if require(seq, 'camera_model') != "pinhole":
        fail("unsupported camera model")
    if require(catalog, 'meta') != require(seq, 'catalog'):
        fail("catalog provenance mismatch")
    camera = require(seq, 'camera')
    width, height = require(camera, 'width'), require(camera, 'height')
    f, cx, cy = (float(finite(name, require(camera, name))) for name in ("f_over_p", "cx", "cy"))
    if cx != (width-1)/2 or cy != (height-1)/2:
        fail("unsupported principal point")
    K = np.array([[f, 0., cx], [0., f, cy], [0., 0., 1.]])
    n_frames = len(require(require(seq, 'motion'), 'matrix'))
    if n_frames < 2 or require(require(require(seq, 'config_effective'), 'dataset'), 'n_frames') != n_frames:
        fail("inconsistent sequence frame count")
    ref = require(seq, 'reference_index')
    if type(ref) is not int or not 0 <= ref < n_frames:
        fail("reference_index out of range")
    corners = np.array([[-.5, -.5], [width-.5, -.5], [width-.5, height-.5], [-.5, height-.5]])
    probes = np.vstack((corners, [[cx, cy]]))
    metas = [json.loads((directory/f"meta_{k:03d}.json").read_text(encoding="utf-8")) for k in range(n_frames)]
    matrices = [finite(f"frame {k} matrix", require(m, "matrix")) for k, m in enumerate(metas)]
    expected_H = [rebuilt_homography(K, R, matrices[ref]) for R in matrices]
    worst = dict(attitude=0., wcs=0., homography=0., star_native=0., star_reference=0., candidate_reference=0.)
    counts = dict(checked=0, star_sources=0, candidates=0, labeled=[0, 0, 0], blended=0, no_flux=0)

    canvas = require(seq, 'canvas')
    origin, size = require(canvas, 'origin'), require(canvas, 'size')
    if not all(type(v) is int for v in origin+size) or min(size) <= 0:
        fail("canvas origin/size must be integers with positive size")
    if list(require(canvas, 'translation')) != [-origin[0], -origin[1]]:
        fail("canvas translation differs from -origin")
    T = np.array([[1., 0., -origin[0]], [0., 1., -origin[1]], [0., 0., 1.]])
    footprints = [apply(H, corners) for H in expected_H]
    union = np.vstack(footprints)
    # Containment and tightness of D26/4.12, tolerant to floor/ceil at exact edges.
    low = union.min(axis=0)-(np.array(origin)-.5)
    high = (np.array(origin)+np.array(size)-.5)-union.max(axis=0)
    if np.any(low < -H_TOL) or np.any(high < -H_TOL) or np.any(low >= 1+H_TOL) or np.any(high >= 1+H_TOL):
        fail("canvas origin/size does not tightly cover the rebuilt footprints")

    for k, meta in enumerate(metas):
        R = matrices[k]
        if require(meta, 'frame_index') != k:
            fail(f"meta_{k:03d} frame_index")
        for name in ("matrix", "quaternion", "H_to_ref"):
            if not np.array_equal(finite(f"sequence motion {name}", require(seq, 'motion')[name][k]),
                                  finite(f"frame {k} {name}", require(meta, name))):
                fail(f"sequence motion {name}[{k}] differs from meta_{k:03d}")
        q = finite("quaternion", require(meta, 'quaternion'))
        err = float(np.max(abs(Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()-R)))
        worst["attitude"] = max(worst["attitude"], within("quaternion/matrix", err, 1e-12))
        # Rebuild the local east/north camera basis directly from pointing+roll.
        a, d, roll = finite("pointing_roll_rad", require(meta, 'pointing_roll_rad'))
        east = np.array([-math.sin(a), math.cos(a), 0.])
        north = np.array([-math.sin(d)*math.cos(a), -math.sin(d)*math.sin(a), math.cos(d)])
        forward = np.array([math.cos(d)*math.cos(a), math.cos(d)*math.sin(a), math.sin(d)])
        # This project's roll=0 has camera x=east, camera y=north.
        pointing = np.array([math.cos(roll)*east+math.sin(roll)*north,
                             -math.sin(roll)*east+math.cos(roll)*north, forward])
        within("pointing/roll", np.max(abs(pointing-R)), 1e-10)
        recorded_north = float(finite("north_angle_rad", require(meta, 'north_angle_rad')))
        if not -math.pi < recorded_north <= math.pi:
            fail("north position angle outside (-pi, pi]")
        # A full-period difference: an error of exactly pi must fail.
        within("north position angle", abs(math.remainder(recorded_north-north_angle(R, a, d), 2*math.pi)), 1e-10)
        ba, bd = finite("boresight_rad", require(meta, 'boresight_rad'))
        bore = np.array([math.cos(bd)*math.cos(ba), math.cos(bd)*math.sin(ba), math.sin(bd)])
        within("boresight", np.max(abs(bore-R[2])), 1e-12)
        phi, theta, psi = finite("euler313_rad", require(meta, 'euler313_rad'))
        def r3(t):
            return np.array([[math.cos(t), math.sin(t), 0.], [-math.sin(t), math.cos(t), 0.], [0., 0., 1.]])
        r1 = np.array([[1., 0., 0.], [0., math.cos(theta), math.sin(theta)],
                       [0., -math.sin(theta), math.cos(theta)]])
        within("Euler", np.max(abs(r3(phi) @ r1 @ r3(psi)-R)), 1e-10)

        with np.load(directory/f"objects_{k:03d}.npz", allow_pickle=False) as table:
            _check_objects_arrays(table)
            sources = sum((records(table[name]) for name in
                           ("sources_stars", "sources_points", "sources_streaks")), [])
            sources.sort(key=lambda row: require(row, 'source_index'))
            if [require(row, 'source_index') for row in sources] != list(range(len(sources))):
                fail("source_index must be continuous and unique")
            candidates = sum((records(table[name]) for name in ("stars", "points", "streaks")), [])
            npz_H = np.array(table["H_to_ref"])
            finite("objects frame_index", table["frame_index"])
            finite("objects size", table["size"])
            if table["frame_index"] != k or table["size"].tolist() != [width, height]:
                fail(f"objects_{k:03d} frame_index/size")
        check_values([sources, candidates, npz_H.tolist()])
        for name, H in ((f"meta_{k:03d} H_to_ref", require(meta, 'H_to_ref')), (f"objects_{k:03d} H_to_ref", npz_H)):
            worst["homography"] = max(worst["homography"], check_homography(name, H, expected_H[k], probes))
        worst["homography"] = max(worst["homography"], check_homography(
            f"canvas H_to_canvas[{k}]", require(canvas, 'H_to_canvas')[k], T @ expected_H[k], probes))
        within(f"canvas footprint[{k}]", np.max(abs(finite("footprint", require(canvas, 'footprints')[k])-footprints[k])), H_TOL)

        for key, value in require(meta, 'WCS').items():
            if type(value) in (int, float):
                finite(f"WCS {key}", value)
        wcs = WCS(fits.Header(require(meta, 'WCS')))
        with fits.open(directory/f"frame_{k:03d}.fits.fz") as hdus:
            fits_wcs = WCS(hdus[1].header)
            if hdus[1].data.shape != (height, width):
                fail("FITS shape")
        # Check every WCS even in a star-free field, using native image corners.
        pixels = np.array([[0., 0.], [width-1., 0.], [width-1., height-1.], [0., height-1.], [cx, cy]])
        vectors = np.column_stack(((pixels[:, 0]-cx)/f, (pixels[:, 1]-cy)/f, np.ones(len(pixels)))) @ R
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        world = np.column_stack((np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 360,
                                 np.degrees(np.arcsin(vectors[:, 2]))))
        for actual in (wcs, fits_wcs):
            worst["wcs"] = max(worst["wcs"], within("WCS field", np.max(abs(actual.all_world2pix(world, 0)-pixels)), 1e-3))

        check_partition(sources, candidates, width, height)
        counts["candidates"] += len(candidates)
        counts["no_flux"] += sum(require(c, 'ignore_reason_default') == "no_flux" for c in candidates)
        counts["blended"] += sum(len(require(c, 'members')) > 1 for c in candidates)

        # Every star source, labeled or not: catalog -> native -> reference.
        projected = {}
        stars = [s for s in sources if require(s, 'class_id') == 0]
        for s in stars:
            key = (require(s, 'tyc1'), require(s, 'tyc2'))
            if key not in require(catalog, 'index'):
                fail(f"TYC {key} not in catalog")
            vec = R @ require(catalog, 'xyz')[require(catalog, 'index')[key]]
            if vec[2] <= 0:
                fail(f"TYC {key} behind the camera")
            projected[require(s, 'source_index')] = np.array([cx, cy])+f*vec[:2]/vec[2]
        if stars:
            native = np.array([projected[require(s, 'source_index')] for s in stars])
            recorded = finite("star source x/y", [[require(s, 'x'), require(s, 'y')] for s in stars])
            worst["star_native"] = max(worst["star_native"], within(
                "catalog reprojection", np.max(abs(native-recorded)), POSITION_TOL))
            recorded_ref = finite("star source x_ref/y_ref", [[require(s, 'x_ref'), require(s, 'y_ref')] for s in stars])
            worst["star_reference"] = max(worst["star_reference"], within(
                "reference reprojection", np.max(abs(apply(expected_H[k], native)-recorded_ref)), POSITION_TOL))
            counts["star_sources"] += len(stars)
            labeled_members = [i for c in candidates if require(c, 'class_id') == 0 and require(c, 'status_default') == "labeled"
                               for i in require(c, 'members')]
            if labeled_members:
                world = np.array([[require(catalog, 'ra_deg')[require(catalog, 'index')[(sources[i]["tyc1"], sources[i]["tyc2"])]],
                                   require(catalog, 'dec_deg')[require(catalog, 'index')[(sources[i]["tyc1"], sources[i]["tyc2"])]]]
                                  for i in labeled_members])
                expected = np.array([projected[i] for i in labeled_members])
                for actual in (wcs, fits_wcs):
                    worst["wcs"] = max(worst["wcs"], within(
                        "WCS reprojection", np.max(abs(actual.all_world2pix(world, 0)-expected)), 1e-3))

        for c in candidates:
            xy = finite(f"candidate {require(c, 'label_group_id')} x/y", [require(c, 'x'), require(c, 'y')])
            xy_ref = finite(f"candidate {require(c, 'label_group_id')} x_ref/y_ref", [require(c, 'x_ref'), require(c, 'y_ref')])
            if require(c, 'class_id') == 0:
                flux = finite("member flux_e", [sources[i]["flux_e"] for i in require(c, 'members')])
                if np.any(flux <= 0):
                    fail(f"candidate {require(c, 'label_group_id')} has non-positive member flux")
                # Blended stars use member flux weights (plan M4).
                centroid = (flux[:, None]*np.array([projected[i] for i in require(c, 'members')])).sum(axis=0)/flux.sum()
                worst["star_native"] = max(worst["star_native"], within(
                    "catalog reprojection", np.max(abs(centroid-xy)), POSITION_TOL))
                xy = centroid
                if require(c, 'status_default') == "labeled":
                    counts["checked"] += 1
            if require(c, 'status_default') == "labeled":
                counts["labeled"][require(c, 'class_id')] += 1
            # Reference coordinates come from the independent native center.
            worst["candidate_reference"] = max(worst["candidate_reference"], within(
                "candidate reference coordinate", np.max(abs(apply(expected_H[k], xy[None])[0]-xy_ref)),
                POSITION_TOL if require(c, 'class_id') == 0 else H_TOL))
    return dict(environment=environment(), algorithm_versions=require(seq, 'algorithm_versions'), n_frames=n_frames,
                max_error_px=worst["star_native"], max_reference_error_px=worst["star_reference"],
                max_candidate_reference_error_px=worst["candidate_reference"],
                max_homography_error_px=worst["homography"], max_wcs_error_px=worst["wcs"],
                max_attitude_error=worst["attitude"], **counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.directory, args.catalog)
    report["modules_clean"] = not any(n == "starsim" or n.startswith("starsim.") for n in sys.modules)
    if not report["modules_clean"]:
        raise VerificationError("starsim was imported during verification")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
