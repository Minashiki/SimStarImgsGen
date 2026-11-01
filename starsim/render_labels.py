"""Derived dense supervision; sparse candidates remain the only truth."""

import copy
import math

import numpy as np

from starsim.labels import assign_status


def _corner_sign(H, width, height):
    corners = np.array([[-.5, -.5, 1], [width-.5, -.5, 1],
                        [width-.5, height-.5, 1], [-.5, height-.5, 1]])
    with np.errstate(over="ignore", invalid="ignore"):
        denominator = corners @ H[2]
    if not np.all(np.isfinite(denominator)) or not (
            np.all(denominator > 0) or np.all(denominator < 0)):
        raise ValueError("homography footprint denominators must have the same sign")
    return np.sign(denominator[0])


def _map(H, x, y, sign):
    x, y = np.broadcast_arrays(x, y)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mapped = np.stack((x, y, np.ones_like(x)), axis=-1) @ H.T
        result = mapped[..., :2] / mapped[..., 2:]
    if (not np.all(np.isfinite(mapped)) or np.any(mapped[..., 2] == 0)
            or not np.all(np.isfinite(result))):
        raise ValueError("homography mapping has non-finite numerator, denominator or quotient")
    if np.any(np.sign(mapped[..., 2]) != sign):
        raise ValueError("homography mapping denominator signs are inconsistent")
    return result


def validate_homography(H, width, height):
    """Validate the raw float64 representation, without rescaling it (D60)."""
    raw = np.asarray(H)
    if raw.dtype.kind not in "iuf" or raw.shape != (3, 3):
        raise ValueError("homography must be a numeric 3x3 array")
    H = raw.astype(np.float64)
    if not np.all(np.isfinite(H)) or not np.any(H):
        raise ValueError("homography must be finite and nonzero")
    x = np.array([-.5, width-.5, width-.5, -.5])
    y = np.array([-.5, -.5, height-.5, height-.5])
    polygon = _map(H, x, y, _corner_sign(H, width, height))
    with np.errstate(over="ignore", invalid="ignore"):
        denominator = np.stack((x, y, np.ones(4)), axis=-1) @ H[2]
        edges = np.roll(polygon, -1, axis=0)-polygon
        following = np.roll(edges, -1, axis=0)
        cross = edges[:, 0]*following[:, 1]-edges[:, 1]*following[:, 0]
    if not (np.all(denominator > 0) or np.all(denominator < 0)):
        raise ValueError("homography footprint denominators must have the same sign")
    if not np.all(np.isfinite(cross)) or not (np.all(cross > 0) or np.all(cross < 0)):
        raise ValueError("homography footprint must be strictly convex")
    return H


def encode_coordinate(x, stride, origin=0):
    scaled = (x-origin+0.5)/stride-0.5
    index = math.floor(scaled+0.5)
    return index, scaled-index


def decode_coordinate(index, offset, stride, origin=0):
    return stride*(index+offset+0.5)-0.5+origin


def footprint(frame, H):
    H = validate_homography(H, frame.width, frame.height)
    return _map(H, np.array([-0.5, frame.width-0.5, frame.width-0.5, -0.5]),
                            np.array([-0.5, -0.5, frame.height-0.5, frame.height-0.5]),
                _corner_sign(H, frame.width, frame.height))


def canvas_geometry(objects_seq, cfg):
    if cfg.get("label_frame", "native") == "native" or cfg.get("canvas", "union") == "reference":
        return (0, 0), (objects_seq[0].width, objects_seq[0].height)
    corners = np.concatenate([footprint(f, f.H_to_ref) for f in objects_seq])
    low = np.floor(corners.min(axis=0)+0.5).astype(int)
    high = np.ceil(corners.max(axis=0)+0.5).astype(int)
    return tuple(map(int, low)), tuple(map(int, high-low))


def valid_mask(polygon, origin, size, stride):
    """Positive-area convex footprint/block intersection (separating axes).

    A stride block is exactly the union of its native pixel squares, so
    this is also the 'any native pixel valid' aggregation in D57.
    """
    width, height = size
    rows, cols = math.ceil(height/stride), math.ceil(width/stride)
    x0 = origin[0] + np.arange(cols)*stride-0.5
    y0 = origin[1] + np.arange(rows)*stride-0.5
    x1 = origin[0] + np.minimum((np.arange(cols)+1)*stride, width)-0.5
    y1 = origin[1] + np.minimum((np.arange(rows)+1)*stride, height)-0.5
    valid = np.ones((rows, cols), dtype=bool)
    edges = np.roll(polygon, -1, axis=0)-polygon
    axes = [np.array([1., 0.]), np.array([0., 1.])]
    axes.extend(np.array([-e[1], e[0]]) for e in edges)
    for ax, ay in axes:
        projected = polygon @ np.array([ax, ay])
        low = np.minimum(ax*x0, ax*x1)[None, :] + np.minimum(ay*y0, ay*y1)[:, None]
        high = np.maximum(ax*x0, ax*x1)[None, :] + np.maximum(ay*y0, ay*y1)[:, None]
        valid &= (high > projected.min()) & (low < projected.max())
    return valid.astype(np.uint8)


def _geometry(candidate, H, sign):
    x, y = _map(H, candidate["x"], candidate["y"], sign)
    length, angle = 0.0, 0.0
    if candidate["class_id"] == 2:
        a = _map(H, candidate["x1"], candidate["y1"], sign)
        b = _map(H, candidate["x2"], candidate["y2"], sign)
        length = float(np.linalg.norm(b-a))
        angle = math.atan2(b[1]-a[1], b[0]-a[0])
    if not math.isfinite(length) or not math.isfinite(angle):
        raise ValueError("non-finite projected geometry")
    return float(x), float(y), length, angle


def render_dense(objects_seq, frame_k, H_k_to_target, cfg):
    """Render one frame using the same homography as the aligned image.

    In reference mode cfg['origin'] / cfg['size'] select an aligned canvas.
    Native mode requires exact identity H and forbids both canvas keys.
    Collision outcomes are arrays alongside the supervision, never mutations
    of sparse status. Offsets remain float32 in memory and on disk.
    """
    objects = copy.deepcopy(objects_seq)
    assign_status(objects, cfg)
    return _render_frame(objects, frame_k, H_k_to_target, cfg)


def _render_frame(objects, frame_k, H_k_to_target, cfg, *, canvas=None):
    frame = objects[frame_k]
    reference = cfg.get("label_frame", "native") == "reference"
    if reference:
        H = validate_homography(H_k_to_target, frame.width, frame.height)
    else:
        if np.asarray(H_k_to_target).dtype.kind not in "iuf":
            raise ValueError("native homography must be numeric")
        H = np.asarray(H_k_to_target, dtype=np.float64)
        if not np.array_equal(H, np.eye(3)):
            raise ValueError("native mode requires the exact identity homography")
        if "origin" in cfg or "size" in cfg:
            raise ValueError("native mode does not accept origin or size")
    sign = _corner_sign(H, frame.width, frame.height)
    origin, size = canvas_geometry(objects, cfg) if canvas is None else canvas
    origin = tuple(cfg.get("origin", origin))
    size = tuple(cfg.get("size", size))
    stride = cfg.get("stride", 1)
    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be an integer >= 1")
    if (len(origin) != 2 or len(size) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in (*origin, *size))
            or min(size) < 1):
        raise ValueError("canvas requires integer origin and positive integer size")
    width, height = size
    hs, ws = math.ceil(height/stride), math.ceil(width/stride)
    shape = (3, hs, ws)
    result = dict(heatmap=np.zeros(shape, np.float32), offset=np.zeros((3, 2, hs, ws), np.float32),
                  reg_mask=np.zeros(shape, np.uint8), ignore=np.zeros(shape, np.uint8),
                  streak_geom=np.zeros(shape, np.float32), origin=np.array(origin),
                  stride=np.array(stride), size=np.array(size), H=H.copy())
    if reference:
        result["valid"] = valid_mask(footprint(frame, H), origin, size, stride)
    if cfg.get("motion", False):
        result["motion"] = np.zeros((3, 2, hs, ws), np.float32)
        result["motion_mask"] = np.zeros(shape, np.uint8)
    if cfg.get("streak_line", False):
        result["streak_line"] = np.zeros((hs, ws), np.float32)
    winners, collisions, outcomes, geometry, cropped = {}, [], [], {}, []
    for c in frame.candidates:
        if c["status_default"] != "labeled":
            continue
        x, y = _map(H, c["x"], c["y"], sign)
        u, v = x-origin[0], y-origin[1]
        gid, cls = c["label_group_id"], c["class_id"]
        if not (-.5 <= u < width-.5 and -.5 <= v < height-.5):
            if not reference:
                raise ValueError("native labeled center outside image")
            cropped.append(gid)
            continue
        ix, ox = encode_coordinate(u, stride, 0)
        iy, oy = encode_coordinate(v, stride, 0)
        if not (0 <= ix < ws and 0 <= iy < hs):
            raise ValueError("center cannot be encoded within dense grid")
        geometry[gid] = (ix, iy, ox, oy)
        if reference and not result["valid"][iy, ix]:
            raise ValueError("retained center outside valid footprint")
        cell = (cls, iy, ix)
        previous = winners.get(cell)
        if previous is None:
            winners[cell] = c
        elif (-c["flux_e"], c["stable_key"]) < (-previous["flux_e"], previous["stable_key"]):
            collisions.append((previous["label_group_id"], gid))
            winners[cell] = c
        else:
            collisions.append((gid, previous["label_group_id"]))
    # Redirect earlier losers when the provisional winner was itself replaced.
    replacements = dict(collisions)
    for loser, winner in collisions:
        while winner in replacements:
            winner = replacements[winner]
        outcomes.append((loser, winner))
    result["collision"] = np.asarray(sorted(outcomes), dtype=np.int64).reshape(-1, 2)
    result["retained"] = np.array(sorted(c["label_group_id"] for c in winners.values()), dtype=np.int64)
    labeled = [c["label_group_id"] for c in frame.candidates if c["status_default"] == "labeled"]
    result["cropped"] = np.array(sorted(cropped), dtype=np.int64)
    retained = set(result["retained"].tolist())
    losers = [a for a, b in outcomes]
    partition = result["retained"].tolist()+losers+cropped
    if (len(partition) != len(set(partition)) or sorted(partition) != sorted(labeled)
            or any(winner not in retained for loser, winner in outcomes)):
        raise ValueError("invalid retained/collision/cropped partition")
    # Local windows keep memory independent of the number of candidates.
    losing = {a for a, b in outcomes}
    prev_H = None
    for c in frame.candidates:
        gid, cls = c["label_group_id"], c["class_id"]
        ignored = c["status_default"] == "ignore" or gid in losing or gid in cropped
        if ignored:
            # Ignore the projected support, including the visible part of an
            # off-image streak. Bounding boxes are conservative supervision masks.
            boxes = [frame.sources[i]["box"] for i in c["members"] if frame.sources[i]["box"][0] >= 0]
            for r0, r1, c0, c1 in boxes:
                quad = _map(H, np.array([c0-.5, c1+.5, c1+.5, c0-.5]),
                                        np.array([r0-.5, r0-.5, r1+.5, r1+.5]), sign)
                low = np.floor((quad.min(axis=0)-origin+0.5)/stride)
                high = np.ceil((quad.max(axis=0)-origin+0.5)/stride)
                low = np.clip(low, (0, 0), (ws, hs)).astype(int)
                high = np.clip(high, (0, 0), (ws, hs)).astype(int)
                if np.all(low < high):
                    result["ignore"][cls, low[1]:high[1], low[0]:high[0]] = 1
            continue
        ix, iy, ox, oy = geometry[gid]
        _, _, length, angle = _geometry(c, H, sign)
        sigma = float(cfg.get("sigma_hm", 1.0))
        if cfg.get("sigma_mode", "fixed") == "adaptive":
            sigma = max(sigma, c.get("sigma_psf", 1.0)/stride)
        along = math.sqrt(sigma*sigma+(length/stride)**2/12)
        radius = math.ceil(4*along)
        a, b = max(0, iy-radius), min(hs, iy+radius+1)
        d, e = max(0, ix-radius), min(ws, ix+radius+1)
        yy, xx = np.mgrid[a:b, d:e]
        dx, dy = xx-ix, yy-iy
        u, v = dx*math.cos(angle)+dy*math.sin(angle), -dx*math.sin(angle)+dy*math.cos(angle)
        gaussian = np.exp(-.5*((u/along)**2+(v/sigma)**2))
        patch = result["heatmap"][cls, a:b, d:e]
        np.maximum(patch, gaussian, out=patch)
        result["offset"][cls, :, iy, ix] = (ox, oy)
        result["reg_mask"][cls, iy, ix] = 1
        if cls == 2:
            result["streak_geom"][:, iy, ix] = (length, math.cos(2*angle), math.sin(2*angle))
            if "streak_line" in result:
                distance2 = v*v+np.maximum(abs(u)-length/(2*stride), 0)**2
                patch = result["streak_line"][a:b, d:e]
                np.maximum(patch, np.exp(-.5*distance2/(sigma*sigma)), out=patch)
        if "motion" in result and frame_k > 0:
            prev = next((p for p in objects[frame_k-1].candidates
                         if p["identity"] == c["identity"] and p["in_image"]), None)
            if prev is not None:
                if prev_H is None:
                    previous_frame = objects[frame_k-1]
                    prev_H = (H @ np.linalg.inv(frame.H_to_ref) @ previous_frame.H_to_ref
                              if reference else np.eye(3))
                    prev_H = validate_homography(prev_H, previous_frame.width, previous_frame.height)
                    prev_sign = _corner_sign(prev_H, previous_frame.width, previous_frame.height)
                px, py = _map(prev_H, prev["x"], prev["y"], prev_sign)
                cx, cy = _map(H, c["x"], c["y"], sign)
                result["motion"][cls, :, iy, ix] = (px-cx, py-cy)
                result["motion_mask"][cls, iy, ix] = 1
    # Positive peaks take priority over every ignore, including collisions.
    positive = result["reg_mask"].astype(bool)
    result["ignore"][positive] = 0
    result["heatmap"][(result["ignore"] != 0) & ~positive] = 0
    return result


def decode_dense(dense):
    """3×3 NMS and offset decoding of recoverable sparse fields."""
    from scipy.ndimage import maximum_filter

    heat = dense["heatmap"].astype(np.float32)
    peaks = (heat == maximum_filter(heat, size=(1, 3, 3), mode="constant")) & (heat == 1)
    decoded = []
    for cls, row, col in np.argwhere(peaks):
        ox, oy = dense["offset"][cls, :, row, col]
        x = decode_coordinate(col, float(ox), int(dense["stride"]), dense["origin"][0])
        y = decode_coordinate(row, float(oy), int(dense["stride"]), dense["origin"][1])
        length, cos2, sin2 = dense["streak_geom"][:, row, col]
        decoded.append(dict(class_id=int(cls), x=float(x), y=float(y),
                            length=float(length), angle=.5*math.atan2(float(sin2), float(cos2))))
    return decoded


def main():
    """Regenerate dense files from disk truth with chosen stride/thresholds."""
    import argparse
    from pathlib import Path
    from starsim.io import (load_objects, load_sequence, save_dense, prepare_output_directory,
                            write_labels_manifest, _sequence_count)

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--label-frame", choices=("native", "reference"))
    parser.add_argument("--snr-eff-min", type=float)
    args = parser.parse_args()
    if args.stride is not None and args.stride < 1:
        parser.error("--stride must be >= 1")
    sequence = load_sequence(args.directory/"sequence.json")
    cfg = dict(sequence["config_effective"]["labels"])
    if args.stride is not None:
        cfg["stride"] = args.stride
    if args.label_frame is not None:
        cfg["label_frame"] = args.label_frame
    if args.snr_eff_min is not None:
        cfg["label_snr_eff_min"] = args.snr_eff_min
    from starsim.config import validate_label_cfg
    validate_label_cfg(cfg)
    objects = [load_objects(args.directory/f"objects_{k:03d}.npz")
               for k in range(_sequence_count(sequence))]
    assign_status(objects, cfg)
    out = prepare_output_directory(args.output_directory)
    for k, frame in enumerate(objects):
        H = frame.H_to_ref if cfg["label_frame"] == "reference" else np.eye(3)
        save_dense(out/f"labels_{k:03d}.npz", _render_frame(objects, k, H, cfg))
    write_labels_manifest(out, args.directory, cfg, objects)


if __name__ == "__main__":
    main()
