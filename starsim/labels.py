"""Sparse source/candidate truth, D52–D56. No noisy pixels enter decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from starsim.camera import apply_homography, unproject_icrs
from starsim.photometry import local_sigma, peak_snr
from starsim.psf import MOFFAT_ALPHA_PER_SIGMA, MOFFAT_BETA
from starsim.render import integrate_pixel
from starsim.targets import classify_measured_snr

LABELS_VERSION = 3


@dataclass
class FrameLabels:
    frame_index: int
    width: int
    height: int
    sources: list[dict]
    candidates: list[dict]
    H_to_ref: np.ndarray


def climb_peak(image, row, col, cache=None):
    """Strict-equality 8-connected plateaus; row-major ties (D54)."""
    cache = {} if cache is None else cache
    start = (int(row), int(col))
    path = set()
    current = start
    height, width = image.shape
    while current not in cache:
        value = image[current]
        plateau = {current}
        todo = [current]
        higher = set()
        while todo:
            r, c = todo.pop()
            for rr in range(max(0, r - 1), min(height, r + 2)):
                for cc in range(max(0, c - 1), min(width, c + 2)):
                    cell = (rr, cc)
                    if image[cell] == value and cell not in plateau:
                        plateau.add(cell)
                        todo.append(cell)
                    elif image[cell] > value:
                        higher.add(cell)
        path.update(plateau)
        if not higher:
            current = min(plateau)
            cache[current] = current
            break
        current = min(higher, key=lambda cell: (-image[cell], cell))
    end = cache[current]
    for cell in sorted(path):
        cache[cell] = end
    return end


@njit(cache=True)
def _star_stamp(r0, r1, c0, c1, x, y, flux, sr, st, rx, ry, core, halo, ia2):
    stamp = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=np.float64)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            stamp[r-r0, c-c0] = integrate_pixel(
                r, c, x, y, flux, sr, st, rx, ry, core, halo, ia2, MOFFAT_BETA)
    return stamp


def _inside(x, y, width, height):
    return -0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5


def build_frame_labels(frame, sensor, psf, camera, motion, population, catalog, *, tiers=None, catalog_index=None):
    """Consume one iter_frames result; retain only source-level truth.

    Canonical global indices follow catalog index, then track_id. Exposure
    owners are remapped before any grouping. Temporary stamps are discarded.
    """
    stars, targets, exposure = frame.stars, frame.targets, frame.exposure
    k = frame.frame_index
    if tiers is None:
        tiers = {"bright": {"snr": [10, 300]}, "faint": {"snr": [4, 10]}, "ultra": {"snr": [1.5, 3]}}
    if catalog_index is None:
        catalog_index = {(int(a), int(b)): i for i, (a, b) in
                         enumerate(zip(catalog.tyc1, catalog.tyc2))}
    star_order = sorted(range(len(stars.x)), key=lambda i:
                        catalog_index[(int(stars.tyc1[i]), int(stars.tyc2[i]))])
    target_order = sorted(range(len(targets.track_id)), key=lambda i: int(targets.track_id[i]))
    old_order = star_order + [len(stars.x) + i for i in target_order]
    remap = {old: new for new, old in enumerate(old_order)}
    pop_index = {int(t): i for i, t in enumerate(population.track_id)}
    sources, stamps = [], []
    for old in old_order:
        static = old < len(stars.x)
        data = stars if static else targets
        i = old if static else old - len(stars.x)
        x = float(data.x[i] if static else data.x_c[i])
        y = float(data.y[i] if static else data.y_c[i])
        box = tuple(int(getattr(data, name)[i]) for name in
                    ("box_row0", "box_row1", "box_col0", "box_col1"))
        r0, r1, c0, c1 = box
        if static and r0 >= 0:
            sr, st, si, rx, ry = psf.axes(x, y)
            alpha = MOFFAT_ALPHA_PER_SIGMA * si
            stamp = _star_stamp(r0, r1, c0, c1, x, y, float(data.flux_e[i]),
                                sr, st, rx, ry,
                                (1-psf.halo_fraction)/(2*math.pi*sr*st),
                                psf.halo_fraction*(MOFFAT_BETA-1)/(math.pi*alpha*alpha),
                                1/(alpha*alpha))
        else:
            stamp = np.empty((0, 0)) if static else targets.stamps[i]
        clipped = bool(exposure.peak_clipped[old] if static else exposure.center_clipped[i])
        owner = int(exposure.peak_owner[old] if static else exposure.center_owner[i])
        peak_owner = int(exposure.peak_owner[old])
        source = dict(source_index=len(sources), class_id=0 if static else int(data.kind[i])+1,
                      x=x, y=y, flux_e=float(data.flux_e[i]), box=list(box),
                      has_flux=bool(data.flux_in_image[i]), in_image=bool(data.in_image[i]),
                      flux_in_image=bool(data.flux_in_image[i]),
                      peak_row=int(data.peak_row[i]), peak_col=int(data.peak_col[i]),
                      peak_clipped=bool(exposure.peak_clipped[old]),
                      peak_owner=remap.get(peak_owner, -1),
                      decision_clipped=clipped, decision_owner=remap.get(owner, -1),
                      s_self=float(exposure.s_self[old]), sigma_local=float(exposure.sigma_local[old]),
                      snr=float(exposure.snr[old]), sigma_psf=float(psf.sigma_iso(x, y)),
                      raw_group_id=int(frame.blend.group_id[i]) if static else -1)
        source["x_ref"], source["y_ref"] = map(float, apply_homography(motion.H_to_ref[k], x, y))
        if static:
            pair = (int(stars.tyc1[i]), int(stars.tyc2[i]))
            ci = catalog_index[pair]
            source.update(id=f"{pair[0]}-{pair[1]}", tyc1=pair[0], tyc2=pair[1],
                          catalog_index=ci, ra=float(catalog.ra_deg[ci]), dec=float(catalog.dec_deg[ci]),
                          vmag=float(stars.mag[i]))
        else:
            p = pop_index[int(targets.track_id[i])]
            source.update(id=str(int(targets.track_id[i])), track_id=int(targets.track_id[i]),
                          center_row=int(targets.center_row[i]), center_col=int(targets.center_col[i]),
                          center_clipped=bool(exposure.center_clipped[i]),
                          center_owner=remap.get(int(exposure.center_owner[i]), -1),
                          mag_eq=float(population.mag_eq[p, k]),
                          tier_design=str(population.tier_design[p]),
                          tier_measured=classify_measured_snr(source["snr"], tiers),
                          duty_eff=float(population.duty[p]), t_exp_eff=float(population.t_exp_eff[p]))
            for name in ("x1", "y1", "x2", "y2", "x1_ref", "y1_ref", "x2_ref", "y2_ref",
                         "L_true", "L_native", "angle_rad", "angle_ref", "x_vis", "y_vis",
                         "vis_x1", "vis_y1", "vis_x2", "vis_y2", "L_vis", "vis_frac"):
                source[name] = float(getattr(population, name)[p, k])
            source["truncated"] = bool(population.truncated[p, k])
            source["entering"] = bool(k > 0 and population.in_image[p, k] and not population.in_image[p, k-1])
            source["exiting"] = bool(k+1 < population.in_image.shape[1] and population.in_image[p, k]
                                     and not population.in_image[p, k+1])
            source["never_full"] = bool(population.never_full[p])
            va = apply_homography(motion.H_to_ref[k], source["vis_x1"], source["vis_y1"])
            vb = apply_homography(motion.H_to_ref[k], source["vis_x2"], source["vis_y2"])
            if source["vis_frac"] == 0:
                va = vb = np.array([-1., -1.])
            source.update(vis_x1_ref=float(va[0]), vis_y1_ref=float(va[1]),
                          vis_x2_ref=float(vb[0]), vis_y2_ref=float(vb[1]),
                          x_vis_ref=float((va[0]+vb[0])/2), y_vis_ref=float((va[1]+vb[1])/2),
                          L_vis_ref=float(np.linalg.norm(vb-va)))
            lo, hi = max(0, k-1), min(population.x_ref.shape[1]-1, k+1)
            source["vx_ref"] = float((population.x_ref[p, hi]-population.x_ref[p, lo])/(hi-lo))
            source["vy_ref"] = float((population.y_ref[p, hi]-population.y_ref[p, lo])/(hi-lo))
            vector = unproject_icrs(camera, motion.matrix[k], x, y)
            source.update(ra=float(np.degrees(np.arctan2(vector[1], vector[0])) % 360),
                          dec=float(np.degrees(np.arcsin(vector[2]))))
        if not source["has_flux"] and source["in_image"]:
            raise ValueError("in-image source without flux")
        sources.append(source)
        stamps.append(stamp)
    light = np.zeros((camera.height, camera.width), dtype=np.float64)
    for source, stamp in zip(sources, stamps):
        r0, r1, c0, c1 = source["box"]
        if r0 >= 0:
            light[r0:r1+1, c0:c1+1] += stamp
    pixel_scale = sensor.vignetting * sensor.prnu
    light *= pixel_scale
    groups, dominated = pixel_groups(sources, stamps, light, psf, pixel_scale)
    candidates = []
    for gid, members in enumerate(groups):
        records = [sources[i] for i in members]
        candidate = {"class_id": records[0]["class_id"]}
        if candidate["class_id"] == 0:
            total = sum(s["flux_e"] for s in records)
            x = sum(s["x"]*s["flux_e"] for s in records)/total
            y = sum(s["y"]*s["flux_e"] for s in records)/total
        else:
            from starsim.io import CANDIDATE_TARGET
            candidate.update({key: records[0][key] for key in CANDIDATE_TARGET})
            x, y, total = (records[0][key] for key in ("x", "y", "flux_e"))
        candidate.update(label_group_id=gid, members=list(members), blended=len(members)>1,
                         components=[s["id"] for s in records], flux_e=total, x=x, y=y,
                         sigma_psf=float(psf.sigma_iso(x, y)),
                         in_image=_inside(x, y, camera.width, camera.height),
                         has_flux=any(s["has_flux"] for s in records),
                         dominant_source=dominated.get(members, -1),
                         representative_member=max(members, key=lambda i: (sources[i]["flux_e"], -i)))
        candidate["identity"] = ("star:" + ";".join(f"{a}-{b}" for a, b in sorted(
            (s["tyc1"], s["tyc2"]) for s in records)) if candidate["class_id"] == 0
            else f"target:{candidate['track_id']}")
        candidate["stable_key"] = (sorted([s["tyc1"], s["tyc2"]] for s in records)
                                   if candidate["class_id"] == 0 else [candidate["track_id"]])
        if candidate["class_id"] == 0:
            candidate["x_ref"], candidate["y_ref"] = map(float, apply_homography(motion.H_to_ref[k], x, y))
            direction = np.zeros(3)
            for s in records:
                ra, dec = np.radians([s["ra"], s["dec"]])
                direction += s["flux_e"]*np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])
            direction /= np.linalg.norm(direction)
            candidate.update(ra=float(np.degrees(np.arctan2(direction[1], direction[0])) % 360),
                             dec=float(np.degrees(np.arcsin(direction[2]))),
                             vmag=-2.5*math.log10(sum(10**(-0.4*s["vmag"]) for s in records)))
        else:
            for key in ("x", "y", "x_ref", "y_ref", "flux_e", "ra", "dec"):
                candidate[key] = records[0][key]
        clipped = [s["decision_owner"] for s in records if s["decision_clipped"]]
        candidate.update(saturated=any(o in members for o in clipped),
                         in_foreign_sat=any(o >= 0 and o not in members for o in clipped),
                         in_defect_sat=-1 in clipped)
        signal, row, col = group_peak(members, sources, stamps)
        sigma, snr = 0.0, 0.0
        if row >= 0:
            signal *= float(sensor.vignetting[row, col]*sensor.prnu[row, col])
            sigma = local_sigma(float(exposure.expectation_e[row, col]), signal,
                                float(sensor.dark_e[row, col]), sensor.read_e)
            snr = peak_snr(signal, sigma)
        candidate.update(group_peak_row=row, group_peak_col=col, s_group=signal,
                         sigma_local=sigma, snr=snr)
        for i in members:
            sources[i]["candidate_id"] = gid
            sources[i]["label_group_id"] = gid
        candidates.append(candidate)
    result = FrameLabels(k, camera.width, camera.height, sources, candidates, motion.H_to_ref[k].copy())
    assert_partition(result)
    return result


def _contribution(source, stamp, row, col):
    r0, r1, c0, c1 = source["box"]
    return float(stamp[row-r0, col-c0]) if r0 <= row <= r1 and c0 <= col <= c1 and r0 >= 0 else 0.0


def pixel_groups(sources, stamps, light, psf, pixel_scale=None):
    """Frozen-round unions on canonical source order; targets never merge."""
    groups = {}
    isolated = []
    for i, s in enumerate(sources):
        if not s["has_flux"] or s["class_id"] != 0:
            isolated.append((i,))
        else:
            groups.setdefault(s["raw_group_id"], []).append(i)
    active = sorted(tuple(v) for v in groups.values())
    # Inclusive boxes: the exact filter below retains boundary pixels and
    # canonical enumeration retains the frozen contribution tie break.
    cells = {}
    for j, source in enumerate(sources):
        r0, r1, c0, c1 = source['box']
        if r0 >= 0:
            for rr in range(r0//32, r1//32+1):
                for cc in range(c0//32, c1//32+1):
                    cells.setdefault((rr, cc), []).append(j)
    cache, ends = {}, {}
    for i, s in enumerate(sources):
        if s["class_id"] == 0 and s["peak_row"] >= 0 and s["s_self"] > 0:
            row, col = climb_peak(light, s["peak_row"], s["peak_col"], cache)
            scale = 1.0 if pixel_scale is None else float(pixel_scale[row, col])
            covering = [j for j in cells.get((row//32, col//32), [])
                        if sources[j]["box"][0] <= row <= sources[j]["box"][1]
                        and sources[j]["box"][2] <= col <= sources[j]["box"][3]]
            ends[i] = max(covering, key=lambda j:
                          (_contribution(sources[j], stamps[j], row, col)*scale, -j))
    while True:
        parent = list(range(len(active)))
        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i
        membership = {i: g for g, members in enumerate(active) for i in members}
        centers = []
        for members in active:
            flux = sum(sources[i]["flux_e"] for i in members)
            centers.append(tuple(sum(sources[i][axis]*sources[i]["flux_e"] for i in members)/flux
                                 for axis in ("x", "y")))
        dominant = []
        for g, members in enumerate(active):
            rep = max(members, key=lambda i: (sources[i]["flux_e"], -i))
            owner = ends.get(rep, rep)
            dominant.append(owner)
            other = membership.get(owner)
            if other is not None and other != g:
                x, y = centers[g]
                xx, yy = centers[other]
                if math.hypot(x-xx, y-yy) <= 2*psf.sigma_iso(x, y):
                    a, b = root(g), root(other)
                    parent[max(a, b)] = min(a, b)
        merged = {}
        for g, members in enumerate(active):
            merged.setdefault(root(g), []).extend(members)
        updated = sorted(tuple(sorted(v)) for v in merged.values())
        if updated == active:
            dominated = {members: owner for members, owner in zip(active, dominant) if owner not in members}
            return sorted(active + isolated), dominated
        active = updated


def group_peak(members, sources, stamps):
    valid = [i for i in members if sources[i]["box"][0] >= 0]
    if not valid:
        return 0.0, -1, -1
    r0 = min(sources[i]["box"][0] for i in valid)
    r1 = max(sources[i]["box"][1] for i in valid)
    c0 = min(sources[i]["box"][2] for i in valid)
    c1 = max(sources[i]["box"][3] for i in valid)
    image = np.zeros((r1-r0+1, c1-c0+1))
    for i in valid:
        a, b, c, d = sources[i]["box"]
        image[a-r0:b-r0+1, c-c0:d-c0+1] += stamps[i]
    row, col = np.unravel_index(np.argmax(image), image.shape)
    return float(image[row, col]), int(row+r0), int(col+c0)


def assert_partition(frame):
    identities = [c["identity"] for c in frame.candidates]
    if len(identities) != len(set(identities)):
        raise ValueError("candidate identity must be unique within a frame")
    members = [i for c in frame.candidates for i in c["members"]]
    if sorted(members) != list(range(len(frame.sources))):
        raise ValueError("sources must form a disjoint candidate partition")
    for gid, c in enumerate(frame.candidates):
        if not c["members"] or c["label_group_id"] != gid or not all(
                frame.sources[i]["candidate_id"] == gid for i in c["members"]):
            raise ValueError("candidate reverse links do not match partition")
        records = [frame.sources[i] for i in c["members"]]
        if any(not r["has_flux"] for r in records) and (len(records) != 1 or c["has_flux"]):
            raise ValueError("no-flux source must form its own no-flux candidate")
        if any(r["class_id"] != c["class_id"] for r in records):
            raise ValueError("candidate mixes source classes")


def assign_status(objects_seq, cfg):
    """Recompute statuses from sparse truth, without changing identity windows."""
    identities = {}
    star_identities = {}
    for frame in objects_seq:
        ids = [c["identity"] for c in frame.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate identity must be unique within a frame")
        for c in frame.candidates:
            identities.setdefault(c["identity"], []).append((frame.frame_index, c))
            if c["class_id"] == 0:
                for component in c["components"]:
                    star_identities.setdefault(component, set()).add(c["identity"])
    all_windows = {}
    if cfg.get("snr_window", "all") == "all":
        for identity, observations in identities.items():
            visible = [(k, v) for k, v in observations if v["in_image"]]
            all_windows[identity] = (math.sqrt(sum(v["snr"]**2 for k, v in visible)), len(visible))
    for frame in objects_seq:
        for source in frame.sources:
            if source.get("class_id") == 0:
                source["identity_count"] = len(star_identities[source["id"]])
        for c in frame.candidates:
            window = cfg.get("snr_window", "all")
            if window == "all":
                c["snr_eff"], c["identity_frame_count"] = all_windows[c["identity"]]
            else:
                lo = frame.frame_index - (int(window)-1)//2
                observations = [(k, v) for k, v in identities[c["identity"]]
                                if lo <= k < lo+int(window) and v["in_image"]]
                c["snr_eff"] = math.sqrt(sum(v["snr"]**2 for k, v in observations))
                c["identity_frame_count"] = len(observations)
            if c["class_id"] == 0:
                c["member_identity_counts"] = [len(star_identities[s]) for s in c["components"]]
            reason = ""
            if not c["has_flux"]:
                reason = "no_flux"
            elif c["dominant_source"] >= 0:
                reason = "pixel_dominated"
            elif c["in_foreign_sat"] or c["in_defect_sat"]:
                reason = "clipped_by_other"
            elif not c["in_image"]:
                reason = "center_outside"
            elif (c["snr_eff"] < cfg.get("label_snr_eff_min", 3.0) if cfg.get("snr_mode", "effective") == "effective"
                  else c["snr"] < cfg.get("label_snr_min", 1.0)):
                reason = "low_snr"
            elif c.get("never_full", False) and cfg.get("never_full_policy", "label") == "ignore":
                reason = "never_full"
            c["ignore_reason_default"] = reason
            c["status_default"] = "ignore" if reason else "labeled"
            if (c["status_default"] == "ignore") != bool(reason):
                raise ValueError("status and ignore reason disagree")
