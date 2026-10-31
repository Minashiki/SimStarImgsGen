"""Tycho-1 catalog loading, npz cache, and cone queries on the unit sphere."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import time
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import scipy.io
from numpy.lib import format as npformat
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

CACHE_KEYS: tuple[str, ...] = ("tyc1", "tyc2", "vmag", "ra_deg", "dec_deg", "xyz")
CACHE_FORMAT_VERSION = 1
META_KEY = "meta"
TYCHO_MAT_COLUMNS = ("TYC1", "TYC2", "TYC3", "Vmag", "RA_deg", "Dec_deg")
_INT32_MIN = int(np.iinfo(np.int32).min)
_INT32_MAX = int(np.iinfo(np.int32).max)
_XYZ_ATOL = 1e-12
# Fixed zip timestamp so rebuilding one source file is byte-identical.
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def default_mat_path() -> Path:
    return Path(__file__).resolve().parents[1] / "Tycho1" / "Tycho1.mat"


def default_cache_path() -> Path:
    return Path(__file__).resolve().parents[1] / "cache" / "tycho1.npz"


def icrs_unit_vector(ra_deg: np.ndarray | float, dec_deg: np.ndarray | float) -> np.ndarray:
    """ICRS unit vector ``[cosδ cosα, cosδ sinα, sinδ]``."""
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    alpha = np.deg2rad(ra)
    delta = np.deg2rad(dec)
    xyz = np.stack(
        (
            np.cos(delta) * np.cos(alpha),
            np.cos(delta) * np.sin(alpha),
            np.sin(delta),
        ),
        axis=-1,
    )
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("ICRS direction has zero norm")
    return xyz / norm


def chord_length(radius_deg: float) -> float:
    """Euclidean chord on the unit sphere for an angular radius."""
    if radius_deg < 0.0:
        raise ValueError(f"radius_deg must be >= 0, got {radius_deg}")
    if radius_deg > 180.0:
        raise ValueError(f"radius_deg must be <= 180, got {radius_deg}")
    return float(2.0 * np.sin(np.deg2rad(radius_deg) / 2.0))


def search_radius_deg(
    half_diagonal_deg: float,
    margin_px: float,
    pixel_scale_arcsec: float,
) -> float:
    """Cone radius = FOV half-diagonal plus a pixel safety margin.

    The margin is the PSF radius plus the maximum single-frame displacement,
    expressed in pixels.
    """
    if half_diagonal_deg < 0.0 or margin_px < 0.0 or pixel_scale_arcsec < 0.0:
        raise ValueError("half-diagonal, margin, and pixel scale must be >= 0")
    return float(half_diagonal_deg + margin_px * pixel_scale_arcsec / 3600.0)


def covering_cone(
    centers_xyz: np.ndarray,
    radius_deg: float,
) -> tuple[np.ndarray, float]:
    """One cone that covers several cones of the same angular radius.

    The center is the renormalized mean of the input directions. The radius is
    the greatest angular distance from that center, plus ``radius_deg``.
    """
    centers = np.asarray(centers_xyz, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
        raise ValueError("centers_xyz must have shape (N, 3) with N >= 1")
    if radius_deg < 0.0:
        raise ValueError(f"radius_deg must be >= 0, got {radius_deg}")
    mean = centers.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        raise ValueError("frame centers average to the zero vector")
    mean = mean / norm
    cosine = np.clip(centers @ mean, -1.0, 1.0)
    cover_deg = float(np.rad2deg(np.arccos(cosine)).max() + radius_deg)
    return mean, cover_deg


def _check_tycho_table(table: np.ndarray) -> None:
    """Require TYC3 == 1 and integral, positive, int32 TYC1/TYC2 on every row."""
    if table.ndim != 2 or table.shape[1] != 6:
        raise ValueError(
            f"Tyc must have shape (N, 6) columns {TYCHO_MAT_COLUMNS}, got {getattr(table, 'shape', None)}"
        )
    problems: list[str] = []
    tyc3 = table[:, 2]
    bad_tyc3 = tyc3 != 1
    if np.any(bad_tyc3):
        first = int(np.flatnonzero(bad_tyc3)[0])
        problems.append(
            f"TYC3 必须全为 1：{int(bad_tyc3.sum())} 行不合格，"
            f"首个下标 {first}，值 {tyc3[first]!r}"
        )
    for column, name in ((0, "TYC1"), (1, "TYC2")):
        values = table[:, column]
        finite = np.isfinite(values)
        rounded = np.rint(values)
        integral = finite & (values == rounded)
        in_range = integral & (rounded > 0) & (rounded >= _INT32_MIN) & (rounded <= _INT32_MAX)
        bad = ~in_range
        if np.any(bad):
            first = int(np.flatnonzero(bad)[0])
            problems.append(
                f"{name} 必须是大于 0 且在 int32 范围内的整数："
                f"{int(bad.sum())} 行不合格，首个下标 {first}，值 {values[first]!r}"
            )
    if problems:
        raise ValueError("第谷星表不满足编号前提：\n" + "\n".join(problems))


def read_tycho_mat(path: Path | str) -> tuple[dict[str, np.ndarray], dict]:
    """Read ``Tyc``, drop NaN Vmag rows, and return ``(arrays, meta)``.

    ``meta`` contains ``n_source_rows``, ``n_dropped``, and ``dropped_ids``.
    Dropped ``(TYC1, TYC2)`` pairs are also written to the log. The id
    preconditions are checked on the whole table, including rows that are dropped.
    """
    path = Path(path)
    loaded = scipy.io.loadmat(path, squeeze_me=False)
    if "Tyc" not in loaded:
        raise KeyError(f"{path} has no 'Tyc' variable")
    table = np.asarray(loaded["Tyc"], dtype=np.float64)
    _check_tycho_table(table)
    nan_mag = np.isnan(table[:, 3])
    n_dropped = int(nan_mag.sum())
    dropped_ids: list[list[int]] = []
    if n_dropped:
        dropped = table[nan_mag]
        dropped_ids = [
            [int(np.rint(row[0])), int(np.rint(row[1]))] for row in dropped
        ]
        pairs = ", ".join(f"({tyc1}, {tyc2})" for tyc1, tyc2 in dropped_ids)
        logger.warning("丢弃 %d 颗 Vmag 为 NaN 的星: %s", n_dropped, pairs)
    keep = table[~nan_mag]
    if np.any(~np.isfinite(keep)):
        raise ValueError(f"{path} has non-finite catalog values besides NaN Vmag")
    ra_deg = np.ascontiguousarray(keep[:, 4], dtype=np.float64)
    dec_deg = np.ascontiguousarray(keep[:, 5], dtype=np.float64)
    arrays = {
        "tyc1": np.rint(keep[:, 0]).astype(np.int32),
        "tyc2": np.rint(keep[:, 1]).astype(np.int32),
        "vmag": np.ascontiguousarray(keep[:, 3], dtype=np.float64),
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "xyz": np.ascontiguousarray(icrs_unit_vector(ra_deg, dec_deg), dtype=np.float64),
    }
    meta = {
        "n_source_rows": int(table.shape[0]),
        "n_dropped": n_dropped,
        "dropped_ids": dropped_ids,
    }
    return arrays, meta


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def catalog_meta(mat_path: Path | str, read_meta: dict) -> dict:
    """Cache metadata for one source file. No build timestamp."""
    path = Path(mat_path)
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "source_name": path.name,
        "source_sha256": sha256_file(path),
        "source_size": path.stat().st_size,
        "n_source_rows": int(read_meta["n_source_rows"]),
        "n_dropped": int(read_meta["n_dropped"]),
        "dropped_ids": read_meta["dropped_ids"],
        "frame": "ICRS",
        "epoch": "J1991.25",
        "proper_motion": False,
        "columns": list(CACHE_KEYS),
    }


def save_cache(path: Path | str, arrays: dict[str, np.ndarray], meta: dict) -> None:
    """Write a cache atomically and publish it only after a successful read-back.

    The temporary file is ``{name}.tmp-{pid}`` in the destination directory
    (for example ``tycho1.npz.tmp-1234``). Arrays are written through the open
    file handle, so the name is not given an extra ``.npz`` suffix. A failed
    validation or replace deletes the temporary file. ``PermissionError`` from
    ``os.replace`` is not retried.
    """
    path = Path(path)
    missing = [key for key in CACHE_KEYS if key not in arrays]
    if missing:
        raise KeyError(f"catalog arrays missing {missing}")
    if not isinstance(meta, dict):
        raise TypeError("meta must be a mapping")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    payload = {key: arrays[key] for key in CACHE_KEYS}
    payload[META_KEY] = _encode_meta(meta)
    try:
        with tmp.open("wb") as handle:
            _savez_deterministic(handle, payload)
            handle.flush()
            os.fsync(handle.fileno())
        arrays2, meta2 = load_cache(tmp)
        StarCatalog(arrays2, meta2)
        for key in CACHE_KEYS:
            if not np.array_equal(arrays[key], arrays2[key]):
                raise ValueError(f"缓存回读与写入不一致：{key}")
        if _meta_json(meta) != _meta_json(meta2):
            raise ValueError("缓存回读与写入的 meta 不一致")
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_cache(path: Path | str) -> tuple[dict[str, np.ndarray], dict]:
    """Load arrays and meta. The npz must contain exactly the data keys plus meta."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        present = set(data.files)
        required = set(CACHE_KEYS) | {META_KEY}
        missing = sorted(required - present)
        extra = sorted(present - required)
        if missing or extra:
            raise ValueError(
                f"{path} 的缓存键不符合格式 v{CACHE_FORMAT_VERSION}："
                f"缺少 {missing}，多余 {extra}。"
                "请运行 scripts/build_catalog_cache.py 重新构建星表缓存。"
            )
        meta = _decode_meta(np.array(data[META_KEY], copy=True))
        arrays = {key: np.array(data[key], copy=True) for key in CACHE_KEYS}
    if arrays["vmag"].ndim == 1:
        _check_meta(meta, int(arrays["vmag"].shape[0]))
    return arrays, meta


def cleanup_orphan_temps(out: Path | str, *, max_age_s: float = 3600) -> None:
    """Delete ``{name}.tmp-*`` files left behind when a build is killed.

    A file older than ``max_age_s`` is removed and logged at info. A newer
    file is left in place: another build may still be writing it. Process
    ids are not consulted, because on Windows ``os.kill(pid, 0)`` terminates
    the process.
    """
    out = Path(out)
    directory = out.parent
    if not directory.is_dir():
        return
    now = time.time()
    for tmp in directory.glob(out.name + ".tmp-*"):
        if not tmp.is_file():
            continue
        try:
            age = now - tmp.stat().st_mtime
        except OSError:
            continue
        if age > max_age_s:
            try:
                tmp.unlink()
            except OSError:
                logger.warning("无法删除过期临时文件 %s", tmp)
                continue
            logger.info("删除过期临时文件 %s", tmp.name)
        else:
            logger.warning("临时文件 %s 仍较新，可能有并发构建，未删除", tmp.name)


def build_cache(mat_path: Path | str, out_path: Path | str) -> dict:
    """Write the npz cache. Returns kept/dropped counts and the cache meta."""
    out_path = Path(out_path)
    cleanup_orphan_temps(out_path)
    arrays, read_meta = read_tycho_mat(mat_path)
    meta = catalog_meta(mat_path, read_meta)
    save_cache(out_path, arrays, meta)
    return {
        "n_kept": int(arrays["vmag"].shape[0]),
        "n_dropped": int(read_meta["n_dropped"]),
        "meta": meta,
    }


class StarCatalog:
    """In-memory Tycho-1 catalog with a unit-vector KD tree.

    Every constructor copies the six columns, validates them, then marks
    them read-only. ``meta`` is required and must agree with the row count.
    ``meta`` 是构造时的快照，与调用方传入的字典相互独立；修改 catalog.meta
    不在支持范围内。
    """

    def __init__(self, arrays: dict[str, np.ndarray], meta: dict):
        missing = [key for key in CACHE_KEYS if key not in arrays]
        if missing:
            raise KeyError(f"catalog arrays missing {missing}")
        self.meta = copy.deepcopy(meta)
        self._tree: cKDTree | None = None
        self._load_and_validate(arrays)
        _check_meta(self.meta, len(self))

    def __len__(self) -> int:
        return int(self.vmag.shape[0])

    @classmethod
    def from_mat(cls, path: Path | str) -> StarCatalog:
        arrays, read_meta = read_tycho_mat(path)
        return cls(arrays, catalog_meta(path, read_meta))

    @classmethod
    def from_cache(cls, path: Path | str) -> StarCatalog:
        arrays, meta = load_cache(path)
        return cls(arrays, meta)

    @property
    def tree(self) -> cKDTree:
        if self._tree is None:
            self._tree = cKDTree(np.asarray(self.xyz, dtype=np.float64))
        return self._tree

    def cone_query(self, ra_deg: float, dec_deg: float, radius_deg: float) -> np.ndarray:
        """Indices of stars within ``radius_deg`` of the given ICRS direction.

        The KD-tree radius is the chord ``2·sin(r/2)``. Indices are sorted
        in ascending catalog order. ``render_static_stars`` sorts again
        after its magnitude cut so the painted order stays that catalog order.
        """
        center = icrs_unit_vector(ra_deg, dec_deg).reshape(3)
        return self._query_chord(center, chord_length(float(radius_deg)))

    def query_sequence_union(
        self,
        ra_deg: np.ndarray | float,
        dec_deg: np.ndarray | float,
        radius_deg: float,
    ) -> np.ndarray:
        """One query covering the union of per-frame cones.

        Each frame contributes a cone of ``radius_deg`` about its own center.
        """
        ra = np.atleast_1d(np.asarray(ra_deg, dtype=np.float64))
        dec = np.atleast_1d(np.asarray(dec_deg, dtype=np.float64))
        if ra.shape != dec.shape:
            raise ValueError("ra_deg and dec_deg must have the same shape")
        centers = icrs_unit_vector(ra, dec)
        if centers.ndim == 1:
            centers = centers.reshape(1, 3)
        mean, cover_deg = covering_cone(centers, float(radius_deg))
        if cover_deg >= 180.0:
            return np.arange(len(self), dtype=np.int64)
        return self._query_chord(mean, chord_length(cover_deg))

    def _load_and_validate(self, arrays: dict[str, np.ndarray]) -> None:
        raw = {key: np.asarray(arrays[key]) for key in CACHE_KEYS}
        n = int(raw["vmag"].shape[0]) if raw["vmag"].ndim == 1 else None
        if n is None:
            raise ValueError(f"vmag 必须是一维数组，得到形状 {raw['vmag'].shape}")
        if raw["xyz"].shape != (n, 3):
            raise ValueError(f"xyz 必须具有形状 ({n}, 3)，得到 {raw['xyz'].shape}")
        for name in ("tyc1", "tyc2", "ra_deg", "dec_deg"):
            if raw[name].shape != (n,):
                raise ValueError(f"{name} 必须具有形状 ({n},)，得到 {raw[name].shape}")
        self.tyc1 = np.array(_as_tyc_id(raw["tyc1"], "tyc1"), dtype=np.int32, order="C", copy=True)
        self.tyc2 = np.array(_as_tyc_id(raw["tyc2"], "tyc2"), dtype=np.int32, order="C", copy=True)
        self.vmag = np.array(raw["vmag"], dtype=np.float64, order="C", copy=True)
        self.ra_deg = np.array(raw["ra_deg"], dtype=np.float64, order="C", copy=True)
        self.dec_deg = np.array(raw["dec_deg"], dtype=np.float64, order="C", copy=True)
        self.xyz = np.array(raw["xyz"], dtype=np.float64, order="C", copy=True)
        self._check_finite()
        self._check_ranges()
        self._check_unique_ids()
        self._check_xyz_matches_radec()
        for column in (self.tyc1, self.tyc2, self.vmag, self.ra_deg, self.dec_deg, self.xyz):
            column.setflags(write=False)

    def _check_finite(self) -> None:
        for name, column in (
            ("vmag", self.vmag),
            ("ra_deg", self.ra_deg),
            ("dec_deg", self.dec_deg),
            ("xyz", self.xyz),
        ):
            bad = ~np.isfinite(column)
            if column.ndim == 2:
                bad = np.any(bad, axis=1)
            if np.any(bad):
                first = int(np.flatnonzero(bad)[0])
                raise ValueError(
                    f"{name} 含非有限值：{int(bad.sum())} 行，"
                    f"首个下标 {first}，TYC {int(self.tyc1[first])}-{int(self.tyc2[first])}"
                )

    def _check_ranges(self) -> None:
        ra_bad = (self.ra_deg < 0.0) | (self.ra_deg >= 360.0)
        if np.any(ra_bad):
            first = int(np.flatnonzero(ra_bad)[0])
            deviation = float(np.maximum(-self.ra_deg, self.ra_deg - 360.0)[ra_bad].max())
            raise ValueError(
                f"ra_deg 超出 [0, 360)：{int(ra_bad.sum())} 行，最大偏差 {deviation:.6g}°，"
                f"首个下标 {first}，TYC {int(self.tyc1[first])}-{int(self.tyc2[first])}"
            )
        dec_bad = (self.dec_deg < -90.0) | (self.dec_deg > 90.0)
        if np.any(dec_bad):
            first = int(np.flatnonzero(dec_bad)[0])
            outside = np.maximum(-90.0 - self.dec_deg, self.dec_deg - 90.0)
            raise ValueError(
                f"dec_deg 超出 [-90, 90]：{int(dec_bad.sum())} 行，"
                f"最大偏差 {float(outside[dec_bad].max()):.6g}°，"
                f"首个下标 {first}，TYC {int(self.tyc1[first])}-{int(self.tyc2[first])}"
            )

    def _check_unique_ids(self) -> None:
        if np.any(self.tyc1 <= 0) or np.any(self.tyc2 <= 0):
            bad = (self.tyc1 <= 0) | (self.tyc2 <= 0)
            first = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"TYC 编号必须大于 0：{int(bad.sum())} 行，"
                f"首个下标 {first}，TYC {int(self.tyc1[first])}-{int(self.tyc2[first])}"
            )
        keys = (self.tyc1.astype(np.int64) << 32) | self.tyc2.astype(np.int64)
        _unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        if counts.shape[0] != keys.shape[0]:
            bad = counts[inverse] > 1
            first = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"(TYC1, TYC2) 不唯一：{int(bad.sum())} 行重复，"
                f"首个下标 {first}，TYC {int(self.tyc1[first])}-{int(self.tyc2[first])}"
            )

    def _check_xyz_matches_radec(self) -> None:
        expected = icrs_unit_vector(self.ra_deg, self.dec_deg)
        component = np.abs(self.xyz - expected).max(axis=1)
        bad = component >= _XYZ_ATOL
        if not np.any(bad):
            return
        cosine = np.clip(np.sum(self.xyz * expected, axis=1), -1.0, 1.0)
        angle = np.arccos(cosine)
        first = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"xyz 与赤经赤纬不一致：{int(bad.sum())} 行，"
            f"最大偏差 {float(angle[bad].max()):.6g} rad，"
            f"首个下标 {first}，TYC {int(self.tyc1[first])}-{int(self.tyc2[first])}"
        )

    def _query_chord(self, center_xyz: np.ndarray, chord: float) -> np.ndarray:
        found = self.tree.query_ball_point(np.asarray(center_xyz, dtype=np.float64), r=float(chord))
        return np.sort(np.asarray(found, dtype=np.int64))


def _as_tyc_id(values: np.ndarray, name: str) -> np.ndarray:
    if np.issubdtype(values.dtype, np.floating):
        finite = np.isfinite(values)
        rounded = np.rint(values)
        bad = ~finite | (values != rounded)
        if np.any(bad):
            first = int(np.flatnonzero(bad)[0])
            raise ValueError(f"{name} 含非整数：首个下标 {first}，值 {values[first]!r}")
        return np.ascontiguousarray(rounded.astype(np.int32))
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} 必须是整数数组，得到 {values.dtype}")
    if values.dtype == np.int32 and values.flags.c_contiguous:
        return values
    return np.ascontiguousarray(values, dtype=np.int32)


def _check_meta(meta: dict, n: int) -> None:
    """Reject cache metadata that does not describe ``n`` rows."""
    if not isinstance(meta, dict):
        raise TypeError("meta must be a mapping")
    problems: list[str] = []
    version = meta.get("format_version")
    if version != CACHE_FORMAT_VERSION:
        problems.append(
            f"缓存格式版本是 {version!r}，需要 {CACHE_FORMAT_VERSION}。"
            "请运行 scripts/build_catalog_cache.py 重新构建星表缓存。"
        )
    n_source = meta.get("n_source_rows")
    n_dropped = meta.get("n_dropped")
    source_ok = isinstance(n_source, int) and not isinstance(n_source, bool)
    dropped_ok = isinstance(n_dropped, int) and not isinstance(n_dropped, bool)
    if not source_ok or not dropped_ok:
        problems.append(
            f"n_source_rows 与 n_dropped 必须是整数，得到 {n_source!r}、{n_dropped!r}"
        )
    elif n_source - n_dropped != n:
        problems.append(
            f"n_source_rows - n_dropped = {n_source - n_dropped}，与星数 {n} 不一致"
        )
    dropped_ids = meta.get("dropped_ids")
    if not isinstance(dropped_ids, list):
        problems.append(f"dropped_ids 必须是列表，得到 {type(dropped_ids).__name__}")
    elif dropped_ok and len(dropped_ids) != n_dropped:
        problems.append(
            f"len(dropped_ids) = {len(dropped_ids)}，与 n_dropped {n_dropped} 不一致"
        )
    if meta.get("columns") != list(CACHE_KEYS):
        problems.append(f"columns 必须是 {list(CACHE_KEYS)}，得到 {meta.get('columns')!r}")
    if meta.get("frame") != "ICRS":
        problems.append(f"frame 必须是 'ICRS'，得到 {meta.get('frame')!r}")
    if meta.get("epoch") != "J1991.25":
        problems.append(f"epoch 必须是 'J1991.25'，得到 {meta.get('epoch')!r}")
    if problems:
        raise ValueError("缓存 meta 与数据不一致：\n" + "\n".join(problems))


def _encode_meta(meta: dict) -> np.ndarray:
    return np.array(_meta_json(meta))


def _decode_meta(raw: np.ndarray) -> dict:
    try:
        text = raw.item()
    except ValueError as exc:
        raise ValueError("缓存 meta 必须是 0 维字符串数组") from exc
    if not isinstance(text, str):
        text = str(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"缓存 meta 不是 JSON：{exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("缓存 meta JSON 必须是对象")
    return parsed


def _meta_json(meta: dict) -> str:
    return json.dumps(meta, sort_keys=True, ensure_ascii=True)


def _savez_deterministic(handle, arrays: dict[str, np.ndarray]) -> None:
    """Write an uncompressed npz with a fixed timestamp and key order."""
    with zipfile.ZipFile(handle, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for key in list(CACHE_KEYS) + [META_KEY]:
            buffer = BytesIO()
            npformat.write_array(buffer, np.asanyarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(filename=f"{key}.npy", date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, buffer.getvalue())
