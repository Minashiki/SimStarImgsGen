"""M0 acceptance tests for the delivered Tycho-1 cache and cone query."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import zipfile

import numpy as np
import pytest

from starsim.catalog import (
    CACHE_FORMAT_VERSION,
    CACHE_KEYS,
    META_KEY,
    StarCatalog,
    _check_tycho_table,
    cleanup_orphan_temps,
    default_cache_path,
    default_mat_path,
    icrs_unit_vector,
    load_cache,
    read_tycho_mat,
    save_cache,
    search_radius_deg,
)

# SIMBAD J2000 for Alcyone (eta Tau): 03h47m29.07655s, +24d06m18.488s.
ALCYONE_RA_DEG = 15.0 * (3.0 + 47.0 / 60.0 + 29.07655 / 3600.0)
ALCYONE_DEC_DEG = 24.0 + 6.0 / 60.0 + 18.488 / 3600.0
PLEIADES_RA_DEG = 56.75
PLEIADES_DEC_DEG = 24.12


class _ListHandler(logging.Handler):
    def __init__(self, records: list[str]):
        super().__init__(level=logging.WARNING)
        self.records = records

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_table(n: int) -> np.ndarray:
    table = np.zeros((n, 6), dtype=np.float64)
    table[:, 0] = np.arange(1, n + 1)
    table[:, 1] = 1
    table[:, 2] = 1
    table[:, 3] = 8.0
    table[:, 4] = np.linspace(1.0, 359.0, n)
    table[:, 5] = np.linspace(-20.0, 20.0, n)
    return table


def _arrays_and_meta(table: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    ra = np.ascontiguousarray(table[:, 4], dtype=np.float64)
    dec = np.ascontiguousarray(table[:, 5], dtype=np.float64)
    arrays = {
        "tyc1": np.rint(table[:, 0]).astype(np.int32),
        "tyc2": np.rint(table[:, 1]).astype(np.int32),
        "vmag": np.ascontiguousarray(table[:, 3], dtype=np.float64),
        "ra_deg": ra,
        "dec_deg": dec,
        "xyz": np.ascontiguousarray(icrs_unit_vector(ra, dec), dtype=np.float64),
    }
    meta = {
        "format_version": CACHE_FORMAT_VERSION,
        "source_name": "synthetic.mat",
        "source_sha256": "0" * 64,
        "source_size": 0,
        "n_source_rows": int(table.shape[0]),
        "n_dropped": 0,
        "dropped_ids": [],
        "frame": "ICRS",
        "epoch": "J1991.25",
        "proper_motion": False,
        "columns": list(CACHE_KEYS),
    }
    return arrays, meta


def _write_unchecked(path, arrays: dict, meta: dict) -> None:
    payload = {key: arrays[key] for key in arrays}
    payload[META_KEY] = np.array(json.dumps(meta, sort_keys=True))
    with path.open("wb") as handle:
        np.savez(handle, **payload)


def _flip_npy_payload_byte(path, member: str) -> None:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
    with path.open("r+b") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        if header[:4] != b"PK\x03\x04":
            raise AssertionError("missing zip local header")
        name_len = int.from_bytes(header[26:28], "little")
        extra_len = int.from_bytes(header[28:30], "little")
        data_offset = info.header_offset + 30 + name_len + extra_len
        handle.seek(data_offset)
        magic = handle.read(8)
        major = magic[6]
        handle.seek(data_offset + 8)
        if major == 1:
            header_len = int.from_bytes(handle.read(2), "little")
            payload = data_offset + 10 + header_len
        else:
            header_len = int.from_bytes(handle.read(4), "little")
            payload = data_offset + 12 + header_len
        handle.seek(payload + 16)
        byte = handle.read(1)
        handle.seek(payload + 16)
        handle.write(bytes([byte[0] ^ 0x01]))


@pytest.fixture(scope="module")
def built():
    records: list[str] = []
    logger = logging.getLogger("starsim.catalog")
    previous = logger.level
    handler = _ListHandler(records)
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        arrays, meta = read_tycho_mat(default_mat_path())
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    return {"arrays": arrays, "meta": meta, "logs": records}


@pytest.fixture(scope="module")
def catalog():
    path = default_cache_path()
    if not path.is_file():
        pytest.skip("先运行 scripts/build_catalog_cache.py")
    return StarCatalog.from_cache(path)


def _separation_arcsec(ra1, dec1, ra2, dec2) -> float:
    first = icrs_unit_vector(ra1, dec1).reshape(-1, 3)
    second = icrs_unit_vector(ra2, dec2).reshape(3)
    cosine = np.clip(first @ second, -1.0, 1.0)
    arcsec = np.rad2deg(np.arccos(cosine)) * 3600.0
    return float(np.asarray(arcsec).reshape(-1)[0])


def test_nan_vmag_stars_are_dropped_and_logged(built):
    n_kept = len(built["arrays"]["vmag"])
    assert built["meta"]["n_dropped"] == 72
    assert built["meta"]["n_source_rows"] == 1_051_966
    assert n_kept + built["meta"]["n_dropped"] == 1_051_966
    assert len(built["meta"]["dropped_ids"]) == 72
    assert np.isfinite(built["arrays"]["vmag"]).all()
    text = "\n".join(built["logs"])
    assert "丢弃 72 颗 Vmag 为 NaN 的星" in text


def test_delivered_cache_matches_mat(built):
    path = default_cache_path()
    if not path.is_file():
        pytest.skip("先运行 scripts/build_catalog_cache.py")
    catalog = StarCatalog.from_cache(path)
    assert catalog.meta["source_sha256"] == _sha256(default_mat_path())
    assert catalog.meta["n_dropped"] == 72
    assert catalog.meta["epoch"] == "J1991.25"
    assert catalog.meta["frame"] == "ICRS"
    assert catalog.meta["proper_motion"] is False
    assert catalog.meta["format_version"] == CACHE_FORMAT_VERSION
    for key in ("tyc1", "tyc2", "vmag", "ra_deg", "dec_deg"):
        np.testing.assert_array_equal(getattr(catalog, key), built["arrays"][key])
    np.testing.assert_allclose(catalog.xyz, built["arrays"]["xyz"], rtol=0.0, atol=1e-12)


def test_delivered_cache_load_time():
    path = default_cache_path()
    if not path.is_file():
        pytest.skip("先运行 scripts/build_catalog_cache.py")
    started = time.perf_counter()
    loaded = StarCatalog.from_cache(path)
    _ = loaded.tree
    elapsed = time.perf_counter() - started
    assert elapsed < 1.0, f"from_cache + tree took {elapsed:.3f} s"


def test_catalog_coordinates_are_unique_unit_vectors(catalog):
    assert len(catalog) == 1_051_966 - 72
    norm = np.linalg.norm(catalog.xyz, axis=1)
    assert np.max(np.abs(norm - 1.0)) < 1e-12
    assert np.isfinite(catalog.ra_deg).all()
    assert np.isfinite(catalog.dec_deg).all()
    assert catalog.dec_deg.min() >= -90.0
    assert catalog.dec_deg.max() <= 90.0
    assert catalog.ra_deg.min() >= 0.0
    assert catalog.ra_deg.max() < 360.0


def test_pleiades_cone_finds_alcyone(catalog):
    idx = catalog.cone_query(PLEIADES_RA_DEG, PLEIADES_DEC_DEG, 1.0)
    hit = np.flatnonzero((catalog.tyc1[idx] == 1800) & (catalog.tyc2[idx] == 2202))
    assert hit.size == 1
    star = idx[hit[0]]
    assert catalog.vmag[star] == pytest.approx(2.87, abs=0.05)
    separation = _separation_arcsec(
        catalog.ra_deg[star],
        catalog.dec_deg[star],
        ALCYONE_RA_DEG,
        ALCYONE_DEC_DEG,
    )
    assert separation < 1.0
    assert np.count_nonzero(catalog.vmag[idx] < 5.0) >= 5

    center = icrs_unit_vector(PLEIADES_RA_DEG, PLEIADES_DEC_DEG).reshape(3)
    cosine = np.clip(catalog.xyz[idx] @ center, -1.0, 1.0)
    assert np.rad2deg(np.arccos(cosine)).max() <= 1.0 + 1e-8


def test_sequence_union_covers_each_frame_cone(catalog):
    ra = np.array([PLEIADES_RA_DEG, 57.2])
    dec = np.array([PLEIADES_DEC_DEG, 24.3])
    first = catalog.cone_query(ra[0], dec[0], 1.0)
    second = catalog.cone_query(ra[1], dec[1], 1.0)
    union = catalog.query_sequence_union(ra, dec, 1.0)
    assert set(first.tolist()).issubset(set(union.tolist()))
    assert set(second.tolist()).issubset(set(union.tolist()))


def test_search_radius_and_negative_radius(catalog):
    assert search_radius_deg(4.97, 164.0, 6.188) == pytest.approx(
        4.97 + 164.0 * 6.188 / 3600.0
    )
    with pytest.raises(ValueError):
        search_radius_deg(4.97, -1.0, 6.188)
    with pytest.raises(ValueError):
        catalog.cone_query(0.0, 0.0, -0.1)


def test_cache_crc_corruption_rejected(tmp_path):
    arrays, meta = _arrays_and_meta(_synthetic_table(1000))
    path = tmp_path / "small.npz"
    save_cache(path, arrays, meta)
    _flip_npy_payload_byte(path, "dec_deg.npy")
    with pytest.raises(zipfile.BadZipFile):
        StarCatalog.from_cache(path)


def test_cache_content_corruption_rejected(tmp_path):
    table = _synthetic_table(1000)
    arrays, meta = _arrays_and_meta(table)
    arrays["dec_deg"] = arrays["dec_deg"].copy()
    arrays["dec_deg"][257:266] = 0.0
    path = tmp_path / "bad_dec.npz"
    _write_unchecked(path, arrays, meta)
    with pytest.raises(ValueError, match=r"首个下标 257") as caught:
        StarCatalog.from_cache(path)
    assert "9 行" in str(caught.value)
    assert "rad" in str(caught.value)


def test_cache_nan_rejected(tmp_path):
    arrays, meta = _arrays_and_meta(_synthetic_table(32))
    arrays["vmag"] = arrays["vmag"].copy()
    arrays["vmag"][4] = np.nan
    path = tmp_path / "nan.npz"
    _write_unchecked(path, arrays, meta)
    with pytest.raises(ValueError, match=r"vmag 含非有限值：1 行，首个下标 4"):
        StarCatalog.from_cache(path)


def test_cache_keys_and_version(tmp_path):
    arrays, meta = _arrays_and_meta(_synthetic_table(8))
    missing = tmp_path / "missing.npz"
    _write_unchecked(missing, {key: arrays[key] for key in CACHE_KEYS if key != "dec_deg"}, meta)
    with pytest.raises(ValueError, match="dec_deg"):
        load_cache(missing)

    extra = tmp_path / "extra.npz"
    extra_arrays = dict(arrays)
    extra_arrays["bonus"] = np.zeros(8, dtype=np.float64)
    _write_unchecked(extra, extra_arrays, meta)
    with pytest.raises(ValueError, match="bonus"):
        load_cache(extra)

    wrong = tmp_path / "version.npz"
    bad_meta = dict(meta)
    bad_meta["format_version"] = 99
    _write_unchecked(wrong, arrays, bad_meta)
    with pytest.raises(ValueError, match="build_catalog_cache.py"):
        load_cache(wrong)


def test_save_cache_atomic(tmp_path, monkeypatch):
    arrays, meta = _arrays_and_meta(_synthetic_table(16))
    path = tmp_path / "catalog.npz"
    save_cache(path, arrays, meta)
    original = path.read_bytes()

    def refuse_replace(*_args, **_kwargs):
        raise PermissionError("busy")

    monkeypatch.setattr("starsim.catalog.os.replace", refuse_replace)
    with pytest.raises(PermissionError):
        save_cache(path, arrays, meta)
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_tycho_mat_preconditions():
    bad_component = _synthetic_table(4)
    bad_component[2, 2] = 3
    with pytest.raises(ValueError, match="TYC3"):
        _check_tycho_table(bad_component)
    fractional = _synthetic_table(4)
    fractional[1, 0] = 1.5
    with pytest.raises(ValueError, match="TYC1"):
        _check_tycho_table(fractional)
    nan_tyc1 = _synthetic_table(4)
    nan_tyc1[0, 0] = np.nan
    with pytest.raises(ValueError, match="TYC1"):
        _check_tycho_table(nan_tyc1)
    zero_tyc2 = _synthetic_table(4)
    zero_tyc2[1, 1] = 0
    with pytest.raises(ValueError, match="TYC2"):
        _check_tycho_table(zero_tyc2)
    negative = _synthetic_table(4)
    negative[2, 0] = -1
    with pytest.raises(ValueError, match="TYC1"):
        _check_tycho_table(negative)
    overflow = _synthetic_table(4)
    overflow[3, 0] = float(2**31)
    with pytest.raises(ValueError, match="TYC1"):
        _check_tycho_table(overflow)


def test_save_cache_is_byte_identical(tmp_path):
    arrays, meta = _arrays_and_meta(_synthetic_table(32))
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    save_cache(first, arrays, meta)
    save_cache(second, arrays, meta)
    assert first.read_bytes() == second.read_bytes()


def test_catalog_meta_is_a_snapshot_of_the_caller_dict():
    arrays, meta = _arrays_and_meta(_synthetic_table(8))
    catalog = StarCatalog(arrays, meta)
    meta["epoch"] = "J2000"
    meta["dropped_ids"].append([1, 1])
    assert catalog.meta is not meta
    assert catalog.meta["epoch"] == "J1991.25"
    assert catalog.meta["dropped_ids"] == []


def test_catalog_columns_are_private_readonly_copies():
    arrays, meta = _arrays_and_meta(_synthetic_table(16))
    catalog = StarCatalog(arrays, meta)
    for key in CACHE_KEYS:
        column = getattr(catalog, key)
        assert not np.shares_memory(column, arrays[key])
        assert column.flags.writeable is False
        with pytest.raises(ValueError):
            column.flat[0] = column.flat[0]
    original = float(catalog.vmag[0])
    arrays["vmag"][0] = 12345.0
    assert catalog.vmag[0] == original
    found = catalog.cone_query(float(catalog.ra_deg[3]), float(catalog.dec_deg[3]), 2.0)
    assert 3 in set(found.tolist())


def test_from_mat_and_from_cache_share_meta():
    cache = default_cache_path()
    mat = default_mat_path()
    if not cache.is_file() or not mat.is_file():
        pytest.skip("先运行 scripts/build_catalog_cache.py")
    assert StarCatalog.from_mat(mat).meta == StarCatalog.from_cache(cache).meta


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (lambda meta: meta.__setitem__("n_source_rows", meta["n_source_rows"] + 1), "n_source_rows"),
        (lambda meta: meta.__setitem__("dropped_ids", [[1, 1]]), "dropped_ids"),
        (lambda meta: meta.__setitem__("columns", ["tyc1"]), "columns"),
        (lambda meta: meta.__setitem__("frame", "FK5"), "frame"),
        (lambda meta: meta.__setitem__("epoch", "J2000"), "epoch"),
        (lambda meta: meta.__setitem__("n_dropped", 1), "n_dropped"),
    ),
)
def test_meta_must_agree_with_the_rows(mutate, match):
    arrays, meta = _arrays_and_meta(_synthetic_table(8))
    bad = dict(meta)
    mutate(bad)
    with pytest.raises(ValueError, match=match):
        StarCatalog(arrays, bad)


def test_orphan_temps_older_than_an_hour_are_deleted(tmp_path, caplog):
    out = tmp_path / "tycho1.npz"
    old = tmp_path / "tycho1.npz.tmp-1"
    recent = tmp_path / "tycho1.npz.tmp-2"
    other = tmp_path / "other.tmp-3"
    old.write_bytes(b"old")
    recent.write_bytes(b"new")
    other.write_bytes(b"keep")
    past = time.time() - 7200
    os.utime(old, (past, past))
    with caplog.at_level(logging.INFO, logger="starsim.catalog"):
        cleanup_orphan_temps(out)
    assert not old.exists()
    assert recent.exists()
    assert other.exists()
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "删除过期临时文件" in text
    assert "可能有并发构建" in text
