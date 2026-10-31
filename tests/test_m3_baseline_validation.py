"""Negative checks for baseline provenance, completeness, and freeze failures."""

import copy
import json
import sys

import pytest

from tests import freeze_m3_baseline as baseline


@pytest.fixture
def comparison(monkeypatch, tmp_path):
    manifest = json.loads(baseline.OUT.read_text(encoding="utf-8"))
    group = manifest["groups"]["small"]
    records = {item["index"]: copy.deepcopy(item) for item in group["sequences"]}
    loaded = baseline._load_quiet()
    monkeypatch.setattr(baseline, "_load_quiet", lambda: loaded)
    monkeypatch.setattr(baseline.StarCatalog, "from_cache", lambda path: None)
    catalog_identity = copy.deepcopy(group["catalog"])
    monkeypatch.setattr(baseline, "catalog_identity", lambda catalog: catalog_identity)
    yaml_hashes = {
        "camera_default.yaml": group["camera_yaml_sha256"],
        "generator_default.yaml": group["generator_yaml_sha256"],
    }
    monkeypatch.setattr(baseline, "sha256_file", lambda path: yaml_hashes[path.name])
    calls = []

    def run_one(loaded, catalog, derived, index):
        calls.append(index)
        return copy.deepcopy(records[index])

    monkeypatch.setattr(baseline, "run_one", run_one)
    monkeypatch.setattr(baseline, "OUT", tmp_path / "manifest.json")
    return manifest, records, calls


def write_manifest(manifest):
    baseline.OUT.write_text(json.dumps(manifest), encoding="utf-8")


def test_unchanged_baseline_is_accepted(comparison):
    manifest, records, calls = comparison
    write_manifest(manifest)
    assert baseline.baseline_problems("small") == []
    assert calls == [0, 1]


@pytest.mark.parametrize("field,reason", [
    ("environment", "environment"),
    ("covered_modules", "covered_modules"),
    ("algorithm_versions", "algorithm_versions"),
    ("camera_yaml_sha256", "camera_yaml"),
    ("generator_yaml_sha256", "generator_yaml"),
    ("catalog", "catalog"),
    ("root_seed", "root_seed"),
    ("side", "side"),
])
@pytest.mark.parametrize("missing", [False, True])
def test_provenance_failure_stops_before_generation(comparison, field, reason, missing):
    manifest, records, calls = comparison
    group = manifest["groups"]["small"]
    if missing:
        del group[field]
    else:
        group[field] = "different"
    write_manifest(manifest)
    assert baseline.baseline_problems("small") == [f"preflight: {reason}"]
    assert calls == []


@pytest.mark.parametrize("damage,expected", [
    ("generated_frames", "seq 0 frame count got 0 manifest 10 != 10"),
    ("manifest_frames", "seq 0 frame count got 10 manifest 9 != 10"),
    ("sequence", "seq 1 missing"),
    ("frame_field", "seq 0 frame 0 fields"),
    ("population_field", "seq 0 population fields"),
    ("sensor", "seq 0 sensor fields"),
    ("background", "seq 0 sensor background_e"),
    ("n_frames", "small n_frames 9 != 10"),
    ("effective", "seq 0 effective"),
])
def test_incomplete_or_changed_baseline_is_rejected(comparison, damage, expected):
    manifest, records, calls = comparison
    group = manifest["groups"]["small"]
    record = group["sequences"][0]
    if damage == "generated_frames":
        records[0]["frames"] = []
    elif damage == "manifest_frames":
        record["frames"].pop()
    elif damage == "sequence":
        group["sequences"].pop()
    elif damage == "frame_field":
        del record["frames"][0]["snr"]
    elif damage == "population_field":
        del record["population"]["track_id"]
    elif damage == "sensor":
        del record["sensor"]
    elif damage == "background":
        records[0]["sensor"]["background_e"] = "different"
    elif damage == "n_frames":
        group["n_frames"] = 9
    elif damage == "effective":
        record["effective"]["targets"]["per_frame"] = -1
    write_manifest(manifest)
    assert expected in baseline.baseline_problems("small")


def test_single_group_freeze_preserves_file_when_other_group_fails(comparison, monkeypatch):
    manifest, records, calls = comparison
    write_manifest(manifest)
    before = baseline.OUT.read_bytes()
    checked = []

    def reject(group):
        checked.append(group)
        return ["preflight: environment"]

    monkeypatch.setattr(baseline, "baseline_problems", reject)
    monkeypatch.setattr(baseline, "_build_group", lambda *args: pytest.fail("must not freeze"))
    monkeypatch.setattr(sys, "argv", ["freeze_m3_baseline.py", "--group", "small"])
    with pytest.raises(SystemExit, match="other group failed"):
        baseline.main()
    assert checked == ["default"]
    assert baseline.OUT.read_bytes() == before


def test_all_group_freeze_preserves_file_when_second_group_fails(comparison, monkeypatch):
    manifest, records, calls = comparison
    write_manifest(manifest)
    before = baseline.OUT.read_bytes()
    built = []

    def build(loaded, catalog, derived, indices, side, provenance):
        built.append(indices)
        if indices == baseline.GROUP_INDICES["default"]:
            raise RuntimeError("second group failed")
        return {}

    monkeypatch.setattr(baseline, "_build_group", build)
    monkeypatch.setattr(sys, "argv", ["freeze_m3_baseline.py", "--group", "all"])
    with pytest.raises(RuntimeError, match="second group failed"):
        baseline.main()
    assert built == [baseline.GROUP_INDICES["small"], baseline.GROUP_INDICES["default"]]
    assert baseline.OUT.read_bytes() == before


def test_geometry_checker_rejects_joint_length_and_fraction_corruption(monkeypatch):
    from tests import test_m3_targets as targets

    sample = targets.sample_targets

    def corrupt(*args, **kwargs):
        population = sample(*args, **kwargs)
        population.L_native *= 2.0
        population.vis_frac /= 2.0
        return population

    monkeypatch.setattr(targets, "sample_targets", corrupt)
    record = targets._population_record((0, 6, 400, 5))
    assert any("L_native" in problem for problem in record["problems"])


@pytest.mark.parametrize("damage", ["extra", "missing", "duplicate", "order", "mapping"])
def test_invalid_coverage_stops_generation(comparison, damage):
    manifest, _, calls = comparison
    covered = manifest["groups"]["small"]["covered_modules"]
    if damage == "extra":
        covered.append("labels")
    elif damage == "missing":
        covered.pop()
    elif damage == "duplicate":
        covered.append(covered[0])
    elif damage == "order":
        covered.reverse()
    else:
        manifest["groups"]["small"]["covered_modules"] = dict.fromkeys(covered)
    write_manifest(manifest)
    assert baseline.baseline_problems("small") == ["preflight: covered_modules"]
    assert calls == []


@pytest.mark.parametrize("value", [999, True, 1.0])
def test_covered_version_is_strict(comparison, value):
    manifest, _, calls = comparison
    manifest["groups"]["small"]["algorithm_versions"]["psf"] = value
    write_manifest(manifest)
    assert baseline.baseline_problems("small") == ["preflight: algorithm_versions"]
    assert calls == []


def test_uncovered_module_is_accepted(comparison, monkeypatch):
    manifest, _, calls = comparison
    current = baseline.algorithm_versions()
    monkeypatch.setattr(baseline, "algorithm_versions", lambda: {**current, "labels": 1})
    write_manifest(manifest)
    assert baseline.baseline_problems("small") == []
    assert calls == [0, 1]


def test_missing_current_covered_module_is_rejected(comparison, monkeypatch):
    manifest, _, calls = comparison
    current = baseline.algorithm_versions()
    del current["psf"]
    monkeypatch.setattr(baseline, "algorithm_versions", lambda: current)
    write_manifest(manifest)
    assert baseline.baseline_problems("small") == ["preflight: algorithm_versions"]
    assert calls == []


def test_coverage_migration_preserves_sequences_and_rejects_repeat(comparison):
    from tests import migrate_m3_covered_modules as migration

    manifest, _, _ = comparison
    for group in manifest["groups"].values():
        del group["covered_modules"]
        del group["migrations"]
    before = copy.deepcopy(manifest)
    write_manifest(manifest)
    migration.main()
    after = json.loads(baseline.OUT.read_text(encoding="utf-8"))
    assert migration.sequence_hashes(before) == migration.sequence_hashes(after)
    for name, group in after["groups"].items():
        assert group.pop("covered_modules") == list(baseline.M3_COVERED_MODULES)
        assert group.pop("migrations") == ["add covered_modules"]
        assert group == before["groups"][name]
    contents = baseline.OUT.read_bytes()
    with pytest.raises(ValueError, match="already present"):
        migration.main()
    assert baseline.OUT.read_bytes() == contents
