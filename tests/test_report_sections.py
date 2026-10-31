import json

from tests.report_sections import section_problems, write_section


def test_updating_blend_preserves_wide_psf(tmp_path):
    path = tmp_path / "report.json"
    write_section(path, "wide_psf", {"sigma": 1.5})
    before = json.dumps(json.loads(path.read_text())["sections"]["wide_psf"])
    write_section(path, "blend_rate", {"median": 0.01})
    after = json.loads(path.read_text())
    assert json.dumps(after["sections"]["wide_psf"]) == before
    assert section_problems(path, ("blend_rate", "wide_psf")) == []


def test_missing_and_stale_sections_are_reported(tmp_path):
    path = tmp_path / "report.json"
    write_section(path, "wide_psf", {})
    report = json.loads(path.read_text())
    report["sections"]["wide_psf"]["environment"]["python"] = "old"
    path.write_text(json.dumps(report))
    assert section_problems(path, ("blend_rate", "wide_psf")) == [
        "请在 genstarimg 里重新生成 blend_rate 段",
        "请在 genstarimg 里重新生成 wide_psf 段",
    ]
    write_section(path, "wide_psf", {})
    report = json.loads(path.read_text())
    report["sections"]["wide_psf"]["algorithm_versions"]["psf"] = 999
    path.write_text(json.dumps(report))
    assert section_problems(path, ("wide_psf",)) == ["请在 genstarimg 里重新生成 wide_psf 段"]
