"""Keep independently generated report sections and their provenance."""

import json
import os
from pathlib import Path

from starsim import algorithm_versions
from starsim.runtime import runtime_environment


def write_section(path, name, data):
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    report = {"sections": report.get("sections", {})}
    report.setdefault("sections", {})[name] = {
        "environment": runtime_environment(),
        "algorithm_versions": algorithm_versions(),
        "data": data,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def section_problems(path, required):
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    sections = report.get("sections", {})
    return [f"请在 genstarimg 里重新生成 {name} 段" for name in required
            if name not in sections
            or sections[name].get("environment") != runtime_environment()
            or sections[name].get("algorithm_versions") != algorithm_versions()]
