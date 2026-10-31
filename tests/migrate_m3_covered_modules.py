"""Add coverage provenance without refreezing any M3 sequences."""

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests import freeze_m3_baseline as baseline
from tests.runtime_env import require_genstarimg


def sequence_hashes(manifest):
    return {
        name: [hashlib.sha256(json.dumps(seq, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
               for seq in group["sequences"]]
        for name, group in manifest["groups"].items()
    }


def main():
    require_genstarimg()
    manifest = json.loads(baseline.OUT.read_text(encoding="utf-8"))
    before = copy.deepcopy(manifest)
    hashes = sequence_hashes(manifest)
    for group in manifest["groups"].values():
        if "covered_modules" in group:
            raise ValueError("covered_modules already present; refusing a second migration")
        if "migrations" in group:
            raise ValueError("migrations already present")
        if set(group["algorithm_versions"]) != set(baseline.M3_COVERED_MODULES):
            raise ValueError("algorithm_versions does not cover exactly the M3 modules")
        group["covered_modules"] = list(baseline.M3_COVERED_MODULES)
        group["migrations"] = ["add covered_modules"]
    assert sequence_hashes(manifest) == hashes
    for name, group in manifest["groups"].items():
        assert {k: v for k, v in group.items() if k not in ("covered_modules", "migrations")} == before["groups"][name]
    baseline._atomic_write(manifest)


if __name__ == "__main__":
    main()
