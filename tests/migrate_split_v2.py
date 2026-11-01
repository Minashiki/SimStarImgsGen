"""D78: split is outside M2 coverage; migrate only its version provenance."""
import json
from pathlib import Path

from starsim.io import write_json


def main():
    path = Path(__file__).resolve().parents[1]/'artifacts/m2_report.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    if any(section['algorithm_versions']['split'] != 1 for section in report['sections'].values()):
        raise ValueError('requires split 1 in every M2 section; refusing repeat')
    for section in report['sections'].values():
        section['algorithm_versions']['split'] = 2
        section.setdefault('migrations', []).append('D78: split 1→2; split is not used by M2; data unchanged')
    write_json(path, report)


if __name__ == '__main__':
    main()
