"""Generate and independently validate evidence for the complete quick suite."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from starsim.source_scope import SOURCE_PATHS

ROOT = Path(__file__).resolve().parents[1]
COMMAND = ('-m', 'pytest', '-c', str(ROOT/'pytest.ini'), '--rootdir', str(ROOT),
           '-p', 'tests.evidence_plugin', '-m', 'not slow',
           '--basetemp', str(ROOT/'.tmp/pytest_evidence'))
SUMMARY = ROOT/'artifacts/m5_quick_tests.json'
JUNIT = ROOT/'artifacts/m5_quick_tests.xml'


class PytestFailure(ValueError):
    def __init__(self, code):
        super().__init__(f'pytest exited with {code}')
        self.code = code


def git(*args):
    return subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', *args],
                          cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_digest():
    names = sorted(set(git('ls-files', '--cached', '--others', '--exclude-standard',
                           '--', *SOURCE_PATHS).splitlines()))
    return hashlib.sha256(''.join(f'{name}\0{sha256(ROOT/name)}\n' for name in names).encode()).hexdigest()


def environment():
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    env.pop('PYTEST_ADDOPTS', None)
    env.pop('PYTEST_PLUGINS', None)
    return env


def invoke(*args):
    (ROOT/'.tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
        path = Path(directory)/'collection.json'
        result = subprocess.run([sys.executable, *COMMAND, *args], cwd=ROOT,
                                env=dict(environment(), STARSIM_COLLECTION=str(path)))
        collection = json.loads(path.read_text()) if path.exists() else None
    return result.returncode, collection


def collect():
    code, collection = invoke('--collect-only', '-q')
    if code:
        raise PytestFailure(code)
    if not collection or not collection['node_ids']:
        raise ValueError('quick collection failed or selected zero tests')
    if len(set(collection['node_ids'])) != len(collection['node_ids']):
        raise ValueError('duplicate collected node ID')
    return collection


def require_committed_sources():
    from starsim.metadata import git_provenance
    if git_provenance()['git_dirty'] is not False:
        raise ValueError('commit source changes before generating formal evidence')
    for name in ('pytest.ini', 'conftest.py'):
        if (ROOT/name).exists():
            git('ls-files', '--error-unmatch', '--', name)
            git('cat-file', '-e', f'HEAD:{name}')


def parse_junit(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    ids = []
    for case in cases:
        properties = case.findall('./properties/property[@name="node_id"]')
        if len(properties) != 1 or not properties[0].get('value'):
            raise ValueError('missing or duplicate node ID property')
        ids.append(properties[0].get('value'))
    counts = dict(tests=len(cases), failures=len(list(root.iter('failure'))),
                  errors=len(list(root.iter('error'))), skipped=len(list(root.iter('skipped'))))
    if not ids or len(set(ids)) != len(ids) or any(counts[k] for k in ('failures', 'errors', 'skipped')):
        raise ValueError('JUnit is empty, duplicated, failed, errored or skipped')
    for suite in root.iter('testsuite'):
        for key in counts:
            actual = len(suite.findall('testcase')) if key == 'tests' else len(suite.findall(f'testcase/{key[:-1] if key != "skipped" else key}'))
            if int(suite.get(key, '0')) != actual:
                raise ValueError('JUnit declared counts differ')
    return dict(counts, node_ids=sorted(ids))


def validate(quick):
    if quick['source_sha256'] != code_digest() or quick['junit_sha256'] != sha256(JUNIT):
        raise ValueError('test evidence source or JUnit hash differs')
    parsed = parse_junit(JUNIT)
    current = collect()
    if quick['collection'] != current or parsed != quick['junit'] or parsed['node_ids'] != sorted(current['node_ids']):
        raise ValueError('test evidence collection or JUnit identities differ')


def main():
    SUMMARY.unlink(missing_ok=True)
    require_committed_sources()
    before = code_digest()
    try:
        collection = collect()
    except PytestFailure as exc:
        return exc.code
    code, executed = invoke(f'--junitxml={JUNIT}')
    if code:
        return code
    parsed = parse_junit(JUNIT)
    if collection != executed or parsed['node_ids'] != sorted(collection['node_ids']) or before != code_digest():
        raise ValueError('collection, execution or source changed')
    quick = dict(source_sha256=before, junit_sha256=sha256(JUNIT), collection=collection,
                 junit=parsed, command=[sys.executable, *COMMAND])
    temporary = SUMMARY.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(quick, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, SUMMARY)
    return 0


if __name__ == '__main__':
    sys.exit(main())
