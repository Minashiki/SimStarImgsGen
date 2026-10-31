"""Named streams are the only random source, and repeats are bit-exact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from starsim.rng import (
    RNG_SCHEME_VERSION,
    SequenceRng,
    assert_distinct,
    name_key,
    root_seed,
    sequence_seed,
    stream,
)

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from rng_scan import scan_source

ROOT = Path(__file__).resolve().parents[1]
_BYPASSES = (
    "import numpy as np\nnp.random.Generator(np.random.PCG64(42))\n",
    "import numpy as np\nnp.random.SeedSequence(42)\n",
    "import numpy as np\nnp.random.standard_normal(4)\n",
    "import numpy as np\nnp.random.exponential(1.0)\n",
    "import random\nrandom.random()\n",
    "import scipy.stats\nscipy.stats.norm.rvs()\n",
    "import numpy.random as npr\nnpr.normal()\n",
    "import numpy\nnumpy.random.normal()\n",
    "from numpy.random import normal\nnormal()\n",
    "from numpy import random\nrandom.normal()\n",
    'import numpy as np\ngetattr(np, "random")\n',
    "from scipy import stats\nstats.norm.rvs()\n",
    "import numpy.linalg\nnumpy.random.normal()\n",
    "from numpy import *\nrandom.normal()\n",
    "import numpy as np\nxp = np\nxp.random.normal()\n",
    "import numpy as np\nxp = np\nyp = xp\nyp.random.normal()\n",
    'import importlib\nimportlib.import_module("numpy.random")\n',
    '__import__("random")\n',
    'import importlib\nimportlib.import_module("numpy").random\n',
    'from importlib import import_module\nimport_module("numpy.random")\n',
    "import numpy as np\nxp: object = np\nxp.random.normal()\n",
    "import scipy\nscipy.stats.norm.sf()\n",
    "import scipy as sp\nsp.stats.bootstrap()\n",
    "import scipy.io\nscipy.stats.norm.sf()\n",
    'import scipy as sp\ngetattr(sp, "stats")\n',
    'from importlib import import_module\nimport_module(name="numpy.random")\n',
    '__import__(name="random")\n',
    'from importlib import import_module\nxp = import_module("numpy")\nxp.random.normal()\n',
    'xp = __import__("numpy.linalg")\nxp.random.normal()\n',
    "import numpy as np\na = b = np\nb.random.normal()\n",
    '__import__("scipy.io").stats\n',
    'from importlib import import_module\nimport_module("scipy").stats\n',
    '__import__("numpy.linalg", fromlist=[]).random\n',
    '__import__("numpy.linalg", fromlist=()).random\n',
    '__import__("numpy.linalg", fromlist=None).random\n',
    '__import__("numpy.linalg", None, None, []).random\n',
    "from starsim.rng import stream\nstream(seq, 'motion')\n",
    "from starsim.rng import stream as draw\ndraw(seq, 'motion')\n",
    "import starsim.rng as r\nr.stream(seq, 'motion')\n",
    "from starsim import rng as r\nr.stream(seq, 'motion')\n",
    "import starsim.rng\nstarsim.rng.stream(seq, 'motion')\n",
)
_STARSIM_BYPASSES = (
    "import os\nos.urandom(8)\n",
    "import os as o\no.urandom(8)\n",
    "from os import urandom\n",
    "import uuid\nuuid.uuid4()\n",
    "import uuid as u\nu.uuid4()\n",
    "from uuid import uuid1\n",
    "import secrets\n",
    "from secrets import token_bytes\n",
    '__import__("secrets")\n',
    'hash("x")\n',
    "class C:\n    def __hash__(self):\n        return hash(self.name)\n",
)
_NOT_RNG = (
    "import numpy.linalg as la\nla.norm(x)\n",
    "from scipy.special import ndtr\nndtr(0.0)\n",
    "def draw(rng):\n    return rng.random(5)\n",
    'from importlib import import_module\nimport_module("numpy.linalg").norm\n',
    'from importlib import import_module\nimport_module("numpy.linalg").random\n',
    '__import__("numpy.linalg", fromlist=["norm"]).random\n',
    '__import__("numpy.linalg", None, None, ["norm"]).random\n',
    "def stream(seq, name):\n    return seq\nstream(seq, name)\n",
    "r = None\nr.stream()\n",
    "import scipy.io\nscipy.io.loadmat(p)\n",
)
_STARSIM_ALLOWED = (
    "import os\nos.replace(a, b)\nos.getpid()\n",
    'import uuid\nuuid.uuid5(uuid.NAMESPACE_DNS, "x")\n',
    "class C:\n    def __hash__(self):\n        return 1\n",
    "def f(obj):\n    return obj.hash()\n",
    "import hashlib\nhashlib.sha256(b'x')\n",
)
_ALLOWED = '''
"""np.random.normal lives in this docstring."""
import numpy as np

def draw(rng: np.random.Generator) -> np.random.Generator:
    # np.random.normal(0, 1)
    note = "np.random.normal"
    if isinstance(rng, np.random.SeedSequence):
        return rng
    return rng
'''


def test_named_stream_is_bit_exact_and_separated():
    root = np.random.SeedSequence(20261027)
    first = sequence_seed(root, 0)
    again = sequence_seed(root, 0)
    draws = lambda seq, name: stream(seq, name).uniform(0.0, 1.0, size=8)
    np.testing.assert_array_equal(draws(first, "config/psf.sigma_px"), draws(again, "config/psf.sigma_px"))
    assert not np.array_equal(draws(first, "config/psf.sigma_px"), draws(first, "motion"))
    other = sequence_seed(root, 1)
    assert not np.array_equal(draws(first, "config/psf.sigma_px"), draws(other, "config/psf.sigma_px"))


def test_assert_distinct_lists_colliding_names(monkeypatch):
    monkeypatch.setattr("starsim.rng.name_key", lambda name: 1)
    with pytest.raises(ValueError, match=r"alpha") as caught:
        assert_distinct(["alpha", "beta"])
    assert "beta" in str(caught.value)
    assert name_key("config/psf.sigma_px") != name_key("motion")


def test_stream_rejects_bad_arguments():
    root = np.random.SeedSequence(1)
    with pytest.raises(TypeError):
        stream("not-a-seed", "motion")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        stream(root, 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        stream(root, "")
    with pytest.raises(ValueError):
        sequence_seed(root, -1)
    with pytest.raises(TypeError):
        sequence_seed(root, True)  # type: ignore[arg-type]


def test_root_seed_rejects_bool_and_negative():
    with pytest.raises(TypeError):
        root_seed(True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        root_seed(-1)
    assert isinstance(root_seed(0), np.random.SeedSequence)


def test_sequence_rng_rejects_a_second_issue_and_hash_collisions(monkeypatch):
    rngs = SequenceRng(sequence_seed(root_seed(1), 0))
    rngs.get("noise")
    with pytest.raises(ValueError, match="已发放过一次"):
        rngs.get("noise")

    monkeypatch.setattr("starsim.rng.name_key", lambda name: 7)
    collided = SequenceRng(sequence_seed(root_seed(1), 1))
    collided.get("alpha")
    with pytest.raises(ValueError, match="alpha") as caught:
        collided.get("beta")
    assert "beta" in str(caught.value)


def test_sequence_rng_issued_order_and_metadata_round_trip():
    rngs = SequenceRng(sequence_seed(root_seed(20261027), 3))
    first = rngs.get("motion").normal(size=16)
    rngs.get("noise/hot_pixels")
    assert rngs.issued == ("motion", "noise/hot_pixels")
    meta = json.loads(json.dumps(rngs.metadata()))
    assert set(meta) == {"rng_scheme", "entropy", "spawn_key", "pool_size", "issued"}
    assert meta["rng_scheme"] == RNG_SCHEME_VERSION
    assert isinstance(meta["entropy"], str)
    assert isinstance(meta["pool_size"], int)
    assert meta["issued"] == ["motion", "noise/hot_pixels"]
    restored = SequenceRng.from_metadata(meta)
    assert restored.issued == ()
    np.testing.assert_array_equal(first, restored.get("motion").normal(size=16))


def _valid_metadata() -> dict:
    rngs = SequenceRng(sequence_seed(root_seed(1), 0))
    rngs.get("motion")
    return json.loads(json.dumps(rngs.metadata()))


_BAD_METADATA = (
    ("missing rng_scheme", lambda meta: meta.pop("rng_scheme")),
    ("rng_scheme 2", lambda meta: meta.__setitem__("rng_scheme", 2)),
    ("entropy +5", lambda meta: meta.__setitem__("entropy", "+5")),
    ("entropy space", lambda meta: meta.__setitem__("entropy", " 5")),
    ("entropy underscore", lambda meta: meta.__setitem__("entropy", "1_000")),
    ("entropy negative", lambda meta: meta.__setitem__("entropy", "-1")),
    ("entropy int", lambda meta: meta.__setitem__("entropy", 5)),
    ("spawn_key negative", lambda meta: meta.__setitem__("spawn_key", [-1])),
    ("spawn_key float", lambda meta: meta.__setitem__("spawn_key", [1.0])),
    ("spawn_key bool", lambda meta: meta.__setitem__("spawn_key", [True])),
    ("pool_size 3", lambda meta: meta.__setitem__("pool_size", 3)),
    ("missing pool_size", lambda meta: meta.pop("pool_size")),
    ("extra key", lambda meta: meta.__setitem__("extra", 1)),
    ("issued not a list", lambda meta: meta.__setitem__("issued", "motion")),
    ("spawn_key extra", lambda meta: meta.__setitem__("spawn_key", [*meta["spawn_key"], 0])),
    ("spawn_key wrong domain", lambda meta: meta.__setitem__("spawn_key", [0, meta["spawn_key"][1]])),
    ("spawn_key length 1", lambda meta: meta.__setitem__("spawn_key", meta["spawn_key"][:1])),
    ("spawn_key bool index", lambda meta: meta.__setitem__("spawn_key", [meta["spawn_key"][0], True])),
    ("spawn_key negative index", lambda meta: meta.__setitem__("spawn_key", [meta["spawn_key"][0], -1])),
    ("integer key", lambda meta: meta.__setitem__(1, 0)),
    ("issued empty", lambda meta: meta.__setitem__("issued", [""])),
    ("issued duplicate", lambda meta: meta.__setitem__("issued", ["motion", "motion"])),
)


@pytest.mark.parametrize(("label", "mutate"), _BAD_METADATA, ids=[item[0] for item in _BAD_METADATA])
def test_from_metadata_rejects_invalid_input(label, mutate):
    meta = _valid_metadata()
    mutate(meta)
    with pytest.raises(ValueError, match="无法从 metadata 重建序列"):
        SequenceRng.from_metadata(meta)


@pytest.mark.parametrize("source", _BYPASSES)
def test_scan_rejects_rng_bypasses(source):
    assert scan_source(source, "sample.py")


@pytest.mark.parametrize("source", _STARSIM_BYPASSES)
def test_scan_rejects_starsim_only_bypasses(source):
    assert scan_source(source, "starsim/sample.py")
    assert scan_source(source, "scripts/sample.py") == []


@pytest.mark.parametrize("source", _NOT_RNG)
def test_scan_allows_unrelated_imports_and_generator_methods(source):
    assert scan_source(source, "sample.py") == []


@pytest.mark.parametrize("source", _STARSIM_ALLOWED)
def test_scan_allows_non_rng_uses_inside_starsim(source):
    assert scan_source(source, "starsim/sample.py") == []


def test_scan_allows_annotations_isinstance_and_comments():
    assert scan_source(_ALLOWED, "sample.py") == []


_RELATIVE_STREAM_CALLS = (
    ("starsim/x.py", "from .rng import stream\nstream(seq, 'motion')\n", "relative stream"),
    ("starsim/x.py", "from .rng import stream as s\ns(seq, 'motion')\n", "relative alias"),
    ("starsim/x.py", "from . import rng\nrng.stream(seq, 'motion')\n", "relative module"),
    ("starsim/x.py", "from . import rng as r\nr.stream(seq, 'motion')\n", "relative module alias"),
    ("starsim/__init__.py", "from .rng import stream\nstream(seq, 'motion')\n", "init relative"),
    ("starsim/sub/x.py", "from ..rng import stream\nstream(seq, 'motion')\n", "parent relative"),
)
_NOT_TARGET_STREAM = (
    ("starsim/x.py", "from ..rng import stream\nstream(seq, 'motion')\n", "above top-level"),
    ("starsim/x.py", "from .starsim.rng import stream\nstream(seq, 'motion')\n", "nested name"),
    ("scripts/g.py", "from .rng import stream\nstream(seq, 'motion')\n", "scripts relative"),
)
_STAR_STREAM = (
    ("starsim/x.py", "from starsim.rng import *\n", "absolute star"),
    ("starsim/x.py", "from .rng import *\n", "relative star"),
    ("scripts/g.py", "from starsim.rng import *\n", "scripts star"),
)


@pytest.mark.parametrize(
    ("filename", "source"),
    [item[:2] for item in _RELATIVE_STREAM_CALLS],
    ids=[item[2] for item in _RELATIVE_STREAM_CALLS],
)
def test_scan_rejects_relative_stream_calls(filename, source):
    assert scan_source(source, filename) == [f"{filename}:2: 调用 starsim.rng.stream"]


@pytest.mark.parametrize(
    ("filename", "source"),
    [item[:2] for item in _NOT_TARGET_STREAM],
    ids=[item[2] for item in _NOT_TARGET_STREAM],
)
def test_scan_ignores_imports_that_do_not_resolve_to_rng(filename, source):
    assert scan_source(source, filename) == []


@pytest.mark.parametrize(
    ("filename", "source"),
    [item[:2] for item in _STAR_STREAM],
    ids=[item[2] for item in _STAR_STREAM],
)
def test_scan_rejects_star_import_of_rng_on_the_import_line(filename, source):
    assert scan_source(source, filename) == [f"{filename}:1: 星号导入 starsim.rng"]


def test_package_does_not_export_stream():
    import starsim

    assert "stream" not in starsim.__all__
    assert not hasattr(starsim, "stream")


def test_scan_exemption_is_the_rng_module_path():
    text = "import random\n"
    assert scan_source(text, "starsim/rng.py") == []
    assert scan_source(text, "scripts/rng.py")
    assert scan_source(text, "starsim/pkg/rng.py")
    assert scan_source(text, "scripts/foo/starsim/rng.py")
    assert scan_source(text, (ROOT / "starsim" / "rng.py").as_posix())


def test_project_code_does_not_draw_random_numbers_outside_rng():
    offenders: list[str] = []
    for folder in ("starsim", "scripts"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            offenders.extend(scan_source(path.read_text(encoding="utf-8"), relative))
    assert offenders == []
