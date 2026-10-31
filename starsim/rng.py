"""Named random streams for bit-exact dataset regeneration.

Production code obtains the dataset root from ``root_seed`` and named
generators from ``SequenceRng``. ``stream`` is the low-level constructor:
it starts over on every call, so other modules must not call it.
``sequence.json`` (M4) stores ``SequenceRng.metadata()``. Replay one
sequence with ``SequenceRng.from_metadata``; that is the only production
entry, and it rejects a bad record before any sample is drawn.

Stream names look like ``config/psf.sigma_px``. Later milestones use names
such as ``motion``, ``noise/hot_pixels``, and ``targets/...``.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

# 修改种子的派生结构才会改变已有种子的抽样结果，必须同时把版本号加一：
# 域标签（_STREAM_DOMAIN、_SEQUENCE_DOMAIN）、name_key 的哈希算法或截取方式、
# 位生成器（PCG64）、根种子结构。
# 只收紧 from_metadata 的入口校验时不升版本：合法记录的抽样结果不变。
RNG_SCHEME_VERSION = 1

# Separate key spaces so a sequence index cannot collide with a named stream.
_STREAM_DOMAIN = 0x5354_5245
_SEQUENCE_DOMAIN = 0x5345_5155
_METADATA_KEYS = ("rng_scheme", "entropy", "spawn_key", "pool_size", "issued")
_ABSENT = object()


def name_key(name: str) -> int:
    """First 8 bytes of SHA-256, as a little-endian unsigned 64-bit integer."""
    _require_name(name)
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def stream_seed(seq_ss: np.random.SeedSequence, name: str) -> np.random.SeedSequence:
    """Seed sequence for one named stream of ``seq_ss``."""
    _require_seed_sequence(seq_ss)
    return np.random.SeedSequence(
        seq_ss.entropy,
        spawn_key=tuple(seq_ss.spawn_key) + (_STREAM_DOMAIN, name_key(name)),
        pool_size=seq_ss.pool_size,
    )


def stream(seq_ss: np.random.SeedSequence, name: str) -> np.random.Generator:
    """PCG64 generator for ``name``. The bit generator is explicit and stable.

    This is a pure function: every call with the same arguments returns the
    same sequence from the beginning. Production code takes generators from
    ``SequenceRng.get`` so a name cannot be issued twice.
    """
    return np.random.Generator(np.random.PCG64(stream_seed(seq_ss, name)))


def root_seed(seed: int) -> np.random.SeedSequence:
    """Dataset root seed. ``generate.py`` must get its root from here.

    ``seed`` is a non-negative integer. Booleans are rejected.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be a non-negative integer, got {seed!r}")
    value = int(seed)
    if value < 0:
        raise ValueError(f"seed must be >= 0, got {value}")
    return np.random.SeedSequence(value)


class SequenceRng:
    """Issues each named stream of one sequence at most once.

    ``get`` records the name, rejects a second issue of that name, and
    rejects a 64-bit ``name_key`` collision with a name already issued.
    """

    def __init__(self, seq_ss: np.random.SeedSequence) -> None:
        _require_seed_sequence(seq_ss)
        self._seq_ss = seq_ss
        self._issued: list[str] = []
        self._keys: dict[int, str] = {}

    def get(self, name: str) -> np.random.Generator:
        """Return the generator for ``name``.

        A second call with the same name raises ``ValueError``. A different
        name whose ``name_key`` matches an issued name also raises, and the
        message lists both names.
        """
        _require_name(name)
        if name in self._issued:
            raise ValueError(f"随机流 '{name}' 已发放过一次")
        key = name_key(name)
        previous = self._keys.get(key)
        if previous is not None:
            raise ValueError(f"随机流名字哈希碰撞：[{previous}, {name}]")
        self._issued.append(name)
        self._keys[key] = name
        return stream(self._seq_ss, name)

    @property
    def issued(self) -> tuple[str, ...]:
        """Names in the order ``get`` issued them."""
        return tuple(self._issued)

    def metadata(self) -> dict:
        """JSON-safe seed description.

        ``rng_scheme`` is ``RNG_SCHEME_VERSION``. ``entropy`` is a decimal
        string: a random root can be 128 bits, and JSON numbers above 2**53
        are not exact. ``pool_size`` is part of the record because replay
        depends on it. ``spawn_key`` and ``issued`` are lists. ``issued`` is
        an audit trail; ``from_metadata`` does not re-issue those names.
        """
        return {
            "rng_scheme": RNG_SCHEME_VERSION,
            "entropy": str(self._seq_ss.entropy),
            "spawn_key": [int(item) for item in self._seq_ss.spawn_key],
            "pool_size": int(self._seq_ss.pool_size),
            "issued": list(self._issued),
        }

    @classmethod
    def from_metadata(cls, meta: object) -> SequenceRng:
        """从 ``metadata()`` 重建一个序列。这是生产代码重放单个序列的唯一入口。

        重放契约：只重放由 ``root_seed`` → ``sequence_seed`` 生产链路产生的记录。
        ``SequenceRng`` 构造器可以接受任意 ``SeedSequence``，但那样得到的记录
        不保证能通过 ``from_metadata``。

        版本号规则：只是收紧入口校验时，不升 ``RNG_SCHEME_VERSION``，因为合法记录的
        抽样结果没有变化。修改种子的派生结构（域标签、``name_key``、位生成器、
        根种子结构）时才必须升版本号。

        ``issued`` 只用于审计，不参与重建。不合法的输入在入口汇总后一次报
        ``ValueError``。新实例的已发放列表为空。
        """
        if not isinstance(meta, dict):
            raise ValueError(f"metadata 必须是 dict，得到 {type(meta).__name__}")
        problems: list[str] = []
        required = set(_METADATA_KEYS)
        missing = sorted((key for key in _METADATA_KEYS if key not in meta), key=repr)
        extra = sorted((key for key in meta if key not in required), key=repr)
        if missing:
            problems.append("缺少字段：" + "、".join(repr(key) for key in missing))
        if extra:
            problems.append("多余字段：" + "、".join(repr(key) for key in extra))
        scheme = meta.get("rng_scheme", _ABSENT)
        entropy = meta.get("entropy", _ABSENT)
        spawn_key = meta.get("spawn_key", _ABSENT)
        pool_size = meta.get("pool_size", _ABSENT)
        issued = meta.get("issued", _ABSENT)
        if scheme is not _ABSENT and (type(scheme) is not int or scheme != RNG_SCHEME_VERSION):
            problems.append(f"rng_scheme 必须是 {RNG_SCHEME_VERSION}，得到 {scheme!r}")
        if entropy is not _ABSENT and (
            not isinstance(entropy, str) or re.fullmatch(r"0|[1-9][0-9]*", entropy) is None
        ):
            problems.append(f"entropy 必须是十进制非负整数字符串，得到 {entropy!r}")
        if spawn_key is not _ABSENT and not _valid_spawn_key(spawn_key):
            problems.append(
                f"spawn_key 必须是长度为 2 的 list，[0] 等于 {_SEQUENCE_DOMAIN}，"
                f"[1] 是 >= 0 的 int，得到 {spawn_key!r}"
            )
        if pool_size is not _ABSENT and (type(pool_size) is not int or pool_size < 4):
            problems.append(f"pool_size 必须是 >= 4 的 int，得到 {pool_size!r}")
        if issued is not _ABSENT and not _valid_issued(issued):
            problems.append(f"issued 必须是由非空且不重复的 str 组成的 list，得到 {issued!r}")
        if problems:
            raise ValueError("无法从 metadata 重建序列：\n" + "\n".join(problems))
        return cls(
            np.random.SeedSequence(
                int(meta["entropy"]),
                spawn_key=tuple(meta["spawn_key"]),
                pool_size=meta["pool_size"],
            )
        )


def sequence_seed(root_ss: np.random.SeedSequence, index: int) -> np.random.SeedSequence:
    """Seed sequence for dataset sequence ``index``. Does not advance ``root_ss``."""
    _require_seed_sequence(root_ss)
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise TypeError(f"index must be a non-negative integer, got {index!r}")
    index = int(index)
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")
    return np.random.SeedSequence(
        root_ss.entropy,
        spawn_key=tuple(root_ss.spawn_key) + (_SEQUENCE_DOMAIN, index),
        pool_size=root_ss.pool_size,
    )


def assert_distinct(names) -> None:
    """Raise if any two names share a ``name_key``. The message lists the clash."""
    groups: dict[int, list[str]] = {}
    for name in names:
        groups.setdefault(name_key(name), []).append(name)
    conflicts = [pair for pair in groups.values() if len(pair) > 1]
    if conflicts:
        listed = "; ".join("[" + ", ".join(group) + "]" for group in conflicts)
        raise ValueError(f"stream names collide: {listed}")


def _valid_spawn_key(value: object) -> bool:
    """Production records are ``[_SEQUENCE_DOMAIN, index]`` from ``sequence_seed``."""
    if not isinstance(value, list) or len(value) != 2:
        return False
    head, index = value
    return type(head) is int and head == _SEQUENCE_DOMAIN and type(index) is int and index >= 0


def _valid_issued(value: object) -> bool:
    """Audit list: non-empty strings, no duplicates. Not used to rebuild the sequence."""
    if not isinstance(value, list):
        return False
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or item == "" or item in seen:
            return False
        seen.add(item)
    return True


def _require_seed_sequence(seq_ss: object) -> None:
    if not isinstance(seq_ss, np.random.SeedSequence):
        raise TypeError("seq_ss must be numpy.random.SeedSequence")


def _require_name(name: object) -> None:
    if not isinstance(name, str):
        raise TypeError(f"name must be a non-empty string, got {type(name).__name__}")
    if name == "":
        raise ValueError("name must be a non-empty string")
