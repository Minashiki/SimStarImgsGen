"""D12 runtime pin shared by the test session, freezes, and reports.

The acceptance environment is the ``genstarimg`` conda env. A direct launch
of ``envs\\genstarimg\\python.exe`` without activation is not this pin:
the DLL search path is incomplete and the process dies with ``0xc06d007f``.
"""

from __future__ import annotations

from starsim.runtime import runtime_environment

# Versions the session guard and the reports both use.
EXPECTED_VERSIONS = {
    "python": "3.12.0",
    "numpy": "2.5.4",
    "numba": "0.68.0",
    "llvmlite": "0.50.0",
    "scipy": "1.18.1",
}

ENV_HINT = "请用 genstarimg"


def environment_problems() -> list[str]:
    """Human-readable mismatches against :data:`EXPECTED_VERSIONS`.

    An import failure is itself a mismatch: the hint is still to use
    ``genstarimg``.
    """
    try:
        current = runtime_environment()
    except Exception as exc:
        return [f"无法读取运行环境（{type(exc).__name__}: {exc}）"]
    problems = []
    for name, expected in EXPECTED_VERSIONS.items():
        got = current.get(name)
        if got != expected:
            problems.append(f"{name}={got}，需要 {expected}")
    return problems


def require_genstarimg() -> None:
    """Abort the process when this interpreter is not the D12 pin."""
    problems = environment_problems()
    if not problems:
        return
    detail = "\n".join(problems)
    raise SystemExit(f"{ENV_HINT}\n{detail}")
