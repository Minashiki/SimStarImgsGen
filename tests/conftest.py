import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.runtime_env import ENV_HINT, environment_problems


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: long-running checks; collected and run by default",
    )


def pytest_sessionstart(session):
    """Stop the session unless this interpreter is the genstarimg pin.

    Base (Python 3.12.7, numpy 1.26.4, numba 0.60) and an un-activated
    ``envs\\genstarimg\\python.exe`` are both the wrong way to run these tests.
    """
    problems = environment_problems()
    if problems:
        detail = "\n".join(problems)
        pytest.exit(f"{ENV_HINT}\n{detail}", returncode=3)

