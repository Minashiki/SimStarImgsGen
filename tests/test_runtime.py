from starsim.runtime import runtime_environment
from tests.runtime_env import runtime_environment as acceptance_environment


def test_runtime_environment_reexport():
    assert runtime_environment() == acceptance_environment()
    assert set(runtime_environment()) == {"python", "numpy", "numba", "llvmlite", "scipy", "cpu"}
