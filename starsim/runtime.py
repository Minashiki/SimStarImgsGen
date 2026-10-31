"""Runtime provenance shared by reports and baseline checks."""

import sys


def runtime_environment() -> dict[str, str]:
    """Package versions and the host CPU name."""
    import llvmlite
    import llvmlite.binding
    import numba
    import numpy
    import scipy

    return {
        "python": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "numpy": numpy.__version__,
        "numba": numba.__version__,
        "llvmlite": llvmlite.__version__,
        "scipy": scipy.__version__,
        "cpu": llvmlite.binding.get_host_cpu_name(),
    }


