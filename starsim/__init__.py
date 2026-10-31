"""Multi-frame star-field sequence simulator."""

from starsim.attitude import (
    boresight_radec,
    matrix_to_pointing_roll,
    matrix_to_quaternion,
    pointing_roll_to_matrix,
    quaternion_to_matrix,
)
from starsim.blend import BLEND_VERSION
from starsim.camera import Camera, homography, project, unproject, wcs_from_attitude
from starsim.catalog import StarCatalog
from starsim.config import (
    CONFIG_SAMPLING_VERSION,
    AppConfig,
    DerivedCamera,
    load_config,
    sample_int_inclusive,
    sample_log_uniform,
    sample_sequence_params,
    sample_uniform,
)
from starsim.motion import MOTION_ALGORITHM_VERSION, SequenceMotion, simulate_sequence
from starsim.noise import NOISE_VERSION
from starsim.photometry import PHOTOMETRY_VERSION
from starsim.psf import PSF_VERSION
from starsim.render import RENDER_VERSION
from starsim.rng import SequenceRng, root_seed, sequence_seed
from starsim.sequence import SEQUENCE_VERSION
from starsim.targets import TARGETS_VERSION


def algorithm_versions() -> dict:
    """Return a new mapping of the module versions M4 will record.

    Keys are the modules implemented so far: config sampling, motion,
    photometry, the PSF, rendering, noise, blending, targets, and the
    frame loop. ``rng_scheme`` is carried only by ``SequenceRng.metadata()``.
    Callers use this mapping directly; there is no nested
    ``algorithm_versions`` key.

    Constants stay in their modules. Each call builds a new dict, so a
    caller can modify the result without changing the next call.

    Only modules that exist today are included. Label and metadata
    versions join when those modules are implemented.

    This is an identifier, not per-module compatible replay. M4, not this
    function, will reject a record whose key set differs or whose values
    are not ``type is int`` and equal to the current versions. Missing keys
    and extra keys are both listed. Non-string keys are reported the same
    way as D33 metadata. A mismatch rejects the whole record.

    Records written before ``CONFIG_SAMPLING_VERSION`` existed are outside
    this replay promise.
    """
    return {
        "config_sampling": CONFIG_SAMPLING_VERSION,
        "motion": MOTION_ALGORITHM_VERSION,
        "photometry": PHOTOMETRY_VERSION,
        "psf": PSF_VERSION,
        "render": RENDER_VERSION,
        "noise": NOISE_VERSION,
        "blend": BLEND_VERSION,
        "targets": TARGETS_VERSION,
        "sequence": SEQUENCE_VERSION,
    }


__all__ = [
    "BLEND_VERSION",
    "CONFIG_SAMPLING_VERSION",
    "MOTION_ALGORITHM_VERSION",
    "NOISE_VERSION",
    "PHOTOMETRY_VERSION",
    "PSF_VERSION",
    "RENDER_VERSION",
    "SEQUENCE_VERSION",
    "TARGETS_VERSION",
    "AppConfig",
    "Camera",
    "DerivedCamera",
    "SequenceMotion",
    "SequenceRng",
    "StarCatalog",
    "algorithm_versions",
    "boresight_radec",
    "homography",
    "load_config",
    "matrix_to_pointing_roll",
    "matrix_to_quaternion",
    "pointing_roll_to_matrix",
    "project",
    "quaternion_to_matrix",
    "root_seed",
    "sample_int_inclusive",
    "sample_log_uniform",
    "sample_sequence_params",
    "sample_uniform",
    "sequence_seed",
    "simulate_sequence",
    "unproject",
    "wcs_from_attitude",
]
