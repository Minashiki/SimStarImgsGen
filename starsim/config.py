"""Load, merge, and validate simulator configuration.

Derived camera quantities are computed from ``sensor`` and replace any
hand-written ``derived`` block. A hand-written ``fov_deg`` is only a
consistency check.
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from starsim.rng import SequenceRng, assert_distinct

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_PATH = PROJECT_ROOT / "configs" / "camera_default.yaml"
DEFAULT_GENERATOR_PATH = PROJECT_ROOT / "configs" / "generator_default.yaml"

# Relative |derived - expected| / |derived| above this emits a warning.
FOV_RELATIVE_TOLERANCE = 0.01

# Drawn once per sequence, each from its own named stream. Speed and
# acceleration stay intervals: motion.py redraws them inside rejection
# sampling and writes the accepted values back onto the effective copy.
_SEQUENCE_UNIFORM_PATHS: tuple[tuple[str, ...], ...] = (
    ("photometry", "mag_limit"),
    ("psf", "sigma_px"),
    ("psf", "field_growth"),
    ("psf", "halo_fraction"),
    ("background", "gradient_frac"),
    ("background", "smooth_field_frac"),
    ("background", "vignetting_corner"),
    ("sensor_noise", "read_e"),
    ("sensor_noise", "dark_e_per_s"),
    ("sensor_noise", "prnu"),
    ("sensor_noise", "gain_e_per_adu"),
    ("sensor_noise", "bias_adu"),
    ("sensor_noise", "full_well_e"),
    ("sensor_noise", "row_col_fpn_e"),
    ("artifacts", "hot_pixel_frac"),
    ("artifacts", "dead_pixel_frac"),
    ("artifacts", "cosmic_rays_per_frame"),
    ("camera_motion", "jitter_px"),
)
_SEQUENCE_INT_PATHS: tuple[tuple[str, ...], ...] = (
    ("dataset", "n_frames"),
    ("targets", "per_frame"),
)
_SKY_PATH: tuple[str, ...] = ("background", "sky_e")
_SAMPLED_PATHS: tuple[tuple[str, ...], ...] = (
    _SEQUENCE_INT_PATHS + _SEQUENCE_UNIFORM_PATHS + (_SKY_PATH,)
)

assert_distinct(["config/" + ".".join(path) for path in _SAMPLED_PATHS])

# 只限制「经过 sample_int_inclusive 抽样」的字段。dataset.seed 不在此列。
# seed 仍然适用「整数必须能转换为有限浮点」。这是明确选定的配置支持范围：
# 2**1023 可以通过，10**400 不行。这是有意的契约差异：配置入口比 root_seed
# 严格，root_seed(10**400) 可以直接调用，配置却会拒绝。
# 比较用精确整数。转成 float 会把 2**63 - 1 舍入成 2**63。
_INT64_MIN = -2**63
_INT64_MAX = 2**63 - 1
_SAMPLED_INT_DOTTED = frozenset(".".join(path) for path in _SEQUENCE_INT_PATHS)

# 配置抽样版本。M4 写入记录的 config_sampling。
# 常量留在本模块。algorithm_versions() 每次返回一份新建的映射
# {"config_sampling": 1, "motion": 2}，不含 rng_scheme。rng_scheme 只由
# SequenceRng.metadata() 承载。
# M1 只定义常量和汇总函数，不实现记录写入和重放校验。
#
# 版本号保护合法配置的配置抽样结果，以及合法序列记录中受重放契约保护的输出
# 和相关随机数消耗。仅拒绝既有配置契约已禁止、且不可能参与任何合法记录的输入，
# 不递增版本。包级辅助函数的直接调用入口收紧，须单独记载。
# 输入原先合法，但既不可能产生任何合法序列记录，也不影响配置抽样结果的，
# 收紧时不递增版本，修订记录写明「收紧支持范围」。
#
# 本常量是第一版基线，内容是修复后的配置抽样规则：整数全程精确处理，
# 以及 int64 支持范围。在此之前没有版本标识的记录，不在这套重放承诺之内。
# config_effective 本身就是可观察输出，不能用「旧结果其实没变」来论证。
# int64 收紧已并入这个基线：超范围整数原先能进入 config_effective。标量和
# 等端点区间原先直接返回、不调用 NumPy，超出 int64 的精确整数本来可以成功返回。
#
# 不递增的例子：
# - 拒绝标量 NaN/Inf。配置契约本来就禁止。这是公共函数的行为变更：收紧直接调用入口。
# - 浮点字段上的超大整数原先抛出未捕获的 OverflowError，现改为带路径的 ValueError。
#   只是异常规范化。浮点跨度检查也属于这一类。
# - 收紧相机派生值支持范围。f_over_p 为 inf 或极小时，nominal_step 拒绝每一次尝试，
#   不可能产生合法序列；sample_sequence_params 也不读相机派生值。
# derive_camera 本身不归任何版本管。它的公式一旦改变，会通过 K 和 f_over_p
# 影响运动输出，按共享辅助函数的归属递增 motion。
#
# 递增本版本：字段分布、整数处理、配置抽样规则改变了原本能成功的输入的结果，
# 或改变了随机数消耗量。随机流改名（例如 config/<path>）递增使用该流的模块，
# 不递增 RNG_SCHEME_VERSION。sample_uniform 的数值改写同时递增 motion：
# 它既用于配置抽样，也用于 motion 抽 s₀ 和 a。
# 一次修改若改变多个模块的行为，对应的多个版本都要递增。
CONFIG_SAMPLING_VERSION = 1


@dataclass(frozen=True)
class Constraint:
    """One field's value rule.

    ``interval`` requires both endpoints to lie in range and ``lo <= hi``.
    ``scalar`` requires one number in range. ``either`` accepts both forms
    and is used for fields sampled once per sequence.
    ``camera_motion.speed_px_per_frame`` is stricter: equal endpoints are
    rejected in ``_check_camera_speed_interval``. A zero-width interval
    constrains both the nominal step and the jittered displacement, and is
    not an acceptance interval for rejection sampling.
    """

    kind: Literal["scalar", "interval", "either"]
    integer: bool = False
    lo: float | None = None
    hi: float | None = None
    lo_open: bool = False
    hi_open: bool = False
    nullable: bool = False


def _c(
    kind: Literal["scalar", "interval", "either"],
    *,
    integer: bool = False,
    lo: float | None = None,
    hi: float | None = None,
    lo_open: bool = False,
    hi_open: bool = False,
    nullable: bool = False,
) -> Constraint:
    return Constraint(kind, integer, lo, hi, lo_open, hi_open, nullable)


# Every generator leaf belongs to exactly one of the five tables below.
# ``*`` matches one path component (fraction names and tier names).
_CONSTRAINTS: dict[tuple[str, ...], Constraint] = {
    ("dataset", "n_sequences"): _c("scalar", integer=True, lo=1),
    ("dataset", "n_frames"): _c("either", integer=True, lo=2),
    ("dataset", "seed"): _c("scalar", integer=True, lo=0),
    ("dataset", "workers"): _c("scalar", integer=True, lo=1),
    ("dataset", "split", "fractions", "*"): _c("scalar", lo=0, hi=1),
    ("dataset", "split", "caps_per_split"): _c("scalar", integer=True, lo=1),
    ("dataset", "split", "cap_radius_deg"): _c("scalar", lo=0, hi=90, lo_open=True),
    ("dataset", "split", "guard_deg"): _c("scalar", lo=0),
    ("dataset", "split", "min_seqs_for_assert"): _c("scalar", integer=True, lo=1),
    ("timing", "frame_interval_s"): _c("scalar", lo=0, lo_open=True),
    ("timing", "exposure_s_nominal"): _c("scalar", lo=0, lo_open=True),
    ("pointing", "min_labeled_stars"): _c("scalar", integer=True, lo=0),
    ("pointing", "galactic_lat_min_deg"): _c("scalar", lo=0, hi=90, nullable=True),
    ("camera_motion", "speed_px_per_frame"): _c("interval", lo=0, lo_open=True),
    ("camera_motion", "accel_px_per_frame2"): _c("interval"),
    ("camera_motion", "jitter_px"): _c("either", lo=0),
    ("photometry", "mag_limit"): _c("either", lo=0, hi=20),
    ("photometry", "snr_at_limit"): _c("scalar", lo=0, lo_open=True),
    ("photometry", "render_mag_margin"): _c("scalar", lo=0),
    ("psf", "sigma_px"): _c("either", lo=0, lo_open=True),
    ("psf", "field_growth"): _c("either", lo=0),
    ("psf", "max_ellipticity"): _c("scalar", lo=0, hi=1, hi_open=True),
    ("psf", "halo_fraction"): _c("either", lo=0, hi=1),
    ("background", "sky_e"): _c("either", lo=0, lo_open=True),
    ("background", "gradient_frac"): _c("either", lo=0, hi=1),
    ("background", "smooth_field_frac"): _c("either", lo=0, hi=1),
    ("background", "vignetting_corner"): _c("either", lo=0, hi=1, hi_open=True),
    ("sensor_noise", "read_e"): _c("either", lo=0),
    ("sensor_noise", "dark_e_per_s"): _c("either", lo=0),
    ("sensor_noise", "prnu"): _c("either", lo=0, hi=1, hi_open=True),
    ("sensor_noise", "gain_e_per_adu"): _c("either", lo=0, lo_open=True),
    ("sensor_noise", "bias_adu"): _c("either", lo=0),
    ("sensor_noise", "full_well_e"): _c("either", lo=0, lo_open=True),
    ("sensor_noise", "row_col_fpn_e"): _c("either", lo=0),
    ("artifacts", "hot_pixel_frac"): _c("either", lo=0, hi=1),
    ("artifacts", "hot_cluster_frac"): _c("scalar", lo=0, hi=1),
    ("artifacts", "hot_cluster_size_px"): _c("interval", integer=True, lo=2),
    ("artifacts", "rts_frac"): _c("scalar", lo=0, hi=1),
    ("artifacts", "rts_p_flip"): _c("interval", lo=0, hi=1),
    ("artifacts", "dead_pixel_frac"): _c("either", lo=0, hi=1),
    ("artifacts", "cosmic_rays_per_frame"): _c("either", lo=0),
    ("targets", "per_frame"): _c("either", integer=True, lo=1),
    ("targets", "streak_fraction"): _c("scalar", lo=0, hi=1),
    ("targets", "min_visible_frames"): _c("scalar", integer=True, lo=1),
    ("targets", "point", "speed_px_per_frame"): _c("interval", lo=0, lo_open=True),
    ("targets", "point", "smear_px"): _c("interval", lo=0),
    ("targets", "streak", "speed_px_per_frame"): _c("interval", lo=0, lo_open=True),
    ("targets", "streak", "duty"): _c("interval", lo=0, hi=1, lo_open=True),
    ("targets", "streak", "min_length_px"): _c("scalar", lo=0, lo_open=True),
    ("targets", "streak", "angle_jitter_deg"): _c("interval", lo=0),
    ("targets", "track_deviation", "strict_fraction"): _c("scalar", lo=0, hi=1),
    ("targets", "track_deviation", "curve_amp_px"): _c("interval", lo=0),
    ("targets", "track_deviation", "rand_cross_px"): _c("interval", lo=0),
    ("targets", "track_deviation", "rand_along_px"): _c("interval", lo=0),
    ("targets", "track_deviation", "rand_rel_cap"): _c("scalar", lo=0, lo_open=True),
    ("targets", "track_deviation", "clip_sigma"): _c("scalar", lo=0, lo_open=True),
    ("targets", "track_deviation", "max_track_dev_px"): _c("scalar", lo=0, lo_open=True),
    ("targets", "tiers", "*", "snr"): _c("interval", lo=0, lo_open=True),
    ("targets", "tiers", "*", "frac"): _c("scalar", lo=0, hi=1),
    ("targets", "brightness_jitter_sigma"): _c("scalar", lo=0),
    ("labels", "stride"): _c("scalar", integer=True, lo=1),
    ("labels", "sigma_hm"): _c("scalar", lo=0, lo_open=True),
    ("labels", "blend_search_sigma"): _c("scalar", lo=0, lo_open=True),
    ("labels", "label_snr_eff_min"): _c("scalar", lo=0),
    ("labels", "label_snr_min"): _c("scalar", lo=0),
    ("synthetic_faint_stars", "max_snr"): _c("scalar", lo=0),
}
_ENUMS: dict[tuple[str, ...], frozenset] = {
    ("dataset", "image_format"): frozenset({"fits_rice", "png16", "tiff_zstd"}),
    ("dataset", "split", "mode"): frozenset({"sky", "random"}),
    ("labels", "label_frame"): frozenset({"native", "reference"}),
    ("labels", "canvas"): frozenset({"union", "reference"}),
    ("labels", "sigma_mode"): frozenset({"fixed", "adaptive"}),
    ("labels", "snr_mode"): frozenset({"effective", "single"}),
    ("labels", "never_full_policy"): frozenset({"label", "ignore"}),
    ("targets", "flux_design_noise"): frozenset({"trajectory_mean"}),
    ("sensor_noise", "adc_bits"): frozenset({12, 14, 16}),
}
_BOOL_PATHS: frozenset[tuple[str, ...]] = frozenset({
    ("dataset", "export_registered"),
    ("dataset", "split", "stratify_by_glat"),
    ("camera_motion", "slew_during_exposure"),
    ("background", "log_uniform"),
    ("artifacts", "blooming"),
    ("labels", "save_dense"),
    ("labels", "per_class_offset"),
    ("labels", "streak_geom"),
    ("labels", "streak_line"),
    ("labels", "motion"),
    ("synthetic_faint_stars", "enabled"),
})
# Path -> why the field is not a plain constraint. Checkers are _CUSTOM_DISPATCH.
_CUSTOM_PATHS: dict[tuple[str, ...], str] = {
    ("dataset", "out_dir"): "非空字符串",
    ("camera_motion", "reference_frame"): "'middle' 或小于 n_frames 下界的非负整数",
    ("labels", "snr_window"): "'all' 或正整数",
}
_EXEMPT: dict[tuple[str, ...], str] = {}

_CAMERA_CONSTRAINTS: dict[tuple[str, ...], Constraint] = {
    ("sensor", "resolution", "width"): _c("scalar", integer=True, lo=1),
    ("sensor", "resolution", "height"): _c("scalar", integer=True, lo=1),
    ("sensor", "pixel_size_mm"): _c("scalar", lo=0, lo_open=True),
    ("sensor", "focal_length_mm"): _c("scalar", lo=0, lo_open=True),
    ("sensor", "aperture_mm"): _c("scalar", lo=0, lo_open=True),
    ("sensor", "spectral_range_nm", "min"): _c("scalar", lo=0, lo_open=True),
    ("sensor", "spectral_range_nm", "max"): _c("scalar", lo=0, lo_open=True),
    ("derived", "fov_deg", "width"): _c("scalar", lo=0, lo_open=True),
    ("derived", "fov_deg", "height"): _c("scalar", lo=0, lo_open=True),
}
_CAMERA_ENUMS: dict[tuple[str, ...], frozenset] = {}
_CAMERA_BOOL_PATHS: frozenset[tuple[str, ...]] = frozenset()
_CAMERA_CUSTOM_PATHS: dict[tuple[str, ...], str] = {}
_CAMERA_EXEMPT: dict[tuple[str, ...], str] = {}


@dataclass
class DerivedCamera:
    """Camera geometry derived from the sensor description."""

    width: int
    height: int
    pixel_size_mm: float
    focal_length_mm: float
    aperture_mm: float
    fov_width_deg: float
    fov_height_deg: float
    pixel_scale_arcsec: float
    half_diagonal_deg: float
    f_over_p: float
    cx: float
    cy: float
    K: np.ndarray

    def as_dict(self) -> dict:
        """JSON-friendly view. ``K`` is a nested list."""
        return {
            "fov_deg": {
                "width": self.fov_width_deg,
                "height": self.fov_height_deg,
            },
            "pixel_scale_arcsec": self.pixel_scale_arcsec,
            "half_diagonal_deg": self.half_diagonal_deg,
            "f_over_p": self.f_over_p,
            "cx": self.cx,
            "cy": self.cy,
            "K": np.asarray(self.K, dtype=np.float64).tolist(),
        }


@dataclass
class AppConfig:
    """Merged camera and generator configuration."""

    camera: dict
    generator: dict
    derived: DerivedCamera


def load_yaml(path: Path | str) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge mappings. Lists and scalars in ``override`` replace."""
    if not isinstance(base, dict) or not isinstance(override, dict):
        raise TypeError("deep_merge expects two mappings")
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def is_interval(value: object) -> bool:
    """True for a length-2 numeric range. Booleans are not numeric here."""
    if isinstance(value, (str, bytes)) or isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    return all(_is_number(item) for item in value)


# 超大整数转 float 时的哨兵。NaN 和 Inf 能转成浮点，不是这个哨兵。
_FLOAT_OVERFLOW = object()


def _to_float_or_overflow(value: object) -> float | object:
    """Convert with ``float``. ``OverflowError`` returns :data:`_FLOAT_OVERFLOW`.

    The name stops at conversion. NaN and Inf come back as those floats;
    the caller decides whether a non-finite result is its own error.
    Constraint checks and cross-field checks both use this function, so an
    operand that cannot be converted is judged the same way in both places.
    """
    try:
        return float(value)
    except OverflowError:
        return _FLOAT_OVERFLOW


def _relation_float(value: object) -> float | None:
    """Float for one cross-field operand, or ``None`` when that relation is skipped.

    ``None`` means the operand is not a number, or :func:`_to_float_or_overflow`
    returned the overflow sentinel. The constraint table already reported the
    overflow, so the relation check must not report the same value again.
    """
    if not _is_number(value):
        return None
    number = _to_float_or_overflow(value)
    if number is _FLOAT_OVERFLOW:
        return None
    return number


def uniform_span_is_finite(lo: float, hi: float) -> bool:
    """True when ``hi - lo`` is a finite float.

    Config validation and motion share this predicate so the two checks
    cannot drift. It does not sample, and it does not modify or replace the
    caller's interval. The caller supplies any field path when it reports a
    failure. Linear and log-uniform intervals both use this test. A
    log-uniform interval has positive endpoints, so ``hi - lo`` is at most
    ``hi`` and is finite whenever those endpoints are. Integer intervals are
    not judged here.
    """
    return math.isfinite(hi - lo)


def _interval_floats(spec: object) -> tuple[float, float]:
    """Finite endpoints, or ``ValueError`` carrying the interval.

    An integer outside the finite float range used to raise ``OverflowError``.
    That is now this ``ValueError``. Successful intervals are unchanged.
    """
    lo = _to_float_or_overflow(spec[0])
    hi = _to_float_or_overflow(spec[1])
    if (
        lo is _FLOAT_OVERFLOW
        or hi is _FLOAT_OVERFLOW
        or not math.isfinite(lo)
        or not math.isfinite(hi)
    ):
        raise ValueError(f"interval bounds must be finite, got {spec!r}")
    return lo, hi


def _scalar_float(spec: object) -> float:
    """Finite scalar. Non-finite values and unconvertible integers are rejected.

    公共函数的行为变更：收紧直接调用入口。配置契约本来就禁止 NaN 和 Inf，
    不递增版本。超大整数原先是未捕获的 ``OverflowError``，现为 ``ValueError``。
    """
    number = _to_float_or_overflow(spec)
    if number is _FLOAT_OVERFLOW or not math.isfinite(number):
        raise ValueError(f"scalar must be finite, got {spec!r}")
    return number


def sample_uniform(rng: np.random.Generator, spec: object) -> float:
    """Sample U[lo, hi] when ``spec`` is an interval, otherwise return the scalar.

    An interval whose span is not finite is an unsupported sampling input.
    The uniform formula itself is unchanged, so an ordinary interval keeps
    the same bits. This entry has no field path; the error carries the
    interval. A non-finite scalar is rejected. That tightens the direct-call
    entry: the configuration contract already forbids those inputs, so the
    version does not increase.
    """
    _require_generator(rng)
    if is_interval(spec):
        lo, hi = _interval_floats(spec)
        _check_bounds(lo, hi, spec)
        if not uniform_span_is_finite(lo, hi):
            raise ValueError(f"interval span must be finite, got {spec!r}")
        if lo == hi:
            return lo
        return float(rng.uniform(lo, hi))
    if _is_number(spec):
        return _scalar_float(spec)
    raise TypeError(f"expected a number or [lo, hi] interval, got {spec!r}")


def sample_log_uniform(rng: np.random.Generator, spec: object) -> float:
    """Sample log-uniform on a positive interval, or return a positive scalar.

    Non-finite scalars are rejected. That is a direct-call tightening, not a
    version bump: a legal configuration cannot contain them.
    """
    _require_generator(rng)
    if is_interval(spec):
        lo, hi = _interval_floats(spec)
        _check_bounds(lo, hi, spec)
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError(f"log-uniform bounds must be positive, got {spec!r}")
        if lo == hi:
            return lo
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if _is_number(spec):
        value = _scalar_float(spec)
        if value <= 0.0:
            raise ValueError(f"log-uniform scalar must be positive, got {spec!r}")
        return value
    raise TypeError(f"expected a number or [lo, hi] interval, got {spec!r}")


def sample_int_inclusive(rng: np.random.Generator, spec: object) -> int:
    """Sample an integer uniformly from an inclusive interval, or return a scalar int.

    Both the scalar and either endpoint must lie in ``[-2**63, 2**63 - 1]``.
    The comparison is exact and does not pass through ``float``. This narrows
    the supported range: a scalar or an equal-endpoint interval used to return
    without calling NumPy, so an exact integer outside int64 used to succeed.
    ``rng.integers(..., endpoint=True)`` states the closed interval directly.
    It is not what makes the int64 upper bound legal; that was the rounding
    in :func:`_lenient_int`.
    """
    _require_generator(rng)
    if is_interval(spec):
        lo = _lenient_int(spec[0], spec)
        hi = _lenient_int(spec[1], spec)
        if hi < lo:
            raise ValueError(f"interval lower bound exceeds upper bound: {spec!r}")
        _require_sampled_int64(lo, hi, spec)
        if lo == hi:
            return lo
        return int(rng.integers(lo, hi, endpoint=True))
    if _is_number(spec):
        value = _lenient_int(spec, spec)
        _require_sampled_int64(value, value, spec)
        return value
    raise TypeError(f"expected an integer or [lo, hi] interval, got {spec!r}")


def _eval_float(operation):
    """Run one existing arithmetic step. ``OverflowError`` becomes the sentinel.

    The operation itself is unchanged. This only stops an integer that cannot
    convert to float from escaping as an uncaught exception.
    """
    try:
        return float(operation())
    except OverflowError:
        return _FLOAT_OVERFLOW


def _shown_number(number: object) -> str:
    if number is _FLOAT_OVERFLOW:
        return "超出有限浮点范围"
    return repr(number)


def _not_finite(number: object) -> bool:
    if number is _FLOAT_OVERFLOW or not isinstance(number, float):
        return True
    return not math.isfinite(number)


def _not_strictly_positive(number: object) -> bool:
    if _not_finite(number):
        return True
    return number <= 0.0


def _reject_positive(number: object, label: str, problems: list[str]) -> None:
    if _not_strictly_positive(number):
        problems.append(f"{label} 必须有限且严格大于 0，得到 {_shown_number(number)}")


def _reject_finite(number: object, label: str, problems: list[str]) -> None:
    if _not_finite(number):
        problems.append(f"{label} 必须有限，得到 {_shown_number(number)}")


def derive_camera(sensor: dict, expected_derived: dict | None = None) -> DerivedCamera:
    """Derive FOV, plate scale, principal point, and ``K`` from ``sensor``.

    ``expected_derived`` is the hand-written ``derived`` block, if any. Its
    ``fov_deg`` is compared with the computed field of view and is not copied
    into the result.

    The formulas are unchanged. Intermediate ``atan`` arguments, and
    ``half_diagonal_mm`` itself, must be finite and greater than 0, because
    overflow through ``atan`` would otherwise look like a 180° or 90° field.
    Final values are then checked by what they mean: the scale-like quantities
    must be finite and strictly positive, and ``cx`` and ``cy`` must be
    finite and may be 0. Each quantity is reported once. Any failure is
    raised before ``K`` is built.

    ``K`` is assembled only after those checks, in the fixed layout
    ``[[f/p, 0, cx], [0, f/p, cy], [0, 0, 1]]``. ``f_over_p`` has already
    been required to be finite and strictly positive. Finiteness of the
    three inputs would not give that sign: an underflow to 0 is still
    finite, so the positive check is what carries the sign. ``cx`` and
    ``cy`` have already been required to be finite. With the fixed layout,
    every element is finite, the two focal diagonals are strictly positive,
    and the other entries stay 0 or 1. Those properties are not checked
    again.

    This tightens the supported camera configuration. It does not increment
    a version. ``derive_camera`` itself is not versioned; changing the
    formulas would increment the motion version, because ``K`` and
    ``f_over_p`` affect motion output.
    """
    if not isinstance(sensor, dict):
        raise TypeError("sensor must be a mapping")
    resolution = sensor.get("resolution")
    if not isinstance(resolution, dict):
        raise ValueError("sensor.resolution must be a mapping with width and height")
    width = _positive_int(resolution.get("width"), "sensor.resolution.width")
    height = _positive_int(resolution.get("height"), "sensor.resolution.height")
    pixel_size_mm = _positive_float(sensor.get("pixel_size_mm"), "sensor.pixel_size_mm")
    focal_length_mm = _positive_float(sensor.get("focal_length_mm"), "sensor.focal_length_mm")
    aperture_mm = _positive_float(sensor.get("aperture_mm"), "sensor.aperture_mm")

    # Same parenthesization as before. Naming the intermediates does not reorder the arithmetic.
    width_arg = _eval_float(lambda: (width * pixel_size_mm / 2.0) / focal_length_mm)
    height_arg = _eval_float(lambda: (height * pixel_size_mm / 2.0) / focal_length_mm)
    half_diagonal_mm = _eval_float(lambda: pixel_size_mm * math.hypot(width, height) / 2.0)
    if half_diagonal_mm is _FLOAT_OVERFLOW:
        diag_arg = _FLOAT_OVERFLOW
    else:
        diag_arg = _eval_float(lambda: half_diagonal_mm / focal_length_mm)
    fov_width_deg = (
        _FLOAT_OVERFLOW if not isinstance(width_arg, float) else math.degrees(2.0 * math.atan(width_arg))
    )
    fov_height_deg = (
        _FLOAT_OVERFLOW
        if not isinstance(height_arg, float)
        else math.degrees(2.0 * math.atan(height_arg))
    )
    pixel_scale_arcsec = _eval_float(
        lambda: (pixel_size_mm / focal_length_mm) * (180.0 / math.pi) * 3600.0
    )
    half_diagonal_deg = (
        _FLOAT_OVERFLOW if not isinstance(diag_arg, float) else math.degrees(math.atan(diag_arg))
    )
    f_over_p = _eval_float(lambda: focal_length_mm / pixel_size_mm)
    cx = _eval_float(lambda: (width - 1) / 2.0)
    cy = _eval_float(lambda: (height - 1) / 2.0)

    problems: list[str] = []
    width_inputs = "sensor.resolution.width、sensor.pixel_size_mm、sensor.focal_length_mm"
    height_inputs = "sensor.resolution.height、sensor.pixel_size_mm、sensor.focal_length_mm"
    diagonal_inputs = "sensor.resolution.width、sensor.resolution.height、sensor.pixel_size_mm"
    diagonal_ratio_inputs = diagonal_inputs + "、sensor.focal_length_mm"
    scale_inputs = "sensor.focal_length_mm、sensor.pixel_size_mm"
    _reject_positive(width_arg, f"{width_inputs}: 半宽视场的 atan 参数", problems)
    _reject_positive(height_arg, f"{height_inputs}: 半高视场的 atan 参数", problems)
    _reject_positive(half_diagonal_mm, f"{diagonal_inputs}: half_diagonal_mm", problems)
    _reject_positive(diag_arg, f"{diagonal_ratio_inputs}: 半对角视场的 atan 参数", problems)
    _reject_positive(f_over_p, f"{scale_inputs}: f_over_p", problems)
    _reject_positive(fov_width_deg, f"{width_inputs}: fov_deg.width", problems)
    _reject_positive(fov_height_deg, f"{height_inputs}: fov_deg.height", problems)
    _reject_positive(pixel_scale_arcsec, f"{scale_inputs}: pixel_scale_arcsec", problems)
    _reject_positive(half_diagonal_deg, f"{diagonal_ratio_inputs}: half_diagonal_deg", problems)
    _reject_finite(cx, "sensor.resolution.width: cx", problems)
    _reject_finite(cy, "sensor.resolution.height: cy", problems)
    if problems:
        raise ValueError("相机派生值超出支持范围：\n" + "\n".join(problems))
    intrinsic = np.array(
        [
            [f_over_p, 0.0, cx],
            [0.0, f_over_p, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    _warn_fov_mismatch(fov_width_deg, fov_height_deg, expected_derived)
    return DerivedCamera(
        width=width,
        height=height,
        pixel_size_mm=pixel_size_mm,
        focal_length_mm=focal_length_mm,
        aperture_mm=aperture_mm,
        fov_width_deg=fov_width_deg,
        fov_height_deg=fov_height_deg,
        pixel_scale_arcsec=pixel_scale_arcsec,
        half_diagonal_deg=half_diagonal_deg,
        f_over_p=f_over_p,
        cx=cx,
        cy=cy,
        K=intrinsic,
    )


def load_config(
    camera_path: Path | str | None = None,
    generator_path: Path | str | None = None,
    overrides: dict | None = None,
) -> AppConfig:
    """Load defaults, then deep-merge optional files and ``overrides``.

    Unknown keys, types, and bounds are reported together by
    ``validate_generator`` and ``validate_camera``. That runs before
    ``derived`` is replaced, so a hand-written ``fov_deg`` is still checked.
    ``overrides`` may contain ``camera`` and ``generator`` partial trees.
    The returned ``camera["derived"]`` is always the computed block.
    Hand-written ``derived`` may contain only ``fov_deg``.
    """
    camera_default = load_yaml(DEFAULT_CAMERA_PATH)
    generator_default = load_yaml(DEFAULT_GENERATOR_PATH)
    camera = copy.deepcopy(camera_default)
    generator = copy.deepcopy(generator_default)
    if camera_path is not None:
        camera = deep_merge(camera, load_yaml(camera_path))
    if generator_path is not None:
        generator = deep_merge(generator, load_yaml(generator_path))
    if overrides:
        camera, generator = _apply_overrides(camera, generator, overrides)
    validate_generator(generator)
    validate_camera(camera)

    sensor = camera.get("sensor")
    if not isinstance(sensor, dict):
        raise ValueError("camera.sensor is required")
    expected_derived = camera.get("derived") if isinstance(camera.get("derived"), dict) else None
    derived = derive_camera(sensor, expected_derived)
    camera["derived"] = derived.as_dict()
    return AppConfig(camera=camera, generator=generator, derived=derived)


def sample_sequence_params(cfg: AppConfig | dict, rngs: SequenceRng) -> dict:
    """Return a full effective copy of the merged generator config.

    The copy starts as ``deepcopy`` of the generator mapping. Each path in
    the sampling list is then replaced by one draw from ``rngs.get``, named
    ``config/<dotted path>``. ``sky_e`` uses ``config/background.sky_e``
    whether the draw is log-uniform or uniform. ``n_frames`` and
    ``targets.per_frame`` are inclusive integers.

    Each of those names is issued once. Calling this again on the same
    ``SequenceRng`` raises ``ValueError`` because the names were already
    issued. Draw another copy from a new ``SequenceRng`` built on the same
    seed sequence.

    ``camera_motion.speed_px_per_frame`` and ``camera_motion.accel_px_per_frame2``
    are not drawn here. ``motion.py`` redraws them inside the rejection loop
    from the ``motion`` stream and writes the accepted scalars back onto this
    same copy. Until that happens, both fields remaining as intervals is the
    expected state. An equal-endpoint speed interval is rejected by
    ``validate_generator``: the interval must satisfy ``lo < hi``. A
    zero-width interval is not an acceptance interval for rejection
    sampling. The small-angle derivation is in plan section 4.2.
    ``jitter_px`` is drawn once per sequence.

    The input config is not modified. Metadata (M4) stores this copy as
    ``config_effective`` and the pre-sample merge as ``config_unsampled``.
    Do not add a ``sampled_paths`` field.
    """
    if not isinstance(rngs, SequenceRng):
        raise TypeError("rngs must be starsim.rng.SequenceRng")
    generator = _generator_mapping(cfg)
    effective = copy.deepcopy(generator)
    for path in _SEQUENCE_INT_PATHS:
        drawn = sample_int_inclusive(rngs.get(_stream_name(path)), _lookup(generator, path))
        _assign_path(effective, path, drawn)
    for path in _SEQUENCE_UNIFORM_PATHS:
        drawn = sample_uniform(rngs.get(_stream_name(path)), _lookup(generator, path))
        _assign_path(effective, path, drawn)
    sky_rng = rngs.get(_stream_name(_SKY_PATH))
    sky_spec = _lookup(generator, _SKY_PATH)
    log_uniform = _lookup(generator, ("background", "log_uniform"))
    if type(log_uniform) is not bool:
        raise TypeError(f"background.log_uniform must be bool, got {log_uniform!r}")
    if log_uniform:
        sky = sample_log_uniform(sky_rng, sky_spec)
    else:
        sky = sample_uniform(sky_rng, sky_spec)
    _assign_path(effective, _SKY_PATH, sky)
    return effective


def validate_generator(gen: dict) -> None:
    """Check generator values. All problems are reported in one ``ValueError``.

    Order: unknown keys, then types and bounds, then cross-field checks.
    """
    if not isinstance(gen, dict):
        raise TypeError("generator config must be a mapping")
    problems: list[str] = []
    _collect_unknown_keys(load_yaml(DEFAULT_GENERATOR_PATH), gen, "generator", problems)
    _apply_constraints(gen, _CONSTRAINTS, problems)
    _apply_enums(gen, _ENUMS, problems)
    _apply_bools(gen, _BOOL_PATHS, problems)
    labels = gen.get("labels")
    if isinstance(labels, dict):
        for key in ("per_class_offset", "streak_geom"):
            if labels.get(key) is False:
                problems.append(f"labels.{key}: false is not supported")
    for checker in _CUSTOM_DISPATCH.values():
        checker(gen, problems)
    _apply_generator_relations(gen, problems)
    if problems:
        raise ValueError("generator 配置校验失败：\n" + "\n".join(problems))


def validate_label_cfg(cfg):
    """Apply the generator label contract to derived CLI overrides."""
    if not isinstance(cfg, dict):
        raise ValueError("labels must be a mapping")
    gen = {"labels": cfg}
    problems = []
    _apply_constraints(gen, {p: v for p, v in _CONSTRAINTS.items() if p[0] == "labels"}, problems)
    _apply_enums(gen, {p: v for p, v in _ENUMS.items() if p[0] == "labels"}, problems)
    _apply_bools(gen, {p for p in _BOOL_PATHS if p[0] == "labels"}, problems)
    _check_snr_window(gen, problems)
    for key in ("per_class_offset", "streak_geom"):
        if cfg.get(key) is False:
            problems.append(f"labels.{key}: false is not supported")
    if problems:
        raise ValueError("labels 配置校验失败：\n" + "\n".join(problems))


def validate_camera(camera: dict) -> None:
    """Check camera values before ``derived`` is overwritten."""
    if not isinstance(camera, dict):
        raise TypeError("camera config must be a mapping")
    problems: list[str] = []
    _collect_unknown_keys(load_yaml(DEFAULT_CAMERA_PATH), camera, "camera", problems)
    _apply_constraints(camera, _CAMERA_CONSTRAINTS, problems)
    _apply_enums(camera, _CAMERA_ENUMS, problems)
    _apply_bools(camera, _CAMERA_BOOL_PATHS, problems)
    _check_spectral_order(camera, problems)
    if problems:
        raise ValueError("camera 配置校验失败：\n" + "\n".join(problems))


def _collect_unknown_keys(reference: dict, override: dict, prefix: str, problems: list[str]) -> None:
    for key, value in override.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in reference:
            problems.append(f"{path}: 未知键")
            continue
        ref_value = reference[key]
        if isinstance(ref_value, dict) and not isinstance(value, dict):
            problems.append(f"{path}: 期望映射，得到 {type(value).__name__}")
            continue
        if not isinstance(ref_value, dict) and isinstance(value, dict):
            problems.append(f"{path}: 期望标量或列表，得到映射")
            continue
        if isinstance(ref_value, dict) and isinstance(value, dict):
            _collect_unknown_keys(ref_value, value, path, problems)


def _apply_overrides(camera: dict, generator: dict, overrides: dict) -> tuple[dict, dict]:
    if not isinstance(overrides, dict):
        raise TypeError("overrides must be a mapping")
    unknown = set(overrides) - {"camera", "generator"}
    if unknown:
        raise ValueError(
            'overrides may only contain "camera" and "generator", '
            f"got {sorted(unknown)}"
        )
    if overrides.get("camera") is not None:
        if not isinstance(overrides["camera"], dict):
            raise TypeError('overrides["camera"] must be a mapping')
        camera = deep_merge(camera, overrides["camera"])
    if overrides.get("generator") is not None:
        if not isinstance(overrides["generator"], dict):
            raise TypeError('overrides["generator"] must be a mapping')
        generator = deep_merge(generator, overrides["generator"])
    return camera, generator


def _warn_fov_mismatch(
    fov_width_deg: float,
    fov_height_deg: float,
    expected_derived: dict | None,
) -> None:
    if not isinstance(expected_derived, dict):
        return
    fov = expected_derived.get("fov_deg")
    if not isinstance(fov, dict):
        return
    problems: list[str] = []
    for axis, computed in (("width", fov_width_deg), ("height", fov_height_deg)):
        if axis not in fov or fov[axis] is None:
            continue
        expected = _to_float_or_overflow(fov[axis])
        if expected is _FLOAT_OVERFLOW:
            continue
        if computed == 0.0:
            relative = math.inf
        else:
            relative = abs(computed - expected) / abs(computed)
        if relative > FOV_RELATIVE_TOLERANCE:
            problems.append(
                f"{axis}: 期望 {expected:g}°，推导 {computed:.3f}°，"
                f"相对偏差 {relative * 100:.2f}%"
            )
    if problems:
        warnings.warn(
            "配置中的 fov_deg 与由 sensor 推导的视场角相对偏差超过 1%，"
            "已用推导值覆盖。" + "；".join(problems),
            UserWarning,
            stacklevel=3,
        )


def _generator_mapping(cfg: AppConfig | dict) -> dict:
    if isinstance(cfg, AppConfig):
        return cfg.generator
    if isinstance(cfg, dict):
        if "generator" in cfg and isinstance(cfg["generator"], dict):
            return cfg["generator"]
        if "dataset" in cfg or "photometry" in cfg:
            return cfg
    raise TypeError("cfg must be an AppConfig or a generator mapping")


def _lookup(mapping: dict, path: tuple[str, ...]) -> object:
    node: object = mapping
    walked: list[str] = []
    for key in path:
        walked.append(key)
        if not isinstance(node, dict) or key not in node:
            raise KeyError(".".join(walked))
        node = node[key]
    return node


def _assign_path(dest: dict, path: tuple[str, ...], value: object) -> None:
    node = dest
    for key in path[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[path[-1]] = value


def _stream_name(path: tuple[str, ...]) -> str:
    return "config/" + ".".join(path)


_MISSING = object()


def _resolve(mapping: dict, pattern: tuple[str, ...]) -> list[tuple[str, object]]:
    """Expand ``pattern`` against ``mapping``.

    A missing key is ``_MISSING``. A present ``null`` stays ``None``, so a
    nullable field is not reported as absent.
    """
    found: list[tuple[str, object]] = []

    def walk(node: object, index: int, prefix: tuple[str, ...]) -> None:
        if index == len(pattern):
            found.append((".".join(prefix), node))
            return
        token = pattern[index]
        if token == "*":
            if not isinstance(node, dict) or not node:
                found.append((".".join(prefix), _MISSING))
                return
            for key, child in node.items():
                walk(child, index + 1, prefix + (str(key),))
            return
        if not isinstance(node, dict) or token not in node:
            if "*" in pattern[index:]:
                found.append((".".join(prefix + (token,)), _MISSING))
            else:
                found.append((".".join(prefix + pattern[index:]), _MISSING))
            return
        walk(node[token], index + 1, prefix + (token,))

    walk(mapping, 0, ())
    return found


def _apply_constraints(
    mapping: dict,
    table: dict[tuple[str, ...], Constraint],
    problems: list[str],
) -> None:
    for pattern, constraint in table.items():
        for path, value in _resolve(mapping, pattern):
            if value is _MISSING:
                problems.append(f"{path}: 缺少")
                continue
            _check_constraint(value, constraint, path, problems)


def _apply_enums(
    mapping: dict,
    table: dict[tuple[str, ...], frozenset],
    problems: list[str],
) -> None:
    for pattern, allowed in table.items():
        kinds = {type(item) for item in allowed}
        expected = "或".join(sorted(kind.__name__ for kind in kinds))
        for path, value in _resolve(mapping, pattern):
            if value is _MISSING:
                problems.append(f"{path}: 缺少")
                continue
            if type(value) not in kinds:
                problems.append(
                    f"{path}: 类型应为 {expected}，得到 {type(value).__name__} {value!r}"
                )
                continue
            if value not in allowed:
                problems.append(f"{path}: 非法枚举值 {value!r}，允许 {sorted(allowed)}")


def _apply_bools(
    mapping: dict,
    paths: frozenset[tuple[str, ...]],
    problems: list[str],
) -> None:
    for pattern in sorted(paths):
        for path, value in _resolve(mapping, pattern):
            if value is _MISSING:
                problems.append(f"{path}: 缺少")
            elif type(value) is not bool:
                problems.append(f"{path}: 必须是布尔值，得到 {value!r}")


def _check_constraint(value: object, constraint: Constraint, path: str, problems: list[str]) -> None:
    if constraint.kind == "interval":
        _check_interval_value(value, constraint, path, problems)
        return
    if constraint.kind == "either" and is_interval(value):
        _check_interval_value(value, constraint, path, problems)
        return
    _check_scalar(value, constraint, path, problems)


def _check_scalar(value: object, constraint: Constraint, path: str, problems: list[str]) -> None:
    if constraint.nullable and value is None:
        return
    if is_interval(value) or isinstance(value, (list, tuple)):
        problems.append(f"{path}: 需要标量，得到 {value!r}")
        return
    if isinstance(value, bool) or not _is_number(value):
        problems.append(f"{path}: 需要数字，得到 {value!r}")
        return
    if constraint.integer:
        _check_integer_scalar(value, constraint, path, problems)
        return
    number = _to_float_or_overflow(value)
    if number is _FLOAT_OVERFLOW:
        problems.append(f"{path}: 数值超出有限浮点范围，得到 {value!r}")
        return
    if not math.isfinite(number):
        problems.append(f"{path}: 需要有限数，得到 {value!r}")
        return
    if not _within(number, constraint):
        problems.append(f"{path}: 必须 {_bound_text(constraint)}，得到 {value!r}")


def _check_integer_scalar(value: object, constraint: Constraint, path: str, problems: list[str]) -> None:
    """Exact integer checks. The value is never converted to float for comparison."""
    try:
        number = _strict_int(value, path)
    except ValueError as exc:
        problems.append(str(exc))
        return
    if not _integer_in_finite_float_range(number):
        problems.append(f"{path}: 整数超出有限浮点范围，得到 {value!r}")
        return
    if _sampled_int_out_of_range(path, number):
        problems.append(_sampled_int_range_message(path, value))
        return
    if not _within_int(number, constraint):
        problems.append(f"{path}: 必须 {_bound_text(constraint)}，得到 {value!r}")


def _check_interval_value(value: object, constraint: Constraint, path: str, problems: list[str]) -> None:
    if not is_interval(value):
        problems.append(f"{path}: 需要区间，得到 {value!r}")
        return
    lo_raw, hi_raw = value[0], value[1]
    if constraint.integer:
        _check_integer_interval(value, constraint, path, problems)
        return
    if not _is_number(lo_raw) or not _is_number(hi_raw):
        problems.append(f"{path}: 端点必须是数字，得到 {value!r}")
        return
    lo = _to_float_or_overflow(lo_raw)
    hi = _to_float_or_overflow(hi_raw)
    if lo is _FLOAT_OVERFLOW or hi is _FLOAT_OVERFLOW:
        problems.append(f"{path}: 数值超出有限浮点范围，得到 {value!r}")
        return
    if not math.isfinite(lo) or not math.isfinite(hi):
        problems.append(f"{path}: 端点必须是有限数，得到 {value!r}")
        return
    if lo > hi:
        problems.append(f"{path}: 区间下界大于上界，得到 {value!r}")
        return
    if not uniform_span_is_finite(lo, hi):
        problems.append(f"{path}: 区间跨度必须有限，得到 {value!r}")
        return
    if not _within(lo, constraint) or not _within(hi, constraint):
        problems.append(f"{path}: 端点必须 {_bound_text(constraint)}，得到 {value!r}")


def _check_integer_interval(value: object, constraint: Constraint, path: str, problems: list[str]) -> None:
    """Exact endpoint order and bounds. Integer intervals have no float span check."""
    parsed: list[int] = []
    failed = False
    for endpoint in (value[0], value[1]):
        try:
            parsed.append(_strict_int(endpoint, path))
        except ValueError as exc:
            problems.append(str(exc))
            failed = True
    if failed:
        return
    lo, hi = parsed
    if not _integer_in_finite_float_range(lo) or not _integer_in_finite_float_range(hi):
        problems.append(f"{path}: 整数超出有限浮点范围，得到 {value!r}")
        return
    if lo > hi:
        problems.append(f"{path}: 区间下界大于上界，得到 {value!r}")
        return
    if _sampled_int_out_of_range(path, lo, hi):
        problems.append(_sampled_int_range_message(path, value))
        return
    if not _within_int(lo, constraint) or not _within_int(hi, constraint):
        problems.append(f"{path}: 端点必须 {_bound_text(constraint)}，得到 {value!r}")


def _integer_in_finite_float_range(value: int) -> bool:
    """False when ``float(value)`` cannot produce a finite float.

    Used only to turn that overflow into a collected problem. Comparisons
    use the original ``int``. The conversion is :func:`_to_float_or_overflow`.
    ``math.isfinite`` on a huge int raises ``OverflowError`` as well, so it
    is not the test.
    """
    number = _to_float_or_overflow(value)
    if number is _FLOAT_OVERFLOW:
        return False
    return math.isfinite(number)


def _int_lower_limit(bound: float, open_bound: bool) -> int:
    """Smallest integer ``x`` with ``x > bound`` or ``x >= bound``.

    The bound in the constraint table is a float. ``x`` stays an int:
    closed means ``x >= ceil(bound)``, open means ``x >= floor(bound) + 1``.
    """
    if open_bound:
        return math.floor(bound) + 1
    return math.ceil(bound)


def _int_upper_limit(bound: float, open_bound: bool) -> int:
    """Largest integer ``x`` with ``x < bound`` or ``x <= bound``."""
    if open_bound:
        return math.ceil(bound) - 1
    return math.floor(bound)


def _within_int(number: int, constraint: Constraint) -> bool:
    if constraint.lo is not None and number < _int_lower_limit(constraint.lo, constraint.lo_open):
        return False
    if constraint.hi is not None and number > _int_upper_limit(constraint.hi, constraint.hi_open):
        return False
    return True


def _sampled_int_out_of_range(path: str, *values: int) -> bool:
    if path not in _SAMPLED_INT_DOTTED:
        return False
    return any(value < _INT64_MIN or value > _INT64_MAX for value in values)


def _sampled_int_range_message(path: str, value: object) -> str:
    return f"{path}: 抽样整数必须落在 [{_INT64_MIN}, {_INT64_MAX}]，得到 {value!r}"


def _require_sampled_int64(lo: int, hi: int, spec: object) -> None:
    """Direct sampling entry. ``lo <= hi`` is the caller's responsibility."""
    if lo < _INT64_MIN or hi > _INT64_MAX:
        raise ValueError(
            "sampled integers must lie in "
            f"[{_INT64_MIN}, {_INT64_MAX}], got {spec!r}"
        )


def _within(number: float, constraint: Constraint) -> bool:
    if constraint.lo is not None:
        if constraint.lo_open:
            if number <= constraint.lo:
                return False
        elif number < constraint.lo:
            return False
    if constraint.hi is not None:
        if constraint.hi_open:
            if number >= constraint.hi:
                return False
        elif number > constraint.hi:
            return False
    return True


def _bound_text(constraint: Constraint) -> str:
    lo = constraint.lo
    hi = constraint.hi
    if lo is None and hi is None:
        return "是有限数"
    if lo is not None and hi is None:
        op = ">" if constraint.lo_open else "≥"
        return f"{op} {lo:g}"
    if lo is None and hi is not None:
        op = "<" if constraint.hi_open else "≤"
        return f"{op} {hi:g}"
    left = "(" if constraint.lo_open else "["
    right = ")" if constraint.hi_open else "]"
    return f"落在 {left}{lo:g}, {hi:g}{right}"


def _apply_generator_relations(gen: dict, problems: list[str]) -> None:
    fractions = _lookup_silent(gen, ("dataset", "split", "fractions"))
    if isinstance(fractions, dict):
        _check_fraction_sum(fractions, "dataset.split.fractions", problems)
    tiers = _lookup_silent(gen, ("targets", "tiers"))
    if tiers is not None:
        _check_tiers(tiers, problems)
    _check_exposure_within_frame(gen, problems)
    _check_camera_speed_interval(gen, problems)
    _check_deviation_floor(gen, problems)


def equal_speed_endpoint_message(lo: float, hi: float) -> str:
    """Shared text for a zero-width camera speed interval.

    Config validation and ``simulate_sequence`` both raise this, so the
    wording cannot drift. It states the configuration contract and does not
    claim that a floating-point sample is impossible.
    """
    shown = [float(lo), float(hi)]
    return (
        "camera_motion.speed_px_per_frame: 速度区间必须满足 lo < hi。"
        "零宽区间同时约束名义步进和含抖动的实际位移，不适合作为拒绝采样的接受区间。"
        f"得到 {shown!r}。"
    )


def _check_camera_speed_interval(gen: dict, problems: list[str]) -> None:
    """Reject ``[s, s]`` for the camera speed interval.

    A zero-width interval constrains both the nominal step and the jittered
    displacement, and is not an acceptance interval for rejection sampling.
    The continuous-model derivation is in plan section 4.2. Acceleration may
    still be ``[0, 0]``.
    """
    spec = _lookup_silent(gen, ("camera_motion", "speed_px_per_frame"))
    if not is_interval(spec):
        return
    lo = _relation_float(spec[0])
    hi = _relation_float(spec[1])
    if lo is None or hi is None:
        return
    if not math.isfinite(lo) or not math.isfinite(hi):
        return
    if lo != hi:
        return
    problems.append(equal_speed_endpoint_message(lo, hi))


def _number_span(spec: object) -> tuple[float, float] | None:
    """Lower and upper ends of a scalar or an interval.

    ``None`` means a cross-field check should skip this operand. The
    constraint table has already reported a value that is not a finite number.
    """
    if is_interval(spec):
        lo = _relation_float(spec[0])
        hi = _relation_float(spec[1])
        if lo is None or hi is None:
            return None
        if not math.isfinite(lo) or not math.isfinite(hi):
            return None
        return lo, hi
    number = _relation_float(spec)
    if number is None or not math.isfinite(number):
        return None
    return number, number


def exceeds_cap(value: float, cap: float, *, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    """True when ``value`` is above ``cap`` by more than the shared tolerance.

    Config validation and ``targets._capped_uniform`` both call this with
    ``rel_tol=1e-9`` and ``abs_tol=0``. A floor that is only float dust above
    the cap is not an empty intersection.
    """
    return value > cap and not math.isclose(value, cap, rel_tol=rel_tol, abs_tol=abs_tol)


def _check_deviation_floor(gen: dict, problems: list[str]) -> None:
    """Reject a deviation floor above ``rand_rel_cap`` times the class speed floor.

    A point class is enabled when ``streak_fraction`` can be below 1, and a
    streak class when it can be above 0. ``strict_fraction`` of 1 draws no
    deviation, so this rule does not apply. One infeasible speed rejects the
    whole config.
    """
    streak = _number_span(_lookup_silent(gen, ("targets", "streak_fraction")))
    strict = _number_span(_lookup_silent(gen, ("targets", "track_deviation", "strict_fraction")))
    cap_span = _number_span(_lookup_silent(gen, ("targets", "track_deviation", "rand_rel_cap")))
    if streak is None or strict is None or cap_span is None:
        return
    if strict[0] >= 1.0:
        return
    classes: list[str] = []
    if streak[0] < 1.0:
        classes.append("point")
    if streak[1] > 0.0:
        classes.append("streak")
    cap_factor = cap_span[0]
    for klass in classes:
        speed = _number_span(_lookup_silent(gen, ("targets", klass, "speed_px_per_frame")))
        if speed is None:
            continue
        cap = cap_factor * speed[0]
        for field in ("rand_cross_px", "rand_along_px"):
            bounds = _number_span(_lookup_silent(gen, ("targets", "track_deviation", field)))
            if bounds is None:
                continue
            floor = bounds[0]
            if exceeds_cap(floor, cap, rel_tol=1e-9, abs_tol=0.0):
                shown_floor = format(floor, ".12g")
                shown_cap = format(cap, ".12g")
                shown_factor = format(cap_factor, ".12g")
                shown_speed = format(speed[0], ".12g")
                problems.append(
                    f"targets.{klass}.{field}: 下界 {shown_floor} 大于 cap {shown_cap}"
                    f"（rand_rel_cap {shown_factor} × {klass}.speed_px_per_frame 的下界 {shown_speed}）。"
                )


def _check_out_dir(gen: dict, problems: list[str]) -> None:
    path, value = _resolve(gen, ("dataset", "out_dir"))[0]
    if value is _MISSING:
        problems.append(f"{path}: 缺少")
        return
    if not isinstance(value, str) or value == "":
        problems.append(f"{path}: 必须是非空字符串，得到 {value!r}")


def _check_snr_window(gen: dict, problems: list[str]) -> None:
    path, window = _resolve(gen, ("labels", "snr_window"))[0]
    if window is _MISSING:
        problems.append(f"{path}: 缺少")
        return
    if window is None:
        problems.append(f"{path}: 必须是 'all' 或正整数，得到 None")
        return
    if window == "all":
        return
    try:
        count = _strict_int(window, path)
    except ValueError as exc:
        problems.append(str(exc))
        return
    if count < 1:
        problems.append(f"{path}: 必须是 'all' 或正整数，得到 {count}")


def _check_fraction_sum(group: object, path: str, problems: list[str]) -> None:
    if not isinstance(group, dict) or not group:
        problems.append(f"{path}: 需要一组非负比例")
        return
    total = 0.0
    for value in group.values():
        number = _relation_float(value)
        if number is None or not math.isfinite(number) or not 0.0 <= number <= 1.0:
            return
        total += number
    if abs(total - 1.0) >= 1e-9:
        problems.append(f"{path}: 比例之和为 {total}，与 1 的差超过 1e-9")


def _check_tiers(tiers: object, problems: list[str]) -> None:
    if not isinstance(tiers, dict) or not tiers:
        problems.append("targets.tiers: 需要至少一个档位")
        return
    fracs: list[float] = []
    intervals: list[tuple[float, float, str]] = []
    fracs_ok = True
    for name, tier in tiers.items():
        if not isinstance(tier, dict):
            problems.append(f"targets.tiers.{name}: 期望映射，得到 {type(tier).__name__}")
            fracs_ok = False
            continue
        frac = tier.get("frac")
        frac_number = _relation_float(frac)
        if frac_number is not None and math.isfinite(frac_number) and 0.0 <= frac_number <= 1.0:
            fracs.append(frac_number)
        else:
            fracs_ok = False
        snr = tier.get("snr")
        if not is_interval(snr):
            continue
        lo = _relation_float(snr[0])
        hi = _relation_float(snr[1])
        if lo is None or hi is None:
            continue
        if math.isfinite(lo) and math.isfinite(hi) and lo <= hi:
            intervals.append((lo, hi, name))
    if fracs_ok and abs(sum(fracs) - 1.0) >= 1e-9:
        problems.append(f"targets.tiers: 比例之和为 {sum(fracs)}，与 1 的差超过 1e-9")
    intervals.sort(key=lambda item: (item[0], item[1], item[2]))
    for (lo, hi, name), (lo_next, hi_next, name_next) in zip(intervals, intervals[1:]):
        if lo_next < hi:
            problems.append(
                f"targets.tiers.{name}.snr [{lo:g}, {hi:g}] 与 "
                f"targets.tiers.{name_next}.snr [{lo_next:g}, {hi_next:g}] 重叠"
            )


def _check_frames_and_reference(gen: dict, problems: list[str]) -> None:
    n_lo = _optional_lower_int(_lookup_silent(gen, ("dataset", "n_frames")))
    min_visible_int = _optional_int(_lookup_silent(gen, ("targets", "min_visible_frames")))
    if n_lo is not None and min_visible_int is not None and n_lo < min_visible_int:
        problems.append(
            f"dataset.n_frames: 下界 {n_lo} 小于 targets.min_visible_frames {min_visible_int}"
        )
    path, reference = _resolve(gen, ("camera_motion", "reference_frame"))[0]
    if reference is _MISSING:
        problems.append(f"{path}: 缺少")
        return
    if reference is None:
        if n_lo is None:
            problems.append(f"{path}: 必须是 'middle' 或非负整数帧号，得到 None")
        else:
            problems.append(f"{path}: 必须是 'middle' 或 0 <= 帧号 < {n_lo}，得到 None")
        return
    if reference == "middle":
        return
    try:
        frame = _strict_int(reference, path)
    except ValueError as exc:
        problems.append(str(exc))
        return
    if n_lo is None:
        return
    if frame < 0 or frame >= n_lo:
        problems.append(f"{path}: 必须是 'middle' 或 0 <= 帧号 < {n_lo}，得到 {frame}")


def _check_exposure_within_frame(gen: dict, problems: list[str]) -> None:
    frame = _relation_float(_lookup_silent(gen, ("timing", "frame_interval_s")))
    exposure = _relation_float(_lookup_silent(gen, ("timing", "exposure_s_nominal")))
    if frame is None or exposure is None:
        return
    if exposure > frame:
        problems.append(
            f"timing.exposure_s_nominal: {exposure:g} 大于 "
            f"timing.frame_interval_s {frame:g}"
        )


def _check_spectral_order(camera: dict, problems: list[str]) -> None:
    low = _relation_float(_lookup_silent(camera, ("sensor", "spectral_range_nm", "min")))
    high = _relation_float(_lookup_silent(camera, ("sensor", "spectral_range_nm", "max")))
    if low is None or high is None:
        return
    if low >= high:
        problems.append(
            "sensor.spectral_range_nm: min 必须小于 max，"
            f"得到 min={low:g}, max={high:g}"
        )


def _lookup_silent(mapping: dict, path: tuple[str, ...]) -> object | None:
    node: object = mapping
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return _lenient_int(value, value)
    except ValueError:
        return None


def _optional_lower_int(spec: object) -> int | None:
    if spec is None:
        return None
    try:
        return _lower_int(spec)
    except ValueError:
        return None


_CUSTOM_DISPATCH = {
    ("dataset", "out_dir"): _check_out_dir,
    ("camera_motion", "reference_frame"): _check_frames_and_reference,
    ("labels", "snr_window"): _check_snr_window,
}
assert set(_CUSTOM_DISPATCH) == set(_CUSTOM_PATHS)


def _lower_int(spec: object) -> int:
    if is_interval(spec):
        lo = _lenient_int(spec[0], spec)
        hi = _lenient_int(spec[1], spec)
        if hi < lo:
            raise ValueError(f"interval lower bound exceeds upper bound: {spec!r}")
        return lo
    return _lenient_int(spec, spec)


def _require_generator(rng: object) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")


def _is_number(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _check_bounds(lo: float, hi: float, spec: object) -> None:
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError(f"interval bounds must be finite, got {spec!r}")
    if hi < lo:
        raise ValueError(f"interval lower bound exceeds upper bound: {spec!r}")


def _strict_int(value: object, path: str) -> int:
    """配置校验专用：要求 type(value) is int。

    会拒绝 4.0、True、np.int64。YAML 和 CLI overrides 产出的整数都是 Python int；
    如果要把 numpy 计算结果传进 overrides，先转成 int()。
    """
    if type(value) is not int:
        raise ValueError(f"{path}: 需要整数，得到 {type(value).__name__} {value!r}")
    return value


def _lenient_int(value: object, spec: object) -> int:
    """Accept integral floats. Sampling and cross-field checks use this.

    ``int`` and ``np.integer`` stay exact: ``int(value)`` never passes
    through ``float``. Only a floating-point input is tested for being
    finite and integral. Config validation uses ``_strict_int``, which
    already rejects floats, so the float branch is reached only by a
    direct caller. ``_optional_int`` and ``_optional_lower_int`` call
    this function, so they keep the same exact integers.

    New config fields must go through ``_strict_int``. This looser check
    stays only so a cross-field rule does not report the same value again
    after the constraint table already rejected it.
    """
    if isinstance(value, (bool, np.bool_)) or not _is_number(value):
        raise ValueError(f"expected an integer bound, got {spec!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"expected an integer bound, got {spec!r}")
    return int(number)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(value, (int, np.integer)):
        number = int(value)
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        number = int(value)
    else:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {number}")
    return number


def _positive_float(value: object, name: str) -> float:
    if not _is_number(value):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    number = _to_float_or_overflow(value)
    if number is _FLOAT_OVERFLOW or not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number
