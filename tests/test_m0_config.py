"""M0 acceptance tests for configuration loading and sequence sampling."""

from __future__ import annotations

import copy
import math
import warnings

import numpy as np
import pytest

import starsim.config as config_mod
from starsim.config import (
    equal_speed_endpoint_message,
    deep_merge,
    is_interval,
    load_config,
    sample_int_inclusive,
    sample_log_uniform,
    sample_sequence_params,
    sample_uniform,
)
from starsim.rng import SequenceRng, root_seed, sequence_seed


@pytest.fixture
def cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return load_config()


def _generator(rng_seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(rng_seed))


def _at(mapping: dict, path: tuple[str, ...]):
    node = mapping
    for key in path:
        node = node[key]
    return node


def _contained(value, spec) -> bool:
    if is_interval(spec):
        return float(spec[0]) <= float(value) <= float(spec[1]) and not isinstance(value, list)
    return value == spec


def test_default_derived_fov_warns_and_overwrites_handwritten_value():
    with pytest.warns(UserWarning, match=r"超过 1%.*6\.5") as caught:
        loaded = load_config()
    assert len(caught) == 1
    assert round(loaded.derived.fov_width_deg, 3) == 7.032
    assert round(loaded.derived.fov_height_deg, 3) == 7.032
    assert round(loaded.derived.pixel_scale_arcsec, 3) == 6.188
    assert round(loaded.derived.half_diagonal_deg, 2) == 4.97

    derived = loaded.camera["derived"]
    assert derived["fov_deg"]["width"] == pytest.approx(loaded.derived.fov_width_deg)
    assert derived["fov_deg"]["height"] == pytest.approx(loaded.derived.fov_height_deg)
    assert derived["fov_deg"]["width"] > 7.0
    assert loaded.derived.cx == pytest.approx((4096 - 1) / 2.0)
    assert loaded.derived.cy == pytest.approx((4096 - 1) / 2.0)
    assert loaded.derived.f_over_p == pytest.approx(300.0 / 0.009)
    np.testing.assert_allclose(
        loaded.derived.K,
        [
            [loaded.derived.f_over_p, 0.0, loaded.derived.cx],
            [0.0, loaded.derived.f_over_p, loaded.derived.cy],
            [0.0, 0.0, 1.0],
        ],
    )
    assert loaded.derived.K.shape == (3, 3)
    assert loaded.camera["sensor"]["resolution"] == {"width": 4096, "height": 4096}
    assert loaded.generator["pointing"]["galactic_lat_min_deg"] is None


def test_fov_warning_threshold(cfg):
    width = cfg.derived.fov_width_deg
    height = cfg.derived.fov_height_deg
    matching = {"camera": {"derived": {"fov_deg": {"width": width, "height": height}}}}
    within = {
        "camera": {
            "derived": {"fov_deg": {"width": width * 1.005, "height": height * 1.005}}
        }
    }
    outside = {"camera": {"derived": {"fov_deg": {"width": width * 1.02, "height": height}}}}
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        load_config(overrides=matching)
        load_config(overrides=within)
    with pytest.warns(UserWarning, match=r"超过 1%"):
        overridden = load_config(overrides=outside)
    assert overridden.derived.fov_width_deg == pytest.approx(width)


def test_deep_merge_replaces_lists_and_does_not_mutate_base():
    base = {"a": {"b": 1, "c": [1, 2]}, "d": 3}
    merged = deep_merge(base, {"a": {"c": [9], "e": 4}})
    assert merged == {"a": {"b": 1, "c": [9], "e": 4}, "d": 3}
    merged["a"]["c"].append(8)
    assert base == {"a": {"b": 1, "c": [1, 2]}, "d": 3}

    replaced = deep_merge({"a": {"b": [1, 2], "keep": 1}}, {"a": {"b": [9]}})
    assert replaced == {"a": {"b": [9], "keep": 1}}


def test_interval_sampling_primitives():
    rng = _generator(0)
    assert is_interval([1, 2])
    assert is_interval((0.0, 1.0))
    assert not is_interval([1, 2, 3])
    assert not is_interval(3)
    assert not is_interval([True, False])

    assert sample_uniform(rng, 4) == 4.0
    assert sample_log_uniform(rng, 4) == 4.0
    assert sample_int_inclusive(rng, 4) == 4
    assert sample_int_inclusive(_generator(0), [8, 8]) == 8

    inclusive = _generator(1)
    draws = [sample_int_inclusive(inclusive, [8, 16]) for _ in range(5000)]
    assert min(draws) == 8
    assert max(draws) == 16

    uniform_rng = _generator(2)
    uniform = np.array([sample_uniform(uniform_rng, [0.0, 1.0]) for _ in range(8000)])
    assert uniform.min() >= 0.0
    assert uniform.max() <= 1.0
    assert uniform.mean() == pytest.approx(0.5, abs=0.02)

    log_rng = _generator(3)
    log_samples = np.array([sample_log_uniform(log_rng, [50.0, 3000.0]) for _ in range(8000)])
    assert log_samples.min() >= 50.0
    assert log_samples.max() <= 3000.0
    assert np.median(log_samples) == pytest.approx(np.sqrt(50.0 * 3000.0), rel=0.08)

    with pytest.raises(ValueError):
        sample_uniform(_generator(0), [2, 1])
    with pytest.raises(ValueError):
        sample_log_uniform(_generator(0), [0, 10])
    with pytest.raises(TypeError):
        sample_uniform(np.random.RandomState(0), [0, 1])


def test_effective_config_samples_scalars_and_keeps_motion_ranges(cfg):
    before = copy.deepcopy(cfg.generator)
    root = root_seed(20261027)
    effective = sample_sequence_params(cfg, SequenceRng(sequence_seed(root, 0)))
    again = sample_sequence_params(cfg, SequenceRng(sequence_seed(root, 0)))
    assert effective == again
    assert cfg.generator == before

    assert effective["dataset"]["n_frames"] == 10
    assert _contained(effective["photometry"]["mag_limit"], before["photometry"]["mag_limit"])
    assert isinstance(effective["photometry"]["mag_limit"], float)
    assert _contained(effective["targets"]["per_frame"], before["targets"]["per_frame"])
    assert isinstance(effective["targets"]["per_frame"], int)
    assert _contained(effective["camera_motion"]["jitter_px"], before["camera_motion"]["jitter_px"])
    assert isinstance(effective["camera_motion"]["jitter_px"], float)
    assert _contained(effective["background"]["sky_e"], before["background"]["sky_e"])

    assert effective["camera_motion"]["speed_px_per_frame"] == [5, 100]
    assert effective["camera_motion"]["accel_px_per_frame2"] == [-2, 2]
    assert effective["targets"]["point"]["speed_px_per_frame"] == [3, 50]
    assert effective["targets"]["streak"]["duty"] == [0.2, 0.9]
    assert effective["targets"]["tiers"]["bright"]["snr"] == [10, 300]
    assert effective["targets"]["tiers"]["faint"]["snr"] == [4, 10]
    assert effective["targets"]["tiers"]["ultra"]["snr"] == [1.5, 3]
    assert effective["artifacts"]["rts_p_flip"] == [0.1, 0.3]
    assert effective["artifacts"]["hot_cluster_size_px"] == [2, 3]
    assert effective["labels"]["label_frame"] == before["labels"]["label_frame"]
    assert effective["sensor_noise"]["adc_bits"] == 16


def test_removing_one_sampled_path_does_not_change_the_others(cfg, monkeypatch):
    root = root_seed(20261027)
    full = sample_sequence_params(cfg, SequenceRng(sequence_seed(root, 4)))
    paths = list(config_mod._SEQUENCE_UNIFORM_PATHS)
    dropped = paths[len(paths) // 2]
    monkeypatch.setattr(
        config_mod,
        "_SEQUENCE_UNIFORM_PATHS",
        tuple(path for path in paths if path != dropped),
    )
    partial = sample_sequence_params(cfg, SequenceRng(sequence_seed(root, 4)))
    for path in config_mod._SAMPLED_PATHS:
        if path == dropped:
            continue
        assert _at(partial, path) == _at(full, path)
    assert _at(partial, dropped) == _at(cfg.generator, dropped)


def test_sky_background_follows_log_uniform_flag(cfg):
    root = root_seed(7)
    log_values = np.array(
        [
            sample_sequence_params(cfg, SequenceRng(sequence_seed(root, index)))["background"]["sky_e"]
            for index in range(4000)
        ]
    )
    assert log_values.min() >= 50.0
    assert log_values.max() <= 3000.0
    assert np.median(log_values) == pytest.approx(np.sqrt(50.0 * 3000.0), rel=0.1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        linear_cfg = load_config(
            overrides={"generator": {"background": {"log_uniform": False}}}
        )
    linear_root = root_seed(8)
    linear_values = np.array(
        [
            sample_sequence_params(linear_cfg, SequenceRng(sequence_seed(linear_root, index)))["background"]["sky_e"]
            for index in range(4000)
        ]
    )
    assert linear_values.mean() == pytest.approx((50.0 + 3000.0) / 2.0, rel=0.05)


def test_unknown_keys_fractions_enums_and_tier_overlap_are_rejected():
    with pytest.raises(ValueError, match=r"generator\.photomtry"):
        load_config(overrides={"generator": {"photomtry": {"mag_limit": [10, 12]}}})
    with pytest.raises(ValueError, match=r"camera\.derived\.pixel_scale_arcsec"):
        load_config(overrides={"camera": {"derived": {"pixel_scale_arcsec": 1.0}}})
    with pytest.raises(ValueError, match=r"dataset\.split\.fractions") as fractions:
        load_config(
            overrides={
                "generator": {
                    "dataset": {"split": {"fractions": {"train": 0.5, "val": 0.5, "test": 0.5}}}
                }
            }
        )
    assert "1.5" in str(fractions.value)
    with pytest.raises(ValueError, match=r"dataset\.image_format") as both:
        load_config(
            overrides={
                "generator": {
                    "dataset": {
                        "image_format": "jpeg",
                        "split": {"fractions": {"train": 0.5, "val": 0.5, "test": 0.5}},
                    }
                }
            }
        )
    text = str(both.value)
    assert "dataset.image_format" in text
    assert "dataset.split.fractions" in text
    with pytest.raises(ValueError, match="重叠"):
        load_config(
            overrides={"generator": {"targets": {"tiers": {"ultra": {"snr": [1.5, 5]}}}}}
        )
    with pytest.raises(ValueError, match="min_visible_frames"):
        load_config(overrides={"generator": {"dataset": {"n_frames": [1, 4]}}})


def test_invalid_sensor_and_override_keys_raise():
    with pytest.raises(ValueError):
        load_config(overrides={"camera": {"sensor": {"pixel_size_mm": -1}}})
    with pytest.raises(ValueError):
        load_config(overrides={"sensor": {}})


def _nest(path: str, value):
    keys = path.split(".")
    node: dict = {}
    cursor = node
    for key in keys[:-1]:
        child: dict = {}
        cursor[key] = child
        cursor = child
    cursor[keys[-1]] = value
    return node


def _leaves(node, prefix=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, prefix + (str(key),))
        return
    yield prefix


def _matches(path: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    if len(path) != len(pattern):
        return False
    return all(item == "*" or item == concrete for item, concrete in zip(pattern, path))


def _assert_coverage(tree: dict, categories: tuple[tuple[str, set], ...]) -> None:
    leaves = list(_leaves(tree))
    for leaf in leaves:
        hits = [
            name
            for name, patterns in categories
            if any(_matches(leaf, pattern) for pattern in patterns)
        ]
        assert hits and len(hits) == 1, f"{'.'.join(leaf)} -> {hits}"
    for name, patterns in categories:
        for pattern in patterns:
            assert any(_matches(leaf, pattern) for leaf in leaves), (name, pattern)


def test_default_yaml_leaves_match_constraint_tables():
    generator = config_mod.load_yaml(config_mod.DEFAULT_GENERATOR_PATH)
    camera = config_mod.load_yaml(config_mod.DEFAULT_CAMERA_PATH)
    _assert_coverage(
        generator,
        (
            ("constraints", set(config_mod._CONSTRAINTS)),
            ("enums", set(config_mod._ENUMS)),
            ("bools", set(config_mod._BOOL_PATHS)),
            ("custom", set(config_mod._CUSTOM_PATHS)),
            ("exempt", set(config_mod._EXEMPT)),
        ),
    )
    _assert_coverage(
        camera,
        (
            ("constraints", set(config_mod._CAMERA_CONSTRAINTS)),
            ("enums", set(config_mod._CAMERA_ENUMS)),
            ("bools", set(config_mod._CAMERA_BOOL_PATHS)),
            ("custom", set(config_mod._CAMERA_CUSTOM_PATHS)),
            ("exempt", set(config_mod._CAMERA_EXEMPT)),
        ),
    )


_REJECTED = (
    ("generator", "camera_motion.speed_px_per_frame", [100, 5]),
    ("generator", "camera_motion.accel_px_per_frame2", [2, -2]),
    ("generator", "targets.point.speed_px_per_frame", [50, 3]),
    ("generator", "targets.streak.speed_px_per_frame", [100, 20]),
    ("generator", "targets.streak.duty", [0.9, 0.2]),
    ("generator", "targets.point.smear_px", [1.0, 0.0]),
    ("generator", "targets.streak.angle_jitter_deg", [2, 0]),
    ("generator", "targets.track_deviation.curve_amp_px", [2, 0]),
    ("generator", "targets.track_deviation.rand_cross_px", [1.0, 0]),
    ("generator", "artifacts.hot_cluster_size_px", [3, 2]),
    ("generator", "psf.sigma_px", [-1, 1.5]),
    ("generator", "sensor_noise.prnu", [-0.02, 0.02]),
    ("generator", "camera_motion.jitter_px", [-0.3, 0.3]),
    ("generator", "photometry.mag_limit", [-5, 12]),
    ("generator", "targets.tiers.ultra.snr", [-1, 3]),
    ("generator", "timing.exposure_s_nominal", 5.0),
    ("generator", "psf.max_ellipticity", 1),
    ("generator", "background.vignetting_corner", [0, 1]),
    ("generator", "targets.streak.duty", [0, 0.5]),
    ("generator", "dataset.split.cap_radius_deg", 0),
    ("camera", "derived.fov_deg.width", -1),
    ("camera", "sensor.aperture_mm", 0),
    ("camera", "sensor.spectral_range_nm.min", 900),
)


@pytest.mark.parametrize(("section", "path", "value"), _REJECTED)
def test_constraint_counterexamples_are_rejected(section, path, value):
    with pytest.raises(ValueError) as caught:
        load_config(overrides={section: _nest(path, value)})
    text = str(caught.value)
    assert path in text or path.rsplit(".", 1)[0] in text


_BOOL_FIELDS = (
    "dataset.export_registered",
    "dataset.split.stratify_by_glat",
    "camera_motion.slew_during_exposure",
    "background.log_uniform",
    "artifacts.blooming",
    "labels.save_dense",
    "labels.per_class_offset",
    "labels.streak_geom",
    "labels.streak_line",
    "labels.motion",
    "synthetic_faint_stars.enabled",
)


@pytest.mark.parametrize("path", _BOOL_FIELDS)
def test_bool_fields_reject_non_bools(path):
    with pytest.raises(ValueError, match="布尔"):
        load_config(overrides={"generator": _nest(path, "no")})


def test_equal_camera_speed_endpoints_are_rejected_and_the_reason_is_stated():
    with pytest.raises(ValueError) as caught:
        load_config(overrides={"generator": {"camera_motion": {"speed_px_per_frame": [50, 50]}}})
    text = str(caught.value)
    assert equal_speed_endpoint_message(50.0, 50.0) in text
    assert "lo < hi" in text
    assert "不适合作为拒绝采样的接受区间" in text
    assert "f·tan" not in text
    assert "概率" not in text
    assert "不可能" not in text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        load_config(
            overrides={
                "generator": {
                    "camera_motion": {
                        "speed_px_per_frame": [50, 50.1],
                        "accel_px_per_frame2": [0, 0],
                    }
                }
            }
        )


def test_open_and_closed_endpoints_that_are_inside_pass():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        load_config(
            overrides={
                "generator": {
                    "psf": {"max_ellipticity": 0},
                    "dataset": {"split": {"cap_radius_deg": 90}},
                    "targets": {"streak": {"duty": [1, 1]}},
                    "pointing": {"galactic_lat_min_deg": 90},
                }
            }
        )
        load_config(overrides={"generator": {"pointing": {"galactic_lat_min_deg": None}}})


_STRICT_TYPES = (
    ("generator", "dataset.image_format", ["fits_rice"]),
    ("generator", "labels.label_frame", {"a": 1}),
    ("generator", "sensor_noise.adc_bits", 16.0),
    ("generator", "labels.stride", 4.0),
    ("generator", "dataset.n_sequences", 500.0),
    ("generator", "artifacts.hot_cluster_size_px", [2.0, 3]),
    ("generator", "labels.snr_window", 4.0),
    ("generator", "camera_motion.reference_frame", 4.0),
)


@pytest.mark.parametrize(("section", "path", "value"), _STRICT_TYPES)
def test_strict_types_and_unhashable_enums_are_rejected(section, path, value):
    with pytest.raises(ValueError) as caught:
        load_config(overrides={section: _nest(path, value)})
    assert path in str(caught.value)


def test_enum_type_error_is_aggregated_with_another_field():
    with pytest.raises(ValueError) as caught:
        load_config(
            overrides={
                "generator": {
                    "dataset": {"image_format": ["fits_rice"]},
                    "labels": {"stride": 4.0},
                }
            }
        )
    text = str(caught.value)
    assert "dataset.image_format" in text
    assert "labels.stride" in text


@pytest.mark.parametrize(
    "path",
    ["dataset.out_dir", "labels.snr_window", "camera_motion.reference_frame"],
)
def test_explicit_null_on_custom_fields_is_rejected(path):
    with pytest.raises(ValueError) as caught:
        load_config(overrides={"generator": _nest(path, None)})
    text = str(caught.value)
    assert path in text
    assert "None" in text


def test_sampling_twice_on_one_sequence_rng_is_rejected(cfg):
    rngs = SequenceRng(sequence_seed(root_seed(1), 0))
    sample_sequence_params(cfg, rngs)
    with pytest.raises(ValueError, match="已发放过一次"):
        sample_sequence_params(cfg, rngs)


def test_nonfinite_float_span_is_reported_with_the_field_path():
    with pytest.raises(ValueError) as caught:
        load_config(
            overrides={
                "generator": {
                    "camera_motion": {"accel_px_per_frame2": [-1e308, 1e308]},
                    "labels": {"stride": 4.0},
                }
            }
        )
    text = str(caught.value)
    assert "camera_motion.accel_px_per_frame2" in text
    assert "区间跨度必须有限" in text
    assert "[-1e+308, 1e+308]" in text
    assert "labels.stride" in text
    assert not config_mod.uniform_span_is_finite(-1e308, 1e308)
    assert config_mod.uniform_span_is_finite(-1e300, 1e300)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        load_config(
            overrides={"generator": {"camera_motion": {"accel_px_per_frame2": [-1e300, 1e300]}}}
        )
        load_config(overrides={"generator": {"background": {"sky_e": [1e-300, 1e300]}}})
    drawn = sample_log_uniform(_generator(0), [1e-300, 1e300])
    assert 1e-300 <= drawn <= 1e300

    with pytest.raises(ValueError, match=r"\[-1e\+308, 1e\+308\]"):
        sample_uniform(_generator(0), [-1e308, 1e308])
    ordinary = sample_uniform(_generator(1), [-1e300, 1e300])
    assert -1e300 <= ordinary <= 1e300


def test_large_exact_integer_survives_sampling_and_lenient_int():
    exact = 9007199254740993  # 2**53 + 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = load_config(overrides={"generator": {"targets": {"per_frame": exact}}})
    assert loaded.generator["targets"]["per_frame"] == exact
    effective = sample_sequence_params(loaded, SequenceRng(sequence_seed(root_seed(1), 3)))
    assert effective["targets"]["per_frame"] == exact
    assert config_mod._lenient_int(2**63 - 1, 2**63 - 1) == 2**63 - 1
    assert config_mod._lenient_int(np.int64(2**63 - 1), np.int64(2**63 - 1)) == 2**63 - 1
    assert config_mod._lenient_int(4.0, 4.0) == 4
    with pytest.raises(ValueError):
        config_mod._lenient_int(4.5, 4.5)


def test_integer_constraint_compares_exactly_and_aggregates_overflow():
    with pytest.raises(ValueError) as caught:
        load_config(
            overrides={
                "generator": {
                    "dataset": {
                        "n_sequences": 10**400,
                        "n_frames": [2**53 + 1, 2**53],
                    }
                }
            }
        )
    text = str(caught.value)
    assert "dataset.n_sequences" in text
    assert "整数超出有限浮点范围" in text
    assert "dataset.n_frames" in text
    assert "下界大于上界" in text

    closed = config_mod.Constraint("scalar", integer=True, lo=2.5, hi=10.5)
    assert config_mod._within_int(3, closed)
    assert not config_mod._within_int(2, closed)
    assert config_mod._within_int(10, closed)
    assert not config_mod._within_int(11, closed)
    opened = config_mod.Constraint(
        "scalar", integer=True, lo=2.0, hi=10.0, lo_open=True, hi_open=True
    )
    assert config_mod._within_int(3, opened)
    assert not config_mod._within_int(2, opened)
    assert config_mod._within_int(9, opened)
    assert not config_mod._within_int(10, opened)
    negative = config_mod.Constraint("scalar", integer=True, lo=-1.5, hi=-0.5)
    assert config_mod._within_int(-1, negative)
    assert not config_mod._within_int(-2, negative)
    assert not config_mod._within_int(0, negative)


def test_sampled_integers_keep_the_int64_range_and_seed_does_not():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = load_config(
            overrides={
                "generator": {
                    "dataset": {"seed": 2**100, "n_sequences": 2**63, "n_frames": 2**63 - 1},
                    "targets": {"per_frame": [2**63 - 1, 2**63 - 1]},
                }
            }
        )
    assert loaded.generator["dataset"]["seed"] == 2**100
    assert loaded.generator["dataset"]["n_sequences"] == 2**63
    assert loaded.generator["dataset"]["n_frames"] == 2**63 - 1

    with pytest.raises(ValueError, match="抽样整数") as high:
        load_config(overrides={"generator": {"targets": {"per_frame": 2**63}}})
    assert "targets.per_frame" in str(high.value)
    with pytest.raises(ValueError, match="抽样整数") as low:
        load_config(
            overrides={"generator": {"dataset": {"n_frames": [-2**63 - 1, 4]}}}
        )
    assert "dataset.n_frames" in str(low.value)

    assert sample_int_inclusive(_generator(0), 2**63 - 1) == 2**63 - 1
    assert sample_int_inclusive(_generator(0), -2**63) == -2**63
    assert sample_int_inclusive(_generator(0), [2**63 - 1, 2**63 - 1]) == 2**63 - 1
    assert sample_int_inclusive(_generator(0), [-2**63, -2**63]) == -2**63
    high_draws = [
        sample_int_inclusive(_generator(i), [2**63 - 5, 2**63 - 1]) for i in range(40)
    ]
    low_draws = [
        sample_int_inclusive(_generator(i), [-2**63, -2**63 + 5]) for i in range(40)
    ]
    assert min(high_draws) >= 2**63 - 5 and max(high_draws) <= 2**63 - 1
    assert min(low_draws) >= -2**63 and max(low_draws) <= -2**63 + 5
    for spec in (
        2**63,
        -2**63 - 1,
        [0, 2**63],
        [-2**63 - 1, 0],
        [2**63, 2**63],
        [-2**63 - 1, -2**63 - 1],
    ):
        with pytest.raises(ValueError, match="sampled integers"):
            sample_int_inclusive(_generator(0), spec)


def test_endpoint_true_matches_exclusive_upper_bound_bit_for_bit():
    """Lock the current-environment identity of the closed integer draw.

    ``integers(lo, hi, endpoint=True)`` and ``integers(lo, hi + 1)`` agreed
    on values and on the PCG64 state for these intervals. The full int64
    interval is included. ``hi + 1`` stays in the test; production code uses
    ``endpoint=True`` so the upper endpoint does not have to be incremented.
    """
    intervals = (
        (1, 10),
        (0, 1),
        (-5, 5),
        (8, 16),
        (-2**63, -2**63 + 10),
        (2**63 - 10, 2**63 - 1),
        (1, 2**63 - 1),
        (-2**63, 2**63 - 1),
    )
    for seed in range(200):
        for lo, hi in intervals:
            reference = np.random.Generator(np.random.PCG64(seed))
            closed = np.random.Generator(np.random.PCG64(seed))
            helper = np.random.Generator(np.random.PCG64(seed))
            reference_draws = [int(reference.integers(lo, hi + 1)) for _ in range(4)]
            closed_draws = [int(closed.integers(lo, hi, endpoint=True)) for _ in range(4)]
            helper_draws = [sample_int_inclusive(helper, [lo, hi]) for _ in range(4)]
            assert reference_draws == closed_draws == helper_draws, (seed, lo, hi)
            assert reference.bit_generator.state == closed.bit_generator.state
            assert reference.bit_generator.state == helper.bit_generator.state


def test_float_fields_reject_integers_outside_the_finite_range():
    huge = 10**400
    cases = (
        ("timing.frame_interval_s", huge),
        ("camera_motion.speed_px_per_frame", [1, huge]),
        ("camera_motion.accel_px_per_frame2", [-huge, 1]),
        ("background.sky_e", [50, huge]),
    )
    for path, value in cases:
        with pytest.raises(ValueError) as caught:
            load_config(overrides={"generator": _nest(path, value)})
        text = str(caught.value)
        assert path in text
        assert "数值超出有限浮点范围" in text
        assert "OverflowError" not in text

    with pytest.raises(ValueError) as caught:
        load_config(
            overrides={
                "generator": {
                    "timing": {"frame_interval_s": huge},
                    "labels": {"stride": 4.0},
                }
            }
        )
    text = str(caught.value)
    assert text.count("timing.frame_interval_s") == 1
    assert "数值超出有限浮点范围" in text
    assert "labels.stride" in text
    assert "timing.exposure_s_nominal" not in text

    with pytest.raises(ValueError) as caught:
        load_config(
            overrides={
                "generator": {
                    "timing": {"exposure_s_nominal": huge},
                    "labels": {"stride": 4.0},
                }
            }
        )
    text = str(caught.value)
    assert text.count("timing.exposure_s_nominal") == 1
    assert "大于" not in text
    assert "labels.stride" in text

    with pytest.raises(ValueError) as caught:
        load_config(
            overrides={
                "generator": {
                    "camera_motion": {"speed_px_per_frame": [1, huge]},
                    "labels": {"stride": 4.0},
                }
            }
        )
    text = str(caught.value)
    assert text.count("camera_motion.speed_px_per_frame") == 1
    assert "不适合作为拒绝采样的接受区间" not in text
    assert "labels.stride" in text

    with pytest.raises(ValueError) as caught:
        load_config(
            overrides={
                "camera": {
                    "sensor": {
                        "pixel_size_mm": huge,
                        "aperture_mm": 0,
                        "spectral_range_nm": {"min": huge},
                    }
                }
            }
        )
    text = str(caught.value)
    assert text.count("sensor.pixel_size_mm") == 1
    assert "数值超出有限浮点范围" in text
    assert "sensor.aperture_mm" in text
    assert text.count("sensor.spectral_range_nm.min") == 1
    assert "min 必须小于 max" not in text


def test_sample_entries_reject_huge_intervals_and_nonfinite_scalars():
    huge = [1, 10**400]
    for sampler in (sample_uniform, sample_log_uniform):
        with pytest.raises(ValueError) as caught:
            sampler(_generator(0), huge)
        assert repr(huge) in str(caught.value)
        for value in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="finite"):
                sampler(_generator(0), value)
    assert sample_uniform(_generator(0), 4.0) == 4.0
    assert sample_log_uniform(_generator(0), 4.0) == 4.0


def test_seed_finite_float_limit_is_stricter_than_root_seed():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = load_config(overrides={"generator": {"dataset": {"seed": 2**1023}}})
    assert loaded.generator["dataset"]["seed"] == 2**1023
    with pytest.raises(ValueError, match="整数超出有限浮点范围") as caught:
        load_config(overrides={"generator": {"dataset": {"seed": 10**400}}})
    assert "dataset.seed" in str(caught.value)
    assert root_seed(10**400).entropy == 10**400


def test_default_camera_derived_values_stay_on_the_same_formulas():
    width = height = 4096
    pixel_size_mm = 0.009
    focal_length_mm = 300.0
    with pytest.warns(UserWarning, match="超过 1%"):
        loaded = load_config()
    derived = loaded.derived
    assert derived.fov_width_deg == math.degrees(
        2.0 * math.atan((width * pixel_size_mm / 2.0) / focal_length_mm)
    )
    assert derived.fov_height_deg == math.degrees(
        2.0 * math.atan((height * pixel_size_mm / 2.0) / focal_length_mm)
    )
    assert derived.pixel_scale_arcsec == (pixel_size_mm / focal_length_mm) * (180.0 / math.pi) * 3600.0
    half_diagonal_mm = pixel_size_mm * math.hypot(width, height) / 2.0
    assert derived.half_diagonal_deg == math.degrees(math.atan(half_diagonal_mm / focal_length_mm))
    assert derived.f_over_p == focal_length_mm / pixel_size_mm
    assert derived.cx == (width - 1) / 2.0
    assert derived.cy == (height - 1) / 2.0
    assert round(derived.fov_width_deg, 3) == 7.032
    assert round(derived.pixel_scale_arcsec, 3) == 6.188
    assert round(derived.half_diagonal_deg, 2) == 4.97


def test_camera_derived_values_reject_nonfinite_or_masked_results():
    with pytest.raises(ValueError) as caught:
        load_config(overrides={"camera": {"sensor": {"pixel_size_mm": 5e-324}}})
    text = str(caught.value)
    assert "sensor.pixel_size_mm" in text
    assert "sensor.focal_length_mm" in text
    details = [line for line in text.splitlines() if line.startswith("sensor.")]
    assert details == [
        "sensor.focal_length_mm、sensor.pixel_size_mm: f_over_p 必须有限且严格大于 0，得到 inf",
        "sensor.focal_length_mm、sensor.pixel_size_mm: pixel_scale_arcsec 必须有限且严格大于 0，得到 0.0",
    ]
    quantities = (
        "半宽视场的 atan 参数",
        "半高视场的 atan 参数",
        "half_diagonal_mm",
        "半对角视场的 atan 参数",
        "f_over_p",
        "fov_deg.width",
        "fov_deg.height",
        "pixel_scale_arcsec",
        "half_diagonal_deg",
        ": cx",
        ": cy",
    )
    assert all(text.count(label) <= 1 for label in quantities)

    with pytest.raises(ValueError) as caught:
        load_config(
            overrides={
                "camera": {
                    "sensor": {
                        "resolution": {"width": 2**1000, "height": 2**1000},
                        "pixel_size_mm": 1e8,
                    }
                }
            }
        )
    text = str(caught.value)
    assert "sensor.resolution.width" in text
    assert "sensor.pixel_size_mm" in text
    assert "atan 参数" in text

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = load_config(
            overrides={"camera": {"sensor": {"resolution": {"width": 1, "height": 1}}}}
        )
    assert loaded.derived.cx == 0.0
    assert loaded.derived.cy == 0.0
    assert loaded.derived.K[0, 2] == 0.0
    assert loaded.derived.K[1, 2] == 0.0
    assert loaded.derived.K[0, 0] > 0.0
    assert loaded.derived.K[2, 2] == 1.0
