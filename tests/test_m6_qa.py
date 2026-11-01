"""M6 offline pixel metrics and registration contracts."""
import numpy as np
import pytest

from starsim.qa_report import histogram, separability, threshold_scan
from starsim.visualize import map_points, registered_image
from tests.rng_scan import scan_source
from tests.test_m4_labels import disk_sequence


@pytest.mark.parametrize('text', [
    'from scipy.stats import norm\nnorm.sf(3)',
    'import scipy as sp\nsp.stats.norm.cdf(0)',
    'from scipy import stats as s\ns.norm.logsf(4)',
])
def test_stats_deterministic_allowed(text):
    assert scan_source(text, 'starsim/qa_report.py') == []


@pytest.mark.parametrize('text', [
    'from scipy.stats import bootstrap as b\nb(x)',
    'import scipy.stats as s\ns.permutation_test(x)',
    'from scipy import stats\ns = stats\ns.monte_carlo_test(x)',
    'from scipy.stats import qmc',
    'import scipy.stats.qmc as q',
    'import scipy.stats as s\ngetattr(s, "bootstrap")(x)',
    'from importlib import import_module\ns=import_module("scipy.stats")\ns.qmc.Sobol(2)',
    'from scipy.stats import norm\nnorm.rvs()',
])
def test_stats_random_blocked(text):
    assert scan_source(text, 'scripts/qa.py')


def test_threshold_ties_and_odd_core():
    counts = np.bincount([1, 2, 3, 4, 4, 5], minlength=65536)
    result = threshold_scan([3, 4, 4, 7, 8], counts)
    assert result['threshold_adu'] == 4
    assert result['recall'] == .8
    assert result['false_positive_pixels'] == 3
    # Exhaustively check that no larger threshold can retain half the core.
    assert all(np.mean(np.array([3, 4, 4, 7, 8]) >= t) < .5 for t in range(5, 10))
    with pytest.raises(ValueError, match='no background'):
        threshold_scan([3], np.zeros(65536))


def test_separability_requires_both_tiers_and_inclusive_budget():
    rows = [dict(tier_design='ultra', false_positive_pixels=11)]*9
    rows += [dict(tier_design='ultra', false_positive_pixels=10)]
    rows += [dict(tier_design='bright', false_positive_pixels=10)]*99
    rows += [dict(tier_design='bright', false_positive_pixels=11)]
    assert separability(rows, 10)['passed']
    assert not separability([], 10)['passed']


def test_registration_stars_fixed_targets_move_hot_pixels_reverse():
    H = np.array([[1., 0, -2], [0, 1, 0], [0, 0, 1]])
    # Camera motion displaces stars +2 native pixels; H restores them.
    assert np.array_equal(map_points(H, [[5, 4]]), [[3, 4]])
    # A moving target advances another 3 pixels in reference coordinates.
    assert np.array_equal(map_points(H, [[8, 4]]), [[6, 4]])
    # Fixed detector hot pixel moves -2 in the registered view.
    assert np.array_equal(map_points(H, [[7, 4]]), [[5, 4]])
    image = np.zeros((10, 12)); image[4, 5] = 123
    warped, scale = registered_image(image, H, (12, 10), max_size=12)
    assert warped[4, 3] == 123
    assert np.isnan(warped[4, -1])
    assert scale == (1, 1)


def test_preview_half_pixel_scaling_and_projective_inverse():
    yy, xx = np.indices((12, 16))
    image = xx+2*yy
    warped, scale = registered_image(image, np.eye(3), (16, 12), max_size=8)
    assert scale == (.5, .5)
    assert warped[1, 2] == 4.5+2*2.5
    H = np.array([[1, .01, 2], [.02, 1, -1], [.0001, -.0002, 1.]])
    points = [[3, 4], [8, 9]]
    assert np.allclose(map_points(np.linalg.inv(H), map_points(H, points)), points)


def test_empty_histogram_has_no_fake_median():
    result = histogram([])
    assert result['count'] == sum(result['counts']) == 0
    assert result['median'] is None


def test_offline_report_and_all_overlay_modes_preserve_inputs(disk_sequence, tmp_path):
    from PIL import Image
    from starsim.io import _sha256
    from starsim.qa_report import build_report, plot_report
    from starsim.visualize import visualize

    source = disk_sequence.directory
    before = {p.name: _sha256(p) for p in source.iterdir() if p.is_file()}
    report = build_report([source])
    assert len(report['frames']) == len(disk_sequence.objects)
    assert report['threshold_experiment']['max_false_positive_pixels'] == 10
    assert report['histograms']['hot_snr']['count'] == report['histograms']['hot_empirical_snr']['count']
    for f, original in zip(report['frames'], disk_sequence.objects):
        assert sum(f['class_counts']) == len(original.sources)
    for mode in ('native', 'reference', 'union'):
        path = visualize(source, tmp_path/mode, mode=mode, max_size=128)
        with Image.open(path) as animation:
            assert animation.n_frames == len(disk_sequence.objects)
            assert max(animation.size) <= 128
    plot_report(report, tmp_path/'plots')
    assert len(list((tmp_path/'plots').glob('*.png'))) == 4
    assert before == {p.name: _sha256(p) for p in source.iterdir() if p.is_file()}
