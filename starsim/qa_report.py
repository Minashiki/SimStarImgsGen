"""Offline M6 statistics and empirical pixel-threshold experiments."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from starsim.io import load_objects, load_sequence, write_json
from starsim.metadata import git_provenance
from starsim.psf import PsfModel
from starsim.render import render_segment_stamp, segment_exposure_radius, segment_sample_count
from starsim.runtime import runtime_environment


def histogram(values, bins=30):
    values = np.asarray(values, dtype=float)
    counts, edges = np.histogram(values, bins=bins)
    return dict(count=len(values), counts=counts.tolist(), edges=edges.tolist(),
                median=float(np.median(values)) if len(values) else None)


def threshold_scan(core_values, background_counts):
    """Highest integer ADU threshold retaining at least half the core pixels.

    Detection uses image >= threshold, including ties. A full uint16 background
    histogram makes the scan exact without sorting a whole production image.
    """
    core = np.asarray(core_values)
    if not core.size:
        raise ValueError('core must contain at least one pixel')
    if not np.sum(background_counts):
        raise ValueError('no background pixels remain for the threshold experiment')
    required = (core.size+1)//2
    threshold = int(np.partition(core, core.size-required)[core.size-required])
    return dict(threshold_adu=threshold, core_pixels=int(core.size),
                recall=float(np.mean(core >= threshold)),
                false_positive_pixels=int(np.sum(background_counts[threshold:])),
                background_pixels=int(np.sum(background_counts)))


def frame_experiment(image, frame, psf, noise_sigma):
    """Core = visible noiseless self-signal >= half its visible peak.

    Background excludes every rendered source stamp, and retains sensor defects
    and cosmic rays. Clipped/truncated and overlapping targets are included.
    """
    background = np.ones(image.shape, dtype=bool)
    for source in frame.sources:
        r0, r1, c0, c1 = source['box']
        if source['has_flux'] and r0 >= 0:
            background[r0:r1+1, c0:c1+1] = False
    counts = np.bincount(image[background].astype(np.int64), minlength=65536)
    rows = []
    for source in frame.sources:
        if source['class_id'] == 0 or not source['has_flux']:
            continue
        endpoints = [source[key] for key in ('x1', 'y1', 'x2', 'y2')]
        radius = segment_exposure_radius(source['flux_e'], *endpoints,
            segment_sample_count(*endpoints), psf, noise_sigma)
        stamp, r0, c0, r1, c1, *_ = render_segment_stamp(
            frame.height, frame.width, source['x1'], source['y1'],
            source['x2'], source['y2'], source['flux_e'], psf, shared_radius=int(radius))
        if [r0, r1, c0, c1] != source['box']:
            raise ValueError('reconstructed target stamp differs from saved source box')
        core = stamp >= stamp.max()*.5
        measurement = threshold_scan(image[r0:r1+1, c0:c1+1][core], counts)
        rows.append(dict(track_id=source['track_id'], class_id=source['class_id'],
                         tier_design=source['tier_design'], tier_measured=source['tier_measured'],
                         **measurement))
    return rows


def separability(rows, max_false_positives):
    result = {}
    for tier in ('ultra', 'faint', 'bright'):
        selected = [r for r in rows if r['tier_design'] == tier]
        classified = [r['false_positive_pixels'] <= max_false_positives for r in selected]
        rate = float(np.mean(classified)) if classified else None
        result[tier] = dict(count=len(selected), separable_fraction=rate,
                            inseparable_fraction=1-rate if rate is not None else None)
    ultra, bright = result['ultra'], result['bright']
    result['passed'] = (ultra['count'] > 0 and bright['count'] > 0
                        and ultra['inseparable_fraction'] >= .9
                        and bright['separable_fraction'] >= .99)
    return result


def build_report(sequence_dirs, *, max_false_positives=10):
    if max_false_positives is not None and max_false_positives < 0:
        raise ValueError('max_false_positives must be nonnegative')
    values = {k: [] for k in ('camera_speed_px_per_frame', 'target_speed_px_per_frame',
                              'star_magnitude', 'star_snr', 'target_snr', 'hot_snr',
                              'hot_empirical_snr')}
    frames, experiments, provenance = [], [], []
    design, measured, transition = Counter(), Counter(), Counter()
    for directory in sequence_dirs:
        source = Path(directory)
        meta = load_sequence(source/'sequence.json')
        provenance.append(dict(path=str(source.resolve()), git_commit=meta['git_commit'],
                               git_dirty=meta['git_dirty'], rng=meta['rng']))
        psf = PsfModel(**meta['psf'])
        values['camera_speed_px_per_frame'].extend(meta['motion']['speed_px_per_frame'])
        population = meta['population']
        # Adjacent reference-center displacement, not nominal design speed.
        dx = np.diff(np.asarray(population['x_ref']), axis=1)
        dy = np.diff(np.asarray(population['y_ref']), axis=1)
        values['target_speed_px_per_frame'].extend(np.hypot(dx, dy).ravel())
        for k in range(len(meta['motion']['H_to_ref'])):
            frame = load_objects(source/f'objects_{k:03d}.npz')
            fm = json.loads((source/f'meta_{k:03d}.json').read_text(encoding='utf-8'))
            image = fits.getdata(source/f'frame_{k:03d}.fits.fz')
            stats = fm['statistics']
            counts = [sum(s['class_id'] == cls for s in frame.sources) for cls in range(3)]
            frames.append(dict(sequence=str(source.resolve()), frame=k, class_counts=counts,
                star_to_target_ratio=counts[0]/sum(counts[1:]) if sum(counts[1:]) else None,
                ignore_fraction=stats['ignore_fraction'],
                outside_with_flux=sum(not s['in_image'] and s['has_flux'] for s in frame.sources),
                never_full=sum(s.get('never_full', False) for s in frame.sources)))
            for s in frame.sources:
                if s['class_id'] == 0:
                    values['star_magnitude'].append(s['vmag'])
                    values['star_snr'].append(s['snr'])
                else:
                    values['target_snr'].append(s['snr'])
                    design[s['tier_design']] += 1
                    measured[s['tier_measured']] += 1
                    transition[f"{s['tier_design']}->{s['tier_measured']}"] += 1
            # The saved RTS state contains every frame. Nominal hot SNR uses
            # that frame's exact level and reference background sigma. Also
            # report realized contrast separately: it includes noise, clipping
            # and any astronomical source overlapping the detector pixel.
            sensor, noise = meta['sensor'], meta['noise']
            for i, (row, col) in enumerate(zip(sensor['hot_row'], sensor['hot_col'])):
                level = sensor['hot_level_a'][i] if sensor['hot_state'][i][k] == 0 else sensor['hot_level_b'][i]
                values['hot_snr'].append(level/fm['background']['sigma_ref'])
                patch = image[max(0, row-3):row+4, max(0, col-3):col+4]
                contrast = (float(image[row, col])-float(np.median(patch)))*noise['gain_e_per_adu']
                values['hot_empirical_snr'].append(contrast/fm['background']['sigma_ref'])
            for row in frame_experiment(image, frame, psf, fm['background']['sigma_ref']):
                experiments.append(dict(sequence=str(source.resolve()), frame=k, **row))
    return dict(schema_version=1, environment=runtime_environment(), **git_provenance(),
        sources=provenance, histograms={key: histogram(v) for key, v in values.items()},
        tiers=dict(design=dict(design), measured=dict(measured), transitions=dict(transition)),
        snr_definitions=dict(target_snr='saved pre-noise self peak / local background sigma',
            hot_snr='saved frame-specific hot level / reference background sigma; pre-clipping',
            hot_empirical_snr='realized FITS pixel minus 7x7 median, in electrons / reference sigma'),
        frames=frames, threshold_experiment=dict(
            core='visible noiseless self-signal >= 0.5 * visible peak',
            background='all native pixels outside all rendered source boxes; defects retained',
            detection='raw uint16 ADU >= threshold; highest threshold with core recall >= 0.5',
            unit='target-frame observation; grouped by design tier',
            max_false_positive_pixels=max_false_positives, measurements=experiments,
            acceptance=separability(experiments, max_false_positives)
                       if max_false_positives is not None else None))


def plot_report(report, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (name, hist) in zip(axes.ravel(), list(report['histograms'].items())[:6]):
        ax.stairs(hist['counts'], hist['edges'])
        ax.set(title=name, ylabel='observations')
    fig.savefig(output/'histograms.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for key, label in [('hot_snr', 'hot: saved frame level / reference sigma'),
                       ('target_snr', 'target: saved self signal / local sigma')]:
        hist = report['histograms'][key]
        density = np.asarray(hist['counts']) / (max(hist['count'], 1)*np.diff(hist['edges']))
        ax.stairs(density, hist['edges'], label=label)
    ax.set(xlabel='SNR (definitions differ; see legend)', ylabel='probability density',
           xscale='symlog', yscale='log')
    ax.legend(fontsize=8)
    fig.savefig(output/'hot_target_snr.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for tier in ('ultra', 'faint', 'bright'):
        values = sorted(r['false_positive_pixels'] for r in report['threshold_experiment']['measurements']
                        if r['tier_design'] == tier)
        if values:
            ax.step(values, np.arange(1, len(values)+1)/len(values), label=tier)
    ax.set(xlabel='background false positive pixels at 50% core recall',
           ylabel='fraction of target-frame observations', xscale='symlog')
    ax.legend()
    fig.savefig(output/'threshold_separability.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    tiers = sorted(set(report['tiers']['design']) | set(report['tiers']['measured']))
    x = np.arange(len(tiers))
    for offset, field in [(-.2, 'design'), (.2, 'measured')]:
        axes[0].bar(x+offset, [report['tiers'][field].get(t, 0) for t in tiers],
                    width=.4, label=field)
    axes[0].set_xticks(x, tiers, rotation=30)
    axes[0].legend()
    axes[0].set_title('Design / measured tiers (source-frame counts)')
    for cls, name in enumerate(('stars', 'points', 'streaks')):
        axes[1].plot([f['class_counts'][cls] for f in report['frames']], label=name)
        axes[2].plot([f['ignore_fraction'][cls] for f in report['frames']], label=name)
    axes[1].set(title='Sources per frame', yscale='symlog', xlabel='frame observation')
    axes[2].set(title='Ignore grid fraction', xlabel='frame observation')
    axes[1].legend(); axes[2].legend()
    fig.savefig(output/'tiers_counts_ignore.png', dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='sequence directory or dataset directory')
    parser.add_argument('output', type=Path)
    parser.add_argument('--max-false-positives', type=int, default=10)
    parser.add_argument('--assert-thresholds', action='store_true',
                        help='exit with status 1 if ultra/bright acceptance fails')
    args = parser.parse_args()
    sequences = [args.input] if (args.input/'sequence.json').is_file() else sorted(args.input.glob('seq_*'))
    if not sequences:
        parser.error('no sequences found')
    report = build_report(sequences, max_false_positives=args.max_false_positives)
    plot_report(report, args.output)
    write_json(args.output/'qa_report.json', report)
    print(args.output/'qa_report.json')
    if args.assert_thresholds and not report['threshold_experiment']['acceptance']['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
