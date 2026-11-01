"""Seeded sky caps and sequence-count splits; all angles stored in degrees."""
import math
import warnings

import numpy as np

from starsim.motion import galactic_latitude_rad
from starsim.rng import SequenceRng, root_seed, sequence_seed

SPLIT_VERSION = 2


def separation(a, b):
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def vector(ra, dec):
    ra, dec = np.radians([ra, dec])
    return np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])


def assign(cfg, seed, count):
    rng = SequenceRng(sequence_seed(root_seed(seed), 0)).get('dataset/assign')
    names = ('train', 'val', 'test')
    order = rng.permutation(3)
    fractions = np.array([cfg['fractions'][name] for name in names], dtype=np.float64)
    counts = np.zeros(3, dtype=np.int64)
    assignments = []
    for i in range(count):
        deficits = fractions*(i+1)-counts
        best = int(order[0])
        for candidate in order[1:]:
            if deficits[candidate] > deficits[best]:
                best = int(candidate)
        counts[best] += 1
        assignments.append(names[best])
    return assignments


def make_caps(cfg, seed):
    rng = SequenceRng(sequence_seed(root_seed(seed), 0)).get('dataset/caps')
    caps = []
    if cfg['mode'] == 'sky':
        radius = cfg['cap_radius_deg']
        for name in ('val', 'test'):
            for i in range(cfg['caps_per_split']):
                for _ in range(100000):
                    ra = float(rng.uniform(0, 360))
                    dec = float(np.degrees(np.arcsin(rng.uniform(-1, 1))))
                    glat = abs(math.degrees(galactic_latitude_rad(math.radians(ra), math.radians(dec))))
                    if cfg['stratify_by_glat'] and not ((0, 20, 50)[i % 3] <= glat < (20, 50, 91)[i % 3]):
                        continue
                    v = vector(ra, dec)
                    if any(separation(v, vector(c['ra_deg'], c['dec_deg'])) <= 2*radius for c in caps):
                        continue
                    caps.append(dict(split=name, ra_deg=ra, dec_deg=dec, radius_deg=radius))
                    break
                else:
                    raise ValueError('cannot place non-overlapping sky caps with these parameters')
    return caps


def make_splits(cfg, seed, count):
    names = ('train', 'val', 'test')
    assignments = assign(cfg, seed, count)
    caps = make_caps(cfg, seed)
    # Deterministic equal-area quadrature, no extra random stream.
    z = 1-2*(np.arange(32768)+.5)/32768
    phi = np.arange(len(z))*math.pi*(3-math.sqrt(5))
    grid = np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z))
    masks = {name: np.zeros(len(z), bool) for name in ('val', 'test')}
    excluded = np.zeros(len(z), bool)
    for c in caps:
        dot = grid @ vector(c['ra_deg'], c['dec_deg'])
        masks[c['split']] |= dot >= math.cos(math.radians(c['radius_deg']))
        excluded |= dot >= math.cos(math.radians(min(180, c['radius_deg']+cfg['guard_deg'])))
    area = {name: float(mask.mean()) for name, mask in masks.items()}
    area['train'] = float((~excluded).mean()) if caps else 1.
    guard_loss = float(excluded.mean()-sum(area[n] for n in ('val', 'test')))
    if cfg['mode'] == 'random':
        area = dict.fromkeys(names, 1.)
    return dict(mode=cfg['mode'], assignments=assignments, caps=caps,
                guard_deg=cfg['guard_deg'], area_fraction=area,
                guard_loss_fraction=guard_loss,
                area_method='32768-point equal-area Fibonacci quadrature')


def accepts(regions, split, boresight):
    if regions['mode'] == 'random':
        return True
    distances = [(c, separation(boresight, vector(c['ra_deg'], c['dec_deg']))) for c in regions['caps']]
    if split == 'train':
        return all(d > c['radius_deg']+regions['guard_deg'] for c, d in distances)
    return any(c['split'] == split and d <= c['radius_deg'] for c, d in distances)


def audit(regions, records, minimum):
    if regions['mode'] == 'sky':
        for a in records:
            if a['split'] == 'train':
                for b in records:
                    if b['split'] != 'train' and separation(a['boresight'], b['boresight']) <= 11.6:
                        raise AssertionError('train/validation/test sky footprint leakage')
    distributions = {}
    for name in ('train', 'val', 'test'):
        group = [r for r in records if r['split'] == name]
        counts = [n for r in group for n in r['star_counts']]
        distributions[name] = dict(sequences=len(group), frames=len(counts),
            median=float(np.median(counts)) if counts else None,
            minimum=min(counts) if counts else None, maximum=max(counts) if counts else None)
    if all(d['sequences'] >= minimum for d in distributions.values()):
        medians = [d['median'] for d in distributions.values()]
        if min(medians) <= 0 or max(medians)/min(medians) >= 1.3:
            raise AssertionError('split median star counts differ by >= 30%')
    else:
        warnings.warn('too few sequences per split for the 30% star-count assertion', UserWarning)
    return distributions
