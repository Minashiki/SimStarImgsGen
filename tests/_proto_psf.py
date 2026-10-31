"""Throwaway timing prototype. Not part of the suite."""

import math
import time

import numpy as np
from numba import njit


@njit(cache=True)
def render_one(sigma, radius, nsub):
    size = int(2 * radius + 3)
    img = np.zeros((size, size))
    cx = size // 2
    cy = size // 2
    inv = 1.0 / (2.0 * math.pi * sigma * sigma)
    area = nsub * nsub
    for row in range(size):
        for col in range(size):
            acc = 0.0
            for iy in range(nsub):
                dy = (row + (iy + 0.5) / nsub - 0.5) - cy
                dy2 = dy * dy
                for ix in range(nsub):
                    dx = (col + (ix + 0.5) / nsub - 0.5) - cx
                    acc += math.exp(-0.5 * (dx * dx + dy2) / (sigma * sigma))
            img[row, col] = inv * acc / area
    return img


@njit(cache=True)
def render_many(image, xs, ys, flux, sigma, radius, nsub):
    h, w = image.shape
    inv = 1.0 / (2.0 * math.pi * sigma * sigma)
    n = xs.shape[0]
    area = nsub * nsub
    sig2 = sigma * sigma
    for i in range(n):
        x0 = xs[i]
        y0 = ys[i]
        c0 = int(math.floor(x0 - radius))
        c1 = int(math.ceil(x0 + radius))
        r0 = int(math.floor(y0 - radius))
        r1 = int(math.ceil(y0 + radius))
        if c0 < 0:
            c0 = 0
        if r0 < 0:
            r0 = 0
        if c1 >= w:
            c1 = w - 1
        if r1 >= h:
            r1 = h - 1
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                acc = 0.0
                for iy in range(nsub):
                    dy = (row + (iy + 0.5) / nsub - 0.5) - y0
                    dy2 = dy * dy
                    for ix in range(nsub):
                        dx = (col + (ix + 0.5) / nsub - 0.5) - x0
                        acc += math.exp(-0.5 * (dx * dx + dy2) / sig2)
                image[row, col] += flux * inv * acc / area


def main():
    render_one(1.0, 8, 5)
    for sigma in (0.7, 1.0, 1.5):
        for nsig in (5, 6, 8):
            radius = int(math.ceil(nsig * sigma))
            img = render_one(sigma, radius, 5)
            print(f"sigma={sigma} nsig={nsig} radius={radius} sum={img.sum():.6f}")

    image = np.zeros((512, 512))
    rng = np.random.default_rng(0)
    xs = rng.uniform(20, 490, 100)
    ys = rng.uniform(20, 490, 100)
    render_many(image, xs, ys, 1000.0, 1.0, 8.0, 5)
    image = np.zeros((4096, 4096))
    xs = rng.uniform(20, 4070, 5000)
    ys = rng.uniform(20, 4070, 5000)
    t0 = time.perf_counter()
    render_many(image, xs, ys, 1000.0, 1.0, 8.0, 5)
    print("5000 r=8", time.perf_counter() - t0, "sum", float(image.sum()))


if __name__ == "__main__":
    main()
