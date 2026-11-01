"""Offline native overlays and registered animations from saved sequences."""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

from starsim.io import load_objects, load_sequence


COLORS = ('#48dcff', '#66ff66', '#ff9944')


def map_points(H, points):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ np.asarray(H).T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def registered_image(image, H, size, *, max_size=768):
    """Inverse-map native pixel centers to a bounded preview of the canvas.

    size is (width, height); H maps native centers to canvas centers.
    Preview centers follow the same half-pixel convention as dense labels.
    """
    width, height = size
    scale = min(1., max_size / max(width, height))
    out_width, out_height = max(1, round(width*scale)), max(1, round(height*scale))
    sx, sy = out_width/width, out_height/height
    yy, xx = np.indices((out_height, out_width), dtype=float)
    points = np.column_stack(((xx.ravel()+.5)/sx-.5, (yy.ravel()+.5)/sy-.5))
    native = map_points(np.linalg.inv(H), points)
    values = map_coordinates(np.asarray(image, dtype=float), [native[:, 1], native[:, 0]],
                             order=1, mode='constant', cval=np.nan, prefilter=False)
    return values.reshape(out_height, out_width), (sx, sy)


def stretch(image, limits):
    lo, hi = limits
    normalized = np.clip((image-lo)/max(hi-lo, 1.), 0, 1)
    return np.nan_to_num(np.arcsinh(8*normalized)/np.arcsinh(8)*255).astype(np.uint8)


def overlay(image, frame, H, scale, limits, *, hot_pixels=(), title=''):
    rgb = Image.fromarray(stretch(image, limits)).convert('RGB')
    draw = ImageDraw.Draw(rgb)

    def pixel(points):
        xy = map_points(H, points)
        return (xy+.5)*np.asarray(scale)-.5

    for candidate in frame.candidates:
        x, y = pixel([[candidate['x'], candidate['y']]])[0]
        color = COLORS[candidate['class_id']]
        if candidate['status_default'] == 'labeled':
            draw.ellipse((x-4, y-4, x+4, y+4), outline=color, width=1)
        else:
            draw.line((x-3, y-3, x+3, y+3), fill=color)
            draw.line((x-3, y+3, x+3, y-3), fill=color)
        if candidate['class_id'] == 2:
            a, b = pixel([[candidate['x1'], candidate['y1']], [candidate['x2'], candidate['y2']]])
            draw.line((*a, *b), fill=color)
    for x, y in pixel(hot_pixels):
        draw.rectangle((x-1, y-1, x+1, y+1), outline='#ff55ff')
    draw.rectangle((0, 0, rgb.width, 30), fill='black')
    draw.text((5, 3), title, fill='white')
    draw.text((5, 16), 'cyan stars | green points | orange streaks | magenta hot; x = ignore', fill='white')
    return rgb


def visualize(sequence_dir, output_dir, *, mode='union', max_size=768, duration_ms=250):
    source, output = Path(sequence_dir), Path(output_dir)
    if mode not in ('native', 'reference', 'union'):
        raise ValueError('mode must be native, reference or union')
    if max_size < 1 or duration_ms < 1:
        raise ValueError('max_size and duration_ms must be positive')
    meta = load_sequence(source/'sequence.json')
    n = len(meta['motion']['H_to_ref'])
    sensor = meta['sensor']
    hot = np.column_stack((sensor['hot_col'], sensor['hot_row']))
    animation = []
    limits = None
    output.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        frame = load_objects(source/f'objects_{k:03d}.npz')
        image = fits.getdata(source/f'frame_{k:03d}.fits.fz')
        if limits is None:
            limits = tuple(np.percentile(image, [1, 99.8]))
        if mode == 'union':
            H, size = np.asarray(meta['canvas']['H_to_canvas'][k]), meta['canvas']['size']
        else:
            H = np.eye(3) if mode == 'native' else frame.H_to_ref
            size = (frame.width, frame.height)
        preview, scale = registered_image(image, H, size, max_size=max_size)
        rendered = overlay(preview, frame, H, scale, limits, hot_pixels=hot,
                           title=f'{source.name} / {mode} / frame {k}')
        rendered.save(output/f'{mode}_{k:03d}.png')
        animation.append(rendered)
    path = output/f'{mode}.gif'
    animation[0].save(path, save_all=True, append_images=animation[1:],
                      duration=duration_ms, loop=0, disposal=2)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--mode', choices=['native', 'reference', 'union'], default='union')
    parser.add_argument('--max-size', type=int, default=768)
    args = parser.parse_args()
    print(visualize(args.sequence_dir, args.output_dir, mode=args.mode, max_size=args.max_size))


if __name__ == '__main__':
    main()
