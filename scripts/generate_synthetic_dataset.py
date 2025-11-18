"""(Removed) Synthetic data generator omitted from simplified demo.

This script was removed to keep the first commit minimal. Use the data collector
to create real mouth-crop samples if needed.
"""

__all__ = []

import argparse
from pathlib import Path
import random
import math
from PIL import Image, ImageDraw


def draw_neutral(draw, w, h):
    # small horizontal line
    draw.line([(w*0.3, h*0.55), (w*0.7, h*0.55)], fill='black', width=4)


def draw_frown(draw, w, h):
    # downturned arc
    bbox = [w*0.25, h*0.4, w*0.75, h*0.7]
    draw.arc(bbox, start=200, end=340, fill='black', width=6)


def draw_smile(draw, w, h):
    # upturned arc
    bbox = [w*0.25, h*0.45, w*0.75, h*0.75]
    draw.arc(bbox, start=20, end=160, fill='black', width=6)


def draw_tongue(draw, w, h):
    # large open mouth with central tongue circle
    bbox = [w*0.3, h*0.4, w*0.7, h*0.8]
    draw.ellipse(bbox, outline='black', width=6)
    # tongue
    tbox = [w*0.45, h*0.62, w*0.55, h*0.72]
    draw.ellipse(tbox, fill='red')


def draw_open(draw, w, h):
    # big open circular mouth
    bbox = [w*0.35, h*0.45, w*0.65, h*0.78]
    draw.ellipse(bbox, outline='black', width=6)


PRIMITIVES = {
    'neutral': draw_neutral,
    'frown': draw_frown,
    'smile': draw_smile,
    'tongue': draw_tongue,
    'open': draw_open,
}


def make_image(kind, size=(224,224)):
    w, h = size
    im = Image.new('RGB', size, color=(255, 220, 200))
    draw = ImageDraw.Draw(im)
    # add small noise background elements
    for _ in range(10):
        x = random.randint(0, w)
        y = random.randint(0, h)
        draw.point((x, y), fill=(200, 180, 180))

    PRIMITIVES[kind](draw, w, h)
    return im


def generate(output_dir: str, n_per_class: int = 50):
    out = Path(output_dir)
    train_dir = out / 'train'
    val_dir = out / 'val'
    classes = list(PRIMITIVES.keys())
    for d in [train_dir, val_dir]:
        for c in classes:
            (d / c).mkdir(parents=True, exist_ok=True)

    # generate
    for c in classes:
        for i in range(n_per_class):
            im = make_image(c)
            im.save(train_dir / c / f'{c}_{i:03d}.png')
    # small val set
    for c in classes:
        for i in range(max(5, n_per_class//10)):
            im = make_image(c)
            im.save(val_dir / c / f'{c}_val_{i:03d}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/synthetic_mouths')
    parser.add_argument('--per-class', type=int, default=80)
    args = parser.parse_args()
    generate(args.out, n_per_class=args.per_class)
    print('Generated synthetic data at', args.out)


if __name__ == '__main__':
    main()
