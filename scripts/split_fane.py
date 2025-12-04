"""Split a folder of class images into train/val ImageFolder layout.

Defaults target the FANE dataset already downloaded locally:
  python scripts/split_fane.py --source fane_data --out data/fane_split --val-frac 0.2

Expected source layout:
  source/
    happy/
    sad/
    angry/
    neutral/

Creates:
  out/train/<class>/...
  out/val/<class>/...
"""

import argparse
import random
import shutil
from pathlib import Path


def split_class_dir(src_class_dir: Path, dst_train: Path, dst_val: Path, val_frac: float, seed: int = 42) -> None:
    images = [p for p in src_class_dir.iterdir() if p.is_file()]
    if not images:
        return

    random.Random(seed).shuffle(images)
    split_idx = int(len(images) * (1.0 - val_frac))
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    (dst_train / src_class_dir.name).mkdir(parents=True, exist_ok=True)
    (dst_val / src_class_dir.name).mkdir(parents=True, exist_ok=True)

    for p in train_imgs:
        shutil.copy2(p, dst_train / src_class_dir.name / p.name)
    for p in val_imgs:
        shutil.copy2(p, dst_val / src_class_dir.name / p.name)


def main():
    parser = argparse.ArgumentParser(description="Split FANE (or similar) dataset into train/val ImageFolder layout.")
    parser.add_argument("--source", default="fane_data", help="Path to source class folders (e.g., fane_data)")
    parser.add_argument("--out", default="data/fane_split", help="Output root for train/ and val/")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Fraction of images to put in val split")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible splits")
    args = parser.parse_args()

    src_root = Path(args.source)
    if not src_root.exists():
        raise FileNotFoundError(f"Source folder not found: {src_root}")

    out_root = Path(args.out)
    train_dir = out_root / "train"
    val_dir = out_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in src_root.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class subfolders found in {src_root}")

    for class_dir in class_dirs:
        split_class_dir(class_dir, train_dir, val_dir, args.val_frac, seed=args.seed)

    print(f"Done. Train/val splits written to {out_root}")


if __name__ == "__main__":
    main()
