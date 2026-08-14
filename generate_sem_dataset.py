"""Generate a small synthetic SEM-style paired restoration dataset for local testing.

This generator is intentionally for development only. It does not represent official KLA data.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from src.utils.image_io import dimensions_summary, list_images, write_rgb_image


def generate_sem_pattern(size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a varied semiconductor-like pattern with lines, vias, and local defects."""
    image = np.full((size, size), rng.uniform(0.10, 0.18), dtype=np.float32)
    spacing = int(rng.integers(34, 58))
    line_width = int(rng.integers(2, 6))
    line_value = float(rng.uniform(0.60, 0.88))
    horizontal_offset = int(rng.integers(0, spacing))
    vertical_offset = int(rng.integers(0, spacing))

    for coordinate in range(horizontal_offset, size, spacing):
        cv2.line(image, (0, coordinate), (size - 1, coordinate), line_value, line_width)
    for coordinate in range(vertical_offset, size, spacing):
        cv2.line(image, (coordinate, 0), (coordinate, size - 1), line_value, line_width)

    via_pitch = spacing * int(rng.integers(2, 4))
    radius = int(rng.integers(6, 15))
    for x in range(via_pitch // 2, size, via_pitch):
        for y in range(via_pitch // 2, size, via_pitch):
            center = (int(x + rng.integers(-3, 4)), int(y + rng.integers(-3, 4)))
            cv2.circle(image, center, radius, float(rng.uniform(0.48, 0.76)), thickness=-1)
            if rng.random() < 0.65:
                cv2.circle(image, center, max(2, radius // 2), float(rng.uniform(0.12, 0.32)), thickness=-1)

    # Add repeated rectangular pads and a few local process-like defects.
    for _ in range(int(rng.integers(12, 25))):
        width = int(rng.integers(spacing, spacing * 3))
        height = int(rng.integers(max(8, spacing // 3), spacing))
        x = int(rng.integers(0, max(1, size - width)))
        y = int(rng.integers(0, max(1, size - height)))
        cv2.rectangle(image, (x, y), (x + width, y + height), float(rng.uniform(0.30, 0.56)), thickness=-1)
        cv2.rectangle(image, (x, y), (x + width, y + height), float(rng.uniform(0.65, 0.90)), thickness=1)

    for _ in range(int(rng.integers(8, 18))):
        x = int(rng.integers(0, size))
        y = int(rng.integers(0, size))
        radius = int(rng.integers(2, 8))
        cv2.circle(image, (x, y), radius, float(rng.uniform(0.02, 0.30)), thickness=-1)

    low_frequency = cv2.GaussianBlur(
        rng.normal(0.0, 1.0, (size, size)).astype(np.float32), (0, 0), sigmaX=size / 28
    )
    low_frequency /= np.max(np.abs(low_frequency)) + 1e-6
    image = np.clip(image + 0.025 * low_frequency, 0.0, 1.0)
    return np.repeat(image[:, :, None], 3, axis=2)


def apply_sem_degradation(clean_image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply downsampling, blur, additive Gaussian noise, and multiplicative speckle noise."""
    height, width = clean_image.shape[:2]
    low_resolution = cv2.resize(clean_image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    blur_sigma = float(rng.uniform(0.5, 1.8))
    blurred = cv2.GaussianBlur(low_resolution, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    speckle = 1.0 + rng.normal(0.0, rng.uniform(0.04, 0.16), blurred.shape).astype(np.float32)
    gaussian_noise = rng.normal(0.0, rng.uniform(0.02, 0.08), blurred.shape).astype(np.float32)
    degraded = blurred * speckle + gaussian_noise
    restored_size = cv2.resize(degraded, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(restored_size, 0.0, 1.0).astype(np.float32)


def generate_split(output_root: Path, split: str, count: int, size: int, seed: int, overwrite: bool) -> None:
    """Create a single train/validation split with matching GT and NoisyLR filenames."""
    gt_dir = output_root / split / "GT"
    noisy_dir = output_root / split / "NoisyLR"
    if overwrite:
        for directory in (gt_dir, noisy_dir):
            if directory.exists():
                shutil.rmtree(directory)
    gt_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    existing = list(gt_dir.glob("*.png")) + list(noisy_dir.glob("*.png"))
    if existing:
        raise FileExistsError(
            f"{split} already contains generated images. Use --overwrite to regenerate {output_root / split}."
        )

    split_offset = 0 if split == "train" else 10_000
    rng = np.random.default_rng(seed + split_offset)
    for index in range(count):
        clean = generate_sem_pattern(size, rng)
        degraded = apply_sem_degradation(clean, rng)
        filename = f"sem_sample_{index:03d}.png"
        write_rgb_image(gt_dir / filename, clean)
        write_rgb_image(noisy_dir / filename, degraded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired synthetic SEM restoration data.")
    parser.add_argument("--output-root", type=Path, default=Path("data"), help="Dataset root directory.")
    parser.add_argument("--train-count", type=int, default=40, help="Number of training pairs.")
    parser.add_argument("--val-count", type=int, default=10, help="Number of validation pairs.")
    parser.add_argument("--size", type=int, default=1024, help="Square output size for every generated image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated split folders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.size % 2 != 0:
        raise ValueError("--size must be a positive even integer because the degradation downsamples by two.")
    if args.train_count <= 0 or args.val_count <= 0:
        raise ValueError("--train-count and --val-count must both be positive.")

    generate_split(args.output_root, "train", args.train_count, args.size, args.seed, args.overwrite)
    generate_split(args.output_root, "val", args.val_count, args.size, args.seed, args.overwrite)

    for split, expected_count in (("train", args.train_count), ("val", args.val_count)):
        gt_paths = list_images(args.output_root / split / "GT")
        noisy_paths = list_images(args.output_root / split / "NoisyLR")
        if len(gt_paths) != expected_count or len(noisy_paths) != expected_count:
            raise RuntimeError(f"Generated {split} counts do not match the requested pair count.")
        if [path.name for path in gt_paths] != [path.name for path in noisy_paths]:
            raise RuntimeError(f"Generated {split} filenames are not paired.")
        dimensions = dimensions_summary(gt_paths + noisy_paths)
        expected_dimensions = {(args.size, args.size): expected_count * 2}
        if dimensions != expected_dimensions:
            raise RuntimeError(f"Generated {split} dimensions are incorrect: {dimensions}")
        print(f"{split}: {expected_count} matched pairs, all {args.size}x{args.size}")

    print(f"Synthetic SEM dataset created at: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
