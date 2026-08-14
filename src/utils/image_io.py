"""Image input/output utilities with a consistent RGB float32 convention."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(directory: str | Path) -> list[Path]:
    """Return supported image files in deterministic filename order."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def _normalize_to_float(image: np.ndarray) -> np.ndarray:
    """Convert common OpenCV image dtypes into float32 values near the [0, 1] range."""
    if np.issubdtype(image.dtype, np.integer):
        maximum = float(np.iinfo(image.dtype).max)
        return image.astype(np.float32) / maximum
    return image.astype(np.float32)


def read_rgb_image(path: str | Path) -> np.ndarray:
    """Read an image as HWC RGB float32 without silently changing spatial dimensions."""
    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image shape {image.shape} for {image_path}")
    return _normalize_to_float(image)


def write_rgb_image(path: str | Path, rgb_image: np.ndarray) -> None:
    """Save an HWC RGB float image as an 8-bit image, clipping only at final output."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(rgb_image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HWC RGB image with three channels, got {image.shape}")
    image = np.clip(image, 0.0, 1.0)
    uint8_image = np.rint(image * 255.0).astype(np.uint8)
    bgr_image = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), bgr_image):
        raise IOError(f"Unable to save image: {output_path}")


def choose_output_path(output_dir: str | Path, source_path: Path, output_extension: str | None) -> Path:
    """Construct an output filename while keeping the source stem unchanged."""
    extension = output_extension or source_path.suffix.lower()
    if not extension.startswith("."):
        extension = f".{extension}"
    if extension.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported output extension '{extension}'. Supported values: {sorted(IMAGE_EXTENSIONS)}"
        )
    return Path(output_dir) / f"{source_path.stem}{extension}"


def dimensions_summary(paths: Iterable[Path]) -> dict[tuple[int, int], int]:
    """Return a dimension histogram for lightweight dataset validation."""
    summary: dict[tuple[int, int], int] = {}
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Unable to read image: {path}")
        height, width = image.shape[:2]
        summary[(width, height)] = summary.get((width, height), 0) + 1
    return summary
