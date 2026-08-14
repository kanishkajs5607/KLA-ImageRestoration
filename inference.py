"""Restore every supported image in an input directory using a trained checkpoint."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from src.utils.checkpoints import load_restoration_model
from src.utils.image_io import choose_output_path, list_images, read_rgb_image, write_rgb_image
from src.utils.runtime import ensure_directory, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore images from an input directory to an output directory.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing degraded images.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for restored images.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pth"), help="Trained checkpoint.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Execution device.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for same-sized inputs.")
    parser.add_argument(
        "--upscale-factor",
        type=float,
        default=None,
        help="Resize inputs before restoration; defaults to the checkpoint configuration (normally 1).",
    )
    parser.add_argument(
        "--output-extension",
        default=None,
        help="Optional output extension, e.g. png. The original extension is retained by default.",
    )
    return parser.parse_args()


def _pad_tensor(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad a CHW tensor on its bottom/right side so U-Net downsampling remains aligned."""
    if multiple <= 0:
        raise ValueError("pad_multiple must be positive.")
    height, width = tensor.shape[-2:]
    pad_height = (-height) % multiple
    pad_width = (-width) % multiple
    if pad_height == 0 and pad_width == 0:
        return tensor, (height, width)
    mode = "reflect" if height > 1 and width > 1 else "replicate"
    return functional.pad(tensor, (0, pad_width, 0, pad_height), mode=mode), (height, width)


def _prepare_tensor(image: np.ndarray, upscale_factor: float, pad_multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    if upscale_factor <= 0:
        raise ValueError("--upscale-factor must be positive.")
    if upscale_factor != 1.0:
        height, width = image.shape[:2]
        resized_width = max(1, round(width * upscale_factor))
        resized_height = max(1, round(height * upscale_factor))
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
    return _pad_tensor(tensor, pad_multiple)


def restore_batch(
    model: torch.nn.Module,
    tensors: list[torch.Tensor],
    original_sizes: list[tuple[int, int]],
    source_paths: list[Path],
    output_dir: Path,
    output_extension: str | None,
    device: torch.device,
) -> int:
    """Run a same-sized batch through the model and save each restored output."""
    batch = torch.stack(tensors).to(device, non_blocking=True)
    predictions = torch.clamp(model(batch), 0.0, 1.0).cpu()
    for prediction, original_size, source_path in zip(predictions, original_sizes, source_paths, strict=True):
        height, width = original_size
        rgb = prediction[:, :height, :width].permute(1, 2, 0).numpy()
        write_rgb_image(choose_output_path(output_dir, source_path, output_extension), rgb)
    return len(source_paths)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    device = select_device(args.device)
    model, checkpoint = load_restoration_model(args.checkpoint, device)
    inference_config = checkpoint.get("inference_config", {})
    upscale_factor = float(
        args.upscale_factor if args.upscale_factor is not None else inference_config.get("upscale_factor", 1)
    )
    pad_multiple = int(inference_config.get("pad_multiple", 4))
    input_paths = list_images(args.input_dir)
    if not input_paths:
        raise ValueError(f"No supported images found in {args.input_dir}")
    output_dir = ensure_directory(args.output_dir)

    started = time.perf_counter()
    saved_count = 0
    pending_tensors: list[torch.Tensor] = []
    pending_sizes: list[tuple[int, int]] = []
    pending_paths: list[Path] = []
    pending_shape: tuple[int, ...] | None = None

    def flush() -> None:
        nonlocal saved_count, pending_shape
        if pending_tensors:
            saved_count += restore_batch(
                model,
                pending_tensors,
                pending_sizes,
                pending_paths,
                output_dir,
                args.output_extension,
                device,
            )
            pending_tensors.clear()
            pending_sizes.clear()
            pending_paths.clear()
            pending_shape = None

    with torch.inference_mode():
        for source_path in input_paths:
            image = read_rgb_image(source_path)
            tensor, original_size = _prepare_tensor(image, upscale_factor, pad_multiple)
            tensor_shape = tuple(tensor.shape)
            if pending_tensors and (tensor_shape != pending_shape or len(pending_tensors) >= args.batch_size):
                flush()
            pending_tensors.append(tensor)
            pending_sizes.append(original_size)
            pending_paths.append(source_path)
            pending_shape = tensor_shape
        flush()

    elapsed = time.perf_counter() - started
    print(
        f"Restored {saved_count} image(s) to {output_dir.resolve()} in {elapsed:.3f}s "
        f"({elapsed / max(saved_count, 1):.3f}s/image), device={device}, batch_size={args.batch_size}."
    )


if __name__ == "__main__":
    main()
