"""Evaluate a trained restoration checkpoint against paired validation images."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from inference import _prepare_tensor
from src.metrics.image_metrics import LPIPSMetric, evaluate_batch
from src.utils.checkpoints import load_restoration_model
from src.utils.image_io import list_images, read_rgb_image
from src.utils.runtime import RunningAverage, ensure_directory, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate restoration metrics for a paired image directory.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory of degraded images.")
    parser.add_argument("--target_dir", type=Path, required=True, help="Directory of clean GT images.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pth"), help="Trained checkpoint.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Execution device.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for same-sized images.")
    parser.add_argument("--upscale-factor", type=float, default=None, help="Optional input upscale override.")
    parser.add_argument("--lpips", action="store_true", help="Calculate LPIPS as well as PSNR and SSIM.")
    parser.add_argument("--output-json", type=Path, default=Path("results/validation_metrics.json"), help="Metric report path.")
    return parser.parse_args()


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

    input_paths = {path.name: path for path in list_images(args.input_dir)}
    target_paths = {path.name: path for path in list_images(args.target_dir)}
    if not input_paths:
        raise ValueError(f"No supported input images found in {args.input_dir}")
    if set(input_paths) != set(target_paths):
        missing_targets = sorted(set(input_paths) - set(target_paths))
        missing_inputs = sorted(set(target_paths) - set(input_paths))
        raise ValueError(
            f"Input/target filenames do not match. Missing targets: {missing_targets[:5]}; "
            f"missing inputs: {missing_inputs[:5]}"
        )
    metric_lpips = LPIPSMetric(device) if args.lpips else None
    psnr_average = RunningAverage()
    ssim_average = RunningAverage()
    lpips_average = RunningAverage()
    start = time.perf_counter()

    pending_inputs: list[torch.Tensor] = []
    pending_targets: list[torch.Tensor] = []
    pending_shape: tuple[int, ...] | None = None

    def flush() -> None:
        nonlocal pending_shape
        if not pending_inputs:
            return
        inputs = torch.stack(pending_inputs).to(device, non_blocking=True)
        targets = torch.stack(pending_targets).to(device, non_blocking=True)
        with torch.inference_mode():
            predictions = torch.clamp(model(inputs), 0.0, 1.0)
        metrics = evaluate_batch(predictions, targets)
        count = inputs.shape[0]
        psnr_average.update(metrics["psnr"], count)
        ssim_average.update(metrics["ssim"], count)
        if metric_lpips is not None:
            lpips_average.update(metric_lpips(predictions, targets), count)
        pending_inputs.clear()
        pending_targets.clear()
        pending_shape = None

    for filename in sorted(input_paths):
        input_image = read_rgb_image(input_paths[filename])
        target_image = read_rgb_image(target_paths[filename])
        input_tensor, target_size = _prepare_tensor(input_image, upscale_factor, pad_multiple)
        if target_image.shape[:2] != target_size:
            expected = (target_size[1], target_size[0])
            actual = (target_image.shape[1], target_image.shape[0])
            raise ValueError(
                f"Target size mismatch for {filename}: restored input is {expected}, target is {actual}. "
                "Set --upscale-factor to the correct ground-truth scale."
            )
        target_tensor = torch.from_numpy(np.ascontiguousarray(target_image.transpose(2, 0, 1))).float()
        pad_height = input_tensor.shape[-2] - target_tensor.shape[-2]
        pad_width = input_tensor.shape[-1] - target_tensor.shape[-1]
        if pad_height or pad_width:
            target_tensor = torch.nn.functional.pad(target_tensor, (0, pad_width, 0, pad_height), mode="replicate")
        shape = tuple(input_tensor.shape)
        if pending_inputs and (shape != pending_shape or len(pending_inputs) >= args.batch_size):
            flush()
        pending_inputs.append(input_tensor)
        pending_targets.append(target_tensor)
        pending_shape = shape
    flush()

    report: dict[str, float | int | str | None] = {
        "image_count": len(input_paths),
        "psnr_db": psnr_average.value,
        "ssim": ssim_average.value,
        "lpips": lpips_average.value if metric_lpips is not None else None,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - start,
    }
    ensure_directory(args.output_json.parent)
    with args.output_json.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved metric report to {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
