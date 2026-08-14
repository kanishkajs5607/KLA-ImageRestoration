"""Image-restoration evaluation metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from skimage.metrics import structural_similarity


def _to_hwc_images(tensor: torch.Tensor) -> np.ndarray:
    """Convert a BCHW tensor to clipped HWC numpy images for non-differentiable metrics."""
    return torch.clamp(tensor.detach().float().cpu(), 0.0, 1.0).permute(0, 2, 3, 1).numpy()


def batch_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Mean PSNR for a BCHW batch in the [0, 1] output convention."""
    if prediction.shape != target.shape:
        raise ValueError(f"PSNR shape mismatch: {prediction.shape} vs {target.shape}")
    mse = torch.mean((torch.clamp(prediction, 0.0, 1.0) - torch.clamp(target, 0.0, 1.0)).pow(2), dim=(1, 2, 3))
    values = []
    for value in mse.detach().cpu().tolist():
        values.append(float("inf") if value == 0 else 10.0 * math.log10(1.0 / value))
    finite_values = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite_values)) if finite_values else float("inf")


def batch_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Mean multichannel SSIM for a BCHW batch."""
    if prediction.shape != target.shape:
        raise ValueError(f"SSIM shape mismatch: {prediction.shape} vs {target.shape}")
    predictions = _to_hwc_images(prediction)
    targets = _to_hwc_images(target)
    scores: list[float] = []
    for predicted_image, target_image in zip(predictions, targets, strict=True):
        minimum_dimension = min(predicted_image.shape[:2])
        if minimum_dimension < 3:
            raise ValueError("SSIM requires images at least 3 pixels in both dimensions.")
        window_size = min(7, minimum_dimension if minimum_dimension % 2 == 1 else minimum_dimension - 1)
        scores.append(
            float(
                structural_similarity(
                    target_image,
                    predicted_image,
                    channel_axis=-1,
                    data_range=1.0,
                    win_size=window_size,
                )
            )
        )
    return float(np.mean(scores))


class LPIPSMetric:
    """Optional LPIPS metric wrapper, initialized only on explicit request."""

    def __init__(self, device: torch.device, network: str = "alex") -> None:
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError("LPIPS evaluation was requested but the 'lpips' package is not installed.") from error
        self.model = lpips.LPIPS(net=network).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        prediction = torch.clamp(prediction, 0.0, 1.0) * 2.0 - 1.0
        target = torch.clamp(target, 0.0, 1.0) * 2.0 - 1.0
        return float(self.model(prediction.to(self.device), target.to(self.device)).mean().item())


def evaluate_batch(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Return the always-available validation metrics for an image batch."""
    return {"psnr": batch_psnr(prediction, target), "ssim": batch_ssim(prediction, target)}
