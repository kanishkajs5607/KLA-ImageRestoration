"""Loss functions for training the SEM image-restoration model."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class CharbonnierLoss(nn.Module):
    """A smooth L1-style reconstruction loss that is robust to outliers."""

    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch in Charbonnier loss: {prediction.shape} vs {target.shape}"
            )
        return torch.mean(torch.sqrt((prediction - target).pow(2) + self.epsilon**2))


class GradientLoss(nn.Module):
    """Penalize differences in first-order image gradients to preserve SEM edges."""

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch in gradient loss: {prediction.shape} vs {target.shape}"
            )
        prediction_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
        target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        prediction_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        return functional.l1_loss(prediction_x, target_x) + functional.l1_loss(prediction_y, target_y)


class LPIPSPerceptualLoss(nn.Module):
    """Optional LPIPS term. It is only constructed when explicitly enabled in configuration."""

    def __init__(self, network: str = "alex") -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError("LPIPS loss was requested but the 'lpips' package is not installed.") from error
        self.model = lpips.LPIPS(net=network)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch in LPIPS loss: {prediction.shape} vs {target.shape}"
            )
        prediction = torch.clamp(prediction, 0.0, 1.0) * 2.0 - 1.0
        target = torch.clamp(target, 0.0, 1.0) * 2.0 - 1.0
        return self.model(prediction, target).mean()


class RestorationLoss(nn.Module):
    """Weighted reconstruction, gradient, and optional perceptual loss."""

    def __init__(
        self,
        charbonnier_weight: float = 1.0,
        gradient_weight: float = 0.1,
        perceptual_weight: float = 0.0,
        lpips_network: str = "alex",
    ) -> None:
        super().__init__()
        if min(charbonnier_weight, gradient_weight, perceptual_weight) < 0:
            raise ValueError("Loss weights must be non-negative.")
        self.charbonnier_weight = charbonnier_weight
        self.gradient_weight = gradient_weight
        self.perceptual_weight = perceptual_weight
        self.charbonnier = CharbonnierLoss()
        self.gradient = GradientLoss()
        self.perceptual = (
            LPIPSPerceptualLoss(lpips_network) if perceptual_weight > 0.0 else None
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Model output and target must have identical dimensions: {prediction.shape} vs {target.shape}"
            )
        reconstruction = self.charbonnier(prediction, target)
        edge = self.gradient(prediction, target)
        total = self.charbonnier_weight * reconstruction + self.gradient_weight * edge
        components = {"charbonnier": reconstruction.detach().item(), "gradient": edge.detach().item()}
        if self.perceptual is not None:
            perceptual = self.perceptual(prediction, target)
            total = total + self.perceptual_weight * perceptual
            components["lpips_loss"] = perceptual.detach().item()
        components["total"] = total.detach().item()
        return total, components
