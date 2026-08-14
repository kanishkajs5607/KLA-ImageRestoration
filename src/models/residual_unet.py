"""Compact residual U-Net for same-resolution image restoration."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two-convolution residual block without normalization artifacts."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Inputs are padded to a multiple of four by inference. The guard also keeps direct model
        # usage safe for odd patch sizes.
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((x, skip), dim=1))


class ResidualUNet(nn.Module):
    """A compact U-Net whose output is input plus a learned correction image."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlock(base_channels),
        )
        self.down1 = DownsampleBlock(base_channels, base_channels * 2)
        self.down2 = DownsampleBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = nn.Sequential(ResidualBlock(base_channels * 4), ResidualBlock(base_channels * 4))
        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpsampleBlock(base_channels * 2, base_channels, base_channels)
        self.output = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = x
        skip1 = self.stem(x)
        skip2 = self.down1(skip1)
        features = self.bottleneck(self.down2(skip2))
        features = self.up2(features, skip2)
        features = self.up1(features, skip1)
        return original + self.output(features)


def build_model(
    name: str = "residual_unet", in_channels: int = 3, base_channels: int = 32
) -> nn.Module:
    """Build a supported restoration model from checkpoint/configuration metadata."""
    normalized_name = name.lower().replace("-", "_")
    if normalized_name != "residual_unet":
        raise ValueError(f"Unsupported model '{name}'. Available model: residual_unet")
    return ResidualUNet(in_channels=in_channels, base_channels=base_channels)
