"""Checkpoint loading helpers shared by inference and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.residual_unet import build_model
from src.utils.runtime import optimize_model_for_inference


def load_restoration_model(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a saved model and its metadata onto the requested device."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError(f"Checkpoint at {path} is missing model weights or architecture metadata.")
    model = build_model(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = optimize_model_for_inference(model, device)
    return model, checkpoint
