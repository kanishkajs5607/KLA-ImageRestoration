"""Shared runtime helpers for the KLA image-restoration pipeline."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed common RNGs for repeatable local experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def select_device(requested_device: str = "auto") -> torch.device:
    """Choose CUDA when available unless CPU was explicitly requested."""
    requested_device = requested_device.lower()
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and return it as a Path instance."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping, returning an empty mapping for an empty file."""
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {path} must contain a YAML mapping.")
    return config


class RunningAverage:
    """Numerically stable running average used in epoch logging."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def value(self) -> float:
        return self.total / max(self.count, 1)


def optimize_model_for_inference(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Prepare a model for inference without changing its learned parameters."""
    model.eval()
    if device.type == "cpu":
        # Channels-last is supported by the convolutional model and was benchmarked locally
        # before being enabled here. It reduces CPU convolution overhead without altering outputs.
        model = model.to(memory_format=torch.channels_last)
    return model


def prepare_inference_batch(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move an inference batch and match the model's CPU memory format."""
    batch = batch.to(device, non_blocking=True)
    if device.type == "cpu":
        batch = batch.contiguous(memory_format=torch.channels_last)
    return batch
