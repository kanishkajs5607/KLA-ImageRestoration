"""Paired NoisyLR-to-GT dataset used for restoration training and validation."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.image_io import list_images, read_rgb_image


class PairedRestorationDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Load matched images lazily and optionally return aligned training patches."""

    def __init__(
        self,
        input_dir: str | Path,
        target_dir: str | Path,
        patch_size: int | None = None,
        training: bool = False,
        augment: bool = False,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.patch_size = patch_size
        self.training = training
        self.augment = augment and training

        input_paths = {path.name: path for path in list_images(self.input_dir)}
        target_paths = {path.name: path for path in list_images(self.target_dir)}
        missing_targets = sorted(set(input_paths) - set(target_paths))
        missing_inputs = sorted(set(target_paths) - set(input_paths))
        if missing_targets or missing_inputs:
            details = []
            if missing_targets:
                details.append(f"missing GT for: {', '.join(missing_targets[:5])}")
            if missing_inputs:
                details.append(f"missing NoisyLR for: {', '.join(missing_inputs[:5])}")
            raise ValueError("Input/target filename mismatch: " + "; ".join(details))
        if not input_paths:
            raise ValueError(f"No supported images found in {self.input_dir}")

        self.names = sorted(input_paths)
        self.input_paths = input_paths
        self.target_paths = target_paths

    def __len__(self) -> int:
        return len(self.names)

    @staticmethod
    def _pad_to_patch(image: np.ndarray, patch_size: int) -> np.ndarray:
        height, width = image.shape[:2]
        bottom = max(patch_size - height, 0)
        right = max(patch_size - width, 0)
        if bottom == 0 and right == 0:
            return image
        border_mode = cv2.BORDER_REFLECT_101 if min(height, width) > 1 else cv2.BORDER_REPLICATE
        return cv2.copyMakeBorder(image, 0, bottom, 0, right, border_mode)

    def _crop_pair(self, noisy: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.patch_size is None:
            return noisy, target
        noisy = self._pad_to_patch(noisy, self.patch_size)
        target = self._pad_to_patch(target, self.patch_size)
        height, width = noisy.shape[:2]
        if self.training:
            top = random.randint(0, height - self.patch_size)
            left = random.randint(0, width - self.patch_size)
        else:
            top = (height - self.patch_size) // 2
            left = (width - self.patch_size) // 2
        crop = np.s_[top : top + self.patch_size, left : left + self.patch_size]
        return noisy[crop], target[crop]

    def _augment_pair(self, noisy: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return noisy, target
        if random.random() < 0.5:
            noisy, target = np.fliplr(noisy), np.fliplr(target)
        if random.random() < 0.5:
            noisy, target = np.flipud(noisy), np.flipud(target)
        rotations = random.randint(0, 3)
        if rotations:
            noisy, target = np.rot90(noisy, rotations), np.rot90(target, rotations)
        return np.ascontiguousarray(noisy), np.ascontiguousarray(target)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        name = self.names[index]
        noisy = read_rgb_image(self.input_paths[name])
        target = read_rgb_image(self.target_paths[name])

        # A model predicts at GT resolution. Resize only the degraded input when an official pair
        # uses a lower-resolution NoisyLR image than its corresponding GT image.
        if noisy.shape[:2] != target.shape[:2]:
            target_height, target_width = target.shape[:2]
            noisy = cv2.resize(noisy, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        noisy, target = self._crop_pair(noisy, target)
        noisy, target = self._augment_pair(noisy, target)
        noisy_tensor = torch.from_numpy(np.ascontiguousarray(noisy.transpose(2, 0, 1))).float()
        target_tensor = torch.from_numpy(np.ascontiguousarray(target.transpose(2, 0, 1))).float()
        return {"input": noisy_tensor, "target": target_tensor, "name": name}
