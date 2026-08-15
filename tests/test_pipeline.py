from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from inference import _prepare_tensor, restore_batch
from src.models.residual_unet import build_model
from src.utils.checkpoints import load_restoration_model
from src.utils.image_io import read_rgb_image, write_rgb_image


def make_checkpoint(path: Path) -> None:
    model_config = {"name": "residual_unet", "in_channels": 3, "base_channels": 4}
    model = build_model(**model_config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "inference_config": {"upscale_factor": 1, "pad_multiple": 4},
        },
        path,
    )


def test_image_round_trip(tmp_path: Path) -> None:
    image = np.zeros((17, 19, 3), dtype=np.float32)
    image[:, :, 0] = 0.8
    image[4:12, 5:14, 1] = 0.4
    source = tmp_path / "input.png"
    write_rgb_image(source, image)
    loaded = read_rgb_image(source)
    assert loaded.shape == image.shape
    assert loaded.dtype == np.float32
    assert np.allclose(loaded[:, :, 0].mean(), 0.8, atol=1 / 255)


def test_preprocessing_pads_to_model_multiple() -> None:
    image = np.random.default_rng(1).random((7, 9, 3), dtype=np.float32)
    tensor, original_size = _prepare_tensor(image, upscale_factor=1, pad_multiple=4)
    assert tuple(tensor.shape) == (3, 8, 12)
    assert original_size == (7, 9)


def test_checkpoint_model_loading(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    make_checkpoint(checkpoint)
    model, metadata = load_restoration_model(checkpoint, torch.device("cpu"))
    output = model(torch.rand(1, 3, 16, 16))
    assert output.shape == (1, 3, 16, 16)
    assert metadata["model_config"]["base_channels"] == 4


def test_inference_writes_restored_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    make_checkpoint(checkpoint)
    model, _ = load_restoration_model(checkpoint, torch.device("cpu"))
    source = tmp_path / "sample.png"
    output_dir = tmp_path / "restored"
    write_rgb_image(source, np.random.default_rng(2).random((16, 16, 3), dtype=np.float32))
    input_tensor, original_size = _prepare_tensor(read_rgb_image(source), 1, 4)
    written = restore_batch(
        model,
        [input_tensor],
        [original_size],
        [source],
        output_dir,
        "png",
        torch.device("cpu"),
    )
    output = output_dir / "sample.png"
    assert written == 1
    assert output.is_file()
    assert read_rgb_image(output).shape == (16, 16, 3)
