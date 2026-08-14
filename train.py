"""Train the SEM image-restoration model on paired NoisyLR and GT images."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.datasets.paired_sem import PairedRestorationDataset
from src.losses.restoration import RestorationLoss
from src.metrics.image_metrics import evaluate_batch
from src.models.residual_unet import build_model
from src.utils.runtime import RunningAverage, ensure_directory, load_yaml_config, seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a paired SEM image-restoration model.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"), help="YAML training configuration.")
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override configured batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override configured learning rate.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="Override configured device.")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint to resume from.")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional number of training batches per epoch for a quick smoke test.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
        help="Optional number of validation batches per epoch for a quick smoke test.",
    )
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply command-line overrides without requiring a configuration-file edit."""
    training = config.setdefault("training", {})
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        training["learning_rate"] = args.learning_rate
    if args.device is not None:
        config["device"] = args.device
    return config


def make_loader(
    input_dir: str,
    target_dir: str,
    patch_size: int | None,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    training: bool,
    augment: bool,
    seed: int,
) -> DataLoader:
    dataset = PairedRestorationDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        patch_size=patch_size,
        training=training,
        augment=augment,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: RestorationLoss,
    device: torch.device,
    optimizer: AdamW | None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run one training or validation epoch and calculate real image metrics."""
    is_training = optimizer is not None
    model.train(is_training)
    loss_average = RunningAverage()
    psnr_average = RunningAverage()
    ssim_average = RunningAverage()

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            predictions = model(inputs)
            loss, _ = loss_function(predictions, targets)
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        metrics = evaluate_batch(predictions, targets)
        current_batch_size = inputs.shape[0]
        loss_average.update(loss.item(), current_batch_size)
        psnr_average.update(metrics["psnr"], current_batch_size)
        ssim_average.update(metrics["ssim"], current_batch_size)

    return {"loss": loss_average.value, "psnr": psnr_average.value, "ssim": ssim_average.value}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: AdamW,
    epoch: int,
    best_psnr: float,
    config: dict[str, Any],
) -> None:
    """Save all information needed for reproducible inference or training continuation."""
    checkpoint = {
        "epoch": epoch,
        "best_psnr": best_psnr,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": config["model"],
        "inference_config": config.get("inference", {}),
        "training_config": config["training"],
        "full_config": config,
    }
    torch.save(checkpoint, path)


def write_history_row(history_path: Path, row: dict[str, float | int]) -> None:
    """Append epoch results to a CSV file that can be opened in spreadsheet software."""
    fieldnames = list(row)
    needs_header = not history_path.exists()
    with history_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_yaml_config(args.config), args)
    experiment = config.get("experiment", {})
    seed_everything(int(experiment.get("seed", 42)), bool(experiment.get("deterministic", False)))
    device = select_device(str(config.get("device", "auto")))
    data_config = config["data"]
    training_config = config["training"]
    model_config = config["model"]

    if int(training_config["epochs"]) <= 0 or int(training_config["batch_size"]) <= 0:
        raise ValueError("training.epochs and training.batch_size must be positive.")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("--max-train-batches must be positive when supplied.")
    if args.max_validation_batches is not None and args.max_validation_batches <= 0:
        raise ValueError("--max-validation-batches must be positive when supplied.")

    pin_memory = bool(data_config.get("pin_memory", True)) and device.type == "cuda"
    train_loader = make_loader(
        data_config["train_input_dir"],
        data_config["train_target_dir"],
        data_config.get("patch_size"),
        int(training_config["batch_size"]),
        int(data_config.get("num_workers", 0)),
        pin_memory,
        training=True,
        augment=bool(data_config.get("augment", True)),
        seed=int(experiment.get("seed", 42)),
    )
    validation_loader = make_loader(
        data_config["val_input_dir"],
        data_config["val_target_dir"],
        data_config.get("validation_patch_size"),
        int(training_config["batch_size"]),
        int(data_config.get("num_workers", 0)),
        pin_memory,
        training=False,
        augment=False,
        seed=int(experiment.get("seed", 42)) + 1,
    )

    model = build_model(**model_config).to(device)
    loss_function = RestorationLoss(**config.get("loss", {})).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    checkpoint_dir = ensure_directory(training_config.get("checkpoint_dir", "checkpoints"))
    results_dir = ensure_directory(training_config.get("results_dir", "results"))
    history_path = results_dir / "training_history.csv"

    start_epoch = 1
    best_psnr = float("-inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint.get("best_psnr", best_psnr))
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")

    print(
        f"Device: {device} | Model: {model_config['name']} | "
        f"Train pairs: {len(train_loader.dataset)} | Validation pairs: {len(validation_loader.dataset)}"
    )
    for epoch in range(start_epoch, int(training_config["epochs"]) + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, loss_function, device, optimizer, max_batches=args.max_train_batches
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model,
                validation_loader,
                loss_function,
                device,
                optimizer=None,
                max_batches=args.max_validation_batches,
            )
        seconds = time.perf_counter() - epoch_start
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_psnr": train_metrics["psnr"],
            "train_ssim": train_metrics["ssim"],
            "val_loss": validation_metrics["loss"],
            "val_psnr": validation_metrics["psnr"],
            "val_ssim": validation_metrics["ssim"],
            "seconds": seconds,
        }
        write_history_row(history_path, row)
        print(
            f"Epoch {epoch:03d}/{training_config['epochs']} | "
            f"train loss {train_metrics['loss']:.5f}, PSNR {train_metrics['psnr']:.3f}, SSIM {train_metrics['ssim']:.4f} | "
            f"val loss {validation_metrics['loss']:.5f}, PSNR {validation_metrics['psnr']:.3f}, SSIM {validation_metrics['ssim']:.4f} | "
            f"{seconds:.1f}s"
        )

        if validation_metrics["psnr"] > best_psnr:
            best_psnr = validation_metrics["psnr"]
            save_checkpoint(checkpoint_dir / "best_model.pth", model, optimizer, epoch, best_psnr, config)
            print(f"Saved new best checkpoint with validation PSNR {best_psnr:.3f} dB.")
        save_checkpoint(checkpoint_dir / "last_checkpoint.pth", model, optimizer, epoch, best_psnr, config)
        if epoch % int(training_config.get("save_every", 5)) == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}.pth", model, optimizer, epoch, best_psnr, config)

    print(f"Training complete. Best validation PSNR: {best_psnr:.3f} dB")
    print(f"Best checkpoint: {checkpoint_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
