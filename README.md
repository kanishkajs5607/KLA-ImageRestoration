# KLA SEM Image Restoration

This repository contains a **runnable, CPU-compatible image-restoration pipeline** for paired semiconductor SEM-style images. The pipeline learns to map a degraded, noisy low-resolution observation (`NoisyLR`) to a clean ground-truth image (`GT`). It includes synthetic data generation for local development, training, standalone directory inference, paired validation, and an interactive local web interface.

> **Important:** the generated SEM images in this repository are synthetic development data. They are not official KLA competition data and must not be represented as such.

## Objective

The project restores semiconductor-like images affected by downsampling, blur, additive Gaussian noise, and multiplicative speckle noise. It is designed so that an evaluator or student can run inference without editing source files:

```text
python inference.py --input_dir <input_directory> --output_dir <output_directory>
```

The program loads a checkpoint, processes every supported image in the input directory, writes restored images to the output directory, preserves filename stems, and creates the output directory if it does not already exist.

| Capability | Implementation |
|---|---|
| Input/target convention | Matched filenames in `NoisyLR/` and `GT/` folders |
| Local development data | Deterministic synthetic SEM-style generator |
| Model | Compact residual U-Net that predicts a correction image |
| Training size | Random aligned patches, while source images remain 1024×1024 |
| Validation | PSNR and SSIM; LPIPS is optional on explicit request |
| Runtime | CPU by default when CUDA is unavailable; CUDA used automatically when available |
| Interactive UI | Streamlit upload, preview, restore, comparison, download, and reset workflow |
| Reproducibility | YAML configuration, seeds, CSV logs, saved checkpoints, and checkpoint metadata |

## Repository layout

```text
KLA-ImageRestoration/
├── configs/
│   └── train.yaml                 # Default experiment configuration
├── src/
│   ├── datasets/paired_sem.py     # Lazy paired image loading and aligned patches
│   ├── losses/restoration.py      # Charbonnier, gradient, optional LPIPS loss
│   ├── metrics/image_metrics.py   # PSNR, SSIM, optional LPIPS metric
│   ├── models/residual_unet.py    # Residual U-Net architecture
│   └── utils/                     # Image I/O, checkpoints, runtime helpers
├── app.py                          # Interactive Streamlit application
├── generate_sem_dataset.py        # Synthetic 1024×1024 paired-data generator
├── train.py                       # Training and checkpointing command
├── inference.py                   # Standalone directory inference command
├── evaluate.py                    # Paired validation metrics command
├── tests/test_pipeline.py         # Automated pipeline tests
├── requirements.txt
└── README.md
```

The following local folders are created at runtime and intentionally ignored by Git:

```text
data/
checkpoints/
results/
```

## Installation and setup

The project targets **Python 3.12.9 on Windows** and also runs on other supported Python environments. It does not require CUDA; the code selects `cuda` only when `torch.cuda.is_available()` is true and otherwise uses CPU.

### Windows PowerShell

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux/macOS shell

```bash
source venv/bin/activate
pip install -r requirements.txt
```

If PyTorch fails to import on a CPU-only Windows machine with a `torch\lib\c10.dll` error, install the Microsoft Visual C++ Redistributable and install a CPU PyTorch wheel appropriate for your environment. Do not change a working environment unnecessarily.

## Generate local synthetic SEM data

Run the generator from the repository root:

```bash
python generate_sem_dataset.py --overwrite
```

It creates exactly **40 training pairs** and **10 validation pairs**. Every generated GT and NoisyLR image is **1024×1024**, and each pair has the same filename.

```text
data/
├── train/
│   ├── GT/        # 40 clean 1024×1024 PNG images
│   └── NoisyLR/   # 40 matched degraded 1024×1024 PNG images
└── val/
    ├── GT/        # 10 clean 1024×1024 PNG images
    └── NoisyLR/   # 10 matched degraded 1024×1024 PNG images
```

The synthetic generator creates line grids, repeated via-like circular structures, rectangular pads, and local defects. It then applies a development-only degradation sequence consisting of downsampling by two, Gaussian blur, multiplicative speckle noise, additive Gaussian noise, and bicubic resizing to the original 1024×1024 dimensions.

## Model architecture

The selected architecture is a **compact two-level residual U-Net**. It has an encoder for multi-scale context, decoder skip connections for fine edge recovery, and a residual output head. Rather than predicting the entire clean image from scratch, the model predicts a correction which is added to the degraded input. This is a practical choice for a limited local dataset because the input already contains meaningful structural information and the model can focus capacity on denoising and detail recovery.

The default model uses 32 base channels and no normalization layers, avoiding batch-statistics instability with small batches. The model is fully convolutional and is made shape-safe during inference by padding to a multiple of four and cropping the result back to the expected output size.

| Component | Default design decision | Rationale |
|---|---|---|
| Input | Three-channel RGB float image | Safe handling for standard common image formats |
| Encoder/decoder depth | Two downsampling stages | Balances context with CPU-friendly size |
| Output | Input + learned residual | Preserves observed structure and learns restoration corrections |
| Training patch | 256×256 aligned patch | Avoids loading full 1024×1024 images into training batches |
| Data augmentation | Flips and 90° rotations | Applies the same geometric transform to each input/target pair |
| Final output | Clipped only before file writing | Keeps training predictions unconstrained while saving valid images |

## Training

The default configuration is in `configs/train.yaml`. It controls data paths, model width, patch size, batch size, learning rate, loss weights, checkpoint paths, and inference padding/upscale defaults. Do not edit source code to change ordinary experiment settings.

Run the standard training command after generating the local data:

```bash
python train.py --config configs/train.yaml
```

For a short CPU smoke test, use one epoch and one image per batch:

```bash
python train.py --config configs/train.yaml --epochs 1 --batch-size 1 --device cpu --max-train-batches 1 --max-validation-batches 1
```

Training saves the following ignored local artifacts:

```text
checkpoints/best_model.pth
checkpoints/last_checkpoint.pth
checkpoints/epoch_XXX.pth        # at the configured save interval
results/training_history.csv
```

The best checkpoint is selected by validation PSNR. Checkpoints include model architecture metadata, optimizer state, the full training configuration, and inference configuration so that inference does not depend on manual code edits.

### Loss functions

The default loss is a weighted combination of a Charbonnier reconstruction loss and a gradient loss. The Charbonnier term rewards pixel fidelity while being less sensitive to isolated high-error pixels than pure MSE. The gradient term encourages preservation of line and edge structure. LPIPS is implemented as an optional perceptual term but is disabled by default (`perceptual_weight: 0.0`) so that local CPU smoke tests do not unexpectedly require perceptual-network weights.

| Loss | Default weight | Purpose |
|---|---:|---|
| Charbonnier | 1.0 | Robust pixel-level reconstruction |
| Gradient L1 | 0.1 | Preserve structural edge detail |
| LPIPS | 0.0 | Optional perceptual similarity term |

## Interactive application

The repository includes a local Streamlit interface for interactive testing. It lets a user upload a JPG, JPEG, or PNG image, preview the original, start restoration, see a loading indicator, compare the original and restored images side by side, download a restored PNG, and reset the upload control.

Start it from the repository root after training a checkpoint:

```bash
streamlit run app.py
```

The browser UI defaults to `checkpoints/best_model.pth`, automatically selects CUDA when available, and otherwise runs on CPU. The sidebar allows the checkpoint path, device selection, upscale factor, and maximum input dimension to be changed without editing source code. Very large uploads are resized down to the configured maximum dimension before inference to limit local memory usage. Invalid, corrupted, unsupported, or missing images and checkpoints produce user-facing error messages instead of an unhandled traceback.

The UI is a local application intended for development and demonstration. It does not upload images to an external service and does not require an API key.

## Standalone inference

After training creates `checkpoints/best_model.pth`, restore an input directory with:

```bash
python inference.py --input_dir data/val/NoisyLR --output_dir results/restored
```

Useful optional arguments are shown below.

```bash
python inference.py \
  --input_dir path/to/new_images \
  --output_dir results/spot_output \
  --checkpoint checkpoints/best_model.pth \
  --device cpu \
  --batch-size 1
```

Supported input formats are PNG, JPG/JPEG, TIFF, and BMP. For matching-sized images, inference groups images into batches when the requested batch size permits. It reports measured end-to-end wall-clock time covering input reading, preprocessing, transfer, model execution, postprocessing, and output writing.

### Lower-resolution official inputs

The local synthetic dataset stores NoisyLR images resized back to GT resolution. If a future dataset instead provides an input that is physically smaller than the required output, use the configured inference upscale factor. For example, an image requiring a two-fold spatial expansion may be restored with:

```bash
python inference.py \
  --input_dir path/to/low_resolution_inputs \
  --output_dir results/restored \
  --checkpoint checkpoints/best_model.pth \
  --upscale-factor 2
```

Use the actual scale specified by the dataset rather than assuming a value. During paired training, the loader safely resizes a smaller NoisyLR image to the GT dimensions before aligned cropping.

## Validation and evaluation

When GT is available, compute actual validation metrics with:

```bash
python evaluate.py \
  --input_dir data/val/NoisyLR \
  --target_dir data/val/GT \
  --checkpoint checkpoints/best_model.pth \
  --device cpu
```

The evaluator writes a JSON report to `results/validation_metrics.json` and prints the measured values. It reports **PSNR** and **SSIM** by default. To request LPIPS as well, use:

```bash
python evaluate.py \
  --input_dir data/val/NoisyLR \
  --target_dir data/val/GT \
  --checkpoint checkpoints/best_model.pth \
  --lpips
```

LPIPS may download or initialize its reference perceptual network the first time it is used. If that environment step fails, the command reports the real error rather than substituting a fabricated LPIPS value.

## Replacing synthetic data with official paired data

Keep the paired folder contract and filename matching:

```text
path/to/official_data/
├── train/
│   ├── GT/
│   └── NoisyLR/
└── val/
    ├── GT/
    └── NoisyLR/
```

Then update the four paths in `configs/train.yaml`. Preserve matching filenames. The data loader validates filename equality and raises an explicit error if an input has no matching target or vice versa. For official hidden test data, use `inference.py` only; no targets are needed and the script automatically creates the output directory.

If externally sourced datasets, pretrained models, or weights are added for a competition submission, document their name, URL, license, and permitted use before submission.

## Automated tests

Run the focused tests after installing the requirements:

```bash
pytest -q
```

The test suite covers RGB image round-tripping, shape-safe padding, checkpoint/model loading, inference tensor dimensions, and restored output generation. It uses temporary files and a small randomly initialized checkpoint; it does not claim restoration quality and does not replace training and validation on official data.

## Reproducibility and experiment hygiene

The configuration records the experiment seed and deterministic option. Training uses a lazy dataset loader rather than keeping all source images in RAM. Every training epoch appends loss, PSNR, SSIM, and elapsed seconds to `results/training_history.csv`. The checkpoint stores the configuration used to create it.

The repository intentionally ignores generated images, datasets, checkpoints, virtual environments, results, environment files, and editor-specific files. Do not commit data, model weights, secrets, or `.env` files to the public repository.

## Limitations

This implementation is a reproducible local baseline, not a claim of optimized competition performance. The synthetic SEM data is intentionally small and cannot demonstrate generalization to confidential or hidden KLA data. CPU training on 1024×1024 imagery can be slow; the default training path uses 256×256 patches to make local experimentation practical. Results, PSNR, SSIM, LPIPS, throughput, and visual quality must be reported only after they have actually been measured on the relevant data and hardware.

For a competition-ready submission, run controlled experiments on the official paired data, compare at least one baseline against the final method, measure end-to-end inference time on the target hardware, retain the actual output examples and failure cases, and document every external data/model resource.
