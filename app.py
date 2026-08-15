"""Local Streamlit UI for the KLA image-restoration pipeline."""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from inference import _prepare_tensor
from src.utils.checkpoints import load_restoration_model
from src.utils.runtime import select_device


DEFAULT_CHECKPOINT = "checkpoints/best_model.pth"


st.set_page_config(
    page_title="AI Image Restoration",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, requested_device: str):
    """Cache model weights between Streamlit reruns while preserving device selection."""
    device = select_device(requested_device)
    model, checkpoint = load_restoration_model(checkpoint_path, device)
    return model, checkpoint, device


def uploaded_to_rgb(uploaded_file) -> np.ndarray:
    """Decode an uploaded file to an RGB float32 HWC image."""
    try:
        with Image.open(uploaded_file) as image:
            image = image.convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("The uploaded file is not a valid readable image.") from error
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("The uploaded image must contain three RGB channels.")
    return array


def limit_image_size(image: np.ndarray, max_dimension: int) -> tuple[np.ndarray, bool]:
    """Downsize unusually large uploads before inference to protect local CPU memory."""
    height, width = image.shape[:2]
    largest_dimension = max(height, width)
    if largest_dimension <= max_dimension:
        return image, False
    scale = max_dimension / largest_dimension
    resized = Image.fromarray(np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)).resize(
        (max(1, round(width * scale)), max(1, round(height * scale)),),
        Image.Resampling.LANCZOS,
    )
    return np.asarray(resized, dtype=np.float32) / 255.0, True


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def restore_uploaded_image(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    upscale_factor: float,
    pad_multiple: int,
) -> tuple[Image.Image, float, tuple[int, int]]:
    """Restore one decoded upload and return a displayable image plus runtime metadata."""
    tensor, output_size = _prepare_tensor(image, upscale_factor, pad_multiple)
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = torch.clamp(model(tensor.unsqueeze(0).to(device)), 0.0, 1.0)[0].cpu()
    elapsed = time.perf_counter() - started
    height, width = output_size
    restored = prediction[:, :height, :width].permute(1, 2, 0).numpy()
    restored_image = Image.fromarray(np.rint(restored * 255.0).astype(np.uint8), mode="RGB")
    return restored_image, elapsed, (width, height)


def main() -> None:
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "restored_result" not in st.session_state:
        st.session_state.restored_result = None

    st.title("AI Image Restoration")
    st.write(
        "Upload a degraded SEM-style image, restore it with the trained residual U-Net, "
        "compare the result with the original, and download the enhanced output."
    )

    with st.sidebar:
        st.header("Restoration settings")
        checkpoint_path = st.text_input("Checkpoint path", value=DEFAULT_CHECKPOINT)
        device_choice = st.selectbox("Execution device", ["auto", "cpu", "cuda"], index=0)
        upscale_factor = st.number_input("Upscale factor", min_value=0.1, value=1.0, step=0.5)
        max_dimension = st.number_input("Maximum input dimension", min_value=128, value=2048, step=128)
        st.caption("The default checkpoint is created by the training pipeline. CUDA is used only when available.")

    col_upload, col_actions = st.columns([4, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a degraded image",
            type=["jpg", "jpeg", "png"],
            key=f"uploader_{st.session_state.uploader_key}",
            help="Supported formats: JPG, JPEG, and PNG.",
        )
    with col_actions:
        st.write("")
        st.write("")
        if st.button("Reset", use_container_width=True):
            st.session_state.uploader_key += 1
            st.session_state.restored_result = None
            st.rerun()

    if uploaded_file is None:
        st.info("Upload an image to begin restoration.")
        return

    try:
        original = uploaded_to_rgb(uploaded_file)
    except ValueError as error:
        st.error(str(error))
        return

    original, was_resized = limit_image_size(original, int(max_dimension))
    if was_resized:
        st.warning(
            f"The upload was resized to protect local memory. Maximum dimension: {int(max_dimension)} pixels."
        )

    original_display = Image.fromarray(np.rint(original * 255.0).astype(np.uint8), mode="RGB")
    st.subheader("Original image")
    st.image(original_display, use_container_width=True)

    restore_clicked = st.button("Restore image", type="primary", use_container_width=True)
    if restore_clicked:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            st.error(
                f"Checkpoint not found at '{checkpoint_path}'. Train the model first or choose a valid checkpoint path."
            )
            return
        try:
            with st.spinner("Loading the restoration model and processing the image..."):
                model, checkpoint_metadata, device = load_model(str(checkpoint), device_choice)
                configured_scale = checkpoint_metadata.get("inference_config", {}).get("upscale_factor", 1)
                effective_scale = float(upscale_factor) if upscale_factor != 1.0 else float(configured_scale)
                pad_multiple = int(checkpoint_metadata.get("inference_config", {}).get("pad_multiple", 4))
                restored, elapsed, output_size = restore_uploaded_image(
                    original, model, device, effective_scale, pad_multiple
                )
            st.session_state.restored_result = {
                "original": original_display,
                "restored": restored,
                "runtime": elapsed,
                "output_size": output_size,
                "device": str(device),
            }
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            st.error(f"Restoration failed: {error}")
            return

    result = st.session_state.restored_result
    if result is None:
        st.caption("Click **Restore image** to run the AI model.")
        return

    st.subheader("Restoration comparison")
    comparison_left, comparison_right = st.columns(2)
    with comparison_left:
        st.caption("Original")
        st.image(result["original"], use_container_width=True)
    with comparison_right:
        st.caption("Restored")
        st.image(result["restored"], use_container_width=True)

    output_bytes = image_to_png_bytes(result["restored"])
    metrics_left, metrics_right = st.columns(2)
    metrics_left.metric("Output size", f"{result['output_size'][0]} × {result['output_size'][1]}")
    metrics_right.metric("Model runtime", f"{result['runtime']:.3f} s")
    st.caption(f"Execution device: {result['device']}")
    st.download_button(
        "Download restored PNG",
        data=output_bytes,
        file_name="restored_image.png",
        mime="image/png",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
