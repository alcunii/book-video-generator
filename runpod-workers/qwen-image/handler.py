"""
Qwen-Image-2512 RunPod Serverless Handler

Generates images from text prompts using the Qwen/Qwen-Image-2512
diffusion model. Model weights are downloaded on first startup to a
RunPod network volume at /runpod-volume/models/. Once downloaded,
they persist across all worker restarts.

Model: Qwen-Image-2512 (~20B params, BF16, ~55 GB on disk)
  - Text encoder: Qwen2.5-VL-7B-Instruct (~14 GB)
  - Diffusion transformer: MMDiT (~26 GB)
  - VAE: (~0.5 GB)
Target GPU: RTX 4090 (24 GB) with enable_model_cpu_offload()

API contract (matches runpod_client.py / Agent 04):

Input:
{
    "input": {
        "prompt": "A serene person meditating...",
        "negative_prompt": "text, watermark, blurry...",
        "width": 928,
        "height": 1664,
        "num_inference_steps": 30,
        "cfg_scale": 4.0,
        "seed": 42
    }
}

Output:
{
    "image_base64": "<base64 PNG>",
    "seed": 42,
    "width": 928,
    "height": 1664
}
"""

import base64
import io
import os
import traceback

import runpod
import torch

# ---------------------------------------------------------------------------
# Paths on the network volume
# ---------------------------------------------------------------------------
VOLUME_DIR = os.environ.get("RUNPOD_VOLUME_DIR", "/runpod-volume")
MODELS_DIR = os.path.join(VOLUME_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "qwen-image-2512")
MODEL_REPO = "Qwen/Qwen-Image-2512"

# Force HuggingFace cache onto the network volume
HF_CACHE_DIR = os.path.join(VOLUME_DIR, "hf_cache")
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = HF_CACHE_DIR
os.environ["TMPDIR"] = os.path.join(VOLUME_DIR, "tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# Global pipeline instance (loaded once per worker lifecycle)
_pipeline = None


# ---------------------------------------------------------------------------
# Model download (runs once per network volume lifetime)
# ---------------------------------------------------------------------------
def ensure_model():
    """Download model weights to network volume if not already present."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Check for a sentinel file that indicates a complete download
    config_path = os.path.join(MODEL_PATH, "model_index.json")
    if os.path.exists(config_path):
        print(f"Model found at {MODEL_PATH}")
        return

    print(f"Downloading {MODEL_REPO} to {MODEL_PATH} ...")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN")

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_PATH,
        token=hf_token if hf_token else None,
    )
    print(f"Model saved to {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------
def get_pipeline():
    """Load Qwen-Image-2512 pipeline with cpu_offload. Cached across requests."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    print("Loading Qwen-Image-2512 pipeline...")
    from diffusers import DiffusionPipeline

    _pipeline = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )

    # Cast VAE to float32 to prevent NaN → black images
    if hasattr(_pipeline, "vae") and _pipeline.vae is not None:
        _pipeline.vae.to(dtype=torch.float32)

    # Memory optimizations for 24GB VRAM
    # sequential offload moves individual layers on/off GPU (not whole components)
    # — slower but the ~26GB transformer won't fit in 24GB VRAM as a single block
    _pipeline.enable_sequential_cpu_offload()
    if hasattr(_pipeline, "enable_attention_slicing"):
        _pipeline.enable_attention_slicing("max")
    if hasattr(_pipeline, "enable_vae_slicing"):
        _pipeline.enable_vae_slicing()

    print("Pipeline loaded and ready.")
    return _pipeline


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
def handler(event):
    """RunPod serverless handler - generates image from text prompt."""
    input_data = event.get("input", {})

    prompt = input_data.get("prompt")
    if not prompt:
        return {"error": "Missing required field: prompt"}

    try:
        negative_prompt = str(input_data.get("negative_prompt", ""))
        width = int(input_data.get("width", 928))
        height = int(input_data.get("height", 1664))
        num_inference_steps = int(input_data.get("num_inference_steps", 30))
        cfg_scale = float(input_data.get("cfg_scale", 4.0))
    except (ValueError, TypeError) as e:
        return {"error": f"Invalid parameter: {e}"}

    seed = input_data.get("seed")
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    seed = int(seed)

    # Enhance prompts (matches Agent 04 behavior)
    enhanced_prompt = (
        f"{prompt}, "
        "Ultra HD, 4K, cinematic composition, professional photography"
    )

    full_negative = (
        f"{negative_prompt}, "
        "text, words, letters, watermark, signature, blurry, "
        "low quality, distorted"
    ) if negative_prompt else (
        "text, words, letters, watermark, signature, blurry, "
        "low quality, distorted"
    )

    print(
        f"Generating: {width}x{height} steps={num_inference_steps} "
        f"cfg={cfg_scale} seed={seed}"
    )

    try:
        pipe = get_pipeline()

        # Device-agnostic generator (required for cpu_offload)
        generator = torch.Generator("cpu").manual_seed(seed)

        with torch.no_grad():
            image = pipe(
                prompt=enhanced_prompt,
                negative_prompt=full_negative,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=cfg_scale,
                generator=generator,
            ).images[0]

        # Encode to base64 PNG
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        size_kb = len(buffer.getvalue()) / 1024
        print(f"Done: {size_kb:.0f} KB, {width}x{height}")

        return {
            "image_base64": image_base64,
            "seed": seed,
            "width": width,
            "height": height,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Startup: download model (if needed) and pre-load pipeline
# ---------------------------------------------------------------------------
print("Starting Qwen-Image-2512 handler...")
ensure_model()
get_pipeline()
print("Handler ready.")

runpod.serverless.start({"handler": handler})
