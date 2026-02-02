"""
MusicGen-medium RunPod Serverless Handler

Generates music from text prompts using the facebook/musicgen-medium
model. Model weights are downloaded on first startup to a RunPod
network volume at /runpod-volume/models/. Once downloaded, they
persist across all worker restarts.

Model: facebook/musicgen-medium (~1.5B params, ~6 GB on disk)
Output: WAV audio at 32 kHz sample rate
Target GPUs: L4 / T4 / A40

API contract (matches runpod_client.py):

Input:
{
    "input": {
        "prompt": "Calm ambient music with soft piano...",
        "duration": 30,
        "temperature": 1.0,
        "top_k": 250,
        "top_p": 0.0,
        "seed": 42
    }
}

Output:
{
    "audio_base64": "<base64 WAV>",
    "seed": 42,
    "duration": 30,
    "sample_rate": 32000,
    "format": "wav"
}
"""

import base64
import io
import os
import traceback

import runpod
import scipy.io.wavfile as wavfile
import torch

# ---------------------------------------------------------------------------
# Paths on the network volume
# ---------------------------------------------------------------------------
VOLUME_DIR = os.environ.get("RUNPOD_VOLUME_DIR", "/runpod-volume")
MODELS_DIR = os.path.join(VOLUME_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "musicgen-medium")
MODEL_REPO = "facebook/musicgen-medium"

# Force HuggingFace cache onto the network volume (guard against missing dir)
HF_CACHE_DIR = os.path.join(VOLUME_DIR, "hf_cache")
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = HF_CACHE_DIR

_tmpdir = os.path.join(VOLUME_DIR, "tmp")
try:
    os.makedirs(_tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = _tmpdir
except OSError:
    print(f"WARNING: Could not create {_tmpdir}, using system default TMPDIR")

# Global model instance (loaded once per worker lifecycle)
_model = None


# ---------------------------------------------------------------------------
# Model download (runs once per network volume lifetime)
# ---------------------------------------------------------------------------
def _model_is_complete():
    """Check that the model weights (not just config) are present."""
    if not os.path.isdir(MODEL_PATH):
        return False
    # state_dict.bin is the main ~3.3GB weights file for musicgen-medium
    weights = os.path.join(MODEL_PATH, "state_dict.bin")
    config = os.path.join(MODEL_PATH, "config.json")
    return os.path.exists(weights) and os.path.exists(config)


def ensure_model():
    """Download model weights to network volume if not already present."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    if _model_is_complete():
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

    if not _model_is_complete():
        raise RuntimeError(
            f"Download finished but model files are incomplete at {MODEL_PATH}. "
            "Check disk space on the network volume."
        )

    print(f"Model saved to {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def get_model():
    """Load MusicGen-medium model. Cached across requests."""
    global _model
    if _model is not None:
        return _model

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. MusicGen requires a GPU. "
            "Check that the NVIDIA driver and CUDA toolkit are installed."
        )

    print(f"Loading MusicGen-medium model on {torch.cuda.get_device_name(0)}...")
    from audiocraft.models import MusicGen

    _model = MusicGen.get_pretrained(MODEL_PATH)
    print(f"Model loaded on {next(_model.lm.parameters()).device}")
    return _model


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
def handler(event):
    """RunPod serverless handler - generates music from text prompt."""
    input_data = event.get("input", {})

    prompt = input_data.get("prompt")
    if not prompt:
        return {"error": "Missing required field: prompt"}

    try:
        duration = int(input_data.get("duration", 30))
        temperature = float(input_data.get("temperature", 1.0))
        top_k = int(input_data.get("top_k", 250))
        top_p = float(input_data.get("top_p", 0.0))
    except (ValueError, TypeError) as e:
        return {"error": f"Invalid parameter: {e}"}

    # Clamp duration (MusicGen works best up to 30s, can do 60s)
    duration = max(1, min(60, duration))

    seed = input_data.get("seed")
    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    seed = int(seed)

    print(f"Generating: duration={duration}s seed={seed} prompt='{prompt[:80]}'")

    try:
        musicgen = get_model()

        # Set generation parameters
        musicgen.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Generate audio
        with torch.inference_mode():
            wav = musicgen.generate([prompt])

        sample_rate = musicgen.sample_rate

        # wav shape: (batch=1, channels, samples) -> squeeze to (samples,)
        audio_data = wav[0].cpu().numpy().squeeze()

        # Normalize float audio to int16 for WAV export
        audio_int16 = (audio_data * 32767).astype("int16")

        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_int16)
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        size_kb = len(buffer.getvalue()) / 1024
        print(f"Done: {size_kb:.0f} KB, {duration}s at {sample_rate} Hz")

        return {
            "audio_base64": audio_base64,
            "seed": seed,
            "duration": duration,
            "sample_rate": sample_rate,
            "format": "wav",
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Startup: download model (if needed) and pre-load
# ---------------------------------------------------------------------------
print("Starting MusicGen-medium handler...")
ensure_model()
get_model()
print("Handler ready.")

runpod.serverless.start({"handler": handler})
