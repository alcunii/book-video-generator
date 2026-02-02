# =================================================================
# MusicGen-medium RunPod Serverless Worker (Clean Build)
#
# This image contains only code and dependencies (~8-10 GB).
# Model weights (~6 GB) are downloaded at first startup to a
# RunPod network volume, then persist across all worker restarts.
#
# Model: facebook/musicgen-medium (~1.5B params, ~6 GB on disk)
# Output: WAV audio at 32 kHz sample rate
# Target GPUs: L4 / T4 / A40
#
# Build:
#   docker build -t youruser/musicgen:latest .
#
# Push:
#   docker push youruser/musicgen:latest
#
# RunPod setup:
#   1. Create network volume (10+ GB) in your preferred region
#   2. Create template:
#      - Docker image: youruser/musicgen:latest
#      - Container disk: 15 GB
#      - Volume mount: /runpod-volume
#      - Env var: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   3. Create serverless endpoint from template
#      - GPU: L4 / T4 / A40
#      - Attach your network volume
# =================================================================

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ----- System dependencies -----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git \
    ffmpeg \
    libsndfile1 \
    pkg-config \
    libavformat-dev libavcodec-dev libavutil-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ----- PyTorch 2.6 + torchaudio with CUDA 12.4 -----
RUN pip install --no-cache-dir \
    "torch~=2.6" \
    "torchaudio~=2.6" \
    --index-url https://download.pytorch.org/whl/cu124

# ----- Audiocraft (Meta's music generation library) -----
RUN pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/audiocraft.git"

# ----- RunPod SDK + download tools + audio export -----
RUN pip install --no-cache-dir \
    runpod \
    "huggingface_hub[hf_transfer]" \
    scipy

# Verify critical imports at build time
RUN python3 -c "import runpod; print(f'runpod {runpod.__version__}')" && \
    python3 -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')" && \
    python3 -c "import torchaudio; print(f'torchaudio {torchaudio.__version__}')" && \
    python3 -c "from audiocraft.models import MusicGen; print('audiocraft OK')" && \
    python3 -c "import scipy; print(f'scipy {scipy.__version__}')"

# ----- Application -----
WORKDIR /app
COPY handler.py /app/handler.py
COPY test_input.json /test_input.json

ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["python3", "-u", "/app/handler.py"]
