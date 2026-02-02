# =================================================================
# Qwen-Image-2512 RunPod Serverless Worker (Clean Build)
#
# This image contains only code and dependencies (~12-15 GB).
# Model weights (~55 GB) are downloaded at first startup to a
# RunPod network volume, then persist across all worker restarts.
#
# Model: Qwen/Qwen-Image-2512 (~20B params, BF16, ~55 GB)
#   - Text encoder: Qwen2.5-VL-7B-Instruct (~14 GB)
#   - Diffusion transformer: MMDiT (~26 GB)
#   - VAE: (~0.5 GB)
# Target GPU: RTX 4090 (24 GB) with enable_sequential_cpu_offload()
#
# Build:
#   docker build -t youruser/qwen-image:latest .
#
# Push:
#   docker push youruser/qwen-image:latest
#
# RunPod setup:
#   1. Create network volume (75+ GB) in your preferred region
#   2. Create template:
#      - Docker image: youruser/qwen-image:latest
#      - Container disk: 20 GB
#      - Volume mount: /runpod-volume
#      - Env var: HF_TOKEN=hf_xxxx (if model is gated)
#      - Env var: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   3. Create serverless endpoint from template
#      - GPU: RTX 4090 (24 GB)
#      - Attach your network volume
# =================================================================

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ----- System dependencies -----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ----- PyTorch 2.6 with CUDA 12.4 -----
RUN pip install --no-cache-dir \
    "torch~=2.6" \
    --index-url https://download.pytorch.org/whl/cu124

# ----- Diffusers from git (QwenImagePipeline not in any PyPI release) -----
RUN pip install --no-cache-dir \
    "git+https://github.com/huggingface/diffusers.git" \
    "transformers>=4.57.0" \
    "accelerate>=0.33.0" \
    "safetensors>=0.4.0" \
    "Pillow>=10.0.0" \
    "sentencepiece" \
    "protobuf"

# ----- RunPod SDK + download tools -----
RUN pip install --no-cache-dir runpod "huggingface_hub[hf_transfer]"

# Verify critical imports at build time
RUN python3 -c "import runpod; print(f'runpod {runpod.__version__}')" && \
    python3 -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}')" && \
    python3 -c "from diffusers import DiffusionPipeline; print('diffusers OK')" && \
    python3 -c "import transformers; print(f'transformers {transformers.__version__}')" && \
    python3 -c "import accelerate; print(f'accelerate {accelerate.__version__}')"

# ----- Application -----
WORKDIR /app
COPY handler.py /app/handler.py
COPY test_input.json /test_input.json

ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["python3", "-u", "/app/handler.py"]
