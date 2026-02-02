# Qwen-Image-2512 RunPod Serverless Worker
# Model weights are NOT baked in — downloaded at runtime to network volume

FROM runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy handler code
COPY handler.py /app/handler.py
COPY test_input.json /test_input.json

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Start the handler
CMD ["python", "-u", "/app/handler.py"]
