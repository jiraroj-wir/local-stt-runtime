# CUDA runtime base for faster-whisper/CTranslate2 GPU execution.
# Adjust this image later if faster-whisper/CTranslate2 CUDA runtime requirements change.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/huggingface
ENV XDG_CACHE_HOME=/cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements-container.txt /tmp/requirements-container.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-container.txt

ENTRYPOINT ["python3"]
