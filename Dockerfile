FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libfreetype6-dev \
        nano && \
    rm -rf /var/lib/apt/lists/*

# Copy Code
COPY ./ /usr/app/
WORKDIR /usr/app

# Install Python dependencies. The base image already provides the pinned
# torch/torchvision CUDA builds from requirements.txt.
RUN python -m pip install --upgrade pip && \
    grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/requirements-no-torch.txt && \
    python -m pip install -r /tmp/requirements-no-torch.txt && \
    python -c "import torch, torchvision; assert torch.__version__.startswith('2.2.0'), torch.__version__; assert torchvision.__version__.startswith('0.17.0'), torchvision.__version__; print('Verified PyTorch stack:', torch.__version__, torchvision.__version__)"
