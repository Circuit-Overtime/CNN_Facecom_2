# Base image with CUDA and Python 3.10 support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-distutils \
    curl git wget zip unzip \
    && apt-get clean

# Symlink python3.10 to python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set workdir
WORKDIR /workspace

# Create virtual environment
RUN python -m venv tf-gpu-env

# Copy and install requirements into the venv
COPY requirements.txt .
RUN /workspace/tf-gpu-env/bin/pip install --upgrade pip \
    && /workspace/tf-gpu-env/bin/pip install -r requirements.txt

# Copy your source files (optional)
COPY testA.py .
COPY testB.py .


