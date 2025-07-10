# Base image with CUDA and Python 3.10 support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-distutils \
    curl git wget zip unzip \
    && apt-get clean

# Link python3.10 as default
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Create virtual environment
RUN python -m venv tf-gpu-env

# Copy and install Python requirements
COPY requirements.txt .
RUN /workspace/tf-gpu-env/bin/pip install --upgrade pip \
    && /workspace/tf-gpu-env/bin/pip install -r requirements.txt

# Copy source code (optional)
COPY testA.py .
COPY testB.py .

# Create and download model files into models/
RUN mkdir -p /workspace/models && \
    wget -O /workspace/models/TASK_A.h5 https://github.com/Circuit-Overtime/CNN_Facecom_2/releases/download/publish102/TASK_A.h5 && \
    wget -O /workspace/models/TASK_B.h5 https://github.com/Circuit-Overtime/CNN_Facecom_2/releases/download/publish101/TASK_B.h5

# Make the venv activation available in bash
SHELL ["/bin/bash", "-c"]
RUN echo "source /workspace/tf-gpu-env/bin/activate" >> ~/.bashrc
