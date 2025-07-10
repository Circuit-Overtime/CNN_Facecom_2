#!/bin/bash

set -e  # Exit on any error
IMAGE_NAME="tf-gpu-builder"
CONTAINER_NAME="temp-container"

echo "[+] Creating container from image: $IMAGE_NAME"
docker create --name $CONTAINER_NAME $IMAGE_NAME

echo "[+] Copying virtual environment..."
docker cp $CONTAINER_NAME:/workspace/tf-gpu-env ./tf-gpu-env

echo "[+] Copying model files..."
docker cp $CONTAINER_NAME:/workspace/models ./models

echo "[+] Copying source scripts..."
docker cp $CONTAINER_NAME:/workspace/testA.py .
docker cp $CONTAINER_NAME:/workspace/testB.py .

echo "[+] Cleaning up container..."
docker rm $CONTAINER_NAME

echo "[✅] Done. Activate your venv with:"
echo "     source tf-gpu-env/bin/activate"
echo "[✅] Run your tests with:"
echo "     python testA.py"
echo "     python testB.py"

