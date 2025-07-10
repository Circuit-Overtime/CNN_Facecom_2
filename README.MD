# 🧠 CNN Facecom — Quick Start Guide

This guide explains how to set up and run the **Gender Classification** (Task A) and **Face Verification** (Task B) solutions, including model requirements, environment setup, and model architectures.

---

## 1. Setup Environment (via Docker 🐳)

The easiest way to run the models is by using our pre-configured Docker container with a built-in Python 3.10 virtual environment (`tf-gpu-env`). This setup ensures full compatibility with TensorFlow-GPU and all dependencies — without needing to install anything manually.

> 🧠 Tested on:  
> - ✅ Windows 11 + Docker Desktop + WSL2  
> - ✅ Ubuntu 22.04 (x86_64 AMD CPU, with NVIDIA GPU)  
> - ✅ macOS (M1/M2 not recommended unless using CPU-only)

> ⏳ **Time to setup:** ~30 minutes (with 22 Mbps internet)
### 🛠️ Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop) (or native Docker Engine on Linux)
  - ✅ **WSL2 backend enabled** (for Windows)
  - ✅ **GPU support enabled** (if using TensorFlow-GPU)
- Internet connection ≥ **22 Mbps** (see time note below)
- GPU drivers installed:
  - NVIDIA Driver (≥ R515+)
  - NVIDIA Container Toolkit (for GPU access inside Docker)


### ⚡ Setup Steps

> 🧪 This script builds the Docker image, installs all Python dependencies inside a virtualenv, downloads pretrained models, and extracts everything to your host.

```bash
# Clone the repo
git clone https://github.com/Circuit-Overtime/CNN_Facecom_2.git
cd CNN_Facecom_2

# 🖥️ If you're on Windows with WSL2:
wsl bash install.sh

# 🐧 If you're on native Linux/macOS:
bash install.sh

# ✅ Activate the virtual environment
source tf-gpu-env/bin/activate

# ▶️ Run the test scripts
python testA.py   # Gender Classification
python testB.py   # Face Verification
```
---


## 2. Download Pretrained Models (Exclusive Download if you want)
> (Don't worry docker will do it by itself no need to do this manually)

- **Task A (Gender Classification):**  
    Download `TASK_A.h5` from [releases](https://github.com/Circuit-Overtime/CNN_Facecom_2/releases/tag/publish102) and place it in the `models/` directory.

- **Task B (Face Verification):**  
    Download `TASK_B.h5` from [releases](https://github.com/Circuit-Overtime/CNN_Facecom_2/releases/tag/publish101) and place it in the `models/` directory.

---

## 3. Prepare Data

- **Task A:**  
    - Place validation/test images in:
        - `E:\CNN_Facecom_2\Data\Task_A\val\male\`
        - `E:\CNN_Facecom_2\Data\Task_A\val\female\`

- **Task B:**  
    - Place validation/test folders in:
        - `E:\CNN_Facecom_2\Data\Task_B\val\`
    - Each identity folder should contain one or many reference image and a `distortion/` subfolder with distorted images.

---

## 4. Run the Scripts

### Task A: Gender Classification

```bash
python testA.py
```

- Evaluates all images in the male and female folders.
- Prints a classification report (precision, recall, F1-score).
- Optionally, set `SHOW_GRADCAM = True` in `testA.py` to visualize Grad-CAM overlays (requires GPU).

### Task B: Face Verification

```bash
python testB.py
```

- Evaluates face matching for all identities and their distorted images.
- Prints match results and overall accuracy.

---

## 5. Model Architectures

### Task A: Gender Classification (VGG19)

- **Base:** VGG19 (pretrained, `include_top=False`)
- **Custom Head:**
        - GlobalAveragePooling2D
        - BatchNormalization
        - Dense(512, relu) → Dropout(0.5)
        - Dense(256, relu) → Dropout(0.3)
        - Dense(2, softmax)
- **Loss:** Focal loss (`gamma=2.0, alpha=0.5`)
- **Optimizer:** Adam (`lr=1e-4` initial, `1e-5` fine-tuning)
- **Training:**  
        - Initial: Only custom head trainable  
        - Fine-tuning: Last 8 layers of VGG19 unfrozen

**Visualization:**
```
Input (224x224x3)
     │
[VGG19 base (frozen/unfrozen)]
     │
GlobalAveragePooling2D
     │
BatchNormalization
     │
Dense(512, relu) → Dropout(0.5)
     │
Dense(256, relu) → Dropout(0.3)
     │
Dense(2, softmax)
```

---

### Task B: Face Verification (Triplet Network, ResNet50)

- **Embedding Model:**
        - ResNet50 (pretrained, frozen for initial training)
        - GlobalAveragePooling2D
        - Dense(256)
        - L2 normalization
- **Triplet Model:**
        - Three inputs: Anchor, Positive (distorted), Negative
        - Shared embedding model for all inputs
        - Embeddings concatenated
        - **Loss:** Triplet loss (margin=0.3)

**Diagram:**
```
Anchor Img ─┐
            │
Positive ───┼─> [Shared Embedding Model] ──┬─> [Triplet Loss]
Negative ───┘                              │
                                           └─> [Concatenation]
```

---

## References

- [Task A Model Release](https://github.com/Circuit-Overtime/CNN_vedic_2/releases/tag/publish102)
- [Task B Model Release](https://github.com/Circuit-Overtime/CNN_vedic_2/releases/tag/publish101)
- [TensorFlow GPU Install Guide](https://www.tensorflow.org/install/gpu)

---
