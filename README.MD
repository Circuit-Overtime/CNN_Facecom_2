# 🧠 CNN Facecom — Quick Start Guide

This guide explains how to set up and run the **Gender Classification** (Task A) and **Face Verification** (Task B) solutions, including model requirements, environment setup, and model architectures.

---


# 1. Setup Environment 

## 🧬 Conda Environment (Recommended)

For guaranteed reproducibility across systems, we provide a ready-to-use Conda environment file directly in this repository.

### 🔧 Setup Instructions (CPU/GPU Compatible)

1. **Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)**

   - ✅ During installation, **check the box to add Conda to your PATH**.
   - ✅ After installing, open a new terminal and run:
     ```bash
     conda init powershell  # or conda init bash/zsh depending on your shell
     ```
   - ✅ Restart the terminal to apply changes. (close and open the same terminal again)

2. **Create the Conda environment from the included YAML file**:
   ```bash
   conda env create -f facecom_env.yml
   ```

3. **Activate the Environment**
    ```bash
    conda activate facecom
    ```

4. **Run inference For Gender Prediction**:
    ```bash
    python PRODUCTION/Task_A/inference/inference_vgg19_updated.py
    ```
5. **Run inference For Face Classification**:
    ```bash
    python PRODUCTION/Task_B/inference/verify_face.py
    ```
6. **Deactivate the Environment**
    ```bash
    conda deactivate
    ```

---



## 2. Download Pretrained Models

- **Task A (Gender Classification):**  
    Download `TASK_A.h5` from [releases](https://github.com/Circuit-Overtime/CNN_Facecom_2/releases/tag/publish102) and place it in the `models/` directory.

- **Task B (Face Verification):**  
    Download `TASK_B.h5` from [releases](https://github.com/Circuit-Overtime/CNN_Facecom_2/releases/tag/publish101) and place it in the `models/` directory.

---

## 3. Prepare Data

- **Task A:**  
    - Place validation/test images in:
        - `E:\CNN_vedic\Data\Task_A\val\male\`
        - `E:\CNN_vedic\Data\Task_A\val\female\`

- **Task B:**  
    - Place validation/test folders in:
        - `E:\CNN_vedic\Data\Task_B\val\`
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