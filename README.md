# Occlusion-Aware Face Recognition Pipeline

End-to-end workflow for:

* building a **bio database**,
* extracting **gallery features** (23-point landmarks + 128-D encodings),
* evaluating thresholds **τ** for 1:1 and 1\:N,
* running the **pipeline** over folders or a **live camera**, with full logging.

---

## Contents

* [Prerequisites](#prerequisites)
* [One-time Setup](#one-time-setup)
* [Required Model Files](#required-model-files)
* [1) Build the Bio DB](#1-build-the-bio-db)
* [2) Extract Gallery Features](#2-extract-gallery-features)
* [3) Evaluate & Pick Thresholds (τ)](#3-evaluate--pick-thresholds-τ)
* [4) Run the Pipeline (Folder / Camera)](#4-run-the-pipeline-folder--camera)
* [“Must Look” Facts](#must-look)

---

## Prerequisites

* Python (with `pip`)
* A C/C++ toolchain suitable for your platform (needed by some CV packages)
* A webcam (for camera mode)

> If you only need CPU inference, standard wheels work. If you plan to use GPU, ensure your CUDA/cuDNN stack matches the installed packages.

---

## One-time Setup

### macOS/Linux

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U opencv-python dlib face_recognition numpy tensorflow
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U opencv-python dlib face_recognition numpy tensorflow
```

---

## Required Model Files

Place these files in `models/`:

* `haarcascade_frontalface_default.xml`
* `shape_predictor_68_face_landmarks.dat`
* `cnn_model.h5`  (your mask model)

**Optional (for Yunet detector):**

* `face_detection_yunet_2023mar.onnx`

---

## 1) Build the Bio DB

Folders → one row per identity.

```bash
python build_bio_db.py \
  --images-root /data/people_root \
  --bios-json   /data/people_bios.json \
  --db          Eagle_Eye_Detection_Pipeline.db
```

---

## 2) Extract Gallery Features

Normalized **23-pt** landmarks + **128-D** encodings.

```bash
python build_facial_data.py \
  --db               Eagle_Eye_Detection_Pipeline.db \
  --cascade          models/haarcascade_frontalface_default.xml \
  --shape-predictor  models/shape_predictor_68_face_landmarks.dat \
  --min-face-size    96 \
  --batch            200
```

---

## 3) Evaluate & Pick Thresholds (τ)

### 1:1 operating point (typical)

```bash
python build_pairs_and_metrics.py \
  --db Eagle_Eye_Detection_Pipeline.db \
  --impostor-mode stratified --stratified-per-row 8 --impostor-max 100000 \
  --target-fmr 0.01 \
  --seed 123 \
  --notes "baseline τ (FMR=1%)"
```

> **Note** the printed `run_id` (e.g., `run_id=7`).

### 1\:N requirement (e.g., FPIR ≤ 1% with N=5000)

```bash
python build_pairs_and_metrics.py \
  --db Eagle_Eye_Detection_Pipeline.db \
  --impostor-mode stratified --stratified-per-row 8 --impostor-max 200000 \
  --target-fpir 0.01 --gallery-size 5000 \
  --seed 123 \
  --notes "1:N τ for FPIR≤1% (N=5k)"
```

> **Note** the printed `run_id` (e.g., `run_id=8`).

---

## 4) Run the Pipeline (Folder / Camera)

### Folder mode (Yunet ONNX)

```bash
python pipeline.py \
  --detector yunet \
  --yunet-model models/face_detection_yunet_2023mar.onnx \
  --shape-predictor models/shape_predictor_68_face_landmarks.dat \
  --mask-model models/cnn_model.h5 \
  --db Eagle_Eye_Detection_Pipeline.db \
  --eval-run-id 5 \
  --images /data/to_score \
  --mask-try-both --mask-th-low 0.25 --mask-th-high 0.80 --mask-temp 1.0 \
  --distance enc \
  --log-comparisons \
  --log-dir logs
```

### Camera mode (Yunet ONNX)

```bash
python pipeline.py \
  --detector yunet \
  --yunet-model models/face_detection_yunet_2023mar.onnx \
  --shape-predictor models/shape_predictor_68_face_landmarks.dat \
  --mask-model models/cnn_model.h5 \
  --db Eagle_Eye_Detection_Pipeline.db \
  --eval-run-id 5 \
  --camera \
  --mask-try-both --mask-th-low 0.25 --mask-th-high 0.80 --mask-temp 1.0 \
  --distance enc \
  --log-comparisons \
  --log-dir logs
```

> Replace `--eval-run-id` with the run ID you recorded in step 3.

---

## Must Look

* The pipeline loads **τ** from `eval_metrics(run_id=...)`.
* **23-pt normalization** for probes exactly matches training.
* Uses **RMS** (landmarks) and **1 − cos** (encodings).
* Writes an **audit row per face** to `logs/runtime_matches.csv`:

  * distance, τ, decision, run\_id, gallery size, expected FPIR.

---

## Speed Tips

* Use `--min-face-size 96–128` to skip tiny faces (biggest speed win).
* Keep `num_jitters=1` and `models="small"` for encodings.
* Re-run **step 3** whenever the gallery grows by \~20–30% (τ may need tightening for 1\:N).
* Monitor `logs/runtime_matches.csv` and `logs/faces_log.csv` for drift checks.
* For large galleries during evaluation, use:

  ```
  --impostor-mode random --impostor-max 300000
  ```

  Then confirm τ with a deeper run later.

---

### Windows PowerShell line-continuation examples (optional)

PowerShell uses backticks ( \` ) to split long lines:

```powershell
python pipeline.py `
  --detector yunet `
  --yunet-model models/face_detection_yunet_2023mar.onnx `
  --shape-predictor models/shape_predictor_68_face_landmarks.dat `
  --mask-model models/mask_detector.h5 `
  --db Eagle_Eye_Detection_Pipeline.db `
  --eval-run-id 5 `
  --images .\data\to_score `
  --mask-try-both --mask-th-low 0.25 --mask-th-high 0.80 --mask-temp 1.0 `
  --distance enc `
  --log-comparisons `
  --log-dir logs
```
---

## Training: CNN (mask detector) and SVM (identity)

This repo supports two trainable components:

* **CNN mask detector** → outputs `P(masked)` per face (used by the routing logic).
* **SVM identity classifier (optional)** → predicts `person_id` from features (128-D encodings or 23-pt landmarks).

**Notebooks:**

* `notebooks/cnn_training.ipynb`
* `notebooks/train_svm.ipynb`

## 0) Environment

```bash
python -m venv .venv
# or: conda create -n eagleeye python=3.10 -y && conda activate eagleeye
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
# If you don't have it already:
# pip install opencv-python dlib face_recognition numpy pandas scikit-learn tensorflow jupyter
```

## 1) Data layout

### 1.1 Mask detector (CNN)

Binary classification: **masked** vs **nomask**.

```
data/mask_train/
  masked/
    img_001.jpg
    ...
  nomask/
    img_101.jpg
    ...
data/mask_val/
  masked/
  nomask/
```

### 1.2 SVM identity (optional)

Multiclass classification: `person_id` as the class label. Use either:

* **Encodings branch** (recommended for no-mask): 128-D vectors from `face_recognition`.
* **Landmarks branch** (works under masks): flattened, normalized 23×2 points → **46-D** vector.

## 2) Train the CNN mask detector

Open the notebook and run all cells:

```bash
jupyter notebook notebooks/cnn_training.ipynb
```

**The notebook:**

* Builds a small CNN.
* Trains on `data/mask_train/` with validation on `data/mask_val/`.
* Exports the model to `models/cnn_model.h5`.  *(If you prefer `models/mask_detector.h5`, just rename and use that in the CLI.)*

**Verify before using in the pipeline:**

* Final validation accuracy / F1.
* Confusion matrix (balanced precision/recall between masked and nomask).
* Saved file exists: `models/cnn_model.h5`.

**Use in the pipeline:**

```bash
python pipeline.py \
  --shape-predictor models/shape_predictor_68_face_landmarks.dat \
  --yunet-model models/face_detection_yunet_2023mar.onnx \
  --mask-model models/cnn_model.h5 \
  --db Eagle_Eye_Detection_Pipeline.db \
  --eval-run-id <your_run_id> \
  --images /path/to/images \
  --mask-class-index 1 \
  --mask-th-low 0.25 --mask-th-high 0.75 --mask-try-both \
  --log-dir logs
```

> `--mask-class-index` must point to the **“Masked”** class index from your CNN.

## 3) Reproducibility

```python
import numpy as np, random, tensorflow as tf
np.random.seed(42); random.seed(42); tf.random.set_seed(42)
```

* Log versions: `tensorflow.__version__`, `opencv.__version__`, `sklearn.__version__`.
* Save training configs (hyper-params, class indices) to `models/<modelname>.json`.

## 4) Suggested hyper-parameters (starting points)

**CNN**

* Optimizer: Adam (lr=1e-3 → cosine decay / ReduceLROnPlateau)
* Batch size: 32
* Epochs: 15–30 (early stopping on val loss)
* Augmentation: random horizontal flip, color jitter, mild blur; avoid heavy occlusions that break labels
* Input size: match your model (e.g., 128×128)

**SVM**

* Encodings: `SVC(C=10, gamma='scale', kernel='rbf', probability=True)`
* Landmarks: `SVC(C=5, gamma='scale', kernel='rbf', probability=True)`
