# Occlusion-Aware Face Recognition System

End-to-end workflow for:

* Building a **bio database**
* Extracting **gallery features** (23-point landmarks + 128-D encodings)
* Evaluating thresholds **τ** for 1:1 and 1:N
* Running the **pipeline** over folders or a **live camera**, with full logging
* **Assessing mask-probability calibration** (Brier Score, ECE, reliability diagram)

---

## Contents

* [Prerequisites](#prerequisites)
* [One-time Setup](#one-time-setup)
* [Required Model Files](#required-model-files)
* [1) Build the Bio DB](#1-build-the-bio-db)
* [2) Extract Gallery Features](#2-extract-gallery-features)
* [3) Evaluate & Pick Thresholds (τ)](#3-evaluate--pick-thresholds-τ)
* [4) Run the Pipeline (Folder / Camera)](#4-run-the-pipeline-folder--camera)
* [5) Mask-Probability Calibration](#5-mask-probability-calibration)
* [Must Look](#must-look)
* [Speed Tips](#speed-tips)

---

## Prerequisites

* Python (with `pip`)
* A C/C++ compiler (needed by some CV packages)
* A webcam (optional)

If you only need CPU inference, standard wheels work. For GPU usage, ensure your CUDA/cuDNN versions match your installed packages.

---

## One-time Setup

### macOS/Linux

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U opencv-python dlib face_recognition numpy tensorflow pandas matplotlib
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U opencv-python dlib face_recognition numpy tensorflow pandas matplotlib
```

---

## Required Model Files

Place the following in `models/`:

* `haarcascade_frontalface_default.xml`
* `shape_predictor_68_face_landmarks.dat`
* `cnn_model.h5` — mask classifier
* *(Optional)* `face_detection_yunet_2023mar.onnx`

---

## 1) Build the Bio DB

```bash
python build_bio_db.py \
  --images-root /data/people_root \
  --bios-json   /data/people_bios.json \
  --db          Eagle_Eye_Detection_Pipeline.db
```

---

## 2) Extract Gallery Features

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

### 1:1 Operating Point

```bash
python build_pairs_and_metrics.py \
  --db Eagle_Eye_Detection_Pipeline.db \
  --impostor-mode stratified \
  --stratified-per-row 8 \
  --impostor-max 100000 \
  --target-fmr 0.01 \
  --seed 123 \
  --notes "baseline τ (FMR=1%)"
```

### 1:N Operating Point (FPIR guidance)

```bash
python build_pairs_and_metrics.py \
  --db Eagle_Eye_Detection_Pipeline.db \
  --impostor-mode stratified \
  --stratified-per-row 8 \
  --impostor-max 200000 \
  --target-fpir 0.01 \
  --gallery-size 5000 \
  --seed 123 \
  --notes "1:N τ for FPIR≤1% (N=5k)"
```

---

## 4) Run the Pipeline (Folder / Camera)

### Folder Mode (Yunet)

```bash
python pipeline.py \
  --detector yunet \
  --yunet-model models/face_detection_yunet_2023mar.onnx \
  --shape-predictor models/shape_predictor_68_face_landmarks.dat \
  --mask-model models/cnn_model.h5 \
  --db Eagle_Eye_Detection_Pipeline.db \
  --eval-run-id 5 \
  --images /data/to_score \
  --mask-try-both --mask-th-low 0.25 --mask-th-high 0.80 \
  --distance enc \
  --log-comparisons \
  --log-dir logs
```

---

## 5) **Mask-Probability Calibration**

Evaluate whether the mask classifier’s output (`mask_prob`) corresponds to *true* masked frequency.

We use:

* `runtime_matches.csv` → contains `mask_prob`, `mask_route`, `mask_th_low`, `mask_th_high`
* Optionally: a ground-truth mask CSV
* Or infer truth:

  ```python
  true_masked = (mask_route == "landmarks")
  ```

---

### **Step 1 — Compute Brier Score**

The Brier score measures the mean squared difference between predicted probabilities and binary truth:

```
Brier Score = (1/N) Σ (pᵢ − yᵢ)²
```

* pᵢ = mask_prob
* yᵢ = 1 if masked else 0

Lower = better calibration.

---

### **Step 2 — Compute Expected Calibration Error (ECE)**

Divide probability [0,1] into bins (e.g., 10).
For each bin **Bₘ**:

```
ECE = Σₘ (|Bₘ| / N) * | acc(Bₘ) − conf(Bₘ) |
```

Where:

* **acc(Bₘ)** = fraction truly masked in the bin
* **conf(Bₘ)** = average predicted mask_prob in the bin
* **|Bₘ|** = number of samples in the bin

Low ECE = well calibrated.

---

### **Step 3 — Reliability Diagram**

Produces a plot:

* x-axis: average predicted mask_prob per bin
* y-axis: actual masked fraction
* diagonal: perfect calibration

Expected output:

```
reliability_diagram.png
```

---

### **Step 4 — Run the Calibration Script**

```bash
python calibrate_mask_prob.py \
  --runtime-matches logs/runtime_matches.csv \
  --out-dir calibration/
```

Outputs:

```
calibration_summary.txt
reliability_diagram.png
```

Example summary:

```
Brier Score: 0.031245
Expected Calibration Error (ECE): 0.042180
```

You may report in Results:

> "Mask-probability calibration yielded a Brier score of 0.031 and ECE of 0.042, indicating well-calibrated mask-confidence predictions."

---

## Must Look

* The pipeline loads **τ** from `eval_metrics(run_id=...)`
* Landmarks branch uses **RMS** distance; embeddings use **Euclidean / 1 − cosine**
* Every probe writes an audit row to `logs/runtime_matches.csv`:

  * distance, τ, decision
  * mask_prob, mask_route
  * timings (detect, mask_pred, branch_time)

---

## Speed Tips

* Increase `--min-face-size 96–128`
* Use `--models small` in `face_recognition`
* Recompute τ whenever gallery grows 20–30%
* Use:

```bash
--impostor-mode random --impostor-max 300000
```

for large-scale sweep approximations.

---
