#!/usr/bin/env python3
"""
pipeline.py

End-to-end face pipeline with evaluated thresholds (τ), calibrated mask routing,
and selectable face detector (YuNet default; Haar optional):

- Detect faces: YuNet (ONNX via OpenCV FaceDetectorYN) or Haar.
- Mask classifier (.h5) → calibrated P(masked) with temperature.
- Routing:
    • If P(masked) ≥ th_high → landmarks branch (dlib 68 → 23 pts → normalize → RMS distance).
    • If P(masked) ≤ th_low  → encodings branch (128-D → Euclidean or 1−cos).
    • If th_low < P(masked) < th_high and --mask-try-both:
        Evaluate BOTH branches, accept via τ; if both accepted, pick normalized margin; else smaller distance.
- Galleries load from Facial_Data (so scoring matches evaluation).
- Loads τ from eval_metrics(run_id); decision = (distance ≤ τ).
- Records timings per face; legacy logs + audit log with τ/decision/FPIR + mask info.
- Keyboard (camera): R(start), P(pause), C(capture once), S(save), Q/ESC(quit).
- Folder mode: iterates only real image extensions; optional save (JSON filename→bio).

Requires:
  pip install opencv-python dlib face_recognition numpy tensorflow
  (YuNet is included in OpenCV 4.7+ as FaceDetectorYN; model .onnx needed)
"""

import argparse
import csv
import json
import math
import sqlite3
import sys
import time
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import dlib
import face_recognition
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime

# ---------- Constants ----------

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}

# 23-point subset (1-based iBUG indices): 17-21, 22-26, 36-41, 42-47, 27
SUBSET_IDX_1BASED = list(range(17, 22)) + list(range(22, 27)) + list(range(36, 42)) + list(range(42, 48)) + [27]
SUBSET_IDX = [i - 1 for i in SUBSET_IDX_1BASED]  # 0-based for dlib output

# For normalization with 68 points
LEYE_68 = list(range(36, 42))  # 36..41
REYE_68 = list(range(42, 48))  # 42..47

# ---------- Dataclasses for Logging ----------

@dataclass
class FaceTiming:
    source: str
    frame_idx: Optional[int]
    face_idx: int
    label_masked: int
    t_detect_ms: float
    t_mask_pred_ms: float
    t_landmarks_ms: float
    t_encoding_ms: float
    t_match_total_ms: float
    t_total_face_ms: float
    branch: str
    match_table: Optional[str]
    best_rowid: Optional[int]
    best_distance: Optional[float]

@dataclass
class ComparisonTiming:
    source: str
    frame_idx: Optional[int]
    face_idx: int
    table: str
    rowid: int
    distance: float
    t_compare_ms: float

# ---------- Utility: Files, DB ----------

def list_images(dir_path: Path) -> List[Path]:
    return [p for p in sorted(dir_path.rglob("*"))
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]

def ensure_db_dir(db_path: Path) -> None:
    if db_path.exists() and db_path.is_dir():
        raise RuntimeError(f"--db points to a directory, not a file: {db_path}")
    db_path.parent.mkdir(parents=True, exist_ok=True)

# ---------- Thresholds (τ) + Gallery from Facial_Data ----------

def load_thresholds(conn: sqlite3.Connection, run_id: int, override_json: Optional[str] = None):
    rows = conn.execute(
        "SELECT branch, threshold, fmr_achieved FROM eval_metrics WHERE run_id=?",
        (run_id,)
    ).fetchall()
    if not rows:
        raise RuntimeError(f"No eval_metrics rows found for run_id={run_id}. "
                           "Run build_pairs_and_metrics.py first.")
    thr = {br: {"tau": float(t), "fmr_achieved": float(f)} for br, t, f in rows}
    if override_json:
        for br, v in json.loads(override_json).items():
            thr.setdefault(br, {})["tau"] = float(v)
    return thr  # {'landmarks': {'tau', 'fmr_achieved'}, 'encodings': {...}}

def expected_fpir(fmr_achieved: Optional[float], gallery_size: int) -> Optional[float]:
    if fmr_achieved is None:
        return None
    f = max(0.0, min(1.0, float(fmr_achieved)))
    N = max(0, int(gallery_size))
    return 1.0 - (1.0 - f) ** N

def load_gallery_from_facial_data(conn: sqlite3.Connection):
    """
    Returns:
      lm_rows: List[(rowid, 23x2 np.ndarray)]  -- normalized, as stored
      enc_rows: List[(rowid, 128 np.ndarray)]
    """
    lm_rows, enc_rows = [], []
    for rowid, jlm, jenc in conn.execute(
        "SELECT rowid, face_landmarks, face_encodings FROM Facial_Data"
    ):
        try:
            lm = np.array(json.loads(jlm), np.float32)
            if lm.shape == (23, 2) and np.all(np.isfinite(lm)):
                lm_rows.append((int(rowid), lm))
        except Exception:
            pass
        try:
            enc = np.array(json.loads(jenc), np.float32).reshape(-1)
            if enc.shape == (128,) and np.all(np.isfinite(enc)):
                enc_rows.append((int(rowid), enc))
        except Exception:
            pass
    return lm_rows, enc_rows

# ---------- Distances & Landmark Normalization ----------

def rms_landmarks(a23x2: np.ndarray, b23x2: np.ndarray) -> float:
    diff = a23x2 - b23x2
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))

def euclidean_enc(a128: np.ndarray, b128: np.ndarray) -> float:
    d = a128 - b128
    return float(np.sqrt(np.dot(d, d)))

def cosine_enc(a128: np.ndarray, b128: np.ndarray) -> float:
    na = float(np.linalg.norm(a128)); nb = float(np.linalg.norm(b128))
    if na == 0.0 or nb == 0.0:
        return 1.0
    sim = float(np.dot(a128, b128) / (na * nb))
    return 1.0 - sim  # distance-like

def eye_center(pts68: np.ndarray, idxs: List[int]) -> np.ndarray:
    return np.mean(pts68[idxs, :], axis=0)

def extract_23(pts68: np.ndarray) -> np.ndarray:
    return pts68[SUBSET_IDX, :]

def normalize_23_from_68(pts68: np.ndarray, pts23: np.ndarray, eps: float = 1e-6) -> Optional[np.ndarray]:
    """
    Translate by eye-midpoint, scale by inter-ocular distance.
    Returns (23,2) float32 or None if invalid scale.
    """
    cL = eye_center(pts68, LEYE_68)
    cR = eye_center(pts68, REYE_68)
    inter = float(np.linalg.norm(cR - cL))
    if inter <= eps:
        return None
    center = (cL + cR) * 0.5
    return ((pts23 - center) / inter).astype(np.float32)

# ---------- DB (optional save to legacy per-branch tables) ----------

def ensure_table(conn: sqlite3.Connection, table: str, kind: str) -> None:
    """
    kind: "landmarks" -> 23x2 JSON under column 'landmarks'
          "encodings" -> 128 JSON under column 'encodings'
    """
    if kind == "landmarks":
        with conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    first_name TEXT NOT NULL,
                    last_name  TEXT NOT NULL,
                    age        INTEGER NOT NULL,
                    landmarks  TEXT NOT NULL
                )
                """
            )
    elif kind == "encodings":
        with conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    first_name TEXT NOT NULL,
                    last_name  TEXT NOT NULL,
                    age        INTEGER NOT NULL,
                    encodings  TEXT NOT NULL
                )
                """
            )
    else:
        raise ValueError("kind must be 'landmarks' or 'encodings'")

def insert_landmarks(conn: sqlite3.Connection, table: str, first: str, last: str, age: int, pts23_norm: np.ndarray) -> None:
    payload = json.dumps([[float(x), float(y)] for x, y in pts23_norm.tolist()], separators=(",", ":"))
    with conn:
        conn.execute(f"INSERT INTO {table} (first_name,last_name,age,landmarks) VALUES (?,?,?,?)",
                     (first, last, int(age), payload))

def insert_encoding(conn: sqlite3.Connection, table: str, first: str, last: str, age: int, enc128: np.ndarray) -> None:
    payload = json.dumps([float(x) for x in enc128.tolist()], separators=(",", ":"))
    with conn:
        conn.execute(f"INSERT INTO {table} (first_name,last_name,age,encodings) VALUES (?,?,?,?)",
                     (first, last, int(age), payload))

# ---------- Face Enc/LM Helpers ----------

def predict_68_pts(gray: np.ndarray, rect: Tuple[int, int, int, int], predictor: dlib.shape_predictor) -> np.ndarray:
    x, y, w, h = rect
    drect = dlib.rectangle(left=int(x), top=int(y), right=int(x + w), bottom=int(y + h))
    shape = predictor(gray, drect)
    pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)
    return pts

def encode_face_bgr(img_bgr: np.ndarray, rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = rect
    top, right, bottom, left = y, x + w, y + h, x
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(img_rgb, known_face_locations=[(top, right, bottom, left)],
                                           num_jitters=1, model="small")
    if not encs:
        return None
    return np.asarray(encs[0], dtype=np.float32)

# ---------- Mask probability (calibrated), routing ----------

def _temp_scale_probs(probs: np.ndarray, temp: float) -> np.ndarray:
    if abs(float(temp) - 1.0) < 1e-6:
        return probs
    p = np.power(probs, 1.0 / float(temp))
    s = p.sum()
    return p / s if s > 0 else probs

def infer_mask_prob(img_bgr_face, model, mask_class_index=1, temp=1.0):
    """
    Returns (p_masked, probs_list). Assumes model outputs a 2-class probability vector.
    No re-softmax; optional temperature applied in probability space.
    """
    shape = model.inputs[0].shape
    h, w = int(shape[1]), int(shape[2])
    face_rgb = cv2.cvtColor(img_bgr_face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (w, h), interpolation=cv2.INTER_AREA)
    x = face_resized.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    y = np.asarray(model.predict(x, verbose=0))

    # Preferred path: 2-class probs
    if y.ndim == 2 and y.shape[1] >= 2:
        probs = y[0].astype(np.float64)
        probs = np.clip(probs, 0.0, 1.0)
        s = probs.sum()
        if s > 0 and not (0.99 <= s <= 1.01):
            probs = probs / s
        probs = _temp_scale_probs(probs, temp)
        idx = int(np.clip(mask_class_index, 0, probs.size - 1))
        return float(probs[idx]), probs.tolist()

    # Fallback: treat scalar as sigmoid for masked
    val = float(y.ravel()[0])
    p_masked = 1.0 / (1.0 + math.exp(-val))
    p_masked = float(np.clip(p_masked, 1e-8, 1.0 - 1e-8))
    return p_masked, [p_masked, 1.0 - p_masked]

def route_branch_by_mask_prob(p_mask: float, th_low: float, th_high: float) -> str:
    """
    Returns 'landmarks' if sure of mask, 'encodings' if sure of no mask, 'both' if uncertain.
    """
    if p_mask >= th_high:
        return "landmarks"
    if p_mask <= th_low:
        return "encodings"
    return "both"

# ---------- YuNet integration (and Haar fallback) ----------

def _backend_id(name: str) -> int:
    return {
        "default": cv2.dnn.DNN_BACKEND_DEFAULT,
        "opencv":  cv2.dnn.DNN_BACKEND_OPENCV,
        "cuda":    cv2.dnn.DNN_BACKEND_CUDA,
    }[name]

def _target_id(name: str) -> int:
    return {
        "cpu":       cv2.dnn.DNN_TARGET_CPU,
        "cuda":      cv2.dnn.DNN_TARGET_CUDA,
        "cuda_fp16": cv2.dnn.DNN_TARGET_CUDA_FP16,
    }[name]

def load_yunet(model_path: str, input_size: Tuple[int, int], backend: str, target: str):
    # OpenCV >= 4.7: FaceDetectorYN.create; some builds expose FaceDetectorYN_create
    if hasattr(cv2, "FaceDetectorYN"):
        det = cv2.FaceDetectorYN.create(
            model=model_path, config="", input_size=input_size,
            score_threshold=0.6, nms_threshold=0.3, top_k=5000,
            backend_id=_backend_id(backend), target_id=_target_id(target)
        )
    else:
        det = cv2.FaceDetectorYN_create(
            model_path, "", input_size, 0.6, 0.3, 5000,
            _backend_id(backend), _target_id(target)
        )
    if det is None:
        raise SystemExit(f"[ERR] Failed to load YuNet model: {model_path}")
    return det

def detect_faces_yunet(detector, frame_bgr: np.ndarray, min_face: int) -> List[Tuple[int, int, int, int]]:
    """
    Normalize FaceDetectorYN.detect outputs across OpenCV builds:
    returns list of (x,y,w,h) boxes sorted by area desc.
    """
    h, w = frame_bgr.shape[:2]
    detector.setInputSize((w, h))  # MUST set per-frame size

    out = detector.detect(frame_bgr)

    faces_arr = None
    if out is None:
        faces_arr = None
    elif isinstance(out, tuple):
        for item in out:
            if isinstance(item, np.ndarray):
                faces_arr = item
                break
        if faces_arr is None and len(out) == 2 and isinstance(out[0], (int, np.integer)) and isinstance(out[1], np.ndarray):
            faces_arr = out[1]
    elif isinstance(out, np.ndarray):
        faces_arr = out
    else:
        faces_arr = None

    if faces_arr is None or faces_arr.size == 0:
        return []

    if faces_arr.ndim == 1:
        faces_arr = faces_arr.reshape(1, -1)

    rects: List[Tuple[int, int, int, int]] = []
    for f in faces_arr:
        x, y, ww, hh = map(int, f[:4])
        if ww >= min_face and hh >= min_face:
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            ww = max(1, min(ww, w - x))
            hh = max(1, min(hh, h - y))
            rects.append((x, y, ww, hh))

    rects.sort(key=lambda r: r[2] * r[3], reverse=True)
    return rects

class FaceDetector:
    def __init__(self, kind: str, args):
        self.kind = kind
        self.min_face = args.min_face_size
        if kind == "haar":
            if not args.cascade:
                raise SystemExit("[ERR] --cascade is required when --detector haar")
            self.cascade = cv2.CascadeClassifier(args.cascade)
            if self.cascade.empty():
                raise SystemExit(f"[ERR] Failed to load Haar cascade: {args.cascade}")
            self.yunet = None
        else:
            if not args.yunet_model:
                raise SystemExit("[ERR] --yunet-model is required when --detector yunet")
            self.yunet = load_yunet(args.yunet_model, (320, 320), args.dnn_backend, args.dnn_target)
            self.cascade = None

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.kind == "haar":
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(self.min_face, self.min_face))
            faces_list = faces.tolist() if hasattr(faces, "tolist") else list(faces)
            faces_list.sort(key=lambda r: int(r[2]) * int(r[3]), reverse=True)
            return faces_list
        else:
            return detect_faces_yunet(self.yunet, frame_bgr, self.min_face)

# ---------- Prompt helpers ----------

def prompt_person_info() -> Tuple[str, str, int]:
    first = input("Enter first_name: ").strip()
    last = input("Enter last_name: ").strip()
    while True:
        age_s = input("Enter age (integer): ").strip()
        try:
            return first, last, int(age_s)
        except ValueError:
            print("Age must be an integer.")

def prompt_bios_mapping_json() -> Dict[str, Dict[str, Any]]:
    while True:
        p = input("Path to bios JSON (filename -> {first_name,last_name,age}): ").strip().strip('"').strip("'")
        path = Path(p)
        if not path.exists() or not path.is_file():
            print("File not found. Try again.")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                print("Expected object mapping filename -> bio dict.")
                continue
            norm = {}
            for k, v in data.items():
                if not isinstance(v, dict) or not {"first_name", "last_name", "age"} <= set(v.keys()):
                    raise ValueError(f"Bad entry for {k}")
                norm[Path(k).name] = {
                    "first_name": str(v["first_name"]).strip(),
                    "last_name": str(v["last_name"]).strip(),
                    "age": int(v["age"])
                }
            return norm
        except Exception as e:
            print(f"Invalid JSON: {e}")

# ---------- Logs (legacy + audit-grade runtime log) ----------

def prepare_logs(log_dir: Path) -> Tuple[Path, Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    faces_csv = log_dir / "faces_log.csv"
    comps_csv = log_dir / "comparisons_log.csv"
    runtime_csv = log_dir / "runtime_matches.csv"
    if not faces_csv.exists():
        with open(faces_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "source", "frame_idx", "face_idx", "label_masked", "branch", "match_table",
                "best_rowid", "best_distance", "t_detect_ms", "t_mask_pred_ms", "t_landmarks_ms",
                "t_encoding_ms", "t_match_total_ms", "t_total_face_ms"
            ])
    if not comps_csv.exists():
        with open(comps_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source", "frame_idx", "face_idx", "table", "rowid", "distance", "t_compare_ms"])
    if not runtime_csv.exists():
        with open(runtime_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp_iso", "source", "frame_idx", "face_idx", "branch",
                "distance", "tau", "decision", "eval_run_id", "gallery_size", "expected_fpir", "best_rowid",
                "mask_prob", "mask_th_low", "mask_th_high", "mask_route"
            ])
    return faces_csv, comps_csv, runtime_csv

def append_face_log(faces_csv: Path, rec: FaceTiming) -> None:
    with open(faces_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            rec.source, rec.frame_idx, rec.face_idx, rec.label_masked, rec.branch, rec.match_table,
            rec.best_rowid, rec.best_distance, rec.t_detect_ms, rec.t_mask_pred_ms, rec.t_landmarks_ms,
            rec.t_encoding_ms, rec.t_match_total_ms, rec.t_total_face_ms
        ])

def append_comp_logs(comps_csv: Path, rows: Iterable[ComparisonTiming]) -> None:
    with open(comps_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([r.source, r.frame_idx, r.face_idx, r.table, r.rowid, r.distance, r.t_compare_ms])

def append_runtime_log(runtime_csv: Path,
                       source: str, frame_idx: Optional[int], face_idx: int,
                       branch: str, distance: Optional[float], tau: Optional[float],
                       decision: Optional[bool], eval_run_id: int,
                       gallery_size: int, expected_fpir_val: Optional[float],
                       best_rowid: Optional[int],
                       mask_prob: Optional[float], th_low: float, th_high: float, mask_route: str) -> None:
    with open(runtime_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.utcnow().isoformat(timespec="seconds") + "Z",
            source, frame_idx, face_idx, branch,
            f"{distance:.9f}" if distance is not None else "",
            f"{tau:.9f}" if tau is not None else "",
            str(bool(decision)) if decision is not None else "",
            eval_run_id,
            gallery_size,
            f"{expected_fpir_val:.9g}" if expected_fpir_val is not None else "",
            best_rowid,
            f"{mask_prob:.6f}" if mask_prob is not None else "",
            th_low, th_high, mask_route
        ])

def log_all_lm_comparisons(probe23n: np.ndarray, lm_rows, src, frame_idx, face_idx, table, comps_csv):
    rows = []
    for rid, g in lm_rows:
        t0 = time.perf_counter()
        d = rms_landmarks(probe23n, g)
        t_ms = (time.perf_counter() - t0) * 1000.0
        rows.append(ComparisonTiming(src, frame_idx, face_idx, table, rid, d, t_ms))
    append_comp_logs(comps_csv, rows)

def log_all_enc_comparisons(probe_enc: np.ndarray, enc_rows, metric: str, src, frame_idx, face_idx, table, comps_csv):
    rows = []
    for rid, g in enc_rows:
        t0 = time.perf_counter()
        d = cosine_enc(probe_enc, g) if metric == "cos" else euclidean_enc(probe_enc, g)
        t_ms = (time.perf_counter() - t0) * 1000.0
        rows.append(ComparisonTiming(src, frame_idx, face_idx, table, rid, d, t_ms))
    append_comp_logs(comps_csv, rows)

# ---------- Argmin matchers ----------

def argmin_landmarks(probe23n: np.ndarray, lm_rows: List[Tuple[int, np.ndarray]]) -> Tuple[Optional[int], Optional[float]]:
    if not lm_rows:
        return None, None
    best_r, best_d = None, float("inf")
    for rid, g in lm_rows:
        d = rms_landmarks(probe23n, g)
        if d < best_d:
            best_d = d
            best_r = rid
    return best_r, best_d

def argmin_encodings(enc: np.ndarray, enc_rows: List[Tuple[int, np.ndarray]], metric: str) -> Tuple[Optional[int], Optional[float]]:
    if not enc_rows:
        return None, None
    best_r, best_d = None, float("inf")
    if metric == "cos":
        for rid, g in enc_rows:
            d = cosine_enc(enc, g)
            if d < best_d:
                best_d = d
                best_r = rid
    else:
        for rid, g in enc_rows:
            d = euclidean_enc(enc, g)
            if d < best_d:
                best_d = d
                best_r = rid
    return best_r, best_d

# ---------- Modes ----------

def run_images_mode(args) -> None:
    images_dir = Path(args.images)
    imgs = list_images(images_dir)
    if not imgs:
        print("[WARN] No images found with valid extensions.")
        return

    # Detector
    detector = FaceDetector(args.detector, args)

    # Predictor + mask model (with warm-up)
    predictor = dlib.shape_predictor(args.shape_predictor)
    mask_model = load_model(args.mask_model)
    hh, ww = int(mask_model.inputs[0].shape[1]), int(mask_model.inputs[0].shape[2])
    _ = mask_model.predict(np.zeros((1, hh, ww, 3), dtype="float32"), verbose=0)

    # DB
    db_path = Path(args.db).expanduser()
    ensure_db_dir(db_path)
    conn = sqlite3.connect(str(db_path))

    # Thresholds + gallery
    thr = load_thresholds(conn, args.eval_run_id, args.tau_override)
    lm_rows, enc_rows = load_gallery_from_facial_data(conn)

    # Optional saving (legacy tables)
    bios_map = None
    if args.save_matches:
        bios_map = prompt_bios_mapping_json()
        ensure_table(conn, args.landmarks_table, "landmarks")
        ensure_table(conn, args.encodings_table, "encodings")

    faces_csv, comps_csv, runtime_csv = prepare_logs(Path(args.log_dir))
    print(f"[INFO] Found {len(imgs)} images. Starting processing...")

    for img_idx, img_path in enumerate(imgs):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Unreadable: {img_path.name}")
            continue

        t0_face = time.perf_counter()
        t0 = time.perf_counter()
        faces = detector.detect(img)
        t_detect_ms = (time.perf_counter() - t0) * 1000.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if not faces:
            rec = FaceTiming(source=img_path.name, frame_idx=None, face_idx=0, label_masked=-1,
                             t_detect_ms=t_detect_ms, t_mask_pred_ms=0.0, t_landmarks_ms=0.0,
                             t_encoding_ms=0.0, t_match_total_ms=0.0,
                             t_total_face_ms=(time.perf_counter() - t0_face) * 1000.0, branch="none",
                             match_table=None, best_rowid=None, best_distance=None)
            append_face_log(faces_csv, rec)
            continue

        for face_idx, rect in enumerate(faces, start=1):
            x, y, w, h = rect
            face_bgr = img[y:y + h, x:x + w].copy()

            # P(masked)
            t0 = time.perf_counter()
            p_mask, _probs = infer_mask_prob(
                face_bgr, mask_model,
                mask_class_index=args.mask_class_index,
                temp=args.mask_temp
            )
            t_mask_ms = (time.perf_counter() - t0) * 1000.0
            route = route_branch_by_mask_prob(p_mask, args.mask_th_low, args.mask_th_high)
            label = 1 if p_mask >= 0.5 else 0

            # Prepare candidates
            best_rowid = None
            best_dist = None
            t_lm_ms = 0.0
            t_enc_ms = 0.0
            t_match_total = 0.0
            branch_decided = None

            # Landmarks candidate if needed
            lm_candidate = (None, None)
            if route in ("landmarks", "both"):
                t0 = time.perf_counter()
                pts68 = predict_68_pts(gray, rect, predictor)
                pts23 = extract_23(pts68)
                pts23n = normalize_23_from_68(pts68, pts23)
                if args.log_comparisons and pts23n is not None and lm_rows:
                    log_all_lm_comparisons(pts23n, lm_rows, img_path.name, None, face_idx, args.landmarks_table, comps_csv)
                t_lm_ms = (time.perf_counter() - t0) * 1000.0
                if pts23n is not None and lm_rows:
                    lm_candidate = argmin_landmarks(pts23n, lm_rows)

                if args.save_matches and bios_map and pts23n is not None:
                    bio = bios_map.get(img_path.name)
                    if bio:
                        insert_landmarks(conn, args.landmarks_table, bio["first_name"], bio["last_name"], bio["age"], pts23n)

            # Encodings candidate if needed
            enc_candidate = (None, None)
            if route in ("encodings", "both"):
                t0 = time.perf_counter()
                enc = encode_face_bgr(img, rect)
                if args.log_comparisons and enc is not None and enc_rows:
                    log_all_enc_comparisons(enc, enc_rows, args.distance, img_path.name, None, face_idx, args.encodings_table, comps_csv)
                t_enc_ms = (time.perf_counter() - t0) * 1000.0
                if enc is not None and enc_rows:
                    enc_candidate = argmin_encodings(enc, enc_rows, args.distance)

                if args.save_matches and bios_map and enc is not None:
                    bio = bios_map.get(img_path.name)
                    if bio:
                        insert_encoding(conn, args.encodings_table, bio["first_name"], bio["last_name"], bio["age"], enc)

            # Decide final branch using τ
            tau_lm = thr.get("landmarks", {}).get("tau")
            tau_en = thr.get("encodings", {}).get("tau")
            fmr_lm = thr.get("landmarks", {}).get("fmr_achieved")
            fmr_en = thr.get("encodings", {}).get("fmr_achieved")

            def accept_lm(d): return (d is not None and tau_lm is not None and d <= float(tau_lm))
            def accept_en(d): return (d is not None and tau_en is not None and d <= float(tau_en))

            if route == "landmarks":
                best_rowid, best_dist = lm_candidate
                branch_decided = "landmarks"
            elif route == "encodings":
                best_rowid, best_dist = enc_candidate
                branch_decided = "encodings"
            else:
                if args.mask_try_both:
                    ok_lm = accept_lm(lm_candidate[1])
                    ok_en = accept_en(enc_candidate[1])
                    if ok_lm and not ok_en:
                        best_rowid, best_dist = lm_candidate
                        branch_decided = "landmarks"
                    elif ok_en and not ok_lm:
                        best_rowid, best_dist = enc_candidate
                        branch_decided = "encodings"
                    elif ok_lm and ok_en:
                        nl = (float(tau_lm) - lm_candidate[1]) / max(float(tau_lm), 1e-9) if tau_lm is not None and lm_candidate[1] is not None else -1e9
                        ne = (float(tau_en) - enc_candidate[1]) / max(float(tau_en), 1e-9) if tau_en is not None and enc_candidate[1] is not None else -1e9
                        if nl >= ne:
                            best_rowid, best_dist = lm_candidate
                            branch_decided = "landmarks"
                        else:
                            best_rowid, best_dist = enc_candidate
                            branch_decided = "encodings"
                    else:
                        a = lm_candidate[1] if lm_candidate[1] is not None else float("inf")
                        b = enc_candidate[1] if enc_candidate[1] is not None else float("inf")
                        if a <= b:
                            best_rowid, best_dist = lm_candidate
                            branch_decided = "landmarks"
                        else:
                            best_rowid, best_dist = enc_candidate
                            branch_decided = "encodings"
                else:
                    if p_mask >= 0.5:
                        best_rowid, best_dist = lm_candidate
                        branch_decided = "landmarks"
                    else:
                        best_rowid, best_dist = enc_candidate
                        branch_decided = "encodings"

            # Decision + audit log
            tau = thr.get(branch_decided, {}).get("tau")
            fmr_ach = thr.get(branch_decided, {}).get("fmr_achieved")
            N = len(lm_rows) if branch_decided == "landmarks" else len(enc_rows)
            decision = (best_dist is not None and tau is not None and best_dist <= float(tau))
            append_runtime_log(runtime_csv, img_path.name, None, face_idx, branch_decided,
                               best_dist, float(tau) if tau is not None else None, decision,
                               args.eval_run_id, N, expected_fpir(fmr_ach, N), best_rowid,
                               mask_prob=p_mask, th_low=args.mask_th_low, th_high=args.mask_th_high, mask_route=route)

            # Legacy per-face timing log
            t_total_face_ms = (time.perf_counter() - t0_face) * 1000.0
            rec = FaceTiming(
                source=img_path.name, frame_idx=None, face_idx=face_idx, label_masked=label,
                t_detect_ms=t_detect_ms, t_mask_pred_ms=t_mask_ms, t_landmarks_ms=t_lm_ms,
                t_encoding_ms=t_enc_ms, t_match_total_ms=t_match_total,
                t_total_face_ms=t_total_face_ms, branch=branch_decided,
                match_table=(args.landmarks_table if branch_decided == "landmarks" else args.encodings_table),
                best_rowid=best_rowid, best_distance=best_dist
            )
            append_face_log(faces_csv, rec)

    conn.close()
    print("[DONE] Folder mode complete.")

def run_camera_mode(args) -> None:
    # Detector
    detector = FaceDetector(args.detector, args)

    # Predictor + mask model (with warm-up)
    predictor = dlib.shape_predictor(args.shape_predictor)
    mask_model = load_model(args.mask_model)
    hh, ww = int(mask_model.inputs[0].shape[1]), int(mask_model.inputs[0].shape[2])
    _ = mask_model.predict(np.zeros((1, hh, ww, 3), dtype="float32"), verbose=0)

    db_path = Path(args.db).expanduser()
    ensure_db_dir(db_path)
    conn = sqlite3.connect(str(db_path))

    thr = load_thresholds(conn, args.eval_run_id, args.tau_override)
    lm_rows, enc_rows = load_gallery_from_facial_data(conn)

    if args.save_matches:
        ensure_table(conn, args.landmarks_table, "landmarks")
        ensure_table(conn, args.encodings_table, "encodings")

    faces_csv, comps_csv, runtime_csv = prepare_logs(Path(args.log_dir))

    cam_index = args.camera_index
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("[ERR] Cannot open camera.")
        conn.close()
        return

    paused = True
    frame_idx = 0
    staged_branch = None
    staged_feature = None
    staged_pts68 = None

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("[ERR] Camera read failed.")
                    break
                frame_idx += 1
            else:
                ok, frame = cap.read()
                if not ok:
                    print("[ERR] Camera read failed.")
                    break

            display = frame.copy()

            # Detect faces
            t0_face = time.perf_counter()
            t0 = time.perf_counter()
            faces = detector.detect(frame)
            t_detect_ms = (time.perf_counter() - t0) * 1000.0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if faces:
                for face_idx, rect in enumerate(faces, start=1):
                    x, y, w, h = rect
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    face_bgr = frame[y:y + h, x:x + w].copy()
                    t0 = time.perf_counter()
                    p_mask, _probs = infer_mask_prob(
                        face_bgr, mask_model,
                        mask_class_index=args.mask_class_index,
                        temp=args.mask_temp
                    )
                    t_mask_ms = (time.perf_counter() - t0) * 1000.0
                    route = route_branch_by_mask_prob(p_mask, args.mask_th_low, args.mask_th_high)
                    label = 1 if p_mask >= 0.5 else 0

                    best_rowid = None
                    best_dist = None
                    branch_decided = ""
                    t_lm_ms = 0.0
                    t_enc_ms = 0.0
                    t_match_total = 0.0

                    lm_candidate = (None, None)
                    if route in ("landmarks", "both"):
                        t0 = time.perf_counter()
                        pts68 = predict_68_pts(gray, rect, predictor)
                        pts23 = extract_23(pts68)
                        pts23n = normalize_23_from_68(pts68, pts23)
                        if args.log_comparisons and pts23n is not None and lm_rows:
                            log_all_lm_comparisons(pts23n, lm_rows, "camera", frame_idx, face_idx, args.landmarks_table, comps_csv)
                        t_lm_ms = (time.perf_counter() - t0) * 1000.0
                        if pts23n is not None and lm_rows:
                            lm_candidate = argmin_landmarks(pts23n, lm_rows)
                        if face_idx == 1:
                            staged_branch = "landmarks"
                            staged_feature = pts23n
                            staged_pts68 = pts68

                    enc_candidate = (None, None)
                    if route in ("encodings", "both"):
                        t0 = time.perf_counter()
                        enc = encode_face_bgr(frame, rect)
                        if args.log_comparisons and enc is not None and enc_rows:
                            log_all_enc_comparisons(enc, enc_rows, args.distance, "camera", frame_idx, face_idx, args.encodings_table, comps_csv)
                        t_enc_ms = (time.perf_counter() - t0) * 1000.0
                        if enc is not None and enc_rows:
                            enc_candidate = argmin_encodings(enc, enc_rows, args.distance)
                        if face_idx == 1:
                            staged_branch = "encodings"
                            staged_feature = enc
                            staged_pts68 = None

                    tau_lm = thr.get("landmarks", {}).get("tau")
                    tau_en = thr.get("encodings", {}).get("tau")
                    fmr_lm = thr.get("landmarks", {}).get("fmr_achieved")
                    fmr_en = thr.get("encodings", {}).get("fmr_achieved")

                    def accept_lm(d): return (d is not None and tau_lm is not None and d <= float(tau_lm))
                    def accept_en(d): return (d is not None and tau_en is not None and d <= float(tau_en))

                    if route == "landmarks":
                        best_rowid, best_dist = lm_candidate
                        branch_decided = "landmarks"
                    elif route == "encodings":
                        best_rowid, best_dist = enc_candidate
                        branch_decided = "encodings"
                    else:
                        if args.mask_try_both:
                            ok_lm = accept_lm(lm_candidate[1])
                            ok_en = accept_en(enc_candidate[1])
                            if ok_lm and not ok_en:
                                best_rowid, best_dist = lm_candidate
                                branch_decided = "landmarks"
                            elif ok_en and not ok_lm:
                                best_rowid, best_dist = enc_candidate
                                branch_decided = "encodings"
                            elif ok_lm and ok_en:
                                nl = (float(tau_lm) - lm_candidate[1]) / max(float(tau_lm), 1e-9) if tau_lm is not None and lm_candidate[1] is not None else -1e9
                                ne = (float(tau_en) - enc_candidate[1]) / max(float(tau_en), 1e-9) if tau_en is not None and enc_candidate[1] is not None else -1e9
                                if nl >= ne:
                                    best_rowid, best_dist = lm_candidate
                                    branch_decided = "landmarks"
                                else:
                                    best_rowid, best_dist = enc_candidate
                                    branch_decided = "encodings"
                            else:
                                a = lm_candidate[1] if lm_candidate[1] is not None else float("inf")
                                b = enc_candidate[1] if enc_candidate[1] is not None else float("inf")
                                if a <= b:
                                    best_rowid, best_dist = lm_candidate
                                    branch_decided = "landmarks"
                                else:
                                    best_rowid, best_dist = enc_candidate
                                    branch_decided = "encodings"
                        else:
                            if p_mask >= 0.5:
                                best_rowid, best_dist = lm_candidate
                                branch_decided = "landmarks"
                            else:
                                best_rowid, best_dist = enc_candidate
                                branch_decided = "encodings"

                    tau = thr.get(branch_decided, {}).get("tau")
                    fmr_ach = thr.get(branch_decided, {}).get("fmr_achieved")
                    N = len(lm_rows) if branch_decided == "landmarks" else len(enc_rows)
                    decision = (best_dist is not None and tau is not None and best_dist <= float(tau))
                    append_runtime_log(runtime_csv, "camera", frame_idx, face_idx, branch_decided,
                                       best_dist, float(tau) if tau is not None else None, decision,
                                       args.eval_run_id, N, expected_fpir(fmr_ach, N), best_rowid,
                                       mask_prob=p_mask, th_low=args.mask_th_low, th_high=args.mask_th_high, mask_route=route)

                    rec = FaceTiming(
                        source="camera", frame_idx=frame_idx, face_idx=face_idx, label_masked=label,
                        t_detect_ms=t_detect_ms, t_mask_pred_ms=t_mask_ms, t_landmarks_ms=t_lm_ms,
                        t_encoding_ms=t_enc_ms, t_match_total_ms=t_match_total,
                        t_total_face_ms=(time.perf_counter() - t0_face) * 1000.0,
                        branch=branch_decided, match_table=(args.landmarks_table if branch_decided == "landmarks" else args.encodings_table),
                        best_rowid=best_rowid, best_distance=best_dist
                    )
                    append_face_log(faces_csv, rec)

            else:
                staged_branch = None
                staged_feature = None
                staged_pts68 = None

            # HUD
            cv2.putText(display, "R:start  P:pause  C:capture  S:save  Q:quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if paused:
                cv2.putText(display, "PAUSED", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            if staged_branch:
                cv2.putText(display, f"STAGED: {staged_branch}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

            cv2.imshow("Pipeline - Camera", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q
                break
            elif key in (ord('r'), ord('R')):
                paused = False
            elif key in (ord('p'), ord('P')):
                paused = True
            elif key in (ord('c'), ord('C')):
                paused = False
            elif key in (ord('s'), ord('S')):
                if staged_feature is None or staged_branch is None:
                    print("[WARN] Nothing staged to save.")
                else:
                    first, last, age = prompt_person_info()
                    if staged_branch == "landmarks":
                        if staged_feature is not None:
                            ensure_table(conn, args.landmarks_table, "landmarks")
                            insert_landmarks(conn, args.landmarks_table, first, last, age, staged_feature)
                            print("[OK] Landmarks saved.")
                    else:
                        if staged_feature is not None:
                            ensure_table(conn, args.encodings_table, "encodings")
                            insert_encoding(conn, args.encodings_table, first, last, age, staged_feature)
                            print("[OK] Encoding saved.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        conn.close()
        print("[DONE] Camera mode closed.")

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Face pipeline with evaluated thresholds: mask -> landmarks(23, normalized)/no-mask -> encodings(128), YuNet/Haar detection.")
    # Detector options
    ap.add_argument("--detector", choices=["haar", "yunet"], default="yunet", help="Face detector to use (default: yunet)")
    ap.add_argument("--cascade", help="Path to Haar cascade XML (haarcascade_frontalface_default.xml) [required if --detector haar]")
    ap.add_argument("--yunet-model", type=str, default="models/face_detection_yunet_2023mar.onnx", help="Path to YuNet ONNX model")
    ap.add_argument("--dnn-backend", choices=["default", "opencv", "cuda"], default="default", help="DNN backend for YuNet")
    ap.add_argument("--dnn-target", choices=["cpu", "cuda", "cuda_fp16"], default="cpu", help="DNN target for YuNet")

    # Landmark predictor + mask model
    ap.add_argument("--shape-predictor", required=True, help="Path to dlib shape_predictor_68_face_landmarks.dat")
    ap.add_argument("--mask-model", required=True, help="Path to Keras .h5 mask classifier")

    # DB & thresholds
    ap.add_argument("--db", default="Eagle_Eye_Detection_Pipeline.db", help="SQLite DB path")
    ap.add_argument("--eval-run-id", type=int, required=True, help="Use thresholds from eval_metrics(run_id)")
    ap.add_argument("--tau-override", type=str, help='JSON like {"landmarks":0.12,"encodings":0.48} to override τ')

    # Mask routing configuration
    ap.add_argument("--mask-class-index", type=int, default=1, help="Index of 'Masked' class in model output")
    ap.add_argument("--mask-th-high", dest="mask_th_high", type=float, default=0.75, help="Route to landmarks when P(mask) ≥ this")
    ap.add_argument("--mask-th-low", dest="mask_th_low", type=float, default=0.25, help="Route to encodings when P(mask) ≤ this")
    ap.add_argument("--mask-try-both", action="store_true", help="In uncertainty band, evaluate BOTH branches and decide via τ")
    ap.add_argument("--mask-temp", type=float, default=1.0, help="Temperature for mask probabilities (1=no change). T>1 flattens; T<1 sharpens.")

    # Legacy per-branch tables (optional saving)
    ap.add_argument("--landmarks-table", default="unmasked_faces", help="(Optional save) Table for 23 landmarks")
    ap.add_argument("--encodings-table", default="masked_faces", help="(Optional save) Table for 128-D encodings")

    # Source modes
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", type=str, help="Directory of images to process")
    src.add_argument("--camera", action="store_true", help="Use webcam")

    # Matching options
    ap.add_argument("--mode", choices=["1to1", "1toN"], default="1toN", help="Matching mode (default: 1toN)")
    ap.add_argument("--target-rowid", type=int, help="(1:1) rowid to compare against (legacy tables)")
    ap.add_argument("--target-table", choices=["landmarks", "encodings"], help="(1:1) table of target rowid")
    ap.add_argument("--probe-landmarks", type=str, help="(1:1) path to 23x2 JSON probe")
    ap.add_argument("--probe-encoding", type=str, help="(1:1) path to 128 JSON probe")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1), help="(Unused in current argmin path; reserved)")
    ap.add_argument("--distance", choices=["enc", "cos"], default="enc", help="Distance for encodings: enc(Euclidean) or cos(1-cosine).")
    ap.add_argument("--min-face-size", type=int, default=80, help="Minimum face size (pixels) for detection")
    ap.add_argument("--save-matches", action="store_true", help="(Folder/Camera) Allow saving with S (camera) or per-image with bios JSON (folder)")
    ap.add_argument("--camera-index", type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument("--log-dir", type=str, default="logs", help="Directory to write CSV logs")
    ap.add_argument("--log-comparisons", action="store_true",
                    help="Log every gallery comparison to comparisons_log.csv (slower)")

    args = ap.parse_args()

    # Conditional arg checks
    if args.detector == "haar" and not args.cascade:
        print("[ERR] --cascade is required when --detector haar")
        sys.exit(1)
    if args.detector == "yunet" and not args.yunet_model:
        print("[ERR] --yunet-model is required when --detector yunet")
        sys.exit(1)

    # Validate 1:1 inputs
    if args.mode == "1to1":
        has_rowid = args.target_rowid is not None and args.target_table is not None
        has_probe = (args.probe_landmarks is not None) ^ (args.probe_encoding is not None)
        if not (has_rowid or has_probe):
            print("[ERR] For --mode 1to1, provide either --target-rowid + --target-table OR one of --probe-landmarks/--probe-encoding.")
            sys.exit(1)

    if args.images:
        images_dir = Path(args.images)
        if not images_dir.exists() or not images_dir.is_dir():
            print("[ERR] --images must be a valid directory.")
            sys.exit(1)
        run_images_mode(args)
    else:
        run_camera_mode(args)

if __name__ == "__main__":
    main()
