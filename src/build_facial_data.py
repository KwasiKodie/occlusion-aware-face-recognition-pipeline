#!/usr/bin/env python3
"""
build_facial_data.py

Create and populate `Facial_Data` from images listed in `Bio_Data`:

- Reads:   Bio_Data(person_id TEXT, first_name TEXT, last_name TEXT, age INT, file_path TEXT)
- Writes:  Facial_Data(person_id TEXT, face_landmarks TEXT, face_encodings TEXT)
           Where:
             - face_landmarks = JSON of 23x2 normalized points (translation & scale normalized)
             - face_encodings = JSON of 128-D face embedding

Pipeline per image:
  1) Load image from Bio_Data.file_path (skip if unreadable / not a valid image)
  2) Detect all faces via Haar on grayscale; pick the LARGEST bbox
  3) Landmarks: dlib 68 → select subset [17..21, 22..26, 36..41, 42..47, 27] (1-based iBUG)
     - Normalize points: subtract eye-midpoint center, divide by inter-ocular distance (scale-invariant)
  4) Encodings: compute 128-D embedding with face_recognition (using chosen bbox)
  5) Insert row into Facial_Data with (person_id, json(23x2), json(128))

Notes:
- Requires: opencv-python, dlib, face_recognition, numpy
- Enforces foreign key on person_id referencing Bio_Data(person_id) (PRAGMA foreign_keys=ON)
- Commits in batches for speed and prints a clear summary at the end

Usage:
  python build_facial_data.py \
    --db Eagle_Eye_Detection_Pipeline.db \
    --cascade models/haarcascade_frontalface_default.xml \
    --shape-predictor models/shape_predictor_68_face_landmarks.dat
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import dlib
import numpy as np
import face_recognition

# ---------- Configuration ----------

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}

# 23-point subset: iBUG 68 indices (1-based in spec) → 0-based for arrays
SUBSET_1B = list(range(17, 22)) + list(range(22, 27)) + list(range(36, 42)) + list(range(42, 48)) + [27]
SUBSET_0B = [i - 1 for i in SUBSET_1B]

# Eye indices (0-based in the full 68-set) for inter-ocular normalization
LEYE_68 = list(range(36, 42))  # 36-41
REYE_68 = list(range(42, 48))  # 42-47


# ---------- DB Helpers ----------

def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    # Foreign keys ON
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_tables(conn: sqlite3.Connection) -> None:
    with conn:
        # Must have Bio_Data
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Bio_Data';")
        if cur.fetchone() is None:
            raise RuntimeError("Bio_Data table not found in the database.")

        # Canonical parent with UNIQUE key
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Persons (
                person_id TEXT PRIMARY KEY
            );
        """)

        # If Facial_Data exists, see who it references
        parent = None
        cur = conn.execute("SELECT \"table\" FROM pragma_foreign_key_list('Facial_Data') LIMIT 1;")
        row = cur.fetchone()
        parent = row[0] if row else None

        if parent and parent != "Persons":
            # Migrate old Facial_Data -> new FK to Persons
            conn.execute("ALTER TABLE Facial_Data RENAME TO Facial_Data_old;")
            conn.execute("""
                CREATE TABLE Facial_Data (
                    person_id      TEXT NOT NULL,
                    face_landmarks TEXT NOT NULL,
                    face_encodings TEXT NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES Persons(person_id)
                );
            """)
            # Copy rows that have a valid parent (safe if Persons already synced)
            conn.execute("""
                INSERT INTO Facial_Data (person_id, face_landmarks, face_encodings)
                SELECT fd.person_id, fd.face_landmarks, fd.face_encodings
                FROM Facial_Data_old fd
                WHERE fd.person_id IN (SELECT person_id FROM Persons);
            """)
            conn.execute("DROP TABLE Facial_Data_old;")
        else:
            # Fresh create if missing
            conn.execute("""
                CREATE TABLE IF NOT EXISTS Facial_Data (
                    person_id      TEXT NOT NULL,
                    face_landmarks TEXT NOT NULL,
                    face_encodings TEXT NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES Persons(person_id)
                );
            """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fd_person ON Facial_Data(person_id);")


def sync_persons(conn: sqlite3.Connection) -> None:
    # Ensure all person_ids from Bio_Data exist in Persons BEFORE inserting into Facial_Data
    with conn:
        conn.execute("""
            INSERT OR IGNORE INTO Persons(person_id)
            SELECT DISTINCT person_id FROM Bio_Data;
        """)


def stream_bio_rows(conn: sqlite3.Connection) -> List[Tuple[str, str]]:
    """
    Returns a list of (person_id, file_path) from Bio_Data.
    We only read the columns we need.
    """
    rows = conn.execute(
        "SELECT person_id, file_path FROM Bio_Data ORDER BY person_id, file_path;"
    ).fetchall()
    return [(str(pid), str(fp)) for (pid, fp) in rows]


# ---------- Vision Helpers ----------

def detect_faces(gray: np.ndarray, cascade: cv2.CascadeClassifier, min_face: int) -> List[Tuple[int, int, int, int]]:
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face))
    # Sort by area (desc), return list of (x,y,w,h)
    faces_list = faces.tolist() if hasattr(faces, "tolist") else list(faces)
    faces_list.sort(key=lambda r: int(r[2]) * int(r[3]), reverse=True)
    return faces_list

def predict_68(gray: np.ndarray, rect: Tuple[int, int, int, int], predictor: dlib.shape_predictor) -> np.ndarray:
    x, y, w, h = rect
    drect = dlib.rectangle(left=int(x), top=int(y), right=int(x + w), bottom=int(y + h))
    shape = predictor(gray, drect)
    pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)
    return pts

def select_23(pts68: np.ndarray) -> np.ndarray:
    return pts68[SUBSET_0B, :]  # shape (23,2)

def eye_center(pts68: np.ndarray, idxs: List[int]) -> np.ndarray:
    return np.mean(pts68[idxs, :], axis=0)

def normalize_points_23(pts68: np.ndarray, pts23: np.ndarray, eps: float = 1e-6) -> Optional[np.ndarray]:
    """
    Normalize 23 points by translation & scale:
      - translation: subtract the midpoint between eye centers
      - scale: divide by inter-ocular distance (||centerR - centerL||)
    Returns 23x2 array of normalized floats, or None if scale near zero.
    """
    cL = eye_center(pts68, LEYE_68)
    cR = eye_center(pts68, REYE_68)
    inter_oc = float(np.linalg.norm(cR - cL))
    if inter_oc <= eps:
        return None
    center = (cL + cR) * 0.5
    normed = (pts23 - center) / inter_oc
    return normed.astype(np.float32)

def encode_face_128(img_bgr: np.ndarray, rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = rect
    top, right, bottom, left = y, x + w, y + h, x
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(
        img_rgb,
        known_face_locations=[(int(top), int(right), int(bottom), int(left))],
        num_jitters=1,
        model="small"
    )
    if not encs:
        return None
    return np.asarray(encs[0], dtype=np.float32)  # (128,)


# ---------- Main Worker ----------

def process_image(file_path: Path,
                  cascade: cv2.CascadeClassifier,
                  predictor: dlib.shape_predictor,
                  min_face_size: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Returns: (norm_23: (23,2) or None, enc_128: (128,) or None, reason_if_failed)
    """
    if not file_path.exists() or not file_path.is_file():
        return None, None, "missing_file"

    if file_path.suffix.lower() not in VALID_IMAGE_EXTS:
        return None, None, "invalid_ext"

    img = cv2.imread(str(file_path))
    if img is None:
        return None, None, "imread_failed"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, cascade, min_face=min_face_size)
    if not faces:
        return None, None, "no_face"

    # Largest face = faces[0]
    rect = faces[0]

    # Landmarks
    try:
        pts68 = predict_68(gray, rect, predictor)
    except Exception:
        return None, None, "landmark_failed"

    pts23 = select_23(pts68)
    norm23 = normalize_points_23(pts68, pts23)
    if norm23 is None:
        return None, None, "bad_scale"

    # Encodings
    enc128 = encode_face_128(img, rect)
    if enc128 is None:
        return None, None, "encoding_failed"

    return norm23, enc128, ""

def insert_row(conn: sqlite3.Connection, person_id: str,
               norm23: np.ndarray, enc128: np.ndarray) -> None:
    jlmk = json.dumps([[float(x), float(y)] for x, y in norm23.tolist()], separators=(",", ":"))
    jenc = json.dumps([float(x) for x in enc128.tolist()], separators=(",", ":"))
    with conn:
        conn.execute(
            "INSERT INTO Facial_Data (person_id, face_landmarks, face_encodings) VALUES (?,?,?)",
            (person_id, jlmk, jenc)
        )


# ---------- CLI & Orchestration ----------

def main():
    ap = argparse.ArgumentParser(description="Populate Facial_Data with normalized 23-point landmarks and 128-D encodings.")
    ap.add_argument("--db", required=True, help="SQLite DB path (e.g., Eagle_Eye_Detection_Pipeline.db)")
    ap.add_argument("--cascade", required=True, help="Path to Haar cascade XML (e.g., haarcascade_frontalface_default.xml)")
    ap.add_argument("--shape-predictor", required=True, help="Path to dlib shape_predictor_68_face_landmarks.dat")
    ap.add_argument("--batch", type=int, default=100, help="Commit every N inserts (default: 100)")
    ap.add_argument("--min-face-size", type=int, default=80, help="Minimum face size in pixels for detection (default: 80)")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser()
    cascade_path = Path(args.cascade)
    predictor_path = Path(args.shape_predictor)

    # Load models
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise SystemExit(f"[ERR] Failed to load Haar cascade: {cascade_path}")

    try:
        predictor = dlib.shape_predictor(str(predictor_path))
    except Exception as e:
        raise SystemExit(f"[ERR] Failed to load dlib shape predictor: {e}")

    # DB setup
    conn = open_db(db_path)
    try:
        ensure_tables(conn)
        sync_persons(conn)
        bio_rows = stream_bio_rows(conn)
        total = len(bio_rows)
        if total == 0:
            print("[INFO] No rows found in Bio_Data.")
            return

        # Stats
        processed = 0
        inserted = 0
        skipped_missing = 0
        skipped_ext = 0
        skipped_imread = 0
        skipped_no_face = 0
        skipped_landmark = 0
        skipped_bad_scale = 0
        skipped_encoding = 0

        # Transaction batching
        conn.execute("BEGIN")
        for idx, (person_id, fpath) in enumerate(bio_rows, start=1):
            processed += 1
            path = Path(fpath)

            norm23, enc128, reason = process_image(
                path, cascade=cascade, predictor=predictor, min_face_size=args.min_face_size
            )

            if norm23 is None or enc128 is None:
                if reason == "missing_file":  skipped_missing += 1
                elif reason == "invalid_ext": skipped_ext += 1
                elif reason == "imread_failed": skipped_imread += 1
                elif reason == "no_face": skipped_no_face += 1
                elif reason == "landmark_failed": skipped_landmark += 1
                elif reason == "bad_scale": skipped_bad_scale += 1
                elif reason == "encoding_failed": skipped_encoding += 1
                else: pass
            else:
                insert_row(conn, person_id, norm23, enc128)
                inserted += 1

            # Batch commit
            if (idx % args.batch) == 0:
                conn.commit()
                print(f"[PROGRESS] {idx}/{total} processed | inserted={inserted}")

        conn.commit()

        # Summary
        print("\n=== Facial_Data Build Summary ===")
        print(f"DB: {db_path}")
        print(f"Total Bio_Data rows:         {total}")
        print(f"Processed images:            {processed}")
        print(f"Inserted rows:               {inserted}")
        print("Skipped breakdown:")
        print(f"  missing file:              {skipped_missing}")
        print(f"  invalid extension:         {skipped_ext}")
        print(f"  imread failed:             {skipped_imread}")
        print(f"  no face detected:          {skipped_no_face}")
        print(f"  landmark failure:          {skipped_landmark}")
        print(f"  bad inter-ocular scale:    {skipped_bad_scale}")
        print(f"  encoding failure:          {skipped_encoding}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
