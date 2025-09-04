#!/usr/bin/env python3
"""
build_pairs_and_metrics.py

Purpose
-------
Create genuine/impostor pairs from Facial_Data, compute branch-specific distances
(23-pt landmarks RMS distance; 128-D encoding Euclidean), sweep thresholds to hit
a target FMR (or one derived from target FPIR and gallery size), and store:
  - eval_runs:    one row per evaluation run
  - eval_pairs:   per-pair distances (genuine & impostor) for each branch
  - eval_metrics: per-branch operating point (threshold, FNMR@FMR, counts)

Inputs (DB schema pre-existing)
-------------------------------
Facial_Data(person_id TEXT, face_landmarks TEXT(JSON 23x2), face_encodings TEXT(JSON len 128))

Key CLI Options
---------------
--impostor-mode {random|stratified|all}
--impostor-max N             (cap total impostor pairs; ignored for 'all' if not set)
--stratified-per-row K       (for stratified: per probe row, K impostor rows sampled)
--target-fmr α               (1:1 target FMR)
--target-fpir β --gallery-size N   (1:N: derive FMR ≈ β/N; exact inversion used)
--seed S                     (reproducible sampling)
--notes "..."                (stored in eval_runs)

Usage
-----
python build_pairs_and_metrics.py --db Eagle_Eye_Detection_Pipeline.db \
  --impostor-mode stratified --stratified-per-row 5 --impostor-max 50000 \
  --target-fmr 0.01 \
  --notes "baseline thresholds"
"""
import argparse
import json
import math
import random
import sqlite3
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# ---------------------------- Distance helpers ----------------------------

def landmarks_distance_rms(a23x2: np.ndarray, b23x2: np.ndarray) -> float:
    """
    RMS Euclidean distance across corresponding points (expects normalized 23x2).
    Smaller = more similar.
    """
    A = np.asarray(a23x2, dtype=float)
    B = np.asarray(b23x2, dtype=float)
    assert A.shape == (23, 2) and B.shape == (23, 2), "landmarks must be 23x2"
    diff = A - B
    return float(np.sqrt((diff * diff).sum(axis=1).mean()))

def enc_distance_euclidean(a128: np.ndarray, b128: np.ndarray) -> float:
    """
    Euclidean distance between 128-D encodings.
    Smaller = more similar.
    """
    a = np.asarray(a128, dtype=float).reshape(-1)
    b = np.asarray(b128, dtype=float).reshape(-1)
    assert a.shape == (128,) and b.shape == (128,), "encodings must be length 128"
    d = a - b
    return float(np.sqrt(np.dot(d, d)))

# ---------------------------- DB setup ------------------------------------

def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def ensure_eval_tables(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at    TEXT NOT NULL,
                notes         TEXT,
                impostor_mode TEXT NOT NULL,
                impostor_max  INTEGER,
                stratified_per_row INTEGER,
                target_fmr    REAL,
                target_fpir   REAL,
                gallery_size  INTEGER,
                fmr_used      REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_pairs (
                run_id     INTEGER NOT NULL,
                branch     TEXT CHECK(branch IN ('landmarks','encodings')) NOT NULL,
                label      TEXT CHECK(label IN ('genuine','impostor'))     NOT NULL,
                left_rowid INTEGER NOT NULL,
                right_rowid INTEGER NOT NULL,
                distance   REAL NOT NULL,
                PRIMARY KEY (run_id, branch, left_rowid, right_rowid)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pairs_run_branch ON eval_pairs(run_id, branch)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pairs_label ON eval_pairs(label)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_metrics (
                run_id       INTEGER NOT NULL,
                branch       TEXT NOT NULL,
                fmr_target   REAL NOT NULL,
                fmr_achieved REAL NOT NULL,
                threshold    REAL NOT NULL,
                fnmr         REAL NOT NULL,
                genuine_n    INTEGER NOT NULL,
                impostor_n   INTEGER NOT NULL,
                fpir_target  REAL,
                gallery_size INTEGER,
                fmr_required REAL,
                PRIMARY KEY (run_id, branch)
            )
        """)

def create_run(conn: sqlite3.Connection,
               impostor_mode: str,
               impostor_max: Optional[int],
               stratified_per_row: Optional[int],
               target_fmr: Optional[float],
               target_fpir: Optional[float],
               gallery_size: Optional[int],
               fmr_used: float,
               notes: Optional[str]) -> int:
    with conn:
        cur = conn.execute("""
            INSERT INTO eval_runs(created_at, notes, impostor_mode, impostor_max, stratified_per_row,
                                  target_fmr, target_fpir, gallery_size, fmr_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.utcnow().isoformat(timespec="seconds")+"Z", notes or "",
              impostor_mode, impostor_max, stratified_per_row, target_fmr, target_fpir, gallery_size, fmr_used))
        return int(cur.lastrowid)

def insert_pairs(conn: sqlite3.Connection,
                 run_id: int,
                 branch: str,
                 label: str,
                 rows: Iterable[Tuple[int, int, float]]) -> int:
    """
    rows: iterable of (left_rowid, right_rowid, distance); left_rowid < right_rowid enforced here.
    Returns count inserted.
    """
    data = []
    for l, r, d in rows:
        if l == r:
            continue
        if r < l:
            l, r = r, l
        data.append((run_id, branch, label, int(l), int(r), float(d)))
    if not data:
        return 0
    with conn:
        conn.executemany("""
            INSERT OR IGNORE INTO eval_pairs(run_id, branch, label, left_rowid, right_rowid, distance)
            VALUES (?, ?, ?, ?, ?, ?)
        """, data)
    return len(data)

def insert_metrics(conn: sqlite3.Connection,
                   run_id: int,
                   branch: str,
                   fmr_target: float,
                   fmr_achieved: float,
                   threshold: float,
                   fnmr: float,
                   genuine_n: int,
                   impostor_n: int,
                   fpir_target: Optional[float],
                   gallery_size: Optional[int],
                   fmr_required: Optional[float]) -> None:
    with conn:
        conn.execute("""
            INSERT OR REPLACE INTO eval_metrics(run_id, branch, fmr_target, fmr_achieved, threshold, fnmr,
                                                genuine_n, impostor_n, fpir_target, gallery_size, fmr_required)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, branch, fmr_target, fmr_achieved, threshold, fnmr,
              genuine_n, impostor_n, fpir_target, gallery_size, fmr_required))

# ---------------------------- Data loading --------------------------------

def load_facial_data(conn: sqlite3.Connection) -> Dict[str, List[Tuple[int, np.ndarray, np.ndarray]]]:
    """
    Returns: dict person_id -> list of (rowid, landmarks_23x2, enc_128)
    Skips rows with bad/missing JSON.
    """
    people: Dict[str, List[Tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    cur = conn.execute("SELECT rowid, person_id, face_landmarks, face_encodings FROM Facial_Data")
    bad_lm = bad_enc = 0
    total = 0
    for rowid, pid, jlm, jenc in cur.fetchall():
        total += 1
        try:
            lm = np.array(json.loads(jlm), dtype=np.float32)
            if lm.shape != (23, 2) or not np.all(np.isfinite(lm)):
                bad_lm += 1
                continue
        except Exception:
            bad_lm += 1
            continue
        try:
            enc = np.array(json.loads(jenc), dtype=np.float32).reshape(-1)
            if enc.shape != (128,) or not np.all(np.isfinite(enc)):
                bad_enc += 1
                continue
        except Exception:
            bad_enc += 1
            continue
        people[str(pid)].append((int(rowid), lm, enc))
    if not people:
        raise RuntimeError("No valid rows in Facial_Data (parsed landmarks/encodings are empty).")
    return people

# ---------------------------- Pair builders --------------------------------

def make_genuine_pairs(people: Dict[str, List[Tuple[int, np.ndarray, np.ndarray]]]
                      ) -> Tuple[List[Tuple[int,int,float]], List[Tuple[int,int,float]]]:
    """
    Returns (landmarks_pairs, enc_pairs) where each is a list of (rowid_a, rowid_b, distance).
    """
    lm_pairs: List[Tuple[int,int,float]] = []
    enc_pairs: List[Tuple[int,int,float]] = []
    for pid, rows in people.items():
        if len(rows) < 2:
            continue
        for (ra, lma, enca), (rb, lmb, encb) in combinations(rows, 2):
            lm_pairs.append((ra, rb, landmarks_distance_rms(lma, lmb)))
            enc_pairs.append((ra, rb, enc_distance_euclidean(enca, encb)))
    return lm_pairs, enc_pairs

def make_impostor_pairs_random(people: Dict[str, List[Tuple[int,np.ndarray,np.ndarray]]],
                               max_pairs: int,
                               rng: random.Random) -> Tuple[List[Tuple[int,int,float]], List[Tuple[int,int,float]]]:
    """
    Sample impostor pairs uniformly by drawing two distinct persons and one random row from each.
    """
    pids = list(people.keys())
    if len(pids) < 2:
        return [], []
    seen = set()
    lm_pairs: List[Tuple[int,int,float]] = []
    enc_pairs: List[Tuple[int,int,float]] = []
    attempts = 0
    target = max_pairs if max_pairs else 0
    while (not target) or (len(lm_pairs) < target):
        attempts += 1
        if attempts > (target * 20 if target else 100000):
            break  # safety stop
        pa, pb = rng.sample(pids, 2)
        ra = rng.choice(people[pa])
        rb = rng.choice(people[pb])
        a_id, a_lm, a_enc = ra
        b_id, b_lm, b_enc = rb
        key = (min(a_id, b_id), max(a_id, b_id))
        if key in seen:
            continue
        seen.add(key)
        lm_pairs.append((key[0], key[1], landmarks_distance_rms(a_lm, b_lm)))
        enc_pairs.append((key[0], key[1], enc_distance_euclidean(a_enc, b_enc)))
        if not target and len(lm_pairs) >= 100000:  # unbounded guard
            break
    return lm_pairs, enc_pairs

def make_impostor_pairs_stratified(people: Dict[str, List[Tuple[int,np.ndarray,np.ndarray]]],
                                   per_row: int,
                                   max_pairs: Optional[int],
                                   rng: random.Random) -> Tuple[List[Tuple[int,int,float]], List[Tuple[int,int,float]]]:
    """
    For each row, sample 'per_row' opponent rows from *other* persons.
    """
    pids = list(people.keys())
    if len(pids) < 2:
        return [], []
    # Pre-build flattened list of (rowid, lm, enc, pid)
    flat: List[Tuple[int, np.ndarray, np.ndarray, str]] = []
    for pid, rows in people.items():
        for (rid, lm, enc) in rows:
            flat.append((rid, lm, enc, pid))

    lm_pairs: List[Tuple[int,int,float]] = []
    enc_pairs: List[Tuple[int,int,float]] = []
    seen = set()
    for rid, lm, enc, pid in flat:
        # collect candidates from other persons
        # sample with replacement to avoid heavy filtering costs
        draw = 0
        while draw < per_row:
            opp_pid = rng.choice(pids)
            if opp_pid == pid or not people[opp_pid]:
                continue
            opp_rid, opp_lm, opp_enc = rng.choice(people[opp_pid])
            key = (min(rid, opp_rid), max(rid, opp_rid))
            if key in seen:
                continue
            seen.add(key)
            lm_pairs.append((key[0], key[1], landmarks_distance_rms(lm, opp_lm)))
            enc_pairs.append((key[0], key[1], enc_distance_euclidean(enc, opp_enc)))
            draw += 1
            if max_pairs and len(lm_pairs) >= max_pairs:
                return lm_pairs, enc_pairs
    return lm_pairs, enc_pairs

def make_impostor_pairs_all(people: Dict[str, List[Tuple[int,np.ndarray,np.ndarray]]],
                            max_pairs: Optional[int] = None
                           ) -> Tuple[List[Tuple[int,int,float]], List[Tuple[int,int,float]]]:
    """
    All cross-person pairs (can be very large). Optional cap with max_pairs.
    """
    lm_pairs: List[Tuple[int,int,float]] = []
    enc_pairs: List[Tuple[int,int,float]] = []
    pids = list(people.keys())
    for i in range(len(pids)):
        for j in range(i+1, len(pids)):
            A = people[pids[i]]
            B = people[pids[j]]
            for (ra, lma, enca) in A:
                for (rb, lmb, encb) in B:
                    key = (min(ra, rb), max(ra, rb))
                    lm_pairs.append((key[0], key[1], landmarks_distance_rms(lma, lmb)))
                    enc_pairs.append((key[0], key[1], enc_distance_euclidean(enca, encb)))
                    if max_pairs and len(lm_pairs) >= max_pairs:
                        return lm_pairs, enc_pairs
    return lm_pairs, enc_pairs

# ---------------------------- Threshold sweep ------------------------------

def derive_fmr_from_fpir(fpir_target: float, gallery_size: int) -> float:
    """
    Solve for FMR in 1 - (1 - FMR)^N = FPIR  =>  FMR = 1 - (1 - FPIR)^(1/N)
    """
    if gallery_size <= 0:
        raise ValueError("gallery_size must be positive to derive FMR from FPIR.")
    fpir_target = float(max(0.0, min(1.0, fpir_target)))
    return 1.0 - (1.0 - fpir_target) ** (1.0 / gallery_size)

def pick_threshold_for_fmr(d_imp: np.ndarray, target_fmr: float) -> Tuple[float, float]:
    """
    For distance scores where smaller=better:
      FMR(tau) = fraction of impostor distances <= tau.
    Choose the smallest tau giving FMR <= target_fmr.
    Returns (tau, fmr_achieved).
    """
    M = len(d_imp)
    if M == 0:
        return float("inf"), 0.0
    target_fmr = float(max(0.0, min(1.0, target_fmr)))
    d_sorted = np.sort(d_imp)
    # k_max such that (k_max+1)/M <= target_fmr  => k_max = floor(target_fmr*M) - 1
    k_max = math.floor(target_fmr * M) - 1
    if k_max < 0:
        # require FMR=0: choose tau below min impostor
        tau = float(d_sorted[0]) - 1e-12
        return tau, 0.0
    k_max = min(k_max, M - 1)
    tau = float(d_sorted[k_max])
    fmr_achieved = float((k_max + 1) / M)
    # ensure monotone: if next value is same, we could move left; but this is fine for reporting
    return tau, fmr_achieved

def compute_fnmr(d_gen: np.ndarray, tau: float) -> float:
    if len(d_gen) == 0:
        return 0.0
    return float(np.mean(d_gen > tau))

# ---------------------------- Orchestrator ---------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build genuine/impostor pairs, sweep thresholds, store metrics.")
    ap.add_argument("--db", required=True, help="SQLite DB path (e.g., Eagle_Eye_Detection_Pipeline.db)")
    ap.add_argument("--impostor-mode", choices=["random", "stratified", "all"], default="stratified",
                    help="How to form impostor pairs (default: stratified)")
    ap.add_argument("--impostor-max", type=int, default=50000,
                    help="Max impostor pairs (cap to avoid explosion). For 'all', set higher or 0 for unlimited.")
    ap.add_argument("--stratified-per-row", type=int, default=5,
                    help="Stratified mode: number of impostor opponents sampled per row (default: 5)")
    ap.add_argument("--target-fmr", type=float, default=0.01,
                    help="Target FMR for 1:1 (default: 0.01)")
    ap.add_argument("--target-fpir", type=float, default=None,
                    help="Target FPIR for 1:N (optional; if set, derives FMR from FPIR and gallery size)")
    ap.add_argument("--gallery-size", type=int, default=None,
                    help="Gallery size N for 1:N derivation of FMR (required if --target-fpir is set)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible sampling (default: 42)")
    ap.add_argument("--notes", type=str, default="", help="Free-text run notes")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    db_path = Path(args.db)
    conn = open_db(db_path)
    ensure_eval_tables(conn)

    # Load data
    people = load_facial_data(conn)

    # Build genuine pairs
    lm_gen, enc_gen = make_genuine_pairs(people)

    # Build impostor pairs
    if args.impostor_mode == "random":
        lm_imp, enc_imp = make_impostor_pairs_random(people, args.impostor_max, rng)
    elif args.impostor_mode == "stratified":
        lm_imp, enc_imp = make_impostor_pairs_stratified(people, args.stratified_per_row, args.impostor_max, rng)
    else:
        lm_imp, enc_imp = make_impostor_pairs_all(people, args.impostor_max if args.impostor_max and args.impostor_max > 0 else None)

    # Convert to numpy for sweeps
    d_gen = {
        "landmarks": np.array([d for _, _, d in lm_gen], dtype=float),
        "encodings": np.array([d for _, _, d in enc_gen], dtype=float),
    }
    d_imp = {
        "landmarks": np.array([d for _, _, d in lm_imp], dtype=float),
        "encodings": np.array([d for _, _, d in enc_imp], dtype=float),
    }

    # Determine FMR to use
    fmr_required = None
    fmr_used = args.target_fmr
    if args.target_fpir is not None:
        if args.gallery_size is None:
            raise SystemExit("[ERR] --gallery-size is required when --target-fpir is provided.")
        fmr_required = derive_fmr_from_fpir(args.target_fpir, args.gallery_size)
        fmr_used = min(args.target_fmr, fmr_required) if args.target_fmr is not None else fmr_required

    # Create run
    run_id = create_run(conn,
                        impostor_mode=args.impostor_mode,
                        impostor_max=args.impostor_max,
                        stratified_per_row=args.stratified_per_row if args.impostor_mode == "stratified" else None,
                        target_fmr=args.target_fmr,
                        target_fpir=args.target_fpir,
                        gallery_size=args.gallery_size,
                        fmr_used=fmr_used,
                        notes=args.notes)

    # Insert pairs for both branches
    ins_gen_lm = insert_pairs(conn, run_id, "landmarks", "genuine", lm_gen)
    ins_gen_enc = insert_pairs(conn, run_id, "encodings", "genuine", enc_gen)
    ins_imp_lm = insert_pairs(conn, run_id, "landmarks", "impostor", lm_imp)
    ins_imp_enc = insert_pairs(conn, run_id, "encodings", "impostor", enc_imp)

    # Sweep and store metrics per branch
    summary_rows = []
    for branch in ("landmarks", "encodings"):
        if len(d_imp[branch]) == 0 or len(d_gen[branch]) == 0:
            print(f"[WARN] Branch '{branch}' has insufficient pairs (gen={len(d_gen[branch])}, imp={len(d_imp[branch])}); skipping metrics.")
            continue
        tau, fmr_ach = pick_threshold_for_fmr(d_imp[branch], fmr_used)
        fnmr = compute_fnmr(d_gen[branch], tau)
        insert_metrics(conn, run_id, branch, fmr_used, fmr_ach, tau, fnmr,
                       genuine_n=len(d_gen[branch]), impostor_n=len(d_imp[branch]),
                       fpir_target=args.target_fpir, gallery_size=args.gallery_size,
                       fmr_required=fmr_required)
        summary_rows.append((branch, tau, fmr_ach, fnmr, len(d_gen[branch]), len(d_imp[branch])))

    # Pretty summary
    print("\n=== Evaluation Summary ===")
    print(f"DB: {db_path}")
    print(f"Run ID: {run_id} | Mode: {args.impostor_mode} | Seed: {args.seed}")
    if args.target_fpir is not None:
        print(f"Target FPIR: {args.target_fpir}  Gallery size: {args.gallery_size}  -> Required FMR: {fmr_required:.6g}")
    print(f"Target FMR used: {fmr_used:.6g}")
    print("Pair counts inserted to eval_pairs:")
    print(f"  Genuine  landmarks: {ins_gen_lm:>8} | encodings: {ins_gen_enc:>8}")
    print(f"  Impostor landmarks: {ins_imp_lm:>8} | encodings: {ins_imp_enc:>8}")
    print("\nOperating points per branch (distance, smaller is better):")
    if summary_rows:
        for branch, tau, fmr_ach, fnmr, nG, nI in summary_rows:
            print(f"  {branch:10s} | τ={tau:.6f} | FMR={fmr_ach:.6g} | FNMR={fnmr:.6g} | gen={nG} imp={nI}")
    else:
        print("  (no branches with sufficient data)")

    print("\nTips:")
    print(" - Use --impostor-mode stratified (default) with --stratified-per-row K to scale smoothly.")
    print(" - For 1:N, set --target-fpir and --gallery-size; script derives stricter FMR automatically.")
    print(" - Rerun with different seeds to assess sampling variance of impostor pairs.")

if __name__ == "__main__":
    main()
