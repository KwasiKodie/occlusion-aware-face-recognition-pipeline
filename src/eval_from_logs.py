#!/usr/bin/env python3
"""
Rebuild comparisons_log.csv from runtime/faces logs, then compute operating-point sweeps
(ROC: TAR vs FAR; DET: FNMR vs FMR) per branch and for the routed system.
Also extract EER, TAR@FAR=1e-2/1e-3, FNMR@FMR=1e-2/1e-3 with bootstrap 95% CIs.
"""

import argparse, re, json, sqlite3, sys, math, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------
STEM_PID = re.compile(r"^([a-z]+_[a-z]+)_[0-9]+$", re.IGNORECASE)

def infer_person_from_stem(stem: str) -> Optional[str]:
    m = STEM_PID.match(stem)
    return m.group(1).lower() if m else None

def safe_float(x):
    try: return float(x)
    except: return np.nan

def binomial_ci(k, n, alpha=0.05):
    # Wilson score interval
    if n == 0: return (np.nan, np.nan)
    z = 1.95996398454005  # ~95%
    phat = k / n
    denom = 1 + z**2/n
    center = (phat + z*z/(2*n)) / denom
    half = z * math.sqrt((phat*(1-phat) + z*z/(4*n))/n) / denom
    return max(0.0, center - half), min(1.0, center + half)

def bootstrap_ci(values: np.ndarray, stat_fn, B=2000, alpha=0.05, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    vals = values.copy()
    V = []
    n = len(vals)
    if n == 0: return (np.nan, np.nan)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        V.append(stat_fn(vals[idx]))
    V = np.sort(np.array(V))
    lo = np.quantile(V, alpha/2)
    hi = np.quantile(V, 1 - alpha/2)
    return float(lo), float(hi)

# -----------------------------
# ROC/DET from scores & labels
# -----------------------------
def sweep_thresholds(scores, labels, direction="lower_better"):
    """Return arrays of thresholds, FAR, TAR (verification), FMR, FNMR (DET).
    labels: 1 for genuine, 0 for impostor.
    direction: 'lower_better' means accept if score <= tau.
    """
    df = pd.DataFrame({"s": scores, "y": labels}).dropna()
    if df.empty:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    # candidate thresholds: unique scores plus ±inf guards
    uniq = np.unique(df["s"].values)
    # Include +/- inf to cover ends
    taus = np.concatenate(([-np.inf], uniq, [np.inf]))
    FAR_list, TAR_list, FMR_list, FNMR_list = [], [], [], []
    for tau in taus:
        if direction == "lower_better":
            accept = (df["s"].values <= tau)
        else:
            accept = (df["s"].values >= tau)
        y = df["y"].values.astype(int)
        # positives = genuine
        P = (y == 1)
        N = (y == 0)
        TP = (accept & P).sum()
        FP = (accept & N).sum()
        FN = ((~accept) & P).sum()
        TN = ((~accept) & N).sum()
        TAR = TP / P.sum() if P.sum() > 0 else np.nan
        FAR = FP / N.sum() if N.sum() > 0 else np.nan
        FMR = FAR
        FNMR = 1 - TAR if not np.isnan(TAR) else np.nan
        FAR_list.append(FAR); TAR_list.append(TAR); FMR_list.append(FMR); FNMR_list.append(FNMR)
    return taus, np.array(FAR_list), np.array(TAR_list), np.array(FMR_list), np.array(FNMR_list)

def interp_at(FAR, TAR, target):
    """Linear interpolate TAR@FAR=target; FAR must be non-decreasing over taus."""
    mask = ~np.isnan(FAR) & ~np.isnan(TAR)
    f, t = FAR[mask], TAR[mask]
    if len(f) < 2: return np.nan
    # remove duplicates in FAR
    uniqF, idx = np.unique(f, return_index=True)
    t = t[idx]
    f = uniqF
    if target < f.min() or target > f.max():
        # if target below min FAR, return TAR at min FAR
        if target < f.min(): return t[f.argmin()]
        # if target above max FAR, return TAR at max FAR
        return t[f.argmax()]
    return float(np.interp(target, f, t))

def eer_from_curves(FAR, FNMR):
    mask = ~np.isnan(FAR) & ~np.isnan(FNMR)
    if mask.sum() < 2: return np.nan
    f, g = FAR[mask], FNMR[mask]
    # Find point where |FAR - FNMR| is minimized; linearly refine
    idx = np.argmin(np.abs(f - g))
    return float( (f[idx] + g[idx]) / 2.0 )

# -----------------------------
# Build comparisons_log
# -----------------------------
def make_comparisons(runtime_csv: Path, faces_csv: Path, db_path: Path) -> pd.DataFrame:
    """Join runtime + face info + gallery person IDs, create 1/0 labels."""

    rm = pd.read_csv(runtime_csv)
    # print("\n[DEBUG] Runtime log column names detected:")
    # for i, c in enumerate(rm.columns):
        # print(f"  {i}: '{c}' (len={len(c)}) repr={repr(c)}")

    if faces_csv and Path(faces_csv).exists():
        fc = pd.read_csv(faces_csv)
        rm = rm.merge(fc[["source", "branch", "best_rowid"]], on=["source", "branch"], how="left")

    # Map gallery rowid → person_id
    con = sqlite3.connect(str(db_path))
    gid = pd.read_sql_query("SELECT rowid, person_id FROM Facial_Data", con)
    con.close()
    gid["person_id"] = gid["person_id"].astype(str).str.strip().str.lower().str.rstrip("_")

    # Normalize column names aggressively
    def clean_col(c):
        return (
            str(c)
            .encode("utf-8", "ignore")
            .decode("utf-8")
            .replace("\ufeff", "")   # remove BOM
            .strip()
            .lower()
        )

    rm.columns = [clean_col(c) for c in rm.columns]
    # print(f"[INFO] Normalized column names: {rm.columns.tolist()}")


    # --- robust gallery-rowid lookup ---
    # rm.columns = [c.strip().lower() for c in rm.columns]  # normalize all column names
    # candidates = [c for c in rm.columns if c in ("best_rowid", "gallery_rowid", "rowid")]
    candidates = [c for c in rm.columns if "best_rowid" in c or "gallery_rowid" in c or c == "rowid"]

    if candidates:
        # Prefer best_rowid_x (runtime_matches) over best_rowid_y if both exist
        best_col = sorted(candidates)[0]
        print(f"[INFO] Using gallery link column: {best_col}")
    else:
        print("[WARN] No best_rowid/gallery_rowid column found in runtime_matches; gallery IDs cannot be mapped.")
        best_col = None

    if best_col:
        rm["_gallery_person_id"] = rm[best_col].map(
            dict(zip(gid["rowid"], gid["person_id"]))
        )
    else:
        rm["_gallery_person_id"] = np.nan


    # Derive probe_person_id from filename
    def normalize_probe(name: str) -> str:
        if pd.isna(name): return np.nan
        stem = Path(str(name)).stem.lower()
        # remove numeric suffixes like _1 or _12
        stem = re.sub(r"_[0-9]+$", "", stem)
        return stem.strip().rstrip("_")

    def normalize_pid(pid):
        if pd.isna(pid): return np.nan
        return str(pid).strip().lower().rstrip("_")


    rm["_probe_person_id"] = rm["source"].apply(normalize_probe)

    pids_probe = rm["_probe_person_id"].apply(normalize_pid)
    pids_gallery = rm["_gallery_person_id"].apply(normalize_pid)

    rm["_label"] = np.where(
        pids_probe.notna() & pids_gallery.notna() & (pids_probe == pids_gallery), 1,
        np.where(
            pids_probe.notna() & pids_gallery.notna(), 0, np.nan
        )
    )


    # Assign labels (1 = same person, 0 = different)
    # rm["_label"] = np.where(
        # rm["_probe_person_id"].notna() & rm["_gallery_person_id"].notna() &
        # (rm["_probe_person_id"] == rm["_gallery_person_id"]), 1,
        # np.where(
        #     rm["_probe_person_id"].notna() & rm["_gallery_person_id"].notna(), 0, np.nan
        # )
    # )

    # Drop unused
    cols = ["source", "branch", "distance", "tau", "eval_run_id", "_probe_person_id",
            "_gallery_person_id", "_label"]
    comp = rm[cols].rename(columns={"_label": "label"})
    return comp


# -----------------------------
# τ from DB
# -----------------------------
def load_taus_from_db(db_path: Path, run_id: Optional[int]) -> Dict[str, float]:
    """Extract branch-specific thresholds τ from eval_metrics(run_id, branch, threshold)."""
    taus = {}
    if db_path is None or not db_path.exists() or run_id is None:
        return taus

    con = sqlite3.connect(db_path)
    try:
        q = """
        SELECT branch, threshold
        FROM eval_metrics
        WHERE run_id = ?
        """
        df = pd.read_sql_query(q, con, params=(run_id,))
        for _, r in df.iterrows():
            br = str(r["branch"]).strip().lower()
            taus[br] = float(r["threshold"])
    except Exception as e:
        print("[WARN] load_taus:", e)
    finally:
        con.close()

    return taus


# -----------------------------
# Routed system scores
# -----------------------------
def routed_scores(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse per-branch scores to a single per-probe score using the branch actually used.
    If try-both occurred and both branches present for same probe, take min distance (best)."""
    # Group by probe_src; choose best (min distance)
    keep = df.dropna(subset=["label"])
    if keep.empty: return np.array([]), np.array([])
    # grp = keep.groupby("probe_src", as_index=False).apply
    grp = keep.groupby("source", as_index=False).apply(
        lambda g: pd.Series({
            "label": g["label"].iloc[0],  # assume same true label across branches
            "score": g["distance"].min()
        })
    ).reset_index(drop=True)
    return grp["score"].values, grp["label"].values

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-matches", required=True)
    ap.add_argument("--faces-log", required=False)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out-dir", default="eval_outputs")
    ap.add_argument("--run-id", type=int, required=False, help="Run ID (replaces eval-run-id)")
    ap.add_argument("--eval-run-id", type=int, required=False)
    ap.add_argument("--far-points", type=float, nargs="+", default=[1e-2, 1e-3])
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build comparisons_log.csv
    comp = make_comparisons(Path(args.runtime_matches),
                            Path(args.faces_log) if args.faces_log else None,
                            Path(args.db))
    comp_path = out_dir / "comparisons_log.csv"
    comp.to_csv(comp_path, index=False)
    print(f"[OK] comparisons_log.csv -> {comp_path}")

    # 2) Split by branch (clean)
    comp_clean = comp.dropna(subset=["label","distance","branch"])
    branches = sorted(comp_clean["branch"].str.lower().unique().tolist())

    # 3) Load τ from DB for reference (if run_id present in log, prefer that)
    run_id = args.run_id or args.eval_run_id
    # Handle eval_run_id vs run_id naming differences safely
    if ("run_id" not in comp.columns) and ("eval_run_id" not in comp.columns):
        print("[INFO] No run_id/eval_run_id column found in comparisons log; skipping DB τ overlay.")
        taus = {}
    else:
        col = "run_id" if "run_id" in comp.columns else "eval_run_id"
        if np.all(pd.isna(comp[col])) and run_id is None:
            print("[INFO] All run_id values NaN; using provided --eval-run-id if given.")
            taus = {}
        else:
            if run_id is None:
                vals = comp[col].dropna().astype(int)
                run_id = int(vals.mode().iloc[0]) if not vals.empty else None
            taus = load_taus_from_db(Path(args.db), run_id)
            print(f"[INFO] τ from DB for run_id={run_id}: {taus}")


    # 4) Per-branch ROC/DET and metrics
    metrics_rows = []
    for br in branches:
        sub = comp_clean[comp_clean["branch"].str.lower() == br]
        scores = sub["distance"].values.astype(float)
        labels = sub["label"].values.astype(int)
        taus_arr, FAR, TAR, FMR, FNMR = sweep_thresholds(scores, labels, "lower_better")
        # EER
        EER = eer_from_curves(FAR, FNMR)

        # target points
        row = {"scope": f"branch:{br}", "EER": EER}
        for fp in args.far_points:
            row[f"TAR@FAR={fp}"] = interp_at(FAR, TAR, fp)
            # FNMR@FMR is 1 - TAR@FAR
            row[f"FNMR@FMR={fp}"] = (1 - row[f"TAR@FAR={fp}"]) if not math.isnan(row[f"TAR@FAR={fp}"]) else np.nan

        # Bootstrap CIs over trials (resampling indices)
        rng = np.random.default_rng(42)
        n = len(scores)
        if n >= 5:
            def stat_TAR_at(fp):
                def stat_fn(idx):
                    sc = scores[idx]; lb = labels[idx]
                    t, f, _, _ = sweep_thresholds(sc, lb, "lower_better")[1:5]
                    return interp_at(t, f, fp)  # TAR@FAR
                return stat_fn

            # EER CI
            def stat_eer(idx):
                sc = scores[idx]; lb = labels[idx]
                _, FARb, _, FMRb, FNMRb = sweep_thresholds(sc, lb, "lower_better")
                return eer_from_curves(FARb, FNMRb)

            lo, hi = bootstrap_ci(np.arange(n), lambda idx: stat_eer(idx), B=args.bootstrap)
            row["EER_CI95"] = (lo, hi)
            for fp in args.far_points:
                lo, hi = bootstrap_ci(np.arange(n), lambda idx, fp=fp: stat_TAR_at(fp)(idx), B=args.bootstrap)
                row[f"TAR@FAR={fp}_CI95"] = (lo, hi)
                row[f"FNMR@FMR={fp}_CI95"] = (1 - hi, 1 - lo) if not (math.isnan(lo) or math.isnan(hi)) else (np.nan, np.nan)
        else:
            row["EER_CI95"] = (np.nan, np.nan)
            for fp in args.far_points:
                row[f"TAR@FAR={fp}_CI95"] = (np.nan, np.nan)
                row[f"FNMR@FMR={fp}_CI95"] = (np.nan, np.nan)

        metrics_rows.append(row)

        # Plots
        fig, ax = plt.subplots(figsize=(5,4), dpi=160)
        ax.plot(FAR, TAR, lw=2, label=f"{br}")
        ax.set_xlabel("FAR"); ax.set_ylabel("TAR"); ax.set_title(f"ROC — {br}")
        ax.grid(alpha=0.3)
        # mark τ from DB if available (convert τ→point by finding closest threshold)
        if br in taus and len(taus_arr)>0:
            # find closest index where tau matches (distance grid)
            idx = np.argmin(np.abs(taus_arr - taus[br]))
            if not np.isnan(FAR[idx]) and not np.isnan(TAR[idx]):
                ax.scatter([FAR[idx]], [TAR[idx]], c="k", s=40, zorder=3, label="operating τ")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"roc_{br}.png"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5,4), dpi=160)
        ax.plot(FMR, FNMR, lw=2, label=f"{br}")
        ax.set_xlabel("FMR"); ax.set_ylabel("FNMR"); ax.set_title(f"DET — {br}")
        ax.grid(alpha=0.3)
        if br in taus and len(taus_arr)>0:
            idx = np.argmin(np.abs(taus_arr - taus[br]))
            if not np.isnan(FMR[idx]) and not np.isnan(FNMR[idx]):
                ax.scatter([FMR[idx]], [FNMR[idx]], c="k", s=40, zorder=3, label="operating τ")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"det_{br}.png"); plt.close(fig)

    # 5) Routed system (collapse by probe, min distance when both branches present)
    scores_routed, labels_routed = routed_scores(comp_clean)
    if len(scores_routed) > 0:
        taus_arr, FAR, TAR, FMR, FNMR = sweep_thresholds(scores_routed, labels_routed, "lower_better")
        # Summaries
        row = {"scope": "routed", "EER": eer_from_curves(FAR, FNMR)}
        for fp in args.far_points:
            row[f"TAR@FAR={fp}"] = interp_at(FAR, TAR, fp)
            row[f"FNMR@FMR={fp}"] = 1 - row[f"TAR@FAR={fp}"] if not math.isnan(row[f"TAR@FAR={fp}"]) else np.nan

        n = len(scores_routed)
        if n >= 5:
            def stat_eer(idx):
                sc = scores_routed[idx]; lb = labels_routed[idx]
                _, FARb, _, FMRb, FNMRb = sweep_thresholds(sc, lb, "lower_better")
                return eer_from_curves(FARb, FNMRb)
            lo, hi = bootstrap_ci(np.arange(n), lambda idx: stat_eer(idx), B=args.bootstrap)
            row["EER_CI95"] = (lo, hi)
            for fp in args.far_points:
                def stat_tar(idx, fp=fp):
                    sc = scores_routed[idx]; lb = labels_routed[idx]
                    tFAR, tTAR, _, _ = sweep_thresholds(sc, lb, "lower_better")[1:5]
                    return interp_at(tFAR, tTAR, fp)
                lo, hi = bootstrap_ci(np.arange(n), stat_tar, B=args.bootstrap)
                row[f"TAR@FAR={fp}_CI95"] = (lo, hi)
                row[f"FNMR@FMR={fp}_CI95"] = (1 - hi, 1 - lo) if not (math.isnan(lo) or math.isnan(hi)) else (np.nan, np.nan)
        else:
            row["EER_CI95"] = (np.nan, np.nan)
            for fp in args.far_points:
                row[f"TAR@FAR={fp}_CI95"] = (np.nan, np.nan)
                row[f"FNMR@FMR={fp}_CI95"] = (np.nan, np.nan)

        metrics_rows.append(row)

        # Plots
        fig, ax = plt.subplots(figsize=(5,4), dpi=160)
        ax.plot(FAR, TAR, lw=2, label="routed")
        ax.set_xlabel("FAR"); ax.set_ylabel("TAR"); ax.set_title("ROC — routed system")
        ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
        fig.savefig(out_dir / "roc_routed.png"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5,4), dpi=160)
        ax.plot(FMR, FNMR, lw=2, label="routed")
        ax.set_xlabel("FMR"); ax.set_ylabel("FNMR"); ax.set_title("DET — routed system")
        ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
        fig.savefig(out_dir / "det_routed.png"); plt.close(fig)

    # 6) Save metrics table
    mdf = pd.DataFrame(metrics_rows)
    mdf.to_csv(out_dir / "sweep_metrics_summary.csv", index=False)
    print("[OK] sweep plots ->", out_dir)

if __name__ == "__main__":
    main()
