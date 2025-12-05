#!/usr/bin/env python3
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def compute_calibration_metrics(csv_path, out_dir, bins=10):
    df = pd.read_csv(csv_path)
    if "mask_prob" not in df.columns:
        raise ValueError("mask_prob column not found")

    # Infer binary ground truth (1 = masked)
    df["true_masked"] = df["mask_route"].eq("landmarks").astype(int)

    p = df["mask_prob"].clip(0, 1)
    y = df["true_masked"]

    # --- Brier score
    brier = np.mean((p - y) ** 2)

    # --- ECE
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_ids = np.digitize(p, bin_edges) - 1
    ece = 0.0
    calib = []
    for b in range(bins):
        bin_mask = bin_ids == b
        if not np.any(bin_mask):
            continue
        conf = np.mean(p[bin_mask])
        acc = np.mean(y[bin_mask])
        ece += np.abs(acc - conf) * (np.sum(bin_mask) / len(df))
        calib.append((conf, acc, np.sum(bin_mask)))

    calib = np.array(calib)
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True, parents=True)

    # --- Plot reliability diagram
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(calib[:, 0], calib[:, 1], "o-", color="C0", label="Model")
    ax.set_xlabel("Predicted mask probability")
    ax.set_ylabel("Empirical masked fraction")
    ax.set_title(f"Mask Probability Calibration\nECE={ece:.4f}, Brier={brier:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "reliability_diagram.png")
    plt.close(fig)

    # --- Save metrics
    with open(out_dir / "calibration_summary.txt", "w") as f:
        f.write(f"Brier Score: {brier:.6f}\n")
        f.write(f"Expected Calibration Error (ECE): {ece:.6f}\n")

    print(f"[OK] Calibration metrics -> {out_dir}")
    print(f"Brier={brier:.6f}, ECE={ece:.6f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute calibration metrics for mask probabilities")
    ap.add_argument("--csv", required=True, help="runtime_matches.csv")
    ap.add_argument("--out-dir", default="./calibration_out", help="Output directory")
    ap.add_argument("--bins", type=int, default=10, help="Number of bins for ECE")
    args = ap.parse_args()
    compute_calibration_metrics(args.csv, args.out_dir, args.bins)
