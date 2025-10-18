#!/usr/bin/env python3
"""
Analyze temporal localization results (Step 6).

Reads:  reports/summary.csv
Writes: reports/analysis/
  - metrics_overall.json
  - metrics_per_class.csv
  - hist_mae.png
  - calibration.png

Expected CSV columns (robust to missing ones):
  required:  video_id, event_label, mae
  optional:  rmse, inside, hit_0.25, hit_0.5, hit_1.0, confidence, entropy, migt, piw
"""

import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
def safe_mean(series):
    return float(series.dropna().mean()) if len(series.dropna()) else float("nan")

def present(cols, name):
    return name in cols

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def bin_confidence(df, conf_col="confidence", target_col="piw", n_bins=10):
    """Return calibration table: bin-wise mean confidence vs accuracy (PiW or Inside)."""
    if conf_col not in df.columns:  # nothing to calibrate
        return None
    # pick a target: prefer piw, else inside, else hit_0.5
    if target_col not in df.columns:
        for alt in ["inside", "hit_0.5", "hit_0.25", "hit_1.0"]:
            if alt in df.columns:
                target_col = alt
                break
        if target_col not in df.columns:
            return None

    valid = df[[conf_col, target_col]].dropna()
    if valid.empty:
        return None
    # clip to [0,1] for safety
    conf = valid[conf_col].clip(0, 1)
    tgt = valid[target_col].astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, bins, right=True)
    rows = []
    for b in range(1, n_bins + 1):
        mask = idx == b
        if mask.sum() == 0:
            rows.append({
                "bin": b,
                "bin_left": bins[b-1],
                "bin_right": bins[b],
                "mean_conf": np.nan,
                "mean_acc": np.nan,
                "count": 0
            })
        else:
            rows.append({
                "bin": b,
                "bin_left": bins[b-1],
                "bin_right": bins[b],
                "mean_conf": float(conf[mask].mean()),
                "mean_acc": float(tgt[mask].mean()),
                "count": int(mask.sum())
            })
    return pd.DataFrame(rows)

# ---------- main ----------
def main(args):
    ensure_outdir(args.out_dir)

    df = pd.read_csv(args.summary_csv)
    cols = set(df.columns)

    # Normalize common column names (accept both "hit@0.5" and "hit_0.5")
    rename_map = {}
    for cand in ["hit@0.25", "hit_0.25"]:
        if cand in cols: rename_map[cand] = "hit_0.25"
    for cand in ["hit@0.5", "hit_0.5"]:
        if cand in cols: rename_map[cand] = "hit_0.5"
    for cand in ["hit@1.0", "hit_1.0"]:
        if cand in cols: rename_map[cand] = "hit_1.0"
    if rename_map:
        df = df.rename(columns=rename_map)
        cols = set(df.columns)

    # Overall metrics
    overall = {
        "num_clips": int(len(df)),
        "mae_mean": safe_mean(df["mae"]) if "mae" in cols else None,
        "mae_median": float(df["mae"].median()) if "mae" in cols else None,
    }
    if present(cols, "rmse"):
        overall["rmse_mean"] = safe_mean(df["rmse"])
    if present(cols, "inside"):
        overall["inside_mean"] = safe_mean(df["inside"])
    for k in ["hit_0.25", "hit_0.5", "hit_1.0"]:
        if present(cols, k):
            overall[f"{k}_mean"] = safe_mean(df[k])
    if present(cols, "migt"):
        overall["migt_mean"] = safe_mean(df["migt"])
        overall["migt_median"] = float(df["migt"].median())
    if present(cols, "piw"):
        overall["piw_mean"] = safe_mean(df["piw"])
    if present(cols, "confidence"):
        overall["confidence_mean"] = safe_mean(df["confidence"])
    if present(cols, "entropy"):
        overall["entropy_mean"] = safe_mean(df["entropy"])

    # Save overall metrics
    with open(os.path.join(args.out_dir, "metrics_overall.json"), "w") as f:
        json.dump(overall, f, indent=2)
    print("✓ metrics_overall.json")

    # Per-class metrics
    if "event_label" in cols:
        per_class_rows = []
        grp = df.groupby("event_label")
        for label, g in grp:
            row = {
                "event_label": label,
                "count": int(len(g)),
                "mae_mean": safe_mean(g["mae"]) if "mae" in g.columns else None,
                "mae_median": float(g["mae"].median()) if "mae" in g.columns else None,
            }
            for k in ["rmse", "inside", "hit_0.25", "hit_0.5", "hit_1.0", "migt", "piw", "confidence", "entropy"]:
                if k in g.columns:
                    row[f"{k}_mean"] = safe_mean(g[k])
            per_class_rows.append(row)
        per_class = pd.DataFrame(per_class_rows).sort_values("event_label")
        per_class.to_csv(os.path.join(args.out_dir, "metrics_per_class.csv"), index=False)
        print("✓ metrics_per_class.csv")
    else:
        print("⚠️  'event_label' not found; skipping per-class table.")

    # Plots
    # 1) Histogram of MAE
    if "mae" in cols:
        plt.figure(figsize=(6,4))
        df["mae"].dropna().plot(kind="hist", bins=args.hist_bins, alpha=0.8)
        plt.xlabel("MAE (seconds)"); plt.ylabel("Count"); plt.title("MAE Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "hist_mae.png"), dpi=160)
        plt.close()
        print("✓ hist_mae.png")
    else:
        print("⚠️  'mae' not found; skipping hist_mae.png")

    # 2) Calibration plot (accuracy vs confidence)
    calib = bin_confidence(df, conf_col="confidence", target_col="piw", n_bins=args.bins)
    if calib is not None:
        plt.figure(figsize=(5.5,5.5))
        # Plot only bins with counts > 0
        c = calib[calib["count"] > 0]
        plt.plot(c["mean_conf"], c["mean_acc"], marker="o")
        lims = [0,1]
        plt.plot(lims, lims, "--", linewidth=1)  # ideal diagonal
        plt.xlim(0,1); plt.ylim(0,1)
        plt.xlabel("Mean confidence (max p(t))")
        plt.ylabel("Accuracy (PiW / Inside)")
        plt.title("Calibration Curve")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "calibration.png"), dpi=160)
        plt.close()
        print("✓ calibration.png")
    else:
        print("⚠️  Could not build calibration (need 'confidence' and a target column like 'piw'/'inside').")

    print("\nDone. Outputs in:", args.out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", default="reports/summary.csv")
    ap.add_argument("--out_dir", default="reports/analysis")
    ap.add_argument("--bins", type=int, default=10, help="bins for calibration")
    ap.add_argument("--hist_bins", type=int, default=40, help="bins for MAE histogram")
    args = ap.parse_args()
    main(args)