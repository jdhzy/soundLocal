#!/usr/bin/env python3
"""
Schema-flexible analyzer for temporal localization results.

Reads:  reports/summary.csv (any reasonable schema)
Writes: reports/analysis/
  - metrics_overall.json
  - metrics_per_class.csv  (if a label column is found)
  - hist_mae.png           (if MAE is computable)
  - calibration.png        (if confidence + target available)

It will derive missing metrics (MAE, Inside, Hit@δ, PiW) if it finds:
- predicted time:  t_pred_sec | t_hat | pred_sec | pred_time_sec
- GT span:         (gt_start_sec|start_s|start_sec) &
                   (gt_end_sec|end_s|end_sec)
  or GT midpoint:  gt_mid_sec
- confidence:      confidence | peak_prob | max_prob
"""

import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
def ensure_outdir(path): os.makedirs(path, exist_ok=True)
def safe_mean(x): 
    s = pd.Series(x).dropna()
    return float(s.mean()) if len(s) else float("nan")

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

def derive_metrics(df):
    cols = set(df.columns)

    # pick fields
    pred_c = pick_col(df, ["t_pred_sec","t_hat","pred_sec","pred_time_sec"])
    start_c = pick_col(df, ["gt_start_sec","start_s","start_sec"])
    end_c   = pick_col(df, ["gt_end_sec","end_s","end_sec"])
    mid_c   = pick_col(df, ["gt_mid_sec"])

    # derive midpoint if needed
    if mid_c is None and start_c and end_c:
        df["gt_mid_sec"] = (df[start_c] + df[end_c]) / 2.0
        mid_c = "gt_mid_sec"

    # derive MAE if possible
    if "mae" not in cols and pred_c and mid_c:
        df["mae"] = (df[pred_c] - df[mid_c]).abs()

    # derive Inside and PiW if possible
    if start_c and end_c and pred_c:
        inside = (df[pred_c] >= df[start_c]) & (df[pred_c] <= df[end_c])
        df["inside"] = inside.astype(float)
        # Peak-in-Window: same as inside for single-peak prediction
        if "piw" not in cols: df["piw"] = df["inside"]

    # derive Hit@δ at common thresholds if possible
    if pred_c and mid_c:
        for delta in [0.25, 0.5, 1.0]:
            col = f"hit_{delta}"
            if col not in cols:
                df[col] = (df[pred_c] - df[mid_c]).abs() <= delta
                df[col] = df[col].astype(float)

    # normalize confidence name if present
    conf_c = pick_col(df, ["confidence","peak_prob","max_prob"])
    if conf_c and "confidence" not in cols:
        df["confidence"] = df[conf_c]

    # normalize label column if present
    label_c = pick_col(df, ["event_label","label","class","category"])
    if label_c and label_c != "event_label":
        df = df.rename(columns={label_c: "event_label"})

    return df

def bin_confidence(df, conf_col="confidence", target_cols=("piw","inside","hit_0.5"), n_bins=10):
    if conf_col not in df.columns: return None
    tgt = None
    for t in target_cols:
        if t in df.columns:
            tgt = t; break
    if tgt is None: return None

    valid = df[[conf_col, tgt]].dropna()
    if valid.empty: return None
    conf = valid[conf_col].clip(0,1).astype(float)
    acc  = valid[tgt].astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, bins, right=True)
    rows = []
    for b in range(1, n_bins+1):
        mask = idx == b
        if mask.sum() == 0:
            rows.append({"bin":b,"bin_left":bins[b-1],"bin_right":bins[b],
                         "mean_conf":np.nan,"mean_acc":np.nan,"count":0})
        else:
            rows.append({"bin":b,"bin_left":bins[b-1],"bin_right":bins[b],
                         "mean_conf":float(conf[mask].mean()),
                         "mean_acc":float(acc[mask].mean()),
                         "count":int(mask.sum())})
    return pd.DataFrame(rows)

# ---------- main ----------
def main(args):
    ensure_outdir(args.out_dir)
    df = pd.read_csv(args.summary_csv)

    # normalize hit column names if someone used hit@0.5 style
    rename_map = {}
    for src, dst in [("hit@0.25","hit_0.25"), ("hit@0.5","hit_0.5"), ("hit@1.0","hit_1.0")]:
        if src in df.columns: rename_map[src] = dst
    if rename_map: df = df.rename(columns=rename_map)

    # derive what’s missing
    df = derive_metrics(df)
    cols = set(df.columns)

    # ---- overall metrics
    overall = {"num_clips": int(len(df))}
    if "mae" in cols:
        overall["mae_mean"]   = safe_mean(df["mae"])
        overall["mae_median"] = float(df["mae"].median())
    for k in ["rmse","inside","hit_0.25","hit_0.5","hit_1.0","piw","confidence","entropy","migt"]:
        if k in cols:
            overall[f"{k}_mean"] = safe_mean(df[k])

    with open(os.path.join(args.out_dir, "metrics_overall.json"), "w") as f:
        json.dump(overall, f, indent=2)
    print("✓ metrics_overall.json")

    # ---- per-class table
    if "event_label" in cols:
        g = df.groupby("event_label")
        rows = []
        for label, sub in g:
            row = {"event_label": label, "count": int(len(sub))}
            for k in ["mae","rmse","inside","hit_0.25","hit_0.5","hit_1.0","piw","confidence","entropy","migt"]:
                if k in sub.columns:
                    row[f"{k}_mean"] = safe_mean(sub[k])
            rows.append(row)
        pd.DataFrame(rows).sort_values("event_label").to_csv(
            os.path.join(args.out_dir, "metrics_per_class.csv"), index=False)
        print("✓ metrics_per_class.csv")
    else:
        print("⚠️  no label column found; skipping per-class table.")

    # ---- plots
    if "mae" in cols:
        plt.figure(figsize=(6,4))
        df["mae"].dropna().plot(kind="hist", bins=args.hist_bins, alpha=0.85)
        plt.xlabel("MAE (seconds)"); plt.ylabel("Count"); plt.title("MAE Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "hist_mae.png"), dpi=160); plt.close()
        print("✓ hist_mae.png")
    else:
        print("⚠️  MAE not available; skipping hist_mae.png")

    calib = bin_confidence(df, conf_col="confidence", target_cols=("piw","inside","hit_0.5"), n_bins=args.bins)
    if calib is not None:
        c = calib[calib["count"] > 0]
        plt.figure(figsize=(5.5,5.5))
        plt.plot(c["mean_conf"], c["mean_acc"], marker="o")
        plt.plot([0,1],[0,1],"--",linewidth=1)
        plt.xlim(0,1); plt.ylim(0,1)
        plt.xlabel("Mean confidence"); plt.ylabel("Accuracy (PiW/Inside)")
        plt.title("Calibration Curve")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "calibration.png"), dpi=160); plt.close()
        print("✓ calibration.png")
    else:
        print("⚠️  calibration not available (need confidence + target).")

    print("\nDone. Outputs in:", args.out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", default="reports/summary.csv")
    ap.add_argument("--out_dir", default="reports/analysis")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--hist_bins", type=int, default=40)
    args = ap.parse_args()
    main(args)