# src/analyze_results.py
import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

# ---- Headless-safe plotting setup ----
HAVE_PLT = False
try:
    import matplotlib  # noqa: F401
    # If backend not specified, use non-interactive one for clusters
    if "MPLBACKEND" not in os.environ:
        import matplotlib as _mpl  # type: ignore
        _mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


# -------- Column alias handling --------
ALIASES: Dict[str, List[str]] = {
    "video_id":      ["video_id", "yt_id", "vid", "id"],
    "t_pred_sec":    ["t_pred_sec", "pred_t", "pred_time_sec", "t_hat"],
    "gt_start_sec":  ["gt_start_sec", "start_s", "gt_start"],
    "gt_end_sec":    ["gt_end_sec", "end_s", "gt_end"],
    "gt_mid_sec":    ["gt_mid_sec", "mid_s", "midpoint"],
    "mae":           ["mae", "mae_mid"],
    "inside":        ["inside", "piw", "inside_interval"],
    "event_label":   ["event_label", "label", "class"],
    "confidence":    ["confidence", "peak_prob", "p_peak"],
    # Hit@δ handled separately below
}

HIT_KEYS = [
    ("hit_0.25", ["hit_0.25", "hit@0.25"]),
    ("hit_0.5",  ["hit_0.5",  "hit@0.5"]),
    ("hit_1.0",  ["hit_1.0",  "hit@1.0"]),
]


def rename_by_alias(df: pd.DataFrame) -> pd.DataFrame:
    # direct aliases
    for canonical, cands in ALIASES.items():
        for c in cands:
            if c in df.columns and canonical not in df.columns:
                df = df.rename(columns={c: canonical})
                break
    # hits
    for canonical, cands in HIT_KEYS:
        for c in cands:
            if c in df.columns and canonical not in df.columns:
                df = df.rename(columns={c: canonical})
                break
    return df


def derive_missing(df: pd.DataFrame) -> pd.DataFrame:
    # compute midpoint if we have start/end
    if "gt_mid_sec" not in df.columns and {"gt_start_sec", "gt_end_sec"}.issubset(df.columns):
        df["gt_mid_sec"] = (df["gt_start_sec"].astype(float) + df["gt_end_sec"].astype(float)) / 2.0

    # mae if we have pred and mid
    if "mae" not in df.columns and {"t_pred_sec", "gt_mid_sec"}.issubset(df.columns):
        df["mae"] = (df["t_pred_sec"].astype(float) - df["gt_mid_sec"].astype(float)).abs()

    # inside if we have span + pred
    if "inside" not in df.columns and {"gt_start_sec", "gt_end_sec", "t_pred_sec"}.issubset(df.columns):
        t = df["t_pred_sec"].astype(float)
        s = df["gt_start_sec"].astype(float)
        e = df["gt_end_sec"].astype(float)
        df["inside"] = ((t >= s) & (t <= e)).astype(float)

    # Hit@δ from mae
    for name, delta in [("hit_0.25", 0.25), ("hit_0.5", 0.5), ("hit_1.0", 1.0)]:
        if name not in df.columns and "mae" in df.columns:
            df[name] = (df["mae"].astype(float) <= delta).astype(float)
    return df


def safe_mean(x: pd.Series) -> float:
    if x is None or len(x) == 0:
        return float("nan")
    return float(np.nanmean(pd.to_numeric(x, errors="coerce")))


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_hist_mae(df: pd.DataFrame, out_dir: str, bins: int):
    if not HAVE_PLT or "mae" not in df.columns:
        print("⚠️  MAE not available or plotting disabled; skipping hist.")
        return
    vals = pd.to_numeric(df["mae"], errors="coerce").dropna().values
    if vals.size == 0:
        print("⚠️  No valid MAE values; skipping hist.")
        return
    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=bins)
    plt.xlabel("MAE (sec)")
    plt.ylabel("Count")
    plt.title("MAE distribution")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "hist_mae.png"))
    plt.close()
    print("✓ hist_mae.png")


def calibration_curve(df: pd.DataFrame, out_dir: str, bins: int):
    if not HAVE_PLT or "confidence" not in df.columns:
        print("⚠️  no 'confidence'; skipping calibration.")
        return

    target_col = "inside" if "inside" in df.columns else ("hit_0.5" if "hit_0.5" in df.columns else None)
    if target_col is None:
        print("⚠️  no target column ('inside' or 'hit_0.5'); skipping calibration.")
        return

    conf = pd.to_numeric(df["confidence"], errors="coerce").clip(0, 1)
    target = pd.to_numeric(df[target_col], errors="coerce")
    mask = conf.notna() & target.notna()
    conf, target = conf[mask].values, target[mask].values
    if conf.size == 0:
        print("⚠️  No valid rows for calibration.")
        return

    bins_edges = np.linspace(0, 1, bins + 1)
    inds = np.digitize(conf, bins_edges, right=False) - 1
    acc, conf_avg, counts = [], [], []
    for b in range(bins):
        sel = inds == b
        if not np.any(sel):
            acc.append(np.nan); conf_avg.append(np.nan); counts.append(0)
        else:
            acc.append(float(np.mean(target[sel])))
            conf_avg.append(float(np.mean(conf[sel])))
            counts.append(int(np.sum(sel)))

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Ideal")
    plt.plot(conf_avg, acc, marker="o", label="Empirical")
    plt.xlabel("Confidence (bin mean)")
    plt.ylabel(f"Accuracy ({target_col})")
    plt.title("Calibration")
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "calibration.png"))
    plt.close()

    save_json(
        {"bin_conf_mean": conf_avg, "bin_acc": acc, "bin_counts": counts, "target": target_col},
        os.path.join(out_dir, "calibration_bins.json"),
    )
    print("✓ calibration.png")


def per_class_table(df: pd.DataFrame, out_dir: str):
    if "event_label" not in df.columns:
        print("⚠️  no label column; skipping per-class table.")
        return

    cols = [c for c in ["mae", "inside", "hit_0.25", "hit_0.5", "hit_1.0"] if c in df.columns]
    if not cols:
        print("⚠️  no metrics to group; skipping per-class table.")
        return

    # Ensure numeric for aggregation
    dfg = df.copy()
    for c in cols:
        dfg[c] = pd.to_numeric(dfg[c], errors="coerce")

    tab = (
        dfg.groupby("event_label")[cols]
        .mean()
        .reset_index()
        .sort_values(by=("mae" if "mae" in cols else cols[0]))
    )
    os.makedirs(out_dir, exist_ok=True)
    tab.to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)
    print("✓ per_class_metrics.csv")


import datetime


def main(args):
    base_is_analysis = args.out_dir == "reports/analysis" or args.out_dir.endswith("/analysis")

    if base_is_analysis:
        # use tag if provided, otherwise fallback to timestamp
        if args.run_tag:
            folder = args.run_tag
        else:
            folder = datetime.datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        args.out_dir = os.path.join(args.out_dir, folder)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"→ Analysis outputs will be saved to: {args.out_dir}")

    df = pd.read_csv(args.summary_csv)
    print("Columns (raw):", list(df.columns))

    df = rename_by_alias(df)
    df = derive_missing(df)
    print("Columns (canonicalized):", list(df.columns))

    # Overall metrics
    metrics = {}
    if "mae" in df.columns:
        mae_vals = pd.to_numeric(df["mae"], errors="coerce")
        metrics["mae_mean"] = float(np.nanmean(mae_vals))
        metrics["mae_median"] = float(np.nanmedian(mae_vals))
    if "inside" in df.columns:
        metrics["inside_mean"] = safe_mean(df["inside"])

    for k, _ in HIT_KEYS:
        if k in df.columns:
            metrics[f"{k}_mean"] = safe_mean(df[k])

    os.makedirs(args.out_dir, exist_ok=True)
    save_json(metrics, os.path.join(args.out_dir, "metrics_overall.json"))
    print("✓ metrics_overall.json")

    # Plots / tables
    plot_hist_mae(df, args.out_dir, args.hist_bins)
    calibration_curve(df, args.out_dir, args.bins)
    per_class_table(df, args.out_dir)

    # Save a canonicalized copy
    df.to_csv(os.path.join(args.out_dir, "summary_canonical.csv"), index=False)
    print(f"Done. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", default="reports/summary.csv")
    ap.add_argument("--out_dir", default="reports/analysis")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--hist_bins", type=int, default=40)
    ap.add_argument("--run_tag", default="", help="Optional name for this run (e.g., 'tau007_multi').")
    args = ap.parse_args()
    main(args)