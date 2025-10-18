# -*- coding: utf-8 -*-
"""
Temporal localization from sound (frozen MC3 embeddings, AVE dataset)

Inputs
------
Video embeddings: cache/vid_emb/<vid>.npz
  - emb: (Tv, D)
  - centers_sec: (Tv,)
Audio embeddings: cache/aud_emb/<vid>__L<ms>.npz
  - emb: (Ka, D)       # sliding crops across the clip
  - centers_sec: (Ka,)
Annotations: data/ave/ave_annotations.csv
  - columns: ['video_id','event_label','start_s','end_s','split']
  - video_id may contain class and YT id. We robustly extract YT id.

Outputs
-------
reports/summary.csv   # per-clip metrics
reports/curves/*.png  # (optional) similarity + PDF plots
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def softmax(x, tau=0.07):
    x = (x - np.max(x)) / max(tau, 1e-6)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)

def cosine_sim(a, B):
    """a: (D,), B: (T,D) -> (T,)"""
    a = a / (np.linalg.norm(a) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return B @ a

_YT_RE = re.compile(r'([A-Za-z0-9_-]{6,})')

def extract_youtube_id(s):
    """
    Robustly pull a YT-like id (e.g., RUhOCu3LNXM) out of a string.
    Works for 'RUhOCu3LNXM' or 'Church,bell&RUhOCu3LNXM&good&'
    """
    # exact id if file stem
    if re.fullmatch(_YT_RE, s):
        return s
    m = _YT_RE.search(s)
    return m.group(1) if m else s

def pick_audio_query(audio_npz, mode="middle"):
    """Choose one audio embedding vector to use as the query."""
    A = audio_npz["emb"]        # (Ka, D)
    centers = audio_npz["centers_sec"]  # (Ka,)
    if A.shape[0] == 0:
        return None, None
    if mode == "middle":
        mid = np.median(centers)
        idx = int(np.argmin(np.abs(centers - mid)))
    else:
        # default: middle
        mid = np.median(centers)
        idx = int(np.argmin(np.abs(centers - mid)))
    return A[idx], centers[idx]

def load_annotation_map(csv_path):
    """
    Returns dict: {yt_id: (start_s, end_s)}
    """
    df = pd.read_csv(csv_path)
    # Normalize id
    yt = df["video_id"].astype(str).apply(extract_youtube_id)
    df = df.assign(yt_id=yt)
    # Prefer one interval per clip; if duplicated, take the widest interval
    df["len"] = df["end_s"] - df["start_s"]
    df = df.sort_values("len", ascending=False)
    keep = df.drop_duplicates(subset=["yt_id"], keep="first")
    return {r["yt_id"]: (float(r["start_s"]), float(r["end_s"])) for _, r in keep.iterrows()}

def evaluate_hit(pred_t, gt_start, gt_end, deltas=(0.25, 0.5, 1.0)):
    mid = 0.5 * (gt_start + gt_end)
    dist = abs(pred_t - mid)
    res = {f"hit@{d}": float(dist <= d) for d in deltas}
    res["mae_mid"] = dist
    # also: inside-interval?
    res["inside"] = float((pred_t >= gt_start) and (pred_t <= gt_end))
    return res

def plot_curve(out_png, t_vid, sim, pdf, pred_t, gt_start, gt_end, title):
    plt.figure(figsize=(8, 4.5))
    ax1 = plt.gca()
    ax1.plot(t_vid, sim, label="cosine sim")
    ax1.set_xlabel("video time (s)")
    ax1.set_ylabel("similarity")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(t_vid, pdf, linestyle="--", label="PDF (softmax)")
    ax2.set_ylabel("probability")

    # GT interval
    ax1.axvspan(gt_start, gt_end, color="orange", alpha=0.2, label="GT interval")
    # Pred peak
    ax1.axvline(pred_t, color="red", alpha=0.8, linestyle=":", label="pred peak")

    plt.title(title)
    # Combined legend
    lines, labels = [], []
    for ax in (ax1, ax2):
        L = ax.get_legend_handles_labels()
        lines += L[0]; labels += L[1]
    plt.legend(lines, labels, loc="upper right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def main(args):
    os.makedirs(args.curve_dir, exist_ok=True)

    # --- Load GT spans and labels from AVE annotations ---
    # Expect a CSV with columns like: video_id,event_label,start_s,end_s,split
    ann_df = pd.read_csv(args.annotations_csv)
    # Normalize id key from the annotations
    if "video_id" not in ann_df.columns:
        # fallbacks: yt_id or first column
        if "yt_id" in ann_df.columns:
            ann_df = ann_df.rename(columns={"yt_id": "video_id"})
        else:
            ann_df = ann_df.rename(columns={ann_df.columns[0]: "video_id"})
    # Build GT span and label maps (take first label if multiple rows per id)
    gt_map = {}
    label_map = {}
    for vid, sub in ann_df.groupby("video_id"):
        # prefer explicit start/end columns; fallbacks to 'start_s'/'end_s'
        if {"start_s","end_s"}.issubset(sub.columns):
            s, e = float(sub["start_s"].iloc[0]), float(sub["end_s"].iloc[0])
        elif {"gt_start","gt_end"}.issubset(sub.columns):
            s, e = float(sub["gt_start"].iloc[0]), float(sub["gt_end"].iloc[0])
        else:
            # If no segment, assume whole 10s clip (AVE default)
            s, e = 0.0, 10.0
        gt_map[extract_youtube_id(vid)] = (s, e)
        if "event_label" in sub.columns:
            label_map[extract_youtube_id(vid)] = str(sub["event_label"].iloc[0])

    # --- Index available embeddings ---
    vid_npzs = sorted(glob(os.path.join(args.vid_emb_dir, "*.npz")))
    aud_npzs = sorted(glob(os.path.join(args.aud_emb_dir, f"*__L{int(args.L_sec*1000)}ms.npz")))

    vids = {extract_youtube_id(os.path.splitext(os.path.basename(p))[0]): p for p in vid_npzs}
    auds = {}
    for p in aud_npzs:
        stem = os.path.splitext(os.path.basename(p))[0]           # <vid>__Lxxxxms
        vid = stem.split("__L")[0]
        auds[extract_youtube_id(vid)] = p

    keys = sorted(set(vids.keys()) & set(auds.keys()) & set(gt_map.keys()))
    if len(keys) == 0:
        print("[warn] No overlapping keys between embeddings and annotations. Check paths.")
        print(f"vid dir: {args.vid_emb_dir}")
        print(f"aud dir: {args.aud_emb_dir}")
        print(f"ann csv: {args.annotations_csv}")
        return

    print(f"→ Evaluating {len(keys)} clips | τ={args.tau} | L={args.L_sec}s | backend=cosine+softmax")

    rows = []
    n_plotted = 0

    for k in tqdm(keys, desc="Evaluate", unit="clip"):
        vnpz = np.load(vids[k])
        anpz = np.load(auds[k])

        V = vnpz["emb"]              # (Tv, D)
        t_vid = vnpz["centers_sec"]  # (Tv,)
        a_vec, a_center = pick_audio_query(anpz, mode=args.audio_pick)

        if a_vec is None or V.shape[0] == 0:
            continue

        # --- similarity + PDF ---
        sim = cosine_sim(a_vec, V)                   # (Tv,)
        z = sim / float(args.tau)
        z = z - z.max()                              # numerical stability
        p = np.exp(z)
        p = p / (p.sum() + 1e-12)                    # PDF over time

        pred_idx = int(np.argmax(p))
        pred_t = float(t_vid[pred_idx])
        confidence = float(p[pred_idx])
        entropy = float(-(p * (np.log(p + 1e-12))).sum())  # natural-log entropy

        # --- eval vs GT ---
        gt_start, gt_end = gt_map[k]
        metrics = evaluate_hit(pred_t, gt_start, gt_end, deltas=tuple(args.hit_deltas))

        # --- assemble canonical row ---
        row = {
            "video_id": k,                    # canonical id key
            "t_pred_sec": pred_t,             # canonical pred time
            "gt_start_sec": gt_start,         # canonical gt start
            "gt_end_sec": gt_end,             # canonical gt end
            "gt_mid_sec": (gt_start + gt_end) / 2.0,
            "audio_center": float(a_center),
            "tau": float(args.tau),
            "L_sec": float(args.L_sec),
            "confidence": confidence,         # NEW
            "entropy": entropy,               # NEW
            # expand hit metrics to canonical names the analyzer expects
            "hit_0.25": metrics.get("hit@0.25", 0.0),
            "hit_0.5":  metrics.get("hit@0.5",  0.0),
            "hit_1.0":  metrics.get("hit@1.0",  0.0),
            "mae":      metrics.get("mae_mid", abs(pred_t - (gt_start+gt_end)/2.0)),
            "inside":   metrics.get("inside", float(gt_start <= pred_t <= gt_end)),
        }
        # optional label
        if k in label_map:
            row["event_label"] = label_map[k]

        rows.append(row)

        # --- optional plots ---
        if n_plotted < args.plot_n:
            title = f"{k} | pred={pred_t:.2f}s | GT=[{gt_start:.2f},{gt_end:.2f}] | conf={confidence:.2f}"
            out_png = os.path.join(args.curve_dir, f"{k}.png")
            plot_curve(out_png, t_vid, sim, p, pred_t, gt_start, gt_end, title)
            n_plotted += 1

    # --- Save summary ---
    if len(rows) > 0:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
        df.to_csv(args.summary_csv, index=False)

        # quick print aggregate
        agg = {
            "N": len(df),
            "MAE_mean": float(df["mae"].mean()),
            "Inside_mean": float(df["inside"].mean()),
            "Hit@0.25": float(df["hit_0.25"].mean()),
            "Hit@0.5":  float(df["hit_0.5"].mean()),
            "Hit@1.0":  float(df["hit_1.0"].mean()),
            "Conf_mean": float(df["confidence"].mean()),
            "Ent_mean":  float(df["entropy"].mean()),
        }
        print("→ Summary:", agg)
    else:
        print("[warn] No rows evaluated. Did keys match?)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid_emb_dir", default="cache/vid_emb")
    ap.add_argument("--aud_emb_dir", default="cache/aud_emb")
    ap.add_argument("--annotations_csv", default="data/ave/ave_annotations.csv")
    ap.add_argument("--summary_csv", default="reports/summary.csv")
    ap.add_argument("--curve_dir", default="reports/curves")
    ap.add_argument("--L_sec", type=float, default=1.0, help="audio crop length to use (seconds)")
    ap.add_argument("--tau", type=float, default=0.07, help="softmax temperature")
    ap.add_argument("--audio_pick", choices=["middle"], default="middle",
                    help="which audio crop to use as query (default: middle)")
    ap.add_argument("--plot_n", type=int, default=20, help="#examples to plot")
    ap.add_argument("--hit_deltas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    args = ap.parse_args()
    main(args)