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
import math

def zscore(x: np.ndarray) -> np.ndarray:
    m, s = float(x.mean()), float(x.std() + 1e-8)
    return (x - m) / s

def gaussian_smooth(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if sigma <= 0: 
        return x
    # 1D gaussian kernel (odd size ≈ 6σ+1)
    k = int(max(3, 2*round(3*sigma)+1))
    r = k // 2
    grid = np.arange(-r, r+1)
    ker = np.exp(-(grid**2)/(2*sigma**2))
    ker = (ker / ker.sum()).astype(np.float32)
    return np.convolve(x, ker, mode="same")

def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    x = (x / max(tau, 1e-6)).astype(np.float32)
    x -= x.max()
    p = np.exp(x); p /= p.sum() + 1e-8
    return p

def soft_argmax(times: np.ndarray, p: np.ndarray) -> float:
    # return E[t] under PDF p(t)
    return float((times * p).sum())

def entropy(p: np.ndarray) -> float:
    q = p + 1e-12
    return float(-(q*np.log(q)).sum())

def adapt_tau(scores: np.ndarray, target_H: float = 3.0, 
              tau_min: float = 0.02, tau_max: float = 0.2, 
              iters: int = 10) -> float:
    """Binary search tau so that entropy(softmax(scores/tau)) ≈ target_H."""
    lo, hi = tau_min, tau_max
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        H = entropy(softmax(scores, mid))
        if H > target_H:
            # too flat → decrease tau
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)


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

def pick_audio_query(audio_npz, mode="middle", offsets=None, reduce="mean"):
    """
    Choose one or multiple audio embeddings to use as query.

    Parameters
    ----------
    audio_npz : np.load result
        Contains "emb" (K, D) and "centers_sec" (K,).
    mode : {"middle", "multi"}
        "middle"  → single center crop (old behavior)
        "multi"   → multiple offsets around center
    offsets : list[float]
        Offsets (in seconds) from the center to sample for multi-crop mode.
        Example: [0, 0.25, -0.25]
    reduce : {"mean", "max"}
        How to fuse multiple crops if using mode="multi".

    Returns
    -------
    (np.ndarray, float) or (List[np.ndarray], float)
    """
    A = audio_npz["emb"]                # (Ka, D)
    centers = audio_npz["centers_sec"]  # (Ka,)
    if A.shape[0] == 0:
        return None, None

    mid = np.median(centers)

    if mode == "middle" or offsets is None:
        idx = int(np.argmin(np.abs(centers - mid)))
        return A[idx], centers[idx]

    # multi-crop mode
    idxs = []
    for off in offsets:
        target = mid + off
        idx = int(np.argmin(np.abs(centers - target)))
        idx = np.clip(idx, 0, len(centers) - 1)
        idxs.append(idx)

    A_sel = A[idxs]  # (n, D)

    if reduce == "mean":
        A_fused = A_sel.mean(axis=0)
    elif reduce == "max":
        A_fused = A_sel.max(axis=0)
    else:
        raise ValueError(f"Unknown reduce mode: {reduce}")

    return A_fused, mid

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
    import numpy as np
    import pandas as pd
    import os
    from glob import glob
    from tqdm import tqdm

    # ------- small helpers (local, no extra imports) -------
    def cosine_sim(a_vec, V):
        # V: (Tv, D), a_vec: (D,)
        return (V @ a_vec) / (np.linalg.norm(V, axis=1) + 1e-8)

    def softmax(sim, tau):
        z = sim / float(tau)
        z -= z.max()
        p = np.exp(z)
        p /= (p.sum() + 1e-12)
        return p

    def entropy(p):
        return float(-(p * np.log(p + 1e-12)).sum())

    def soft_argmax(t_vid, p):
        # expectation of time under PDF
        return float((t_vid * p).sum())

    def zscore(x):
        m, s = x.mean(), x.std() + 1e-8
        return (x - m) / s

    def gaussian_smooth(x, sigma):
        if sigma <= 0:
            return x
        # lightweight 1D Gaussian (no SciPy needed)
        # kernel size ≈ 6σ+1, clipped odd
        k = max(3, int(6 * sigma) | 1)
        center = k // 2
        grid = np.arange(k) - center
        g = np.exp(-(grid ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return np.convolve(x, g, mode="same")

    def adapt_tau(sim, target_H=3.5, iters=20, lo=1e-3, hi=1.0):
        # binary search τ to match desired entropy
        for _ in range(iters):
            mid = (lo + hi) / 2.0
            p = softmax(sim, tau=mid)
            H = entropy(p)
            if H > target_H:
                # too flat → lower τ
                lo, hi = lo, mid
            else:
                # too peaky → raise τ
                lo, hi = mid, hi
        return (lo + hi) / 2.0

    def pick_audio_single(anpz, mode="middle"):
        A = anpz["emb"]               # (Ka, D)
        centers = anpz["centers_sec"] # (Ka,)
        if A.shape[0] == 0:
            return None, None
        if mode == "middle":
            mid = np.median(centers)
            idx = int(np.argmin(np.abs(centers - mid)))
        else:
            # fallback to middle
            mid = np.median(centers)
            idx = int(np.argmin(np.abs(centers - mid)))
        return A[idx], float(centers[idx])

    def pick_audio_multi(anpz, offsets, reduce="mean"):
        A = anpz["emb"]
        centers = anpz["centers_sec"]
        if A.shape[0] == 0:
            return [], None
        chosen_vecs, chosen_centers = [], []
        for off in offsets:
            idx = int(np.argmin(np.abs(centers - off)))
            chosen_vecs.append(A[idx])
            chosen_centers.append(centers[idx])
        # representative center (for logging)
        a_center = float(np.mean(chosen_centers)) if len(chosen_centers) else None
        return chosen_vecs, a_center

    # -------------------------------------------------------

    os.makedirs(args.curve_dir, exist_ok=True)

    # --- Load GT spans + (optional) labels from AVE annotations ---
    ann_df = pd.read_csv(args.annotations_csv)
    if "video_id" not in ann_df.columns:
        if "yt_id" in ann_df.columns:
            ann_df = ann_df.rename(columns={"yt_id": "video_id"})
        else:
            ann_df = ann_df.rename(columns={ann_df.columns[0]: "video_id"})

    gt_map, label_map = {}, {}
    for vid, sub in ann_df.groupby("video_id"):
        if {"start_s", "end_s"}.issubset(sub.columns):
            s, e = float(sub["start_s"].iloc[0]), float(sub["end_s"].iloc[0])
        elif {"gt_start", "gt_end"}.issubset(sub.columns):
            s, e = float(sub["gt_start"].iloc[0]), float(sub["gt_end"].iloc[0])
        else:
            s, e = 0.0, 10.0  # default AVE 10 s clip
        key = extract_youtube_id(vid)
        gt_map[key] = (s, e)
        if "event_label" in sub.columns:
            label_map[key] = str(sub["event_label"].iloc[0])

    # --- Index available embeddings ---
    vid_npzs = sorted(glob(os.path.join(args.vid_emb_dir, "*.npz")))
    aud_npzs = sorted(glob(os.path.join(args.aud_emb_dir, f"*__L{int(args.L_sec*1000)}ms.npz")))

    vids = {extract_youtube_id(os.path.splitext(os.path.basename(p))[0]): p for p in vid_npzs}
    auds = {}
    for p in aud_npzs:
        stem = os.path.splitext(os.path.basename(p))[0]   # <vid>__Lxxxxms
        vid = stem.split("__L")[0]
        auds[extract_youtube_id(vid)] = p

    keys = sorted(set(vids.keys()) & set(auds.keys()) & set(gt_map.keys()))
    if len(keys) == 0:
        print("[warn] No overlapping keys between embeddings and annotations. Check paths.")
        print(f"vid dir: {args.vid_emb_dir}")
        print(f"aud dir: {args.aud_emb_dir}")
        print(f"ann csv: {args.annotations_csv}")
        raise SystemExit

    print(f"→ Evaluating {len(keys)} clips | τ={args.tau} | L={args.L_sec}s | audio_pick={args.audio_pick}")

    # Parse multi-audio relative offsets (seconds) once (used only if audio_pick=multi)
    rel_offsets = []
    if args.audio_pick == "multi":
        rel_offsets = [float(x) for x in args.multi_offsets.split(",") if x.strip()]

    rows, n_plotted = [], 0

    for k in tqdm(keys, desc="Evaluate", unit="clip"):
        vnpz = np.load(vids[k])
        anpz = np.load(auds[k])

        V = vnpz["emb"].astype(np.float32)                # (Tv, D)
        t_vid = vnpz["centers_sec"].astype(np.float32)    # (Tv,)
        if V.shape[0] == 0:
            continue

        # --- choose audio query (single or multi) ---
        if args.audio_pick == "multi":
            centers = anpz["centers_sec"].astype(np.float32)
            base = float(np.median(centers)) if centers.size > 0 else 5.0
            abs_offsets = [base + r for r in rel_offsets]  # relative -> absolute
            a_list, a_center = pick_audio_multi(anpz, offsets=abs_offsets, reduce=args.multi_reduce)
            # a_list: list of (D,) vectors; a_center: float (mean of chosen centers)
        else:
            a_vec, a_center = pick_audio_single(anpz, mode="middle")
            a_list = [a_vec] if a_vec is not None else []

        if len(a_list) == 0:
            continue

        # --- similarity → PDF (per-crop) → fuse ---
        pdfs, sims_for_plot = [], []
        for a_vec in a_list:
            if a_vec is None:
                continue
            sim = cosine_sim(a_vec, V)  # (Tv,)
            if args.score_zscore:
                sim = zscore(sim)
            if args.score_smooth_sigma > 0:
                sim = gaussian_smooth(sim, sigma=args.score_smooth_sigma)

            tau_use = adapt_tau(sim, target_H=args.tau_adapt) if args.tau_adapt > 0 else args.tau
            p = softmax(sim, tau=tau_use)

            sims_for_plot.append(sim)
            pdfs.append(p)

        if len(pdfs) == 0:
            continue

        pdfs = np.stack(pdfs, 0)  # (K, Tv)
        p_fused = pdfs.max(axis=0) if args.multi_reduce == "max" else pdfs.mean(axis=0)
        sim_plot = np.stack(sims_for_plot, 0).mean(axis=0) if len(sims_for_plot) else p_fused

        # prediction (hard or soft)
        if args.pred_softargmax:
            pred_t = soft_argmax(t_vid, p_fused)
            pred_idx = int(np.argmax(p_fused))
        else:
            pred_idx = int(np.argmax(p_fused))
            pred_t = float(t_vid[pred_idx])

        confidence = float(p_fused[pred_idx])
        ent = entropy(p_fused)

        # --- evaluate vs GT ---
        gt_start, gt_end = gt_map[k]
        metrics = evaluate_hit(pred_t, gt_start, gt_end, deltas=tuple(args.hit_deltas))

        # --- row (canonical + legacy for downstream) ---
        row = {
            # canonical columns
            "video_id": k,
            "t_pred_sec": pred_t,
            "gt_start_sec": gt_start,
            "gt_end_sec": gt_end,
            "gt_mid_sec": (gt_start + gt_end) / 2.0,
            "audio_center": float(a_center) if a_center is not None else float("nan"),
            "tau": (float(args.tau) if args.tau_adapt <= 0 else float("nan")),
            "tau_adapt": float(args.tau_adapt),
            "L_sec": float(args.L_sec),
            "confidence": confidence,
            "entropy": ent,
            "hit_0.25": metrics.get("hit@0.25", 0.0),
            "hit_0.5":  metrics.get("hit@0.5",  0.0),
            "hit_1.0":  metrics.get("hit@1.0",  0.0),
            "mae":      metrics.get("mae_mid", abs(pred_t - (gt_start + gt_end)/2.0)),
            "inside":   metrics.get("inside", float(gt_start <= pred_t <= gt_end)),
            # legacy names (kept for analyzer compatibility)
            "yt_id": k,
            "pred_t": pred_t,
            "gt_start": gt_start,
            "gt_end": gt_end,
            "conf": confidence,
            **metrics,
        }
        if k in label_map:
            row["event_label"] = label_map[k]
        rows.append(row)

        # --- optional plot ---
        if n_plotted < args.plot_n:
            title = f"{k} | pred={pred_t:.2f}s | GT=[{gt_start:.2f},{gt_end:.2f}] | conf={confidence:.2f}"
            out_png = os.path.join(args.curve_dir, f"{k}.png")
            plot_curve(out_png, t_vid, sim_plot, p_fused, pred_t, gt_start, gt_end, title)
            n_plotted += 1

    # --- Save summary CSV and print aggregate ---
    if len(rows) == 0:
        print("[warn] No rows evaluated. Did keys match?")
        return

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
    df.to_csv(args.summary_csv, index=False)

    agg = {
        "N": len(df),
        "MAE_mean": float(df["mae"].mean()),
        "Inside_mean": float(df["inside"].mean()),
        "Hit@0.25": float(df["hit_0.25"].mean()) if "hit_0.25" in df.columns else float("nan"),
        "Hit@0.5":  float(df["hit_0.5"].mean())  if "hit_0.5"  in df.columns else float("nan"),
        "Hit@1.0":  float(df["hit_1.0"].mean())  if "hit_1.0"  in df.columns else float("nan"),
        "Conf_mean": float(df["confidence"].mean()) if "confidence" in df.columns else float("nan"),
        "Ent_mean":  float(df["entropy"].mean())    if "entropy"    in df.columns else float("nan"),
    }
    print("→ Summary:", agg)

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
    ap.add_argument("--log_sim_stats", action="store_true",
                help="Write per-clip similarity diagnostics to CSV")
    ap.add_argument("--sim_log_csv", default="reports/sim_stats.csv",
                    help="Path to write similarity diagnostics CSV")
    ap.add_argument("--print_sim_every", type=int, default=0,
                    help="If >0, print a sample of similarity stats every N clips")

    # I/O
    ap.add_argument("--vid_emb_dir", default="cache/vid_emb")
    ap.add_argument("--aud_emb_dir", default="cache/aud_emb")
    ap.add_argument("--annotations_csv", default="data/ave/ave_annotations.csv")
    ap.add_argument("--summary_csv", default="reports/summary.csv")
    ap.add_argument("--curve_dir", default="reports/curves")

    # Core hyperparams
    ap.add_argument("--L_sec", type=float, default=1.0, help="Audio crop length (seconds)")
    ap.add_argument("--tau", type=float, default=0.07, help="Softmax temperature")

    # Audio query strategy
    ap.add_argument("--audio_pick", choices=["middle", "multi"], default="middle",
                    help="Use single middle crop or fuse multiple audio crops.")
    ap.add_argument("--multi_offsets", type=str, default="0,0.25,-0.25",
                    help="For --audio_pick=multi: RELATIVE seconds around the clip middle "
                         "(e.g., '0,0.25,-0.25'). These are added to the median audio-center.")
    ap.add_argument("--multi_reduce", choices=["mean", "max"], default="mean",
                    help="How to fuse per-crop PDFs for multi: mean or max.")

    # Score shaping & prediction
    ap.add_argument("--score_zscore", action="store_true",
                    help="Z-score the similarity per clip before softmax.")
    ap.add_argument("--score_smooth_sigma", type=float, default=0.0,
                    help="Gaussian smoothing sigma (frames) for similarity; 0 disables.")
    ap.add_argument("--pred_softargmax", action="store_true",
                    help="Use soft-argmax of the fused PDF (expectation of time).")
    ap.add_argument("--tau_adapt", type=float, default=0.0,
                    help="If >0, adapt τ per-clip to reach this entropy (ignores --tau).")

    # Plotting & evaluation
    ap.add_argument("--plot_n", type=int, default=20, help="How many examples to plot.")
    ap.add_argument("--hit_deltas", type=float, nargs="+", default=[0.25, 0.5, 1.0],
                    help="Hit@δ thresholds in seconds.")

    args = ap.parse_args()
    main(args)