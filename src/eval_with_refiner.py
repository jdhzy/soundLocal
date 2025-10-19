#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, json
from glob import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# ------------------ headless-safe plotting ------------------
HAVE_PLT = False
try:
    import matplotlib  # noqa
    if "MPLBACKEND" not in os.environ:
        import matplotlib as _mpl  # noqa
        _mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def plot_curve_dual(path, t, sim, p_base, p_ref, pred_base_t, pred_ref_t, gt_s, gt_e, title=""):
    if not HAVE_PLT:
        return
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(t, sim, lw=1.0, alpha=0.7, label="similarity", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(t, p_base, lw=1.2, alpha=0.95, label="pdf (base)", color="C1")
    ax2.plot(t, p_ref,  lw=1.2, alpha=0.95, label="pdf (refined)", color="C2")

    ax2.axvline(pred_base_t, color="C1", ls="--", lw=1, alpha=0.9)
    ax2.axvline(pred_ref_t,  color="C2", ls="--", lw=1, alpha=0.9)
    ax1.axvspan(gt_s, gt_e, color="k", alpha=0.08, label="GT span")

    ax1.set_xlabel("time (s)"); ax1.set_ylabel("cos sim"); ax2.set_ylabel("probability")
    ax1.set_title(title)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(path))
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()

# ------------------ numerics/helpers ------------------
def l2normalize(x: np.ndarray, axis=-1, eps=1e-8):
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n

def cosine_sim(a: np.ndarray, V: np.ndarray) -> np.ndarray:
    a = l2normalize(a.astype(np.float32), axis=0)
    V = l2normalize(V.astype(np.float32), axis=1)
    return (V @ a).astype(np.float32)

def zscore(x: np.ndarray, eps=1e-6):
    m, s = x.mean(), x.std()
    s = s if s > eps else eps
    return ((x - m) / s).astype(np.float32)

def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x.astype(np.float32)
    radius = max(1, int(round(3 * sigma)))
    t = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(t**2) / (2 * sigma * sigma))
    k = (k / (k.sum() + 1e-8)).astype(np.float32)
    return np.convolve(x.astype(np.float32), k, mode="same")

def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    z = x.astype(np.float32) / max(tau, 1e-6)
    z = z - z.max()
    p = np.exp(z)
    p = p / (p.sum() + 1e-12)
    return p.astype(np.float32)

def entropy(p: np.ndarray) -> float:
    p = np.clip(p.astype(np.float32), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

def soft_argmax(t: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p.astype(np.float32), 1e-12, 1.0)
    p = p / p.sum()
    return float((t * p).sum())

def adapt_tau(sim: np.ndarray, target_H: float, tol=1e-3, maxit=20) -> float:
    lo, hi = 1e-3, 10.0
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        H = entropy(softmax(sim, tau=mid))
        if H > target_H:
            hi = mid
        else:
            lo = mid
        if abs(H - target_H) < tol:
            break
    return float(0.5 * (lo + hi))

def resample_from_to(y: np.ndarray, t_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    u_src = (t_src - t_src.min()) / max(1e-8, (t_src.max() - t_src.min()))
    u_tgt = (t_tgt - t_tgt.min()) / max(1e-8, (t_tgt.max() - t_tgt.min()))
    y_tgt = np.interp(u_tgt, u_src, y).astype(np.float32)
    s = float(y_tgt.sum())
    if s > 0:
        y_tgt /= s
    return y_tgt

def resample_to_len(y: np.ndarray, t_src: np.ndarray, out_len: int) -> Tuple[np.ndarray, np.ndarray]:
    u_src = (t_src - t_src.min()) / max(1e-8, (t_src.max() - t_src.min()))
    u_tgt = np.linspace(0.0, 1.0, out_len, dtype=np.float32)
    y_tgt = np.interp(u_tgt, u_src, y).astype(np.float32)
    s = float(y_tgt.sum())
    if s > 0:
        y_tgt /= s
    t_tgt = np.linspace(t_src.min(), t_src.max(), out_len, dtype=np.float32)
    return y_tgt, t_tgt

def evaluate_hit(t_hat: float, gs: float, ge: float, deltas=(0.25, 0.5, 1.0)):
    mid = 0.5 * (gs + ge)
    out = {"mae_mid": abs(t_hat - mid), "inside": float(gs <= t_hat <= ge)}
    for d in deltas:
        out[f"hit@{d}"] = float(abs(t_hat - mid) <= d)
    return out

# ------------------ audio picking ------------------
def pick_audio_single(a_npz) -> Tuple[np.ndarray, float]:
    A = a_npz["emb"].astype(np.float32)
    centers = a_npz["centers_sec"].astype(np.float32)
    if A.shape[0] == 0:
        return None, None
    mid = float(np.median(centers))
    idx = int(np.argmin(np.abs(centers - mid)))
    return A[idx], float(centers[idx])

def pick_audio_multi(a_npz, offsets: List[float]) -> Tuple[List[np.ndarray], float]:
    A = a_npz["emb"].astype(np.float32)
    centers = a_npz["centers_sec"].astype(np.float32)
    if A.shape[0] == 0:
        return [], None
    mid = float(np.median(centers))
    want = np.array([mid + d for d in offsets], dtype=np.float32)
    idx = np.clip(np.searchsorted(centers, want), 0, max(0, len(centers) - 1))
    vecs = [A[i] for i in idx]
    return vecs, float(mid)

# ------------------ tiny refiner ------------------
class Refiner(nn.Module):
    def __init__(self, D=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, D),
        )
        self.D = D

    def forward(self, x):
        x = x.float()
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

def load_refiner_from_ckpt(path: str, device: str, D_fallback: int) -> Refiner:
    # try to discover D from checkpoint
    D = D_fallback
    try:
        raw = torch.load(path, map_location=device)
        if isinstance(raw, dict) and "D" in raw:
            D = int(raw["D"])
    except Exception:
        pass

    model = Refiner(D=D).to(device).float().eval()

    # robust state_dict loading
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)  # torch>=2.4 recommends this
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    state_dict = None
    if isinstance(ckpt, nn.Module):
        state_dict = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict")
        if state_dict is None and all(isinstance(k, str) for k in ckpt.keys()):
            state_dict = ckpt
    if state_dict is None:
        # last fallback
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, nn.Module):
            state_dict = ckpt.state_dict()
        elif isinstance(ckpt, dict):
            state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict")

    if state_dict is None:
        raise RuntimeError(f"Unrecognized checkpoint format at {path}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"⚠️  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    return model

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid_emb_dir", default="cache/vid_emb")
    ap.add_argument("--aud_emb_dir", default="cache/aud_emb")
    ap.add_argument("--annotations_csv", default="data/ave/ave_annotations.csv")
    ap.add_argument("--summary_csv", default="reports/summary_refiner.csv")
    ap.add_argument("--curve_dir", default="reports/curves_refiner")

    ap.add_argument("--L_sec", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--audio_pick", choices=["middle", "multi"], default="multi")
    ap.add_argument("--multi_offsets", type=str, default="0,0.25,-0.25")
    ap.add_argument("--multi_reduce", choices=["mean", "max"], default="mean")
    ap.add_argument("--score_zscore", action="store_true")
    ap.add_argument("--score_smooth_sigma", type=float, default=1.0)
    ap.add_argument("--tau_adapt", type=float, default=3.5)
    ap.add_argument("--pred_softargmax", action="store_true")

    ap.add_argument("--refiner_ckpt", default="checkpoints/refiner/refiner.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--grid_D", type=int, default=128)

    ap.add_argument("--hit_deltas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--plot_n", type=int, default=20)
    args = ap.parse_args()

    _ensure_dir(args.curve_dir)

    # --- load GT spans/labels ---
    ann_df = pd.read_csv(args.annotations_csv)
    if "video_id" not in ann_df.columns:
        if "yt_id" in ann_df.columns:
            ann_df = ann_df.rename(columns={"yt_id": "video_id"})
        else:
            ann_df = ann_df.rename(columns={ann_df.columns[0]: "video_id"})
    gt_map = {}
    for vid, sub in ann_df.groupby("video_id"):
        if {"start_s", "end_s"}.issubset(sub.columns):
            s, e = float(sub["start_s"].iloc[0]), float(sub["end_s"].iloc[0])
        elif {"gt_start", "gt_end"}.issubset(sub.columns):
            s, e = float(sub["gt_start"].iloc[0]), float(sub["gt_end"].iloc[0])
        else:
            s, e = 0.0, 10.0
        key = os.path.splitext(os.path.basename(str(vid)))[0]
        gt_map[key] = (s, e)

    # --- load refiner ---
    print(f"Loading refiner checkpoint from {args.refiner_ckpt} ...")
    model = load_refiner_from_ckpt(args.refiner_ckpt, args.device, args.grid_D)
    D = model.D
    print(f"✓ Loaded refiner (D={D})")

    # --- index embeddings ---
    vid_npzs = sorted(glob(os.path.join(args.vid_emb_dir, "*.npz")))
    aud_npzs = sorted(glob(os.path.join(args.aud_emb_dir, f"*__L{int(args.L_sec*1000)}ms.npz")))

    def key_from_vid(p): return os.path.splitext(os.path.basename(p))[0]
    def key_from_aud(p): return os.path.splitext(os.path.basename(p))[0].split("__L")[0]

    vids = {key_from_vid(p): p for p in vid_npzs}
    auds = {key_from_aud(p): p for p in aud_npzs}
    keys = sorted(set(vids.keys()) & set(auds.keys()) & set(gt_map.keys()))
    if not keys:
        print("[error] No overlapping ids across video/audio/annotations.")
        return

    rows = []
    n_plotted = 0
    print(f"→ Evaluating with refiner on {len(keys)} clips.")
    for k in tqdm(keys, desc="Eval+Refine", unit="clip"):
        v_npz = np.load(vids[k])
        a_npz = np.load(auds[k])

        V = v_npz["emb"].astype(np.float32)              # (Tv,D)
        t_vid = v_npz["centers_sec"].astype(np.float32)  # (Tv,)
        if V.shape[0] == 0:
            continue

        # --- audio selection ---
        if args.audio_pick == "multi":
            offsets = [float(x) for x in args.multi_offsets.split(",") if x.strip()]
            a_list, a_center = pick_audio_multi(a_npz, offsets=offsets)
            if len(a_list) == 0:  # safety
                continue
        else:
            a_vec, a_center = pick_audio_single(a_npz)
            if a_vec is None:
                continue
            a_list = [a_vec]

        # --- baseline: similarity -> (optional zscore/smooth) -> softmax -> fuse ---
        pdfs, sims_for_plot = [], []
        for a in a_list:
            sim = cosine_sim(a, V)
            if args.score_zscore:
                sim = zscore(sim)
            if args.score_smooth_sigma > 0:
                sim = gaussian_smooth(sim, args.score_smooth_sigma)
            tau_use = adapt_tau(sim, args.tau_adapt) if args.tau_adapt > 0 else args.tau
            p = softmax(sim, tau=tau_use)
            pdfs.append(p); sims_for_plot.append(sim)

        p_base = (np.stack(pdfs, 0).max(0) if args.multi_reduce == "max"
                  else np.stack(pdfs, 0).mean(0))
        p_base = p_base / (p_base.sum() + 1e-12)
        sim_plot = np.stack(sims_for_plot, 0).mean(0)

        # baseline prediction
        if args.pred_softargmax:
            pred_base_t = soft_argmax(t_vid, p_base)
        else:
            pred_base_t = float(t_vid[int(np.argmax(p_base))])

        # --- refine on a fixed grid of length D, then map back to Tv ---
        p_grid, t_grid = resample_to_len(p_base, t_vid, model.D)
        x = torch.from_numpy(p_grid[None, :]).to(args.device, dtype=torch.float32)
        with torch.inference_mode():
            p_ref = model(x)[0].detach().cpu().numpy().astype(np.float32)  # (D,)
        p_ref_T = resample_from_to(p_ref, t_grid, t_vid)  # (Tv,)

        if args.pred_softargmax:
            pred_ref_t = soft_argmax(t_vid, p_ref_T)
        else:
            pred_ref_t = float(t_vid[int(np.argmax(p_ref_T))])

        conf = float(p_ref_T.max())  # peak prob after refinement

        # --- metrics vs GT ---
        gs, ge = gt_map.get(k, (0.0, 10.0))
        metrics_ref = evaluate_hit(pred_ref_t, gs, ge, tuple(args.hit_deltas))

        rows.append({
            # canonical
            "video_id": k,
            "t_pred_sec": pred_ref_t,
            "gt_start_sec": gs,
            "gt_end_sec": ge,
            "gt_mid_sec": 0.5 * (gs + ge),
            "L_sec": args.L_sec,
            "tau": (args.tau if args.tau_adapt <= 0 else float("nan")),
            "tau_adapt": args.tau_adapt,
            "confidence": conf,
            "entropy": float(entropy(p_ref_T)),
            "hit_0.25": metrics_ref["hit@0.25"],
            "hit_0.5":  metrics_ref["hit@0.5"],
            "hit_1.0":  metrics_ref["hit@1.0"],
            "mae":      metrics_ref["mae_mid"],
            "inside":   metrics_ref["inside"],
            # extras for debugging
            "pred_base_t": pred_base_t,
            "pred_ref_t": pred_ref_t,
        })

        # plot a few examples
        if n_plotted < args.plot_n:
            title = (f"{k} | base={pred_base_t:.2f}s, ref={pred_ref_t:.2f}s | "
                     f"GT=[{gs:.2f},{ge:.2f}]")
            out_png = os.path.join(args.curve_dir, f"{k}.png")
            plot_curve_dual(out_png, t_vid, sim_plot, p_base, p_ref_T,
                            pred_base_t, pred_ref_t, gs, ge, title)
            n_plotted += 1

    # --- write outputs ---
    if not rows:
        print("[warn] No rows written (empty rows).")
        return

    df = pd.DataFrame(rows)
    _ensure_dir(os.path.dirname(args.summary_csv))
    df.to_csv(args.summary_csv, index=False)

    summary = {
        "N": int(len(df)),
        "MAE_mean": float(df["mae"].mean()),
        "MAE_median": float(df["mae"].median()),
        "Inside_mean": float(df["inside"].mean()),
        "Hit@0.25": float(df["hit_0.25"].mean()),
        "Hit@0.5":  float(df["hit_0.5"].mean()),
        "Hit@1.0":  float(df["hit_1.0"].mean()),
        "Conf_mean": float(df["confidence"].mean()),
    }
    print("→ Summary:", summary)

    # logs and quick artifacts into curve_dir
    with open(os.path.join(args.curve_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.curve_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.curve_dir, "summary.txt"), "w") as f:
        f.write("=== Eval+Refiner Summary ===\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Args ===\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    df.sort_values("mae", ascending=False).head(50).to_csv(
        os.path.join(args.curve_dir, "worst50.csv"), index=False
    )
    df.sort_values("mae", ascending=True).head(50).to_csv(
        os.path.join(args.curve_dir, "best50.csv"), index=False
    )

    if HAVE_PLT and "mae" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.hist(df["mae"].values, bins=40)
        plt.xlabel("MAE (sec)"); plt.ylabel("Count"); plt.title("Refined MAE distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(args.curve_dir, "hist_mae_refined.png"), dpi=140)
        plt.close()

if __name__ == "__main__":
    main()