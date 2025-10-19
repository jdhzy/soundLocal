#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, json
from glob import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------- same utilities as trainer ----------

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

def pick_audio_single(a_npz) -> Tuple[np.ndarray, float]:
    A = a_npz["emb"].astype(np.float32)
    centers = a_npz["centers_sec"].astype(np.float32)
    if A.shape[0] == 0:
        return None, None
    mid = float(np.median(centers))
    idx = int(np.argmin(np.abs(centers - mid)))
    return A[idx], float(centers[idx])

def pick_audio_multi(a_npz, offsets: List[float], reduce: str = "mean") -> Tuple[List[np.ndarray], float]:
    A = a_npz["emb"].astype(np.float32)
    centers = a_npz["centers_sec"].astype(np.float32)
    if A.shape[0] == 0:
        return [], None
    mid = float(np.median(centers))
    want = np.array([mid + d for d in offsets], dtype=np.float32)
    idx = np.clip(np.searchsorted(centers, want), 0, max(0, len(centers) - 1))
    vecs = [A[i] for i in idx]
    return vecs, float(mid)

def resample_from_to(y: np.ndarray, t_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """Resample y(t_src) onto t_tgt (both in seconds). Preserves integral ~1."""
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
    # also return seconds grid for convenience
    t_tgt = np.linspace(t_src.min(), t_src.max(), out_len, dtype=np.float32)
    return y_tgt, t_tgt

def evaluate_hit(t_hat: float, gs: float, ge: float, deltas=(0.25, 0.5, 1.0)):
    mid = 0.5 * (gs + ge)
    out = {
        "mae_mid": abs(t_hat - mid),
        "inside": float(gs <= t_hat <= ge),
    }
    for d in deltas:
        out[f"hit@{d}"] = float(abs(t_hat - mid) <= d)
    return out

# ---------- model ----------

class Refiner(torch.nn.Module):
    def __init__(self, D=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(D, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, D),
        )
        self.D = D

    def forward(self, x):
        x = x.float()
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid_emb_dir", default="cache/vid_emb")
    ap.add_argument("--aud_emb_dir", default="cache/aud_emb")
    ap.add_argument("--annotations_csv", default="data/ave/ave_annotations.csv")
    ap.add_argument("--summary_csv", default="reports/summary_refined.csv")
    ap.add_argument("--curve_dir", default="reports/curves_refined")

    ap.add_argument("--L_sec", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--audio_pick", choices=["middle", "multi"], default="multi")
    ap.add_argument("--multi_offsets", type=str, default="0,0.25,-0.25")
    ap.add_argument("--multi_reduce", choices=["mean", "max"], default="mean")
    ap.add_argument("--score_zscore", action="store_true")
    ap.add_argument("--score_smooth_sigma", type=float, default=1.0)
    ap.add_argument("--tau_adapt", type=float, default=3.5)

    ap.add_argument("--refiner_ckpt", default="checkpoints/refiner/refiner.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--grid_D", type=int, default=128)

    ap.add_argument("--hit_deltas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--plot_n", type=int, default=0)

    ap.add_argument("--pred_softargmax", action="store_true",
                help="use soft-argmax for prediction instead of hard argmax")
    
    args = ap.parse_args()

    os.makedirs(args.curve_dir, exist_ok=True)

    # load model
    # --- load model safely ---
    import torch.nn as nn

    def _load_refiner_checkpoint(path, device, model: nn.Module):
        # Try safe load first (suppresses the FutureWarning)
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=device)

        # Normalize to a state_dict format
        state_dict = None
        if isinstance(ckpt, nn.Module):
            state_dict = ckpt.state_dict()
        elif isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state_dict = ckpt["state_dict"]
            elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
                state_dict = ckpt["model_state_dict"]
            else:
                if all(isinstance(k, str) for k in ckpt.keys()):
                    state_dict = ckpt

        # fallback for non-standard saves
        if state_dict is None:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            if isinstance(ckpt, nn.Module):
                state_dict = ckpt.state_dict()
            elif isinstance(ckpt, dict):
                state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or (
                    ckpt if all(isinstance(k, str) for k in ckpt.keys()) else None
                )

        if state_dict is None:
            raise RuntimeError(f"Unrecognized checkpoint format at {path}")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"⚠️  load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}")
            for k in missing[:5]: print("   missing:", k)
            for k in unexpected[:5]: print("   unexpected:", k)

        return model


    # --- instantiate and load ---
    print(f"Loading refiner checkpoint from {args.refiner_ckpt} ...")

    # Build model — if checkpoint has D, use it; else fall back to args.grid_D
    try:
        ckpt = torch.load(args.refiner_ckpt, map_location=args.device)
        D = int(ckpt.get("D", args.grid_D)) if isinstance(ckpt, dict) else args.grid_D
    except Exception:
        D = args.grid_D

    model = Refiner(D=D).to(args.device).float().eval()
    model = _load_refiner_checkpoint(args.refiner_ckpt, args.device, model)
    print(f"✓ Loaded refiner from {args.refiner_ckpt}")

    # index files
    vid_npzs = sorted(glob(os.path.join(args.vid_emb_dir, "*.npz")))
    aud_npzs = sorted(glob(os.path.join(args.aud_emb_dir, f"*__L{int(args.L_sec*1000)}ms.npz")))

    def key_from_vid(p):
        return os.path.splitext(os.path.basename(p))[0]
    def key_from_aud(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        return stem.split("__L")[0]

    vids = {key_from_vid(p): p for p in vid_npzs}
    auds = {key_from_aud(p): p for p in aud_npzs}
    keys = sorted(set(vids.keys()) & set(auds.keys()))
    if not keys:
        print("[error] No overlapping video/audio NPZs. Check paths.")
        return

    rows = []
    print(f"→ Evaluating with refiner on {len(keys)} clips.")
    for k in tqdm(keys, desc="Eval+Refine", unit="clip"):
        v_npz = np.load(vids[k])
        a_npz = np.load(auds[k])

        V = v_npz["emb"].astype(np.float32)
        t_vid = v_npz["centers_sec"].astype(np.float32)
        if V.shape[0] == 0:
            continue

        # baseline p_fused
        if args.audio_pick == "multi":
            offsets = [float(x) for x in args.multi_offsets.split(",") if x.strip()]
            a_list, a_center = pick_audio_multi(a_npz, offsets=offsets, reduce=args.multi_reduce)
            if len(a_list) == 0:
                continue
        else:
            a_vec, a_center = pick_audio_single(a_npz)
            if a_vec is None:
                continue
            a_list = [a_vec]

        pdfs = []
        for a in a_list:
            sim = cosine_sim(a, V)
            if args.score_zscore:
                sim = zscore(sim)
            if args.score_smooth_sigma > 0:
                sim = gaussian_smooth(sim, args.score_smooth_sigma)
            tau_use = adapt_tau(sim, args.tau_adapt) if args.tau_adapt > 0 else args.tau
            p = softmax(sim, tau=tau_use)
            pdfs.append(p)
        pdfs = np.stack(pdfs, 0)
        p_fused = pdfs.max(0) if args.multi_reduce == "max" else pdfs.mean(0)
        p_fused = p_fused / (p_fused.sum() + 1e-12)

        # resample to refiner grid, refine, map back to original Tv
        p_grid, t_grid = resample_to_len(p_fused, t_vid, D)
        x = torch.from_numpy(p_grid[None, :]).to(args.device, dtype=torch.float32)
        with torch.inference_mode():
            p_ref = model(x)[0].detach().cpu().numpy().astype(np.float32)  # (D,)

        # map refined PDF back onto original t_vid grid
        p_ref_T = resample_from_to(p_ref, t_grid, t_vid)  # (Tv,)
        pred_idx = int(np.argmax(p_ref_T))
        pred_t = float(t_vid[pred_idx])
        conf = float(p_ref_T[pred_idx])

        # AVE GT span if present in annotations CSV – not available in npz; assume 0..10s
        gs, ge = 0.0, 10.0
        metrics = evaluate_hit(pred_t, gs, ge, tuple(args.hit_deltas))

        rows.append({
            "video_id": k,
            "t_pred_sec": pred_t,
            "gt_start_sec": gs,
            "gt_end_sec": ge,
            "gt_mid_sec": 0.5 * (gs + ge),
            "L_sec": args.L_sec,
            "tau": args.tau if args.tau_adapt <= 0 else float("nan"),
            "tau_adapt": args.tau_adapt,
            "confidence": conf,
            "entropy": float(entropy(p_ref_T)),
            "hit_0.25": metrics["hit@0.25"],
            "hit_0.5":  metrics["hit@0.5"],
            "hit_1.0":  metrics["hit@1.0"],
            "mae":      metrics["mae_mid"],
            "inside":   metrics["inside"],
        })

    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
        df.to_csv(args.summary_csv, index=False)
        agg = {
            "N": int(len(df)),
            "MAE_mean": float(df["mae"].mean()),
            "Inside_mean": float(df["inside"].mean()),
            "Hit@0.25": float(df["hit_0.25"].mean()),
            "Hit@0.5":  float(df["hit_0.5"].mean()),
            "Hit@1.0":  float(df["hit_1.0"].mean()),
            "Conf_mean": float(df["confidence"].mean()),
        }
        print("→ Summary:", agg)
    else:
        print("[warn] No rows written.")

if __name__ == "__main__":
    main()