#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, math, json
from glob import glob
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ----------------------------- utilities -----------------------------

def l2normalize(x: np.ndarray, axis=-1, eps=1e-8):
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n

def cosine_sim(a: np.ndarray, V: np.ndarray) -> np.ndarray:
    """a: (D,), V: (T,D) -> (T,)"""
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
    # simple discrete 1D gaussian kernel with radius 3*sigma
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
    """Binary search tau s.t. entropy(softmax(sim/tau)) ≈ target_H."""
    lo, hi = 1e-3, 10.0
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        H = entropy(softmax(sim, tau=mid))
        if H > target_H:  # too flat -> lower tau
            hi = mid
        else:
            lo = mid
        if abs(H - target_H) < tol:
            break
    return float(0.5 * (lo + hi))

def pick_audio_single(a_npz) -> Tuple[np.ndarray, float]:
    A = a_npz["emb"].astype(np.float32)            # (Ka, D)
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

def gaussian_target_on_grid(t_vid: np.ndarray, center_s: float, sigma_s: float) -> np.ndarray:
    """Unnormalized gaussian pdf over video centers, then normalized."""
    if sigma_s <= 0:
        # delta-like: put mass at closest index
        idx = int(np.argmin(np.abs(t_vid - center_s)))
        p = np.zeros_like(t_vid, dtype=np.float32)
        p[idx] = 1.0
        return p
    d = ((t_vid - center_s) / sigma_s).astype(np.float32)
    p = np.exp(-0.5 * d * d).astype(np.float32)
    p = p / (p.sum() + 1e-12)
    return p

def resample_to_len(y: np.ndarray, t_src: np.ndarray, out_len: int) -> np.ndarray:
    """Resample curve y(t_src) to uniform u in [0,1] with out_len points."""
    if len(y) == 0:
        return np.zeros((out_len,), dtype=np.float32)
    u_src = (t_src - t_src.min()) / max(1e-8, (t_src.max() - t_src.min()))
    u_tgt = np.linspace(0.0, 1.0, out_len, dtype=np.float32)
    y_tgt = np.interp(u_tgt, u_src, y).astype(np.float32)
    # renormalize if it's a pdf
    s = float(y_tgt.sum())
    if s > 0:
        y_tgt /= s
    return y_tgt

# ----------------------------- model -----------------------------

class Refiner(nn.Module):
    """
    Tiny MLP that refines a time PDF sampled on a fixed grid (D=128).
    Input:  (B, D) baseline PDF (or feature)
    Output: (B, D) refined PDF (softmax-normalized)
    """
    def __init__(self, D: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, D),
        )
        self.D = D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure float32 to match weights
        x = x.float()
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

# ----------------------------- dataset builder -----------------------------

def build_baseline_pdf(v_npz, a_npz, args) -> Tuple[np.ndarray, np.ndarray]:
    V = v_npz["emb"].astype(np.float32)                  # (Tv, D)
    t_vid = v_npz["centers_sec"].astype(np.float32)      # (Tv,)
    if V.shape[0] == 0:
        return None, None

    if args.audio_pick == "multi":
        offsets = [float(x) for x in args.multi_offsets.split(",") if x.strip()]
        a_list, a_center = pick_audio_multi(a_npz, offsets=offsets, reduce=args.multi_reduce)
        if len(a_list) == 0:
            return None, None
    else:
        a_vec, a_center = pick_audio_single(a_npz)
        if a_vec is None:
            return None, None
        a_list = [a_vec]

    pdfs = []
    for a in a_list:
        sim = cosine_sim(a, V)  # (Tv,)
        if args.score_zscore:
            sim = zscore(sim)
        if args.score_smooth_sigma > 0:
            sim = gaussian_smooth(sim, sigma=args.score_smooth_sigma)
        tau_use = adapt_tau(sim, args.tau_adapt) if args.tau_adapt > 0 else args.tau
        p = softmax(sim, tau=tau_use)
        pdfs.append(p)

    pdfs = np.stack(pdfs, axis=0)
    if args.multi_reduce == "max":
        p_fused = pdfs.max(axis=0)
    else:
        p_fused = pdfs.mean(axis=0)
    # normalize again for safety
    p_fused = p_fused / (p_fused.sum() + 1e-12)
    return p_fused.astype(np.float32), t_vid.astype(np.float32)

# ----------------------------- training -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_emb_dir", default="cache/vid_emb")
    parser.add_argument("--aud_emb_dir", default="cache/aud_emb")
    parser.add_argument("--annotations_csv", default="data/ave/ave_annotations.csv")
    parser.add_argument("--L_sec", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--audio_pick", choices=["middle", "multi"], default="multi")
    parser.add_argument("--multi_offsets", type=str, default="0,0.25,-0.25")
    parser.add_argument("--multi_reduce", choices=["mean", "max"], default="mean")
    parser.add_argument("--score_zscore", action="store_true")
    parser.add_argument("--score_smooth_sigma", type=float, default=1.0)
    parser.add_argument("--tau_adapt", type=float, default=3.5)
    parser.add_argument("--sigma_target", type=float, default=0.5, help="GT gaussian sigma in seconds")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", default="checkpoints/refiner")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grid_D", type=int, default=128, help="refiner input/output length")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Index files
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

    # Build dataset
    X, Y = [], []
    # ---- optional: skip training if checkpoint already exists ----
    ckpt_path = os.path.join(args.out_dir, "refiner.pt")
    if os.path.exists(ckpt_path):
        print(f"⚠️  Found existing checkpoint at {ckpt_path}. Skipping retraining.")
        print("   (Delete the file manually if you want to retrain.)")
        return
    
    print(f"Building dataset from {len(keys)} clips...")
    for k in tqdm(keys):
        v_npz = np.load(vids[k])
        a_npz = np.load(auds[k])

        p_fused, t_vid = build_baseline_pdf(v_npz, a_npz, args)
        if p_fused is None:
            continue

        # GT midpoint (AVE: if absent, 5s)
        gt_start = float(v_npz.get("gt_start_sec", 0.0)) if "gt_start_sec" in v_npz else 0.0
        gt_end   = float(v_npz.get("gt_end_sec",   10.0)) if "gt_end_sec"   in v_npz else 10.0
        mid = 0.5 * (gt_start + gt_end)

        tgt_T = gaussian_target_on_grid(t_vid, center_s=mid, sigma_s=args.sigma_target)
        x = resample_to_len(p_fused, t_vid, args.grid_D)
        y = resample_to_len(tgt_T,   t_vid, args.grid_D)

        X.append(x.astype(np.float32))
        Y.append(y.astype(np.float32))

    if len(X) == 0:
        print("[error] Empty dataset.")
        return

    X = torch.from_numpy(np.stack(X, axis=0)).to(args.device, dtype=torch.float32)
    Y = torch.from_numpy(np.stack(Y, axis=0)).to(args.device, dtype=torch.float32)
    print("Dataset:", X.shape, Y.shape)

    # Model / Optim
    model = Refiner(D=args.grid_D).to(args.device).float()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # KLDivLoss expects log-prob input; our model returns prob -> take log
    kld = nn.KLDivLoss(reduction="batchmean")

    best = math.inf
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(X.size(0), device=args.device)
        losses = []
        for i in range(0, X.size(0), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = X[idx]
            yb = Y[idx]

            pb = model(xb)                       # prob
            loss = kld((pb + 1e-12).log(), yb)  # KL(p || y): y is target
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses))
        print(f"[ep {ep}] loss={mean_loss:.4f}")

        # save best
        if mean_loss < best:
            best = mean_loss
            path = os.path.join(args.out_dir, "refiner.pt")
            torch.save({"state_dict": model.state_dict(), "D": args.grid_D}, path)
            print(f"  ✓ saved {path} (best so far)")

if __name__ == "__main__":
    main()