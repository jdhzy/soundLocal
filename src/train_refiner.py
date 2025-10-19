# src/train_refiner.py
import argparse, os, math, json
from glob import glob
from typing import Tuple, List
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- helpers copied/trimmed from your eval ----------
def extract_youtube_id(stem: str) -> str:
    # your stems already are yt ids; keep passthrough
    return stem

def cosine_sim(a_vec: np.ndarray, V: np.ndarray) -> np.ndarray:
    # V: (T,D), a_vec: (D,)
    denom = (np.linalg.norm(V, axis=1) * (np.linalg.norm(a_vec) + 1e-8)) + 1e-8
    return (V @ a_vec) / denom

def softmax(sim: np.ndarray, tau: float) -> np.ndarray:
    z = (sim / max(tau, 1e-6))
    z -= z.max()
    p = np.exp(z)
    return p / (p.sum() + 1e-12)

def zscore(x: np.ndarray) -> np.ndarray:
    mu, sd = x.mean(), x.std() + 1e-8
    return (x - mu) / sd

def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0: return x
    radius = int(3*sigma)
    k = np.arange(-radius, radius+1)
    w = np.exp(-0.5*(k/sigma)**2)
    w /= w.sum()
    return np.convolve(x, w, mode="same")

def adapt_tau(sim: np.ndarray, target_H: float, max_iter=20) -> float:
    # solve tau s.t. entropy(p)=target_H with a tiny binary search
    lo, hi = 1e-3, 1.0
    for _ in range(max_iter):
        mid = (lo+hi)/2
        p = softmax(sim, tau=mid)
        H = -(p * np.log(p + 1e-12)).sum()
        if H > target_H: lo = mid
        else: hi = mid
    return (lo+hi)/2

def pick_audio_single(a_npz) -> Tuple[np.ndarray,float]:
    A = a_npz["emb"]
    centers = a_npz["centers_sec"]
    if len(A)==0: return None, None
    mid = np.median(centers)
    idx = int(np.argmin(np.abs(centers - mid)))
    return A[idx], float(centers[idx])

def pick_audio_multi(a_npz, offsets: List[float], reduce="mean"):
    A = a_npz["emb"]
    centers = a_npz["centers_sec"]
    if len(A)==0: return [], None
    base = float(np.median(centers))
    target_cs = [base+o for o in offsets]
    idxs = [int(np.argmin(np.abs(centers - c))) for c in target_cs]
    vecs = [A[i] for i in idxs]
    return vecs, base

def fuse_pdfs(pdfs: List[np.ndarray], reduce="mean") -> np.ndarray:
    pdfs = np.stack(pdfs,0)
    return pdfs.mean(0) if reduce=="mean" else pdfs.max(0)

def build_pdf(vid_npz, aud_npz, args) -> Tuple[np.ndarray, np.ndarray]:
    V = vid_npz["emb"].astype(np.float32)
    t_vid = vid_npz["centers_sec"].astype(np.float32)
    if V.shape[0]==0: return None, None
    if args.audio_pick == "multi":
        offsets = [float(x) for x in args.multi_offsets.split(",") if x.strip()]
        a_list, a_center = pick_audio_multi(aud_npz, offsets, reduce=args.multi_reduce)
    else:
        a_vec, a_center = pick_audio_single(aud_npz)
        a_list = [a_vec] if a_vec is not None else []
    if len(a_list)==0: return None, None

    pdfs = []
    for a in a_list:
        sim = cosine_sim(a, V)
        if args.score_zscore: sim = zscore(sim)
        if args.score_smooth_sigma>0: sim = gaussian_smooth(sim, args.score_smooth_sigma)
        tau_use = adapt_tau(sim, args.tau_adapt) if args.tau_adapt>0 else args.tau
        pdfs.append(softmax(sim, tau=tau_use))
    p = fuse_pdfs(pdfs, args.multi_reduce)
    return t_vid, p

# ---------- refiner model (MLP on 128-bin logits) ----------
class Refiner(nn.Module):
    def __init__(self, bins=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bins, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, bins)  # logits
        )
    def forward(self, x):  # x: (B, bins)
        return self.net(x)

def resample_to_fixed(t: np.ndarray, p: np.ndarray, bins=128):
    # interpolate PDF p(t) onto [0,1] with 'bins' points
    if t.min()<0: t = t - t.min()
    if t.max()==0: t = t + 1e-6
    ti = (t - t.min()) / (t.max()-t.min())
    grid = np.linspace(0,1,bins)
    pi = np.interp(grid, ti, p)
    pi = np.clip(pi, 1e-12, None)
    pi = pi / pi.sum()
    return grid, pi

def gaussian_target(mid_s: float, t_min: float, t_max: float, bins=128, sigma_s=0.5):
    # make a Gaussian around the mid-point in seconds, sampled on [t_min,t_max]
    grid = np.linspace(t_min, t_max, bins)
    w = np.exp(-0.5*((grid - mid_s)/max(sigma_s,1e-6))**2)
    w = np.clip(w, 1e-12, None)
    w = w / w.sum()
    return w

def kl_loss(pred_log_probs, target_probs):
    # pred_log_probs: (B,bins), target_probs: (B,bins)
    return F.kl_div(pred_log_probs, target_probs, reduction="batchmean")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid_emb_dir", default="cache/vid_emb")
    ap.add_argument("--aud_emb_dir", default="cache/aud_emb")
    ap.add_argument("--annotations_csv", default="data/ave/ave_annotations.csv")
    ap.add_argument("--L_sec", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--audio_pick", choices=["middle","multi"], default="multi")
    ap.add_argument("--multi_offsets", type=str, default="0,0.25,-0.25")
    ap.add_argument("--multi_reduce", choices=["mean","max"], default="mean")
    ap.add_argument("--score_zscore", action="store_true")
    ap.add_argument("--score_smooth_sigma", type=float, default=1.0)
    ap.add_argument("--tau_adapt", type=float, default=3.5)
    # training
    ap.add_argument("--bins", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sigma_target", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", default="checkpoints/refiner")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load annotations
    ann_df = pd.read_csv(args.annotations_csv)
    if "video_id" not in ann_df.columns:
        ann_df = ann_df.rename(columns={"yt_id":"video_id"} if "yt_id" in ann_df.columns else {ann_df.columns[0]:"video_id"})
    gt_map = {}
    for vid, sub in ann_df.groupby("video_id"):
        s = float(sub.get("start_s", sub.get("gt_start", pd.Series([0.0]))).iloc[0])
        e = float(sub.get("end_s", sub.get("gt_end",   pd.Series([10.0]))).iloc[0])
        gt_map[extract_youtube_id(vid)] = (s, e)

    # index NPZs
    vid_npzs = {extract_youtube_id(os.path.splitext(os.path.basename(p))[0]): p
                for p in glob(os.path.join(args.vid_emb_dir,"*.npz"))}
    aud_npzs = {}
    for p in glob(os.path.join(args.aud_emb_dir, f"*__L{int(args.L_sec*1000)}ms.npz")):
        stem = os.path.splitext(os.path.basename(p))[0]
        vid = stem.split("__L")[0]
        aud_npzs[extract_youtube_id(vid)] = p
    keys = sorted(set(vid_npzs) & set(aud_npzs) & set(gt_map))
    print(f"Building dataset from {len(keys)} clips...")

    X, Y = [], []
    for k in tqdm(keys):
        vnpz = np.load(vid_npzs[k])
        anpz = np.load(aud_npzs[k])
        t, p = build_pdf(vnpz, anpz, args)
        if t is None: continue
        # input: resampled PDF on [0,1]
        _, pi = resample_to_fixed(t, p, bins=args.bins)
        # target: Gaussian around midpoint on [t_min,t_max]
        s,e = gt_map[k]
        target = gaussian_target((s+e)/2, t_min=t.min(), t_max=t.max(), bins=args.bins, sigma_s=args.sigma_target)
        X.append(pi.astype(np.float32))
        Y.append(target.astype(np.float32))

    X = torch.tensor(np.stack(X), device=args.device)
    Y = torch.tensor(np.stack(Y), device=args.device)
    print("Dataset:", X.shape, Y.shape)

    model = Refiner(bins=args.bins, hidden=256).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = math.inf

    for ep in range(1, args.epochs+1):
        model.train()
        total = 0.0
        for i in range(0, len(X), args.batch_size):
            xb = X[i:i+args.batch_size]
            yb = Y[i:i+args.batch_size]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)                 # (B,bins)
            logp = F.log_softmax(logits, dim=-1)
            loss = kl_loss(logp, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
        avg = total / len(X)
        print(f"[ep {ep}] loss={avg:.4f}")
        if avg < best:
            best = avg
            ckpt = os.path.join(args.out_dir, "refiner.pt")
            torch.save({"model": model.state_dict(), "bins": args.bins}, ckpt)
            with open(os.path.join(args.out_dir, "train_cfg.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            print(f"  ✓ saved {ckpt} (best so far)")

if __name__ == "__main__":
    main()