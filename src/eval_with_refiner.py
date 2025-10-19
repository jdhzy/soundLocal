# src/eval_with_refiner.py
import argparse, os, json
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

# --- minimal helpers (aligned with train_refiner.py) ---
def extract_youtube_id(stem): return stem
def cosine_sim(a_vec, V):
    denom = (np.linalg.norm(V,axis=1)*(np.linalg.norm(a_vec)+1e-8))+1e-8
    return (V @ a_vec) / denom
def softmax(sim, tau):
    z = (sim/max(tau,1e-6)); z -= z.max()
    p = np.exp(z); return p/(p.sum()+1e-12)
def zscore(x): mu,sd=x.mean(),x.std()+1e-8; return (x-mu)/sd
def gaussian_smooth(x,sigma):
    if sigma<=0: return x
    R=int(3*sigma); k=np.arange(-R,R+1); w=np.exp(-0.5*(k/sigma)**2); w/=w.sum()
    return np.convolve(x,w,mode="same")
def adapt_tau(sim,target_H,max_iter=20):
    lo,hi=1e-3,1.0
    for _ in range(max_iter):
        mid=(lo+hi)/2
        p=softmax(sim,mid)
        H=-(p*np.log(p+1e-12)).sum()
        if H>target_H: lo=mid
        else: hi=mid
    return (lo+hi)/2
def pick_audio_single(a_npz):
    A=a_npz["emb"]; c=a_npz["centers_sec"]
    if len(A)==0: return None,None
    mid=np.median(c); idx=int(np.argmin(np.abs(c-mid)))
    return A[idx], float(c[idx])
def pick_audio_multi(a_npz, offsets, reduce="mean"):
    A=a_npz["emb"]; c=a_npz["centers_sec"]
    if len(A)==0: return [], None
    base=float(np.median(c))
    idxs=[int(np.argmin(np.abs(c-(base+o)))) for o in offsets]
    return [A[i] for i in idxs], base
def fuse_pdfs(pdfs, reduce="mean"):
    pdfs=np.stack(pdfs,0); return pdfs.mean(0) if reduce=="mean" else pdfs.max(0)
def resample_to_fixed(t, p, bins=128):
    if t.min()<0: t=t-t.min()
    if t.max()==0: t=t+1e-6
    ti=(t-t.min())/(t.max()-t.min())
    grid=np.linspace(0,1,bins)
    pi=np.interp(grid, ti, p)
    pi=np.clip(pi,1e-12,None); pi/=pi.sum()
    return grid, pi
def soft_argmax(t, p):  # t (T,), p (T,)
    p = p / (p.sum() + 1e-12)
    return float((t * p).sum())
def entropy(p):
    p = p / (p.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())

# --- refiner ---
class Refiner(torch.nn.Module):
    def __init__(self, bins=128, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(bins, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, bins)
        )
    def forward(self, x):
        return self.net(x)

def evaluate_hit(pred_t, s, e, deltas=(0.25,0.5,1.0)):
    mid = (s+e)/2.0
    mae = abs(pred_t - mid)
    out = {"inside": float(s <= pred_t <= e), "mae_mid": float(mae)}
    for d in deltas:
        out[f"hit@{d}"] = float(mae <= d)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid_emb_dir", default="cache/vid_emb")
    ap.add_argument("--aud_emb_dir", default="cache/aud_emb")
    ap.add_argument("--annotations_csv", default="data/ave/ave_annotations.csv")
    ap.add_argument("--summary_csv", default="reports/summary_refined.csv")
    ap.add_argument("--curve_dir", default="reports/curves_refined")
    ap.add_argument("--L_sec", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--audio_pick", choices=["middle","multi"], default="multi")
    ap.add_argument("--multi_offsets", type=str, default="0,0.25,-0.25")
    ap.add_argument("--multi_reduce", choices=["mean","max"], default="mean")
    ap.add_argument("--score_zscore", action="store_true")
    ap.add_argument("--score_smooth_sigma", type=float, default=1.0)
    ap.add_argument("--tau_adapt", type=float, default=3.5)
    ap.add_argument("--bins", type=int, default=128)
    ap.add_argument("--pred_softargmax", action="store_true")
    # refiner
    ap.add_argument("--refiner_ckpt", default="checkpoints/refiner/refiner.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--plot_n", type=int, default=20)
    ap.add_argument("--hit_deltas", type=float, nargs="+", default=[0.25,0.5,1.0])
    args = ap.parse_args()

    os.makedirs(args.curve_dir, exist_ok=True)

    # load refiner
    ck = torch.load(args.refiner_ckpt, map_location=args.device)
    bins = int(ck.get("bins", args.bins))
    model = Refiner(bins=bins, hidden=256).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded refiner: {args.refiner_ckpt} (bins={bins})")

    # load annotations
    ann_df = pd.read_csv(args.annotations_csv)
    if "video_id" not in ann_df.columns:
        ann_df = ann_df.rename(columns={"yt_id":"video_id"} if "yt_id" in ann_df.columns else {ann_df.columns[0]:"video_id"})
    gt_map = {}
    for vid, sub in ann_df.groupby("video_id"):
        s = float(sub.get("start_s", sub.get("gt_start", pd.Series([0.0]))).iloc[0])
        e = float(sub.get("end_s",   sub.get("gt_end",   pd.Series([10.0]))).iloc[0])
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
    print(f"→ Evaluating {len(keys)} clips with refiner...")

    rows, n_plotted = [], 0
    for k in tqdm(keys, desc="RefinedEval", unit="clip"):
        vnpz = np.load(vid_npzs[k]); anpz = np.load(aud_npzs[k])
        V = vnpz["emb"].astype(np.float32)
        t = vnpz["centers_sec"].astype(np.float32)
        if V.shape[0]==0: continue

        # build base fused pdf (like train_refiner)
        if args.audio_pick=="multi":
            offs = [float(x) for x in args.multi_offsets.split(",") if x.strip()]
            a_list, a_center = pick_audio_multi(anpz, offs, reduce=args.multi_reduce)
        else:
            a_vec, a_center = pick_audio_single(anpz)
            a_list = [a_vec] if a_vec is not None else []
        if len(a_list)==0: continue

        pdfs=[]
        for a in a_list:
            sim = cosine_sim(a,V)
            if args.score_zscore: sim=zscore(sim)
            if args.score_smooth_sigma>0: sim=gaussian_smooth(sim,args.score_smooth_sigma)
            tau_use = adapt_tau(sim,args.tau_adapt) if args.tau_adapt>0 else args.tau
            pdfs.append(softmax(sim,tau_use))
        p_fused = fuse_pdfs(pdfs,args.multi_reduce)

        # resample to fixed bins and run refiner
        grid01, pin = resample_to_fixed(t, p_fused, bins=bins)  # pin sums to 1
        with torch.inference_mode():
            logits = model(torch.tensor(pin[None,:], device=args.device))
            pout = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # map refined bin to original time range
        # get hard/soft predictions
        hard_idx = int(np.argmax(pout))
        hard_t = float(np.interp(grid01[hard_idx], [0,1], [t.min(), t.max()]))
        if args.pred_softargmax:
            soft_t = float(np.interp(float((grid01 * pout).sum()), [0,1], [t.min(), t.max()]))
            pred_t = soft_t
        else:
            pred_t = hard_t

        conf = float(pout.max()); ent = entropy(pout)
        s,e = gt_map[k]
        metr = evaluate_hit(pred_t, s, e, deltas=tuple(args.hit_deltas))

        row = {
            "video_id": k, "t_pred_sec": pred_t,
            "gt_start_sec": s, "gt_end_sec": e, "gt_mid_sec": (s+e)/2.0,
            "audio_center": float(a_center) if a_center is not None else float("nan"),
            "tau": float(args.tau), "tau_adapt": float(args.tau_adapt),
            "L_sec": float(args.L_sec),
            "confidence": conf, "entropy": ent,
            "hit_0.25": metr.get("hit@0.25",0.0),
            "hit_0.5":  metr.get("hit@0.5",0.0),
            "hit_1.0":  metr.get("hit@1.0",0.0),
            "mae": metr.get("mae_mid", abs(pred_t-(s+e)/2.0)),
            "inside": metr.get("inside", float(s<=pred_t<=e)),
            # legacy for analyzer compatibility
            "yt_id": k, "pred_t": pred_t, "gt_start": s, "gt_end": e, "conf": conf,
        }
        rows.append(row)

    if rows:
        os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
        pd.DataFrame(rows).to_csv(args.summary_csv, index=False)
        print(f"✓ saved {args.summary_csv}")
    else:
        print("[warn] no rows evaluated")

if __name__ == "__main__":
    main()