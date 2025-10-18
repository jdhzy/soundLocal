# Multiscale audio embeddings with timestamps for frozen MC3
# Saves one NPZ per (video, length): cache/aud_emb/<vid>__L<ms>.npz
#   emb:         (K, D)  L2-normalized audio embeddings
#   centers_sec: (K,)    center time (sec) of each crop
#   L_sec:       float   crop length in seconds
#   stride_sec:  float   stride in seconds
#   sr:          int     sample rate (16_000)

import os
import argparse
import numpy as np
import torch
import torchaudio as ta
from glob import glob
from typing import Tuple, List
from mc3_frozen import FrozenMC3

def load_wav_mono_16k(path: str) -> Tuple[torch.Tensor, int]:
    wav, sr = ta.load(path)          # (C, T)
    if sr != 16000:
        wav = ta.functional.resample(wav, sr, 16000)
        sr = 16000
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.shape[0] == 1:
        pass
    else:
        wav = wav.unsqueeze(0)
    return wav, sr  # (1, T), 16000

def crop_indices(T: int, sr: int, L_sec: float, stride_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return start/end sample indices and center times (sec) for sliding crops."""
    L = int(round(L_sec * sr))
    S = int(round(stride_sec * sr))
    if L <= 0 or S <= 0 or T <= 0:
        return np.zeros((0,2), np.int64), np.zeros((0,), np.float32)

    starts = list(range(0, max(T - L + 1, 1), S))
    if T < L:
        # Single crop with padding handled by caller (we'll allow a smaller last segment)
        starts = [0]

    idx = []
    centers = []
    for s in starts:
        e = min(s + L, T)
        # Always produce a crop length of L (pad later if needed)
        # Center time in seconds:
        c = (s + min(s + L, T)) / 2.0 / sr
        idx.append([s, e])
        centers.append(c)
    return np.array(idx, dtype=np.int64), np.array(centers, dtype=np.float32)

def get_crops(wav: torch.Tensor, sr: int, idx: np.ndarray, L_sec: float) -> torch.Tensor:
    """Return (K, L) crops, padding end with zeros if needed."""
    L = int(round(L_sec * sr))
    K = idx.shape[0]
    out = torch.zeros((K, L), dtype=wav.dtype)
    for i, (s, e) in enumerate(idx):
        seg = wav[0, s:e]
        Lcur = e - s
        if Lcur >= L:
            out[i] = seg[:L]
        else:
            out[i, :Lcur] = seg
    return out

def parse_lengths(s: str) -> List[float]:
    # "0.5,1.0,2.0" -> [0.5,1.0,2.0]
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if args.device in ("cpu","cuda") else ("cuda" if torch.cuda.is_available() else "cpu")
    model = FrozenMC3(device=device)

    lengths = parse_lengths(args.lengths)
    wavs = sorted(glob(os.path.join(args.wav_dir, "*.wav")))
    if len(wavs) == 0:
        print(f"[warn] no wav files found in {args.wav_dir}")
        return

    for w in wavs:
        vid_id = os.path.splitext(os.path.basename(w))[0]
        wav, sr = load_wav_mono_16k(w)            # (1,T), 16k
        T = wav.shape[-1]
        wav = wav.to(model.device)

        for L_sec in lengths:
            tag_ms = int(round(L_sec * 1000))
            out_npz = os.path.join(args.out_dir, f"{vid_id}__L{tag_ms}ms.npz")
            if os.path.exists(out_npz) and not args.overwrite:
                continue

            idx, centers_s = crop_indices(T, sr, L_sec=L_sec, stride_sec=args.stride_sec)
            if idx.shape[0] == 0:
                np.savez_compressed(out_npz, emb=np.zeros((0,512), np.float32),
                                    centers_sec=np.zeros((0,), np.float32),
                                    L_sec=float(L_sec), stride_sec=float(args.stride_sec), sr=sr)
                print(f"[warn] no crops for {vid_id} L={L_sec}s")
                continue

            crops = get_crops(wav, sr, idx, L_sec=L_sec)  # (K, L)
            # Encode in small batches to save memory
            K = crops.shape[0]
            bs = args.batch_size
            embs = []
            with torch.inference_mode():
                for i in range(0, K, bs):
                    seg = crops[i:i+bs].to(model.device)       # (B, L)
                    emb = model.encode_audio(seg)              # (B, D)
                    embs.append(emb.cpu().numpy())
            embs = np.concatenate(embs, axis=0).astype(np.float32)  # (K, D)

            np.savez_compressed(
                out_npz,
                emb=embs,
                centers_sec=centers_s.astype(np.float32),
                L_sec=float(L_sec),
                stride_sec=float(args.stride_sec),
                sr=int(sr)
            )
            print(f"✓ saved {out_npz} emb{embs.shape} crops={K} L={L_sec}s stride={args.stride_sec}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", default="data/ave/ave_wav")
    ap.add_argument("--out_dir", default="cache/aud_emb")
    ap.add_argument("--lengths", type=str, default="0.5,1.0,2.0", help="comma-separated seconds, e.g. '0.5,1.0,2.0'")
    ap.add_argument("--stride_sec", type=float, default=0.25)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    main(args)
