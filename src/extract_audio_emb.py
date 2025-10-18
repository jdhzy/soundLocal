# src/extract_audio_emb.py
import os
import argparse
import numpy as np
import torch
import torchaudio as ta
import pandas as pd
from glob import glob
from mc3_frozen import FrozenMC3

def load_wav_16k(path):
    wav, sr = ta.load(path)
    if sr != 16000:
        wav = ta.functional.resample(wav, sr, 16000)
    wav = wav.mean(dim=0, keepdim=True)
    return wav

def crop_1s_center(wav):
    T = wav.shape[-1]
    need = 16000
    if T <= need:
        pad = need - T
        left, right = pad // 2, pad - pad // 2
        return torch.nn.functional.pad(wav, (left, right))
    start = (T - need) // 2
    return wav[:, start:start+need]

def crop_1s_from_csv(wav, row):
    mid = (row["start_s"] + row["end_s"]) / 2.0
    start = max(0, int(16000 * (mid - 0.5)))
    end = start + 16000
    if end > wav.shape[-1]:
        end = wav.shape[-1]
        start = max(0, end - 16000)
    return wav[:, start:end]

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    model = FrozenMC3(device=args.device)
    meta = pd.read_csv(args.anno_csv)

    for w in sorted(glob(os.path.join(args.wav_dir, "*.wav"))):
        vid_id = os.path.splitext(os.path.basename(w))[0]
        out_npy = os.path.join(args.out_dir, f"{vid_id}.npy")
        if os.path.exists(out_npy) and not args.overwrite:
            continue

        wav = load_wav_16k(w)
        rows = meta[meta["video_id"].str.contains(vid_id, regex=False)]
        seg = crop_1s_from_csv(wav, rows.iloc[0]) if len(rows) > 0 else crop_1s_center(wav)
        seg = seg.to(model.device)
        emb = model.encode_audio(seg)
        np.save(out_npy, emb.cpu().numpy())
        print(f"✓ saved {out_npy}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", default="data/ave/ave_wav")
    ap.add_argument("--anno_csv", default="data/ave/ave_annotations.csv")
    ap.add_argument("--out_dir", default="cache/aud_emb")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
