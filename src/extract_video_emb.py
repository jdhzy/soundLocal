# src/extract_video_emb.py
import os
import argparse
import numpy as np
import torch
from glob import glob

from mc3_frozen import FrozenMC3
from video_windows import (
    read_video_cv2,
    windowize_with_timestamps,
    multiscale_windowize,
)

def parse_scales(s: str):
    """
    Parse a string like "16:4,32:8" into a tuple of (win, stride) pairs.
    """
    if not s:
        return ()
    out = []
    for part in s.split(","):
        w, st = part.split(":")
        out.append((int(w), int(st)))
    return tuple(out)

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    model = FrozenMC3(device=args.device)

    mp4s = sorted(glob(os.path.join(args.vid_dir, "*.mp4")))
    if len(mp4s) == 0:
        print(f"[warn] no mp4 files found in {args.vid_dir}")
        return

    use_amp = args.fp16 and (args.device == "cuda")

    for mp4 in mp4s:
        vid_id = os.path.splitext(os.path.basename(mp4))[0]
        out_npz = os.path.join(args.out_dir, f"{vid_id}.npz")
        if os.path.exists(out_npz) and not args.overwrite:
            continue

        # read video; be compatible with either 2-tuple or 3-tuple return
        rv = read_video_cv2(mp4, target_fps=args.fps, size=args.size)
        if isinstance(rv, tuple) and len(rv) == 3:
            vid, eff_fps, T = rv
        else:
            # legacy: (vid, eff_fps)
            vid, eff_fps = rv
            T = vid.shape[1] if vid is not None and hasattr(vid, "shape") else 0

        # Handle empty/failed decode
        if vid is None or T == 0:
            np.savez_compressed(
                out_npz,
                emb=np.zeros((0, 512), np.float32),
                centers_sec=np.zeros((0,), np.float32),
                fps=float(args.fps),
                size=int(args.size),
                win=int(args.win),
                stride=int(args.stride),
                scales=(args.scales if args.scales else "")
            )
            print(f"[warn] empty video after sampling: {vid_id}")
            continue

        # Build windows (single-scale or multiscale)
        if args.scales:
            scales = parse_scales(args.scales)
            win_meta = multiscale_windowize(vid, eff_fps, scales)  # {'idx','centers_s'}
        else:
            win_meta = windowize_with_timestamps(vid, eff_fps, args.win, args.stride)

        idx = win_meta["idx"]
        centers_s = win_meta["centers_s"]

        if idx.shape[0] == 0:
            np.savez_compressed(
                out_npz,
                emb=np.zeros((0, 512), np.float32),
                centers_sec=np.zeros((0,), np.float32),
                fps=float(eff_fps),
                size=int(args.size),
                win=int(args.win),
                stride=int(args.stride),
                scales=(args.scales if args.scales else "")
            )
            print(f"[warn] too short for windows: {vid_id}")
            continue

        # Encode in batches
        embs = []
        bs = args.batch_size
        with torch.inference_mode():
            for i in range(0, idx.shape[0], bs):
                batch_idx = idx[i:i+bs]
                # stack clips to (B,C,T,H,W)
                clips = torch.stack([vid[:, s:e] for (s, e) in batch_idx], dim=0).to(model.device)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        emb = model.encode_video(clips)
                else:
                    emb = model.encode_video(clips)
                embs.append(emb.float().cpu().numpy())
        embs = np.concatenate(embs, axis=0).astype(np.float32)  # (N,D)

        # Save NPZ with metadata + timestamps
        np.savez_compressed(
            out_npz,
            emb=embs,
            centers_sec=centers_s.astype(np.float32),
            fps=float(eff_fps),
            size=int(args.size),
            win=int(args.win),
            stride=int(args.stride),
            scales=(args.scales if args.scales else "")
        )
        print(f"✓ saved {out_npz} emb{embs.shape} windows={len(centers_s)} fps={eff_fps:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid_dir", default="data/ave/raw/AVE")
    ap.add_argument("--out_dir", default="cache/vid_emb")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--scales", type=str, default="", help='optional multi-scale, e.g. "16:4,32:8"')
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
