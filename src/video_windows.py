# src/video_windows.py
import cv2
import numpy as np
import torch
from typing import Tuple, Dict

def read_video_cv2(path: str, target_fps: int = 24, size: int = 224) -> Tuple[torch.Tensor, float, int]:
    """Read video, resample frames to ~target_fps, resize to square.
    Returns:
      vid_cthw: (C,T,H,W) float32 in [0,1]
      eff_fps:  effective fps after sampling
      T:        number of frames in the returned tensor
    """
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    # downsample by integer step
    sample_every = max(int(round(src_fps / target_fps)), 1)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % sample_every == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        i += 1
    cap.release()

    if not frames:
        return torch.zeros(0), float(target_fps), 0

    arr = np.stack(frames).astype(np.float32) / 255.0  # (T,H,W,C)
    vid = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()  # (C,T,H,W)
    eff_fps = src_fps / sample_every
    return vid, float(eff_fps), vid.shape[1]

def windowize_with_timestamps(
    vid_cthw: torch.Tensor,
    fps: float,
    win: int,
    stride: int
) -> Dict[str, np.ndarray]:
    """Create sliding windows and per-window center timestamps (sec).
    Returns dict with:
      'idx':       (N,2) start,end frame indices (end exclusive)
      'centers_s': (N,) center timestamps in seconds
    """
    C, T, H, W = vid_cthw.shape
    if T < win:
        return {'idx': np.zeros((0,2), dtype=np.int32), 'centers_s': np.zeros((0,), dtype=np.float32)}
    starts = list(range(0, T - win + 1, stride))
    idx = np.array([[s, s + win] for s in starts], dtype=np.int32)
    # center frame index per window
    centers = (idx[:, 0] + idx[:, 1] - 1) / 2.0  # zero-based index
    centers_s = centers / float(fps)
    return {'idx': idx, 'centers_s': centers_s.astype(np.float32)}

def multiscale_windowize(
    vid_cthw: torch.Tensor,
    fps: float,
    scales: Tuple[Tuple[int,int], ...]
) -> Dict[str, np.ndarray]:
    """scales: tuple of (win, stride) pairs. Returns concatenated idx & centers_s."""
    all_idx = []
    all_centers = []
    for (win, stride) in scales:
        out = windowize_with_timestamps(vid_cthw, fps, win, stride)
        if out['idx'].shape[0] == 0:
            continue
        all_idx.append(out['idx'])
        all_centers.append(out['centers_s'])
    if not all_idx:
        return {'idx': np.zeros((0,2), dtype=np.int32), 'centers_s': np.zeros((0,), dtype=np.float32)}
    idx = np.concatenate(all_idx, axis=0)
    centers_s = np.concatenate(all_centers, axis=0)
    # keep original order (by time), stable sort by centers
    order = np.argsort(centers_s, kind='mergesort')
    return {'idx': idx[order], 'centers_s': centers_s[order]}
