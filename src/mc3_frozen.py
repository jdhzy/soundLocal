# src/mc3_frozen.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta
from torchvision.models.video import mc3_18, MC3_18_Weights

# Prefer 3D channels-last if available (reduces activation mem & speeds convs)
_CHANNELS_LAST_3D = getattr(torch, "channels_last_3d", torch.channels_last)

class L2Norm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

class AudioProjector(nn.Module):
    """Very lightweight audio head over log-mel features → L2-normalized embedding."""
    def __init__(self, emb_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, emb_dim),
            L2Norm(),
        )

    @torch.inference_mode()
    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        # mels: (B, n_mels, T) → (B, 1, n_mels, T)
        x = mels.unsqueeze(1)
        return self.proj(x)

class FrozenMC3(nn.Module):
    """
    Frozen video/audio encoders in a robust, memory-friendly wrapper.

    Key choices for stability on mixed GPU types:
    - Lazy device move: we keep the backbone on CPU until the first encode.
    - FP32 weights; use autocast (FP16) only for activations to avoid BN issues.
    - channels_last_3d memory format to reduce activation memory footprint.
    - Safe normalization buffers moved lazily.
    """
    def __init__(self, device: str | None = None, vid_emb_dim: int = 512, aud_emb_dim: int = 512):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # ---- VIDEO ENCODER (torchvision mc3_18) ----
        weights = MC3_18_Weights.KINETICS400_V1
        backbone = mc3_18(weights=weights)
        # Small projection head to 512 + L2
        self.vid_backbone = backbone
        self.vid_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(512, vid_emb_dim, bias=True),
            L2Norm(),
        )

        # Memory format hint for better perf on Ampere/Lovelace
        self.vid_backbone.to(memory_format=_CHANNELS_LAST_3D)

        # ---- AUDIO ENCODER (log-mel front-end + tiny conv head) ----
        self.melspec = ta.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=64, f_min=50, f_max=8000
        )
        self.db = ta.transforms.AmplitudeToDB()
        self.aud_head = AudioProjector(aud_emb_dim)

        # Preprocessing stats (kept on CPU; moved lazily)
        self.register_buffer("img_mean", torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("img_std",  torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1), persistent=False)

        # Freeze
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        # We will move to device lazily to avoid OOM at __init__
        self._on_device = False

    # ---------- Utilities ----------
    def _ensure_on_device(self):
        """Move buffers & modules to target device once (lazy)."""
        if self._on_device:
            return
        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.vid_backbone.to(self.device, memory_format=_CHANNELS_LAST_3D, non_blocking=True)
            self.vid_head.to(self.device, memory_format=_CHANNELS_LAST_3D, non_blocking=True)
            # audio parts mostly CPU-safe; keep transforms on CPU and move only the projector
            self.aud_head.to(self.device, non_blocking=True)
            # normalization stats to device
            self.img_mean = self.img_mean.to(self.device, non_blocking=True)
            self.img_std  = self.img_std.to(self.device, non_blocking=True)
            self._on_device = True
        except torch.cuda.OutOfMemoryError as e:
            free, total = (torch.cuda.mem_get_info() if self.device.type == "cuda" else (0, 0))
            raise RuntimeError(
                f"OOM moving MC3 to {self.device}. Free/Total={free/1e9:.2f}/{total/1e9:.2f} GB. "
                f"Try lower --batch_size/--size/--win and set "
                f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128"
            ) from e

    @staticmethod
    def _to_5d_chw(video: torch.Tensor) -> torch.Tensor:
        """Ensure (B, C, T, H, W); accept (B, T, C, H, W)."""
        assert video.dim() == 5, f"Expected 5D video, got shape {tuple(video.shape)}"
        if video.shape[1] == 3:
            return video
        if video.shape[2] == 3:  # (B, T, C, H, W)
            return video.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError(f"Cannot infer channel dim from shape {tuple(video.shape)}")

    @staticmethod
    def _maybe_scale_uint(video: torch.Tensor) -> torch.Tensor:
        """Map uint8 [0..255] to float32 [0..1] if needed."""
        if video.dtype == torch.uint8:
            return video.float() / 255.0
        return video

    # ---------- Public API ----------
    @torch.inference_mode()
    def encode_video(self, clip: torch.Tensor) -> torch.Tensor:
        """
        clip: (B, C, T, H, W) or (B, T, C, H, W); dtype uint8 or float; range [0,255] or [0,1].
        Returns: (B, D) L2-normalized embeddings (D==vid_emb_dim).
        """
        self._ensure_on_device()

        # Canonicalize shape + dtype + range
        x = self._to_5d_chw(self._maybe_scale_uint(clip))
        # Normalize per model stats
        x = (x - self.img_mean) / (self.img_std + 1e-8)
        # Place on device with memory format hint
        x = x.to(self.device, memory_format=_CHANNELS_LAST_3D, non_blocking=True)

        # Forward with autocast for activations (weights remain FP32)
        use_amp = (self.device.type == "cuda")
        try:
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = self._forward_backbone(x)
            else:
                feats = self._forward_backbone(x)
            out = self.vid_head(feats)  # (B, D) then L2-normalized by head
            return out
        except torch.cuda.OutOfMemoryError:
            # Free any cached memory and rethrow for the caller to downshift batch/size.
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vid_backbone.stem(x)
        feats = self.vid_backbone.layer1(feats)
        feats = self.vid_backbone.layer2(feats)
        feats = self.vid_backbone.layer3(feats)
        feats = self.vid_backbone.layer4(feats)
        # no head here; head is applied in encode_video
        return feats

    @torch.inference_mode()
    def encode_audio(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """
        wav_16k: (B, T) 16kHz mono float tensor in [-1,1] on CPU or device.
        Returns: (B, D) L2-normalized embeddings.
        """
        # Keep front-end on CPU to avoid GPU mem churn; move projector to device.
        was_cuda = wav_16k.is_cuda
        if was_cuda:
            wav = wav_16k.detach().cpu()
        else:
            wav = wav_16k

        # Log-mel extraction (CPU)
        mels = self.db(self.melspec(wav))  # (B, n_mels, Tm)
        # Per-sample standardization
        mels = (mels - mels.mean(dim=(-2, -1), keepdim=True)) / (mels.std(dim=(-2, -1), keepdim=True) + 1e-6)

        # Project on device
        mels = mels.to(self.device, non_blocking=True)
        return self.aud_head(mels)