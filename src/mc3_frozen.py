# src/mc3_frozen.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta
from torchvision.models.video import mc3_18, MC3_18_Weights
from torch.backends.cuda import is_bf16_supported


_CHANNELS_LAST_3D = getattr(torch, "channels_last_3d", torch.channels_last)

class L2Norm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

class AudioProjector(nn.Module):
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
        self.param_dtype = torch.float32

    @torch.inference_mode()
    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        x = mels.unsqueeze(1)
        return self.proj(x)

def _fp16_except_norms(m: nn.Module):
    """Convert Conv/Linear to half, keep norms in float32 for stability."""
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            mod.float()
        elif isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            mod.half()

'''Helper for ensure_on_device()'''
def _fp32_module(m: torch.nn.Module):
    for p in m.parameters(recurse=True):
        if p.dtype.is_floating_point:
            p.data = p.data.float()
    for b in m.buffers(recurse=True):
        if b.dtype.is_floating_point:
            b.data = b.data.float()

class FrozenMC3(nn.Module):
    def __init__(self, device: str | None = None, vid_emb_dim: int = 512, aud_emb_dim: int = 512):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        weights = MC3_18_Weights.KINETICS400_V1
        self.vid_backbone = mc3_18(weights=weights)
        self.vid_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(512, vid_emb_dim, bias=True),
            L2Norm(),
        )
        self.vid_backbone.to(memory_format=_CHANNELS_LAST_3D)

        self.melspec = ta.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=64, f_min=50, f_max=8000
        )
        self.db = ta.transforms.AmplitudeToDB()
        self.aud_head = AudioProjector(aud_emb_dim)

        self.register_buffer("img_mean", torch.tensor([0.43216, 0.394666, 0.37645]).view(1,3,1,1,1), persistent=False)
        self.register_buffer("img_std",  torch.tensor([0.22803, 0.22145, 0.216989]).view(1,3,1,1,1), persistent=False)

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        self._on_device = False


    def _ensure_on_device(self):
        
        if self._on_device:
            return
        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # ---- VIDEO in mixed precision (fp16), channels-last 3D ----
            _fp16_except_norms(self.vid_backbone)
            self.vid_backbone.to(self.device, memory_format=_CHANNELS_LAST_3D, non_blocking=True)
            self.vid_head.to(self.device, memory_format=_CHANNELS_LAST_3D, non_blocking=True)

            # ---- AUDIO in fp32 ----
            _fp32_module(self.aud_head)
            self.aud_head.to(self.device, dtype=torch.float32, non_blocking=True)

            # mel transforms are ops with buffers; keep them fp32
            self.melspec.to(self.device)
            self.db.to(self.device)

            # stats
            self.img_mean = self.img_mean.to(self.device, non_blocking=True)
            self.img_std  = self.img_std.to(self.device, non_blocking=True)

            self._on_device = True

        except torch.cuda.OutOfMemoryError as e:
            free, total = (torch.cuda.mem_get_info() if self.device.type == "cuda" else (0,0))
            raise RuntimeError(
                f"OOM moving MC3 to {self.device}. Free/Total={free/1e9:.2f}/{total/1e9:.2f} GB. "
                f"Try lower --batch_size/--size/--win and set "
                f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128"
            ) from e

    @staticmethod
    def _to_5d_chw(video: torch.Tensor) -> torch.Tensor:
        assert video.dim() == 5, f"Expected 5D video, got shape {tuple(video.shape)}"
        if video.shape[1] == 3:
            return video
        if video.shape[2] == 3:
            return video.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError(f"Cannot infer channel dim from shape {tuple(video.shape)}")

    @staticmethod
    def _maybe_scale_uint(video: torch.Tensor) -> torch.Tensor:
        if video.dtype == torch.uint8:
            return video.float() / 255.0
        return video

    @torch.inference_mode()
    def encode_video(self, clip: torch.Tensor) -> torch.Tensor:
        self._ensure_on_device()

        x = self._to_5d_chw(self._maybe_scale_uint(clip))
        # Move to GPU first
        x = x.to(self.device, memory_format=_CHANNELS_LAST_3D, non_blocking=True)

        use_amp = (self.device.type == "cuda")
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # match dtype so broadcasting is safe under AMP
                mean = self.img_mean.to(x.dtype)
                std  = (self.img_std.to(x.dtype) + 1e-8)
                x = (x - mean) / std

                feats = self._forward_backbone(x)
        else:
            mean = self.img_mean.to(x.dtype)
            std  = (self.img_std.to(x.dtype) + 1e-8)
            x = (x - mean) / std

            feats = self._forward_backbone(x)

        out = self.vid_head(feats)
        return out

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vid_backbone.stem(x)
        feats = self.vid_backbone.layer1(feats)
        feats = self.vid_backbone.layer2(feats)
        feats = self.vid_backbone.layer3(feats)
        feats = self.vid_backbone.layer4(feats)
        return feats
    

    @torch.inference_mode()
    def encode_audio(self, wav):
        self._ensure_on_device()

        wav = wav.to(self.device, dtype=torch.float32)           # (B, L)
        mels = self.melspec(wav)                                 # (B, n_mels, T)
        mels = self.db(mels)

        # per-clip standardization
        m = mels.mean(dim=(1, 2), keepdim=True)
        s = mels.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        mels = (mels - m) / s                                    # (B, n_mels, T)

        # DO NOT unsqueeze here (forward() already does it)
        return self.aud_head(mels)                               # (B, 512)