# src/mc3_frozen.py
import torch
import torch.nn as nn
import torchaudio as ta
from torchvision.models.video import mc3_18, MC3_18_Weights

class L2Norm(nn.Module):
    def forward(self, x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

class AudioProjector(nn.Module):
    """Simple placeholder audio encoder using log-mel features."""
    def __init__(self, emb_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten(),
            nn.Linear(32*16*16, emb_dim),
            L2Norm(),
        )

    @torch.inference_mode()
    def forward(self, mels):
        x = mels.unsqueeze(1)  # (B,1,n_mels,T)
        return self.proj(x)

class FrozenMC3(nn.Module):
    """Frozen video/audio encoders for SoundingActions-style experiments."""
    def __init__(self, device=None, vid_emb_dim=512, aud_emb_dim=512):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ---- VIDEO ENCODER (torchvision MC3-18 as baseline) ----
        weights = MC3_18_Weights.KINETICS400_V1
        model = mc3_18(weights=weights)
        self.vid_backbone = model
        self.vid_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(512, vid_emb_dim),
            L2Norm(),
        )

        # ---- AUDIO ENCODER ----
        self.melspec = ta.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=64, f_min=50, f_max=8000
        )
        self.db = ta.transforms.AmplitudeToDB()
        self.aud_head = AudioProjector(aud_emb_dim)

        self.eval().to(self.device)
        for p in self.parameters():
            p.requires_grad = False

        # Preprocessing stats for normalization
        self.img_mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1, 3, 1, 1)
        self.img_std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def encode_video(self, clip):
        """clip: (B,C,T,H,W) tensor in [0,1]."""
        x = (clip - self.img_mean) / self.img_std
        feats = self.vid_backbone.stem(x)
        feats = self.vid_backbone.layer1(feats)
        feats = self.vid_backbone.layer2(feats)
        feats = self.vid_backbone.layer3(feats)
        feats = self.vid_backbone.layer4(feats)
        return self.vid_head(feats)

    @torch.inference_mode()
    def encode_audio(self, wav):
        """wav: (B,16000) tensor at 16kHz."""
        mels = self.db(self.melspec(wav))
        mels = (mels - mels.mean()) / (mels.std() + 1e-6)
        return self.aud_head(mels)
