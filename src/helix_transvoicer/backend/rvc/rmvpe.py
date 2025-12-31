"""
RMVPE - Robust Model for Vocal Pitch Estimation.

A U-Net style architecture for accurate pitch extraction,
especially effective with noisy audio.
"""

import logging
from typing import Optional

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("helix.rvc.rmvpe")


class BiGRU(nn.Module):
    """Bidirectional GRU layer."""

    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    """Residual convolutional block."""

    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Encoder(nn.Module):
    """U-Net encoder."""

    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels, momentum=0.01):
        super().__init__()

        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)

        self.layers = nn.ModuleList()
        self.latent_channels = []

        for i in range(n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels if i == 0 else out_channels * (2 ** (i - 1)),
                    out_channels * (2 ** i),
                    kernel_size,
                    n_blocks,
                    momentum,
                )
            )
            self.latent_channels.append(out_channels * (2 ** i))

    def forward(self, x):
        x = self.bn(x)
        concat_tensors = []

        for layer in self.layers:
            x, t = layer(x)
            concat_tensors.append(t)

        return x, concat_tensors


class ResEncoderBlock(nn.Module):
    """Residual encoder block with downsampling."""

    def __init__(self, in_channels, out_channels, kernel_size, n_blocks, momentum):
        super().__init__()

        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))

        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))

        self.kernel_size = kernel_size
        self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return self.pool(x), x


class ResDecoderBlock(nn.Module):
    """Residual decoder block with upsampling."""

    def __init__(self, in_channels, out_channels, stride, n_blocks, momentum):
        super().__init__()

        self.n_blocks = n_blocks

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(stride[0] + 1, stride[1] + 1),
            stride=stride,
            padding=(1, 1),
            output_padding=(0, 0),
        )

        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(out_channels * 2, out_channels, momentum))

        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.up(x)

        # Handle size mismatch
        diff_h = concat_tensor.size(2) - x.size(2)
        diff_w = concat_tensor.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([x, concat_tensor], dim=1)

        for conv in self.conv:
            x = conv(x)

        return x


class Decoder(nn.Module):
    """U-Net decoder."""

    def __init__(self, in_channels, n_decoders, stride, n_blocks, out_channels, momentum=0.01):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(n_decoders):
            self.layers.append(
                ResDecoderBlock(
                    in_channels // (2 ** i) if i == 0 else out_channels * (2 ** (n_decoders - i)),
                    out_channels * (2 ** (n_decoders - i - 1)) if i < n_decoders - 1 else out_channels,
                    stride,
                    n_blocks,
                    momentum,
                )
            )

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-(i + 1)])
        return x


class DeepUnet(nn.Module):
    """Deep U-Net architecture for RMVPE."""

    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels,
            128,
            en_de_layers,
            kernel_size,
            n_blocks,
            en_out_channels,
        )

        self.intermediate = nn.Sequential(
            *[
                ConvBlockRes(
                    en_out_channels * (2 ** (en_de_layers - 1)),
                    en_out_channels * (2 ** (en_de_layers - 1)),
                )
                for _ in range(inter_layers)
            ]
        )

        self.decoder = Decoder(
            en_out_channels * (2 ** (en_de_layers - 1)),
            en_de_layers,
            kernel_size,
            n_blocks,
            en_out_channels,
        )

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class RMVPE(nn.Module):
    """RMVPE - Robust Model for Vocal Pitch Estimation."""

    def __init__(self, checkpoint: dict):
        super().__init__()

        self.mel_extractor = MelSpectrogram(
            n_mels=128,
            sample_rate=16000,
            win_length=1024,
            hop_length=160,
            f_min=30,
            f_max=8000,
        )

        self.unet = DeepUnet(
            kernel_size=(2, 2),
            n_blocks=2,
            en_de_layers=5,
            inter_layers=4,
            in_channels=1,
            en_out_channels=16,
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=(1, 1)),
            nn.Sigmoid(),
        )

        # Load weights if available
        if "model" in checkpoint:
            self.load_state_dict(checkpoint["model"], strict=False)

        self.cents_mapping = 20 * np.arange(360) + 1997.3794084376191

    def forward(self, mel):
        """Forward pass."""
        x = self.unet(mel)
        x = self.output_layer(x)
        return x

    @torch.no_grad()
    def infer_from_audio(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract pitch from audio.

        Args:
            audio: Audio as numpy array
            sr: Sample rate

        Returns:
            F0 contour in Hz
        """
        # Resample if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Compute mel spectrogram
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

        if next(self.parameters()).is_cuda:
            audio_tensor = audio_tensor.cuda()

        mel = self.mel_extractor(audio_tensor)

        # Run model
        output = self(mel)

        # Convert to frequency
        output = output.squeeze().cpu().numpy()

        # Find peak in each frame
        f0 = np.zeros(output.shape[0])
        for i in range(output.shape[0]):
            frame = output[i]
            if frame.max() > 0.1:  # Threshold
                peak_idx = frame.argmax()
                f0[i] = self.cents_mapping[peak_idx] / 100  # Convert cents to Hz
            else:
                f0[i] = 0

        return f0


class MelSpectrogram(nn.Module):
    """Mel spectrogram extractor."""

    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        win_length: int = 1024,
        hop_length: int = 160,
        f_min: float = 30,
        f_max: float = 8000,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max

        # Create mel filterbank
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=win_length,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

        # Hann window
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram."""
        # STFT
        stft = torch.stft(
            audio,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

        # Magnitude
        magnitude = torch.abs(stft)

        # Mel
        mel = torch.matmul(self.mel_basis, magnitude)

        # Log scale
        mel = torch.log(mel.clamp(min=1e-5))

        # Add channel dimension
        mel = mel.unsqueeze(1)

        return mel
