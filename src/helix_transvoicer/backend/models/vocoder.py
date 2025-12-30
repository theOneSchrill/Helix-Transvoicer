"""
Helix Transvoicer - Neural vocoder (HiFi-GAN style).

Converts mel spectrogram to audio waveform.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for 'same' convolution."""
    return (kernel_size * dilation - dilation) // 2


class ResBlock(nn.Module):
    """Residual block for HiFi-GAN generator."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple = (1, 3, 5),
    ):
        super().__init__()

        self.convs1 = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=d,
                padding=get_padding(kernel_size, d),
            )
            for d in dilations
        ])

        self.convs2 = nn.ModuleList([
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            )
            for _ in dilations
        ])

        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = self.activation(x)
            xt = conv1(xt)
            xt = self.activation(xt)
            xt = conv2(xt)
            x = xt + x
        return x


class MultiReceptiveFieldFusion(nn.Module):
    """Multi-receptive field fusion module."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: tuple = (3, 7, 11),
        dilations: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()

        self.resblocks = nn.ModuleList([
            ResBlock(channels, k, d)
            for k, d in zip(kernel_sizes, dilations)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for resblock in self.resblocks:
            if out is None:
                out = resblock(x)
            else:
                out = out + resblock(x)
        return out / len(self.resblocks)


class Vocoder(nn.Module):
    """
    HiFi-GAN style neural vocoder.

    Converts mel spectrogram to audio waveform using:
    - Transposed convolutions for upsampling
    - Multi-receptive field fusion for quality
    - Residual connections
    """

    def __init__(
        self,
        n_mels: int = 80,
        upsample_rates: tuple = (8, 8, 2, 2),
        upsample_kernel_sizes: tuple = (16, 16, 4, 4),
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: tuple = (3, 7, 11),
        resblock_dilations: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        # Initial convolution
        self.conv_pre = nn.Conv1d(n_mels, upsample_initial_channel, 7, padding=3)

        # Upsample layers
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    ch,
                    ch // 2,
                    k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )
            ch = ch // 2
            self.mrfs.append(
                MultiReceptiveFieldFusion(ch, resblock_kernel_sizes, resblock_dilations)
            )

        # Output convolution
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3)

        self.activation = nn.LeakyReLU(0.1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel spectrogram to audio.

        Args:
            mel: Mel spectrogram [batch, time, n_mels] or [batch, n_mels, time]

        Returns:
            Audio waveform [batch, 1, samples]
        """
        # Ensure correct shape [batch, n_mels, time]
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        if mel.shape[1] != self.conv_pre.in_channels:
            mel = mel.transpose(1, 2)

        # Initial convolution
        x = self.conv_pre(mel)

        # Upsample with MRF fusion
        for up, mrf in zip(self.ups, self.mrfs):
            x = self.activation(x)
            x = up(x)
            x = mrf(x)

        # Output convolution
        x = self.activation(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def inference(self, mel: torch.Tensor) -> torch.Tensor:
        """Inference mode."""
        self.eval()
        with torch.no_grad():
            return self.forward(mel)

    @staticmethod
    def get_upsample_factor(upsample_rates: tuple = (8, 8, 2, 2)) -> int:
        """Calculate total upsample factor."""
        factor = 1
        for rate in upsample_rates:
            factor *= rate
        return factor
