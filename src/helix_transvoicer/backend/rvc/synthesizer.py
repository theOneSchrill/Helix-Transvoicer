"""
RVC Synthesizer - Voice conversion synthesis module.

Based on the SynthesizerTrnMs768NSFsid architecture from RVC.
"""

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

logger = logging.getLogger("helix.rvc.synthesizer")


class SynthesizerTrnMs768NSFsid(nn.Module):
    """
    RVC Synthesizer with NSF-HiFiGAN vocoder.

    Converts content features + pitch to audio waveform.
    """

    def __init__(
        self,
        spec_channels: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
    ):
        super().__init__()

        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.sr = sr
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim

        # Text encoder
        self.enc_p = TextEncoder768(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )

        # Decoder (generator)
        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels,
            sr,
        )

        # Flow
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            4,
            gin_channels,
        )

        # Speaker embedding
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def infer(
        self,
        feats: torch.Tensor,
        f0: torch.Tensor,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
    ) -> torch.Tensor:
        """
        Run inference to generate audio.

        Args:
            feats: Content features from HuBERT [B, T, 768]
            f0: Pitch contour [B, T]
            rms_mix_rate: RMS volume mixing rate
            protect: Voiceless consonant protection

        Returns:
            Audio waveform [B, samples]
        """
        # Encode content
        x, m_p, logs_p, x_mask = self.enc_p(feats.transpose(1, 2))

        # Create speaker embedding (use default speaker 0)
        g = self.emb_g(torch.zeros(feats.size(0), dtype=torch.long, device=feats.device))
        g = g.unsqueeze(-1)

        # Apply flow
        z = m_p + torch.randn_like(m_p) * torch.exp(logs_p)
        z = self.flow(z, x_mask, g=g, reverse=True)

        # Decode to audio
        o = self.dec(z * x_mask, f0, g=g)

        return o


class TextEncoder768(nn.Module):
    """Text/Content encoder for 768-dim features (HuBERT)."""

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        # Input projection from 768-dim HuBERT
        self.pre = nn.Conv1d(768, hidden_channels, 1)

        # Transformer layers
        self.attn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(hidden_channels, n_heads, dropout=p_dropout, batch_first=True)
            )
            self.norm_layers.append(nn.LayerNorm(hidden_channels))
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channels, filter_channels, kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.Conv1d(filter_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
                )
            )

        # Output projection
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Encode content features.

        Args:
            x: Input features [B, C, T]

        Returns:
            (output, mean, log_var, mask)
        """
        x_mask = torch.ones(x.size(0), 1, x.size(2), device=x.device)

        x = self.pre(x)

        # Apply transformer layers
        for attn, norm, ffn in zip(self.attn_layers, self.norm_layers, self.ffn_layers):
            # Self-attention
            x_t = x.transpose(1, 2)
            attn_out, _ = attn(x_t, x_t, x_t)
            x = x + attn_out.transpose(1, 2)
            x_t = norm(x.transpose(1, 2)).transpose(1, 2)

            # FFN
            x = x + ffn(x_t)

        # Project to mean and log variance
        stats = self.proj(x)
        m, logs = torch.split(stats, self.out_channels, dim=1)

        return x, m, logs, x_mask


class GeneratorNSF(nn.Module):
    """NSF-HiFiGAN Generator for waveform synthesis."""

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        gin_channels: int,
        sr: int,
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.sr = sr

        # Initial convolution
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, padding=3)

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        # Output convolution
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3, bias=False)

        # Speaker conditioning
        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate audio waveform.

        Args:
            x: Input features [B, C, T]
            f0: Pitch contour [B, T]
            g: Speaker embedding [B, C, 1]

        Returns:
            Audio waveform [B, 1, samples]
        """
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class ResBlock(nn.Module):
    """Residual block for HiFiGAN."""

    def __init__(self, channels: int, kernel_size: int, dilation: List[int]):
        super().__init__()

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilation:
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=d,
                        padding=(kernel_size * d - d) // 2,
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class ResidualCouplingBlock(nn.Module):
    """Residual coupling block for normalizing flow."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.n_flows = 4

        self.flows = nn.ModuleList()
        for i in range(self.n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class ResidualCouplingLayer(nn.Module):
    """Single residual coupling layer."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        mean_only: bool = False,
    ):
        super().__init__()

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (1 if mean_only else 2), 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ):
        x0, x1 = torch.split(x, self.half_channels, dim=1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask

        if self.mean_only:
            m = stats
            logs = torch.zeros_like(m)
        else:
            m, logs = torch.split(stats, self.half_channels, dim=1)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            return torch.cat([x0, x1], dim=1), logs.sum([1, 2])
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            return torch.cat([x0, x1], dim=1)


class WN(nn.Module):
    """WaveNet-style network."""

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding)
            )
            self.res_skip_layers.append(nn.Conv1d(hidden_channels, 2 * hidden_channels, 1))

        if gin_channels > 0:
            self.cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None):
        output = torch.zeros_like(x)

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)

            if g is not None:
                cond_offset = i * 2 * x.size(1)
                g_l = g[:, cond_offset : cond_offset + 2 * x.size(1), :]
                x_in = x_in + g_l

            acts = torch.tanh(x_in[:, : x.size(1)]) * torch.sigmoid(x_in[:, x.size(1) :])

            res_skip_acts = self.res_skip_layers[i](acts)
            x = (x + res_skip_acts[:, : x.size(1)]) * x_mask
            output = output + res_skip_acts[:, x.size(1) :]

        return output * x_mask


class Flip(nn.Module):
    """Flip layer for flow."""

    def forward(self, x: torch.Tensor, *args, reverse: bool = False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            return x, torch.zeros(x.size(0), device=x.device)
        return x
