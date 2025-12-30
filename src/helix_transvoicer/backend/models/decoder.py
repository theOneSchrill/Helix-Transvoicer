"""
Helix Transvoicer - Voice decoder model.

Generates mel spectrogram from content features and speaker embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTransposeBlock(nn.Module):
    """Transposed convolutional block for upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()

        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class AdaptiveInstanceNorm(nn.Module):
    """Adaptive Instance Normalization for style injection."""

    def __init__(self, num_features: int, style_dim: int):
        super().__init__()

        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.style_proj = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # Get style parameters
        style_params = self.style_proj(style)
        gamma, beta = style_params.chunk(2, dim=-1)

        # Reshape for broadcasting
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        # Apply adaptive normalization
        x = self.norm(x)
        return x * (1 + gamma) + beta


class StyleBlock(nn.Module):
    """Decoder block with style injection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)

        self.adain1 = AdaptiveInstanceNorm(out_channels, style_dim)
        self.adain2 = AdaptiveInstanceNorm(out_channels, style_dim)

        self.activation = nn.LeakyReLU(0.2)

        # Skip connection if dimensions don't match
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        x = self.conv1(x)
        x = self.adain1(x, style)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.adain2(x, style)

        return self.activation(x + residual)


class VoiceDecoder(nn.Module):
    """
    Voice decoder for generating mel spectrogram.

    Takes content features and speaker embedding, outputs mel spectrogram.

    Architecture:
    - Transformer decoder for content processing
    - Style blocks for speaker identity injection
    - Mel spectrogram output projection
    """

    def __init__(
        self,
        content_dim: int = 256,
        speaker_dim: int = 256,
        hidden_dim: int = 512,
        n_mels: int = 80,
        num_layers: int = 4,
    ):
        super().__init__()

        self.content_dim = content_dim
        self.speaker_dim = speaker_dim
        self.hidden_dim = hidden_dim

        # Content projection
        self.content_proj = nn.Linear(content_dim, hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Style blocks for speaker injection
        self.style_blocks = nn.ModuleList([
            StyleBlock(hidden_dim, hidden_dim, speaker_dim)
            for _ in range(4)
        ])

        # Mel projection
        self.mel_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_mels),
        )

        # Postnet for mel refinement
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(256, n_mels, kernel_size=5, padding=2),
        )

    def forward(
        self,
        content: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate mel spectrogram from content and speaker.

        Args:
            content: Content features [batch, time, content_dim]
            speaker_embedding: Speaker embedding [batch, speaker_dim]

        Returns:
            Mel spectrogram [batch, time, n_mels]
        """
        # Project content
        x = self.content_proj(content)

        # Create memory for transformer (same as input for self-attention)
        memory = x

        # Transformer decoding
        x = self.transformer(x, memory)

        # Transpose for conv layers [batch, hidden, time]
        x = x.transpose(1, 2)

        # Style injection
        for style_block in self.style_blocks:
            x = style_block(x, speaker_embedding)

        # Back to [batch, time, hidden]
        x = x.transpose(1, 2)

        # Mel projection
        mel = self.mel_proj(x)

        # Postnet refinement
        mel_residual = self.postnet(mel.transpose(1, 2)).transpose(1, 2)
        mel = mel + mel_residual

        return mel

    def inference(
        self,
        content: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Inference mode with no gradient computation."""
        self.eval()
        with torch.no_grad():
            return self.forward(content, speaker_embedding)
