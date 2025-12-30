"""
Helix Transvoicer - Voice encoder models.

Content Encoder: Extracts content/phonetic features (speaker-independent)
Speaker Encoder: Extracts speaker identity embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
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


class ResidualBlock(nn.Module):
    """Residual block for encoders."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class ContentEncoder(nn.Module):
    """
    Content encoder for extracting speaker-independent content features.

    Architecture:
    - Convolutional frontend for spectral feature extraction
    - Transformer encoder for temporal modeling
    - Outputs phonetic/content representation
    """

    def __init__(
        self,
        input_dim: int = 1,  # Raw audio input
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()

        # Audio to spectrogram-like representation
        self.frontend = nn.Sequential(
            ConvBlock(input_dim, 32, kernel_size=7, stride=2, padding=3),
            ConvBlock(32, 64, kernel_size=5, stride=2, padding=2),
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBlock(128, hidden_dim, kernel_size=3, stride=2, padding=1),
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(4)]
        )

        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract content features from audio.

        Args:
            audio: Raw audio tensor [batch, samples] or [batch, 1, samples]

        Returns:
            Content features [batch, time, output_dim]
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Frontend
        x = self.frontend(audio)

        # Residual blocks
        x = self.residual_blocks(x)

        # Transpose for transformer [batch, time, features]
        x = x.transpose(1, 2)

        # Transformer encoding
        x = self.transformer(x)

        # Output projection
        return self.output_proj(x)


class SpeakerEncoder(nn.Module):
    """
    Speaker encoder for extracting speaker identity embedding.

    Architecture:
    - Convolutional layers for spectral processing
    - Recurrent layers for temporal aggregation
    - Attention-based pooling
    - Outputs fixed-size speaker embedding
    """

    def __init__(
        self,
        input_dim: int = 80,  # Mel spectrogram bins
        hidden_dim: int = 256,
        embedding_dim: int = 256,
    ):
        super().__init__()

        # Convolutional frontend
        self.conv_layers = nn.Sequential(
            ConvBlock(input_dim, 128, kernel_size=3),
            ResidualBlock(128),
            ConvBlock(128, 192, kernel_size=3, stride=2),
            ResidualBlock(192),
            ConvBlock(192, hidden_dim, kernel_size=3, stride=2),
            ResidualBlock(hidden_dim),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        # Attention for temporal pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Output embedding projection
        self.embedding_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from mel spectrogram.

        Args:
            mel: Mel spectrogram [batch, n_mels, time] or [batch, time, n_mels]

        Returns:
            Speaker embedding [batch, embedding_dim]
        """
        # Ensure correct shape [batch, n_mels, time]
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        if mel.shape[1] > mel.shape[2]:
            mel = mel.transpose(1, 2)

        # Convolutional layers
        x = self.conv_layers(mel)

        # Transpose for LSTM [batch, time, features]
        x = x.transpose(1, 2)

        # LSTM
        x, _ = self.lstm(x)

        # Attention-based pooling
        attention_weights = self.attention(x)
        attention_weights = F.softmax(attention_weights, dim=1)
        x = torch.sum(x * attention_weights, dim=1)

        # Project to embedding
        embedding = self.embedding_proj(x)

        # L2 normalize
        return F.normalize(embedding, p=2, dim=-1)

    def compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between embeddings."""
        return F.cosine_similarity(embedding1, embedding2, dim=-1)
