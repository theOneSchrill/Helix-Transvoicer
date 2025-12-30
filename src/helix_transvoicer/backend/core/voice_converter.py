"""
Helix Transvoicer - Voice conversion pipeline.

Converts source audio to target voice while preserving content and timing.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helix_transvoicer.backend.core.audio_processor import AudioProcessor, ProcessingConfig
from helix_transvoicer.backend.models.encoder import ContentEncoder, SpeakerEncoder
from helix_transvoicer.backend.models.decoder import VoiceDecoder
from helix_transvoicer.backend.models.vocoder import Vocoder
from helix_transvoicer.backend.utils.audio import AudioUtils
from helix_transvoicer.backend.utils.config import get_settings

logger = logging.getLogger("helix.voice_converter")


@dataclass
class ConversionConfig:
    """Voice conversion configuration."""

    pitch_shift: float = 0.0  # semitones (-12 to +12)
    formant_shift: float = 0.0  # (-1.0 to +1.0)
    smoothing: float = 0.5  # (0 to 1)
    crossfade_ms: float = 20.0  # milliseconds
    preserve_breath: bool = True
    preserve_background: bool = False
    normalize_output: bool = True


@dataclass
class ConversionResult:
    """Result of voice conversion."""

    audio: np.ndarray
    sample_rate: int
    duration: float
    source_duration: float
    model_id: str
    config: ConversionConfig
    processing_time: float
    metadata: Dict = field(default_factory=dict)


class VoiceConverter:
    """
    Voice conversion engine.

    Pipeline:
    1. Preprocess source audio
    2. Extract content features (PPG-like representation)
    3. Extract/load target speaker embedding
    4. Decode to target voice
    5. Vocode to waveform
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        models_dir: Optional[Path] = None,
    ):
        self.settings = get_settings()
        self.device = device or torch.device("cpu")
        self.models_dir = models_dir or self.settings.models_dir

        self.audio_processor = AudioProcessor()

        # Initialize models (lazy loading)
        self._content_encoder: Optional[ContentEncoder] = None
        self._speaker_encoder: Optional[SpeakerEncoder] = None
        self._decoder: Optional[VoiceDecoder] = None
        self._vocoder: Optional[Vocoder] = None

        # Cache for loaded speaker embeddings
        self._speaker_cache: Dict[str, torch.Tensor] = {}

    @property
    def content_encoder(self) -> ContentEncoder:
        """Lazy-load content encoder."""
        if self._content_encoder is None:
            self._content_encoder = ContentEncoder().to(self.device)
            self._content_encoder.eval()
        return self._content_encoder

    @property
    def speaker_encoder(self) -> SpeakerEncoder:
        """Lazy-load speaker encoder."""
        if self._speaker_encoder is None:
            self._speaker_encoder = SpeakerEncoder().to(self.device)
            self._speaker_encoder.eval()
        return self._speaker_encoder

    @property
    def decoder(self) -> VoiceDecoder:
        """Lazy-load voice decoder."""
        if self._decoder is None:
            self._decoder = VoiceDecoder().to(self.device)
            self._decoder.eval()
        return self._decoder

    @property
    def vocoder(self) -> Vocoder:
        """Lazy-load vocoder."""
        if self._vocoder is None:
            self._vocoder = Vocoder().to(self.device)
            self._vocoder.eval()
        return self._vocoder

    def convert(
        self,
        source_audio: Union[str, Path, np.ndarray],
        target_model_id: str,
        config: Optional[ConversionConfig] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ConversionResult:
        """
        Convert source audio to target voice.

        Args:
            source_audio: Path to audio file or audio array
            target_model_id: ID of target voice model
            config: Conversion configuration
            progress_callback: Optional callback for progress updates

        Returns:
            ConversionResult with converted audio
        """
        import time

        start_time = time.time()
        cfg = config or ConversionConfig()

        def update_progress(stage: str, progress: float):
            if progress_callback:
                progress_callback(stage, progress)

        update_progress("Loading audio", 0.0)

        # Load and preprocess source audio
        if isinstance(source_audio, (str, Path)):
            processed = self.audio_processor.process(source_audio)
            audio = processed.audio
            sr = processed.sample_rate
            source_duration = processed.original_duration
        else:
            audio = source_audio
            sr = self.settings.sample_rate
            source_duration = len(audio) / sr

        update_progress("Extracting features", 0.2)

        # Extract content features
        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            content_features = self.content_encoder(audio_tensor)

        update_progress("Loading voice model", 0.4)

        # Get target speaker embedding
        speaker_embedding = self._get_speaker_embedding(target_model_id)

        update_progress("Converting voice", 0.5)

        # Apply pitch shift if specified
        if cfg.pitch_shift != 0:
            content_features = self._apply_pitch_shift(
                content_features,
                cfg.pitch_shift,
            )

        # Decode to mel spectrogram
        with torch.no_grad():
            mel_output = self.decoder(content_features, speaker_embedding)

        update_progress("Synthesizing audio", 0.7)

        # Vocode to waveform
        with torch.no_grad():
            audio_output = self.vocoder(mel_output)
            audio_output = audio_output.squeeze().cpu().numpy()

        update_progress("Post-processing", 0.9)

        # Apply smoothing
        if cfg.smoothing > 0:
            audio_output = self._apply_smoothing(audio_output, cfg.smoothing)

        # Normalize output
        if cfg.normalize_output:
            audio_output = AudioUtils.normalize(audio_output)

        # Apply crossfade at segment boundaries
        if cfg.crossfade_ms > 0:
            audio_output = self._apply_crossfade(
                audio_output,
                sr,
                cfg.crossfade_ms,
            )

        processing_time = time.time() - start_time

        update_progress("Complete", 1.0)

        return ConversionResult(
            audio=audio_output,
            sample_rate=sr,
            duration=len(audio_output) / sr,
            source_duration=source_duration,
            model_id=target_model_id,
            config=cfg,
            processing_time=processing_time,
            metadata={
                "device": str(self.device),
            },
        )

    def _get_speaker_embedding(self, model_id: str) -> torch.Tensor:
        """Load or compute speaker embedding for model."""
        if model_id in self._speaker_cache:
            return self._speaker_cache[model_id]

        # Try to load from model directory
        model_dir = self.models_dir / model_id
        embedding_path = model_dir / "speaker_embedding.npy"

        if embedding_path.exists():
            embedding = np.load(str(embedding_path))
            embedding_tensor = torch.from_numpy(embedding).float().to(self.device)
            self._speaker_cache[model_id] = embedding_tensor
            return embedding_tensor

        # Generate default embedding if model not found
        logger.warning(f"Speaker embedding not found for {model_id}, using default")
        default_embedding = torch.zeros(256).to(self.device)
        return default_embedding

    def _apply_pitch_shift(
        self,
        content_features: torch.Tensor,
        semitones: float,
    ) -> torch.Tensor:
        """Apply pitch shift to content features."""
        # Pitch shift is applied by modifying the F0 component
        # This is a simplified implementation
        pitch_factor = 2 ** (semitones / 12.0)

        # Assuming content features have a pitch channel
        # In practice, this would be more sophisticated
        return content_features * pitch_factor

    def _apply_smoothing(self, audio: np.ndarray, strength: float) -> np.ndarray:
        """Apply temporal smoothing to reduce artifacts."""
        from scipy.ndimage import gaussian_filter1d

        sigma = strength * 3
        if sigma > 0:
            # Apply light smoothing to reduce high-frequency artifacts
            audio = gaussian_filter1d(audio, sigma=sigma)
        return audio

    def _apply_crossfade(
        self,
        audio: np.ndarray,
        sr: int,
        crossfade_ms: float,
    ) -> np.ndarray:
        """Apply crossfade at the edges."""
        fade_samples = int(sr * crossfade_ms / 1000)

        if fade_samples > 0 and len(audio) > 2 * fade_samples:
            # Fade in
            fade_in = np.linspace(0, 1, fade_samples)
            audio[:fade_samples] *= fade_in

            # Fade out
            fade_out = np.linspace(1, 0, fade_samples)
            audio[-fade_samples:] *= fade_out

        return audio

    def clear_cache(self):
        """Clear speaker embedding cache."""
        self._speaker_cache.clear()

    def set_device(self, device: torch.device):
        """Change compute device."""
        self.device = device

        if self._content_encoder is not None:
            self._content_encoder = self._content_encoder.to(device)
        if self._speaker_encoder is not None:
            self._speaker_encoder = self._speaker_encoder.to(device)
        if self._decoder is not None:
            self._decoder = self._decoder.to(device)
        if self._vocoder is not None:
            self._vocoder = self._vocoder.to(device)

        # Update cached embeddings
        for key in self._speaker_cache:
            self._speaker_cache[key] = self._speaker_cache[key].to(device)
