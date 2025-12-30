"""
Helix Transvoicer - Voice conversion pipeline.

Converts source audio to target voice while preserving content and timing.
"""

import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

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

# Maximum chunk size in samples for 8GB GPUs (about 3 seconds at 22050 Hz)
MAX_CHUNK_SAMPLES = 22050 * 3

# Low memory threshold in GB - use chunked processing below this
LOW_MEMORY_THRESHOLD_GB = 10.0


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

        # Architecture config cache (model_id -> config)
        self._arch_cache: Dict[str, Dict] = {}

        # Current architecture dimensions
        self._current_content_dim: int = 256
        self._current_speaker_dim: int = 256

        # Check available GPU memory
        self._low_memory_mode = self._check_low_memory()

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

    def _check_low_memory(self) -> bool:
        """Check if we're running in low memory mode."""
        if not torch.cuda.is_available():
            return False
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return total_memory < LOW_MEMORY_THRESHOLD_GB
        except Exception:
            return False

    def _clear_gpu_memory(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def _load_architecture_config(self, model_id: str) -> Optional[Dict]:
        """Load architecture config for a model."""
        if model_id in self._arch_cache:
            return self._arch_cache[model_id]

        model_path = self.models_dir / model_id

        # Try architecture.json first
        arch_path = model_path / "architecture.json"
        if arch_path.exists():
            try:
                with open(arch_path, "r") as f:
                    config = json.load(f)
                    self._arch_cache[model_id] = config
                    logger.info(f"Loaded architecture config for {model_id}")
                    return config
            except Exception as e:
                logger.warning(f"Failed to load architecture.json: {e}")

        # Fallback: Try to infer from speaker embedding
        embedding_path = model_path / "speaker_embedding.npy"
        if embedding_path.exists():
            try:
                embedding = np.load(str(embedding_path))
                embed_dim = embedding.shape[-1]

                # Infer if low memory mode based on embedding dimension
                if embed_dim == 64:
                    config = {
                        "low_memory_mode": True,
                        "content_encoder": {"hidden_dim": 64, "output_dim": 64, "num_layers": 1},
                        "speaker_encoder": {"hidden_dim": 64, "embedding_dim": 64},
                        "decoder": {"content_dim": 64, "speaker_dim": 64, "hidden_dim": 128, "num_layers": 1},
                    }
                    self._arch_cache[model_id] = config
                    logger.info(f"Inferred low_memory architecture from embedding dim={embed_dim}")
                    return config
            except Exception as e:
                logger.warning(f"Failed to infer architecture from embedding: {e}")

        return None

    def _get_content_encoder(self, arch_config: Optional[Dict] = None) -> ContentEncoder:
        """Get or create content encoder with correct architecture."""
        if arch_config and arch_config.get("low_memory_mode"):
            enc_config = arch_config.get("content_encoder", {})
            hidden_dim = enc_config.get("hidden_dim", 64)
            output_dim = enc_config.get("output_dim", 64)
            num_layers = enc_config.get("num_layers", 1)

            # Check if we need to recreate with different dimensions
            if (self._content_encoder is None or
                self._current_content_dim != output_dim):
                logger.info(f"Creating ContentEncoder with hidden_dim={hidden_dim}, output_dim={output_dim}")
                self._content_encoder = ContentEncoder(
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_layers=num_layers
                ).to(self.device)
                self._content_encoder.eval()
                self._current_content_dim = output_dim
        else:
            if self._content_encoder is None or self._current_content_dim != 256:
                self._content_encoder = ContentEncoder().to(self.device)
                self._content_encoder.eval()
                self._current_content_dim = 256

        return self._content_encoder

    def _get_decoder(self, arch_config: Optional[Dict] = None) -> VoiceDecoder:
        """Get or create decoder with correct architecture."""
        if arch_config and arch_config.get("low_memory_mode"):
            dec_config = arch_config.get("decoder", {})
            content_dim = dec_config.get("content_dim", 64)
            speaker_dim = dec_config.get("speaker_dim", 64)
            hidden_dim = dec_config.get("hidden_dim", 128)
            num_layers = dec_config.get("num_layers", 1)

            if (self._decoder is None or
                self._current_speaker_dim != speaker_dim):
                logger.info(f"Creating VoiceDecoder with content_dim={content_dim}, speaker_dim={speaker_dim}")
                self._decoder = VoiceDecoder(
                    content_dim=content_dim,
                    speaker_dim=speaker_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers
                ).to(self.device)
                self._decoder.eval()
                self._current_speaker_dim = speaker_dim
        else:
            if self._decoder is None or self._current_speaker_dim != 256:
                self._decoder = VoiceDecoder().to(self.device)
                self._decoder.eval()
                self._current_speaker_dim = 256

        return self._decoder

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

        # Clear GPU memory before starting
        self._clear_gpu_memory()

        # Load architecture config for target model
        arch_config = self._load_architecture_config(target_model_id)
        if arch_config:
            logger.info(f"Using architecture config for {target_model_id}: low_memory={arch_config.get('low_memory_mode', False)}")

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

        update_progress("Loading voice model", 0.2)

        # Get target speaker embedding
        speaker_embedding = self._get_speaker_embedding(target_model_id)

        # Get correctly configured encoder and decoder
        content_encoder = self._get_content_encoder(arch_config)
        decoder = self._get_decoder(arch_config)

        update_progress("Extracting features", 0.3)

        # Determine if we need chunked processing
        use_chunked = self._low_memory_mode or len(audio) > MAX_CHUNK_SAMPLES

        if use_chunked:
            logger.info(f"Using chunked processing (audio length: {len(audio)}, max chunk: {MAX_CHUNK_SAMPLES})")
            audio_output = self._convert_chunked(
                audio, sr, speaker_embedding, content_encoder, decoder,
                cfg, arch_config, update_progress
            )
        else:
            # Process entire audio at once
            audio_output = self._convert_single(
                audio, sr, speaker_embedding, content_encoder, decoder,
                cfg, update_progress
            )

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

        # Clear GPU memory after processing
        self._clear_gpu_memory()

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
                "chunked_processing": use_chunked,
                "low_memory_mode": arch_config.get("low_memory_mode", False) if arch_config else False,
            },
        )

    def _convert_single(
        self,
        audio: np.ndarray,
        sr: int,
        speaker_embedding: torch.Tensor,
        content_encoder: ContentEncoder,
        decoder: VoiceDecoder,
        cfg: ConversionConfig,
        update_progress: Callable[[str, float], None],
    ) -> np.ndarray:
        """Convert audio in a single pass (for shorter audio or high-memory GPUs)."""
        with torch.no_grad():
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            content_features = content_encoder(audio_tensor)

            update_progress("Converting voice", 0.5)

            # Apply pitch shift if specified
            if cfg.pitch_shift != 0:
                content_features = self._apply_pitch_shift(content_features, cfg.pitch_shift)

            # Decode to mel spectrogram
            mel_output = decoder(content_features, speaker_embedding)

            update_progress("Synthesizing audio", 0.7)

            # Vocode to waveform
            audio_output = self.vocoder(mel_output)
            audio_output = audio_output.squeeze().cpu().numpy()

        return audio_output

    def _convert_chunked(
        self,
        audio: np.ndarray,
        sr: int,
        speaker_embedding: torch.Tensor,
        content_encoder: ContentEncoder,
        decoder: VoiceDecoder,
        cfg: ConversionConfig,
        arch_config: Optional[Dict],
        update_progress: Callable[[str, float], None],
    ) -> np.ndarray:
        """Convert audio in chunks to avoid OOM on low-memory GPUs."""
        chunk_size = MAX_CHUNK_SAMPLES
        overlap = int(sr * cfg.crossfade_ms / 1000) * 2  # Overlap for crossfading

        # Calculate number of chunks
        num_samples = len(audio)
        chunks = []
        chunk_starts = []

        pos = 0
        while pos < num_samples:
            end = min(pos + chunk_size, num_samples)
            chunks.append(audio[pos:end])
            chunk_starts.append(pos)
            pos = end - overlap if end < num_samples else end

        logger.info(f"Processing {len(chunks)} chunks")

        output_chunks = []
        for i, chunk in enumerate(chunks):
            progress = 0.3 + (0.5 * (i / len(chunks)))
            update_progress(f"Processing chunk {i+1}/{len(chunks)}", progress)

            # Clear memory before each chunk
            self._clear_gpu_memory()

            with torch.no_grad():
                audio_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(self.device)
                content_features = content_encoder(audio_tensor)

                # Apply pitch shift if specified
                if cfg.pitch_shift != 0:
                    content_features = self._apply_pitch_shift(content_features, cfg.pitch_shift)

                # Decode to mel spectrogram
                mel_output = decoder(content_features, speaker_embedding)

                # Vocode to waveform
                chunk_output = self.vocoder(mel_output)
                chunk_output = chunk_output.squeeze().cpu().numpy()

                output_chunks.append(chunk_output)

                # Free tensors
                del audio_tensor, content_features, mel_output
                self._clear_gpu_memory()

        update_progress("Merging chunks", 0.8)

        # Merge chunks with crossfade
        if len(output_chunks) == 1:
            return output_chunks[0]

        return self._merge_chunks_with_crossfade(output_chunks, overlap, sr)

    def _merge_chunks_with_crossfade(
        self,
        chunks: List[np.ndarray],
        overlap: int,
        sr: int,
    ) -> np.ndarray:
        """Merge audio chunks with crossfade to avoid clicks."""
        if not chunks:
            return np.array([])

        if len(chunks) == 1:
            return chunks[0]

        # Calculate total length
        total_length = sum(len(c) for c in chunks) - overlap * (len(chunks) - 1)
        result = np.zeros(total_length)

        pos = 0
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: no fade in
                end_pos = len(chunk) - overlap
                result[:end_pos] = chunk[:end_pos]
                # Crossfade region
                fade_out = np.linspace(1, 0, overlap)
                result[end_pos:end_pos + overlap] = chunk[-overlap:] * fade_out
                pos = end_pos
            elif i == len(chunks) - 1:
                # Last chunk: fade in only
                fade_in = np.linspace(0, 1, overlap)
                result[pos:pos + overlap] += chunk[:overlap] * fade_in
                result[pos + overlap:pos + len(chunk) - overlap] = chunk[overlap:]
            else:
                # Middle chunk: fade in and fade out
                fade_in = np.linspace(0, 1, overlap)
                result[pos:pos + overlap] += chunk[:overlap] * fade_in
                end_pos = pos + len(chunk) - overlap
                result[pos + overlap:end_pos] = chunk[overlap:-overlap]
                fade_out = np.linspace(1, 0, overlap)
                result[end_pos:end_pos + overlap] = chunk[-overlap:] * fade_out
                pos = end_pos

        return result

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
