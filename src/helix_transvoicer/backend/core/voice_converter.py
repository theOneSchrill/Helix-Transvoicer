"""
Helix Transvoicer - Voice conversion pipeline.

Uses RVC (Retrieval-based Voice Conversion) for high-quality voice conversion.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import librosa
import numpy as np

from helix_transvoicer.backend.core.audio_processor import AudioProcessor
from helix_transvoicer.backend.utils.audio import AudioUtils
from helix_transvoicer.backend.utils.config import get_settings

logger = logging.getLogger("helix.voice_converter")

# Try to import RVC
try:
    from helix_transvoicer.backend.rvc.inference import RVCInference
    from helix_transvoicer.backend.rvc.models import RVCModelManager, check_rvc_ready
    RVC_AVAILABLE = True
except ImportError:
    RVC_AVAILABLE = False
    logger.warning("RVC module not available")


@dataclass
class ConversionConfig:
    """Voice conversion configuration - Applio-compatible parameters."""

    # === Pitch Settings ===
    pitch_shift: int = 0  # semitones (-24 to +24)
    f0_method: str = "rmvpe"  # Pitch extraction: crepe, crepe-tiny, rmvpe, fcpe
    hop_length: int = 128  # Hop length for pitch extraction (64-512)

    # === Index/Feature Settings ===
    index_rate: float = 0.0  # Search Feature Ratio (0-1), higher = more index influence
    index_file: Optional[str] = None  # Specific index file to use (auto-detect if None)

    # === Audio Processing ===
    rms_mix_rate: float = 0.4  # Volume Envelope (0-1), 0=input loudness, 1=training set
    protect: float = 0.3  # Protect Voiceless Consonants (0-0.5), 0.5=disabled
    filter_radius: int = 3  # Median filter radius for pitch (0-7)

    # === Split & Clean ===
    split_audio: bool = False  # Split audio into chunks for better results
    autotune: bool = False  # Apply soft autotune (for singing)
    clean_audio: bool = False  # Clean audio with noise detection
    clean_strength: float = 0.4  # Cleaning strength (0-1)

    # === Formant Shifting ===
    formant_shifting: bool = False  # Enable formant shift (male<->female)
    formant_preset: str = "m2f"  # Preset: m2f, f2m, etc.
    formant_quefrency: float = 1.0  # Quefrency (0-16)
    formant_timbre: float = 1.2  # Timbre (0-16)

    # === Post Processing ===
    post_process: bool = False  # Apply post-processing effects
    normalize_output: bool = True  # Normalize output audio

    # === Model Settings ===
    speaker_id: int = 0  # Speaker ID for multi-speaker models
    embedder_model: str = "contentvec"  # contentvec, spin, spin-v2, hubert variants

    # === Export Settings ===
    export_format: str = "wav"  # Output format: wav, mp3, flac, ogg, m4a


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
    Voice conversion engine using RVC.

    Converts audio from one voice to another using trained RVC models.
    """

    def __init__(
        self,
        device=None,
        models_dir: Optional[Path] = None,
    ):
        self.settings = get_settings()
        self.models_dir = models_dir or self.settings.models_dir
        self.audio_processor = AudioProcessor()
        self.device = device

        # RVC inference engine (lazy loaded)
        self._rvc_inference: Optional[RVCInference] = None

    def _get_rvc_inference(self) -> Optional[RVCInference]:
        """Get or create RVC inference engine."""
        if not RVC_AVAILABLE:
            return None

        if self._rvc_inference is None:
            try:
                self._rvc_inference = RVCInference(device=self.device)
            except Exception as e:
                logger.warning(f"Failed to initialize RVC: {e}")
                return None

        return self._rvc_inference

    def _has_rvc_model(self, model_id: str) -> bool:
        """Check if model has RVC files (.pth)."""
        model_dir = self.models_dir / model_id

        if not model_dir.exists():
            return False

        # Check for .pth files
        pth_files = list(model_dir.glob("*.pth"))
        return len(pth_files) > 0

    def _get_rvc_model_path(self, model_id: str) -> Optional[Path]:
        """Get path to RVC model file."""
        model_dir = self.models_dir / model_id

        pth_files = list(model_dir.glob("*.pth"))
        if pth_files:
            return pth_files[0]

        return None

    def _get_rvc_index_path(self, model_id: str) -> Optional[Path]:
        """Get path to RVC index file."""
        model_dir = self.models_dir / model_id

        index_files = list(model_dir.glob("*.index"))
        if index_files:
            return index_files[0]

        return None

    def convert(
        self,
        source_audio: Union[str, Path, np.ndarray],
        target_model_id: str,
        config: Optional[ConversionConfig] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ConversionResult:
        """
        Convert source audio to target voice using RVC.

        Args:
            source_audio: Path to audio file or numpy array
            target_model_id: ID of the target voice model
            config: Conversion configuration
            progress_callback: Optional callback for progress updates

        Returns:
            ConversionResult with converted audio

        Raises:
            RuntimeError: If RVC is not available or model loading fails
        """
        start_time = time.time()
        cfg = config or ConversionConfig()

        def update_progress(stage: str, progress: float):
            if progress_callback:
                progress_callback(stage, progress)

        update_progress("Loading audio", 0.0)

        # Load source audio
        if isinstance(source_audio, (str, Path)):
            audio, sr = librosa.load(str(source_audio), sr=None)
            source_duration = len(audio) / sr
        else:
            audio = source_audio.astype(np.float32)
            sr = self.settings.sample_rate
            source_duration = len(audio) / sr

        update_progress("Checking model", 0.1)

        # Check if RVC is available
        if not RVC_AVAILABLE:
            raise RuntimeError("RVC module not available. Please install required dependencies.")

        # Check if model exists
        if not self._has_rvc_model(target_model_id):
            raise RuntimeError(f"No RVC model found for '{target_model_id}'. "
                             f"Please add a .pth file to the model directory.")

        logger.info(f"Using RVC for {target_model_id}")
        audio_output, sr_out = self._convert_with_rvc(
            audio, sr, target_model_id, cfg, update_progress
        )

        update_progress("Post-processing", 0.9)

        # Normalize
        if cfg.normalize_output:
            audio_output = AudioUtils.normalize(audio_output)

        processing_time = time.time() - start_time

        update_progress("Complete", 1.0)

        return ConversionResult(
            audio=audio_output,
            sample_rate=sr_out,
            duration=len(audio_output) / sr_out,
            source_duration=source_duration,
            model_id=target_model_id,
            config=cfg,
            processing_time=processing_time,
            metadata={
                "method": "rvc",
                "rvc_available": RVC_AVAILABLE,
            },
        )

    def _get_rvc_index_files(self, model_id: str) -> list[Path]:
        """Get all index files for a model."""
        model_dir = self.models_dir / model_id
        return list(model_dir.glob("*.index"))

    def _convert_with_rvc(
        self,
        audio: np.ndarray,
        sr: int,
        model_id: str,
        cfg: ConversionConfig,
        update_progress: Callable[[str, float], None],
    ) -> tuple[np.ndarray, int]:
        """Convert using RVC with Applio-compatible parameters."""
        rvc = self._get_rvc_inference()

        if rvc is None:
            raise RuntimeError("Failed to initialize RVC inference engine")

        update_progress("Loading RVC model", 0.2)

        # Load model
        model_path = self._get_rvc_model_path(model_id)

        # Get index path - use specific file if provided, otherwise auto-detect
        if cfg.index_file:
            index_path = self.models_dir / model_id / cfg.index_file
            if not index_path.exists():
                index_path = self._get_rvc_index_path(model_id)
        else:
            index_path = self._get_rvc_index_path(model_id)

        if not rvc.load_rvc_model(model_path):
            raise RuntimeError(f"Failed to load RVC model: {model_path}")

        update_progress("Converting with RVC", 0.4)

        audio_out, sr_out = rvc.convert(
            audio,
            sr,
            f0_up_key=int(cfg.pitch_shift),
            index_path=index_path,
            index_rate=cfg.index_rate,
            filter_radius=cfg.filter_radius,
            rms_mix_rate=cfg.rms_mix_rate,
            protect=cfg.protect,
            f0_method=cfg.f0_method,
            hop_length=cfg.hop_length,
            split_audio=cfg.split_audio,
            clean_audio=cfg.clean_audio,
            clean_strength=cfg.clean_strength,
            autotune=cfg.autotune,
            formant_shifting=cfg.formant_shifting,
            formant_quefrency=cfg.formant_quefrency,
            formant_timbre=cfg.formant_timbre,
            embedder_model=cfg.embedder_model,
            speaker_id=cfg.speaker_id,
            progress_callback=lambda msg, prog: update_progress(msg, 0.4 + prog * 0.4),
        )

        return audio_out, sr_out

    def is_rvc_available(self) -> bool:
        """Check if RVC is available."""
        return RVC_AVAILABLE

    def has_rvc_model(self, model_id: str) -> bool:
        """Check if model has RVC files."""
        return self._has_rvc_model(model_id)

    def clear_cache(self):
        """Clear caches."""
        pass

    def set_device(self, device):
        """Set device."""
        self.device = device
        self._rvc_inference = None  # Force re-init
