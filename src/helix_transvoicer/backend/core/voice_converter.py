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
    """Voice conversion configuration."""

    # General settings
    pitch_shift: float = 0.0  # semitones (-12 to +12)
    normalize_output: bool = True
    formant_shift: float = 1.0  # Not used in RVC but kept for API compatibility

    # RVC-specific
    index_rate: float = 0.75  # Feature retrieval blend (0-1)
    filter_radius: int = 3  # Pitch median filter
    rms_mix_rate: float = 0.25  # Volume envelope mixing
    protect: float = 0.33  # Protect voiceless consonants


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

    def _convert_with_rvc(
        self,
        audio: np.ndarray,
        sr: int,
        model_id: str,
        cfg: ConversionConfig,
        update_progress: Callable[[str, float], None],
    ) -> tuple[np.ndarray, int]:
        """Convert using RVC."""
        rvc = self._get_rvc_inference()

        if rvc is None:
            raise RuntimeError("Failed to initialize RVC inference engine")

        update_progress("Loading RVC model", 0.2)

        # Load model
        model_path = self._get_rvc_model_path(model_id)
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
