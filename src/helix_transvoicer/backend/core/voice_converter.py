"""
Helix Transvoicer - Voice conversion pipeline using RVC.

Uses Retrieval-based Voice Conversion (RVC) for high-quality voice cloning
that captures speaking style, pitch, and characteristics of the target voice.
"""

import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import librosa
import numpy as np
import soundfile as sf

from helix_transvoicer.backend.core.audio_processor import AudioProcessor
from helix_transvoicer.backend.utils.audio import AudioUtils
from helix_transvoicer.backend.utils.config import get_settings

logger = logging.getLogger("helix.voice_converter")

# Try to import RVC
try:
    from rvc_python.infer import RVCInference
    RVC_AVAILABLE = True
    logger.info("RVC (Retrieval-based Voice Conversion) is available")
except ImportError:
    RVC_AVAILABLE = False
    logger.warning("RVC not available - install with: pip install rvc-python")


@dataclass
class ConversionConfig:
    """Voice conversion configuration."""

    pitch_shift: float = 0.0  # semitones (-12 to +12)
    index_rate: float = 0.75  # Feature retrieval blend (0-1), higher = more like target
    filter_radius: int = 3  # Pitch median filtering
    rms_mix_rate: float = 0.25  # Volume envelope blend (0-1)
    protect: float = 0.33  # Protect voiceless consonants (0-0.5)
    f0_method: str = "rmvpe"  # Pitch extraction: rmvpe, harvest, crepe, pm
    resample_sr: int = 0  # Output sample rate (0 = same as input)
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
    Voice conversion engine using RVC (Retrieval-based Voice Conversion).

    RVC provides high-quality voice conversion that:
    - Captures the target speaker's voice characteristics
    - Preserves the speaking style and mannerisms
    - Maintains natural prosody and intonation
    - Works with just 10+ minutes of training audio
    """

    def __init__(
        self,
        device: Optional[str] = None,
        models_dir: Optional[Path] = None,
    ):
        self.settings = get_settings()
        self.models_dir = models_dir or self.settings.models_dir
        self.audio_processor = AudioProcessor()

        # Determine device
        if device is None:
            import torch
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

        # RVC inference engine (lazy loaded)
        self._rvc: Optional[RVCInference] = None
        self._current_model: Optional[str] = None

        # Cache for model info
        self._model_cache: Dict[str, Dict] = {}

    def _get_rvc(self) -> Optional[RVCInference]:
        """Get or initialize RVC inference engine."""
        if not RVC_AVAILABLE:
            return None

        if self._rvc is None:
            try:
                logger.info(f"Initializing RVC on device: {self.device}")
                self._rvc = RVCInference(device=self.device)
                logger.info("RVC initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RVC: {e}")
                return None

        return self._rvc

    def _find_model_files(self, model_id: str) -> tuple[Optional[Path], Optional[Path]]:
        """Find .pth model file and optional .index file for a model."""
        model_dir = self.models_dir / model_id

        if not model_dir.exists():
            logger.error(f"Model directory not found: {model_dir}")
            return None, None

        # Find .pth file
        pth_files = list(model_dir.glob("*.pth"))
        if not pth_files:
            # Check for RVC subdirectory
            rvc_dir = model_dir / "rvc"
            if rvc_dir.exists():
                pth_files = list(rvc_dir.glob("*.pth"))

        model_path = pth_files[0] if pth_files else None

        # Find .index file (optional but improves quality)
        index_files = list(model_dir.glob("*.index"))
        if not index_files:
            rvc_dir = model_dir / "rvc"
            if rvc_dir.exists():
                index_files = list(rvc_dir.glob("*.index"))

        index_path = index_files[0] if index_files else None

        return model_path, index_path

    def _load_model(self, model_id: str) -> bool:
        """Load RVC model for a voice."""
        if self._current_model == model_id:
            return True

        rvc = self._get_rvc()
        if rvc is None:
            return False

        model_path, index_path = self._find_model_files(model_id)

        if model_path is None:
            logger.error(f"No .pth model file found for {model_id}")
            return False

        try:
            logger.info(f"Loading RVC model: {model_path}")
            rvc.load_model(str(model_path), index_path=str(index_path) if index_path else None)
            self._current_model = model_id
            logger.info(f"Model loaded successfully (index: {index_path is not None})")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False

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
            source_audio: Path to audio file or audio array
            target_model_id: ID of target voice model
            config: Conversion configuration
            progress_callback: Optional callback for progress updates

        Returns:
            ConversionResult with converted audio
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

        update_progress("Loading voice model", 0.2)

        # Try RVC conversion first
        if RVC_AVAILABLE and self._load_model(target_model_id):
            audio_output, sr_out = self._convert_with_rvc(
                audio, sr, target_model_id, cfg, update_progress
            )
        else:
            # Fallback to pitch shifting
            logger.warning("RVC not available, using pitch shift fallback")
            audio_output, sr_out = self._convert_with_pitch_shift(
                audio, sr, target_model_id, cfg, update_progress
            )

        update_progress("Post-processing", 0.9)

        # Normalize output
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
                "method": "rvc" if RVC_AVAILABLE else "pitch_shift",
                "device": self.device,
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
        """Convert audio using RVC."""
        rvc = self._get_rvc()

        update_progress("Preparing audio", 0.3)

        # RVC works with files, so save to temp
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            sf.write(tmp_in.name, audio, sr)
            input_path = tmp_in.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            output_path = tmp_out.name

        try:
            update_progress("Converting voice (RVC)", 0.5)

            # Configure RVC parameters
            rvc.set_params(
                f0method=cfg.f0_method,
                f0up_key=int(cfg.pitch_shift),
                index_rate=cfg.index_rate,
                filter_radius=cfg.filter_radius,
                rms_mix_rate=cfg.rms_mix_rate,
                protect=cfg.protect,
                resample_sr=cfg.resample_sr if cfg.resample_sr > 0 else sr,
            )

            # Run conversion
            rvc.infer_file(input_path, output_path)

            update_progress("Loading result", 0.8)

            # Load converted audio
            audio_out, sr_out = librosa.load(output_path, sr=None)

            return audio_out, sr_out

        finally:
            # Cleanup temp files
            Path(input_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    def _convert_with_pitch_shift(
        self,
        audio: np.ndarray,
        sr: int,
        model_id: str,
        cfg: ConversionConfig,
        update_progress: Callable[[str, float], None],
    ) -> tuple[np.ndarray, int]:
        """Fallback conversion using pitch shifting."""
        update_progress("Analyzing voice", 0.4)

        # Load target voice characteristics if available
        target_pitch = self._load_voice_pitch(model_id)
        source_pitch = self._analyze_pitch(audio, sr)

        # Calculate pitch shift
        if source_pitch > 0 and target_pitch > 0:
            pitch_ratio = target_pitch / source_pitch
            auto_shift = 12 * np.log2(pitch_ratio)
        else:
            auto_shift = 0.0

        total_shift = auto_shift + cfg.pitch_shift

        update_progress("Shifting pitch", 0.6)

        if abs(total_shift) > 0.1:
            audio_out = librosa.effects.pitch_shift(audio, sr=sr, n_steps=total_shift)
        else:
            audio_out = audio.copy()

        return audio_out, sr

    def _analyze_pitch(self, audio: np.ndarray, sr: int) -> float:
        """Analyze mean pitch of audio."""
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C6'),
                sr=sr,
            )
            if voiced_flag is not None:
                voiced_f0 = f0[voiced_flag]
                if len(voiced_f0) > 0:
                    return float(np.nanmean(voiced_f0))
        except Exception as e:
            logger.warning(f"Pitch analysis failed: {e}")
        return 150.0

    def _load_voice_pitch(self, model_id: str) -> float:
        """Load target voice pitch from characteristics file."""
        char_path = self.models_dir / model_id / "voice_characteristics.json"
        if char_path.exists():
            try:
                with open(char_path) as f:
                    data = json.load(f)
                    return data.get("pitch_mean", 150.0)
            except Exception:
                pass
        return 150.0

    def is_rvc_available(self) -> bool:
        """Check if RVC is available."""
        return RVC_AVAILABLE

    def has_rvc_model(self, model_id: str) -> bool:
        """Check if a model has RVC files (.pth)."""
        model_path, _ = self._find_model_files(model_id)
        return model_path is not None

    def get_available_f0_methods(self) -> list[str]:
        """Get available pitch extraction methods."""
        return ["rmvpe", "harvest", "crepe", "pm"]

    def clear_cache(self):
        """Clear model cache."""
        self._model_cache.clear()
        self._current_model = None

    def set_device(self, device: str):
        """Change compute device."""
        if device != self.device:
            self.device = device
            self._rvc = None  # Force re-initialization
            self._current_model = None
