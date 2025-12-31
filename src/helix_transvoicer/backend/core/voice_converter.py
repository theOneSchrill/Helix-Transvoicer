"""
Helix Transvoicer - Voice conversion pipeline.

Supports multiple conversion methods:
1. RVC (Retrieval-based Voice Conversion) - Best quality
2. Praat/Parselmouth - Fallback for pitch/formant manipulation
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

# Try to import Parselmouth
try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    logger.warning("Parselmouth not available - install with: pip install praat-parselmouth")

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

    # Parselmouth-specific
    formant_shift: float = 1.0  # ratio (0.8 = deeper, 1.2 = higher)
    pitch_range: float = 1.0  # pitch variation
    duration_factor: float = 1.0  # speed

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


@dataclass
class VoiceProfile:
    """Voice characteristics profile."""
    pitch_median: float = 150.0
    pitch_min: float = 75.0
    pitch_max: float = 300.0
    formant_shift: float = 1.0


class VoiceConverter:
    """
    Voice conversion engine.

    Uses RVC (Retrieval-based Voice Conversion) when available,
    falls back to Praat/Parselmouth for simpler transformations.
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

        # Cache
        self._profile_cache: Dict[str, VoiceProfile] = {}

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
        Convert source audio to target voice.

        Uses RVC if model has .pth file, otherwise uses Parselmouth.
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

        # Determine conversion method
        use_rvc = self._has_rvc_model(target_model_id) and RVC_AVAILABLE

        if use_rvc:
            logger.info(f"Using RVC for {target_model_id}")
            audio_output, sr_out, method = self._convert_with_rvc(
                audio, sr, target_model_id, cfg, update_progress
            )
        else:
            logger.info(f"Using Parselmouth for {target_model_id}")
            audio_output, sr_out, method = self._convert_with_parselmouth(
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
                "method": method,
                "rvc_available": RVC_AVAILABLE,
                "parselmouth_available": PARSELMOUTH_AVAILABLE,
            },
        )

    def _convert_with_rvc(
        self,
        audio: np.ndarray,
        sr: int,
        model_id: str,
        cfg: ConversionConfig,
        update_progress: Callable[[str, float], None],
    ) -> tuple[np.ndarray, int, str]:
        """Convert using RVC."""
        rvc = self._get_rvc_inference()

        if rvc is None:
            logger.warning("RVC not available, falling back to Parselmouth")
            return self._convert_with_parselmouth(audio, sr, model_id, cfg, update_progress)

        update_progress("Loading RVC model", 0.2)

        # Load model
        model_path = self._get_rvc_model_path(model_id)
        index_path = self._get_rvc_index_path(model_id)

        if not rvc.load_rvc_model(model_path):
            logger.warning("Failed to load RVC model, falling back to Parselmouth")
            return self._convert_with_parselmouth(audio, sr, model_id, cfg, update_progress)

        update_progress("Converting with RVC", 0.4)

        try:
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

            return audio_out, sr_out, "rvc"

        except Exception as e:
            logger.error(f"RVC conversion failed: {e}")
            logger.info("Falling back to Parselmouth")
            return self._convert_with_parselmouth(audio, sr, model_id, cfg, update_progress)

    def _convert_with_parselmouth(
        self,
        audio: np.ndarray,
        sr: int,
        model_id: str,
        cfg: ConversionConfig,
        update_progress: Callable[[str, float], None],
    ) -> tuple[np.ndarray, int, str]:
        """Convert using Parselmouth."""
        update_progress("Loading voice profile", 0.2)

        # Load target voice profile
        target_profile = self._load_voice_profile(model_id)
        source_profile = self._analyze_voice(audio, sr)

        update_progress("Converting voice", 0.4)

        if PARSELMOUTH_AVAILABLE:
            audio_out = self._parselmouth_convert(
                audio, sr, source_profile, target_profile, cfg
            )
        else:
            audio_out = self._librosa_convert(
                audio, sr, source_profile, target_profile, cfg
            )

        return audio_out, sr, "parselmouth" if PARSELMOUTH_AVAILABLE else "librosa"

    def _parselmouth_convert(
        self,
        audio: np.ndarray,
        sr: int,
        source: VoiceProfile,
        target: VoiceProfile,
        cfg: ConversionConfig,
    ) -> np.ndarray:
        """Convert using Parselmouth PSOLA."""
        sound = parselmouth.Sound(audio, sampling_frequency=sr)

        # Calculate pitch ratio
        if source.pitch_median > 0 and target.pitch_median > 0:
            pitch_ratio = target.pitch_median / source.pitch_median
        else:
            pitch_ratio = 1.0

        # Apply user pitch shift
        if cfg.pitch_shift != 0:
            pitch_ratio *= 2 ** (cfg.pitch_shift / 12.0)

        # Formant ratio
        formant_ratio = target.formant_shift * cfg.formant_shift

        try:
            converted = call(
                sound,
                "Change gender",
                75, 600,  # pitch floor, ceiling
                formant_ratio,
                pitch_ratio,
                cfg.pitch_range,
                cfg.duration_factor,
            )

            return converted.values.flatten()

        except Exception as e:
            logger.warning(f"Parselmouth failed: {e}")
            return audio

    def _librosa_convert(
        self,
        audio: np.ndarray,
        sr: int,
        source: VoiceProfile,
        target: VoiceProfile,
        cfg: ConversionConfig,
    ) -> np.ndarray:
        """Fallback using librosa pitch shift."""
        if source.pitch_median > 0 and target.pitch_median > 0:
            pitch_ratio = target.pitch_median / source.pitch_median
            auto_shift = 12 * np.log2(pitch_ratio)
        else:
            auto_shift = 0.0

        total_shift = auto_shift + cfg.pitch_shift

        if abs(total_shift) > 0.1:
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=total_shift)

        return audio.copy()

    def _analyze_voice(self, audio: np.ndarray, sr: int) -> VoiceProfile:
        """Analyze voice characteristics."""
        if PARSELMOUTH_AVAILABLE:
            try:
                sound = parselmouth.Sound(audio, sampling_frequency=sr)
                pitch = call(sound, "To Pitch", 0.0, 75, 600)
                pitch_values = pitch.selected_array['frequency']
                pitch_values = pitch_values[pitch_values > 0]

                if len(pitch_values) > 0:
                    return VoiceProfile(
                        pitch_median=float(np.median(pitch_values)),
                        pitch_min=float(np.min(pitch_values)),
                        pitch_max=float(np.max(pitch_values)),
                    )
            except Exception as e:
                logger.warning(f"Parselmouth analysis failed: {e}")

        # Fallback to librosa
        try:
            f0, voiced, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C6'),
                sr=sr,
            )
            if voiced is not None:
                voiced_f0 = f0[voiced]
                if len(voiced_f0) > 0:
                    return VoiceProfile(
                        pitch_median=float(np.median(voiced_f0)),
                        pitch_min=float(np.nanmin(voiced_f0)),
                        pitch_max=float(np.nanmax(voiced_f0)),
                    )
        except Exception:
            pass

        return VoiceProfile()

    def _load_voice_profile(self, model_id: str) -> VoiceProfile:
        """Load voice profile from model."""
        if model_id in self._profile_cache:
            return self._profile_cache[model_id]

        model_path = self.models_dir / model_id

        # Try voice_characteristics.json
        char_path = model_path / "voice_characteristics.json"
        if char_path.exists():
            try:
                with open(char_path) as f:
                    data = json.load(f)
                    profile = VoiceProfile(
                        pitch_median=data.get("pitch_median", data.get("pitch_mean", 150.0)),
                        pitch_min=data.get("pitch_min", 75.0),
                        pitch_max=data.get("pitch_max", 300.0),
                        formant_shift=data.get("formant_shift", 1.0),
                    )
                    self._profile_cache[model_id] = profile
                    return profile
            except Exception:
                pass

        # Analyze from samples
        samples_dir = model_path / "samples"
        if samples_dir.exists():
            profile = self._analyze_samples(samples_dir)
            if profile:
                self._save_voice_profile(model_path, profile)
                self._profile_cache[model_id] = profile
                return profile

        return VoiceProfile()

    def _analyze_samples(self, samples_dir: Path) -> Optional[VoiceProfile]:
        """Analyze training samples."""
        sample_files = list(samples_dir.glob("*.wav")) + list(samples_dir.glob("*.mp3"))
        all_pitches = []

        for sample in sample_files[:10]:
            try:
                audio, sr = librosa.load(str(sample), sr=22050)
                profile = self._analyze_voice(audio, sr)
                if profile.pitch_median > 0:
                    all_pitches.append(profile.pitch_median)
            except Exception:
                pass

        if all_pitches:
            return VoiceProfile(
                pitch_median=float(np.median(all_pitches)),
                pitch_min=float(np.min(all_pitches)) * 0.8,
                pitch_max=float(np.max(all_pitches)) * 1.2,
            )

        return None

    def _save_voice_profile(self, model_path: Path, profile: VoiceProfile):
        """Save voice profile."""
        char_path = model_path / "voice_characteristics.json"
        try:
            with open(char_path, "w") as f:
                json.dump({
                    "pitch_median": profile.pitch_median,
                    "pitch_mean": profile.pitch_median,
                    "pitch_min": profile.pitch_min,
                    "pitch_max": profile.pitch_max,
                    "formant_shift": profile.formant_shift,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save voice profile: {e}")

    def is_rvc_available(self) -> bool:
        """Check if RVC is available."""
        return RVC_AVAILABLE

    def has_rvc_model(self, model_id: str) -> bool:
        """Check if model has RVC files."""
        return self._has_rvc_model(model_id)

    def clear_cache(self):
        """Clear caches."""
        self._profile_cache.clear()

    def set_device(self, device):
        """Set device."""
        self.device = device
        self._rvc_inference = None  # Force re-init
