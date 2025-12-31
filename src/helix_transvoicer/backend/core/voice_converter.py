"""
Helix Transvoicer - Voice conversion pipeline.

Uses Praat/Parselmouth for professional-grade voice manipulation including
pitch shifting, formant modification, and voice characteristic transfer.
"""

import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import librosa
import numpy as np
import soundfile as sf

from helix_transvoicer.backend.core.audio_processor import AudioProcessor
from helix_transvoicer.backend.utils.audio import AudioUtils
from helix_transvoicer.backend.utils.config import get_settings

logger = logging.getLogger("helix.voice_converter")

# Try to import Parselmouth for professional voice manipulation
try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
    logger.info("Parselmouth (Praat) available for voice conversion")
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    logger.warning("Parselmouth not available - install with: pip install praat-parselmouth")


@dataclass
class ConversionConfig:
    """Voice conversion configuration."""

    pitch_shift: float = 0.0  # semitones (-12 to +12)
    formant_shift: float = 1.0  # ratio (0.8 = deeper, 1.2 = higher)
    pitch_range: float = 1.0  # pitch variation (0.5 = monotone, 1.5 = expressive)
    duration_factor: float = 1.0  # speed (0.8 = faster, 1.2 = slower)
    intensity_factor: float = 1.0  # volume adjustment
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


@dataclass
class VoiceProfile:
    """Voice characteristics profile."""
    pitch_median: float = 150.0  # Hz
    pitch_min: float = 75.0
    pitch_max: float = 300.0
    formant_shift: float = 1.0  # 1.0 = neutral
    intensity: float = 70.0  # dB


class VoiceConverter:
    """
    Voice conversion engine using Praat/Parselmouth.

    Uses professional-grade PSOLA algorithm for:
    - Pitch manipulation (shift and range)
    - Formant shifting (voice character)
    - Duration modification
    - Voice profile matching
    """

    def __init__(
        self,
        device=None,  # Kept for API compatibility
        models_dir: Optional[Path] = None,
    ):
        self.settings = get_settings()
        self.models_dir = models_dir or self.settings.models_dir
        self.audio_processor = AudioProcessor()

        # Cache for voice profiles
        self._profile_cache: Dict[str, VoiceProfile] = {}

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

        update_progress("Loading voice profile", 0.2)

        # Load target voice profile
        target_profile = self._load_voice_profile(target_model_id)

        # Analyze source voice
        source_profile = self._analyze_voice(audio, sr)

        update_progress("Converting voice", 0.3)

        # Convert using Parselmouth if available
        if PARSELMOUTH_AVAILABLE:
            audio_output = self._convert_with_parselmouth(
                audio, sr, source_profile, target_profile, cfg, update_progress
            )
        else:
            # Fallback to librosa
            audio_output = self._convert_with_librosa(
                audio, sr, source_profile, target_profile, cfg, update_progress
            )

        update_progress("Post-processing", 0.9)

        # Normalize output
        if cfg.normalize_output:
            audio_output = AudioUtils.normalize(audio_output)

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
                "method": "parselmouth" if PARSELMOUTH_AVAILABLE else "librosa",
                "source_pitch": source_profile.pitch_median,
                "target_pitch": target_profile.pitch_median,
            },
        )

    def _convert_with_parselmouth(
        self,
        audio: np.ndarray,
        sr: int,
        source_profile: VoiceProfile,
        target_profile: VoiceProfile,
        cfg: ConversionConfig,
        update_progress: Callable[[str, float], None],
    ) -> np.ndarray:
        """Convert audio using Praat/Parselmouth PSOLA algorithm."""

        # Create Parselmouth Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sr)

        update_progress("Analyzing pitch", 0.4)

        # Calculate pitch shift ratio
        if source_profile.pitch_median > 0 and target_profile.pitch_median > 0:
            pitch_ratio = target_profile.pitch_median / source_profile.pitch_median
        else:
            pitch_ratio = 1.0

        # Apply user pitch shift (in semitones)
        if cfg.pitch_shift != 0:
            pitch_ratio *= 2 ** (cfg.pitch_shift / 12.0)

        # Calculate formant shift
        formant_ratio = target_profile.formant_shift * cfg.formant_shift

        update_progress("Manipulating voice", 0.5)

        # Use Praat's "Change gender" function for voice conversion
        # This handles pitch AND formants together using PSOLA
        try:
            converted = call(
                sound,
                "Change gender",
                75,  # pitch floor (Hz)
                600,  # pitch ceiling (Hz)
                formant_ratio,  # formant shift ratio
                pitch_ratio,  # new pitch median (as ratio)
                cfg.pitch_range,  # pitch range factor
                cfg.duration_factor,  # duration factor
            )

            update_progress("Extracting audio", 0.8)

            # Extract audio array
            audio_output = converted.values.flatten()

            return audio_output

        except Exception as e:
            logger.warning(f"Parselmouth conversion failed: {e}, using fallback")
            return self._convert_with_librosa(
                audio, sr, source_profile, target_profile, cfg, update_progress
            )

    def _convert_with_librosa(
        self,
        audio: np.ndarray,
        sr: int,
        source_profile: VoiceProfile,
        target_profile: VoiceProfile,
        cfg: ConversionConfig,
        update_progress: Callable[[str, float], None],
    ) -> np.ndarray:
        """Fallback conversion using librosa."""

        update_progress("Calculating pitch shift", 0.4)

        # Calculate pitch shift in semitones
        if source_profile.pitch_median > 0 and target_profile.pitch_median > 0:
            pitch_ratio = target_profile.pitch_median / source_profile.pitch_median
            auto_shift = 12 * np.log2(pitch_ratio)
        else:
            auto_shift = 0.0

        total_shift = auto_shift + cfg.pitch_shift

        update_progress("Shifting pitch", 0.5)

        # Apply pitch shift
        if abs(total_shift) > 0.1:
            audio_output = librosa.effects.pitch_shift(
                audio, sr=sr, n_steps=total_shift
            )
        else:
            audio_output = audio.copy()

        # Apply formant shift via time stretch + pitch compensation
        if abs(cfg.formant_shift - 1.0) > 0.05:
            update_progress("Shifting formants", 0.7)
            audio_output = self._apply_formant_shift_librosa(
                audio_output, sr, cfg.formant_shift
            )

        return audio_output

    def _apply_formant_shift_librosa(
        self,
        audio: np.ndarray,
        sr: int,
        shift: float,
    ) -> np.ndarray:
        """Apply formant shift using librosa time stretch + pitch."""
        try:
            # Time stretch
            stretched = librosa.effects.time_stretch(audio, rate=shift)

            # Pitch shift to compensate
            compensate = -12 * np.log2(shift)
            result = librosa.effects.pitch_shift(stretched, sr=sr, n_steps=compensate)

            return result
        except Exception as e:
            logger.warning(f"Formant shift failed: {e}")
            return audio

    def _analyze_voice(self, audio: np.ndarray, sr: int) -> VoiceProfile:
        """Analyze voice characteristics of audio."""

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
                logger.warning(f"Parselmouth pitch analysis failed: {e}")

        # Fallback to librosa
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
                    return VoiceProfile(
                        pitch_median=float(np.median(voiced_f0)),
                        pitch_min=float(np.nanmin(voiced_f0)),
                        pitch_max=float(np.nanmax(voiced_f0)),
                    )
        except Exception as e:
            logger.warning(f"Librosa pitch analysis failed: {e}")

        return VoiceProfile()

    def _load_voice_profile(self, model_id: str) -> VoiceProfile:
        """Load voice profile for a model."""

        if model_id in self._profile_cache:
            return self._profile_cache[model_id]

        model_path = self.models_dir / model_id

        # Try to load from voice_characteristics.json
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
            except Exception as e:
                logger.warning(f"Failed to load voice profile: {e}")

        # Try to analyze from samples
        samples_dir = model_path / "samples"
        if samples_dir.exists():
            profile = self._analyze_samples(samples_dir)
            if profile:
                self._save_voice_profile(model_path, profile)
                self._profile_cache[model_id] = profile
                return profile

        # Return default
        logger.warning(f"No voice profile for {model_id}, using defaults")
        return VoiceProfile()

    def _analyze_samples(self, samples_dir: Path) -> Optional[VoiceProfile]:
        """Analyze training samples to build voice profile."""

        sample_files = list(samples_dir.glob("*.wav")) + list(samples_dir.glob("*.mp3"))
        all_pitches = []

        for sample_file in sample_files[:10]:
            try:
                audio, sr = librosa.load(str(sample_file), sr=22050)
                profile = self._analyze_voice(audio, sr)
                if profile.pitch_median > 0:
                    all_pitches.append(profile.pitch_median)
            except Exception as e:
                logger.debug(f"Failed to analyze {sample_file}: {e}")

        if all_pitches:
            return VoiceProfile(
                pitch_median=float(np.median(all_pitches)),
                pitch_min=float(np.min(all_pitches)) * 0.8,
                pitch_max=float(np.max(all_pitches)) * 1.2,
            )

        return None

    def _save_voice_profile(self, model_path: Path, profile: VoiceProfile):
        """Save voice profile to file."""
        char_path = model_path / "voice_characteristics.json"
        try:
            data = {
                "pitch_median": profile.pitch_median,
                "pitch_mean": profile.pitch_median,  # Alias for compatibility
                "pitch_min": profile.pitch_min,
                "pitch_max": profile.pitch_max,
                "formant_shift": profile.formant_shift,
            }
            with open(char_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved voice profile to {char_path}")
        except Exception as e:
            logger.warning(f"Failed to save voice profile: {e}")

    def clear_cache(self):
        """Clear voice profile cache."""
        self._profile_cache.clear()

    def set_device(self, device):
        """Set device (no-op, kept for API compatibility)."""
        pass
