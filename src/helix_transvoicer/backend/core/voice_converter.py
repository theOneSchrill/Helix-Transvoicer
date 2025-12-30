"""
Helix Transvoicer - Voice conversion pipeline.

Converts source audio to target voice using signal processing.
Uses pitch shifting and formant modification for voice transformation.
"""

import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import time

import librosa
import numpy as np
import scipy.signal as signal

from helix_transvoicer.backend.core.audio_processor import AudioProcessor
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


@dataclass
class VoiceCharacteristics:
    """Voice characteristics extracted from training samples."""
    pitch_mean: float = 150.0  # Hz
    pitch_std: float = 50.0
    pitch_min: float = 80.0
    pitch_max: float = 300.0
    formant_shift: float = 0.0  # relative shift


class VoiceConverter:
    """
    Voice conversion engine using signal processing.

    Pipeline:
    1. Load and preprocess source audio
    2. Analyze source pitch characteristics
    3. Load target voice characteristics
    4. Apply pitch shift to match target
    5. Apply formant modification
    6. Post-process and return
    """

    def __init__(
        self,
        device=None,  # Kept for API compatibility
        models_dir: Optional[Path] = None,
    ):
        self.settings = get_settings()
        self.models_dir = models_dir or self.settings.models_dir
        self.audio_processor = AudioProcessor()

        # Cache for voice characteristics
        self._voice_cache: Dict[str, VoiceCharacteristics] = {}

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

        # Load and preprocess source audio
        if isinstance(source_audio, (str, Path)):
            processed = self.audio_processor.process(source_audio)
            audio = processed.audio
            sr = processed.sample_rate
            source_duration = processed.original_duration
        else:
            audio = source_audio.astype(np.float32)
            sr = self.settings.sample_rate
            source_duration = len(audio) / sr

        update_progress("Analyzing voice", 0.2)

        # Analyze source pitch
        source_pitch = self._analyze_pitch(audio, sr)
        logger.info(f"Source pitch: mean={source_pitch.pitch_mean:.1f}Hz")

        update_progress("Loading voice model", 0.3)

        # Load target voice characteristics
        target_voice = self._load_voice_characteristics(target_model_id)
        logger.info(f"Target pitch: mean={target_voice.pitch_mean:.1f}Hz")

        update_progress("Converting voice", 0.4)

        # Calculate pitch shift needed
        if source_pitch.pitch_mean > 0 and target_voice.pitch_mean > 0:
            # Calculate semitones shift to match target pitch
            pitch_ratio = target_voice.pitch_mean / source_pitch.pitch_mean
            auto_pitch_shift = 12 * np.log2(pitch_ratio)
        else:
            auto_pitch_shift = 0.0

        # Combine with user-specified pitch shift
        total_pitch_shift = auto_pitch_shift + cfg.pitch_shift
        logger.info(f"Pitch shift: auto={auto_pitch_shift:.1f}, user={cfg.pitch_shift}, total={total_pitch_shift:.1f} semitones")

        # Apply pitch shift using librosa
        if abs(total_pitch_shift) > 0.1:
            audio_output = librosa.effects.pitch_shift(
                audio,
                sr=sr,
                n_steps=total_pitch_shift,
            )
        else:
            audio_output = audio.copy()

        update_progress("Applying formant shift", 0.6)

        # Apply formant shift if specified
        total_formant_shift = target_voice.formant_shift + cfg.formant_shift
        if abs(total_formant_shift) > 0.05:
            audio_output = self._apply_formant_shift(audio_output, sr, total_formant_shift)

        update_progress("Post-processing", 0.8)

        # Apply smoothing if needed
        if cfg.smoothing > 0:
            audio_output = self._apply_smoothing(audio_output, cfg.smoothing)

        # Normalize output
        if cfg.normalize_output:
            audio_output = AudioUtils.normalize(audio_output)

        # Apply fade in/out
        if cfg.crossfade_ms > 0:
            audio_output = self._apply_crossfade(audio_output, sr, cfg.crossfade_ms)

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
                "source_pitch_mean": source_pitch.pitch_mean,
                "target_pitch_mean": target_voice.pitch_mean,
                "pitch_shift_semitones": total_pitch_shift,
                "formant_shift": total_formant_shift,
            },
        )

    def _analyze_pitch(self, audio: np.ndarray, sr: int) -> VoiceCharacteristics:
        """Analyze pitch characteristics of audio."""
        try:
            # Extract pitch using librosa
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C6'),
                sr=sr,
            )

            # Filter to voiced frames only
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]

            if len(voiced_f0) > 0:
                return VoiceCharacteristics(
                    pitch_mean=float(np.nanmean(voiced_f0)),
                    pitch_std=float(np.nanstd(voiced_f0)),
                    pitch_min=float(np.nanmin(voiced_f0)),
                    pitch_max=float(np.nanmax(voiced_f0)),
                )
        except Exception as e:
            logger.warning(f"Pitch analysis failed: {e}")

        # Return default if analysis fails
        return VoiceCharacteristics()

    def _load_voice_characteristics(self, model_id: str) -> VoiceCharacteristics:
        """Load voice characteristics for a model."""
        if model_id in self._voice_cache:
            return self._voice_cache[model_id]

        model_path = self.models_dir / model_id

        # Try to load voice_characteristics.json
        char_path = model_path / "voice_characteristics.json"
        if char_path.exists():
            try:
                with open(char_path, "r") as f:
                    data = json.load(f)
                    chars = VoiceCharacteristics(
                        pitch_mean=data.get("pitch_mean", 150.0),
                        pitch_std=data.get("pitch_std", 50.0),
                        pitch_min=data.get("pitch_min", 80.0),
                        pitch_max=data.get("pitch_max", 300.0),
                        formant_shift=data.get("formant_shift", 0.0),
                    )
                    self._voice_cache[model_id] = chars
                    return chars
            except Exception as e:
                logger.warning(f"Failed to load voice characteristics: {e}")

        # Try to analyze from training samples
        samples_dir = model_path / "samples"
        if samples_dir.exists():
            chars = self._analyze_training_samples(samples_dir)
            if chars:
                # Save for future use
                self._save_voice_characteristics(model_path, chars)
                self._voice_cache[model_id] = chars
                return chars

        # Return defaults based on common voice types
        logger.warning(f"No voice characteristics found for {model_id}, using defaults")
        default_chars = VoiceCharacteristics()
        self._voice_cache[model_id] = default_chars
        return default_chars

    def _analyze_training_samples(self, samples_dir: Path) -> Optional[VoiceCharacteristics]:
        """Analyze training samples to extract voice characteristics."""
        all_pitches = []

        sample_files = list(samples_dir.glob("*.wav")) + list(samples_dir.glob("*.mp3"))

        for sample_file in sample_files[:10]:  # Limit to first 10 samples
            try:
                audio, sr = librosa.load(str(sample_file), sr=22050)
                f0, voiced_flag, _ = librosa.pyin(
                    audio,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C6'),
                    sr=sr,
                )
                if voiced_flag is not None:
                    voiced_f0 = f0[voiced_flag]
                    all_pitches.extend(voiced_f0[~np.isnan(voiced_f0)])
            except Exception as e:
                logger.debug(f"Failed to analyze {sample_file}: {e}")

        if all_pitches:
            all_pitches = np.array(all_pitches)
            return VoiceCharacteristics(
                pitch_mean=float(np.mean(all_pitches)),
                pitch_std=float(np.std(all_pitches)),
                pitch_min=float(np.min(all_pitches)),
                pitch_max=float(np.max(all_pitches)),
            )

        return None

    def _save_voice_characteristics(self, model_path: Path, chars: VoiceCharacteristics):
        """Save voice characteristics to file."""
        char_path = model_path / "voice_characteristics.json"
        try:
            with open(char_path, "w") as f:
                json.dump({
                    "pitch_mean": chars.pitch_mean,
                    "pitch_std": chars.pitch_std,
                    "pitch_min": chars.pitch_min,
                    "pitch_max": chars.pitch_max,
                    "formant_shift": chars.formant_shift,
                }, f, indent=2)
            logger.info(f"Saved voice characteristics to {char_path}")
        except Exception as e:
            logger.warning(f"Failed to save voice characteristics: {e}")

    def _apply_formant_shift(
        self,
        audio: np.ndarray,
        sr: int,
        shift: float,
    ) -> np.ndarray:
        """
        Apply formant shift using resampling technique.

        Positive shift = higher formants (more feminine/childlike)
        Negative shift = lower formants (more masculine/deeper)
        """
        if abs(shift) < 0.05:
            return audio

        # Formant shift via resampling
        # shift > 0: stretch time, then pitch up
        # shift < 0: compress time, then pitch down
        stretch_factor = 1.0 + shift * 0.3  # Limit the effect

        try:
            # Time stretch
            stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)

            # Pitch shift to compensate (opposite direction)
            compensate_semitones = -12 * np.log2(stretch_factor)
            result = librosa.effects.pitch_shift(
                stretched,
                sr=sr,
                n_steps=compensate_semitones,
            )

            return result
        except Exception as e:
            logger.warning(f"Formant shift failed: {e}")
            return audio

    def _apply_smoothing(self, audio: np.ndarray, strength: float) -> np.ndarray:
        """Apply temporal smoothing to reduce artifacts."""
        from scipy.ndimage import gaussian_filter1d

        sigma = strength * 2
        if sigma > 0:
            audio = gaussian_filter1d(audio, sigma=sigma)
        return audio

    def _apply_crossfade(
        self,
        audio: np.ndarray,
        sr: int,
        crossfade_ms: float,
    ) -> np.ndarray:
        """Apply fade in/out at the edges."""
        fade_samples = int(sr * crossfade_ms / 1000)

        if fade_samples > 0 and len(audio) > 2 * fade_samples:
            # Fade in
            fade_in = np.linspace(0, 1, fade_samples)
            audio = audio.copy()
            audio[:fade_samples] *= fade_in

            # Fade out
            fade_out = np.linspace(1, 0, fade_samples)
            audio[-fade_samples:] *= fade_out

        return audio

    def clear_cache(self):
        """Clear voice characteristics cache."""
        self._voice_cache.clear()

    def set_device(self, device):
        """Set device (no-op for signal processing, kept for API compatibility)."""
        pass
