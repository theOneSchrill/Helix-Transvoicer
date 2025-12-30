"""
Helix Transvoicer - Audio preprocessing pipeline.

Handles resampling, denoising, silence trimming, and alignment.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import scipy.signal
import torch

from helix_transvoicer.backend.utils.audio import AudioUtils
from helix_transvoicer.backend.utils.config import get_settings

logger = logging.getLogger("helix.audio_processor")


@dataclass
class ProcessingConfig:
    """Audio processing configuration."""

    target_sr: int = 22050
    mono: bool = True
    normalize: bool = True
    normalize_db: float = -20.0
    trim_silence: bool = True
    trim_db: float = 30.0
    denoise: bool = False
    denoise_strength: float = 0.5
    min_duration: float = 0.5  # seconds
    max_duration: float = 300.0  # seconds
    skip_max_duration_check: bool = False  # Skip max duration validation (for training with auto-split)


@dataclass
class ProcessedAudio:
    """Result of audio processing."""

    audio: np.ndarray
    sample_rate: int
    duration: float
    original_duration: float
    was_trimmed: bool
    was_normalized: bool
    was_denoised: bool
    quality_score: float
    metadata: Dict = field(default_factory=dict)


class AudioProcessor:
    """
    Audio preprocessing pipeline for voice processing.

    Handles:
    - Format decoding and resampling
    - Denoising (spectral gating)
    - Silence trimming
    - Normalization
    - Quality assessment
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.settings = get_settings()

    def process(
        self,
        audio_path: Union[str, Path],
        config: Optional[ProcessingConfig] = None,
    ) -> ProcessedAudio:
        """
        Process an audio file through the full pipeline.

        Args:
            audio_path: Path to input audio file
            config: Optional processing configuration override

        Returns:
            ProcessedAudio result
        """
        cfg = config or self.config
        audio_path = Path(audio_path)

        logger.info(f"Processing audio: {audio_path.name}")

        # Load audio
        audio, sr = AudioUtils.load_audio(
            audio_path,
            target_sr=cfg.target_sr,
            mono=cfg.mono,
        )
        original_duration = len(audio) / sr

        # Validate duration
        if original_duration < cfg.min_duration:
            raise ValueError(
                f"Audio too short: {original_duration:.2f}s < {cfg.min_duration}s minimum"
            )

        if not cfg.skip_max_duration_check and original_duration > cfg.max_duration:
            raise ValueError(
                f"Audio too long: {original_duration:.2f}s > {cfg.max_duration}s maximum"
            )

        # Denoise if requested
        was_denoised = False
        if cfg.denoise:
            audio = self._denoise(audio, sr, cfg.denoise_strength)
            was_denoised = True

        # Trim silence
        was_trimmed = False
        if cfg.trim_silence:
            trimmed = AudioUtils.trim_silence(audio, top_db=cfg.trim_db)
            if len(trimmed) < len(audio):
                was_trimmed = True
                audio = trimmed

        # Normalize
        was_normalized = False
        if cfg.normalize:
            audio = AudioUtils.normalize(audio, target_db=cfg.normalize_db)
            was_normalized = True

        # Compute quality score
        quality_score = self._compute_quality_score(audio, sr)

        # Get duration
        duration = len(audio) / sr

        return ProcessedAudio(
            audio=audio,
            sample_rate=sr,
            duration=duration,
            original_duration=original_duration,
            was_trimmed=was_trimmed,
            was_normalized=was_normalized,
            was_denoised=was_denoised,
            quality_score=quality_score,
            metadata={
                "source_file": str(audio_path),
                "source_format": audio_path.suffix,
            },
        )

    def process_batch(
        self,
        audio_paths: List[Union[str, Path]],
        config: Optional[ProcessingConfig] = None,
    ) -> List[ProcessedAudio]:
        """Process multiple audio files."""
        results = []
        for path in audio_paths:
            try:
                result = self.process(path, config)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                continue
        return results

    def _denoise(
        self,
        audio: np.ndarray,
        sr: int,
        strength: float = 0.5,
    ) -> np.ndarray:
        """
        Apply spectral gating denoising.

        Uses a simple spectral gating approach:
        1. Estimate noise profile from quiet portions
        2. Apply frequency-domain gating
        """
        # Compute STFT
        n_fft = 2048
        hop_length = 512
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # Estimate noise floor from quietest 10% of frames
        frame_energy = np.sum(magnitude**2, axis=0)
        noise_frames = np.argsort(frame_energy)[: max(1, len(frame_energy) // 10)]
        noise_profile = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)

        # Apply spectral gating
        threshold = noise_profile * (1.0 + strength * 2.0)
        mask = (magnitude > threshold).astype(float)

        # Smooth the mask
        mask = scipy.signal.medfilt2d(mask.astype(np.float32), kernel_size=3)

        # Apply mask
        magnitude_clean = magnitude * mask

        # Reconstruct
        stft_clean = magnitude_clean * np.exp(1j * phase)
        audio_clean = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))

        return audio_clean

    def _compute_quality_score(self, audio: np.ndarray, sr: int) -> float:
        """
        Compute an audio quality score (0-1).

        Considers:
        - Signal-to-noise ratio estimate
        - Clipping detection
        - Silence ratio
        - Dynamic range
        """
        scores = []

        # Check for clipping (samples at max value)
        clipping_ratio = np.mean(np.abs(audio) > 0.99)
        clipping_score = 1.0 - min(clipping_ratio * 10, 1.0)
        scores.append(clipping_score)

        # Check dynamic range
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        if peak > 0:
            crest_factor = peak / (rms + 1e-8)
            # Good crest factor is around 4-10
            dynamic_score = min(crest_factor / 10, 1.0)
            scores.append(dynamic_score)

        # Check silence ratio
        threshold = 0.01
        silence_ratio = np.mean(np.abs(audio) < threshold)
        silence_score = 1.0 - min(silence_ratio * 2, 1.0)
        scores.append(silence_score)

        # Compute energy consistency
        frame_length = sr // 10  # 100ms frames
        if len(audio) > frame_length:
            n_frames = len(audio) // frame_length
            frames = audio[: n_frames * frame_length].reshape(n_frames, frame_length)
            frame_energy = np.sqrt(np.mean(frames**2, axis=1))
            energy_std = np.std(frame_energy) / (np.mean(frame_energy) + 1e-8)
            # Lower variance is better
            consistency_score = 1.0 - min(energy_std, 1.0)
            scores.append(consistency_score)

        return np.mean(scores)

    def extract_features(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> Dict[str, np.ndarray]:
        """
        Extract audio features for voice processing.

        Returns:
            Dictionary containing mel spectrogram, F0, energy, etc.
        """
        settings = self.settings

        # Mel spectrogram
        mel = AudioUtils.compute_mel_spectrogram(
            audio,
            sample_rate=sr,
            n_fft=settings.n_fft,
            hop_length=settings.hop_length,
            n_mels=settings.n_mels,
        )

        # F0 (pitch)
        f0, voiced = AudioUtils.extract_f0(
            audio,
            sample_rate=sr,
            hop_length=settings.hop_length,
        )

        # Energy
        energy = AudioUtils.compute_rms_energy(
            audio,
            frame_length=settings.n_fft,
            hop_length=settings.hop_length,
        )

        # MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13,
            n_fft=settings.n_fft,
            hop_length=settings.hop_length,
        )

        return {
            "mel_spectrogram": mel,
            "f0": f0,
            "voiced_mask": voiced,
            "energy": energy,
            "mfcc": mfccs,
        }

    def align_audio(
        self,
        audio: np.ndarray,
        sr: int,
        reference_duration: float,
    ) -> np.ndarray:
        """
        Time-stretch audio to match reference duration.

        Uses phase vocoder for high-quality time stretching.
        """
        current_duration = len(audio) / sr
        stretch_factor = reference_duration / current_duration

        if abs(stretch_factor - 1.0) < 0.01:
            return audio

        return librosa.effects.time_stretch(audio, rate=1.0 / stretch_factor)
