"""
Helix Transvoicer - Audio utility functions.
"""

import io
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger("helix.audio")


class AudioUtils:
    """Utility class for common audio operations."""

    SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

    @staticmethod
    def load_audio(
        path: Union[str, Path],
        target_sr: int = 22050,
        mono: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file with automatic format detection.

        Args:
            path: Path to audio file
            target_sr: Target sample rate for resampling
            mono: Whether to convert to mono

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if path.suffix.lower() not in AudioUtils.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {path.suffix}")

        try:
            # Use librosa for robust loading
            audio, sr = librosa.load(str(path), sr=target_sr, mono=mono)
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise

    @staticmethod
    def save_audio(
        audio: Union[np.ndarray, torch.Tensor],
        path: Union[str, Path],
        sample_rate: int = 22050,
        format: str = "wav",
        subtype: str = "PCM_16",
    ) -> Path:
        """
        Save audio to file.

        Args:
            audio: Audio data as numpy array or torch tensor
            path: Output path
            sample_rate: Sample rate
            format: Output format
            subtype: Audio subtype (e.g., PCM_16, FLOAT)

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        if audio.ndim > 1:
            audio = audio.squeeze()

        sf.write(str(path), audio, sample_rate, subtype=subtype, format=format)
        return path

    @staticmethod
    def audio_to_bytes(
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 22050,
        format: str = "wav",
    ) -> bytes:
        """Convert audio array to bytes."""
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        if audio.ndim > 1:
            audio = audio.squeeze()

        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format=format)
        buffer.seek(0)
        return buffer.read()

    @staticmethod
    def get_audio_info(path: Union[str, Path]) -> dict:
        """Get audio file information."""
        path = Path(path)
        info = sf.info(str(path))

        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
            "frames": info.frames,
        }

    @staticmethod
    def resample(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    @staticmethod
    def normalize(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalize audio to target dB level."""
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 10 ** (target_db / 20)
            audio = audio * (target_rms / rms)
        return np.clip(audio, -1.0, 1.0)

    @staticmethod
    def trim_silence(
        audio: np.ndarray,
        top_db: float = 30.0,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """Trim leading and trailing silence."""
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        return trimmed

    @staticmethod
    def remove_silence(
        audio: np.ndarray,
        sample_rate: int = 22050,
        top_db: float = 30.0,
        min_silence_duration: float = 0.1,
        keep_short_silence: float = 0.05,
    ) -> np.ndarray:
        """
        Remove silent segments from audio (beginning, middle, and end).

        Args:
            audio: Audio samples
            sample_rate: Sample rate
            top_db: Threshold for silence detection (lower = more aggressive)
            min_silence_duration: Minimum silence duration to remove (seconds)
            keep_short_silence: Keep this much silence between segments (seconds)

        Returns:
            Audio with silence removed
        """
        # Get non-silent intervals
        intervals = librosa.effects.split(
            audio,
            top_db=top_db,
            frame_length=2048,
            hop_length=512,
        )

        if len(intervals) == 0:
            return audio

        # Calculate samples for minimum durations
        min_silence_samples = int(min_silence_duration * sample_rate)
        keep_samples = int(keep_short_silence * sample_rate)

        # Build output by concatenating non-silent segments with short gaps
        segments = []
        prev_end = 0

        for start, end in intervals:
            # Check if gap since last segment is long enough to be considered silence
            gap = start - prev_end
            if gap > min_silence_samples and len(segments) > 0:
                # Add a short silence between segments
                segments.append(np.zeros(keep_samples))

            segments.append(audio[start:end])
            prev_end = end

        if len(segments) == 0:
            return audio

        result = np.concatenate(segments)

        logger.info(f"Silence removal: {len(audio)/sample_rate:.1f}s -> {len(result)/sample_rate:.1f}s "
                    f"({(1 - len(result)/len(audio))*100:.0f}% removed)")

        return result

    @staticmethod
    def compute_mel_spectrogram(
        audio: np.ndarray,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ) -> np.ndarray:
        """Compute mel spectrogram."""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax or sample_rate / 2,
        )
        return librosa.power_to_db(mel, ref=np.max)

    @staticmethod
    def extract_f0(
        audio: np.ndarray,
        sample_rate: int = 22050,
        hop_length: int = 256,
        fmin: float = 50.0,
        fmax: float = 800.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract fundamental frequency (F0) using pYIN.

        Returns:
            Tuple of (f0_values, voiced_flag)
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            sr=sample_rate,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
        )
        return f0, voiced_flag

    @staticmethod
    def compute_rms_energy(
        audio: np.ndarray,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """Compute RMS energy."""
        return librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]
