"""
Helix Transvoicer API - Audio processing endpoints.
"""

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from helix_transvoicer.backend.core.audio_processor import AudioProcessor, ProcessingConfig
from helix_transvoicer.backend.utils.audio import AudioUtils

router = APIRouter()


class ProcessingRequest(BaseModel):
    """Audio processing request."""

    target_sr: int = 22050
    normalize: bool = True
    normalize_db: float = -20.0
    trim_silence: bool = True
    trim_db: float = 30.0
    denoise: bool = False
    denoise_strength: float = 0.5


class AudioInfo(BaseModel):
    """Audio file information."""

    duration: float
    sample_rate: int
    channels: int
    format: str
    frames: int


class ProcessedAudioResponse(BaseModel):
    """Response for processed audio."""

    duration: float
    original_duration: float
    sample_rate: int
    was_trimmed: bool
    was_normalized: bool
    was_denoised: bool
    quality_score: float


@router.post("/preprocess", response_model=ProcessedAudioResponse)
async def preprocess_audio(
    request: Request,
    file: UploadFile = File(...),
    target_sr: int = Query(22050, ge=8000, le=48000),
    normalize: bool = Query(True),
    trim_silence: bool = Query(True),
    denoise: bool = Query(False),
):
    """
    Preprocess an audio file.

    Applies resampling, normalization, silence trimming, and optional denoising.
    """
    processor = AudioProcessor()

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        suffix=Path(file.filename or "audio.wav").suffix,
        delete=False,
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        config = ProcessingConfig(
            target_sr=target_sr,
            normalize=normalize,
            trim_silence=trim_silence,
            denoise=denoise,
        )

        result = processor.process(tmp_path, config)

        return ProcessedAudioResponse(
            duration=result.duration,
            original_duration=result.original_duration,
            sample_rate=result.sample_rate,
            was_trimmed=result.was_trimmed,
            was_normalized=result.was_normalized,
            was_denoised=result.was_denoised,
            quality_score=result.quality_score,
        )

    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/denoise")
async def denoise_audio(
    request: Request,
    file: UploadFile = File(...),
    strength: float = Query(0.5, ge=0.0, le=1.0),
):
    """
    Apply denoising to an audio file.

    Returns the denoised audio as a WAV file.
    """
    processor = AudioProcessor()

    with tempfile.NamedTemporaryFile(
        suffix=Path(file.filename or "audio.wav").suffix,
        delete=False,
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        config = ProcessingConfig(
            denoise=True,
            denoise_strength=strength,
            normalize=True,
            trim_silence=False,
        )

        result = processor.process(tmp_path, config)

        # Convert to bytes
        audio_bytes = AudioUtils.audio_to_bytes(
            result.audio,
            result.sample_rate,
            format="wav",
        )

        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=denoised.wav"
            },
        )

    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/analyze", response_model=AudioInfo)
async def analyze_audio(
    file: UploadFile = File(...),
):
    """
    Analyze an audio file and return its properties.
    """
    with tempfile.NamedTemporaryFile(
        suffix=Path(file.filename or "audio.wav").suffix,
        delete=False,
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        info = AudioUtils.get_audio_info(tmp_path)

        return AudioInfo(
            duration=info["duration"],
            sample_rate=info["sample_rate"],
            channels=info["channels"],
            format=info["format"],
            frames=info["frames"],
        )

    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/extract-features")
async def extract_features(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Extract audio features (mel spectrogram, F0, energy).

    Returns feature arrays as JSON.
    """
    processor = AudioProcessor()

    with tempfile.NamedTemporaryFile(
        suffix=Path(file.filename or "audio.wav").suffix,
        delete=False,
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = processor.process(tmp_path)
        features = processor.extract_features(result.audio, result.sample_rate)

        return {
            "mel_shape": list(features["mel_spectrogram"].shape),
            "f0_length": len(features["f0"]),
            "energy_length": len(features["energy"]),
            "mfcc_shape": list(features["mfcc"].shape),
        }

    finally:
        tmp_path.unlink(missing_ok=True)
