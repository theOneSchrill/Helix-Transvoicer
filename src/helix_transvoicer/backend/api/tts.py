"""
Helix Transvoicer API - Text-to-Speech endpoints.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from helix_transvoicer.backend.core.tts_engine import TTSConfig, TTSEngine
from helix_transvoicer.backend.utils.audio import AudioUtils

router = APIRouter()


class TTSRequest(BaseModel):
    """TTS synthesis request."""

    text: str
    voice_model_id: str
    speed: float = 1.0
    pitch: float = 0.0
    intensity: float = 0.5
    variance: float = 0.5
    emotion: str = "neutral"
    emotion_strength: float = 0.5
    secondary_emotion: Optional[str] = None
    secondary_emotion_blend: float = 0.0
    add_pauses: bool = False
    add_breathing: bool = True
    whisper_mode: bool = False


class TTSResponse(BaseModel):
    """TTS synthesis response."""

    duration: float
    sample_rate: int
    processing_time: float
    phoneme_count: int


class VoiceInfo(BaseModel):
    """Voice model info for TTS."""

    id: str
    name: str
    version: str


@router.post("/synthesize", response_model=TTSResponse)
async def synthesize(request: Request, body: TTSRequest):
    """
    Synthesize speech from text.

    Returns metadata about the synthesis.
    """
    device_manager = request.app.state.device_manager
    model_manager = request.app.state.model_manager

    model = model_manager.get_model(body.voice_model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Voice model not found")

    engine = TTSEngine(
        device=device_manager.device,
        models_dir=model_manager.models_dir,
    )

    config = TTSConfig(
        speed=body.speed,
        pitch=body.pitch,
        intensity=body.intensity,
        variance=body.variance,
        emotion=body.emotion,
        emotion_strength=body.emotion_strength,
        secondary_emotion=body.secondary_emotion,
        secondary_emotion_blend=body.secondary_emotion_blend,
        add_pauses=body.add_pauses,
        add_breathing=body.add_breathing,
        whisper_mode=body.whisper_mode,
    )

    result = engine.synthesize(body.text, body.voice_model_id, config)

    return TTSResponse(
        duration=result.duration,
        sample_rate=result.sample_rate,
        processing_time=result.processing_time,
        phoneme_count=len(result.phonemes),
    )


@router.post("/synthesize/download")
async def synthesize_and_download(request: Request, body: TTSRequest):
    """
    Synthesize speech and return the audio file.
    """
    device_manager = request.app.state.device_manager
    model_manager = request.app.state.model_manager

    model = model_manager.get_model(body.voice_model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Voice model not found")

    engine = TTSEngine(
        device=device_manager.device,
        models_dir=model_manager.models_dir,
    )

    config = TTSConfig(
        speed=body.speed,
        pitch=body.pitch,
        intensity=body.intensity,
        variance=body.variance,
        emotion=body.emotion,
        emotion_strength=body.emotion_strength,
        secondary_emotion=body.secondary_emotion,
        secondary_emotion_blend=body.secondary_emotion_blend,
        add_pauses=body.add_pauses,
        add_breathing=body.add_breathing,
        whisper_mode=body.whisper_mode,
    )

    result = engine.synthesize(body.text, body.voice_model_id, config)

    audio_bytes = AudioUtils.audio_to_bytes(
        result.audio,
        result.sample_rate,
        format="wav",
    )

    return StreamingResponse(
        iter([audio_bytes]),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename=tts_{body.voice_model_id}.wav"
        },
    )


@router.post("/preview")
async def preview(
    request: Request,
    text: str = Query(..., max_length=200),
    voice_model_id: str = Query(...),
    emotion: str = Query("neutral"),
):
    """
    Quick TTS preview with default settings.

    Limited to 200 characters for fast response.
    """
    device_manager = request.app.state.device_manager
    model_manager = request.app.state.model_manager

    model = model_manager.get_model(voice_model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Voice model not found")

    engine = TTSEngine(
        device=device_manager.device,
        models_dir=model_manager.models_dir,
    )

    config = TTSConfig(emotion=emotion)
    result = engine.preview(text, voice_model_id, config)

    audio_bytes = AudioUtils.audio_to_bytes(
        result.audio,
        result.sample_rate,
        format="wav",
    )

    return StreamingResponse(
        iter([audio_bytes]),
        media_type="audio/wav",
    )


@router.get("/voices", response_model=List[VoiceInfo])
async def list_voices(request: Request):
    """List available voice models for TTS."""
    model_manager = request.app.state.model_manager
    models = model_manager.list_models()

    return [
        VoiceInfo(
            id=m.id,
            name=m.name,
            version=m.version,
        )
        for m in models
    ]


@router.get("/emotions")
async def list_emotions():
    """List available emotions for TTS."""
    return {
        "emotions": [
            {"id": "neutral", "name": "Neutral", "valence": 0.0, "arousal": 0.0},
            {"id": "happy", "name": "Happy", "valence": 0.8, "arousal": 0.6},
            {"id": "sad", "name": "Sad", "valence": -0.7, "arousal": -0.4},
            {"id": "angry", "name": "Angry", "valence": -0.6, "arousal": 0.8},
            {"id": "fear", "name": "Fear", "valence": -0.8, "arousal": 0.7},
            {"id": "surprise", "name": "Surprise", "valence": 0.2, "arousal": 0.8},
            {"id": "disgust", "name": "Disgust", "valence": -0.7, "arousal": 0.2},
            {"id": "calm", "name": "Calm", "valence": 0.3, "arousal": -0.6},
            {"id": "excited", "name": "Excited", "valence": 0.7, "arousal": 0.9},
        ]
    }
