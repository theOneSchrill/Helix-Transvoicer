"""
Helix Transvoicer API - Voice conversion endpoints.
"""

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from helix_transvoicer.backend.core.voice_converter import ConversionConfig, VoiceConverter
from helix_transvoicer.backend.services.job_queue import JobQueue, JobStatus, JobType
from helix_transvoicer.backend.utils.audio import AudioUtils

router = APIRouter()


class ConversionRequest(BaseModel):
    """Voice conversion request."""

    target_model_id: str
    pitch_shift: float = 0.0
    formant_shift: float = 0.0
    smoothing: float = 0.5
    crossfade_ms: float = 20.0
    preserve_breath: bool = True
    preserve_background: bool = False
    normalize_output: bool = True


class ConversionResponse(BaseModel):
    """Voice conversion response."""

    duration: float
    source_duration: float
    model_id: str
    processing_time: float


class JobResponse(BaseModel):
    """Background job response."""

    job_id: str
    status: str
    progress: float
    stage: str


@router.post("/voice", response_model=ConversionResponse)
async def convert_voice(
    request: Request,
    file: UploadFile = File(...),
    target_model_id: str = Query(..., description="Target voice model ID"),
    pitch_shift: float = Query(0.0, ge=-12.0, le=12.0),
    formant_shift: float = Query(0.0, ge=-1.0, le=1.0),
    smoothing: float = Query(0.5, ge=0.0, le=1.0),
    normalize: bool = Query(True),
):
    """
    Convert voice in audio file to target voice.

    Synchronous conversion for short audio files.
    """
    device_manager = request.app.state.device_manager
    model_manager = request.app.state.model_manager

    # Verify model exists
    model = model_manager.get_model(target_model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {target_model_id}")

    converter = VoiceConverter(
        device=device_manager.device,
        models_dir=model_manager.models_dir,
    )

    with tempfile.NamedTemporaryFile(
        suffix=Path(file.filename or "audio.wav").suffix,
        delete=False,
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        config = ConversionConfig(
            pitch_shift=pitch_shift,
            formant_shift=formant_shift,
            smoothing=smoothing,
            normalize_output=normalize,
        )

        result = converter.convert(tmp_path, target_model_id, config)

        return ConversionResponse(
            duration=result.duration,
            source_duration=result.source_duration,
            model_id=result.model_id,
            processing_time=result.processing_time,
        )

    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/voice/download")
async def convert_and_download(
    request: Request,
    file: UploadFile = File(...),
    target_model_id: str = Query(...),
    pitch_shift: float = Query(0.0, ge=-12.0, le=12.0),
    smoothing: float = Query(0.5, ge=0.0, le=1.0),
):
    """
    Convert voice and return the converted audio file.
    """
    device_manager = request.app.state.device_manager
    model_manager = request.app.state.model_manager

    model = model_manager.get_model(target_model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {target_model_id}")

    converter = VoiceConverter(
        device=device_manager.device,
        models_dir=model_manager.models_dir,
    )

    with tempfile.NamedTemporaryFile(
        suffix=Path(file.filename or "audio.wav").suffix,
        delete=False,
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        config = ConversionConfig(
            pitch_shift=pitch_shift,
            smoothing=smoothing,
        )

        result = converter.convert(tmp_path, target_model_id, config)

        audio_bytes = AudioUtils.audio_to_bytes(
            result.audio,
            result.sample_rate,
            format="wav",
        )

        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=converted_{target_model_id}.wav"
            },
        )

    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/voice/async", response_model=JobResponse)
async def convert_voice_async(
    request: Request,
    file: UploadFile = File(...),
    target_model_id: str = Query(...),
    pitch_shift: float = Query(0.0, ge=-12.0, le=12.0),
):
    """
    Start asynchronous voice conversion.

    Returns a job ID for tracking progress.
    """
    job_queue: JobQueue = request.app.state.job_queue
    device_manager = request.app.state.device_manager
    model_manager = request.app.state.model_manager

    model = model_manager.get_model(target_model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {target_model_id}")

    # Save file
    with tempfile.NamedTemporaryFile(
        suffix=Path(file.filename or "audio.wav").suffix,
        delete=False,
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = str(tmp.name)

    async def convert_task(progress_callback):
        converter = VoiceConverter(
            device=device_manager.device,
            models_dir=model_manager.models_dir,
        )

        config = ConversionConfig(pitch_shift=pitch_shift)
        result = converter.convert(tmp_path, target_model_id, config, progress_callback)

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

        return {
            "audio": result.audio.tolist(),
            "sample_rate": result.sample_rate,
            "duration": result.duration,
        }

    job = await job_queue.submit(
        JobType.CONVERSION,
        convert_task,
        metadata={"model_id": target_model_id, "source_file": file.filename},
    )

    return JobResponse(
        job_id=job.id,
        status=job.status.value,
        progress=job.progress,
        stage=job.stage,
    )


@router.get("/status/{job_id}", response_model=JobResponse)
async def get_conversion_status(
    request: Request,
    job_id: str,
):
    """Get status of a conversion job."""
    job_queue: JobQueue = request.app.state.job_queue
    job = job_queue.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=job.id,
        status=job.status.value,
        progress=job.progress,
        stage=job.stage,
    )


@router.post("/cancel/{job_id}")
async def cancel_conversion(
    request: Request,
    job_id: str,
):
    """Cancel a running conversion job."""
    job_queue: JobQueue = request.app.state.job_queue

    success = await job_queue.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not cancel job")

    return {"status": "cancelled"}
