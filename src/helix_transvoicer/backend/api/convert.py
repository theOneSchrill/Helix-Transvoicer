"""
Helix Transvoicer API - Voice conversion endpoints (Applio-compatible).
"""

import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from helix_transvoicer.backend.core.voice_converter import ConversionConfig, VoiceConverter
from helix_transvoicer.backend.services.job_queue import JobQueue, JobStatus, JobType
from helix_transvoicer.backend.utils.audio import AudioUtils

router = APIRouter()


class ConversionRequest(BaseModel):
    """Voice conversion request with Applio-compatible parameters."""

    target_model_id: str

    # Pitch settings
    pitch_shift: int = Field(0, ge=-24, le=24, description="Pitch shift in semitones")
    f0_method: str = Field("rmvpe", description="Pitch extraction: crepe, crepe-tiny, rmvpe, fcpe")
    hop_length: int = Field(128, ge=64, le=512, description="Hop length for pitch extraction")

    # Index/Feature settings
    index_rate: float = Field(0.0, ge=0.0, le=1.0, description="Search Feature Ratio")
    index_file: Optional[str] = Field(None, description="Specific index file to use")

    # Audio processing
    rms_mix_rate: float = Field(0.4, ge=0.0, le=1.0, description="Volume Envelope")
    protect: float = Field(0.3, ge=0.0, le=0.5, description="Protect Voiceless Consonants")
    filter_radius: int = Field(3, ge=0, le=7, description="Median filter radius")

    # Split & Clean
    split_audio: bool = Field(False, description="Split audio into chunks")
    autotune: bool = Field(False, description="Apply soft autotune")
    clean_audio: bool = Field(False, description="Apply noise reduction")
    clean_strength: float = Field(0.4, ge=0.0, le=1.0, description="Cleaning strength")

    # Formant shifting
    formant_shifting: bool = Field(False, description="Enable formant shift")
    formant_preset: str = Field("m2f", description="Formant preset")
    formant_quefrency: float = Field(1.0, ge=0.0, le=16.0, description="Quefrency")
    formant_timbre: float = Field(1.2, ge=0.0, le=16.0, description="Timbre")

    # Post processing
    post_process: bool = Field(False, description="Apply post-processing effects")
    normalize_output: bool = Field(True, description="Normalize output audio")

    # Model settings
    speaker_id: int = Field(0, ge=0, description="Speaker ID for multi-speaker models")
    embedder_model: str = Field("contentvec", description="Embedder model")

    # Export settings
    export_format: str = Field("wav", description="Output format: wav, mp3, flac, ogg, m4a")


class ConversionResponse(BaseModel):
    """Voice conversion response."""

    duration: float
    source_duration: float
    model_id: str
    processing_time: float


class IndexFileInfo(BaseModel):
    """Index file information."""

    name: str
    path: str


class ModelFilesResponse(BaseModel):
    """Model files response."""

    model_id: str
    pth_files: List[str]
    index_files: List[IndexFileInfo]


class JobResponse(BaseModel):
    """Background job response."""

    job_id: str
    status: str
    progress: float
    stage: str


@router.get("/models/{model_id}/files", response_model=ModelFilesResponse)
async def get_model_files(
    request: Request,
    model_id: str,
):
    """Get available model and index files for a model."""
    model_manager = request.app.state.model_manager

    model = model_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    model_dir = model_manager.models_dir / model_id

    pth_files = [f.name for f in model_dir.glob("*.pth")]
    index_files = [
        IndexFileInfo(name=f.name, path=str(f))
        for f in model_dir.glob("*.index")
    ]

    return ModelFilesResponse(
        model_id=model_id,
        pth_files=pth_files,
        index_files=index_files,
    )


@router.post("/voice", response_model=ConversionResponse)
async def convert_voice(
    request: Request,
    file: UploadFile = File(...),
    target_model_id: str = Query(..., description="Target voice model ID"),
    # Pitch settings
    pitch_shift: int = Query(0, ge=-24, le=24, description="Pitch shift in semitones"),
    f0_method: str = Query("rmvpe", description="Pitch extraction algorithm"),
    hop_length: int = Query(128, ge=64, le=512, description="Hop length"),
    # Index settings
    index_rate: float = Query(0.0, ge=0.0, le=1.0, description="Search Feature Ratio"),
    index_file: Optional[str] = Query(None, description="Specific index file"),
    # Audio processing
    rms_mix_rate: float = Query(0.4, ge=0.0, le=1.0, description="Volume Envelope"),
    protect: float = Query(0.3, ge=0.0, le=0.5, description="Protect Voiceless Consonants"),
    filter_radius: int = Query(3, ge=0, le=7, description="Median filter radius"),
    # Split & Clean
    split_audio: bool = Query(False, description="Split audio into chunks"),
    autotune: bool = Query(False, description="Apply soft autotune"),
    clean_audio: bool = Query(False, description="Apply noise reduction"),
    clean_strength: float = Query(0.4, ge=0.0, le=1.0, description="Clean strength"),
    # Formant shifting
    formant_shifting: bool = Query(False, description="Enable formant shift"),
    formant_quefrency: float = Query(1.0, ge=0.0, le=16.0, description="Quefrency"),
    formant_timbre: float = Query(1.2, ge=0.0, le=16.0, description="Timbre"),
    # Post processing
    normalize: bool = Query(True, description="Normalize output"),
    # Model settings
    speaker_id: int = Query(0, ge=0, description="Speaker ID"),
    embedder_model: str = Query("contentvec", description="Embedder model"),
):
    """
    Convert voice in audio file to target voice (Applio-compatible).

    Synchronous conversion with full parameter support.
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
            f0_method=f0_method,
            hop_length=hop_length,
            index_rate=index_rate,
            index_file=index_file,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            filter_radius=filter_radius,
            split_audio=split_audio,
            autotune=autotune,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            formant_shifting=formant_shifting,
            formant_quefrency=formant_quefrency,
            formant_timbre=formant_timbre,
            normalize_output=normalize,
            speaker_id=speaker_id,
            embedder_model=embedder_model,
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
    # Pitch settings
    pitch_shift: int = Query(0, ge=-24, le=24),
    f0_method: str = Query("rmvpe"),
    hop_length: int = Query(128, ge=64, le=512),
    # Index settings
    index_rate: float = Query(0.0, ge=0.0, le=1.0),
    index_file: Optional[str] = Query(None),
    # Audio processing
    rms_mix_rate: float = Query(0.4, ge=0.0, le=1.0),
    protect: float = Query(0.3, ge=0.0, le=0.5),
    filter_radius: int = Query(3, ge=0, le=7),
    # Split & Clean
    split_audio: bool = Query(False),
    autotune: bool = Query(False),
    clean_audio: bool = Query(False),
    clean_strength: float = Query(0.4, ge=0.0, le=1.0),
    # Formant shifting
    formant_shifting: bool = Query(False),
    formant_quefrency: float = Query(1.0, ge=0.0, le=16.0),
    formant_timbre: float = Query(1.2, ge=0.0, le=16.0),
    # Post processing
    normalize: bool = Query(True),
    # Model settings
    speaker_id: int = Query(0, ge=0),
    embedder_model: str = Query("contentvec"),
    # Export settings
    export_format: str = Query("wav", description="Output format"),
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
            f0_method=f0_method,
            hop_length=hop_length,
            index_rate=index_rate,
            index_file=index_file,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            filter_radius=filter_radius,
            split_audio=split_audio,
            autotune=autotune,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            formant_shifting=formant_shifting,
            formant_quefrency=formant_quefrency,
            formant_timbre=formant_timbre,
            normalize_output=normalize,
            speaker_id=speaker_id,
            embedder_model=embedder_model,
            export_format=export_format,
        )

        result = converter.convert(tmp_path, target_model_id, config)

        # Determine format and media type
        format_map = {
            "wav": ("audio/wav", "wav"),
            "mp3": ("audio/mpeg", "mp3"),
            "flac": ("audio/flac", "flac"),
            "ogg": ("audio/ogg", "ogg"),
            "m4a": ("audio/mp4", "m4a"),
        }
        media_type, ext = format_map.get(export_format, ("audio/wav", "wav"))

        audio_bytes = AudioUtils.audio_to_bytes(
            result.audio,
            result.sample_rate,
            format=ext,
        )

        return StreamingResponse(
            iter([audio_bytes]),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=converted_{target_model_id}.{ext}"
            },
        )

    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/voice/async", response_model=JobResponse)
async def convert_voice_async(
    request: Request,
    file: UploadFile = File(...),
    target_model_id: str = Query(...),
    # Pitch settings
    pitch_shift: int = Query(0, ge=-24, le=24),
    f0_method: str = Query("rmvpe"),
    # Index settings
    index_rate: float = Query(0.0, ge=0.0, le=1.0),
    # Audio processing
    rms_mix_rate: float = Query(0.4, ge=0.0, le=1.0),
    protect: float = Query(0.3, ge=0.0, le=0.5),
    # Split & Clean
    split_audio: bool = Query(False),
    clean_audio: bool = Query(False),
    # Formant shifting
    formant_shifting: bool = Query(False),
    # Export
    export_format: str = Query("wav"),
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

        config = ConversionConfig(
            pitch_shift=pitch_shift,
            f0_method=f0_method,
            index_rate=index_rate,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
            split_audio=split_audio,
            clean_audio=clean_audio,
            formant_shifting=formant_shifting,
            export_format=export_format,
        )
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
