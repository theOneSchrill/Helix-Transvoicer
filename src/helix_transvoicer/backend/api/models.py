"""
Helix Transvoicer API - Model management endpoints.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel

from helix_transvoicer.backend.core.model_trainer import ModelTrainer, TrainingConfig
from helix_transvoicer.backend.services.job_queue import JobType
from helix_transvoicer.backend.services.model_manager import VoiceModel

router = APIRouter()


class ModelInfo(BaseModel):
    """Voice model information."""

    id: str
    name: str
    version: str
    created_at: str
    updated_at: str
    total_samples: int
    total_duration: float
    quality_score: float
    is_loaded: bool
    emotion_coverage: Dict


class ModelCreateRequest(BaseModel):
    """Request to create a new model."""

    model_id: str
    name: Optional[str] = None


class TrainingRequest(BaseModel):
    """Training request."""

    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.0001
    auto_denoise: bool = True
    augment_data: bool = True


class JobResponse(BaseModel):
    """Job response."""

    job_id: str
    status: str
    progress: float
    stage: str


@router.get("", response_model=List[ModelInfo])
async def list_models(request: Request):
    """List all available voice models."""
    model_manager = request.app.state.model_manager
    models = model_manager.list_models()

    return [
        ModelInfo(
            id=m.id,
            name=m.name,
            version=m.version,
            created_at=m.created_at.isoformat(),
            updated_at=m.updated_at.isoformat(),
            total_samples=m.total_samples,
            total_duration=m.total_duration,
            quality_score=m.quality_score,
            is_loaded=m.is_loaded,
            emotion_coverage=m.emotion_coverage,
        )
        for m in models
    ]


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(request: Request, model_id: str):
    """Get details of a specific model."""
    model_manager = request.app.state.model_manager
    model = model_manager.get_model(model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return ModelInfo(
        id=model.id,
        name=model.name,
        version=model.version,
        created_at=model.created_at.isoformat(),
        updated_at=model.updated_at.isoformat(),
        total_samples=model.total_samples,
        total_duration=model.total_duration,
        quality_score=model.quality_score,
        is_loaded=model.is_loaded,
        emotion_coverage=model.emotion_coverage,
    )


@router.post("", response_model=ModelInfo)
async def create_model(request: Request, body: ModelCreateRequest):
    """Create a new empty voice model."""
    model_manager = request.app.state.model_manager

    try:
        model = await model_manager.create_model(
            body.model_id,
            metadata={"name": body.name} if body.name else None,
        )

        return ModelInfo(
            id=model.id,
            name=model.name,
            version=model.version,
            created_at=model.created_at.isoformat(),
            updated_at=model.updated_at.isoformat(),
            total_samples=model.total_samples,
            total_duration=model.total_duration,
            quality_score=model.quality_score,
            is_loaded=model.is_loaded,
            emotion_coverage=model.emotion_coverage,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{model_id}")
async def delete_model(request: Request, model_id: str):
    """Delete a voice model."""
    model_manager = request.app.state.model_manager

    success = await model_manager.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")

    return {"status": "deleted"}


@router.post("/{model_id}/train", response_model=JobResponse)
async def train_model(
    request: Request,
    model_id: str,
    files: List[UploadFile] = File(...),
    epochs: int = Query(100, ge=10, le=1000),
    batch_size: int = Query(16, ge=1, le=64),
    learning_rate: float = Query(0.0001, ge=0.00001, le=0.01),
    auto_denoise: bool = Query(True),
    augment_data: bool = Query(True),
):
    """
    Start training a voice model from audio samples.

    Returns a job ID for tracking progress.
    """
    job_queue = request.app.state.job_queue
    device_manager = request.app.state.device_manager
    model_manager = request.app.state.model_manager

    if len(files) < 3:
        raise HTTPException(
            status_code=400,
            detail="At least 3 audio samples required for training",
        )

    # Save uploaded files
    temp_paths = []
    for file in files:
        with tempfile.NamedTemporaryFile(
            suffix=Path(file.filename or "audio.wav").suffix,
            delete=False,
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_paths.append(tmp.name)

    async def train_task(progress_callback):
        trainer = ModelTrainer(
            device=device_manager.device,
            models_dir=model_manager.models_dir,
        )

        # Prepare samples
        progress_callback("Preparing samples", 0.1)
        samples = trainer.prepare_samples(
            temp_paths,
            progress_callback=lambda s, p: progress_callback(s, 0.1 + p * 0.2),
        )

        # Configure training
        config = TrainingConfig(
            model_name=model_id,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            auto_denoise=auto_denoise,
            augment_data=augment_data,
        )

        # Train
        def training_progress(p):
            progress_callback(
                f"Training epoch {p.epoch}/{p.total_epochs}",
                0.3 + (p.epoch / p.total_epochs) * 0.6,
            )

        result = trainer.train(samples, config, training_progress)

        # Cleanup temp files
        for path in temp_paths:
            Path(path).unlink(missing_ok=True)

        # Refresh model manager
        await model_manager.refresh()

        progress_callback("Complete", 1.0)

        return {
            "model_id": result.model_id,
            "version": result.version,
            "total_samples": result.total_samples,
            "final_loss": result.final_loss,
            "success": result.success,
            "error": result.error,
        }

    job = await job_queue.submit(
        JobType.TRAINING,
        train_task,
        metadata={"model_id": model_id, "sample_count": len(files)},
    )

    return JobResponse(
        job_id=job.id,
        status=job.status.value,
        progress=job.progress,
        stage=job.stage,
    )


@router.post("/{model_id}/update", response_model=JobResponse)
async def update_model(
    request: Request,
    model_id: str,
    files: List[UploadFile] = File(...),
    epochs: int = Query(20, ge=5, le=100),
):
    """
    Incrementally update a model with new samples.

    Uses fewer epochs and lower learning rate for fine-tuning.
    """
    job_queue = request.app.state.job_queue
    device_manager = request.app.state.device_manager
    model_manager = request.app.state.model_manager

    model = model_manager.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Save uploaded files
    temp_paths = []
    for file in files:
        with tempfile.NamedTemporaryFile(
            suffix=Path(file.filename or "audio.wav").suffix,
            delete=False,
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_paths.append(tmp.name)

    async def update_task(progress_callback):
        trainer = ModelTrainer(
            device=device_manager.device,
            models_dir=model_manager.models_dir,
        )

        progress_callback("Preparing new samples", 0.1)
        samples = trainer.prepare_samples(temp_paths)

        progress_callback("Updating model", 0.3)
        result = trainer.update_model(
            model_id,
            samples,
            epochs=epochs,
        )

        # Cleanup
        for path in temp_paths:
            Path(path).unlink(missing_ok=True)

        await model_manager.refresh()

        progress_callback("Complete", 1.0)

        return {
            "model_id": result.model_id,
            "version": result.version,
            "success": result.success,
        }

    job = await job_queue.submit(
        JobType.MODEL_UPDATE,
        update_task,
        metadata={"model_id": model_id},
    )

    return JobResponse(
        job_id=job.id,
        status=job.status.value,
        progress=job.progress,
        stage=job.stage,
    )


@router.get("/{model_id}/emotions")
async def get_emotion_coverage(request: Request, model_id: str):
    """Get emotion coverage analysis for a model."""
    model_manager = request.app.state.model_manager
    model = model_manager.get_model(model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "model_id": model_id,
        "coverage": model.emotion_coverage,
        "metadata": model.metadata.get("emotion_coverage", {}),
    }


@router.post("/{model_id}/load")
async def load_model(request: Request, model_id: str):
    """Load model into memory for faster inference."""
    model_manager = request.app.state.model_manager

    try:
        await model_manager.load_model(model_id)
        return {"status": "loaded"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{model_id}/unload")
async def unload_model(request: Request, model_id: str):
    """Unload model from memory."""
    model_manager = request.app.state.model_manager
    await model_manager.unload_model(model_id)
    return {"status": "unloaded"}
