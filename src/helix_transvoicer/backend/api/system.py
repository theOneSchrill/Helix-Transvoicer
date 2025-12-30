"""
Helix Transvoicer API - System endpoints.
"""

from typing import Dict, List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class DeviceInfo(BaseModel):
    """Compute device information."""

    name: str
    type: str
    is_gpu: bool
    total_memory: Optional[int] = None
    available_memory: Optional[int] = None
    allocated_memory: Optional[int] = None


class SystemStatus(BaseModel):
    """System status information."""

    status: str
    device: DeviceInfo
    models_loaded: int
    models_total: int
    jobs_pending: int
    jobs_running: int


class JobInfo(BaseModel):
    """Job information."""

    id: str
    type: str
    status: str
    progress: float
    stage: str
    created_at: str
    completed_at: Optional[str] = None


@router.get("/status", response_model=SystemStatus)
async def get_status(request: Request):
    """Get current system status."""
    device_manager = request.app.state.device_manager
    model_manager = request.app.state.model_manager
    job_queue = request.app.state.job_queue

    memory_info = device_manager.get_memory_info()

    return SystemStatus(
        status="ready",
        device=DeviceInfo(
            name=device_manager.device_name,
            type=device_manager.device_type,
            is_gpu=device_manager.is_gpu,
            total_memory=memory_info.get("total_memory"),
            available_memory=memory_info.get("free_memory"),
            allocated_memory=memory_info.get("allocated_memory"),
        ),
        models_loaded=model_manager.get_loaded_count(),
        models_total=model_manager.get_total_count(),
        jobs_pending=job_queue.pending_count,
        jobs_running=job_queue.running_count,
    )


@router.get("/device", response_model=DeviceInfo)
async def get_device(request: Request):
    """Get compute device information."""
    device_manager = request.app.state.device_manager
    memory_info = device_manager.get_memory_info()

    return DeviceInfo(
        name=device_manager.device_name,
        type=device_manager.device_type,
        is_gpu=device_manager.is_gpu,
        total_memory=memory_info.get("total_memory"),
        available_memory=memory_info.get("free_memory"),
        allocated_memory=memory_info.get("allocated_memory"),
    )


@router.get("/jobs", response_model=List[JobInfo])
async def list_jobs(
    request: Request,
    status: Optional[str] = None,
    limit: int = 50,
):
    """List background jobs."""
    from helix_transvoicer.backend.services.job_queue import JobStatus

    job_queue = request.app.state.job_queue

    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status)
        except ValueError:
            pass

    jobs = job_queue.list_jobs(status=status_filter, limit=limit)

    return [
        JobInfo(
            id=job.id,
            type=job.type.value,
            status=job.status.value,
            progress=job.progress,
            stage=job.stage,
            created_at=job.created_at.isoformat(),
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
        )
        for job in jobs
    ]


@router.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job(request: Request, job_id: str):
    """Get details of a specific job."""
    from fastapi import HTTPException

    job_queue = request.app.state.job_queue
    job = job_queue.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobInfo(
        id=job.id,
        type=job.type.value,
        status=job.status.value,
        progress=job.progress,
        stage=job.stage,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str):
    """Cancel a running job."""
    from fastapi import HTTPException

    job_queue = request.app.state.job_queue

    success = await job_queue.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not cancel job")

    return {"status": "cancelled"}


@router.post("/jobs/{job_id}/pause")
async def pause_job(request: Request, job_id: str):
    """Pause a running job."""
    from fastapi import HTTPException

    job_queue = request.app.state.job_queue

    success = await job_queue.pause_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not pause job")

    return {"status": "paused"}


@router.post("/jobs/{job_id}/resume")
async def resume_job(request: Request, job_id: str):
    """Resume a paused job."""
    from fastapi import HTTPException

    job_queue = request.app.state.job_queue

    success = await job_queue.resume_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not resume job")

    return {"status": "resumed"}


@router.post("/cache/clear")
async def clear_cache(request: Request):
    """Clear GPU memory cache."""
    device_manager = request.app.state.device_manager
    device_manager.empty_cache()
    return {"status": "cache_cleared"}


@router.get("/config")
async def get_config(request: Request):
    """Get current configuration."""
    from helix_transvoicer.backend.utils.config import get_settings

    settings = get_settings()

    return {
        "sample_rate": settings.sample_rate,
        "n_fft": settings.n_fft,
        "hop_length": settings.hop_length,
        "n_mels": settings.n_mels,
        "max_audio_duration": settings.max_audio_duration,
        "models_dir": str(settings.models_dir),
        "cache_dir": str(settings.cache_dir),
        "exports_dir": str(settings.exports_dir),
    }
