"""
Helix Transvoicer - Background job queue.

Handles long-running tasks like training and batch conversion.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("helix.job_queue")


class JobStatus(str, Enum):
    """Job status states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Types of background jobs."""

    TRAINING = "training"
    CONVERSION = "conversion"
    BATCH_CONVERSION = "batch_conversion"
    MODEL_UPDATE = "model_update"
    ANALYSIS = "analysis"


@dataclass
class Job:
    """Represents a background job."""

    id: str
    type: JobType
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    stage: str = ""
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class JobQueue:
    """
    Manages background job execution.

    Features:
    - Async job execution
    - Progress tracking
    - Job cancellation
    - Result storage
    """

    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self._jobs: Dict[str, Job] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._shutdown = False

    async def submit(
        self,
        job_type: JobType,
        func: Callable,
        *args,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> Job:
        """
        Submit a new job for execution.

        Args:
            job_type: Type of job
            func: Async function to execute
            *args: Function arguments
            metadata: Optional job metadata
            **kwargs: Function keyword arguments

        Returns:
            Job object
        """
        job_id = str(uuid.uuid4())

        job = Job(
            id=job_id,
            type=job_type,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            metadata=metadata or {},
        )

        self._jobs[job_id] = job

        # Create task
        task = asyncio.create_task(
            self._execute_job(job, func, *args, **kwargs)
        )
        self._tasks[job_id] = task

        logger.info(f"Submitted job {job_id} ({job_type})")
        return job

    async def _execute_job(
        self,
        job: Job,
        func: Callable,
        *args,
        **kwargs,
    ) -> None:
        """Execute a job with semaphore control."""
        async with self._semaphore:
            if self._shutdown:
                job.status = JobStatus.CANCELLED
                return

            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()

            try:
                # Create progress callback
                def progress_callback(stage: str, progress: float):
                    job.stage = stage
                    job.progress = progress

                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, progress_callback=progress_callback, **kwargs)
                else:
                    # Run sync functions in thread pool to avoid blocking event loop
                    import functools
                    partial_func = functools.partial(func, *args, progress_callback=progress_callback, **kwargs)
                    result = await asyncio.to_thread(partial_func)

                job.result = result
                job.status = JobStatus.COMPLETED
                job.progress = 1.0

            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                logger.info(f"Job {job.id} cancelled")

            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
                logger.error(f"Job {job.id} failed: {e}")

            finally:
                job.completed_at = datetime.now()

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        job_type: Optional[JobType] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> List[Job]:
        """List jobs with optional filtering."""
        jobs = list(self._jobs.values())

        if job_type:
            jobs = [j for j in jobs if j.type == job_type]

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by creation time, newest first
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or pending job."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
            return False

        task = self._tasks.get(job_id)
        if task:
            task.cancel()

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()

        logger.info(f"Cancelled job {job_id}")
        return True

    async def wait_for_job(
        self,
        job_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Job]:
        """Wait for a job to complete."""
        task = self._tasks.get(job_id)
        if not task:
            return self._jobs.get(job_id)

        try:
            await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError:
            pass

        return self._jobs.get(job_id)

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Remove old completed jobs."""
        now = datetime.now()
        to_remove = []

        for job_id, job in self._jobs.items():
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.completed_at:
                    age = (now - job.completed_at).total_seconds() / 3600
                    if age > max_age_hours:
                        to_remove.append(job_id)

        for job_id in to_remove:
            del self._jobs[job_id]
            if job_id in self._tasks:
                del self._tasks[job_id]

        return len(to_remove)

    async def shutdown(self) -> None:
        """Shutdown the job queue, cancelling all pending jobs."""
        self._shutdown = True

        # Cancel all running tasks
        for task in self._tasks.values():
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        logger.info("Job queue shutdown complete")

    @property
    def pending_count(self) -> int:
        """Get number of pending jobs."""
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.PENDING)

    @property
    def running_count(self) -> int:
        """Get number of running jobs."""
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)
