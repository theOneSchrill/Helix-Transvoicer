"""
Helix Transvoicer Backend - FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from helix_transvoicer.backend.api import audio, convert, models, tts, system
from helix_transvoicer.backend.services.model_manager import ModelManager
from helix_transvoicer.backend.services.job_queue import JobQueue
from helix_transvoicer.backend.utils.config import get_settings
from helix_transvoicer.backend.utils.device import DeviceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("helix.backend")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown."""
    settings = get_settings()

    # Initialize directories
    for path in [settings.models_dir, settings.cache_dir, settings.exports_dir]:
        Path(path).mkdir(parents=True, exist_ok=True)

    # Initialize device manager
    device_manager = DeviceManager()
    app.state.device_manager = device_manager
    logger.info(f"Device: {device_manager.device} ({device_manager.device_name})")

    # Initialize model manager
    model_manager = ModelManager(
        models_dir=settings.models_dir,
        device=device_manager.device,
    )
    await model_manager.initialize()
    app.state.model_manager = model_manager
    logger.info(f"Loaded {len(model_manager.models)} voice models")

    # Initialize job queue
    job_queue = JobQueue()
    app.state.job_queue = job_queue
    logger.info("Job queue initialized")

    logger.info("Helix Transvoicer backend ready")

    yield

    # Cleanup
    await job_queue.shutdown()
    logger.info("Helix Transvoicer backend shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Helix Transvoicer API",
        description="Studio-grade voice conversion and TTS API",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Local-only, so allow all
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routers
    app.include_router(audio.router, prefix="/api/audio", tags=["Audio"])
    app.include_router(convert.router, prefix="/api/convert", tags=["Conversion"])
    app.include_router(models.router, prefix="/api/models", tags=["Models"])
    app.include_router(tts.router, prefix="/api/tts", tags=["TTS"])
    app.include_router(system.router, prefix="/api/system", tags=["System"])

    @app.get("/")
    async def root() -> dict:
        """Root endpoint with API information."""
        return {
            "name": "Helix Transvoicer API",
            "version": "1.0.0",
            "status": "running",
        }

    @app.get("/health")
    async def health() -> dict:
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc: Exception) -> JSONResponse:
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    return app


app = create_app()


def run_server(host: str = "127.0.0.1", port: int = 8420) -> None:
    """Run the API server."""
    settings = get_settings()
    uvicorn.run(
        "helix_transvoicer.backend.main:app",
        host=host,
        port=port,
        reload=settings.debug,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
