"""
Helix Transvoicer - Configuration management.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="HELIX_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "Helix Transvoicer"
    debug: bool = False

    # Directories
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".helix-transvoicer")
    models_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    exports_dir: Optional[Path] = None

    # Audio settings
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    max_audio_duration: float = 300.0  # 5 minutes max

    # Processing settings
    batch_size: int = 16
    num_workers: int = 4

    # Training settings
    default_epochs: int = 100
    default_learning_rate: float = 0.0001
    checkpoint_interval: int = 10

    # Device settings
    force_cpu: bool = False
    gpu_memory_fraction: float = 0.9

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set derived paths if not explicitly set
        if self.models_dir is None:
            self.models_dir = self.data_dir / "models"
        if self.cache_dir is None:
            self.cache_dir = self.data_dir / "cache"
        if self.exports_dir is None:
            self.exports_dir = self.data_dir / "exports"

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for path in [self.data_dir, self.models_dir, self.cache_dir, self.exports_dir]:
            if path:
                path.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
