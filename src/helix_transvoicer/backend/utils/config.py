"""
Helix Transvoicer - Configuration management.

Windows 11 optimized configuration.
"""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_windows_app_data() -> Path:
    """Get Windows AppData/Local directory for application data."""
    # Use LOCALAPPDATA for Windows (typically C:\Users\<user>\AppData\Local)
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "HelixTransvoicer"

    # Fallback to user home directory
    return Path.home() / "HelixTransvoicer"


def get_windows_documents() -> Path:
    """Get Windows Documents folder for exports."""
    # Try to get Documents folder from environment
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        docs = Path(user_profile) / "Documents" / "HelixTransvoicer"
        return docs

    return get_windows_app_data() / "exports"


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
    platform: str = "windows"

    # Directories (Windows 11 paths)
    data_dir: Path = Field(default_factory=get_windows_app_data)
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
    num_workers: int = 0  # Windows works better with 0 workers for DataLoader

    # Training settings
    default_epochs: int = 100
    default_learning_rate: float = 0.0001
    checkpoint_interval: int = 10

    # Device settings
    force_cpu: bool = False
    gpu_memory_fraction: float = 0.9

    # Windows-specific settings
    use_cuda_graphs: bool = False  # Can cause issues on some Windows systems
    console_encoding: str = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set derived paths if not explicitly set
        if self.models_dir is None:
            self.models_dir = self.data_dir / "models"
        if self.cache_dir is None:
            self.cache_dir = self.data_dir / "cache"
        if self.exports_dir is None:
            self.exports_dir = get_windows_documents()

        # Windows console encoding fix
        if sys.platform == "win32":
            try:
                sys.stdout.reconfigure(encoding=self.console_encoding)
                sys.stderr.reconfigure(encoding=self.console_encoding)
            except Exception:
                pass

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for path in [self.data_dir, self.models_dir, self.cache_dir, self.exports_dir]:
            if path:
                path.mkdir(parents=True, exist_ok=True)

    def get_temp_dir(self) -> Path:
        """Get Windows temp directory for processing."""
        temp = os.environ.get("TEMP") or os.environ.get("TMP")
        if temp:
            temp_path = Path(temp) / "HelixTransvoicer"
            temp_path.mkdir(parents=True, exist_ok=True)
            return temp_path
        return self.cache_dir


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
