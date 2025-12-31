"""
RVC Pre-trained Model Manager.

Downloads and manages pre-trained models required for RVC:
- HuBERT: Content feature extraction
- RMVPE: Pitch (F0) extraction
- Base models: Generator/Discriminator weights
"""

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from helix_transvoicer.backend.utils.config import get_settings

logger = logging.getLogger("helix.rvc.models")

# Pre-trained model definitions
PRETRAINED_MODELS = {
    # HuBERT for content feature extraction
    "hubert_base": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "filename": "hubert_base.pt",
        "subdir": "hubert",
        "size_mb": 360,
        "description": "HuBERT base model for content feature extraction",
    },
    # RMVPE for pitch extraction
    "rmvpe": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        "filename": "rmvpe.pt",
        "subdir": "rmvpe",
        "size_mb": 180,
        "description": "RMVPE model for pitch (F0) extraction",
    },
    # Pre-trained generator (v2, 40k sample rate)
    "pretrained_v2_G40k": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth",
        "filename": "f0G40k.pth",
        "subdir": "pretrained_v2",
        "size_mb": 55,
        "description": "Pre-trained generator (v2, 40k)",
    },
    # Pre-trained discriminator (v2, 40k sample rate)
    "pretrained_v2_D40k": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth",
        "filename": "f0D40k.pth",
        "subdir": "pretrained_v2",
        "size_mb": 55,
        "description": "Pre-trained discriminator (v2, 40k)",
    },
    # Pre-trained generator (v2, 48k sample rate)
    "pretrained_v2_G48k": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth",
        "filename": "f0G48k.pth",
        "subdir": "pretrained_v2",
        "size_mb": 55,
        "description": "Pre-trained generator (v2, 48k)",
    },
    # Pre-trained discriminator (v2, 48k sample rate)
    "pretrained_v2_D48k": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth",
        "filename": "f0D48k.pth",
        "subdir": "pretrained_v2",
        "size_mb": 55,
        "description": "Pre-trained discriminator (v2, 48k)",
    },
}

# Minimum required models for inference
REQUIRED_FOR_INFERENCE = ["hubert_base", "rmvpe"]

# Minimum required models for training
REQUIRED_FOR_TRAINING = [
    "hubert_base",
    "rmvpe",
    "pretrained_v2_G40k",
    "pretrained_v2_D40k",
]


class RVCModelManager:
    """Manages RVC pre-trained models and voice models."""

    def __init__(self, assets_dir: Optional[Path] = None):
        settings = get_settings()
        self.assets_dir = assets_dir or (settings.data_dir / "rvc_assets")
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to a pre-trained model."""
        if model_name not in PRETRAINED_MODELS:
            return None

        info = PRETRAINED_MODELS[model_name]
        return self.assets_dir / info["subdir"] / info["filename"]

    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a model is downloaded."""
        path = self.get_model_path(model_name)
        return path is not None and path.exists()

    def get_missing_models(self, for_training: bool = False) -> List[str]:
        """Get list of missing required models."""
        required = REQUIRED_FOR_TRAINING if for_training else REQUIRED_FOR_INFERENCE
        return [m for m in required if not self.is_model_downloaded(m)]

    def get_download_status(self) -> Dict[str, bool]:
        """Get download status of all models."""
        return {name: self.is_model_downloaded(name) for name in PRETRAINED_MODELS}

    def get_total_download_size(self, models: List[str]) -> int:
        """Get total download size in MB for given models."""
        return sum(
            PRETRAINED_MODELS[m]["size_mb"]
            for m in models
            if m in PRETRAINED_MODELS and not self.is_model_downloaded(m)
        )

    def download_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[bool, str]:
        """Download a single pre-trained model."""
        if model_name not in PRETRAINED_MODELS:
            return False, f"Unknown model: {model_name}"

        if self.is_model_downloaded(model_name):
            return True, f"{model_name} already downloaded"

        info = PRETRAINED_MODELS[model_name]
        target_dir = self.assets_dir / info["subdir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / info["filename"]

        try:
            logger.info(f"Downloading {model_name} ({info['size_mb']}MB)...")

            response = requests.get(info["url"], stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            # Download to temp file first
            temp_path = target_path.with_suffix(".tmp")

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)

                    if progress_callback and total_size:
                        progress = downloaded / total_size
                        progress_callback(f"Downloading {model_name}...", progress)

            # Move to final location
            shutil.move(str(temp_path), str(target_path))

            logger.info(f"Downloaded {model_name} successfully")
            return True, f"Downloaded {model_name}"

        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            # Cleanup temp file
            temp_path = target_path.with_suffix(".tmp")
            if temp_path.exists():
                temp_path.unlink()
            return False, str(e)

    def download_all_required(
        self,
        for_training: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[bool, str]:
        """Download all required pre-trained models."""
        missing = self.get_missing_models(for_training)

        if not missing:
            return True, "All required models already downloaded"

        total = len(missing)
        errors = []

        for i, model_name in enumerate(missing):
            def sub_progress(msg: str, prog: float):
                total_progress = (i + prog) / total
                if progress_callback:
                    progress_callback(msg, total_progress)

            success, message = self.download_model(model_name, sub_progress)
            if not success:
                errors.append(f"{model_name}: {message}")

        if errors:
            return False, "Some downloads failed: " + "; ".join(errors)

        return True, f"Downloaded {total} models successfully"


def download_pretrained_models(
    for_training: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[bool, str]:
    """Convenience function to download required pre-trained models."""
    manager = RVCModelManager()
    return manager.download_all_required(for_training, progress_callback)


def check_rvc_ready(for_training: bool = False) -> Tuple[bool, List[str]]:
    """Check if RVC is ready to use."""
    manager = RVCModelManager()
    missing = manager.get_missing_models(for_training)
    return len(missing) == 0, missing
