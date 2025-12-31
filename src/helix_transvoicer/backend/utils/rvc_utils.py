"""
RVC (Retrieval-based Voice Conversion) utilities.

Provides utilities for importing, managing, and training RVC voice models.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

from helix_transvoicer.backend.utils.config import get_settings

logger = logging.getLogger("helix.rvc_utils")

# Pre-trained model URLs (HuBERT and base models)
PRETRAINED_MODELS = {
    "hubert_base": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "path": "assets/hubert/hubert_base.pt",
        "size_mb": 360,
    },
    "rmvpe": {
        "url": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
        "path": "assets/rmvpe/rmvpe.pt",
        "size_mb": 180,
    },
}


class RVCModelImporter:
    """Import and manage RVC voice models."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.settings = get_settings()
        self.models_dir = models_dir or self.settings.models_dir
        self.assets_dir = self.settings.models_dir.parent / "assets"

    def import_model(
        self,
        source_path: Path,
        model_name: str,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[bool, str]:
        """
        Import an RVC model from a file or directory.

        Supports:
        - Single .pth file
        - Directory containing .pth and optional .index files
        - .zip archive containing model files

        Args:
            source_path: Path to model file, directory, or zip
            model_name: Name for the imported model
            progress_callback: Optional callback(message, progress)

        Returns:
            (success, message) tuple
        """

        def update(msg: str, progress: float = 0):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg, progress)

        try:
            update(f"Importing model: {model_name}", 0.1)

            # Create model directory
            model_dir = self.models_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            source = Path(source_path)

            if source.suffix == ".zip":
                # Extract zip file
                update("Extracting zip archive...", 0.3)
                with zipfile.ZipFile(source, 'r') as zf:
                    zf.extractall(model_dir)

            elif source.suffix == ".pth":
                # Copy single .pth file
                update("Copying model file...", 0.3)
                shutil.copy(source, model_dir / f"{model_name}.pth")

                # Check for matching .index file
                index_file = source.with_suffix(".index")
                if index_file.exists():
                    shutil.copy(index_file, model_dir / f"{model_name}.index")

            elif source.is_dir():
                # Copy directory contents
                update("Copying model directory...", 0.3)
                for file in source.glob("*"):
                    if file.suffix in [".pth", ".index", ".json"]:
                        shutil.copy(file, model_dir / file.name)

            else:
                return False, f"Unsupported file type: {source.suffix}"

            # Verify model files
            update("Verifying model...", 0.8)
            pth_files = list(model_dir.glob("*.pth"))
            if not pth_files:
                shutil.rmtree(model_dir)
                return False, "No .pth model file found"

            # Create metadata
            update("Creating metadata...", 0.9)
            self._create_model_metadata(model_dir, model_name)

            update("Import complete!", 1.0)
            return True, f"Successfully imported model: {model_name}"

        except Exception as e:
            logger.error(f"Failed to import model: {e}")
            return False, str(e)

    def _create_model_metadata(self, model_dir: Path, model_name: str):
        """Create metadata file for imported model."""
        from datetime import datetime

        metadata = {
            "model_id": model_name,
            "version": "1.0.0",
            "type": "rvc",
            "created_at": datetime.now().isoformat(),
            "imported": True,
        }

        pth_files = list(model_dir.glob("*.pth"))
        index_files = list(model_dir.glob("*.index"))

        metadata["rvc_model"] = str(pth_files[0].name) if pth_files else None
        metadata["rvc_index"] = str(index_files[0].name) if index_files else None

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def list_models(self) -> List[Dict]:
        """List all available RVC models."""
        models = []

        if not self.models_dir.exists():
            return models

        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            pth_files = list(model_dir.glob("*.pth"))
            if not pth_files:
                continue

            index_files = list(model_dir.glob("*.index"))

            # Load metadata if available
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            models.append({
                "id": model_dir.name,
                "name": metadata.get("model_id", model_dir.name),
                "type": metadata.get("type", "unknown"),
                "model_file": pth_files[0].name,
                "has_index": len(index_files) > 0,
                "created_at": metadata.get("created_at"),
            })

        return models

    def delete_model(self, model_name: str) -> Tuple[bool, str]:
        """Delete an RVC model."""
        model_dir = self.models_dir / model_name

        if not model_dir.exists():
            return False, f"Model not found: {model_name}"

        try:
            shutil.rmtree(model_dir)
            return True, f"Deleted model: {model_name}"
        except Exception as e:
            return False, str(e)


class RVCAssetManager:
    """Manage RVC pre-trained assets (HuBERT, RMVPE, etc.)."""

    def __init__(self, assets_dir: Optional[Path] = None):
        settings = get_settings()
        self.assets_dir = assets_dir or (settings.models_dir.parent / "assets")

    def check_assets(self) -> Dict[str, bool]:
        """Check which pre-trained assets are available."""
        status = {}
        for name, info in PRETRAINED_MODELS.items():
            path = self.assets_dir / info["path"]
            status[name] = path.exists()
        return status

    def download_asset(
        self,
        asset_name: str,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[bool, str]:
        """Download a pre-trained asset."""
        if asset_name not in PRETRAINED_MODELS:
            return False, f"Unknown asset: {asset_name}"

        info = PRETRAINED_MODELS[asset_name]
        target_path = self.assets_dir / info["path"]

        if target_path.exists():
            return True, f"Asset already exists: {asset_name}"

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {asset_name} ({info['size_mb']}MB)...")

            response = requests.get(info["url"], stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(target_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size:
                        progress_callback(
                            f"Downloading {asset_name}...",
                            downloaded / total_size
                        )

            logger.info(f"Downloaded {asset_name} successfully")
            return True, f"Downloaded {asset_name}"

        except Exception as e:
            logger.error(f"Failed to download {asset_name}: {e}")
            if target_path.exists():
                target_path.unlink()
            return False, str(e)

    def download_all_assets(
        self,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[bool, str]:
        """Download all required pre-trained assets."""
        errors = []

        for i, name in enumerate(PRETRAINED_MODELS.keys()):
            def sub_callback(msg, prog):
                total_prog = (i + prog) / len(PRETRAINED_MODELS)
                if progress_callback:
                    progress_callback(msg, total_prog)

            success, msg = self.download_asset(name, sub_callback)
            if not success:
                errors.append(f"{name}: {msg}")

        if errors:
            return False, "Some assets failed: " + "; ".join(errors)
        return True, "All assets downloaded successfully"


def check_rvc_installation() -> Dict:
    """Check if RVC dependencies are properly installed."""
    status = {
        "rvc_python": False,
        "faiss": False,
        "hubert_model": False,
        "rmvpe_model": False,
    }

    # Check rvc-python
    try:
        from rvc_python.infer import RVCInference
        status["rvc_python"] = True
    except ImportError:
        pass

    # Check faiss
    try:
        import faiss
        status["faiss"] = True
    except ImportError:
        pass

    # Check pre-trained models
    asset_manager = RVCAssetManager()
    asset_status = asset_manager.check_assets()
    status["hubert_model"] = asset_status.get("hubert_base", False)
    status["rmvpe_model"] = asset_status.get("rmvpe", False)

    return status


def install_rvc_dependencies():
    """Install RVC dependencies via pip."""
    packages = [
        "rvc-python>=0.2.0",
        "faiss-cpu>=1.7.0",
    ]

    for package in packages:
        logger.info(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    logger.info("RVC dependencies installed successfully")
