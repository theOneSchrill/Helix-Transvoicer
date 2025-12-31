"""
Voice model utilities.

Provides utilities for importing and managing voice models.
"""

import json
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from helix_transvoicer.backend.utils.config import get_settings

logger = logging.getLogger("helix.model_utils")


class VoiceModelImporter:
    """Import and manage voice models."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.settings = get_settings()
        self.models_dir = models_dir or self.settings.models_dir

    def import_model(
        self,
        source_path: Path,
        model_name: str,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[bool, str]:
        """
        Import a voice model from a file or directory.

        Supports:
        - Directory containing audio samples
        - .zip archive containing audio samples
        - RVC .pth model files (for external RVC usage)

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

            elif source.suffix in [".pth", ".pt"]:
                # Copy model file (RVC or other)
                update("Copying model file...", 0.3)
                shutil.copy(source, model_dir / source.name)

                # Check for matching .index file
                for ext in [".index", ".json"]:
                    companion = source.with_suffix(ext)
                    if companion.exists():
                        shutil.copy(companion, model_dir / companion.name)

            elif source.is_dir():
                # Copy directory contents
                update("Copying files...", 0.3)

                # Create samples directory
                samples_dir = model_dir / "samples"
                samples_dir.mkdir(exist_ok=True)

                # Copy audio files to samples
                audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
                for file in source.iterdir():
                    if file.suffix.lower() in audio_exts:
                        shutil.copy(file, samples_dir / file.name)
                    elif file.suffix in [".pth", ".pt", ".index", ".json"]:
                        shutil.copy(file, model_dir / file.name)

            else:
                return False, f"Unsupported file type: {source.suffix}"

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
        metadata = {
            "model_id": model_name,
            "version": "1.0.0",
            "type": "voice",
            "created_at": datetime.now().isoformat(),
            "imported": True,
        }

        # Check for sample files
        samples_dir = model_dir / "samples"
        if samples_dir.exists():
            sample_count = len(list(samples_dir.glob("*")))
            metadata["sample_count"] = sample_count

        # Check for RVC files
        pth_files = list(model_dir.glob("*.pth"))
        index_files = list(model_dir.glob("*.index"))

        if pth_files:
            metadata["rvc_model"] = str(pth_files[0].name)
        if index_files:
            metadata["rvc_index"] = str(index_files[0].name)

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def list_models(self) -> List[Dict]:
        """List all available voice models."""
        models = []

        if not self.models_dir.exists():
            return models

        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # Check for samples or model files
            samples_dir = model_dir / "samples"
            pth_files = list(model_dir.glob("*.pth"))

            if not samples_dir.exists() and not pth_files:
                continue

            # Load metadata if available
            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            # Count samples
            sample_count = 0
            if samples_dir.exists():
                sample_count = len(list(samples_dir.glob("*")))

            models.append({
                "id": model_dir.name,
                "name": metadata.get("model_id", model_dir.name),
                "type": metadata.get("type", "voice"),
                "sample_count": sample_count,
                "has_rvc": len(pth_files) > 0,
                "created_at": metadata.get("created_at"),
            })

        return models

    def delete_model(self, model_name: str) -> Tuple[bool, str]:
        """Delete a voice model."""
        model_dir = self.models_dir / model_name

        if not model_dir.exists():
            return False, f"Model not found: {model_name}"

        try:
            shutil.rmtree(model_dir)
            return True, f"Deleted model: {model_name}"
        except Exception as e:
            return False, str(e)

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get detailed information about a model."""
        model_dir = self.models_dir / model_name

        if not model_dir.exists():
            return None

        info = {
            "id": model_name,
            "path": str(model_dir),
            "files": [],
        }

        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                info["metadata"] = json.load(f)

        # Load voice characteristics
        char_path = model_dir / "voice_characteristics.json"
        if char_path.exists():
            with open(char_path) as f:
                info["voice_characteristics"] = json.load(f)

        # List files
        for file in model_dir.rglob("*"):
            if file.is_file():
                info["files"].append({
                    "name": str(file.relative_to(model_dir)),
                    "size": file.stat().st_size,
                })

        return info
