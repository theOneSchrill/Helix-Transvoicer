"""
Helix Transvoicer - Voice model manager.

Handles model storage, indexing, versioning, and fast refresh.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from helix_transvoicer.backend.utils.config import get_settings

logger = logging.getLogger("helix.model_manager")


@dataclass
class VoiceModel:
    """Voice model metadata and state."""

    id: str
    name: str
    version: str
    created_at: datetime
    updated_at: datetime
    path: Path
    total_samples: int
    total_duration: float
    emotion_coverage: Dict
    quality_score: float
    is_loaded: bool = False
    model_type: str = "helix"  # "helix" or "rvc"
    metadata: Dict = field(default_factory=dict)


class ModelManager:
    """
    Manages voice models in local storage.

    Features:
    - Model discovery and indexing
    - Metadata management
    - Version control
    - Fast model loading/unloading
    - Model export/import
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        self.settings = get_settings()
        self.models_dir = models_dir or self.settings.models_dir
        self.device = device or torch.device("cpu")

        self._models: Dict[str, VoiceModel] = {}
        self._loaded_models: Dict[str, Dict] = {}

    async def initialize(self) -> None:
        """Initialize model manager and scan for models."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # Clean up any incomplete models from interrupted training
        await self._cleanup_incomplete_models()
        await self.refresh()

    async def _cleanup_incomplete_models(self) -> None:
        """Remove incomplete model directories from interrupted training sessions."""
        if not self.models_dir.exists():
            return

        # Files required for Helix native models
        helix_required_files = ["metadata.json", "speaker_encoder.pt", "decoder.pt", "content_encoder.pt"]

        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # Skip RVC models (have .pth files)
            pth_files = list(model_dir.glob("*.pth"))
            if pth_files:
                logger.debug(f"Skipping RVC model: {model_dir.name}")
                continue

            # Skip if has metadata.json (could be partial but valid)
            if (model_dir / "metadata.json").exists():
                continue

            # Check if it's an incomplete Helix model (has some but not all files)
            has_any_helix_file = any((model_dir / f).exists() for f in helix_required_files[1:])

            if has_any_helix_file:
                missing_files = [f for f in helix_required_files if not (model_dir / f).exists()]
                logger.warning(
                    f"Found incomplete Helix model '{model_dir.name}' (missing: {', '.join(missing_files)}). "
                    f"Removing..."
                )
                try:
                    shutil.rmtree(model_dir)
                    logger.info(f"Removed incomplete model directory: {model_dir.name}")
                except Exception as e:
                    logger.error(f"Failed to remove incomplete model {model_dir.name}: {e}")

    async def refresh(self) -> None:
        """Scan and index all models in storage."""
        self._models.clear()

        if not self.models_dir.exists():
            return

        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                try:
                    model = await self._load_model_metadata(model_dir)
                    if model:
                        self._models[model.id] = model
                except Exception as e:
                    logger.error(f"Failed to load model {model_dir.name}: {e}")

        logger.info(f"Indexed {len(self._models)} voice models")

    async def _load_model_metadata(self, model_dir: Path) -> Optional[VoiceModel]:
        """Load model metadata from directory."""
        metadata_path = model_dir / "metadata.json"

        # Check for RVC model (.pth files)
        pth_files = list(model_dir.glob("*.pth"))
        index_files = list(model_dir.glob("*.index"))
        is_rvc_model = len(pth_files) > 0

        if metadata_path.exists():
            with open(metadata_path) as f:
                data = json.load(f)
        elif is_rvc_model:
            # Auto-generate metadata for RVC models
            now = datetime.now()
            pth_file = pth_files[0]
            data = {
                "model_id": model_dir.name,
                "version": "1.0.0",
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "total_samples": 0,
                "total_duration": 0.0,
                "emotion_coverage": {},
                "model_type": "rvc",
                "rvc_model_file": pth_file.name,
                "rvc_index_file": index_files[0].name if index_files else None,
            }
            # Save the auto-generated metadata
            try:
                with open(metadata_path, "w") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Created metadata for RVC model: {model_dir.name}")
            except Exception as e:
                logger.warning(f"Could not save metadata for {model_dir.name}: {e}")
        else:
            return None

        return VoiceModel(
            id=data.get("model_id", model_dir.name),
            name=data.get("model_id", model_dir.name),
            version=data.get("version", "1.0.0"),
            created_at=datetime.fromisoformat(
                data.get("created_at", datetime.now().isoformat())
            ),
            updated_at=datetime.fromisoformat(
                data.get("updated_at", datetime.now().isoformat())
            ),
            path=model_dir,
            total_samples=data.get("total_samples", 0),
            total_duration=data.get("total_duration", 0.0),
            emotion_coverage=data.get("emotion_coverage", {}),
            quality_score=data.get("quality_metrics", {}).get("overall_quality", 0.0),
            model_type=data.get("model_type", "rvc" if is_rvc_model else "helix"),
            metadata=data,
        )

    @property
    def models(self) -> Dict[str, VoiceModel]:
        """Get all indexed models."""
        return self._models

    def get_model(self, model_id: str) -> Optional[VoiceModel]:
        """Get a specific model by ID."""
        return self._models.get(model_id)

    def list_models(self) -> List[VoiceModel]:
        """List all models sorted by update time."""
        return sorted(
            self._models.values(),
            key=lambda m: m.updated_at,
            reverse=True,
        )

    async def load_model(self, model_id: str) -> Dict:
        """
        Load model weights into memory.

        Returns:
            Dictionary with loaded model components
        """
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        model = self._models.get(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")

        loaded = {}
        settings = get_settings()

        # Load architecture config to create models with correct dimensions
        arch_config = self._load_architecture_config(model.path)

        # Load speaker embedding
        embedding_path = model.path / "speaker_embedding.npy"
        if embedding_path.exists():
            embedding = np.load(str(embedding_path))
            loaded["speaker_embedding"] = torch.from_numpy(embedding).to(self.device)

        # Load encoder with correct architecture
        encoder_path = model.path / "speaker_encoder.pt"
        if encoder_path.exists():
            from helix_transvoicer.backend.models.encoder import SpeakerEncoder

            if arch_config and arch_config.get("low_memory_mode", False):
                se_cfg = arch_config.get("speaker_encoder", {})
                encoder = SpeakerEncoder(
                    input_dim=settings.n_mels,
                    hidden_dim=se_cfg.get("hidden_dim", 64),
                    embedding_dim=se_cfg.get("embedding_dim", 64),
                )
            else:
                encoder = SpeakerEncoder(input_dim=settings.n_mels)

            encoder.load_state_dict(
                torch.load(str(encoder_path), map_location=self.device)
            )
            encoder.to(self.device)
            encoder.eval()
            loaded["encoder"] = encoder

        # Load decoder with correct architecture
        decoder_path = model.path / "decoder.pt"
        if decoder_path.exists():
            from helix_transvoicer.backend.models.decoder import VoiceDecoder

            if arch_config and arch_config.get("low_memory_mode", False):
                dec_cfg = arch_config.get("decoder", {})
                decoder = VoiceDecoder(
                    content_dim=dec_cfg.get("content_dim", 64),
                    speaker_dim=dec_cfg.get("speaker_dim", 64),
                    hidden_dim=dec_cfg.get("hidden_dim", 128),
                    n_mels=settings.n_mels,
                    num_layers=dec_cfg.get("num_layers", 1),
                )
            else:
                decoder = VoiceDecoder(n_mels=settings.n_mels)

            decoder.load_state_dict(
                torch.load(str(decoder_path), map_location=self.device)
            )
            decoder.to(self.device)
            decoder.eval()
            loaded["decoder"] = decoder

        self._loaded_models[model_id] = loaded
        model.is_loaded = True

        logger.info(f"Loaded model: {model_id}")
        return loaded

    def _load_architecture_config(self, model_path: Path) -> Optional[Dict]:
        """Load model architecture configuration."""
        arch_path = model_path / "architecture.json"
        if arch_path.exists():
            with open(arch_path, "r") as f:
                return json.load(f)

        # Fallback: Try to infer from metadata or model weights
        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                if metadata.get("low_memory_mode", False):
                    # Return inferred low memory config
                    return {
                        "low_memory_mode": True,
                        "content_encoder": {"hidden_dim": 64, "output_dim": 64, "num_layers": 1},
                        "speaker_encoder": {"hidden_dim": 64, "embedding_dim": 64},
                        "decoder": {"content_dim": 64, "speaker_dim": 64, "hidden_dim": 128, "num_layers": 1},
                    }

        # Try to infer from model weights
        encoder_path = model_path / "speaker_encoder.pt"
        if encoder_path.exists():
            try:
                state_dict = torch.load(str(encoder_path), map_location="cpu")
                # Check embedding projection output size to detect low memory mode
                if "embedding_proj.0.weight" in state_dict:
                    proj_size = state_dict["embedding_proj.0.weight"].shape[0]
                    if proj_size == 64:
                        logger.info(f"Detected low memory mode model (embedding_dim=64)")
                        return {
                            "low_memory_mode": True,
                            "content_encoder": {"hidden_dim": 64, "output_dim": 64, "num_layers": 1},
                            "speaker_encoder": {"hidden_dim": 64, "embedding_dim": 64},
                            "decoder": {"content_dim": 64, "speaker_dim": 64, "hidden_dim": 128, "num_layers": 1},
                        }
            except Exception as e:
                logger.warning(f"Could not infer architecture from weights: {e}")

        return None

    async def unload_model(self, model_id: str) -> None:
        """Unload model from memory."""
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]

            model = self._models.get(model_id)
            if model:
                model.is_loaded = False

            # Clear GPU cache
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            logger.info(f"Unloaded model: {model_id}")

    async def create_model(
        self,
        model_id: str,
        metadata: Optional[Dict] = None,
    ) -> VoiceModel:
        """Create a new empty model directory."""
        model_dir = self.models_dir / model_id

        if model_dir.exists():
            raise ValueError(f"Model already exists: {model_id}")

        model_dir.mkdir(parents=True)

        # Create initial metadata
        now = datetime.now()
        meta = {
            "model_id": model_id,
            "version": "0.0.0",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "total_samples": 0,
            "total_duration": 0.0,
            "emotion_coverage": {},
            **(metadata or {}),
        }

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        model = VoiceModel(
            id=model_id,
            name=model_id,
            version="0.0.0",
            created_at=now,
            updated_at=now,
            path=model_dir,
            total_samples=0,
            total_duration=0.0,
            emotion_coverage={},
            quality_score=0.0,
            metadata=meta,
        )

        self._models[model_id] = model
        return model

    async def update_metadata(
        self,
        model_id: str,
        updates: Dict,
    ) -> VoiceModel:
        """Update model metadata."""
        model = self._models.get(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")

        # Load existing metadata
        metadata_path = model.path / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Update
        metadata.update(updates)
        metadata["updated_at"] = datetime.now().isoformat()

        # Save
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Refresh model object
        updated_model = await self._load_model_metadata(model.path)
        if updated_model:
            self._models[model_id] = updated_model
            return updated_model

        return model

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        model = self._models.get(model_id)
        if not model:
            return False

        # Unload if loaded
        await self.unload_model(model_id)

        # Delete directory
        shutil.rmtree(model.path)

        # Remove from index
        del self._models[model_id]

        logger.info(f"Deleted model: {model_id}")
        return True

    async def duplicate_model(
        self,
        source_id: str,
        new_id: str,
    ) -> VoiceModel:
        """Duplicate an existing model."""
        source = self._models.get(source_id)
        if not source:
            raise ValueError(f"Source model not found: {source_id}")

        new_dir = self.models_dir / new_id
        if new_dir.exists():
            raise ValueError(f"Model already exists: {new_id}")

        # Copy directory
        shutil.copytree(source.path, new_dir)

        # Update metadata
        metadata_path = new_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        now = datetime.now()
        metadata["model_id"] = new_id
        metadata["created_at"] = now.isoformat()
        metadata["updated_at"] = now.isoformat()
        metadata["version"] = "1.0.0"

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Index new model
        new_model = await self._load_model_metadata(new_dir)
        if new_model:
            self._models[new_id] = new_model
            return new_model

        raise RuntimeError("Failed to duplicate model")

    async def export_model(
        self,
        model_id: str,
        export_path: Path,
    ) -> Path:
        """Export model to a zip file."""
        model = self._models.get(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")

        export_path = Path(export_path)
        if export_path.suffix != ".zip":
            export_path = export_path.with_suffix(".zip")

        shutil.make_archive(
            str(export_path.with_suffix("")),
            "zip",
            model.path,
        )

        logger.info(f"Exported model {model_id} to {export_path}")
        return export_path

    async def import_model(
        self,
        import_path: Path,
        model_id: Optional[str] = None,
    ) -> VoiceModel:
        """Import model from a zip file."""
        import_path = Path(import_path)

        if not import_path.exists():
            raise FileNotFoundError(f"Import file not found: {import_path}")

        # Determine model ID
        if model_id is None:
            model_id = import_path.stem

        model_dir = self.models_dir / model_id
        if model_dir.exists():
            raise ValueError(f"Model already exists: {model_id}")

        # Extract
        shutil.unpack_archive(import_path, model_dir)

        # Update metadata with new ID
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

            metadata["model_id"] = model_id
            metadata["updated_at"] = datetime.now().isoformat()

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        # Index
        model = await self._load_model_metadata(model_dir)
        if model:
            self._models[model_id] = model
            logger.info(f"Imported model: {model_id}")
            return model

        raise RuntimeError("Failed to import model")

    def get_loaded_count(self) -> int:
        """Get number of loaded models."""
        return len(self._loaded_models)

    def get_total_count(self) -> int:
        """Get total number of indexed models."""
        return len(self._models)
