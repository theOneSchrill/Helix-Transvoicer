"""
RVC Model Training.

Trains RVC voice conversion models from audio samples.
"""

import gc
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from helix_transvoicer.backend.rvc.models import RVCModelManager, check_rvc_ready
from helix_transvoicer.backend.utils.config import get_settings

logger = logging.getLogger("helix.rvc.training")


@dataclass
class RVCTrainingConfig:
    """RVC training configuration."""

    model_name: str
    sample_rate: int = 40000  # 40k or 48k
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    save_every_epoch: int = 10
    total_epochs: int = 100

    # Training options
    use_pretrained: bool = True
    version: str = "v2"
    f0_method: str = "rmvpe"

    # Model architecture
    hidden_channels: int = 192
    inter_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6


@dataclass
class RVCTrainingResult:
    """Result of RVC training."""

    model_id: str
    model_path: Path
    epochs_trained: int
    final_loss: float
    total_samples: int
    total_duration: float
    training_time: float
    success: bool
    error: Optional[str] = None


class RVCDataset(Dataset):
    """Dataset for RVC training."""

    def __init__(
        self,
        audio_dir: Path,
        features_dir: Path,
        sample_rate: int = 40000,
        segment_size: int = 16384,
    ):
        self.audio_dir = audio_dir
        self.features_dir = features_dir
        self.sample_rate = sample_rate
        self.segment_size = segment_size

        # Find all processed files
        self.files = sorted(list(features_dir.glob("*.npy")))

        if not self.files:
            raise ValueError(f"No processed files found in {features_dir}")

        logger.info(f"Found {len(self.files)} training samples")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        feat_path = self.files[idx]

        # Load features
        data = np.load(str(feat_path), allow_pickle=True).item()

        audio = data["audio"]
        hubert = data["hubert"]
        f0 = data["f0"]

        # Random segment
        if len(audio) > self.segment_size:
            start = np.random.randint(0, len(audio) - self.segment_size)
            audio = audio[start : start + self.segment_size]

            # Corresponding feature indices
            feat_start = start // 320  # HuBERT hop size
            feat_len = self.segment_size // 320
            hubert = hubert[feat_start : feat_start + feat_len]
            f0 = f0[feat_start : feat_start + feat_len]

        return {
            "audio": torch.from_numpy(audio).float(),
            "hubert": torch.from_numpy(hubert).float(),
            "f0": torch.from_numpy(f0).float(),
        }


class RVCTrainer:
    """
    RVC Model Trainer.

    Trains voice conversion models from audio samples.
    """

    def __init__(
        self,
        config: RVCTrainingConfig,
        output_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.config = config
        self.settings = get_settings()
        self.model_manager = RVCModelManager()

        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.settings.models_dir / config.model_name

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"RVC Trainer using device: {self.device}")

        # Models (lazy loaded)
        self._hubert_model = None
        self._rmvpe_model = None

    def _ensure_pretrained(self):
        """Ensure pre-trained models are available."""
        ready, missing = check_rvc_ready(for_training=True)
        if not ready:
            raise RuntimeError(
                f"Missing pre-trained models: {missing}. "
                "Please download them first using download_pretrained_models(for_training=True)"
            )

    def _load_hubert(self):
        """Load HuBERT model."""
        if self._hubert_model is not None:
            return self._hubert_model

        hubert_path = self.model_manager.get_model_path("hubert_base")

        try:
            from fairseq import checkpoint_utils

            models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                [str(hubert_path)],
                suffix="",
            )
            self._hubert_model = models[0].to(self.device)
            self._hubert_model.eval()
            return self._hubert_model

        except ImportError:
            raise RuntimeError(
                "fairseq is required for training. Install with: "
                "pip install fairseq"
            )

    def _load_rmvpe(self):
        """Load RMVPE model."""
        if self._rmvpe_model is not None:
            return self._rmvpe_model

        rmvpe_path = self.model_manager.get_model_path("rmvpe")
        checkpoint = torch.load(str(rmvpe_path), map_location=self.device)

        from helix_transvoicer.backend.rvc.rmvpe import RMVPE

        self._rmvpe_model = RMVPE(checkpoint)
        self._rmvpe_model.to(self.device)
        self._rmvpe_model.eval()

        return self._rmvpe_model

    def preprocess(
        self,
        audio_files: List[Path],
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Path:
        """
        Preprocess audio files for training.

        Extracts HuBERT features and pitch for each file.

        Args:
            audio_files: List of audio file paths
            progress_callback: Optional progress callback

        Returns:
            Path to features directory
        """
        def update_progress(msg: str, prog: float):
            if progress_callback:
                progress_callback(msg, prog)

        self._ensure_pretrained()

        features_dir = self.output_dir / "features"
        features_dir.mkdir(exist_ok=True)

        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        update_progress("Loading models...", 0.0)

        hubert = self._load_hubert()
        rmvpe = self._load_rmvpe()

        total = len(audio_files)

        for i, audio_path in enumerate(audio_files):
            try:
                update_progress(f"Processing {audio_path.name}...", (i / total) * 0.9)

                # Load audio
                audio, sr = librosa.load(str(audio_path), sr=16000)

                # Copy sample
                sample_path = samples_dir / audio_path.name
                shutil.copy(audio_path, sample_path)

                # Extract HuBERT features
                with torch.no_grad():
                    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
                    hubert_feats = hubert.extract_features(audio_tensor)[0]
                    hubert_feats = hubert_feats.squeeze(0).cpu().numpy()

                # Extract pitch
                f0 = rmvpe.infer_from_audio(audio, 16000)

                # Save features
                feat_data = {
                    "audio": audio,
                    "hubert": hubert_feats,
                    "f0": f0,
                    "sr": sr,
                }
                np.save(features_dir / f"{audio_path.stem}.npy", feat_data)

            except Exception as e:
                logger.warning(f"Failed to process {audio_path}: {e}")

        update_progress("Preprocessing complete", 1.0)

        return features_dir

    def train(
        self,
        audio_files: Optional[List[Path]] = None,
        features_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> RVCTrainingResult:
        """
        Train an RVC model.

        Args:
            audio_files: List of audio files to train on
            features_dir: Pre-processed features directory (if already done)
            progress_callback: Optional progress callback

        Returns:
            RVCTrainingResult
        """
        start_time = time.time()

        def update_progress(msg: str, prog: float):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg, prog)

        try:
            self._ensure_pretrained()

            # Preprocess if needed
            if features_dir is None:
                if audio_files is None:
                    raise ValueError("Either audio_files or features_dir must be provided")
                features_dir = self.preprocess(audio_files, progress_callback)

            update_progress("Initializing training...", 0.1)

            # Create dataset
            dataset = RVCDataset(
                self.output_dir / "samples",
                features_dir,
                sample_rate=self.config.sample_rate,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            # Initialize model
            from helix_transvoicer.backend.rvc.synthesizer import SynthesizerTrnMs768NSFsid

            model = SynthesizerTrnMs768NSFsid(
                spec_channels=1025,
                inter_channels=self.config.inter_channels,
                hidden_channels=self.config.hidden_channels,
                filter_channels=self.config.filter_channels,
                n_heads=self.config.n_heads,
                n_layers=self.config.n_layers,
                kernel_size=3,
                p_dropout=0,
                resblock="1",
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsample_rates=[10, 10, 2, 2],
                upsample_initial_channel=512,
                upsample_kernel_sizes=[16, 16, 4, 4],
                spk_embed_dim=109,
                gin_channels=256,
                sr=self.config.sample_rate,
            ).to(self.device)

            # Load pretrained weights if available
            if self.config.use_pretrained:
                pretrained_g = self.model_manager.get_model_path("pretrained_v2_G40k")
                if pretrained_g and pretrained_g.exists():
                    try:
                        checkpoint = torch.load(str(pretrained_g), map_location=self.device)
                        model.load_state_dict(checkpoint, strict=False)
                        logger.info("Loaded pretrained generator weights")
                    except Exception as e:
                        logger.warning(f"Could not load pretrained weights: {e}")

            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=(0.8, 0.99),
                weight_decay=0.01,
            )

            # Training loop
            model.train()
            total_loss = 0
            steps = 0

            for epoch in range(self.config.epochs):
                epoch_loss = 0

                for batch in dataloader:
                    audio = batch["audio"].to(self.device)
                    hubert = batch["hubert"].to(self.device)
                    f0 = batch["f0"].to(self.device)

                    optimizer.zero_grad()

                    # Forward pass
                    try:
                        output = model.infer(hubert, f0)

                        # Simple reconstruction loss
                        min_len = min(output.size(-1), audio.size(-1))
                        loss = F.l1_loss(output[..., :min_len], audio[..., :min_len].unsqueeze(1))

                        # Backward pass
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        steps += 1

                    except Exception as e:
                        logger.warning(f"Training step failed: {e}")
                        continue

                avg_loss = epoch_loss / max(len(dataloader), 1)
                total_loss = avg_loss

                # Progress update
                progress = 0.1 + (epoch / self.config.epochs) * 0.8
                update_progress(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {avg_loss:.4f}", progress)

                # Save checkpoint
                if (epoch + 1) % self.config.save_every_epoch == 0:
                    self._save_checkpoint(model, epoch + 1)

            # Final save
            update_progress("Saving model...", 0.95)
            model_path = self._save_model(model)

            # Build index
            update_progress("Building search index...", 0.98)
            self._build_index(features_dir)

            # Save metadata
            self._save_metadata(len(dataset), sum(librosa.get_duration(path=str(f)) for f in (self.output_dir / "samples").glob("*")))

            training_time = time.time() - start_time
            update_progress("Training complete!", 1.0)

            return RVCTrainingResult(
                model_id=self.config.model_name,
                model_path=model_path,
                epochs_trained=self.config.epochs,
                final_loss=total_loss,
                total_samples=len(dataset),
                total_duration=0,  # Could calculate from samples
                training_time=training_time,
                success=True,
            )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()

            return RVCTrainingResult(
                model_id=self.config.model_name,
                model_path=self.output_dir,
                epochs_trained=0,
                final_loss=0,
                total_samples=0,
                total_duration=0,
                training_time=time.time() - start_time,
                success=False,
                error=str(e),
            )

    def _save_checkpoint(self, model: nn.Module, epoch: int):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        path = checkpoint_dir / f"checkpoint_{epoch}.pth"
        torch.save(model.state_dict(), path)
        logger.info(f"Saved checkpoint: {path}")

    def _save_model(self, model: nn.Module) -> Path:
        """Save final model."""
        model_path = self.output_dir / f"{self.config.model_name}.pth"

        save_dict = {
            "weight": model.state_dict(),
            "config": {
                "spec_channels": 1025,
                "inter_channels": self.config.inter_channels,
                "hidden_channels": self.config.hidden_channels,
                "filter_channels": self.config.filter_channels,
                "n_heads": self.config.n_heads,
                "n_layers": self.config.n_layers,
                "kernel_size": 3,
                "p_dropout": 0,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [10, 10, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16, 16, 4, 4],
                "spk_embed_dim": 109,
                "gin_channels": 256,
                "sr": self.config.sample_rate,
            },
            "version": self.config.version,
        }

        torch.save(save_dict, model_path)
        logger.info(f"Saved model: {model_path}")

        return model_path

    def _build_index(self, features_dir: Path):
        """Build FAISS index for feature retrieval."""
        try:
            import faiss

            # Collect all HuBERT features
            all_features = []

            for feat_path in features_dir.glob("*.npy"):
                data = np.load(str(feat_path), allow_pickle=True).item()
                hubert = data["hubert"]
                all_features.append(hubert)

            if not all_features:
                logger.warning("No features to build index from")
                return

            # Concatenate features
            features = np.vstack(all_features).astype(np.float32)

            # Build IVF index
            dim = features.shape[1]
            n_clusters = min(int(features.shape[0] ** 0.5), 256)

            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_clusters)

            # Train and add
            index.train(features)
            index.add(features)

            # Save index
            index_path = self.output_dir / f"{self.config.model_name}.index"
            faiss.write_index(index, str(index_path))

            logger.info(f"Built index with {features.shape[0]} vectors: {index_path}")

        except ImportError:
            logger.warning("faiss not available, skipping index creation")
        except Exception as e:
            logger.warning(f"Failed to build index: {e}")

    def _save_metadata(self, sample_count: int, total_duration: float):
        """Save training metadata."""
        metadata = {
            "model_id": self.config.model_name,
            "type": "rvc",
            "version": self.config.version,
            "sample_rate": self.config.sample_rate,
            "created_at": datetime.now().isoformat(),
            "epochs": self.config.epochs,
            "sample_count": sample_count,
            "total_duration": total_duration,
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
