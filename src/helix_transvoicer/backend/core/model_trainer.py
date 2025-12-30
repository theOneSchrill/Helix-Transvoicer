"""
Helix Transvoicer - Voice model training pipeline.

Handles initial training and incremental updates.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from helix_transvoicer.backend.core.audio_processor import AudioProcessor, ProcessedAudio
from helix_transvoicer.backend.core.emotion_analyzer import EmotionAnalyzer
from helix_transvoicer.backend.models.encoder import ContentEncoder, SpeakerEncoder
from helix_transvoicer.backend.models.decoder import VoiceDecoder
from helix_transvoicer.backend.utils.audio import AudioUtils
from helix_transvoicer.backend.utils.config import get_settings

logger = logging.getLogger("helix.model_trainer")


@dataclass
class TrainingConfig:
    """Model training configuration."""

    model_name: str
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    warmup_epochs: int = 5
    checkpoint_interval: int = 10
    auto_denoise: bool = True
    augment_data: bool = True
    fine_tune_vocoder: bool = False


@dataclass
class TrainingSample:
    """A single training sample."""

    audio: np.ndarray
    sample_rate: int
    duration: float
    emotion: str
    emotion_confidence: float
    quality_score: float
    source_path: str


@dataclass
class TrainingProgress:
    """Training progress information."""

    epoch: int
    total_epochs: int
    loss: float
    learning_rate: float
    elapsed_time: float
    eta_seconds: float
    stage: str


@dataclass
class TrainingResult:
    """Result of model training."""

    model_id: str
    version: str
    model_path: Path
    total_samples: int
    total_duration: float
    final_loss: float
    epochs_trained: int
    emotion_coverage: Dict
    training_time: float
    success: bool
    error: Optional[str] = None


class VoiceDataset(Dataset):
    """Dataset for voice model training with automatic chunking for long audio."""

    # Maximum chunk duration in seconds (5 seconds fits comfortably in memory)
    MAX_CHUNK_SECONDS = 5.0

    def __init__(
        self,
        samples: List[TrainingSample],
        augment: bool = True,
    ):
        self.augment = augment
        self.settings = get_settings()

        # Chunk long audio into manageable segments
        self.chunks = []
        for sample in samples:
            chunks = self._chunk_audio(sample)
            self.chunks.extend(chunks)

        logger.info(f"Created {len(self.chunks)} training chunks from {len(samples)} samples")

    def _chunk_audio(self, sample: TrainingSample) -> List[Dict]:
        """Split long audio into chunks."""
        max_samples = int(self.MAX_CHUNK_SECONDS * sample.sample_rate)
        audio = sample.audio

        if len(audio) <= max_samples:
            # Short enough, use as-is
            return [{
                "audio": audio,
                "sample_rate": sample.sample_rate,
                "emotion": sample.emotion,
            }]

        # Split into overlapping chunks (50% overlap for continuity)
        chunks = []
        hop = max_samples // 2
        for start in range(0, len(audio) - max_samples + 1, hop):
            chunk_audio = audio[start:start + max_samples]
            chunks.append({
                "audio": chunk_audio,
                "sample_rate": sample.sample_rate,
                "emotion": sample.emotion,
            })

        # Add final chunk if there's remaining audio
        if len(audio) % hop != 0:
            chunk_audio = audio[-max_samples:]
            chunks.append({
                "audio": chunk_audio,
                "sample_rate": sample.sample_rate,
                "emotion": sample.emotion,
            })

        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        audio = chunk["audio"].copy()
        sample_rate = chunk["sample_rate"]

        # Data augmentation
        if self.augment:
            audio = self._augment(audio, sample_rate)

        # Compute mel spectrogram
        mel = AudioUtils.compute_mel_spectrogram(
            audio,
            sample_rate=sample_rate,
            n_fft=self.settings.n_fft,
            hop_length=self.settings.hop_length,
            n_mels=self.settings.n_mels,
        )

        return {
            "mel": torch.from_numpy(mel).float(),
            "audio": torch.from_numpy(audio).float(),
            "emotion": chunk["emotion"],
        }

    def _augment(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply data augmentation."""
        # Random volume adjustment
        if np.random.random() < 0.5:
            gain = np.random.uniform(0.8, 1.2)
            audio = audio * gain
            audio = np.clip(audio, -1.0, 1.0)

        # Random noise injection
        if np.random.random() < 0.3:
            noise = np.random.randn(len(audio)) * 0.002
            audio = audio + noise
            audio = np.clip(audio, -1.0, 1.0)

        return audio


class ModelTrainer:
    """
    Voice model training engine.

    Handles:
    - Dataset preparation from audio samples
    - Speaker encoder training
    - Voice decoder training
    - Incremental model updates
    - Checkpoint management
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        models_dir: Optional[Path] = None,
    ):
        self.settings = get_settings()
        self.device = device or torch.device("cpu")
        self.models_dir = models_dir or self.settings.models_dir

        self.audio_processor = AudioProcessor()
        self.emotion_analyzer = EmotionAnalyzer(device=self.device)

        self._current_training: Optional[Dict] = None

    def prepare_samples(
        self,
        audio_paths: List[Union[str, Path]],
        progress_callback: Optional[Callable[[str, float], None]] = None,
        remove_silence: bool = True,
    ) -> List[TrainingSample]:
        """
        Prepare training samples from audio files.

        Processes audio, removes silence, and analyzes emotions.

        Args:
            audio_paths: List of paths to audio files
            progress_callback: Callback for progress updates
            remove_silence: Whether to remove silent segments (speeds up training)
        """
        samples = []
        total = len(audio_paths)
        total_original_duration = 0
        total_processed_duration = 0

        for i, path in enumerate(audio_paths):
            try:
                if progress_callback:
                    progress_callback(f"Processing {Path(path).name}", i / total)

                # Process audio
                processed = self.audio_processor.process(path)
                audio = processed.audio
                original_duration = processed.duration
                total_original_duration += original_duration

                # Remove silence to speed up training
                if remove_silence:
                    audio = AudioUtils.remove_silence(
                        audio,
                        sample_rate=processed.sample_rate,
                        top_db=30.0,  # Threshold for silence detection
                        min_silence_duration=0.1,  # Remove silences > 100ms
                        keep_short_silence=0.05,  # Keep 50ms gaps between segments
                    )

                processed_duration = len(audio) / processed.sample_rate
                total_processed_duration += processed_duration

                # Analyze emotion
                emotions = self.emotion_analyzer.analyze(
                    audio,
                    processed.sample_rate,
                )
                primary_emotion = max(emotions, key=emotions.get)
                emotion_confidence = emotions[primary_emotion]

                sample = TrainingSample(
                    audio=audio,
                    sample_rate=processed.sample_rate,
                    duration=processed_duration,
                    emotion=primary_emotion,
                    emotion_confidence=emotion_confidence,
                    quality_score=processed.quality_score,
                    source_path=str(path),
                )
                samples.append(sample)

            except Exception as e:
                logger.error(f"Failed to prepare sample {path}: {e}")
                continue

        if progress_callback:
            progress_callback("Preparation complete", 1.0)

        # Log total silence removed
        if remove_silence and total_original_duration > 0:
            removed_pct = (1 - total_processed_duration / total_original_duration) * 100
            logger.info(f"Total audio: {total_original_duration:.1f}s -> {total_processed_duration:.1f}s "
                        f"({removed_pct:.0f}% silence removed)")

        return samples

    def train(
        self,
        samples: List[TrainingSample],
        config: TrainingConfig,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None,
    ) -> TrainingResult:
        """
        Train a new voice model from samples.

        Args:
            samples: Prepared training samples
            config: Training configuration
            progress_callback: Progress update callback

        Returns:
            TrainingResult with model information
        """
        import time

        start_time = time.time()

        if len(samples) < 1:
            return TrainingResult(
                model_id=config.model_name,
                version="0.0.0",
                model_path=Path(""),
                total_samples=len(samples),
                total_duration=0,
                final_loss=0,
                epochs_trained=0,
                emotion_coverage={},
                training_time=0,
                success=False,
                error="At least 1 training sample required",
            )

        logger.info(f"Starting training for model: {config.model_name}")
        logger.info(f"Samples: {len(samples)}, Epochs: {config.epochs}")

        # Create model directory
        model_dir = self.models_dir / config.model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset and dataloader
        dataset = VoiceDataset(samples, augment=config.augment_data)
        dataloader = DataLoader(
            dataset,
            batch_size=min(config.batch_size, len(samples)),
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

        # Initialize models
        content_encoder = ContentEncoder().to(self.device)
        speaker_encoder = SpeakerEncoder(input_dim=self.settings.n_mels).to(self.device)
        decoder = VoiceDecoder(n_mels=self.settings.n_mels).to(self.device)

        # Initialize optimizer
        params = (
            list(content_encoder.parameters()) +
            list(speaker_encoder.parameters()) +
            list(decoder.parameters())
        )
        optimizer = optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01,
        )

        # Loss function
        criterion = nn.MSELoss()

        # Training loop
        final_loss = 0.0
        self._current_training = {"active": True}

        try:
            for epoch in range(config.epochs):
                if not self._current_training.get("active", True):
                    logger.info("Training cancelled")
                    break

                epoch_loss = 0.0
                content_encoder.train()
                speaker_encoder.train()
                decoder.train()

                for batch in dataloader:
                    mel = batch["mel"].to(self.device)  # [batch, n_mels, time]
                    audio = batch["audio"].to(self.device)  # [batch, samples]

                    optimizer.zero_grad()

                    # Extract content features from audio
                    content_features = content_encoder(audio)  # [batch, time, 256]

                    # Extract speaker embedding from mel
                    speaker_embedding = speaker_encoder(mel)  # [batch, 256]

                    # Decode to mel spectrogram
                    reconstructed = decoder(content_features, speaker_embedding)  # [batch, time, n_mels]

                    # Target mel needs to match reconstructed shape [batch, time, n_mels]
                    target_mel = mel.transpose(1, 2)  # [batch, time, n_mels]

                    # Match sequence lengths (content encoder downsamples audio)
                    min_len = min(reconstructed.shape[1], target_mel.shape[1])
                    reconstructed = reconstructed[:, :min_len, :]
                    target_mel = target_mel[:, :min_len, :]

                    # Compute loss
                    loss = criterion(reconstructed, target_mel)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                epoch_loss /= len(dataloader)
                final_loss = epoch_loss
                scheduler.step()

                # Progress callback
                if progress_callback:
                    elapsed = time.time() - start_time
                    eta = elapsed / (epoch + 1) * (config.epochs - epoch - 1)

                    progress_callback(
                        TrainingProgress(
                            epoch=epoch + 1,
                            total_epochs=config.epochs,
                            loss=epoch_loss,
                            learning_rate=scheduler.get_last_lr()[0],
                            elapsed_time=elapsed,
                            eta_seconds=eta,
                            stage="training",
                        )
                    )

                # Checkpoint
                if (epoch + 1) % config.checkpoint_interval == 0:
                    self._save_checkpoint(
                        model_dir,
                        content_encoder,
                        speaker_encoder,
                        decoder,
                        epoch + 1,
                    )

                # Console progress with visual bar
                progress_pct = (epoch + 1) / config.epochs * 100
                bar_len = 30
                filled = int(bar_len * (epoch + 1) / config.epochs)
                bar = "█" * filled + "░" * (bar_len - filled)
                elapsed = time.time() - start_time
                eta = elapsed / (epoch + 1) * (config.epochs - epoch - 1)
                logger.info(
                    f"Training [{bar}] {progress_pct:5.1f}% | "
                    f"Epoch {epoch + 1}/{config.epochs} | "
                    f"Loss: {epoch_loss:.6f} | "
                    f"ETA: {eta:.0f}s"
                )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TrainingResult(
                model_id=config.model_name,
                version="0.0.0",
                model_path=model_dir,
                total_samples=len(samples),
                total_duration=sum(s.duration for s in samples),
                final_loss=final_loss,
                epochs_trained=epoch + 1 if "epoch" in dir() else 0,
                emotion_coverage={},
                training_time=time.time() - start_time,
                success=False,
                error=str(e),
            )

        finally:
            self._current_training = None

        # Compute speaker embedding from all samples
        speaker_embedding = self._compute_speaker_embedding(
            samples,
            speaker_encoder,
        )

        # Save model
        self._save_model(
            model_dir,
            content_encoder,
            speaker_encoder,
            decoder,
            speaker_embedding,
            config,
            samples,
        )

        # Compute emotion coverage
        emotion_coverage = self._compute_emotion_coverage(samples)

        training_time = time.time() - start_time

        logger.info(f"Training complete in {training_time:.1f}s, final loss: {final_loss:.6f}")

        return TrainingResult(
            model_id=config.model_name,
            version="1.0.0",
            model_path=model_dir,
            total_samples=len(samples),
            total_duration=sum(s.duration for s in samples),
            final_loss=final_loss,
            epochs_trained=config.epochs,
            emotion_coverage=emotion_coverage,
            training_time=training_time,
            success=True,
        )

    def update_model(
        self,
        model_id: str,
        new_samples: List[TrainingSample],
        epochs: int = 20,
        learning_rate: float = 0.00005,
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None,
    ) -> TrainingResult:
        """
        Incrementally update an existing model with new samples.

        Uses a lower learning rate and fewer epochs to fine-tune.
        """
        model_dir = self.models_dir / model_id

        if not model_dir.exists():
            raise ValueError(f"Model not found: {model_id}")

        # Load existing model
        speaker_encoder, decoder = self._load_model(model_dir)

        # Create config for incremental training
        config = TrainingConfig(
            model_name=model_id,
            epochs=epochs,
            learning_rate=learning_rate,
            augment_data=False,  # Less augmentation for fine-tuning
        )

        # Increment version
        metadata = self._load_metadata(model_dir)
        old_version = metadata.get("version", "1.0.0")
        new_version = self._increment_version(old_version)

        # Archive current version
        self._archive_version(model_dir, old_version)

        # Fine-tune with new samples
        result = self.train(new_samples, config, progress_callback)
        result.version = new_version

        return result

    def cancel_training(self):
        """Cancel current training."""
        if self._current_training:
            self._current_training["active"] = False

    def _compute_speaker_embedding(
        self,
        samples: List[TrainingSample],
        encoder: SpeakerEncoder,
    ) -> np.ndarray:
        """Compute average speaker embedding from samples."""
        encoder.eval()
        embeddings = []

        with torch.no_grad():
            for sample in samples:
                mel = AudioUtils.compute_mel_spectrogram(
                    sample.audio,
                    sample_rate=sample.sample_rate,
                )
                mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).to(self.device)
                embedding = encoder(mel_tensor)
                embeddings.append(embedding.cpu().numpy())

        return np.mean(embeddings, axis=0)

    def _compute_emotion_coverage(
        self,
        samples: List[TrainingSample],
    ) -> Dict:
        """Compute emotion coverage statistics."""
        emotions = {}

        for sample in samples:
            emotion = sample.emotion
            if emotion not in emotions:
                emotions[emotion] = {
                    "sample_count": 0,
                    "total_confidence": 0,
                    "total_duration": 0,
                }

            emotions[emotion]["sample_count"] += 1
            emotions[emotion]["total_confidence"] += sample.emotion_confidence
            emotions[emotion]["total_duration"] += sample.duration

        # Normalize
        for emotion in emotions:
            data = emotions[emotion]
            data["confidence"] = data["total_confidence"] / data["sample_count"]
            data["coverage"] = min(data["sample_count"] / 5, 1.0)  # 5+ samples = 100%

        return emotions

    def _save_model(
        self,
        model_dir: Path,
        content_encoder: ContentEncoder,
        speaker_encoder: SpeakerEncoder,
        decoder: VoiceDecoder,
        speaker_embedding: np.ndarray,
        config: TrainingConfig,
        samples: List[TrainingSample],
    ):
        """Save model to disk."""
        # Save model weights
        torch.save(content_encoder.state_dict(), model_dir / "content_encoder.pt")
        torch.save(speaker_encoder.state_dict(), model_dir / "speaker_encoder.pt")
        torch.save(decoder.state_dict(), model_dir / "decoder.pt")

        # Save speaker embedding
        np.save(model_dir / "speaker_embedding.npy", speaker_embedding)

        # Save config
        with open(model_dir / "config.json", "w") as f:
            json.dump(
                {
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                },
                f,
                indent=2,
            )

        # Save metadata
        metadata = {
            "model_id": config.model_name,
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "total_duration": sum(s.duration for s in samples),
            "emotion_coverage": self._compute_emotion_coverage(samples),
        }

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_checkpoint(
        self,
        model_dir: Path,
        content_encoder: ContentEncoder,
        speaker_encoder: SpeakerEncoder,
        decoder: VoiceDecoder,
        epoch: int,
    ):
        """Save training checkpoint."""
        checkpoint_dir = model_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        torch.save(
            {
                "epoch": epoch,
                "content_encoder": content_encoder.state_dict(),
                "speaker_encoder": speaker_encoder.state_dict(),
                "decoder": decoder.state_dict(),
            },
            checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
        )

    def _load_model(
        self,
        model_dir: Path,
    ) -> Tuple[ContentEncoder, SpeakerEncoder, VoiceDecoder]:
        """Load model from disk."""
        content_encoder = ContentEncoder().to(self.device)
        speaker_encoder = SpeakerEncoder(input_dim=self.settings.n_mels).to(self.device)
        decoder = VoiceDecoder(n_mels=self.settings.n_mels).to(self.device)

        content_encoder.load_state_dict(
            torch.load(model_dir / "content_encoder.pt", map_location=self.device)
        )
        speaker_encoder.load_state_dict(
            torch.load(model_dir / "speaker_encoder.pt", map_location=self.device)
        )
        decoder.load_state_dict(
            torch.load(model_dir / "decoder.pt", map_location=self.device)
        )

        return content_encoder, speaker_encoder, decoder

    def _load_metadata(self, model_dir: Path) -> Dict:
        """Load model metadata."""
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}

    def _increment_version(self, version: str) -> str:
        """Increment version string."""
        parts = version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)

    def _archive_version(self, model_dir: Path, version: str):
        """Archive current version before update."""
        versions_dir = model_dir / "versions"
        versions_dir.mkdir(exist_ok=True)

        archive_dir = versions_dir / version
        if not archive_dir.exists():
            archive_dir.mkdir()

            for file in ["content_encoder.pt", "speaker_encoder.pt", "decoder.pt", "metadata.json"]:
                src = model_dir / file
                if src.exists():
                    shutil.copy(src, archive_dir / file)
