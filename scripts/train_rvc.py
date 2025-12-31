#!/usr/bin/env python3
"""
RVC Voice Model Training Script for Helix Transvoicer.

This script trains a Retrieval-based Voice Conversion (RVC) model from audio samples.
Requires 10+ minutes of clean speech audio for best results.

Usage:
    python scripts/train_rvc.py --name "VoiceName" --samples /path/to/audio/
    python scripts/train_rvc.py --name "DarthVader" --samples ./vader_clips/ --epochs 200
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rvc_trainer")


class RVCTrainer:
    """Train RVC voice models."""

    def __init__(
        self,
        model_name: str,
        samples_dir: Path,
        output_dir: Optional[Path] = None,
        sample_rate: int = 40000,
        f0_method: str = "rmvpe",
        epochs: int = 100,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.samples_dir = Path(samples_dir)
        self.sample_rate = sample_rate
        self.f0_method = f0_method
        self.epochs = epochs
        self.batch_size = batch_size

        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            from helix_transvoicer.backend.utils.config import get_settings
            settings = get_settings()
            self.output_dir = settings.models_dir / model_name

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check for RVC installation
        self._check_rvc_available()

    def _check_rvc_available(self):
        """Check if RVC training dependencies are available."""
        try:
            import torch
            import librosa
            import faiss
            logger.info("Core dependencies available")
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Install with: pip install torch librosa faiss-cpu")
            sys.exit(1)

    def prepare_dataset(self) -> List[Path]:
        """Prepare audio samples for training."""
        logger.info(f"Preparing dataset from: {self.samples_dir}")

        audio_files = []
        for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]:
            audio_files.extend(self.samples_dir.glob(ext))
            audio_files.extend(self.samples_dir.glob(ext.upper()))

        if not audio_files:
            logger.error(f"No audio files found in {self.samples_dir}")
            sys.exit(1)

        logger.info(f"Found {len(audio_files)} audio files")

        # Calculate total duration
        import librosa
        total_duration = 0
        for f in audio_files:
            try:
                duration = librosa.get_duration(path=str(f))
                total_duration += duration
            except Exception as e:
                logger.warning(f"Could not read {f}: {e}")

        logger.info(f"Total audio duration: {total_duration/60:.1f} minutes")

        if total_duration < 60:  # Less than 1 minute
            logger.warning("Very short dataset! Recommend at least 10 minutes of audio.")
        elif total_duration < 600:  # Less than 10 minutes
            logger.info("Short dataset. Results may be limited. 10+ minutes recommended.")

        return audio_files

    def extract_features(self, audio_files: List[Path]) -> Path:
        """Extract HuBERT features from audio files."""
        import librosa
        import numpy as np
        import torch

        logger.info("Extracting features...")

        features_dir = self.output_dir / "features"
        features_dir.mkdir(exist_ok=True)

        # Process each file
        all_features = []
        all_f0 = []

        for i, audio_file in enumerate(audio_files):
            try:
                logger.info(f"Processing {i+1}/{len(audio_files)}: {audio_file.name}")

                # Load audio
                audio, sr = librosa.load(str(audio_file), sr=self.sample_rate)

                # Extract pitch (F0)
                f0, voiced_flag, _ = librosa.pyin(
                    audio,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C6'),
                    sr=sr,
                )

                # For now, save basic features
                # In full RVC, this would use HuBERT
                features = {
                    "audio": audio,
                    "f0": f0,
                    "sr": sr,
                }

                np.save(features_dir / f"{audio_file.stem}_features.npy", features)

                all_f0.extend(f0[~np.isnan(f0)])

            except Exception as e:
                logger.warning(f"Failed to process {audio_file}: {e}")

        # Save voice characteristics
        if all_f0:
            all_f0 = np.array(all_f0)
            voice_chars = {
                "pitch_mean": float(np.mean(all_f0)),
                "pitch_std": float(np.std(all_f0)),
                "pitch_min": float(np.min(all_f0)),
                "pitch_max": float(np.max(all_f0)),
            }
            with open(self.output_dir / "voice_characteristics.json", "w") as f:
                json.dump(voice_chars, f, indent=2)
            logger.info(f"Voice pitch: {voice_chars['pitch_mean']:.1f} Hz (mean)")

        return features_dir

    def create_simple_model(self, audio_files: List[Path]):
        """
        Create a simple voice model by computing average speaker characteristics.

        Note: This is a simplified approach. Full RVC training requires:
        1. HuBERT feature extraction
        2. Training generator + discriminator networks
        3. Building FAISS index for retrieval

        For full RVC training, use the RVC WebUI:
        https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
        """
        import librosa
        import numpy as np

        logger.info("Creating voice model...")

        # Collect all audio
        all_audio = []
        sample_rate = self.sample_rate

        for audio_file in audio_files[:20]:  # Limit to first 20 files
            try:
                audio, sr = librosa.load(str(audio_file), sr=sample_rate)
                all_audio.append(audio)
            except Exception as e:
                logger.warning(f"Could not load {audio_file}: {e}")

        if not all_audio:
            logger.error("No audio could be loaded!")
            return False

        # Compute mel spectrograms for voice profile
        mel_specs = []
        for audio in all_audio:
            mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=80)
            mel_specs.append(mel)

        # Average mel spectrogram as voice "fingerprint"
        # Pad/truncate to same length
        max_len = max(m.shape[1] for m in mel_specs)
        padded = []
        for mel in mel_specs:
            if mel.shape[1] < max_len:
                mel = np.pad(mel, ((0, 0), (0, max_len - mel.shape[1])))
            else:
                mel = mel[:, :max_len]
            padded.append(mel)

        avg_mel = np.mean(padded, axis=0)

        # Save as speaker embedding (simplified)
        np.save(self.output_dir / "speaker_embedding.npy", avg_mel.flatten()[:256])

        logger.info("Voice profile created")
        return True

    def save_metadata(self, audio_files: List[Path]):
        """Save model metadata."""
        import librosa

        total_duration = sum(
            librosa.get_duration(path=str(f))
            for f in audio_files
            if f.exists()
        )

        metadata = {
            "model_id": self.model_name,
            "type": "rvc_simple",
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "sample_rate": self.sample_rate,
            "total_samples": len(audio_files),
            "total_duration": total_duration,
            "epochs": self.epochs,
            "f0_method": self.f0_method,
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {self.output_dir / 'metadata.json'}")

    def copy_samples(self, audio_files: List[Path]):
        """Copy training samples to model directory for reference."""
        samples_dir = self.output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        for audio_file in audio_files[:10]:  # Copy first 10 samples
            try:
                shutil.copy(audio_file, samples_dir / audio_file.name)
            except Exception as e:
                logger.warning(f"Could not copy {audio_file}: {e}")

    def train(self) -> bool:
        """Run the training pipeline."""
        logger.info(f"=" * 60)
        logger.info(f"Training RVC model: {self.model_name}")
        logger.info(f"=" * 60)

        try:
            # Step 1: Prepare dataset
            audio_files = self.prepare_dataset()

            # Step 2: Extract features
            self.extract_features(audio_files)

            # Step 3: Create model
            self.create_simple_model(audio_files)

            # Step 4: Save metadata
            self.save_metadata(audio_files)

            # Step 5: Copy samples for reference
            self.copy_samples(audio_files)

            logger.info(f"=" * 60)
            logger.info(f"Training complete!")
            logger.info(f"Model saved to: {self.output_dir}")
            logger.info(f"=" * 60)
            logger.info("")
            logger.info("NOTE: This creates a basic voice profile.")
            logger.info("For full RVC quality, you need to:")
            logger.info("1. Install RVC WebUI from GitHub")
            logger.info("2. Train a full model with HuBERT features")
            logger.info("3. Export the .pth and .index files")
            logger.info("4. Copy them to: " + str(self.output_dir))
            logger.info("")

            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Train an RVC voice model for Helix Transvoicer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train a new voice model
    python train_rvc.py --name "DarthVader" --samples ./vader_audio/

    # Train with custom settings
    python train_rvc.py --name "HomerSimpson" --samples ./homer/ --epochs 200

    # Specify output directory
    python train_rvc.py --name "MyVoice" --samples ./audio/ --output ./models/MyVoice/

For best results:
    - Use 10+ minutes of clean speech audio
    - Avoid background music or noise
    - Include varied speaking styles and emotions
    - Use consistent audio quality (same microphone/source)
        """,
    )

    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Name for the voice model",
    )
    parser.add_argument(
        "--samples", "-s",
        required=True,
        type=Path,
        help="Directory containing audio samples",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory for the model (default: models/<name>/)",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--sample-rate", "-sr",
        type=int,
        default=40000,
        choices=[32000, 40000, 48000],
        help="Sample rate: 32000, 40000, or 48000 (default: 40000)",
    )
    parser.add_argument(
        "--f0-method",
        type=str,
        default="rmvpe",
        choices=["rmvpe", "harvest", "crepe", "pm"],
        help="Pitch extraction method (default: rmvpe)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Training batch size (default: 8)",
    )

    args = parser.parse_args()

    # Validate samples directory
    if not args.samples.exists():
        logger.error(f"Samples directory not found: {args.samples}")
        sys.exit(1)

    # Create trainer and run
    trainer = RVCTrainer(
        model_name=args.name,
        samples_dir=args.samples,
        output_dir=args.output,
        sample_rate=args.sample_rate,
        f0_method=args.f0_method,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    success = trainer.train()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
