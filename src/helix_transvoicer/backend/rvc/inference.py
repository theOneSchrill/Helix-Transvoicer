"""
RVC Voice Conversion Inference.

Performs voice conversion using trained RVC models.
"""

import gc
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from helix_transvoicer.backend.rvc.models import RVCModelManager, check_rvc_ready
from helix_transvoicer.backend.utils.config import get_settings

logger = logging.getLogger("helix.rvc.inference")


class RVCInference:
    """
    RVC Voice Conversion Inference Engine.

    Converts audio from one voice to another using a trained RVC model.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        assets_dir: Optional[Path] = None,
    ):
        self.settings = get_settings()
        self.model_manager = RVCModelManager(assets_dir)

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"RVC Inference using device: {self.device}")

        # Lazy-loaded models
        self._hubert_model = None
        self._rmvpe_model = None
        self._current_rvc_model = None
        self._current_model_path = None

        # Default parameters
        self.sample_rate = 16000  # HuBERT expects 16kHz
        self.hop_length = 320

    def _ensure_pretrained_models(self):
        """Ensure pre-trained models are downloaded."""
        ready, missing = check_rvc_ready(for_training=False)
        if not ready:
            raise RuntimeError(
                f"Missing pre-trained models: {missing}. "
                "Call download_pretrained_models() first."
            )

    def _load_hubert(self):
        """Load HuBERT model for content feature extraction."""
        if self._hubert_model is not None:
            return self._hubert_model

        self._ensure_pretrained_models()

        hubert_path = self.model_manager.get_model_path("hubert_base")
        logger.info(f"Loading HuBERT from {hubert_path}")

        try:
            # Load HuBERT model
            from fairseq import checkpoint_utils

            models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
                [str(hubert_path)],
                suffix="",
            )
            self._hubert_model = models[0].to(self.device)
            self._hubert_model.eval()

            logger.info("HuBERT model loaded successfully")
            return self._hubert_model

        except ImportError:
            logger.warning("fairseq not available, using fallback HuBERT loading")
            # Fallback: try loading as regular PyTorch model
            checkpoint = torch.load(str(hubert_path), map_location=self.device)
            # This requires a compatible model structure
            raise NotImplementedError(
                "HuBERT loading requires fairseq. Install with: "
                "pip install fairseq"
            )

    def _load_rmvpe(self):
        """Load RMVPE model for pitch extraction."""
        if self._rmvpe_model is not None:
            return self._rmvpe_model

        self._ensure_pretrained_models()

        rmvpe_path = self.model_manager.get_model_path("rmvpe")
        logger.info(f"Loading RMVPE from {rmvpe_path}")

        try:
            # Load RMVPE checkpoint
            checkpoint = torch.load(str(rmvpe_path), map_location=self.device)

            # RMVPE is a U-Net style model
            from helix_transvoicer.backend.rvc.rmvpe import RMVPE

            self._rmvpe_model = RMVPE(checkpoint)
            self._rmvpe_model.to(self.device)
            self._rmvpe_model.eval()

            logger.info("RMVPE model loaded successfully")
            return self._rmvpe_model

        except Exception as e:
            logger.error(f"Failed to load RMVPE: {e}")
            raise

    def load_rvc_model(self, model_path: Union[str, Path]) -> bool:
        """
        Load an RVC voice model.

        Args:
            model_path: Path to .pth RVC model file

        Returns:
            True if loaded successfully
        """
        model_path = Path(model_path)

        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False

        if self._current_model_path == model_path:
            return True  # Already loaded

        try:
            logger.info(f"Loading RVC model: {model_path}")

            checkpoint = torch.load(str(model_path), map_location=self.device)

            # Extract model configuration
            if "config" in checkpoint:
                config = checkpoint["config"]
            else:
                # Default v2 config
                config = {
                    "spec_channels": 1025,
                    "inter_channels": 192,
                    "hidden_channels": 192,
                    "filter_channels": 768,
                    "n_heads": 2,
                    "n_layers": 6,
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
                    "sr": 40000,
                }

            # Build and load synthesizer
            from helix_transvoicer.backend.rvc.synthesizer import SynthesizerTrnMs768NSFsid

            self._current_rvc_model = SynthesizerTrnMs768NSFsid(
                config["spec_channels"],
                config["inter_channels"],
                config["hidden_channels"],
                config["filter_channels"],
                config["n_heads"],
                config["n_layers"],
                config["kernel_size"],
                config["p_dropout"],
                config["resblock"],
                config["resblock_kernel_sizes"],
                config["resblock_dilation_sizes"],
                config["upsample_rates"],
                config["upsample_initial_channel"],
                config["upsample_kernel_sizes"],
                config["spk_embed_dim"],
                config["gin_channels"],
                config["sr"],
            )

            # Load weights
            if "weight" in checkpoint:
                self._current_rvc_model.load_state_dict(checkpoint["weight"])
            else:
                self._current_rvc_model.load_state_dict(checkpoint)

            self._current_rvc_model.to(self.device)
            self._current_rvc_model.eval()
            self._current_model_path = model_path

            logger.info(f"RVC model loaded successfully (SR: {config.get('sr', 40000)})")
            return True

        except Exception as e:
            logger.error(f"Failed to load RVC model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def convert(
        self,
        audio: np.ndarray,
        sr: int,
        f0_up_key: int = 0,
        index_path: Optional[Path] = None,
        index_rate: float = 0.75,
        filter_radius: int = 3,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert audio using loaded RVC model.

        Args:
            audio: Input audio as numpy array
            sr: Sample rate of input audio
            f0_up_key: Pitch shift in semitones
            index_path: Optional path to .index file
            index_rate: Index influence (0-1)
            filter_radius: Pitch median filter radius
            rms_mix_rate: RMS volume envelope mixing
            protect: Protect voiceless consonants

        Returns:
            (converted_audio, sample_rate)
        """
        if self._current_rvc_model is None:
            raise RuntimeError("No RVC model loaded. Call load_rvc_model() first.")

        def update_progress(msg: str, prog: float):
            if progress_callback:
                progress_callback(msg, prog)

        update_progress("Preprocessing audio", 0.1)

        # Resample to 16kHz for HuBERT
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            update_progress("Extracting content features", 0.2)

            # Extract HuBERT features
            hubert = self._load_hubert()
            feats = hubert.extract_features(audio_tensor)[0]

            update_progress("Extracting pitch", 0.4)

            # Extract pitch using RMVPE
            rmvpe = self._load_rmvpe()
            f0 = rmvpe.infer_from_audio(audio, 16000)

            # Apply pitch shift
            if f0_up_key != 0:
                f0 = f0 * 2 ** (f0_up_key / 12)

            # Apply median filter
            if filter_radius > 0:
                from scipy.ndimage import median_filter
                f0 = median_filter(f0, size=filter_radius * 2 + 1)

            update_progress("Converting voice", 0.6)

            # Prepare inputs for synthesizer
            f0_tensor = torch.from_numpy(f0).float().unsqueeze(0).to(self.device)

            # Load index for retrieval if available
            if index_path and index_path.exists() and index_rate > 0:
                try:
                    import faiss

                    index = faiss.read_index(str(index_path))
                    # Apply feature retrieval
                    feats_np = feats.cpu().numpy().squeeze()
                    _, indices = index.search(feats_np, 1)
                    # Blend with retrieved features
                    # (simplified - full implementation would retrieve actual features)
                except Exception as e:
                    logger.warning(f"Index loading failed: {e}")

            # Run synthesizer
            audio_out = self._current_rvc_model.infer(
                feats,
                f0_tensor,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
            )

            update_progress("Post-processing", 0.9)

            # Convert output
            audio_out = audio_out.squeeze().cpu().numpy()
            output_sr = 40000  # RVC default output sample rate

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        update_progress("Complete", 1.0)

        return audio_out, output_sr

    def convert_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        **kwargs,
    ) -> bool:
        """
        Convert an audio file.

        Args:
            input_path: Path to input audio file
            output_path: Path for output audio file
            **kwargs: Additional arguments passed to convert()

        Returns:
            True if successful
        """
        try:
            # Load input audio
            audio, sr = librosa.load(str(input_path), sr=None)

            # Convert
            audio_out, sr_out = self.convert(audio, sr, **kwargs)

            # Save output
            import soundfile as sf
            sf.write(str(output_path), audio_out, sr_out)

            return True

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False

    def unload_models(self):
        """Unload all models to free memory."""
        self._hubert_model = None
        self._rmvpe_model = None
        self._current_rvc_model = None
        self._current_model_path = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Models unloaded")
