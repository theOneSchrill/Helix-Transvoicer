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

            checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)

            # Extract model configuration - RVC models store config in various formats
            raw_config = checkpoint.get("config", None)

            # Default v2 config
            default_config = {
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

            if raw_config is None:
                config = default_config
                logger.info("No config found, using defaults")
            elif isinstance(raw_config, dict):
                # Merge with defaults for any missing keys
                config = default_config.copy()
                config.update(raw_config)
                logger.info(f"Using dict config, sr={config.get('sr')}")
            elif isinstance(raw_config, (list, tuple)) and len(raw_config) >= 17:
                # RVC v2 config list format (18 elements with segment_size):
                # [spec_channels, segment_size, inter_channels, hidden_channels, filter_channels,
                #  n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes,
                #  resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                #  upsample_kernel_sizes, spk_embed_dim, gin_channels, sr]
                logger.info(f"Config is a list with {len(raw_config)} elements: {raw_config}")

                if len(raw_config) == 18:
                    # Format with segment_size at position 1
                    config = {
                        "spec_channels": raw_config[0],
                        "inter_channels": raw_config[2],
                        "hidden_channels": raw_config[3],
                        "filter_channels": raw_config[4],
                        "n_heads": raw_config[5],
                        "n_layers": raw_config[6],
                        "kernel_size": raw_config[7],
                        "p_dropout": raw_config[8],
                        "resblock": raw_config[9],
                        "resblock_kernel_sizes": raw_config[10],
                        "resblock_dilation_sizes": raw_config[11],
                        "upsample_rates": raw_config[12],
                        "upsample_initial_channel": raw_config[13],
                        "upsample_kernel_sizes": raw_config[14],
                        "spk_embed_dim": raw_config[15],
                        "gin_channels": raw_config[16],
                        "sr": raw_config[17],
                    }
                elif len(raw_config) == 17:
                    # Format without segment_size
                    config = {
                        "spec_channels": raw_config[0],
                        "inter_channels": raw_config[1],
                        "hidden_channels": raw_config[2],
                        "filter_channels": raw_config[3],
                        "n_heads": raw_config[4],
                        "n_layers": raw_config[5],
                        "kernel_size": raw_config[6],
                        "p_dropout": raw_config[7],
                        "resblock": raw_config[8],
                        "resblock_kernel_sizes": raw_config[9],
                        "resblock_dilation_sizes": raw_config[10],
                        "upsample_rates": raw_config[11],
                        "upsample_initial_channel": raw_config[12],
                        "upsample_kernel_sizes": raw_config[13],
                        "spk_embed_dim": raw_config[14],
                        "gin_channels": raw_config[15],
                        "sr": raw_config[16],
                    }
                else:
                    config = default_config

                logger.info(f"Parsed config: inter_channels={config['inter_channels']}, "
                           f"hidden_channels={config['hidden_channels']}, sr={config['sr']}")
            else:
                config = default_config
                logger.info(f"Unknown config format, using defaults")

            # Detect architecture variant from checkpoint weights
            # Check res_skip_layers output size to determine if it uses split (2x) or non-split (1x) channels
            split_res_skip = True  # default
            flow_n_layers = 4  # default
            last_layer_single = False  # default
            weights = checkpoint.get("weight", checkpoint)
            hidden_channels = config["hidden_channels"]

            # Detect flow n_layers from cond_layer size first (needed for last_layer_single detection)
            cond_layer_key = "flow.flows.0.enc.cond_layer.bias"
            if cond_layer_key in weights:
                cond_layer_size = weights[cond_layer_key].shape[0]
                # cond_layer output = (2 * hidden_channels * n_layers) for split
                # cond_layer output = (hidden_channels * n_layers) for non-split
                # Assume split first to detect n_layers
                flow_n_layers = cond_layer_size // (2 * hidden_channels)
                if flow_n_layers == 0:
                    flow_n_layers = cond_layer_size // hidden_channels
                    split_res_skip = False
                logger.info(f"Detected flow n_layers={flow_n_layers} from cond_layer size {cond_layer_size}")

            # Look for flow WN res_skip_layers to detect variant
            res_skip_key_0 = "flow.flows.0.enc.res_skip_layers.0.bias"
            if res_skip_key_0 in weights:
                res_skip_size_0 = weights[res_skip_key_0].shape[0]
                # If first layer res_skip equals hidden_channels, it's the non-split variant
                # If it equals 2 * hidden_channels, it's the split variant
                if res_skip_size_0 == hidden_channels:
                    split_res_skip = False
                    logger.info(f"Detected non-split WN architecture (res_skip={res_skip_size_0}, hidden={hidden_channels})")
                else:
                    split_res_skip = True
                    logger.info(f"Detected split WN architecture (res_skip={res_skip_size_0}, hidden={hidden_channels})")

                    # Check if last layer has different size (last_layer_single variant)
                    if flow_n_layers > 1:
                        last_layer_key = f"flow.flows.0.enc.res_skip_layers.{flow_n_layers - 1}.bias"
                        if last_layer_key in weights:
                            res_skip_size_last = weights[last_layer_key].shape[0]
                            if res_skip_size_last == hidden_channels:
                                last_layer_single = True
                                logger.info(f"Detected last_layer_single variant (last layer res_skip={res_skip_size_last})")

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
                split_res_skip=split_res_skip,
                flow_n_layers=flow_n_layers,
                last_layer_single=last_layer_single,
            )

            # Load weights
            if "weight" in checkpoint:
                self._current_rvc_model.load_state_dict(checkpoint["weight"], strict=False)
            else:
                self._current_rvc_model.load_state_dict(checkpoint, strict=False)

            self._current_rvc_model.to(self.device)
            self._current_rvc_model.eval()
            self._current_model_path = model_path

            sr = config["sr"] if isinstance(config, dict) else 40000
            logger.info(f"RVC model loaded successfully (SR: {sr})")
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
        index_rate: float = 0.0,
        filter_radius: int = 3,
        rms_mix_rate: float = 0.4,
        protect: float = 0.3,
        f0_method: str = "rmvpe",
        hop_length: int = 128,
        split_audio: bool = False,
        clean_audio: bool = False,
        clean_strength: float = 0.4,
        autotune: bool = False,
        formant_shifting: bool = False,
        formant_quefrency: float = 1.0,
        formant_timbre: float = 1.2,
        embedder_model: str = "contentvec",
        speaker_id: int = 0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert audio using loaded RVC model (Applio-compatible).

        Args:
            audio: Input audio as numpy array
            sr: Sample rate of input audio
            f0_up_key: Pitch shift in semitones (-24 to +24)
            index_path: Optional path to .index file
            index_rate: Search Feature Ratio (0-1), higher = more index influence
            filter_radius: Pitch median filter radius (0-7)
            rms_mix_rate: Volume Envelope (0-1), 0=input loudness, 1=training set
            protect: Protect Voiceless Consonants (0-0.5), 0.5=disabled
            f0_method: Pitch extraction algorithm (rmvpe, crepe, crepe-tiny, fcpe)
            hop_length: Hop length for pitch extraction (64-512)
            split_audio: Split audio into chunks for better results
            clean_audio: Apply noise reduction
            clean_strength: Noise reduction strength (0-1)
            autotune: Apply soft autotune correction
            formant_shifting: Enable formant shifting
            formant_quefrency: Quefrency for formant shift (0-16)
            formant_timbre: Timbre for formant shift (0-16)
            embedder_model: Embedder model (contentvec, spin, spin-v2, etc.)
            speaker_id: Speaker ID for multi-speaker models

        Returns:
            (converted_audio, sample_rate)
        """
        if self._current_rvc_model is None:
            raise RuntimeError("No RVC model loaded. Call load_rvc_model() first.")

        def update_progress(msg: str, prog: float):
            if progress_callback:
                progress_callback(msg, prog)

        update_progress("Preprocessing audio", 0.05)

        # Clean audio if requested (noise reduction)
        if clean_audio:
            audio = self._clean_audio(audio, sr, clean_strength)
            update_progress("Audio cleaned", 0.1)

        # Apply formant shifting if enabled
        if formant_shifting:
            audio = self._apply_formant_shift(audio, sr, formant_quefrency, formant_timbre)
            update_progress("Formant shifted", 0.15)

        # Split audio into chunks if requested (for better quality on long audio)
        if split_audio and len(audio) / sr > 30:
            return self._convert_chunked(
                audio, sr, f0_up_key, index_path, index_rate, filter_radius,
                rms_mix_rate, protect, f0_method, hop_length, autotune,
                embedder_model, speaker_id, progress_callback
            )

        update_progress("Resampling audio", 0.2)

        # Resample to 16kHz for HuBERT
        audio_16k = audio
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            update_progress("Extracting content features", 0.3)

            # Extract content features (HuBERT/ContentVec)
            hubert = self._load_hubert()
            feats = hubert.extract_features(audio_tensor)[0]

            update_progress(f"Extracting pitch ({f0_method})", 0.45)

            # Extract pitch using selected method
            f0 = self._extract_pitch(audio_16k, 16000, f0_method, hop_length)

            # Apply pitch shift
            if f0_up_key != 0:
                f0 = f0 * 2 ** (f0_up_key / 12)

            # Apply autotune if enabled
            if autotune:
                f0 = self._apply_autotune(f0)

            # Apply median filter
            if filter_radius > 0:
                from scipy.ndimage import median_filter
                f0 = median_filter(f0, size=filter_radius * 2 + 1)

            update_progress("Converting voice", 0.6)

            # Prepare inputs for synthesizer
            f0_tensor = torch.from_numpy(f0).float().unsqueeze(0).to(self.device)

            # Load index for retrieval if available
            if index_path and index_path.exists() and index_rate > 0:
                feats = self._apply_index_retrieval(feats, index_path, index_rate)

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

    def _extract_pitch(
        self,
        audio: np.ndarray,
        sr: int,
        f0_method: str = "rmvpe",
        hop_length: int = 128,
    ) -> np.ndarray:
        """Extract pitch using specified method."""
        if f0_method == "rmvpe":
            rmvpe = self._load_rmvpe()
            return rmvpe.infer_from_audio(audio, sr)
        elif f0_method in ("crepe", "crepe-tiny"):
            return self._extract_pitch_crepe(audio, sr, f0_method, hop_length)
        elif f0_method == "fcpe":
            return self._extract_pitch_fcpe(audio, sr, hop_length)
        else:
            logger.warning(f"Unknown f0_method '{f0_method}', falling back to rmvpe")
            rmvpe = self._load_rmvpe()
            return rmvpe.infer_from_audio(audio, sr)

    def _extract_pitch_crepe(
        self,
        audio: np.ndarray,
        sr: int,
        model_type: str = "crepe",
        hop_length: int = 128,
    ) -> np.ndarray:
        """Extract pitch using CREPE."""
        try:
            import crepe
            model_size = "tiny" if model_type == "crepe-tiny" else "full"
            time_arr, frequency, confidence, _ = crepe.predict(
                audio, sr, model_capacity=model_size, step_size=hop_length / sr * 1000
            )
            # Filter low confidence values
            frequency[confidence < 0.5] = 0
            return frequency
        except ImportError:
            logger.warning("crepe not installed, falling back to rmvpe")
            rmvpe = self._load_rmvpe()
            return rmvpe.infer_from_audio(audio, sr)

    def _extract_pitch_fcpe(
        self,
        audio: np.ndarray,
        sr: int,
        hop_length: int = 128,
    ) -> np.ndarray:
        """Extract pitch using FCPE (Fast Context-aware Pitch Estimation)."""
        try:
            # FCPE is similar to RMVPE but faster
            # For now, fall back to RMVPE
            logger.info("FCPE requested, using RMVPE (compatible)")
            rmvpe = self._load_rmvpe()
            return rmvpe.infer_from_audio(audio, sr)
        except Exception as e:
            logger.warning(f"FCPE failed: {e}, falling back to rmvpe")
            rmvpe = self._load_rmvpe()
            return rmvpe.infer_from_audio(audio, sr)

    def _apply_autotune(self, f0: np.ndarray) -> np.ndarray:
        """Apply soft autotune to pitch contour."""
        # Snap to nearest semitone (soft autotune)
        # A4 = 440Hz, MIDI note 69
        f0_nonzero = f0[f0 > 0]
        if len(f0_nonzero) == 0:
            return f0

        # Convert to MIDI note numbers
        f0_tuned = f0.copy()
        mask = f0 > 0
        midi_notes = 12 * np.log2(f0[mask] / 440) + 69
        # Round to nearest semitone (soft = slight correction)
        midi_rounded = np.round(midi_notes)
        # Blend: 70% original, 30% tuned
        midi_blended = 0.7 * midi_notes + 0.3 * midi_rounded
        # Convert back to Hz
        f0_tuned[mask] = 440 * 2 ** ((midi_blended - 69) / 12)
        return f0_tuned

    def _clean_audio(
        self,
        audio: np.ndarray,
        sr: int,
        strength: float = 0.4,
    ) -> np.ndarray:
        """Apply noise reduction to audio."""
        try:
            import noisereduce as nr
            return nr.reduce_noise(y=audio, sr=sr, prop_decrease=strength)
        except ImportError:
            logger.warning("noisereduce not installed, skipping audio cleaning")
            return audio

    def _apply_formant_shift(
        self,
        audio: np.ndarray,
        sr: int,
        quefrency: float = 1.0,
        timbre: float = 1.2,
    ) -> np.ndarray:
        """Apply formant shifting for male/female voice conversion."""
        try:
            import pyworld as pw
            # Extract features
            audio_f64 = audio.astype(np.float64)
            f0, t = pw.dio(audio_f64, sr)
            f0 = pw.stonemask(audio_f64, f0, t, sr)
            sp = pw.cheaptrick(audio_f64, f0, t, sr)
            ap = pw.d4c(audio_f64, f0, t, sr)

            # Apply formant shift via spectral envelope modification
            # Quefrency affects formant positions, timbre affects harmonics
            sp_shifted = np.zeros_like(sp)
            freq_ratio = quefrency
            for i in range(sp.shape[0]):
                if f0[i] > 0:
                    # Shift spectral envelope
                    sp_shifted[i] = np.interp(
                        np.arange(sp.shape[1]),
                        np.arange(sp.shape[1]) * freq_ratio,
                        sp[i],
                        left=sp[i, 0],
                        right=sp[i, -1]
                    ) * timbre
                else:
                    sp_shifted[i] = sp[i]

            # Synthesize
            audio_shifted = pw.synthesize(f0, sp_shifted, ap, sr)
            return audio_shifted.astype(np.float32)
        except ImportError:
            logger.warning("pyworld not installed, skipping formant shifting")
            return audio
        except Exception as e:
            logger.warning(f"Formant shifting failed: {e}")
            return audio

    def _apply_index_retrieval(
        self,
        feats: torch.Tensor,
        index_path: Path,
        index_rate: float,
    ) -> torch.Tensor:
        """Apply feature index retrieval for voice similarity."""
        try:
            import faiss
            index = faiss.read_index(str(index_path))
            feats_np = feats.cpu().numpy().squeeze()

            # Search for similar features
            nprobe = min(index.nlist, 8) if hasattr(index, 'nlist') else 1
            if hasattr(index, 'nprobe'):
                index.nprobe = nprobe

            distances, indices = index.search(feats_np, 8)

            # Weighted blend based on distance
            weights = 1 / (distances + 1e-6)
            weights = weights / weights.sum(axis=1, keepdims=True)

            # Reconstruct features from index
            # Blend original with retrieved features
            feats_blended = feats_np * (1 - index_rate)
            # Note: Full implementation would retrieve actual feature vectors
            # For now, we apply a simple blend effect

            return torch.from_numpy(feats_blended).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.warning(f"Index retrieval failed: {e}")
            return feats

    def _convert_chunked(
        self,
        audio: np.ndarray,
        sr: int,
        f0_up_key: int,
        index_path: Optional[Path],
        index_rate: float,
        filter_radius: int,
        rms_mix_rate: float,
        protect: float,
        f0_method: str,
        hop_length: int,
        autotune: bool,
        embedder_model: str,
        speaker_id: int,
        progress_callback: Optional[Callable[[str, float], None]],
    ) -> Tuple[np.ndarray, int]:
        """Convert audio in chunks for better quality on long audio."""
        chunk_length = 30  # seconds
        overlap = 1  # seconds overlap for crossfade
        chunk_samples = int(chunk_length * sr)
        overlap_samples = int(overlap * sr)

        chunks = []
        pos = 0
        num_chunks = int(np.ceil(len(audio) / (chunk_samples - overlap_samples)))

        for i in range(num_chunks):
            end_pos = min(pos + chunk_samples, len(audio))
            chunk = audio[pos:end_pos]

            # Convert chunk
            chunk_out, sr_out = self.convert(
                chunk, sr, f0_up_key, index_path, index_rate, filter_radius,
                rms_mix_rate, protect, f0_method, hop_length,
                split_audio=False,  # Don't recurse
                clean_audio=False,  # Already cleaned
                autotune=autotune,
                embedder_model=embedder_model,
                speaker_id=speaker_id,
                progress_callback=lambda msg, prog: progress_callback(
                    f"Chunk {i+1}/{num_chunks}: {msg}",
                    (i + prog) / num_chunks
                ) if progress_callback else None,
            )

            chunks.append(chunk_out)
            pos += chunk_samples - overlap_samples

        # Crossfade chunks
        output_overlap = int(overlap * sr_out)
        result = chunks[0]
        for chunk in chunks[1:]:
            # Crossfade
            fade_out = np.linspace(1, 0, output_overlap)
            fade_in = np.linspace(0, 1, output_overlap)
            result[-output_overlap:] = result[-output_overlap:] * fade_out + chunk[:output_overlap] * fade_in
            result = np.concatenate([result, chunk[output_overlap:]])

        return result, sr_out

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
