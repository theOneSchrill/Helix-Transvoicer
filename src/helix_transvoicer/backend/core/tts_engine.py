"""
Helix Transvoicer - Text-to-Speech engine.

Advanced TTS with pitch, speed, intensity, and emotion controls.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from helix_transvoicer.backend.models.vocoder import Vocoder
from helix_transvoicer.backend.utils.audio import AudioUtils
from helix_transvoicer.backend.utils.config import get_settings

logger = logging.getLogger("helix.tts_engine")


# Phoneme set for English
PHONEMES = [
    "<pad>", "<unk>", "<sos>", "<eos>",
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH", "EH", "ER",
    "EY", "F", "G", "HH", "IH", "IY",
    "JH", "K", "L", "M", "N", "NG",
    "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W",
    "Y", "Z", "ZH", " ", ".", ",", "!", "?",
]

PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEMES)}


@dataclass
class TTSConfig:
    """TTS synthesis configuration."""

    speed: float = 1.0  # 0.5 to 2.0
    pitch: float = 0.0  # -12 to +12 semitones
    intensity: float = 0.5  # 0 (soft) to 1 (intense)
    variance: float = 0.5  # 0 (low) to 1 (high)
    emotion: str = "neutral"
    emotion_strength: float = 0.5  # 0 to 1
    secondary_emotion: Optional[str] = None
    secondary_emotion_blend: float = 0.0  # 0 to 1
    add_pauses: bool = False
    add_breathing: bool = True
    whisper_mode: bool = False


@dataclass
class TTSResult:
    """Result of TTS synthesis."""

    audio: np.ndarray
    sample_rate: int
    duration: float
    text: str
    phonemes: List[str]
    config: TTSConfig
    processing_time: float
    metadata: Dict = field(default_factory=dict)


class TextNormalizer:
    """Normalize text for TTS processing."""

    # Number words
    ONES = [
        "", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    TENS = [
        "", "", "twenty", "thirty", "forty",
        "fifty", "sixty", "seventy", "eighty", "ninety",
    ]

    @classmethod
    def normalize(cls, text: str) -> str:
        """Normalize text for TTS."""
        text = text.strip()

        # Expand common abbreviations
        text = cls._expand_abbreviations(text)

        # Convert numbers to words
        text = cls._convert_numbers(text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Normalize punctuation
        text = re.sub(r"[\"']", "", text)
        text = re.sub(r"[;:]", ",", text)

        return text.strip()

    @classmethod
    def _expand_abbreviations(cls, text: str) -> str:
        """Expand common abbreviations."""
        abbreviations = {
            "Mr.": "Mister",
            "Mrs.": "Missus",
            "Ms.": "Miss",
            "Dr.": "Doctor",
            "Prof.": "Professor",
            "Jr.": "Junior",
            "Sr.": "Senior",
            "vs.": "versus",
            "etc.": "etcetera",
            "e.g.": "for example",
            "i.e.": "that is",
        }

        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)

        return text

    @classmethod
    def _convert_numbers(cls, text: str) -> str:
        """Convert numbers to words."""

        def number_to_words(n: int) -> str:
            if n < 20:
                return cls.ONES[n]
            if n < 100:
                return cls.TENS[n // 10] + (
                    " " + cls.ONES[n % 10] if n % 10 else ""
                )
            if n < 1000:
                return (
                    cls.ONES[n // 100]
                    + " hundred"
                    + (" " + number_to_words(n % 100) if n % 100 else "")
                )
            if n < 1000000:
                return (
                    number_to_words(n // 1000)
                    + " thousand"
                    + (" " + number_to_words(n % 1000) if n % 1000 else "")
                )
            return str(n)

        def replace_number(match):
            num = int(match.group())
            return number_to_words(num)

        return re.sub(r"\b\d+\b", replace_number, text)


class GraphemeToPhoneme:
    """Convert graphemes (text) to phonemes."""

    # Simple rule-based G2P (for production, use a proper G2P model)
    GRAPHEME_MAP = {
        "a": "AH",
        "b": "B",
        "c": "K",
        "d": "D",
        "e": "EH",
        "f": "F",
        "g": "G",
        "h": "HH",
        "i": "IH",
        "j": "JH",
        "k": "K",
        "l": "L",
        "m": "M",
        "n": "N",
        "o": "AO",
        "p": "P",
        "q": "K",
        "r": "R",
        "s": "S",
        "t": "T",
        "u": "AH",
        "v": "V",
        "w": "W",
        "x": "K S",
        "y": "Y",
        "z": "Z",
    }

    DIGRAPHS = {
        "ch": "CH",
        "sh": "SH",
        "th": "TH",
        "ph": "F",
        "wh": "W",
        "ng": "NG",
        "ee": "IY",
        "oo": "UW",
        "ou": "AW",
        "ow": "OW",
        "oi": "OY",
        "ai": "EY",
        "ea": "IY",
        "ie": "IY",
        "ck": "K",
    }

    @classmethod
    def convert(cls, text: str) -> List[str]:
        """Convert text to phoneme sequence."""
        text = text.lower()
        phonemes = []
        i = 0

        while i < len(text):
            # Check for digraphs
            if i < len(text) - 1:
                digraph = text[i : i + 2]
                if digraph in cls.DIGRAPHS:
                    phonemes.extend(cls.DIGRAPHS[digraph].split())
                    i += 2
                    continue

            char = text[i]

            # Handle punctuation and spaces
            if char in " .,!?":
                phonemes.append(char)
            elif char in cls.GRAPHEME_MAP:
                phonemes.extend(cls.GRAPHEME_MAP[char].split())

            i += 1

        return phonemes


class DurationPredictor(nn.Module):
    """Predict phoneme durations."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.linear = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict duration for each phoneme."""
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)
        return self.linear(x).squeeze(-1)


class PitchPredictor(nn.Module):
    """Predict pitch contour."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.linear = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict pitch for each frame."""
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)
        return self.linear(x).squeeze(-1)


class AcousticModel(nn.Module):
    """Generate mel spectrogram from phonemes and embeddings."""

    def __init__(
        self,
        vocab_size: int = len(PHONEMES),
        embed_dim: int = 256,
        hidden_dim: int = 512,
        n_mels: int = 80,
        speaker_dim: int = 256,
        emotion_dim: int = 64,
    ):
        super().__init__()

        self.phoneme_embedding = nn.Embedding(vocab_size, embed_dim)
        self.speaker_projection = nn.Linear(speaker_dim, hidden_dim)
        self.emotion_projection = nn.Linear(emotion_dim, hidden_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                batch_first=True,
            ),
            num_layers=4,
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_mels),
        )

        self.duration_predictor = DurationPredictor(embed_dim)
        self.pitch_predictor = PitchPredictor(embed_dim)

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        speaker_embedding: torch.Tensor,
        emotion_embedding: torch.Tensor,
        duration_scale: float = 1.0,
        pitch_shift: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate mel spectrogram from phonemes."""
        # Embed phonemes
        x = self.phoneme_embedding(phoneme_ids)

        # Encode
        x = self.encoder(x)

        # Predict durations
        durations = self.duration_predictor(x)
        durations = torch.clamp(durations * duration_scale, min=1)

        # Predict pitch
        pitch = self.pitch_predictor(x) + pitch_shift

        # Expand to frame level using durations
        expanded = self._expand_by_duration(x, durations.round().long())

        # Project speaker and emotion embeddings
        # Squeeze to 2D [batch, dim] if needed, then project
        speaker_emb = speaker_embedding.squeeze() if speaker_embedding.dim() > 2 else speaker_embedding
        emotion_emb = emotion_embedding.squeeze() if emotion_embedding.dim() > 2 else emotion_embedding

        # Ensure batch dimension exists
        if speaker_emb.dim() == 1:
            speaker_emb = speaker_emb.unsqueeze(0)
        if emotion_emb.dim() == 1:
            emotion_emb = emotion_emb.unsqueeze(0)

        speaker = self.speaker_projection(speaker_emb)
        emotion = self.emotion_projection(emotion_emb)

        # Broadcast to match expanded sequence length
        batch_size, seq_len, _ = expanded.shape
        speaker = speaker.unsqueeze(1).expand(-1, seq_len, -1)
        emotion = emotion.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate and decode
        combined = torch.cat([expanded, speaker, emotion], dim=-1)
        mel = self.decoder(combined)

        return mel, durations, pitch

    def _expand_by_duration(
        self,
        x: torch.Tensor,
        durations: torch.Tensor,
    ) -> torch.Tensor:
        """Expand sequence by durations."""
        batch_size = x.shape[0]
        max_len = int(durations.sum(dim=1).max().item())

        expanded = []
        for b in range(batch_size):
            frames = []
            for i, dur in enumerate(durations[b]):
                frames.append(x[b, i].unsqueeze(0).expand(int(dur.item()), -1))
            if frames:
                expanded.append(torch.cat(frames, dim=0))

        # Pad to max length
        result = torch.zeros(batch_size, max_len, x.shape[-1], device=x.device)
        for b, seq in enumerate(expanded):
            result[b, : len(seq)] = seq

        return result


class TTSEngine:
    """
    Text-to-Speech synthesis engine.

    Features:
    - Text normalization
    - Grapheme-to-phoneme conversion
    - Duration and pitch prediction
    - Speaker embedding support
    - Emotion embedding support
    - Fine-grained control over speed, pitch, intensity
    """

    EMOTIONS = [
        "neutral", "happy", "sad", "angry",
        "fear", "surprise", "disgust", "calm", "excited",
    ]

    def __init__(
        self,
        device: Optional[torch.device] = None,
        models_dir: Optional[Path] = None,
    ):
        self.settings = get_settings()
        self.device = device or torch.device("cpu")
        self.models_dir = models_dir or self.settings.models_dir

        self.text_normalizer = TextNormalizer()
        self.g2p = GraphemeToPhoneme()

        # Initialize models (lazy loading)
        self._acoustic_model: Optional[AcousticModel] = None
        self._vocoder: Optional[Vocoder] = None
        self._current_speaker_dim: int = 256  # Track current speaker dimension

        # Emotion embeddings
        self._emotion_embeddings = self._create_emotion_embeddings()

        # Cache for loaded speaker embeddings
        self._speaker_cache: Dict[str, torch.Tensor] = {}

    def _get_acoustic_model(self, speaker_dim: int = 256) -> AcousticModel:
        """Get acoustic model with correct speaker dimension."""
        # Reinitialize if speaker dimension changed
        if self._acoustic_model is None or self._current_speaker_dim != speaker_dim:
            logger.info(f"Initializing acoustic model with speaker_dim={speaker_dim}")
            self._acoustic_model = AcousticModel(speaker_dim=speaker_dim).to(self.device)
            self._acoustic_model.eval()
            self._current_speaker_dim = speaker_dim
        return self._acoustic_model

    @property
    def acoustic_model(self) -> AcousticModel:
        """Lazy-load acoustic model with default dimension."""
        return self._get_acoustic_model(speaker_dim=256)

    @property
    def vocoder(self) -> Vocoder:
        """Lazy-load vocoder."""
        if self._vocoder is None:
            self._vocoder = Vocoder().to(self.device)
            self._vocoder.eval()
        return self._vocoder

    def _create_emotion_embeddings(self) -> Dict[str, torch.Tensor]:
        """Create default emotion embeddings."""
        embeddings = {}
        for i, emotion in enumerate(self.EMOTIONS):
            emb = torch.zeros(64)
            emb[i * 7 : (i + 1) * 7] = 1.0
            embeddings[emotion] = emb.to(self.device)
        return embeddings

    def synthesize(
        self,
        text: str,
        voice_model_id: str,
        config: Optional[TTSConfig] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            text: Input text
            voice_model_id: ID of voice model to use
            config: TTS configuration
            progress_callback: Progress update callback

        Returns:
            TTSResult with synthesized audio
        """
        import time

        start_time = time.time()
        cfg = config or TTSConfig()

        def update_progress(stage: str, progress: float):
            if progress_callback:
                progress_callback(stage, progress)

        update_progress("Normalizing text", 0.1)

        # Normalize text
        normalized_text = self.text_normalizer.normalize(text)

        update_progress("Converting to phonemes", 0.2)

        # Convert to phonemes
        phonemes = self.g2p.convert(normalized_text)

        # Convert to IDs
        phoneme_ids = [
            PHONEME_TO_ID.get(p, PHONEME_TO_ID["<unk>"])
            for p in phonemes
        ]
        phoneme_tensor = torch.tensor([phoneme_ids], dtype=torch.long, device=self.device)

        update_progress("Loading voice model", 0.3)

        # Get speaker embedding
        speaker_embedding = self._get_speaker_embedding(voice_model_id)

        # Detect speaker embedding dimension and get matching acoustic model
        speaker_dim = speaker_embedding.shape[-1]

        # Get emotion embedding
        emotion_embedding = self._get_emotion_embedding(
            cfg.emotion,
            cfg.emotion_strength,
            cfg.secondary_emotion,
            cfg.secondary_emotion_blend,
        )

        update_progress("Generating speech", 0.5)

        # Get acoustic model with correct speaker dimension
        acoustic_model = self._get_acoustic_model(speaker_dim=speaker_dim)

        # Generate mel spectrogram
        with torch.no_grad():
            mel, durations, pitch = acoustic_model(
                phoneme_tensor,
                speaker_embedding.unsqueeze(0),
                emotion_embedding.unsqueeze(0),
                duration_scale=1.0 / cfg.speed,
                pitch_shift=cfg.pitch,
            )

            # Apply intensity
            if cfg.intensity != 0.5:
                intensity_factor = 0.5 + cfg.intensity
                mel = mel * intensity_factor

            # Apply variance
            if cfg.variance != 0.5:
                noise = torch.randn_like(mel) * (cfg.variance - 0.5) * 0.1
                mel = mel + noise

        update_progress("Synthesizing waveform", 0.7)

        # Vocode
        with torch.no_grad():
            audio = self.vocoder(mel)
            audio = audio.squeeze().cpu().numpy()

        # Apply whisper mode
        if cfg.whisper_mode:
            audio = self._apply_whisper(audio)

        # Add breathing sounds
        if cfg.add_breathing:
            audio = self._add_breathing(audio, self.settings.sample_rate)

        update_progress("Finalizing", 0.9)

        # Normalize
        audio = AudioUtils.normalize(audio)

        processing_time = time.time() - start_time

        update_progress("Complete", 1.0)

        return TTSResult(
            audio=audio,
            sample_rate=self.settings.sample_rate,
            duration=len(audio) / self.settings.sample_rate,
            text=text,
            phonemes=phonemes,
            config=cfg,
            processing_time=processing_time,
            metadata={
                "voice_model": voice_model_id,
                "device": str(self.device),
            },
        )

    def preview(
        self,
        text: str,
        voice_model_id: str,
        config: Optional[TTSConfig] = None,
    ) -> TTSResult:
        """Generate a quick preview (shorter processing)."""
        # Truncate long text for preview
        if len(text) > 100:
            text = text[:100] + "..."

        return self.synthesize(text, voice_model_id, config)

    def list_voices(self) -> List[Dict]:
        """List available voice models."""
        voices = []

        if self.models_dir.exists():
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir():
                    metadata_path = model_dir / "metadata.json"
                    if metadata_path.exists():
                        import json

                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        voices.append(
                            {
                                "id": model_dir.name,
                                "name": metadata.get("model_id", model_dir.name),
                                "version": metadata.get("version", "1.0.0"),
                            }
                        )

        return voices

    def _get_speaker_embedding(self, model_id: str) -> torch.Tensor:
        """Load speaker embedding for model."""
        if model_id in self._speaker_cache:
            return self._speaker_cache[model_id]

        model_dir = self.models_dir / model_id
        embedding_path = model_dir / "speaker_embedding.npy"

        if embedding_path.exists():
            embedding = np.load(str(embedding_path))
            embedding_tensor = torch.from_numpy(embedding).float().to(self.device)
            logger.info(f"Loaded speaker embedding for {model_id}: dim={embedding_tensor.shape[-1]}")
            self._speaker_cache[model_id] = embedding_tensor
            return embedding_tensor

        # Check architecture.json to determine default dimension
        default_dim = 256
        arch_path = model_dir / "architecture.json"
        if arch_path.exists():
            try:
                import json
                with open(arch_path, "r") as f:
                    arch = json.load(f)
                    if arch.get("low_memory_mode", False):
                        default_dim = arch.get("speaker_encoder", {}).get("embedding_dim", 64)
            except Exception:
                pass

        # Return default embedding
        logger.warning(f"Speaker embedding not found for {model_id}, using zeros (dim={default_dim})")
        return torch.zeros(default_dim, device=self.device)

    def _get_emotion_embedding(
        self,
        primary_emotion: str,
        primary_strength: float,
        secondary_emotion: Optional[str],
        secondary_blend: float,
    ) -> torch.Tensor:
        """Get blended emotion embedding."""
        primary_emb = self._emotion_embeddings.get(
            primary_emotion,
            self._emotion_embeddings["neutral"],
        )
        result = primary_emb * primary_strength

        if secondary_emotion and secondary_blend > 0:
            secondary_emb = self._emotion_embeddings.get(
                secondary_emotion,
                self._emotion_embeddings["neutral"],
            )
            result = result * (1 - secondary_blend) + secondary_emb * secondary_blend

        return result

    def _apply_whisper(self, audio: np.ndarray) -> np.ndarray:
        """Apply whisper effect."""
        # Add breathiness noise
        noise = np.random.randn(len(audio)) * 0.05
        audio = audio * 0.7 + noise

        # High-pass filter to remove low frequencies
        from scipy import signal

        b, a = signal.butter(2, 300 / (self.settings.sample_rate / 2), "high")
        audio = signal.filtfilt(b, a, audio)

        return audio

    def _add_breathing(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """Add subtle breathing sounds at pauses."""
        # This is a placeholder - in production, use actual breath samples
        return audio
