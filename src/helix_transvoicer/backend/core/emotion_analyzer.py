"""
Helix Transvoicer - Emotion analysis system.

Detects emotional ranges in audio samples and provides coverage analysis.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helix_transvoicer.backend.utils.audio import AudioUtils
from helix_transvoicer.backend.utils.config import get_settings

logger = logging.getLogger("helix.emotion_analyzer")


# Emotion categories with valence-arousal mapping
EMOTIONS = {
    "neutral": {"valence": 0.0, "arousal": 0.0},
    "happy": {"valence": 0.8, "arousal": 0.6},
    "sad": {"valence": -0.7, "arousal": -0.4},
    "angry": {"valence": -0.6, "arousal": 0.8},
    "fear": {"valence": -0.8, "arousal": 0.7},
    "surprise": {"valence": 0.2, "arousal": 0.8},
    "disgust": {"valence": -0.7, "arousal": 0.2},
    "calm": {"valence": 0.3, "arousal": -0.6},
    "excited": {"valence": 0.7, "arousal": 0.9},
}

EMOTION_LIST = list(EMOTIONS.keys())


@dataclass
class EmotionResult:
    """Result of emotion analysis for a single sample."""

    probabilities: Dict[str, float]
    primary_emotion: str
    primary_confidence: float
    secondary_emotion: Optional[str]
    secondary_confidence: float
    valence: float
    arousal: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class EmotionCoverage:
    """Emotion coverage analysis for a set of samples."""

    coverage_scores: Dict[str, float]  # 0-1 coverage per emotion
    confidence_scores: Dict[str, float]  # Average confidence per emotion
    sample_counts: Dict[str, int]  # Number of samples per emotion
    total_samples: int
    total_duration: float
    gaps: List[str]  # Emotions with low coverage
    recommendations: List[str]  # Suggestions for improvement
    health_score: float  # Overall coverage health 0-1


class EmotionClassifier(nn.Module):
    """Neural network for emotion classification from audio features."""

    def __init__(
        self,
        input_dim: int = 120,  # MFCCs (40) + delta (40) + delta-delta (40)
        hidden_dim: int = 256,
        num_emotions: int = len(EMOTIONS),
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1)

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_emotions),
        )

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch, time, features]

        Returns:
            Emotion logits [batch, num_emotions]
        """
        # Transpose for conv layers
        x = x.transpose(1, 2)

        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Transpose back for LSTM
        x = x.transpose(1, 2)

        # LSTM
        x, _ = self.lstm(x)

        # Self-attention
        x, _ = self.attention(x, x, x)

        # Global pooling
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)

        # Classification
        return self.fc(x)


class EmotionAnalyzer:
    """
    Emotion analysis engine.

    Capabilities:
    - Detect emotions in audio samples
    - Analyze emotion coverage across multiple samples
    - Provide recommendations for improving coverage
    - Map emotions to valence-arousal space
    """

    COVERAGE_THRESHOLD = 0.3  # Minimum coverage to not be considered a gap

    def __init__(
        self,
        device: Optional[torch.device] = None,
    ):
        self.settings = get_settings()
        self.device = device or torch.device("cpu")

        # Initialize classifier (lazy loading)
        self._classifier: Optional[EmotionClassifier] = None

    @property
    def classifier(self) -> EmotionClassifier:
        """Lazy-load emotion classifier."""
        if self._classifier is None:
            self._classifier = EmotionClassifier().to(self.device)
            self._classifier.eval()
        return self._classifier

    def analyze(
        self,
        audio: Union[np.ndarray, str, Path],
        sample_rate: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Analyze emotion in audio sample.

        Args:
            audio: Audio data or path to audio file
            sample_rate: Sample rate (if audio is array)

        Returns:
            Dictionary of emotion probabilities
        """
        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio, sample_rate = AudioUtils.load_audio(
                audio,
                target_sr=self.settings.sample_rate,
            )
        elif sample_rate is None:
            sample_rate = self.settings.sample_rate

        # Extract features
        features = self._extract_features(audio, sample_rate)

        # Run classifier
        with torch.no_grad():
            features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            logits = self.classifier(features_tensor)
            probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Map to emotion labels
        return {emotion: float(probs[i]) for i, emotion in enumerate(EMOTION_LIST)}

    def analyze_detailed(
        self,
        audio: Union[np.ndarray, str, Path],
        sample_rate: Optional[int] = None,
    ) -> EmotionResult:
        """
        Perform detailed emotion analysis.

        Returns:
            EmotionResult with full analysis
        """
        probs = self.analyze(audio, sample_rate)

        # Get primary and secondary emotions
        sorted_emotions = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_emotions[0]
        secondary = sorted_emotions[1] if len(sorted_emotions) > 1 else (None, 0.0)

        # Compute valence and arousal
        valence = sum(
            probs[e] * EMOTIONS[e]["valence"]
            for e in EMOTION_LIST
        )
        arousal = sum(
            probs[e] * EMOTIONS[e]["arousal"]
            for e in EMOTION_LIST
        )

        return EmotionResult(
            probabilities=probs,
            primary_emotion=primary[0],
            primary_confidence=primary[1],
            secondary_emotion=secondary[0] if secondary[1] > 0.1 else None,
            secondary_confidence=secondary[1],
            valence=valence,
            arousal=arousal,
        )

    def analyze_coverage(
        self,
        samples: List[Dict],
    ) -> EmotionCoverage:
        """
        Analyze emotion coverage across multiple samples.

        Args:
            samples: List of dicts with 'audio', 'sample_rate', 'duration'

        Returns:
            EmotionCoverage analysis
        """
        # Initialize counters
        emotion_samples: Dict[str, List[float]] = {e: [] for e in EMOTION_LIST}
        total_duration = 0.0

        # Analyze each sample
        for sample in samples:
            result = self.analyze_detailed(
                sample["audio"],
                sample.get("sample_rate"),
            )

            # Accumulate confidence for primary emotion
            emotion_samples[result.primary_emotion].append(result.primary_confidence)
            total_duration += sample.get("duration", 0.0)

            # Also count secondary emotion with lower weight
            if result.secondary_emotion:
                emotion_samples[result.secondary_emotion].append(
                    result.secondary_confidence * 0.5
                )

        # Compute coverage scores
        coverage_scores = {}
        confidence_scores = {}
        sample_counts = {}

        for emotion in EMOTION_LIST:
            samples_list = emotion_samples[emotion]
            sample_counts[emotion] = len(samples_list)

            if samples_list:
                # Coverage: 5+ samples with high confidence = 100%
                coverage_scores[emotion] = min(len(samples_list) / 5.0, 1.0)
                confidence_scores[emotion] = float(np.mean(samples_list))
            else:
                coverage_scores[emotion] = 0.0
                confidence_scores[emotion] = 0.0

        # Identify gaps
        gaps = [
            emotion
            for emotion, score in coverage_scores.items()
            if score < self.COVERAGE_THRESHOLD
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            coverage_scores,
            sample_counts,
            gaps,
        )

        # Compute health score
        health_score = np.mean(list(coverage_scores.values()))

        return EmotionCoverage(
            coverage_scores=coverage_scores,
            confidence_scores=confidence_scores,
            sample_counts=sample_counts,
            total_samples=len(samples),
            total_duration=total_duration,
            gaps=gaps,
            recommendations=recommendations,
            health_score=health_score,
        )

    def get_emotion_info(self, emotion: str) -> Dict:
        """Get information about an emotion."""
        if emotion not in EMOTIONS:
            return {}

        return {
            "name": emotion,
            "valence": EMOTIONS[emotion]["valence"],
            "arousal": EMOTIONS[emotion]["arousal"],
            "description": self._get_emotion_description(emotion),
        }

    def _extract_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Extract features for emotion classification."""
        # MFCCs (40 coefficients)
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40,
            n_fft=2048,
            hop_length=512,
        )

        # Delta and delta-delta
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

        # Stack features
        features = np.vstack([mfccs, mfcc_delta, mfcc_delta2])

        # Transpose to [time, features]
        features = features.T

        # Truncate or pad to fixed length
        max_len = 300  # ~3 seconds at 100 frames/sec
        if len(features) > max_len:
            features = features[:max_len]
        elif len(features) < max_len:
            pad_width = ((0, max_len - len(features)), (0, 0))
            features = np.pad(features, pad_width, mode="constant")

        return features

    def _generate_recommendations(
        self,
        coverage: Dict[str, float],
        counts: Dict[str, int],
        gaps: List[str],
    ) -> List[str]:
        """Generate recommendations for improving emotion coverage."""
        recommendations = []

        for emotion in gaps:
            count = counts.get(emotion, 0)
            cov = coverage.get(emotion, 0.0)

            if count == 0:
                recommendations.append(
                    f"Record 3-5 samples expressing '{emotion}' emotion (15-30 seconds each)"
                )
            elif cov < 0.3:
                needed = max(1, 5 - count)
                recommendations.append(
                    f"Add {needed} more '{emotion}' samples to improve coverage"
                )

        # General recommendations
        if len(gaps) > 3:
            recommendations.append(
                "Consider recording a diverse emotional range in a single session"
            )

        return recommendations

    def _get_emotion_description(self, emotion: str) -> str:
        """Get description of an emotion."""
        descriptions = {
            "neutral": "Baseline state with no strong emotional content",
            "happy": "Joyful, pleased, or content expression",
            "sad": "Sorrowful, melancholic, or downcast expression",
            "angry": "Frustrated, irritated, or enraged expression",
            "fear": "Scared, anxious, or worried expression",
            "surprise": "Startled or amazed expression",
            "disgust": "Repulsed or averse expression",
            "calm": "Relaxed, peaceful, or serene expression",
            "excited": "Enthusiastic or energetic expression",
        }
        return descriptions.get(emotion, "")

    def visualize_coverage(
        self,
        coverage: EmotionCoverage,
    ) -> Dict:
        """
        Generate visualization data for emotion coverage.

        Returns:
            Dictionary with visualization data for UI
        """
        # Valence-arousal plot data
        plot_data = []
        for emotion in EMOTION_LIST:
            info = EMOTIONS[emotion]
            plot_data.append({
                "emotion": emotion,
                "valence": info["valence"],
                "arousal": info["arousal"],
                "coverage": coverage.coverage_scores[emotion],
                "confidence": coverage.confidence_scores[emotion],
                "samples": coverage.sample_counts[emotion],
            })

        # Bar chart data
        bar_data = [
            {
                "emotion": emotion,
                "coverage": coverage.coverage_scores[emotion] * 100,
                "is_gap": emotion in coverage.gaps,
            }
            for emotion in EMOTION_LIST
        ]

        return {
            "plot_data": plot_data,
            "bar_data": bar_data,
            "health_score": coverage.health_score * 100,
            "gaps": coverage.gaps,
            "recommendations": coverage.recommendations,
        }
