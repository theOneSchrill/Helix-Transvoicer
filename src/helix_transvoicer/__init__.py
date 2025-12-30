"""
Helix Transvoicer - Studio-grade voice conversion and TTS application.

A professional, local-first voice processing application providing:
- Voice conversion with custom-trained target voices
- Custom voice model creation from user samples
- Incremental model updates without full retraining
- Advanced Text-to-Speech with fine-grained controls
- Emotion coverage analysis for voice models
"""

__version__ = "1.0.0"
__author__ = "Helix Audio"

from helix_transvoicer.backend.core.audio_processor import AudioProcessor
from helix_transvoicer.backend.core.voice_converter import VoiceConverter
from helix_transvoicer.backend.core.model_trainer import ModelTrainer
from helix_transvoicer.backend.core.tts_engine import TTSEngine
from helix_transvoicer.backend.core.emotion_analyzer import EmotionAnalyzer

__all__ = [
    "__version__",
    "AudioProcessor",
    "VoiceConverter",
    "ModelTrainer",
    "TTSEngine",
    "EmotionAnalyzer",
]
