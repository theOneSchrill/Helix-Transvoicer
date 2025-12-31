"""
RVC (Retrieval-based Voice Conversion) Module for Helix Transvoicer.

Provides:
- RVC model training from audio samples
- RVC voice conversion inference
- Pre-trained model management (HuBERT, RMVPE)
"""

from helix_transvoicer.backend.rvc.models import RVCModelManager, download_pretrained_models
from helix_transvoicer.backend.rvc.inference import RVCInference
from helix_transvoicer.backend.rvc.training import RVCTrainer

__all__ = [
    "RVCModelManager",
    "RVCInference",
    "RVCTrainer",
    "download_pretrained_models",
]
