# Helix Transvoicer

**Studio-grade voice conversion and text-to-speech application.**

A professional, local-first voice processing application for audio production. Helix Transvoicer provides voice conversion, custom voice model training, and advanced TTS capabilities with complete local operation.

## Features

### Voice Conversion
- Replace voice in existing audio with custom-trained target voice
- Preserve timing, content, and expression
- Fine-grained controls for pitch, formant, and smoothing

### Custom Voice Models
- Train voice models from user-provided samples (WAV/MP3/FLAC)
- Incremental learning - update models without full retraining
- Automatic quality assessment and preprocessing

### Emotion Analysis
- Detect emotional ranges in training samples
- Coverage analysis across 9 emotion categories
- Recommendations for improving model quality

### Advanced TTS
- Text-to-speech with selectable local voice models
- Controls for speed, pitch, intensity, and emotional expression
- Natural breathing and pausing options

## Architecture

```
┌────────────────────────────────────────────────────┐
│                  HELIX TRANSVOICER                  │
├────────────────────────────────────────────────────┤
│  Frontend (CustomTkinter)                          │
│  ├── Voice Converter Panel                         │
│  ├── Model Builder Panel                           │
│  ├── Emotion Coverage Dashboard                    │
│  ├── TTS Studio Panel                              │
│  └── Model Library Panel                           │
├────────────────────────────────────────────────────┤
│  Backend API (FastAPI)                             │
│  ├── Audio Processing (/api/audio)                 │
│  ├── Voice Conversion (/api/convert)               │
│  ├── Model Management (/api/models)                │
│  ├── TTS Synthesis (/api/tts)                      │
│  └── System Status (/api/system)                   │
├────────────────────────────────────────────────────┤
│  Core Processing                                   │
│  ├── AudioProcessor - Preprocessing pipeline       │
│  ├── VoiceConverter - Voice conversion engine      │
│  ├── ModelTrainer - Training & incremental updates │
│  ├── TTSEngine - Speech synthesis                  │
│  └── EmotionAnalyzer - Emotion detection           │
├────────────────────────────────────────────────────┤
│  ML Models                                         │
│  ├── ContentEncoder - Speaker-independent features │
│  ├── SpeakerEncoder - Speaker identity embedding   │
│  ├── VoiceDecoder - Mel spectrogram generation     │
│  └── Vocoder (HiFi-GAN) - Waveform synthesis       │
└────────────────────────────────────────────────────┘
```

## Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU (recommended) or CPU

### Install
```bash
# Clone repository
git clone https://github.com/your-org/helix-transvoicer.git
cd helix-transvoicer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Usage

### Launch Application
```bash
# Full application (UI + backend)
helix

# Backend only (API server)
helix --server-only

# UI only (requires running backend)
helix --ui-only --api-url http://localhost:8420
```

### API Endpoints

The backend API runs at `http://localhost:8420` by default.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/audio/preprocess` | POST | Preprocess audio file |
| `/api/audio/denoise` | POST | Apply denoising |
| `/api/convert/voice` | POST | Convert voice |
| `/api/models` | GET | List voice models |
| `/api/models/{id}/train` | POST | Train model |
| `/api/models/{id}/update` | POST | Incremental update |
| `/api/tts/synthesize` | POST | Synthesize speech |
| `/api/system/status` | GET | System status |

### Python API
```python
from helix_transvoicer import (
    AudioProcessor,
    VoiceConverter,
    ModelTrainer,
    TTSEngine,
    EmotionAnalyzer,
)

# Preprocess audio
processor = AudioProcessor()
result = processor.process("input.wav")

# Convert voice
converter = VoiceConverter(device="cuda")
output = converter.convert("input.wav", "voice_model_id")

# Synthesize speech
tts = TTSEngine(device="cuda")
audio = tts.synthesize("Hello world", "voice_model_id")
```

## Project Structure

```
helix-transvoicer/
├── src/helix_transvoicer/
│   ├── backend/
│   │   ├── api/           # FastAPI routes
│   │   ├── core/          # Processing engines
│   │   ├── models/        # Neural network models
│   │   ├── services/      # Business logic
│   │   └── utils/         # Utilities
│   └── frontend/
│       ├── panels/        # UI panels
│       ├── components/    # UI components
│       ├── styles/        # Theme & styling
│       └── utils/         # API client
├── docs/
│   └── DESIGN.md          # Detailed design document
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Design Principles

- **Local-first**: All processing on-device, no cloud dependencies
- **Expert-oriented**: Full control, no hidden automation
- **Studio-grade**: Professional audio quality and workflows
- **Modular**: Clean separation of frontend, API, and processing
- **GPU-accelerated**: CUDA support with CPU fallback

## Technology Stack

### Backend
- **PyTorch** - Neural network framework
- **torchaudio/librosa** - Audio processing
- **FastAPI** - REST API server
- **Pydantic** - Data validation

### Frontend
- **CustomTkinter** - Modern UI framework
- **Matplotlib** - Visualization

### Models
- **HiFi-GAN** - Neural vocoder
- **Transformer** - Voice encoding/decoding
- **CNN/LSTM** - Emotion classification

## License

MIT License
