# Helix Transvoicer

**Studio-grade voice conversion and text-to-speech application for Windows 11.**

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

## System Requirements

### Minimum
- **OS**: Windows 11 (Build 22000+)
- **CPU**: Intel Core i5 / AMD Ryzen 5 or better
- **RAM**: 8 GB
- **Storage**: 2 GB free space
- **Python**: 3.10 or later

### Recommended (GPU Acceleration)
- **GPU**: NVIDIA RTX 2060 or better (6+ GB VRAM)
- **CUDA**: 11.8 or 12.1
- **RAM**: 16 GB

## Installation

### Quick Setup (PowerShell)

1. **Download** or clone this repository

2. **Open PowerShell as Administrator** and navigate to the folder:
   ```powershell
   cd C:\path\to\Helix-Transvoicer
   ```

3. **Allow script execution** (if needed):
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. **Run the setup script**:
   ```powershell
   .\setup.ps1
   ```

   For CPU-only installation (no NVIDIA GPU):
   ```powershell
   .\setup.ps1 -CudaVersion cpu
   ```

5. **Launch** using the desktop shortcut or:
   ```batch
   run.bat
   ```

### Manual Installation

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch (choose one)
# CUDA 12.1 (recommended for RTX 30/40 series):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (for older GPUs):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# Install Helix Transvoicer
pip install -e .
```

## Usage

### Starting the Application

**Option 1: Desktop Shortcut**
- Double-click "Helix Transvoicer" on your desktop

**Option 2: Batch File**
```batch
run.bat
```

**Option 3: Command Line**
```powershell
.\venv\Scripts\Activate.ps1
helix
```

### Command Line Options

```
helix                      # Full application (UI + backend)
helix --server-only        # API server only
helix --ui-only            # UI only (requires running server)
helix --port 8421          # Custom API port
helix --debug              # Enable debug mode
```

### API Server

The backend API runs at `http://127.0.0.1:8420` by default.

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

API documentation available at: `http://127.0.0.1:8420/docs`

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

## Data Locations

| Data Type | Location |
|-----------|----------|
| Models | `%LOCALAPPDATA%\HelixTransvoicer\models` |
| Cache | `%LOCALAPPDATA%\HelixTransvoicer\cache` |
| Exports | `%USERPROFILE%\Documents\HelixTransvoicer` |

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

## Project Structure

```
Helix-Transvoicer/
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
├── setup.ps1              # Windows setup script
├── run.bat                # Windows launcher
├── run-server.bat         # Server-only launcher
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Troubleshooting

### "Python not found"
Install Python 3.10+ from [python.org](https://www.python.org/downloads/). Check "Add Python to PATH" during installation.

### "CUDA not available" (with NVIDIA GPU)
1. Install latest NVIDIA drivers from [nvidia.com](https://www.nvidia.com/drivers)
2. Reinstall PyTorch with CUDA:
   ```powershell
   pip uninstall torch torchaudio
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### UI appears blurry
Right-click the desktop shortcut → Properties → Compatibility → "Change high DPI settings" → Enable "Override high DPI scaling behavior"

### Antivirus blocks execution
Add the Helix-Transvoicer folder to your antivirus exclusions.

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
