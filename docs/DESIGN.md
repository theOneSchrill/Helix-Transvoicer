# Helix Transvoicer - Conceptual Design Document

## System Overview

Helix Transvoicer is a studio-grade voice processing application for professional audio production. It provides voice conversion, custom voice model training, and advanced text-to-speech capabilities with complete local operation.

### Core Philosophy
- **Local-first**: All processing happens on-device. No cloud dependencies.
- **Expert-oriented**: Full control, no hidden automation, complete transparency.
- **Industrial design**: Technical precision over aesthetic flourish.
- **Modular architecture**: Clean separation of concerns, extensible pipelines.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HELIX TRANSVOICER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         FRONTEND (UI Layer)                          │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │    │
│  │  │  Voice   │ │  Model   │ │ Emotion  │ │   TTS    │ │  Model   │   │    │
│  │  │Converter │ │ Builder  │ │Dashboard │ │  Studio  │ │ Library  │   │    │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │    │
│  └───────┼────────────┼────────────┼────────────┼────────────┼─────────┘    │
│          │            │            │            │            │              │
│          └────────────┴────────────┼────────────┴────────────┘              │
│                                    │                                        │
│                            ┌───────▼───────┐                                │
│                            │   REST API    │                                │
│                            │   (FastAPI)   │                                │
│                            └───────┬───────┘                                │
│                                    │                                        │
│  ┌─────────────────────────────────┼─────────────────────────────────────┐  │
│  │                         BACKEND (Processing Layer)                     │  │
│  │                                 │                                      │  │
│  │  ┌──────────────────────────────┼──────────────────────────────────┐  │  │
│  │  │                      Core Services                               │  │  │
│  │  │  ┌─────────────┐  ┌─────────┴─────────┐  ┌─────────────┐        │  │  │
│  │  │  │   Audio     │  │      Model        │  │   Voice     │        │  │  │
│  │  │  │ Processor   │  │    Manager        │  │  Converter  │        │  │  │
│  │  │  └─────────────┘  └───────────────────┘  └─────────────┘        │  │  │
│  │  │  ┌─────────────┐  ┌───────────────────┐  ┌─────────────┐        │  │  │
│  │  │  │   TTS       │  │     Emotion       │  │   Model     │        │  │  │
│  │  │  │  Engine     │  │    Analyzer       │  │  Trainer    │        │  │  │
│  │  │  └─────────────┘  └───────────────────┘  └─────────────┘        │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                      ML Pipeline                                 │  │  │
│  │  │  ┌─────────────┐  ┌───────────────────┐  ┌─────────────┐        │  │  │
│  │  │  │  Feature    │  │    Neural Voice   │  │  Vocoder    │        │  │  │
│  │  │  │ Extractor   │  │     Encoder       │  │  (HiFi-GAN) │        │  │  │
│  │  │  └─────────────┘  └───────────────────┘  └─────────────┘        │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                      Device Layer                                │  │  │
│  │  │         ┌─────────────────┐    ┌─────────────────┐              │  │  │
│  │  │         │   GPU (CUDA)    │    │   CPU Fallback  │              │  │  │
│  │  │         │   Accelerated   │    │   Processing    │              │  │  │
│  │  │         └─────────────────┘    └─────────────────┘              │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         DATA Layer                                   │    │
│  │  ┌─────────────┐  ┌───────────────────┐  ┌─────────────────────┐    │    │
│  │  │   Model     │  │    Audio Cache    │  │   Configuration     │    │    │
│  │  │  Storage    │  │    & Workspace    │  │    & Metadata       │    │    │
│  │  └─────────────┘  └───────────────────┘  └─────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Processing Pipelines

### 1. Audio Preprocessing Pipeline

```
Input Audio (WAV/MP3/FLAC)
         │
         ▼
┌─────────────────┐
│  Format Decode  │  ← librosa/soundfile
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Resample to   │  ← Target: 22050 Hz or 44100 Hz
│   Standard SR   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Denoise      │  ← Spectral gating / RNNoise
│   (Optional)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Silence Trim    │  ← VAD-based trimming
│  & Normalize    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Alignment     │  ← Forced alignment for training
│   (if needed)   │
└────────┬────────┘
         │
         ▼
Processed Audio Tensor
```

### 2. Voice Conversion Pipeline

```
Source Audio                    Target Voice Model
     │                                  │
     ▼                                  │
┌────────────────┐                      │
│ Preprocessing  │                      │
└───────┬────────┘                      │
        │                               │
        ▼                               │
┌────────────────┐                      │
│   Feature      │                      │
│  Extraction    │                      │
│  (Mel-spec,    │                      │
│   F0, PPG)     │                      │
└───────┬────────┘                      │
        │                               │
        ▼                               ▼
┌─────────────────────────────────────────┐
│         Voice Encoder Network           │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │   Content    │  │    Speaker      │  │
│  │   Encoder    │  │   Embedding     │◄─┼── Target Voice
│  │   (PPG)      │  │    Injection    │  │
│  └──────┬───────┘  └────────┬────────┘  │
│         │                   │           │
│         └─────────┬─────────┘           │
│                   │                     │
│                   ▼                     │
│         ┌─────────────────┐             │
│         │    Decoder      │             │
│         │  (Mel-spec)     │             │
│         └────────┬────────┘             │
└──────────────────┼──────────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │    Vocoder      │
         │   (HiFi-GAN)    │
         └────────┬────────┘
                  │
                  ▼
         Converted Audio Output
```

### 3. Model Training Pipeline

```
Training Samples (WAV/MP3/FLAC)
              │
              ▼
     ┌────────────────┐
     │ Preprocessing  │
     │    Batch       │
     └───────┬────────┘
              │
              ▼
     ┌────────────────┐
     │   Feature      │
     │  Extraction    │
     │ (per sample)   │
     └───────┬────────┘
              │
              ▼
     ┌────────────────┐
     │   Emotion      │
     │   Detection    │◄── Emotion labels extracted
     │   & Labeling   │
     └───────┬────────┘
              │
              ▼
     ┌────────────────┐
     │   Dataset      │
     │  Construction  │
     └───────┬────────┘
              │
              ▼
┌─────────────────────────────────────┐
│        Training Loop                │
│  ┌─────────────────────────────┐    │
│  │   Speaker Encoder Training  │    │
│  │   (Contrastive Learning)    │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │   Voice Decoder Training    │    │
│  │   (Reconstruction Loss)     │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │   Vocoder Fine-tuning       │    │
│  │   (Optional)                │    │
│  └─────────────────────────────┘    │
└──────────────────┬──────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │  Model Export  │
          │  & Metadata    │
          └───────┬────────┘
                  │
                  ▼
          Voice Model (.helix)
```

### 4. Incremental Learning Pipeline

```
Existing Model                New Samples
      │                            │
      ▼                            ▼
┌───────────┐              ┌───────────────┐
│  Load     │              │ Preprocess    │
│  Weights  │              │ & Extract     │
└─────┬─────┘              └───────┬───────┘
      │                            │
      └──────────┬─────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │  Merge Dataset │
        │  (old + new)   │
        └───────┬────────┘
                │
                ▼
        ┌────────────────┐
        │  Fine-tune     │
        │  (Low LR,      │
        │   Few Epochs)  │
        └───────┬────────┘
                │
                ▼
        ┌────────────────┐
        │  Validation    │
        │  (Quality      │
        │   Check)       │
        └───────┬────────┘
                │
                ▼
        Updated Model (v+1)
```

### 5. TTS Pipeline

```
Text Input                    Voice Model
     │                             │
     ▼                             │
┌────────────────┐                 │
│  Text          │                 │
│  Normalization │                 │
└───────┬────────┘                 │
        │                          │
        ▼                          │
┌────────────────┐                 │
│   Phoneme      │                 │
│  Conversion    │                 │
│  (G2P)         │                 │
└───────┬────────┘                 │
        │                          │
        ▼                          ▼
┌─────────────────────────────────────┐
│         TTS Synthesizer             │
│  ┌─────────────────────────────┐    │
│  │   Duration Predictor        │    │
│  │   (speed control)           │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │   Pitch Predictor           │    │
│  │   (pitch control)           │    │
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │   Acoustic Model            │    │
│  │   + Speaker Embedding       │◄───┼── Voice Model
│  │   + Emotion Embedding       │◄───┼── Emotion Bias
│  └─────────────────────────────┘    │
└──────────────────┬──────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │    Vocoder      │
         │   (HiFi-GAN)    │
         └────────┬────────┘
                  │
                  ▼
          Synthesized Audio
```

---

## UI Structure

### Panel Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  HELIX TRANSVOICER                                    [─] [□] [×]           │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │  VOICE CONVERTER │ MODEL BUILDER │ EMOTION MAP │ TTS STUDIO │ LIBRARY │   │
│ └───────────────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │                                 │  │                                 │  │
│  │         PRIMARY PANEL           │  │        SECONDARY PANEL          │  │
│  │                                 │  │                                 │  │
│  │    (Context-dependent main      │  │    (Supporting controls,        │  │
│  │     workspace area)             │  │     settings, details)          │  │
│  │                                 │  │                                 │  │
│  │                                 │  │                                 │  │
│  │                                 │  │                                 │  │
│  │                                 │  │                                 │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                           STATUS BAR                                  │  │
│  │  [GPU: CUDA 12.1 | VRAM: 4.2/8GB]  [Model: voice_alpha_v3]  [Ready] │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1. Voice Converter Panel

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  VOICE CONVERTER                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SOURCE AUDIO                                                               │
│  ┌────────────────────────────────────────────────────────┐                │
│  │  ████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  00:42 / 02:15 │
│  │  [▶ Play]  [⏹ Stop]  [📂 Load File]  [🔊 Preview]      │                │
│  └────────────────────────────────────────────────────────┘                │
│                                                                             │
│  TARGET VOICE                          CONVERSION SETTINGS                  │
│  ┌──────────────────────┐              ┌────────────────────────────────┐  │
│  │                      │              │                                │  │
│  │  [Voice Model ▼]     │              │  Pitch Shift:    [-12]──●──[+12] │
│  │                      │              │  Formant:        [-1.0]──●──[+1.0]│
│  │  ● voice_alpha_v3    │              │  Smoothing:      [0]────●────[1] │
│  │  ○ voice_beta_v1     │              │  Crossfade:      [10ms]──●──[50ms]│
│  │  ○ narrator_deep     │              │                                │  │
│  │                      │              │  □ Preserve breath sounds       │  │
│  │  [Refresh List]      │              │  □ Preserve background noise    │  │
│  └──────────────────────┘              │  ☑ Normalize output             │  │
│                                        └────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         [▶ CONVERT]                                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  OUTPUT                                                                     │
│  ┌────────────────────────────────────────────────────────┐                │
│  │  Processing: ████████████████░░░░░░░░░░░░░░░  67%      │                │
│  │  Stage: Voice encoding (2/4)                           │                │
│  └────────────────────────────────────────────────────────┘                │
│  [💾 Save As...]  [📋 Copy to Clipboard]  [↻ Reset]                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Model Builder Panel

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MODEL BUILDER                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROJECT: New Voice Model                    [📂 Load Project] [💾 Save]   │
│                                                                             │
│  TRAINING SAMPLES                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  #   Filename              Duration   Quality   Emotion    Status    │  │
│  │  ──────────────────────────────────────────────────────────────────  │  │
│  │  1   sample_001.wav        00:12      ████░     Neutral    ✓ Ready   │  │
│  │  2   sample_002.wav        00:08      █████     Happy      ✓ Ready   │  │
│  │  3   sample_003.mp3        00:15      ███░░     Sad        ⚠ Noisy   │  │
│  │  4   sample_004.flac       00:22      █████     Excited    ✓ Ready   │  │
│  │  5   sample_005.wav        00:10      ████░     Calm       ✓ Ready   │  │
│  │                                                                       │  │
│  │  [+ Add Files]  [🗑 Remove Selected]  [🔄 Re-analyze]                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  TRAINING CONFIGURATION                    EMOTION COVERAGE                 │
│  ┌────────────────────────────┐           ┌────────────────────────────┐   │
│  │  Model Name: [voice_new  ] │           │  Neutral:   ████████░░ 80% │   │
│  │  Base Model: [Default ▼  ] │           │  Happy:     ██████░░░░ 60% │   │
│  │  Epochs:     [100       ]  │           │  Sad:       ████░░░░░░ 40% │   │
│  │  Batch Size: [16        ]  │           │  Angry:     ░░░░░░░░░░  0% │   │
│  │  Learning Rate: [0.0001 ]  │           │  Calm:      ██████████ 100%│   │
│  │                            │           │  Excited:   ████████░░ 80% │   │
│  │  ☑ Auto-denoise samples    │           │  ──────────────────────────│   │
│  │  ☑ Augment training data   │           │  ⚠ Missing: Angry, Fear    │   │
│  │  □ Fine-tune vocoder       │           └────────────────────────────┘   │
│  └────────────────────────────┘                                            │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  [▶ START TRAINING]          Est. Time: ~45 min  |  GPU: Available   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  TRAINING PROGRESS                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Epoch: 34/100  |  Loss: 0.0234  |  ETA: 28 min                       │  │
│  │  ████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░  34%       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Emotion Coverage Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  EMOTION MAP                                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MODEL: voice_alpha_v3 (v3.2.1)                     [Change Model ▼]       │
│                                                                             │
│  EMOTION SPECTRUM ANALYSIS                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │           High Arousal                                                │  │
│  │                ▲                                                      │  │
│  │                │     ┌─────────┐                                      │  │
│  │    ┌───────┐   │     │ EXCITED │ ████████ 85%                         │  │
│  │    │ ANGRY │   │     └─────────┘                                      │  │
│  │    │  ░░░  │ 12%         │                                            │  │
│  │    └───────┘   │         │                                            │  │
│  │                │         │                                            │  │
│  │  Negative ─────┼─────────┼────────── Positive                         │  │
│  │  Valence       │         │           Valence                          │  │
│  │                │         │                                            │  │
│  │    ┌───────┐   │     ┌─────────┐                                      │  │
│  │    │  SAD  │   │     │  HAPPY  │ ██████░░ 72%                         │  │
│  │    │ ████  │ 45%     └─────────┘                                      │  │
│  │    └───────┘   │                                                      │  │
│  │                │     ┌─────────┐                                      │  │
│  │                │     │  CALM   │ ██████████ 100%                      │  │
│  │                ▼     └─────────┘                                      │  │
│  │           Low Arousal                                                 │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  DETAILED COVERAGE                         RECOMMENDATIONS                  │
│  ┌────────────────────────────┐           ┌────────────────────────────┐   │
│  │  Emotion     Coverage Conf │           │                            │   │
│  │  ────────────────────────  │           │  ⚠ CRITICAL GAPS:          │   │
│  │  Neutral     ████████ 92%  │           │    • Angry (12% coverage)  │   │
│  │  Happy       ██████░░ 72%  │           │    • Fear (0% coverage)    │   │
│  │  Sad         ████░░░░ 45%  │           │                            │   │
│  │  Angry       █░░░░░░░ 12%  │           │  📋 SUGGESTED SAMPLES:     │   │
│  │  Fear        ░░░░░░░░  0%  │           │    • Record 3-5 angry      │   │
│  │  Surprise    ███░░░░░ 38%  │           │      utterances (15-30s)   │   │
│  │  Disgust     ██░░░░░░ 25%  │           │    • Record 2-3 fearful    │   │
│  │  Calm        ████████100%  │           │      expressions (10-20s)  │   │
│  │  Excited     ████████ 85%  │           │                            │   │
│  └────────────────────────────┘           └────────────────────────────┘   │
│                                                                             │
│  [📊 Export Report]  [🔄 Re-analyze]  [+ Add Samples to Model]             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. TTS Studio Panel

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TTS STUDIO                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TEXT INPUT                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  The quick brown fox jumps over the lazy dog. This sentence contains │  │
│  │  every letter of the alphabet and is perfect for testing voice       │  │
│  │  synthesis quality.                                                   │  │
│  │                                                                       │  │
│  │  Character count: 156  |  Est. duration: ~8 seconds                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  VOICE SELECTION                           VOICE PARAMETERS                 │
│  ┌────────────────────────────┐           ┌────────────────────────────┐   │
│  │                            │           │                            │   │
│  │  ● voice_alpha_v3          │           │  Speed:                    │   │
│  │  ○ voice_beta_v1           │           │  [0.5x]────●────[2.0x]     │   │
│  │  ○ narrator_deep           │           │           1.0x             │   │
│  │  ○ assistant_warm          │           │                            │   │
│  │                            │           │  Pitch:                    │   │
│  │                            │           │  [-12]─────●─────[+12]     │   │
│  │  [🔄 Refresh]              │           │           0 st             │   │
│  └────────────────────────────┘           │                            │   │
│                                           │  Intensity:                │   │
│  EMOTION CONTROL                          │  [Soft]────●────[Intense]  │   │
│  ┌────────────────────────────┐           │          Normal            │   │
│  │                            │           │                            │   │
│  │  Primary Emotion:          │           │  Variance:                 │   │
│  │  [Neutral        ▼]        │           │  [Low]─────●─────[High]    │   │
│  │                            │           │          Medium            │   │
│  │  Emotion Strength:         │           │                            │   │
│  │  [0%]─────────●───[100%]   │           │  □ Add natural pauses      │   │
│  │              65%           │           │  ☑ Breathing sounds        │   │
│  │                            │           │  □ Whisper mode            │   │
│  │  Secondary Blend:          │           │                            │   │
│  │  [None           ▼]  0%    │           └────────────────────────────┘   │
│  └────────────────────────────┘                                            │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  [▶ SYNTHESIZE]  [🔊 Preview]  [⏹ Stop]                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  OUTPUT WAVEFORM                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  ▁▂▄▆█▇▅▃▂▁▁▂▃▅▇█▆▄▂▁▁▂▄▆█▇▅▃▂▁▁▂▃▅▇█▆▄▂▁▁▂▄▆█▇▅▃▂▁             │  │
│  │  [▶]──────────────●───────────────────────────────── 00:03 / 00:08  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  [💾 Save Audio]  [📋 Save to History]  [📊 Spectrogram View]              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5. Model Library Panel

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MODEL LIBRARY                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LOCAL MODELS                              [📂 Open Models Folder]          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  🔊 voice_alpha_v3                                    [★ Default]    │  │
│  │  ├─ Version: 3.2.1  |  Size: 245 MB  |  Created: 2024-01-15          │  │
│  │  ├─ Samples: 47  |  Duration: 12m 34s  |  Quality: ████░             │  │
│  │  ├─ Emotions: Neutral, Happy, Sad, Calm, Excited                     │  │
│  │  └─ [Load] [Edit] [Duplicate] [Export] [Delete]                      │  │
│  │                                                                       │  │
│  │  ──────────────────────────────────────────────────────────────────  │  │
│  │                                                                       │  │
│  │  🔊 voice_beta_v1                                                    │  │
│  │  ├─ Version: 1.0.0  |  Size: 198 MB  |  Created: 2024-01-10          │  │
│  │  ├─ Samples: 23  |  Duration: 6m 45s  |  Quality: ███░░              │  │
│  │  ├─ Emotions: Neutral, Calm                                          │  │
│  │  └─ [Load] [Edit] [Duplicate] [Export] [Delete]                      │  │
│  │                                                                       │  │
│  │  ──────────────────────────────────────────────────────────────────  │  │
│  │                                                                       │  │
│  │  🔊 narrator_deep                                                    │  │
│  │  ├─ Version: 2.1.0  |  Size: 312 MB  |  Created: 2024-01-08          │  │
│  │  ├─ Samples: 89  |  Duration: 28m 12s  |  Quality: █████             │  │
│  │  ├─ Emotions: All covered (92% avg)                                  │  │
│  │  └─ [Load] [Edit] [Duplicate] [Export] [Delete]                      │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  MODEL DETAILS                             QUICK ACTIONS                    │
│  ┌────────────────────────────┐           ┌────────────────────────────┐   │
│  │  Selected: voice_alpha_v3  │           │                            │   │
│  │                            │           │  [+ New Model]             │   │
│  │  Training History:         │           │                            │   │
│  │  • v3.2.1 - Added 5 samples│           │  [📥 Import Model]         │   │
│  │  • v3.2.0 - Emotion update │           │                            │   │
│  │  • v3.1.0 - Quality fix    │           │  [🔄 Refresh Library]      │   │
│  │  • v3.0.0 - Initial train  │           │                            │   │
│  │                            │           │  [⚙ Model Settings]        │   │
│  │  Metadata:                 │           │                            │   │
│  │  • Base: default_encoder   │           └────────────────────────────┘   │
│  │  • Vocoder: hifi_gan_v2    │                                            │
│  │  • Sample Rate: 22050 Hz   │                                            │
│  └────────────────────────────┘                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Lifecycle

### Model File Structure

```
models/
├── voice_alpha_v3/
│   ├── model.helix              # Main model weights
│   ├── config.json              # Model configuration
│   ├── metadata.json            # Model metadata & history
│   ├── speaker_embedding.npy    # Speaker embedding vector
│   ├── vocoder/                 # Vocoder weights (optional fine-tuned)
│   │   └── generator.pt
│   ├── samples/                 # Original training samples (optional)
│   │   ├── sample_001.wav
│   │   └── ...
│   └── versions/                # Version history
│       ├── v3.1.0/
│       └── v3.0.0/
```

### Metadata Schema

```json
{
  "model_id": "voice_alpha_v3",
  "version": "3.2.1",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T14:22:00Z",
  "base_model": "default_encoder_v2",
  "vocoder": "hifi_gan_v2",
  "sample_rate": 22050,
  "training": {
    "total_samples": 47,
    "total_duration_seconds": 754,
    "epochs_trained": 150,
    "final_loss": 0.0189,
    "device": "cuda:0"
  },
  "emotion_coverage": {
    "neutral": {"coverage": 0.92, "confidence": 0.95, "sample_count": 15},
    "happy": {"coverage": 0.72, "confidence": 0.88, "sample_count": 8},
    "sad": {"coverage": 0.45, "confidence": 0.82, "sample_count": 5},
    "angry": {"coverage": 0.12, "confidence": 0.65, "sample_count": 2},
    "fear": {"coverage": 0.0, "confidence": 0.0, "sample_count": 0},
    "surprise": {"coverage": 0.38, "confidence": 0.78, "sample_count": 4},
    "disgust": {"coverage": 0.25, "confidence": 0.72, "sample_count": 3},
    "calm": {"coverage": 1.0, "confidence": 0.97, "sample_count": 12},
    "excited": {"coverage": 0.85, "confidence": 0.91, "sample_count": 10}
  },
  "quality_metrics": {
    "overall_quality": 0.87,
    "clarity": 0.91,
    "naturalness": 0.84,
    "speaker_similarity": 0.89
  },
  "version_history": [
    {"version": "3.2.1", "date": "2024-01-15", "changes": "Added 5 new samples"},
    {"version": "3.2.0", "date": "2024-01-12", "changes": "Emotion coverage update"},
    {"version": "3.1.0", "date": "2024-01-10", "changes": "Quality improvements"}
  ]
}
```

### Model States

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   EMPTY     │───▶│  TRAINING   │───▶│   READY     │───▶│  UPDATING   │
│  (No data)  │    │ (Learning)  │    │ (Usable)    │    │(Fine-tuning)│
└─────────────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
                          │                  │                  │
                          │                  │                  │
                          ▼                  ▼                  ▼
                   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                   │   FAILED    │    │  ARCHIVED   │    │   READY     │
                   │  (Error)    │    │ (Versioned) │    │  (v+1)      │
                   └─────────────┘    └─────────────┘    └─────────────┘
```

---

## Emotion Analysis System

### Emotion Detection Architecture

```
Audio Sample
     │
     ▼
┌────────────────┐
│   Feature      │
│  Extraction    │
│  (MFCC, F0,    │
│   energy, etc) │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│   Emotion      │
│  Classifier    │
│  (Pre-trained  │
│   CNN/LSTM)    │
└───────┬────────┘
        │
        ▼
┌────────────────────────────────────┐
│  Emotion Probability Distribution  │
│  ┌────────────────────────────┐    │
│  │  Neutral:   0.15           │    │
│  │  Happy:     0.62  ◄── Primary    │
│  │  Sad:       0.08           │    │
│  │  Angry:     0.02           │    │
│  │  Fear:      0.01           │    │
│  │  Surprise:  0.05           │    │
│  │  Disgust:   0.01           │    │
│  │  Calm:      0.04           │    │
│  │  Excited:   0.02           │    │
│  └────────────────────────────┘    │
└────────────────────────────────────┘
```

### Coverage Calculation

```python
# Coverage calculation per emotion
coverage = {
    "detected_samples": count of samples where emotion is primary or secondary,
    "coverage_score": weighted sum of emotion probabilities across all samples,
    "confidence": average confidence when emotion is detected,
    "quality": assessment of sample quality for that emotion
}

# Overall coverage health
health_score = weighted_average(all_emotion_coverages)
gaps = emotions where coverage < threshold (e.g., 30%)
recommendations = generate_sample_suggestions(gaps)
```

### Emotion Categories

| Emotion | Valence | Arousal | Description |
|---------|---------|---------|-------------|
| Neutral | 0 | 0 | Baseline, no emotional content |
| Happy | + | + | Joyful, pleased, content |
| Sad | - | - | Sorrowful, melancholic |
| Angry | - | + | Frustrated, irritated, enraged |
| Fear | - | + | Scared, anxious, worried |
| Surprise | 0 | + | Startled, amazed |
| Disgust | - | 0 | Repulsed, averse |
| Calm | 0 | - | Relaxed, peaceful, serene |
| Excited | + | + | Enthusiastic, energetic |

---

## Technology Stack

### Backend
- **Python 3.10+** - Core language
- **PyTorch** - Neural network framework
- **torchaudio** - Audio processing
- **librosa** - Audio analysis
- **FastAPI** - REST API server
- **Pydantic** - Data validation
- **SQLite** - Local metadata storage

### ML Models
- **HiFi-GAN** - Neural vocoder
- **Conformer** - Voice encoder
- **SpeechBrain** - Emotion recognition
- **Montreal Forced Aligner** - Alignment (optional)

### Frontend
- **Python** - UI framework
- **CustomTkinter** / **DearPyGUI** - Modern UI toolkit
- **Matplotlib** - Waveform visualization
- **Pillow** - Image processing

### Device Support
- **CUDA** - NVIDIA GPU acceleration
- **MPS** - Apple Silicon acceleration
- **CPU** - Fallback processing

---

## API Endpoints

### Audio Processing
```
POST /api/audio/preprocess      - Preprocess audio file
POST /api/audio/denoise         - Apply denoising
POST /api/audio/analyze         - Analyze audio properties
```

### Voice Conversion
```
POST /api/convert/voice         - Convert voice in audio
GET  /api/convert/status/{id}   - Get conversion progress
POST /api/convert/cancel/{id}   - Cancel conversion
```

### Model Management
```
GET  /api/models                - List all models
GET  /api/models/{id}           - Get model details
POST /api/models                - Create new model
PUT  /api/models/{id}           - Update model
DELETE /api/models/{id}         - Delete model
POST /api/models/{id}/train     - Start training
POST /api/models/{id}/update    - Incremental update
GET  /api/models/{id}/emotions  - Get emotion coverage
```

### TTS
```
POST /api/tts/synthesize        - Synthesize speech
GET  /api/tts/voices            - List available voices
POST /api/tts/preview           - Quick preview generation
```

### System
```
GET  /api/system/status         - System status (GPU, memory)
GET  /api/system/device         - Device information
POST /api/system/settings       - Update settings
```

---

## Design Principles Implementation

### 1. Studio-Grade
- Professional audio quality (24-bit, high sample rates)
- Accurate metering and visualization
- Non-destructive workflow
- Comprehensive undo/redo

### 2. No Social Features
- No accounts, no cloud sync
- No sharing functionality
- No telemetry or analytics
- Complete privacy

### 3. No Gimmicks
- No AI suggestions or auto-complete
- No "magic" buttons
- Every action explicit
- Full parameter control

### 4. Technology Invisible
- Clean, minimal interface
- No exposed technical jargon
- Sensible defaults
- Progressive disclosure

### 5. Control Explicit
- All settings accessible
- No hidden automation
- Clear cause-and-effect
- Predictable behavior

### 6. Industrial Design
- Dark, focused UI
- Monospace typography for data
- Grid-based layout
- High information density

### 7. Expert-Oriented
- Keyboard shortcuts
- Batch operations
- Scripting support
- Advanced configuration

---

## File Organization

```
helix-transvoicer/
├── src/
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application entry
│   │   ├── api/                 # API routes
│   │   │   ├── __init__.py
│   │   │   ├── audio.py
│   │   │   ├── convert.py
│   │   │   ├── models.py
│   │   │   ├── tts.py
│   │   │   └── system.py
│   │   ├── core/                # Core processing
│   │   │   ├── __init__.py
│   │   │   ├── audio_processor.py
│   │   │   ├── voice_converter.py
│   │   │   ├── model_trainer.py
│   │   │   ├── tts_engine.py
│   │   │   └── emotion_analyzer.py
│   │   ├── models/              # ML model definitions
│   │   │   ├── __init__.py
│   │   │   ├── encoder.py
│   │   │   ├── decoder.py
│   │   │   ├── vocoder.py
│   │   │   └── emotion.py
│   │   ├── services/            # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── model_manager.py
│   │   │   ├── job_queue.py
│   │   │   └── storage.py
│   │   └── utils/               # Utilities
│   │       ├── __init__.py
│   │       ├── audio.py
│   │       ├── device.py
│   │       └── config.py
│   │
│   └── frontend/
│       ├── __init__.py
│       ├── main.py              # UI application entry
│       ├── app.py               # Main application class
│       ├── panels/              # UI panels
│       │   ├── __init__.py
│       │   ├── converter.py
│       │   ├── builder.py
│       │   ├── emotions.py
│       │   ├── tts.py
│       │   └── library.py
│       ├── components/          # Reusable UI components
│       │   ├── __init__.py
│       │   ├── waveform.py
│       │   ├── progress.py
│       │   ├── controls.py
│       │   └── dialogs.py
│       ├── styles/              # UI styling
│       │   ├── __init__.py
│       │   ├── theme.py
│       │   └── colors.py
│       └── utils/               # UI utilities
│           ├── __init__.py
│           └── api_client.py
│
├── data/
│   ├── models/                  # Voice models storage
│   ├── cache/                   # Temporary files
│   └── exports/                 # Exported audio
│
├── config/
│   ├── default.yaml             # Default configuration
│   └── logging.yaml             # Logging configuration
│
├── tests/
│   ├── backend/
│   └── frontend/
│
├── docs/
│   └── DESIGN.md                # This document
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Summary

Helix Transvoicer is designed as a professional, local-first voice processing application that prioritizes:

1. **Complete Control** - Every parameter exposed, no hidden automation
2. **Transparency** - Clear processing states, visible progress, honest feedback
3. **Quality** - Studio-grade audio processing with GPU acceleration
4. **Privacy** - 100% local operation, no cloud dependencies
5. **Efficiency** - Incremental learning, fast model switching, optimized pipelines

The architecture separates concerns cleanly between frontend (UI), backend (API), and core processing (ML), allowing for future extensibility while maintaining a focused, expert-oriented experience.
