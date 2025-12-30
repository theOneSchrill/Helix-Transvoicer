"""
Helix Transvoicer - API client for frontend.
"""

import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import httpx


class APIClient:
    """
    HTTP client for communicating with the Helix backend API.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8420"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=60.0)

    def _url(self, path: str) -> str:
        """Build full URL."""
        return f"{self.base_url}{path}"

    # System endpoints

    def get_status(self) -> Dict:
        """Get system status."""
        try:
            response = self._client.get(self._url("/api/system/status"))
            response.raise_for_status()
            return response.json()
        except Exception:
            return {"status": "offline"}

    def get_device(self) -> Dict:
        """Get device information."""
        response = self._client.get(self._url("/api/system/device"))
        response.raise_for_status()
        return response.json()

    def list_jobs(self, status: Optional[str] = None) -> List[Dict]:
        """List background jobs."""
        params = {}
        if status:
            params["status"] = status
        response = self._client.get(self._url("/api/system/jobs"), params=params)
        response.raise_for_status()
        return response.json()

    def cancel_job(self, job_id: str) -> Dict:
        """Cancel a job."""
        response = self._client.post(self._url(f"/api/system/jobs/{job_id}/cancel"))
        response.raise_for_status()
        return response.json()

    # Model endpoints

    def list_models(self) -> List[Dict]:
        """List all voice models."""
        response = self._client.get(self._url("/api/models"))
        response.raise_for_status()
        return response.json()

    def get_model(self, model_id: str) -> Dict:
        """Get model details."""
        response = self._client.get(self._url(f"/api/models/{model_id}"))
        response.raise_for_status()
        return response.json()

    def create_model(self, model_id: str, name: Optional[str] = None) -> Dict:
        """Create a new model."""
        response = self._client.post(
            self._url("/api/models"),
            json={"model_id": model_id, "name": name},
        )
        response.raise_for_status()
        return response.json()

    def delete_model(self, model_id: str) -> Dict:
        """Delete a model."""
        response = self._client.delete(self._url(f"/api/models/{model_id}"))
        response.raise_for_status()
        return response.json()

    def train_model(
        self,
        model_id: str,
        audio_files: List[Path],
        epochs: int = 100,
        batch_size: int = 16,
    ) -> Dict:
        """Start model training."""
        files = [
            ("files", (f.name, open(f, "rb"), "audio/wav"))
            for f in audio_files
        ]

        try:
            response = self._client.post(
                self._url(f"/api/models/{model_id}/train"),
                files=files,
                params={
                    "epochs": epochs,
                    "batch_size": batch_size,
                },
                timeout=300.0,
            )
            response.raise_for_status()
            return response.json()
        finally:
            for _, (_, f, _) in files:
                f.close()

    def update_model(
        self,
        model_id: str,
        audio_files: List[Path],
        epochs: int = 20,
    ) -> Dict:
        """Incrementally update a model."""
        files = [
            ("files", (f.name, open(f, "rb"), "audio/wav"))
            for f in audio_files
        ]

        try:
            response = self._client.post(
                self._url(f"/api/models/{model_id}/update"),
                files=files,
                params={"epochs": epochs},
                timeout=300.0,
            )
            response.raise_for_status()
            return response.json()
        finally:
            for _, (_, f, _) in files:
                f.close()

    def get_emotion_coverage(self, model_id: str) -> Dict:
        """Get emotion coverage for a model."""
        response = self._client.get(self._url(f"/api/models/{model_id}/emotions"))
        response.raise_for_status()
        return response.json()

    def load_model(self, model_id: str) -> Dict:
        """Load model into memory."""
        response = self._client.post(self._url(f"/api/models/{model_id}/load"))
        response.raise_for_status()
        return response.json()

    def unload_model(self, model_id: str) -> Dict:
        """Unload model from memory."""
        response = self._client.post(self._url(f"/api/models/{model_id}/unload"))
        response.raise_for_status()
        return response.json()

    # Conversion endpoints

    def convert_voice(
        self,
        audio_file: Path,
        target_model_id: str,
        pitch_shift: float = 0.0,
        smoothing: float = 0.5,
    ) -> bytes:
        """Convert voice and return audio bytes."""
        with open(audio_file, "rb") as f:
            response = self._client.post(
                self._url("/api/convert/voice/download"),
                files={"file": (audio_file.name, f, "audio/wav")},
                params={
                    "target_model_id": target_model_id,
                    "pitch_shift": pitch_shift,
                    "smoothing": smoothing,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            return response.content

    def convert_voice_async(
        self,
        audio_file: Path,
        target_model_id: str,
        pitch_shift: float = 0.0,
    ) -> Dict:
        """Start async conversion job."""
        with open(audio_file, "rb") as f:
            response = self._client.post(
                self._url("/api/convert/voice/async"),
                files={"file": (audio_file.name, f, "audio/wav")},
                params={
                    "target_model_id": target_model_id,
                    "pitch_shift": pitch_shift,
                },
            )
            response.raise_for_status()
            return response.json()

    def get_conversion_status(self, job_id: str) -> Dict:
        """Get conversion job status."""
        response = self._client.get(self._url(f"/api/convert/status/{job_id}"))
        response.raise_for_status()
        return response.json()

    # Audio endpoints

    def analyze_audio(self, audio_file: Path) -> Dict:
        """Analyze audio file."""
        with open(audio_file, "rb") as f:
            response = self._client.post(
                self._url("/api/audio/analyze"),
                files={"file": (audio_file.name, f, "audio/wav")},
            )
            response.raise_for_status()
            return response.json()

    def preprocess_audio(
        self,
        audio_file: Path,
        denoise: bool = False,
        normalize: bool = True,
    ) -> Dict:
        """Preprocess audio file."""
        with open(audio_file, "rb") as f:
            response = self._client.post(
                self._url("/api/audio/preprocess"),
                files={"file": (audio_file.name, f, "audio/wav")},
                params={
                    "denoise": denoise,
                    "normalize": normalize,
                },
            )
            response.raise_for_status()
            return response.json()

    def denoise_audio(self, audio_file: Path, strength: float = 0.5) -> bytes:
        """Denoise audio and return bytes."""
        with open(audio_file, "rb") as f:
            response = self._client.post(
                self._url("/api/audio/denoise"),
                files={"file": (audio_file.name, f, "audio/wav")},
                params={"strength": strength},
            )
            response.raise_for_status()
            return response.content

    # TTS endpoints

    def synthesize_speech(
        self,
        text: str,
        voice_model_id: str,
        speed: float = 1.0,
        pitch: float = 0.0,
        emotion: str = "neutral",
        emotion_strength: float = 0.5,
    ) -> bytes:
        """Synthesize speech and return audio bytes."""
        response = self._client.post(
            self._url("/api/tts/synthesize/download"),
            json={
                "text": text,
                "voice_model_id": voice_model_id,
                "speed": speed,
                "pitch": pitch,
                "emotion": emotion,
                "emotion_strength": emotion_strength,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.content

    def preview_tts(
        self,
        text: str,
        voice_model_id: str,
        emotion: str = "neutral",
    ) -> bytes:
        """Quick TTS preview."""
        response = self._client.post(
            self._url("/api/tts/preview"),
            params={
                "text": text[:200],
                "voice_model_id": voice_model_id,
                "emotion": emotion,
            },
        )
        response.raise_for_status()
        return response.content

    def list_voices(self) -> List[Dict]:
        """List available TTS voices."""
        response = self._client.get(self._url("/api/tts/voices"))
        response.raise_for_status()
        return response.json()

    def list_emotions(self) -> Dict:
        """List available emotions."""
        response = self._client.get(self._url("/api/tts/emotions"))
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the client."""
        self._client.close()
