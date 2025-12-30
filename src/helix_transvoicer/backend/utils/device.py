"""
Helix Transvoicer - Device management for CPU/GPU processing.
"""

import logging
import sys
from dataclasses import dataclass
from typing import Optional

import torch

from helix_transvoicer.backend.utils.config import get_settings

logger = logging.getLogger("helix.device")


@dataclass
class DeviceInfo:
    """Information about a compute device."""

    name: str
    type: str  # "cuda", "mps", "cpu"
    index: int
    total_memory: Optional[int] = None  # bytes
    available_memory: Optional[int] = None  # bytes


class DeviceManager:
    """Manages compute device selection and monitoring."""

    def __init__(self):
        self.settings = get_settings()
        self._device: Optional[torch.device] = None
        self._device_info: Optional[DeviceInfo] = None
        self._initialize_device()

    def _initialize_device(self) -> None:
        """Initialize the compute device."""
        # Log system info for debugging
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA compiled: {torch.version.cuda or 'No'}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        else:
            # Log why CUDA might not be available
            if not torch.version.cuda:
                logger.warning("PyTorch was NOT compiled with CUDA support!")
                logger.warning("Install CUDA version: pip install torch --index-url https://download.pytorch.org/whl/cu118")

        if self.settings.force_cpu:
            self._device = torch.device("cpu")
            self._device_info = DeviceInfo(
                name="CPU",
                type="cpu",
                index=0,
            )
            logger.info("Using CPU (forced by settings)")
            return

        # Try CUDA first
        if torch.cuda.is_available():
            device_index = 0
            self._device = torch.device(f"cuda:{device_index}")

            # Set memory fraction
            if self.settings.gpu_memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(
                    self.settings.gpu_memory_fraction,
                    device_index,
                )

            props = torch.cuda.get_device_properties(device_index)
            self._device_info = DeviceInfo(
                name=props.name,
                type="cuda",
                index=device_index,
                total_memory=props.total_memory,
                available_memory=torch.cuda.memory_reserved(device_index),
            )
            logger.info(f"Using CUDA GPU: {props.name} ({props.total_memory // 1024 // 1024} MB)")
            return

        # Try MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
            self._device_info = DeviceInfo(
                name="Apple Silicon",
                type="mps",
                index=0,
            )
            logger.info("Using MPS (Apple Silicon)")
            return

        # Fall back to CPU
        self._device = torch.device("cpu")
        self._device_info = DeviceInfo(
            name="CPU",
            type="cpu",
            index=0,
        )
        logger.info("Using CPU - no GPU acceleration available")

    @property
    def device(self) -> torch.device:
        """Get the current compute device."""
        return self._device

    @property
    def device_type(self) -> str:
        """Get the device type string."""
        return self._device_info.type

    @property
    def device_name(self) -> str:
        """Get the device name."""
        return self._device_info.name

    @property
    def is_gpu(self) -> bool:
        """Check if using GPU acceleration."""
        return self._device_info.type in ("cuda", "mps")

    def get_memory_info(self) -> dict:
        """Get current memory information."""
        info = {
            "device": self.device_name,
            "type": self.device_type,
        }

        if self.device_type == "cuda":
            info["total_memory"] = torch.cuda.get_device_properties(0).total_memory
            info["allocated_memory"] = torch.cuda.memory_allocated(0)
            info["reserved_memory"] = torch.cuda.memory_reserved(0)
            info["free_memory"] = info["total_memory"] - info["allocated_memory"]

        return info

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the compute device."""
        return tensor.to(self._device)

    def empty_cache(self) -> None:
        """Clear GPU memory cache."""
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
        elif self.device_type == "mps":
            torch.mps.empty_cache()
