"""
Helix Transvoicer - Waveform display component.
"""

import customtkinter as ctk
from typing import Optional
import numpy as np

from helix_transvoicer.frontend.styles.theme import HelixTheme


class WaveformDisplay(ctk.CTkCanvas):
    """
    Audio waveform visualization component.
    """

    def __init__(
        self,
        parent,
        width: int = 600,
        height: int = 100,
        **kwargs,
    ):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=HelixTheme.COLORS["bg_tertiary"],
            highlightthickness=0,
            **kwargs,
        )

        self._width = width
        self._height = height
        self._audio_data: Optional[np.ndarray] = None
        self._playhead_position: float = 0.0

        self._draw_empty()

    def _draw_empty(self):
        """Draw empty waveform placeholder."""
        self.delete("all")

        # Draw center line
        center_y = self._height // 2
        self.create_line(
            0, center_y, self._width, center_y,
            fill=HelixTheme.COLORS["border"],
            width=1,
        )

        # Draw placeholder text
        self.create_text(
            self._width // 2, center_y,
            text="No audio loaded",
            fill=HelixTheme.COLORS["text_tertiary"],
            font=HelixTheme.FONTS["small"],
        )

    def set_audio(self, audio: np.ndarray, sample_rate: int = 22050):
        """Set audio data to display."""
        self._audio_data = audio
        self._sample_rate = sample_rate
        self._redraw()

    def _redraw(self):
        """Redraw the waveform."""
        self.delete("all")

        if self._audio_data is None or len(self._audio_data) == 0:
            self._draw_empty()
            return

        # Downsample for display
        samples_per_pixel = max(1, len(self._audio_data) // self._width)
        downsampled = []

        for i in range(0, len(self._audio_data) - samples_per_pixel, samples_per_pixel):
            chunk = self._audio_data[i:i + samples_per_pixel]
            downsampled.append((np.min(chunk), np.max(chunk)))

        if not downsampled:
            self._draw_empty()
            return

        center_y = self._height // 2
        max_amplitude = self._height // 2 - 5

        # Draw waveform
        for i, (min_val, max_val) in enumerate(downsampled):
            y1 = center_y - int(max_val * max_amplitude)
            y2 = center_y - int(min_val * max_amplitude)

            self.create_line(
                i, y1, i, y2,
                fill=HelixTheme.COLORS["accent"],
                width=1,
            )

        # Draw center line
        self.create_line(
            0, center_y, self._width, center_y,
            fill=HelixTheme.COLORS["border"],
            width=1,
        )

        # Draw playhead
        self._draw_playhead()

    def _draw_playhead(self):
        """Draw the playhead indicator."""
        if self._audio_data is None:
            return

        x = int(self._playhead_position * self._width)
        self.create_line(
            x, 0, x, self._height,
            fill=HelixTheme.COLORS["accent_hover"],
            width=2,
            tags="playhead",
        )

    def set_playhead(self, position: float):
        """Set playhead position (0.0 to 1.0)."""
        self._playhead_position = max(0.0, min(1.0, position))
        self.delete("playhead")
        self._draw_playhead()

    def clear(self):
        """Clear the waveform display."""
        self._audio_data = None
        self._playhead_position = 0.0
        self._draw_empty()

    def get_duration(self) -> float:
        """Get audio duration in seconds."""
        if self._audio_data is None:
            return 0.0
        return len(self._audio_data) / self._sample_rate
