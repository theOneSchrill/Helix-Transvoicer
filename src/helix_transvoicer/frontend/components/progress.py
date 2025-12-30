"""
Helix Transvoicer - Modern progress indicator component.
"""

import customtkinter as ctk
from typing import Optional

from helix_transvoicer.frontend.styles.theme import HelixTheme


class ProgressIndicator(ctk.CTkFrame):
    """
    Modern progress indicator with animated feel and status labels.
    """

    def __init__(
        self,
        parent,
        width: int = 400,
        height: int = 70,
        **kwargs,
    ):
        super().__init__(
            parent,
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            corner_radius=HelixTheme.RADIUS["md"],
            **kwargs,
        )

        self._width = width
        self._progress = 0.0
        self._stage = ""
        self._is_error = False

        self._build_ui()

    def _build_ui(self):
        """Build the progress indicator UI."""
        self.grid_columnconfigure(0, weight=1)

        # Top row - Stage and percentage
        top_row = ctk.CTkFrame(self, fg_color="transparent")
        top_row.grid(row=0, column=0, padx=16, pady=(12, 8), sticky="ew")
        top_row.grid_columnconfigure(0, weight=1)

        # Stage label (left)
        self.stage_label = ctk.CTkLabel(
            top_row,
            text="Ready",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
            anchor="w",
        )
        self.stage_label.grid(row=0, column=0, sticky="w")

        # Percentage label (right)
        self.percent_label = ctk.CTkLabel(
            top_row,
            text="0%",
            font=HelixTheme.FONTS["mono_small"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        self.percent_label.grid(row=0, column=1, sticky="e")

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            self,
            width=self._width - 32,
            height=6,
            corner_radius=3,
            fg_color=HelixTheme.COLORS["bg_primary"],
            progress_color=HelixTheme.COLORS["accent"],
        )
        self.progress_bar.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.progress_bar.set(0)

    def set_progress(self, progress: float, stage: Optional[str] = None):
        """
        Set progress value and optionally update stage.

        Args:
            progress: Progress value (0.0 to 1.0)
            stage: Optional stage description
        """
        self._progress = max(0.0, min(1.0, progress))
        self.progress_bar.set(self._progress)
        self.percent_label.configure(text=f"{int(self._progress * 100)}%")

        if stage is not None:
            self._stage = stage
            self.stage_label.configure(text=stage)

        # Reset error state if progressing
        if self._is_error and progress > 0:
            self._reset_colors()

    def set_stage(self, stage: str):
        """Set stage description."""
        self._stage = stage
        self.stage_label.configure(text=stage)

    def reset(self):
        """Reset progress indicator."""
        self._progress = 0.0
        self._stage = ""
        self._is_error = False
        self.progress_bar.set(0)
        self.percent_label.configure(text="0%")
        self.stage_label.configure(
            text="Ready",
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        self._reset_colors()

    def _reset_colors(self):
        """Reset to default colors."""
        self._is_error = False
        self.progress_bar.configure(
            progress_color=HelixTheme.COLORS["accent"],
        )
        self.stage_label.configure(
            text_color=HelixTheme.COLORS["text_secondary"],
        )

    def set_complete(self):
        """Set to complete state."""
        self._progress = 1.0
        self._is_error = False
        self.progress_bar.set(1.0)
        self.progress_bar.configure(
            progress_color=HelixTheme.COLORS["success"],
        )
        self.percent_label.configure(text="100%")
        self.stage_label.configure(
            text="Complete",
            text_color=HelixTheme.COLORS["success"],
        )

    def set_error(self, message: str = "Error"):
        """Set to error state."""
        self._is_error = True
        self.stage_label.configure(
            text=message,
            text_color=HelixTheme.COLORS["error"],
        )
        self.progress_bar.configure(
            progress_color=HelixTheme.COLORS["error"],
        )

    @property
    def progress(self) -> float:
        """Get current progress value."""
        return self._progress

    @property
    def stage(self) -> str:
        """Get current stage description."""
        return self._stage
