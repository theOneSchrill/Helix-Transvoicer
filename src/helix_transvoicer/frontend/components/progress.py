"""
Helix Transvoicer - Progress indicator component.
"""

import customtkinter as ctk
from typing import Optional

from helix_transvoicer.frontend.styles.theme import HelixTheme


class ProgressIndicator(ctk.CTkFrame):
    """
    Progress indicator with label and percentage.
    """

    def __init__(
        self,
        parent,
        width: int = 400,
        height: int = 60,
        **kwargs,
    ):
        super().__init__(
            parent,
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            corner_radius=HelixTheme.RADIUS["sm"],
            **kwargs,
        )

        self._width = width
        self._progress = 0.0
        self._stage = ""

        self._build_ui()

    def _build_ui(self):
        """Build the progress indicator UI."""
        self.grid_columnconfigure(0, weight=1)

        # Stage label
        self.stage_label = ctk.CTkLabel(
            self,
            text="Ready",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
            anchor="w",
        )
        self.stage_label.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="w")

        # Progress bar container
        progress_container = ctk.CTkFrame(
            self,
            fg_color="transparent",
        )
        progress_container.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="ew")
        progress_container.grid_columnconfigure(0, weight=1)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            progress_container,
            width=self._width - 80,
            height=8,
            corner_radius=4,
            fg_color=HelixTheme.COLORS["progress_bg"],
            progress_color=HelixTheme.COLORS["progress_fill"],
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        self.progress_bar.set(0)

        # Percentage label
        self.percent_label = ctk.CTkLabel(
            progress_container,
            text="0%",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
            width=50,
        )
        self.percent_label.grid(row=0, column=1, padx=(10, 0))

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

    def set_stage(self, stage: str):
        """Set stage description."""
        self._stage = stage
        self.stage_label.configure(text=stage)

    def reset(self):
        """Reset progress indicator."""
        self._progress = 0.0
        self._stage = ""
        self.progress_bar.set(0)
        self.percent_label.configure(text="0%")
        self.stage_label.configure(text="Ready")

    def set_complete(self):
        """Set to complete state."""
        self._progress = 1.0
        self.progress_bar.set(1.0)
        self.percent_label.configure(text="100%")
        self.stage_label.configure(text="Complete")

    def set_error(self, message: str = "Error"):
        """Set to error state."""
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
