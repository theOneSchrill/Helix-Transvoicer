"""
Helix Transvoicer - Main application window.
"""

import customtkinter as ctk
from typing import Optional

from helix_transvoicer.frontend.styles.theme import HelixTheme
from helix_transvoicer.frontend.utils.api_client import APIClient
from helix_transvoicer.frontend.panels.converter import ConverterPanel
from helix_transvoicer.frontend.panels.builder import BuilderPanel
from helix_transvoicer.frontend.panels.emotions import EmotionsPanel
from helix_transvoicer.frontend.panels.tts import TTSPanel
from helix_transvoicer.frontend.panels.library import LibraryPanel
from helix_transvoicer.frontend.components.status_bar import StatusBar


class HelixApp:
    """
    Main Helix Transvoicer application.

    A modern, dark-themed UI for voice processing.
    """

    WINDOW_TITLE = "HELIX TRANSVOICER"
    WINDOW_SIZE = (1400, 900)
    MIN_SIZE = (1200, 700)

    def __init__(self, api_url: str = "http://127.0.0.1:8420"):
        # Set theme before creating window
        HelixTheme.apply()

        # Create main window
        self.root = ctk.CTk()
        self.root.title(self.WINDOW_TITLE)
        self.root.geometry(f"{self.WINDOW_SIZE[0]}x{self.WINDOW_SIZE[1]}")
        self.root.minsize(*self.MIN_SIZE)

        # API client
        self.api_client = APIClient(api_url)

        # Current panel
        self._current_panel: Optional[str] = None
        self._panels = {}

        # Build UI
        self._build_ui()

        # Set initial panel
        self._switch_panel("converter")

        # Start status updates
        self._update_status()

    def _build_ui(self):
        """Build the main UI layout."""
        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # Header with tabs
        self._build_header()

        # Main content area
        self.content_frame = ctk.CTkFrame(
            self.root,
            fg_color=HelixTheme.COLORS["bg_primary"],
            corner_radius=0,
        )
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)

        # Status bar
        self.status_bar = StatusBar(self.root, self.api_client)
        self.status_bar.grid(row=2, column=0, sticky="ew")

        # Initialize panels
        self._panels["converter"] = ConverterPanel(self.content_frame, self.api_client)
        self._panels["builder"] = BuilderPanel(self.content_frame, self.api_client)
        self._panels["emotions"] = EmotionsPanel(self.content_frame, self.api_client)
        self._panels["tts"] = TTSPanel(self.content_frame, self.api_client)
        self._panels["library"] = LibraryPanel(self.content_frame, self.api_client)

    def _build_header(self):
        """Build the header with navigation tabs."""
        header = ctk.CTkFrame(
            self.root,
            fg_color=HelixTheme.COLORS["bg_secondary"],
            corner_radius=0,
            height=50,
        )
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        # Logo/title
        title = ctk.CTkLabel(
            header,
            text="HELIX TRANSVOICER",
            font=HelixTheme.FONTS["title"],
            text_color=HelixTheme.COLORS["accent"],
        )
        title.grid(row=0, column=0, padx=20, pady=10)

        # Tab buttons container
        tabs_frame = ctk.CTkFrame(
            header,
            fg_color="transparent",
        )
        tabs_frame.grid(row=0, column=1, sticky="w", padx=20)

        self._tab_buttons = {}
        tab_names = [
            ("converter", "VOICE CONVERTER"),
            ("builder", "MODEL BUILDER"),
            ("emotions", "EMOTION MAP"),
            ("tts", "TTS STUDIO"),
            ("library", "LIBRARY"),
        ]

        for i, (key, label) in enumerate(tab_names):
            btn = ctk.CTkButton(
                tabs_frame,
                text=label,
                font=HelixTheme.FONTS["button"],
                fg_color="transparent",
                hover_color=HelixTheme.COLORS["bg_hover"],
                text_color=HelixTheme.COLORS["text_secondary"],
                corner_radius=0,
                width=130,
                height=40,
                command=lambda k=key: self._switch_panel(k),
            )
            btn.grid(row=0, column=i, padx=2)
            self._tab_buttons[key] = btn

    def _switch_panel(self, panel_name: str):
        """Switch to a different panel."""
        if self._current_panel == panel_name:
            return

        # Hide current panel
        if self._current_panel and self._current_panel in self._panels:
            self._panels[self._current_panel].grid_forget()

        # Update tab styling
        for key, btn in self._tab_buttons.items():
            if key == panel_name:
                btn.configure(
                    fg_color=HelixTheme.COLORS["accent"],
                    text_color=HelixTheme.COLORS["text_primary"],
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=HelixTheme.COLORS["text_secondary"],
                )

        # Show new panel
        self._current_panel = panel_name
        panel = self._panels.get(panel_name)
        if panel:
            panel.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

    def _update_status(self):
        """Periodically update status bar."""
        self.status_bar.update()
        self.root.after(5000, self._update_status)  # Update every 5 seconds

    def run(self):
        """Run the application main loop."""
        self.root.mainloop()

    def quit(self):
        """Quit the application."""
        self.root.quit()
