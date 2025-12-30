"""
Helix Transvoicer - Main application window.

Modern, sleek UI for AI-powered voice processing.
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

    A modern, dark-themed UI for voice processing with AI.
    """

    WINDOW_TITLE = "Helix Transvoicer"
    WINDOW_SIZE = (1440, 900)
    MIN_SIZE = (1200, 700)

    def __init__(self, api_url: str = "http://127.0.0.1:8420"):
        # Set theme before creating window
        HelixTheme.apply()

        # Create main window
        self.root = ctk.CTk()
        self.root.title(self.WINDOW_TITLE)
        self.root.geometry(f"{self.WINDOW_SIZE[0]}x{self.WINDOW_SIZE[1]}")
        self.root.minsize(*self.MIN_SIZE)
        self.root.configure(fg_color=HelixTheme.COLORS["bg_primary"])

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

        # Modern header with navigation
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
        """Build the modern header with navigation."""
        header = ctk.CTkFrame(
            self.root,
            fg_color=HelixTheme.COLORS["bg_secondary"],
            corner_radius=0,
            height=60,
        )
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        header.grid_propagate(False)

        # Logo section
        logo_frame = ctk.CTkFrame(header, fg_color="transparent")
        logo_frame.grid(row=0, column=0, padx=24, pady=0, sticky="w")

        # App logo icon (stylized H)
        logo_icon = ctk.CTkLabel(
            logo_frame,
            text="◆",
            font=("", 24),
            text_color=HelixTheme.COLORS["accent"],
        )
        logo_icon.pack(side="left", padx=(0, 8))

        # App title
        title = ctk.CTkLabel(
            logo_frame,
            text="Helix",
            font=HelixTheme.FONTS["title"],
            text_color=HelixTheme.COLORS["text_primary"],
        )
        title.pack(side="left")

        subtitle = ctk.CTkLabel(
            logo_frame,
            text="Transvoicer",
            font=HelixTheme.FONTS["title"],
            text_color=HelixTheme.COLORS["accent"],
        )
        subtitle.pack(side="left", padx=(4, 0))

        # Navigation tabs container
        nav_frame = ctk.CTkFrame(
            header,
            fg_color="transparent",
        )
        nav_frame.grid(row=0, column=1, sticky="", padx=20)

        self._tab_buttons = {}
        self._tab_indicators = {}

        tab_config = [
            ("converter", "Voice Convert", "🎙"),
            ("builder", "Model Builder", "🔧"),
            ("emotions", "Emotions", "😊"),
            ("tts", "TTS Studio", "🔊"),
            ("library", "Library", "📚"),
        ]

        for i, (key, label, icon) in enumerate(tab_config):
            # Tab container
            tab_container = ctk.CTkFrame(nav_frame, fg_color="transparent")
            tab_container.grid(row=0, column=i, padx=4)

            # Tab button
            btn = ctk.CTkButton(
                tab_container,
                text=f"{icon}  {label}",
                font=HelixTheme.FONTS["button"],
                fg_color="transparent",
                hover_color=HelixTheme.COLORS["bg_hover"],
                text_color=HelixTheme.COLORS["text_tertiary"],
                corner_radius=HelixTheme.RADIUS["md"],
                width=130,
                height=36,
                command=lambda k=key: self._switch_panel(k),
            )
            btn.pack(pady=(12, 4))

            # Active indicator (underline)
            indicator = ctk.CTkFrame(
                tab_container,
                fg_color="transparent",
                height=3,
                width=80,
                corner_radius=2,
            )
            indicator.pack()

            self._tab_buttons[key] = btn
            self._tab_indicators[key] = indicator

        # Right side - version/settings placeholder
        right_frame = ctk.CTkFrame(header, fg_color="transparent")
        right_frame.grid(row=0, column=2, padx=24, sticky="e")

        version_label = ctk.CTkLabel(
            right_frame,
            text="v1.0",
            font=HelixTheme.FONTS["tiny"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        version_label.pack(side="right")

    def _switch_panel(self, panel_name: str):
        """Switch to a different panel with smooth transition."""
        if self._current_panel == panel_name:
            return

        # Hide current panel
        if self._current_panel and self._current_panel in self._panels:
            self._panels[self._current_panel].grid_forget()

        # Update tab styling
        for key, btn in self._tab_buttons.items():
            indicator = self._tab_indicators.get(key)
            if key == panel_name:
                btn.configure(
                    fg_color=HelixTheme.COLORS["bg_hover"],
                    text_color=HelixTheme.COLORS["text_primary"],
                )
                if indicator:
                    indicator.configure(fg_color=HelixTheme.COLORS["accent"])
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=HelixTheme.COLORS["text_tertiary"],
                )
                if indicator:
                    indicator.configure(fg_color="transparent")

        # Show new panel
        self._current_panel = panel_name
        panel = self._panels.get(panel_name)
        if panel:
            panel.grid(row=0, column=0, sticky="nsew", padx=24, pady=24)

    def _update_status(self):
        """Periodically update status bar."""
        self.status_bar.update()
        self.root.after(5000, self._update_status)

    def run(self):
        """Run the application main loop."""
        self.root.mainloop()

    def quit(self):
        """Quit the application."""
        self.api_client.close()
        self.root.quit()
