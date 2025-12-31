"""
Helix Transvoicer - Voice Converter Panel.
"""

import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog
from typing import List, Optional

from helix_transvoicer.frontend.styles.theme import HelixTheme
from helix_transvoicer.frontend.utils.api_client import APIClient
from helix_transvoicer.frontend.components.waveform import WaveformDisplay
from helix_transvoicer.frontend.components.progress import ProgressIndicator
from helix_transvoicer.frontend.components.controls import Slider, DropdownSelect
from helix_transvoicer.frontend.components.dropzone import DropZone


class ConverterPanel(ctk.CTkFrame):
    """
    Voice conversion workspace panel.

    Allows users to:
    - Load source audio
    - Select target voice model
    - Configure conversion parameters
    - Convert and save output
    """

    def __init__(self, parent, api_client: APIClient):
        super().__init__(
            parent,
            fg_color=HelixTheme.COLORS["bg_primary"],
            corner_radius=0,
        )

        self.api_client = api_client
        self._source_path: Optional[Path] = None
        self._output_audio: Optional[bytes] = None

        self._build_ui()
        self._refresh_models()

    def _build_ui(self):
        """Build the converter panel UI."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)

        # Title
        title = ctk.CTkLabel(
            self,
            text="VOICE CONVERTER",
            font=HelixTheme.FONTS["heading"],
            text_color=HelixTheme.COLORS["text_primary"],
        )
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 20))

        # Main content area
        main_frame = ctk.CTkFrame(
            self,
            fg_color=HelixTheme.COLORS["bg_secondary"],
            corner_radius=HelixTheme.RADIUS["md"],
        )
        main_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 20))
        main_frame.grid_columnconfigure(0, weight=1)

        # Source audio section
        self._build_source_section(main_frame)

        # Output section
        self._build_output_section(main_frame)

        # Side panel for settings
        side_frame = ctk.CTkFrame(
            self,
            fg_color=HelixTheme.COLORS["bg_secondary"],
            corner_radius=HelixTheme.RADIUS["md"],
            width=300,
        )
        side_frame.grid(row=1, column=1, sticky="ns")

        # Target voice section
        self._build_target_section(side_frame)

        # Settings section
        self._build_settings_section(side_frame)

        # Convert button
        self.convert_btn = ctk.CTkButton(
            side_frame,
            text="▶ CONVERT",
            height=50,
            **HelixTheme.get_button_style("primary"),
            command=self._on_convert,
        )
        self.convert_btn.pack(fill="x", padx=20, pady=20)

    def _build_source_section(self, parent):
        """Build source audio section."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=20, pady=20)

        # Section title
        title = ctk.CTkLabel(
            section,
            text="SOURCE AUDIO",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(anchor="w", pady=(0, 10))

        # Drop zone for audio files
        self.drop_zone = DropZone(
            section,
            on_files_dropped=self._on_files_dropped,
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg"),
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("FLAC files", "*.flac"),
                ("All files", "*.*"),
            ],
            multiple=False,
            width=600,
            height=100,
            title="Drop audio file here",
            subtitle="or click to browse",
            icon="🎵",
        )
        self.drop_zone.pack(fill="x", pady=10)

        # File info label
        self.source_info = ctk.CTkLabel(
            section,
            text="No file loaded",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        self.source_info.pack(anchor="w")

        # Waveform display
        self.source_waveform = WaveformDisplay(section, width=600, height=80)
        self.source_waveform.pack(fill="x", pady=10)

    def _build_output_section(self, parent):
        """Build output audio section."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=20, pady=20)

        # Section title
        title = ctk.CTkLabel(
            section,
            text="OUTPUT",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(anchor="w", pady=(0, 10))

        # Progress indicator
        self.progress = ProgressIndicator(section, width=600)
        self.progress.pack(fill="x", pady=10)

        # Output waveform
        self.output_waveform = WaveformDisplay(section, width=600, height=80)
        self.output_waveform.pack(fill="x", pady=10)

        # Controls
        controls = ctk.CTkFrame(section, fg_color="transparent")
        controls.pack(fill="x")

        self.save_btn = ctk.CTkButton(
            controls,
            text="💾 Save As...",
            width=120,
            **HelixTheme.get_button_style("secondary"),
            command=self._on_save,
            state="disabled",
        )
        self.save_btn.pack(side="left", padx=(0, 10))

        self.reset_btn = ctk.CTkButton(
            controls,
            text="↻ Reset",
            width=100,
            **HelixTheme.get_button_style("ghost"),
            command=self._on_reset,
        )
        self.reset_btn.pack(side="left")

    def _build_target_section(self, parent):
        """Build target voice selection."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=20, pady=20)

        # Section title
        title = ctk.CTkLabel(
            section,
            text="TARGET VOICE",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(anchor="w", pady=(0, 10))

        # Model selector
        self.model_select = DropdownSelect(
            section,
            label="Voice Model",
            options=["Loading..."],
            width=260,
        )
        self.model_select.pack(fill="x", pady=5)

        # Refresh button
        refresh_btn = ctk.CTkButton(
            section,
            text="🔄 Refresh List",
            width=260,
            height=28,
            **HelixTheme.get_button_style("ghost"),
            command=self._refresh_models,
        )
        refresh_btn.pack(fill="x", pady=5)

    def _build_settings_section(self, parent):
        """Build conversion settings."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=20, pady=20)

        # Section title
        title = ctk.CTkLabel(
            section,
            text="CONVERSION SETTINGS",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(anchor="w", pady=(0, 15))

        # Pitch slider
        self.pitch_slider = Slider(
            section,
            label="Pitch Shift:",
            from_=-12.0,
            to=12.0,
            default=0.0,
            format_str="{:+.0f} st",
            width=260,
        )
        self.pitch_slider.pack(fill="x", pady=5)

        # Formant shift slider
        self.formant_slider = Slider(
            section,
            label="Formant Shift:",
            from_=0.5,
            to=2.0,
            default=1.0,
            format_str="{:.2f}x",
            width=260,
        )
        self.formant_slider.pack(fill="x", pady=5)

        # RVC Index Rate slider
        self.index_rate_slider = Slider(
            section,
            label="Voice Similarity (RVC):",
            from_=0.0,
            to=1.0,
            default=0.75,
            format_str="{:.0%}",
            width=260,
        )
        self.index_rate_slider.pack(fill="x", pady=5)

        # Options
        options_frame = ctk.CTkFrame(section, fg_color="transparent")
        options_frame.pack(fill="x", pady=10)

        self.normalize_var = ctk.BooleanVar(value=True)
        normalize_cb = ctk.CTkCheckBox(
            options_frame,
            text="Normalize output",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
            variable=self.normalize_var,
            fg_color=HelixTheme.COLORS["accent"],
        )
        normalize_cb.pack(anchor="w")

    def _refresh_models(self):
        """Refresh available voice models."""
        try:
            models = self.api_client.list_models()
            model_names = [m["id"] for m in models] if models else ["No models"]
            self.model_select.update_options(model_names)
        except Exception:
            self.model_select.update_options(["Error loading models"])

    def _on_files_dropped(self, files: List[Path]):
        """Handle files dropped or selected via drop zone."""
        if files:
            self._source_path = files[0]
            self.source_info.configure(
                text=f"Loaded: {self._source_path.name}",
                text_color=HelixTheme.COLORS["accent"],
            )
            self.drop_zone.flash_success()
            self.drop_zone.set_title(self._source_path.name)
            self.drop_zone.set_subtitle("Click to change file")
            # In a real implementation, we'd load and display the waveform
            self.source_waveform._draw_empty()

    def _on_convert(self):
        """Handle convert button click."""
        if not self._source_path:
            return

        model_id = self.model_select.get()
        if not model_id or model_id in ["No models", "Loading...", "Error loading models"]:
            return

        self.progress.reset()
        self.progress.set_stage("Converting...")
        self.convert_btn.configure(state="disabled")

        try:
            # Perform conversion
            self._output_audio = self.api_client.convert_voice(
                self._source_path,
                model_id,
                pitch_shift=self.pitch_slider.get(),
                formant_shift=self.formant_slider.get(),
                index_rate=self.index_rate_slider.get(),
            )

            self.progress.set_complete()
            self.save_btn.configure(state="normal")

        except Exception as e:
            self.progress.set_error(str(e)[:50])

        finally:
            self.convert_btn.configure(state="normal")

    def _on_save(self):
        """Handle save output."""
        if not self._output_audio:
            return

        path = filedialog.asksaveasfilename(
            title="Save Converted Audio",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
        )

        if path:
            with open(path, "wb") as f:
                f.write(self._output_audio)

    def _on_reset(self):
        """Reset the converter."""
        self._source_path = None
        self._output_audio = None
        self.source_info.configure(
            text="No file loaded",
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        self.drop_zone.set_title("Drop audio file here")
        self.drop_zone.set_subtitle("or click to browse")
        self.source_waveform.clear()
        self.output_waveform.clear()
        self.progress.reset()
        self.save_btn.configure(state="disabled")
