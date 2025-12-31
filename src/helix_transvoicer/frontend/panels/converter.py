"""
Helix Transvoicer - Voice Converter Panel (Applio-style).

Full-featured voice conversion interface with all RVC parameters.
"""

import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog
from typing import Dict, List, Optional
import threading

from helix_transvoicer.frontend.styles.theme import HelixTheme
from helix_transvoicer.frontend.utils.api_client import APIClient
from helix_transvoicer.frontend.components.waveform import WaveformDisplay
from helix_transvoicer.frontend.components.progress import ProgressIndicator
from helix_transvoicer.frontend.components.controls import Slider, DropdownSelect
from helix_transvoicer.frontend.components.dropzone import DropZone


class CollapsibleSection(ctk.CTkFrame):
    """Collapsible settings section."""

    def __init__(self, parent, title: str, expanded: bool = False, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self._expanded = expanded
        self._title = title

        # Header
        self.header = ctk.CTkFrame(self, fg_color=HelixTheme.COLORS["bg_tertiary"], corner_radius=6)
        self.header.pack(fill="x", pady=(0, 2))

        self.toggle_btn = ctk.CTkButton(
            self.header,
            text=f"{'▼' if expanded else '▶'} {title}",
            anchor="w",
            fg_color="transparent",
            hover_color=HelixTheme.COLORS["bg_secondary"],
            text_color=HelixTheme.COLORS["text_primary"],
            font=HelixTheme.FONTS["button"],
            command=self._toggle,
        )
        self.toggle_btn.pack(fill="x", padx=5, pady=5)

        # Content
        self.content = ctk.CTkFrame(self, fg_color=HelixTheme.COLORS["bg_tertiary"], corner_radius=6)
        if expanded:
            self.content.pack(fill="x", pady=(0, 10), padx=5)

    def _toggle(self):
        self._expanded = not self._expanded
        self.toggle_btn.configure(text=f"{'▼' if self._expanded else '▶'} {self._title}")
        if self._expanded:
            self.content.pack(fill="x", pady=(0, 10), padx=5)
        else:
            self.content.pack_forget()

    def get_content_frame(self) -> ctk.CTkFrame:
        return self.content


class ConverterPanel(ctk.CTkFrame):
    """
    Applio-style voice conversion panel with full RVC parameter support.
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
        self._models: List[Dict] = []
        self._index_files: List[Dict] = []

        self._build_ui()
        self._refresh_models()

    def _build_ui(self):
        """Build the Applio-style converter panel."""
        # Main scrollable container
        self.main_scroll = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            corner_radius=0,
        )
        self.main_scroll.pack(fill="both", expand=True, padx=20, pady=20)
        self.main_scroll.grid_columnconfigure(0, weight=1)
        self.main_scroll.grid_columnconfigure(1, weight=1)

        # Left column - Model & Audio selection
        left_col = ctk.CTkFrame(self.main_scroll, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Right column - Output
        right_col = ctk.CTkFrame(self.main_scroll, fg_color="transparent")
        right_col.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # Build sections
        self._build_model_section(left_col)
        self._build_audio_section(left_col)
        self._build_advanced_settings(left_col)
        self._build_preset_settings(left_col)
        self._build_convert_section(left_col)
        self._build_output_section(right_col)

    def _build_model_section(self, parent):
        """Build voice model and index selection."""
        section = ctk.CTkFrame(parent, fg_color=HelixTheme.COLORS["bg_secondary"], corner_radius=8)
        section.pack(fill="x", pady=(0, 15))

        # Voice Model
        model_frame = ctk.CTkFrame(section, fg_color="transparent")
        model_frame.pack(fill="x", padx=15, pady=15)

        ctk.CTkLabel(
            model_frame,
            text="Voice Model",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            model_frame,
            text="Select the voice model to use for the conversion.",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        ).pack(anchor="w", pady=(0, 5))

        self.model_select = ctk.CTkComboBox(
            model_frame,
            values=["Loading..."],
            font=HelixTheme.FONTS["body"],
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            border_color=HelixTheme.COLORS["border"],
            button_color=HelixTheme.COLORS["accent"],
            dropdown_fg_color=HelixTheme.COLORS["bg_secondary"],
            command=self._on_model_selected,
        )
        self.model_select.pack(fill="x", pady=5)

        # Index File
        index_frame = ctk.CTkFrame(section, fg_color="transparent")
        index_frame.pack(fill="x", padx=15, pady=(0, 15))

        ctk.CTkLabel(
            index_frame,
            text="Index File",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            index_frame,
            text="Select the index file to use for the conversion.",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        ).pack(anchor="w", pady=(0, 5))

        self.index_select = ctk.CTkComboBox(
            index_frame,
            values=["Auto-detect"],
            font=HelixTheme.FONTS["body"],
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            border_color=HelixTheme.COLORS["border"],
            button_color=HelixTheme.COLORS["accent"],
            dropdown_fg_color=HelixTheme.COLORS["bg_secondary"],
        )
        self.index_select.pack(fill="x", pady=5)

        # Buttons
        btn_frame = ctk.CTkFrame(section, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        self.unload_btn = ctk.CTkButton(
            btn_frame,
            text="Unload Voice",
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            hover_color=HelixTheme.COLORS["border"],
            text_color=HelixTheme.COLORS["text_primary"],
            command=self._on_unload,
        )
        self.unload_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.refresh_btn = ctk.CTkButton(
            btn_frame,
            text="Refresh",
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            hover_color=HelixTheme.COLORS["border"],
            text_color=HelixTheme.COLORS["text_primary"],
            command=self._refresh_models,
        )
        self.refresh_btn.grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def _build_audio_section(self, parent):
        """Build audio upload and selection."""
        section = ctk.CTkFrame(parent, fg_color=HelixTheme.COLORS["bg_secondary"], corner_radius=8)
        section.pack(fill="x", pady=(0, 15))

        # Tabs for Single/Batch
        tab_frame = ctk.CTkFrame(section, fg_color="transparent")
        tab_frame.pack(fill="x", padx=15, pady=(15, 5))

        self.single_tab = ctk.CTkButton(
            tab_frame,
            text="Single",
            width=70,
            fg_color=HelixTheme.COLORS["accent"],
            text_color=HelixTheme.COLORS["text_primary"],
        )
        self.single_tab.pack(side="left", padx=(0, 5))

        self.batch_tab = ctk.CTkButton(
            tab_frame,
            text="Batch",
            width=70,
            fg_color="transparent",
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        self.batch_tab.pack(side="left")

        # Drop zone
        audio_frame = ctk.CTkFrame(section, fg_color="transparent")
        audio_frame.pack(fill="x", padx=15, pady=10)

        self.drop_zone = DropZone(
            audio_frame,
            on_files_dropped=self._on_files_dropped,
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
            ],
            multiple=False,
            width=400,
            height=80,
            title="Upload Audio",
            subtitle="Drop audio file or click to browse",
            icon="",
        )
        self.drop_zone.pack(fill="x")

        # Waveform display
        self.source_waveform = WaveformDisplay(audio_frame, width=400, height=60)
        self.source_waveform.pack(fill="x", pady=10)

        # Audio selector
        select_frame = ctk.CTkFrame(section, fg_color="transparent")
        select_frame.pack(fill="x", padx=15, pady=(0, 15))

        ctk.CTkLabel(
            select_frame,
            text="Select Audio",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        ).pack(anchor="w")

        self.audio_select = ctk.CTkComboBox(
            select_frame,
            values=["No audio loaded"],
            font=HelixTheme.FONTS["body"],
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            border_color=HelixTheme.COLORS["border"],
        )
        self.audio_select.pack(fill="x", pady=5)

    def _build_advanced_settings(self, parent):
        """Build advanced settings section."""
        section = CollapsibleSection(parent, "Advanced Settings", expanded=True)
        section.pack(fill="x", pady=(0, 15))
        content = section.get_content_frame()

        # Output Path
        path_frame = ctk.CTkFrame(content, fg_color="transparent")
        path_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            path_frame,
            text="Output Path",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
        ).pack(anchor="w")

        self.output_path = ctk.CTkEntry(
            path_frame,
            placeholder_text="assets/audios/output.wav",
            fg_color=HelixTheme.COLORS["bg_tertiary"],
        )
        self.output_path.pack(fill="x", pady=2)

        # Export Format
        format_frame = ctk.CTkFrame(content, fg_color="transparent")
        format_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            format_frame,
            text="Export Format",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
        ).pack(anchor="w")

        format_options = ctk.CTkFrame(format_frame, fg_color="transparent")
        format_options.pack(fill="x", pady=2)

        self.format_var = ctk.StringVar(value="wav")
        for fmt in ["WAV", "MP3", "FLAC", "OGG", "M4A"]:
            ctk.CTkRadioButton(
                format_options,
                text=fmt,
                variable=self.format_var,
                value=fmt.lower(),
                font=HelixTheme.FONTS["small"],
                fg_color=HelixTheme.COLORS["accent"],
            ).pack(side="left", padx=5)

        # Speaker ID
        speaker_frame = ctk.CTkFrame(content, fg_color="transparent")
        speaker_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            speaker_frame,
            text="Speaker ID",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
        ).pack(anchor="w")

        self.speaker_id = ctk.CTkComboBox(
            speaker_frame,
            values=["0"],
            width=100,
            fg_color=HelixTheme.COLORS["bg_tertiary"],
        )
        self.speaker_id.pack(anchor="w", pady=2)

        # Checkboxes
        checks_frame = ctk.CTkFrame(content, fg_color="transparent")
        checks_frame.pack(fill="x", padx=10, pady=10)

        self.split_audio_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            checks_frame,
            text="Split Audio",
            variable=self.split_audio_var,
            font=HelixTheme.FONTS["small"],
            fg_color=HelixTheme.COLORS["accent"],
        ).pack(anchor="w", pady=2)

        self.autotune_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            checks_frame,
            text="Autotune",
            variable=self.autotune_var,
            font=HelixTheme.FONTS["small"],
            fg_color=HelixTheme.COLORS["accent"],
        ).pack(anchor="w", pady=2)

        self.clean_audio_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            checks_frame,
            text="Clean Audio",
            variable=self.clean_audio_var,
            font=HelixTheme.FONTS["small"],
            fg_color=HelixTheme.COLORS["accent"],
            command=self._toggle_clean_strength,
        ).pack(anchor="w", pady=2)

        # Clean Strength slider
        self.clean_strength_frame = ctk.CTkFrame(content, fg_color="transparent")
        self.clean_strength_frame.pack(fill="x", padx=10)

        self.clean_strength_slider = Slider(
            self.clean_strength_frame,
            label="Clean Strength:",
            from_=0.0,
            to=1.0,
            default=0.4,
            format_str="{:.2f}",
            width=300,
        )
        self.clean_strength_slider.pack(fill="x")
        self.clean_strength_frame.pack_forget()  # Hidden by default

        # Formant Shifting
        formant_frame = ctk.CTkFrame(content, fg_color="transparent")
        formant_frame.pack(fill="x", padx=10, pady=5)

        self.formant_shifting_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            formant_frame,
            text="Formant Shifting",
            variable=self.formant_shifting_var,
            font=HelixTheme.FONTS["small"],
            fg_color=HelixTheme.COLORS["accent"],
            command=self._toggle_formant_options,
        ).pack(anchor="w", pady=2)

        self.formant_options = ctk.CTkFrame(content, fg_color="transparent")

        ctk.CTkLabel(
            self.formant_options,
            text="Browse presets for formanting",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        ).pack(anchor="w", padx=10)

        self.formant_preset = ctk.CTkComboBox(
            self.formant_options,
            values=["m2f", "f2m", "neutral"],
            fg_color=HelixTheme.COLORS["bg_tertiary"],
        )
        self.formant_preset.pack(fill="x", padx=10, pady=5)

        self.quefrency_slider = Slider(
            self.formant_options,
            label="Quefrency:",
            from_=0.0,
            to=16.0,
            default=1.0,
            format_str="{:.1f}",
            width=300,
        )
        self.quefrency_slider.pack(fill="x", padx=10, pady=2)

        self.timbre_slider = Slider(
            self.formant_options,
            label="Timbre:",
            from_=0.0,
            to=16.0,
            default=1.2,
            format_str="{:.1f}",
            width=300,
        )
        self.timbre_slider.pack(fill="x", padx=10, pady=2)

        # Post-Process
        self.post_process_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            content,
            text="Post-Process",
            variable=self.post_process_var,
            font=HelixTheme.FONTS["small"],
            fg_color=HelixTheme.COLORS["accent"],
        ).pack(anchor="w", padx=10, pady=5)

    def _build_preset_settings(self, parent):
        """Build preset settings section."""
        section = CollapsibleSection(parent, "Preset Settings", expanded=True)
        section.pack(fill="x", pady=(0, 15))
        content = section.get_content_frame()

        # Pitch slider (-24 to +24)
        self.pitch_slider = Slider(
            content,
            label="Pitch:",
            from_=-24,
            to=24,
            default=0,
            format_str="{:+.0f}",
            width=300,
        )
        self.pitch_slider.pack(fill="x", padx=10, pady=10)

        # Search Feature Ratio
        self.index_rate_slider = Slider(
            content,
            label="Search Feature Ratio:",
            from_=0.0,
            to=1.0,
            default=0.0,
            format_str="{:.2f}",
            width=300,
        )
        self.index_rate_slider.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            content,
            text="Influence exerted by the index file. Higher = more influence.",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_tertiary"],
            wraplength=280,
        ).pack(anchor="w", padx=10)

        # Volume Envelope
        self.rms_slider = Slider(
            content,
            label="Volume Envelope:",
            from_=0.0,
            to=1.0,
            default=0.4,
            format_str="{:.2f}",
            width=300,
        )
        self.rms_slider.pack(fill="x", padx=10, pady=10)

        # Protect Voiceless Consonants
        self.protect_slider = Slider(
            content,
            label="Protect Voiceless Consonants:",
            from_=0.0,
            to=0.5,
            default=0.3,
            format_str="{:.2f}",
            width=300,
        )
        self.protect_slider.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            content,
            text="Safeguard consonants and breathing sounds. 0.5 = disabled.",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_tertiary"],
            wraplength=280,
        ).pack(anchor="w", padx=10)

        # Pitch extraction algorithm
        f0_frame = ctk.CTkFrame(content, fg_color="transparent")
        f0_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            f0_frame,
            text="Pitch extraction algorithm",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
        ).pack(anchor="w")

        f0_options = ctk.CTkFrame(f0_frame, fg_color="transparent")
        f0_options.pack(fill="x", pady=5)

        self.f0_method_var = ctk.StringVar(value="rmvpe")
        for method in ["crepe", "crepe-tiny", "rmvpe", "fcpe"]:
            ctk.CTkRadioButton(
                f0_options,
                text=method,
                variable=self.f0_method_var,
                value=method,
                font=HelixTheme.FONTS["small"],
                fg_color=HelixTheme.COLORS["accent"],
            ).pack(side="left", padx=5)

        # Embedder Model
        embed_frame = ctk.CTkFrame(content, fg_color="transparent")
        embed_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            embed_frame,
            text="Embedder Model",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
        ).pack(anchor="w")

        embed_options = ctk.CTkFrame(embed_frame, fg_color="transparent")
        embed_options.pack(fill="x", pady=5)

        self.embedder_var = ctk.StringVar(value="contentvec")
        for emb in ["contentvec", "spin", "spin-v2"]:
            ctk.CTkRadioButton(
                embed_options,
                text=emb,
                variable=self.embedder_var,
                value=emb,
                font=HelixTheme.FONTS["small"],
                fg_color=HelixTheme.COLORS["accent"],
            ).pack(side="left", padx=5)

    def _build_convert_section(self, parent):
        """Build convert button and terms."""
        section = ctk.CTkFrame(parent, fg_color=HelixTheme.COLORS["bg_secondary"], corner_radius=8)
        section.pack(fill="x", pady=(0, 15))

        # Terms checkbox
        terms_frame = ctk.CTkFrame(section, fg_color="transparent")
        terms_frame.pack(fill="x", padx=15, pady=(15, 5))

        self.terms_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            terms_frame,
            text="I agree to the terms of use",
            variable=self.terms_var,
            font=HelixTheme.FONTS["small"],
            fg_color=HelixTheme.COLORS["accent"],
        ).pack(anchor="w")

        # Convert button
        self.convert_btn = ctk.CTkButton(
            section,
            text="Convert",
            height=50,
            fg_color=HelixTheme.COLORS["accent"],
            hover_color=HelixTheme.COLORS["accent_hover"],
            text_color=HelixTheme.COLORS["text_primary"],
            font=HelixTheme.FONTS["button"],
            command=self._on_convert,
        )
        self.convert_btn.pack(fill="x", padx=15, pady=15)

    def _build_output_section(self, parent):
        """Build output section."""
        section = ctk.CTkFrame(parent, fg_color=HelixTheme.COLORS["bg_secondary"], corner_radius=8)
        section.pack(fill="both", expand=True)

        # Output Information
        info_frame = ctk.CTkFrame(section, fg_color="transparent")
        info_frame.pack(fill="x", padx=15, pady=15)

        ctk.CTkLabel(
            info_frame,
            text="Output Information",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        ).pack(anchor="w")

        self.output_info = ctk.CTkTextbox(
            info_frame,
            height=80,
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            text_color=HelixTheme.COLORS["text_primary"],
            font=HelixTheme.FONTS["small"],
        )
        self.output_info.pack(fill="x", pady=10)
        self.output_info.insert("1.0", "The output information will be displayed here.")
        self.output_info.configure(state="disabled")

        # Progress
        self.progress = ProgressIndicator(info_frame, width=350)
        self.progress.pack(fill="x", pady=10)

        # Export Audio
        export_frame = ctk.CTkFrame(section, fg_color="transparent")
        export_frame.pack(fill="x", padx=15, pady=(0, 15))

        ctk.CTkLabel(
            export_frame,
            text="Export Audio",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        ).pack(anchor="w")

        self.output_waveform = WaveformDisplay(export_frame, width=350, height=80)
        self.output_waveform.pack(fill="x", pady=10)

        # Export buttons
        btn_frame = ctk.CTkFrame(export_frame, fg_color="transparent")
        btn_frame.pack(fill="x")

        self.save_btn = ctk.CTkButton(
            btn_frame,
            text="Save As...",
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            hover_color=HelixTheme.COLORS["border"],
            text_color=HelixTheme.COLORS["text_primary"],
            command=self._on_save,
            state="disabled",
        )
        self.save_btn.pack(side="left", padx=(0, 10))

        self.reset_btn = ctk.CTkButton(
            btn_frame,
            text="Reset",
            fg_color="transparent",
            hover_color=HelixTheme.COLORS["border"],
            text_color=HelixTheme.COLORS["text_secondary"],
            command=self._on_reset,
        )
        self.reset_btn.pack(side="left")

    def _toggle_clean_strength(self):
        """Toggle clean strength slider visibility."""
        if self.clean_audio_var.get():
            self.clean_strength_frame.pack(fill="x", padx=10)
        else:
            self.clean_strength_frame.pack_forget()

    def _toggle_formant_options(self):
        """Toggle formant options visibility."""
        if self.formant_shifting_var.get():
            self.formant_options.pack(fill="x")
        else:
            self.formant_options.pack_forget()

    def _refresh_models(self):
        """Refresh available voice models."""
        try:
            models = self.api_client.list_models()
            self._models = models if models else []
            model_names = [m["id"] for m in self._models] if self._models else ["No models"]
            self.model_select.configure(values=model_names)
            if model_names:
                self.model_select.set(model_names[0])
                self._on_model_selected(model_names[0])
        except Exception as e:
            self.model_select.configure(values=["Error loading models"])

    def _on_model_selected(self, model_id: str):
        """Handle model selection - load index files."""
        if model_id in ["No models", "Loading...", "Error loading models"]:
            return

        try:
            # Get index files for this model
            files = self.api_client.get_model_files(model_id)
            if files and "index_files" in files:
                index_names = ["Auto-detect"] + [f["name"] for f in files["index_files"]]
                self.index_select.configure(values=index_names)
                self.index_select.set("Auto-detect")
        except Exception:
            self.index_select.configure(values=["Auto-detect"])

    def _on_unload(self):
        """Handle unload voice model."""
        model_id = self.model_select.get()
        if model_id not in ["No models", "Loading...", "Error loading models"]:
            try:
                self.api_client.unload_model(model_id)
            except Exception:
                pass

    def _on_files_dropped(self, files: List[Path]):
        """Handle files dropped."""
        if files:
            self._source_path = files[0]
            self.drop_zone.set_title(self._source_path.name)
            self.drop_zone.flash_success()
            self.audio_select.configure(values=[self._source_path.name])
            self.audio_select.set(self._source_path.name)

    def _on_convert(self):
        """Handle convert button click."""
        if not self._source_path:
            self._update_output_info("Please load an audio file first.")
            return

        if not self.terms_var.get():
            self._update_output_info("Please agree to the terms of use.")
            return

        model_id = self.model_select.get()
        if model_id in ["No models", "Loading...", "Error loading models"]:
            self._update_output_info("Please select a valid voice model.")
            return

        self.progress.reset()
        self.progress.set_stage("Converting...")
        self.convert_btn.configure(state="disabled")

        # Get all parameters
        index_file = self.index_select.get()
        if index_file == "Auto-detect":
            index_file = None

        # Run conversion in thread
        def convert_thread():
            try:
                self._output_audio = self.api_client.convert_voice(
                    self._source_path,
                    model_id,
                    pitch_shift=int(self.pitch_slider.get()),
                    f0_method=self.f0_method_var.get(),
                    index_rate=self.index_rate_slider.get(),
                    index_file=index_file,
                    rms_mix_rate=self.rms_slider.get(),
                    protect=self.protect_slider.get(),
                    split_audio=self.split_audio_var.get(),
                    autotune=self.autotune_var.get(),
                    clean_audio=self.clean_audio_var.get(),
                    clean_strength=self.clean_strength_slider.get() if self.clean_audio_var.get() else 0.4,
                    formant_shifting=self.formant_shifting_var.get(),
                    formant_quefrency=self.quefrency_slider.get() if self.formant_shifting_var.get() else 1.0,
                    formant_timbre=self.timbre_slider.get() if self.formant_shifting_var.get() else 1.2,
                    speaker_id=int(self.speaker_id.get()),
                    embedder_model=self.embedder_var.get(),
                    export_format=self.format_var.get(),
                )

                self.after(0, lambda: self._conversion_complete())

            except Exception as e:
                self.after(0, lambda: self._conversion_error(str(e)))

        thread = threading.Thread(target=convert_thread, daemon=True)
        thread.start()

    def _conversion_complete(self):
        """Handle conversion completion."""
        self.progress.set_complete()
        self.save_btn.configure(state="normal")
        self.convert_btn.configure(state="normal")
        self._update_output_info(f"File {self._source_path.name} inferred successfully.")

    def _conversion_error(self, error: str):
        """Handle conversion error."""
        self.progress.set_error(error[:50])
        self.convert_btn.configure(state="normal")
        self._update_output_info(f"Error: {error}")

    def _update_output_info(self, text: str):
        """Update output info text."""
        self.output_info.configure(state="normal")
        self.output_info.delete("1.0", "end")
        self.output_info.insert("1.0", text)
        self.output_info.configure(state="disabled")

    def _on_save(self):
        """Handle save output."""
        if not self._output_audio:
            return

        ext = self.format_var.get()
        path = filedialog.asksaveasfilename(
            title="Save Converted Audio",
            defaultextension=f".{ext}",
            filetypes=[(f"{ext.upper()} files", f"*.{ext}")],
        )

        if path:
            with open(path, "wb") as f:
                f.write(self._output_audio)
            self._update_output_info(f"Audio saved to {path}")

    def _on_reset(self):
        """Reset the converter."""
        self._source_path = None
        self._output_audio = None
        self.drop_zone.set_title("Upload Audio")
        self.drop_zone.set_subtitle("Drop audio file or click to browse")
        self.audio_select.configure(values=["No audio loaded"])
        self.audio_select.set("No audio loaded")
        self.source_waveform.clear()
        self.output_waveform.clear()
        self.progress.reset()
        self.save_btn.configure(state="disabled")
        self._update_output_info("The output information will be displayed here.")
