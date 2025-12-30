"""
Helix Transvoicer - TTS Studio Panel.
"""

import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog
from typing import Optional

from helix_transvoicer.frontend.styles.theme import HelixTheme
from helix_transvoicer.frontend.utils.api_client import APIClient
from helix_transvoicer.frontend.components.waveform import WaveformDisplay
from helix_transvoicer.frontend.components.controls import Slider, DropdownSelect


class TTSPanel(ctk.CTkFrame):
    """
    Text-to-Speech studio panel.

    Features:
    - Text input
    - Voice selection
    - Emotion control
    - Speed, pitch, intensity controls
    - Preview and save
    """

    EMOTIONS = [
        "neutral", "happy", "sad", "angry",
        "fear", "surprise", "disgust", "calm", "excited"
    ]

    def __init__(self, parent, api_client: APIClient):
        super().__init__(
            parent,
            fg_color=HelixTheme.COLORS["bg_primary"],
            corner_radius=0,
        )

        self.api_client = api_client
        self._output_audio: Optional[bytes] = None

        self._build_ui()
        self._refresh_voices()

    def _build_ui(self):
        """Build the TTS panel UI."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        # Title
        title = ctk.CTkLabel(
            self,
            text="TTS STUDIO",
            font=HelixTheme.FONTS["heading"],
            text_color=HelixTheme.COLORS["text_primary"],
        )
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 20))

        # Main content
        main_frame = ctk.CTkFrame(
            self,
            fg_color=HelixTheme.COLORS["bg_secondary"],
            corner_radius=HelixTheme.RADIUS["md"],
        )
        main_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 20))
        main_frame.grid_columnconfigure(0, weight=1)

        # Text input
        self._build_text_section(main_frame)

        # Output section
        self._build_output_section(main_frame)

        # Side panel
        side_frame = ctk.CTkFrame(
            self,
            fg_color=HelixTheme.COLORS["bg_secondary"],
            corner_radius=HelixTheme.RADIUS["md"],
            width=320,
        )
        side_frame.grid(row=1, column=1, sticky="ns")

        # Voice selection
        self._build_voice_section(side_frame)

        # Voice parameters
        self._build_params_section(side_frame)

        # Emotion control
        self._build_emotion_section(side_frame)

        # Synthesize button
        self.synth_btn = ctk.CTkButton(
            side_frame,
            text="▶ SYNTHESIZE",
            font=HelixTheme.FONTS["button"],
            height=50,
            **HelixTheme.get_button_style("primary"),
            command=self._on_synthesize,
        )
        self.synth_btn.pack(fill="x", padx=20, pady=20)

    def _build_text_section(self, parent):
        """Build text input section."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="both", expand=True, padx=20, pady=20)

        # Title row
        title_row = ctk.CTkFrame(section, fg_color="transparent")
        title_row.pack(fill="x")

        title = ctk.CTkLabel(
            title_row,
            text="TEXT INPUT",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(side="left")

        self.char_count = ctk.CTkLabel(
            title_row,
            text="0 characters",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        self.char_count.pack(side="right")

        # Text input
        self.text_input = ctk.CTkTextbox(
            section,
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            text_color=HelixTheme.COLORS["text_primary"],
            font=HelixTheme.FONTS["body"],
            corner_radius=HelixTheme.RADIUS["sm"],
            height=200,
        )
        self.text_input.pack(fill="both", expand=True, pady=10)
        self.text_input.bind("<KeyRelease>", self._on_text_change)

        # Placeholder
        self.text_input.insert(
            "1.0",
            "Enter text to synthesize...\n\nThe quick brown fox jumps over the lazy dog."
        )

    def _build_output_section(self, parent):
        """Build output section."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=20, pady=20)

        title = ctk.CTkLabel(
            section,
            text="OUTPUT WAVEFORM",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(anchor="w", pady=(0, 10))

        # Waveform
        self.waveform = WaveformDisplay(section, width=600, height=80)
        self.waveform.pack(fill="x", pady=5)

        # Controls
        controls = ctk.CTkFrame(section, fg_color="transparent")
        controls.pack(fill="x", pady=5)

        preview_btn = ctk.CTkButton(
            controls,
            text="🔊 Preview",
            width=100,
            **HelixTheme.get_button_style("secondary"),
            command=self._on_preview,
        )
        preview_btn.pack(side="left", padx=(0, 10))

        self.save_btn = ctk.CTkButton(
            controls,
            text="💾 Save Audio",
            width=120,
            **HelixTheme.get_button_style("secondary"),
            command=self._on_save,
            state="disabled",
        )
        self.save_btn.pack(side="left")

        self.duration_label = ctk.CTkLabel(
            controls,
            text="",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        self.duration_label.pack(side="right")

    def _build_voice_section(self, parent):
        """Build voice selection."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=20, pady=20)

        title = ctk.CTkLabel(
            section,
            text="VOICE SELECTION",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(anchor="w", pady=(0, 10))

        self.voice_select = DropdownSelect(
            section,
            label="Voice Model",
            options=["Loading..."],
            width=280,
        )
        self.voice_select.pack(fill="x")

        refresh_btn = ctk.CTkButton(
            section,
            text="🔄 Refresh",
            height=28,
            **HelixTheme.get_button_style("ghost"),
            command=self._refresh_voices,
        )
        refresh_btn.pack(anchor="w", pady=5)

    def _build_params_section(self, parent):
        """Build voice parameters."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=20, pady=20)

        title = ctk.CTkLabel(
            section,
            text="VOICE PARAMETERS",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(anchor="w", pady=(0, 15))

        # Speed
        self.speed_slider = Slider(
            section,
            label="Speed:",
            from_=0.5,
            to=2.0,
            default=1.0,
            format_str="{:.1f}x",
            width=280,
        )
        self.speed_slider.pack(fill="x", pady=5)

        # Pitch
        self.pitch_slider = Slider(
            section,
            label="Pitch:",
            from_=-12,
            to=12,
            default=0,
            format_str="{:+.0f} st",
            width=280,
        )
        self.pitch_slider.pack(fill="x", pady=5)

        # Intensity
        self.intensity_slider = Slider(
            section,
            label="Intensity:",
            from_=0,
            to=1,
            default=0.5,
            format_str="{:.2f}",
            width=280,
        )
        self.intensity_slider.pack(fill="x", pady=5)

        # Options
        options = ctk.CTkFrame(section, fg_color="transparent")
        options.pack(fill="x", pady=10)

        self.breathing_var = ctk.BooleanVar(value=True)
        breathing_cb = ctk.CTkCheckBox(
            options,
            text="Breathing sounds",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
            variable=self.breathing_var,
            fg_color=HelixTheme.COLORS["accent"],
        )
        breathing_cb.pack(anchor="w", pady=2)

        self.whisper_var = ctk.BooleanVar(value=False)
        whisper_cb = ctk.CTkCheckBox(
            options,
            text="Whisper mode",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
            variable=self.whisper_var,
            fg_color=HelixTheme.COLORS["accent"],
        )
        whisper_cb.pack(anchor="w", pady=2)

    def _build_emotion_section(self, parent):
        """Build emotion control."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=20, pady=20)

        title = ctk.CTkLabel(
            section,
            text="EMOTION CONTROL",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(anchor="w", pady=(0, 10))

        # Primary emotion
        self.emotion_select = DropdownSelect(
            section,
            label="Primary Emotion",
            options=[e.capitalize() for e in self.EMOTIONS],
            default="Neutral",
            width=280,
        )
        self.emotion_select.pack(fill="x")

        # Emotion strength
        self.emotion_strength = Slider(
            section,
            label="Strength:",
            from_=0,
            to=1,
            default=0.5,
            format_str="{:.0%}",
            width=280,
        )
        self.emotion_strength.pack(fill="x", pady=10)

    def _on_text_change(self, event=None):
        """Handle text change."""
        text = self.text_input.get("1.0", "end-1c")
        self.char_count.configure(text=f"{len(text)} characters")

    def _refresh_voices(self):
        """Refresh available voices."""
        try:
            voices = self.api_client.list_voices()
            names = [v["id"] for v in voices] if voices else ["No voices"]
            self.voice_select.update_options(names)
        except Exception:
            self.voice_select.update_options(["Error"])

    def _on_synthesize(self):
        """Synthesize speech."""
        text = self.text_input.get("1.0", "end-1c").strip()
        if not text:
            return

        voice_id = self.voice_select.get()
        if not voice_id or voice_id in ["No voices", "Loading...", "Error"]:
            return

        self.synth_btn.configure(state="disabled")

        try:
            self._output_audio = self.api_client.synthesize_speech(
                text=text,
                voice_model_id=voice_id,
                speed=self.speed_slider.get(),
                pitch=self.pitch_slider.get(),
                emotion=self.emotion_select.get().lower(),
                emotion_strength=self.emotion_strength.get(),
            )

            self.save_btn.configure(state="normal")
            self.duration_label.configure(text="Synthesis complete")

        except Exception as e:
            self.duration_label.configure(
                text=f"Error: {str(e)[:30]}",
                text_color=HelixTheme.COLORS["error"],
            )

        finally:
            self.synth_btn.configure(state="normal")

    def _on_preview(self):
        """Quick preview."""
        text = self.text_input.get("1.0", "end-1c").strip()[:200]
        if not text:
            return

        voice_id = self.voice_select.get()
        if not voice_id or voice_id in ["No voices", "Loading...", "Error"]:
            return

        try:
            self.api_client.preview_tts(
                text=text,
                voice_model_id=voice_id,
                emotion=self.emotion_select.get().lower(),
            )
        except Exception:
            pass

    def _on_save(self):
        """Save synthesized audio."""
        if not self._output_audio:
            return

        path = filedialog.asksaveasfilename(
            title="Save Synthesized Audio",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
        )

        if path:
            with open(path, "wb") as f:
                f.write(self._output_audio)
