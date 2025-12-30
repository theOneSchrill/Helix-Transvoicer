"""
Helix Transvoicer - Emotion Coverage Panel.
"""

import customtkinter as ctk
from typing import Dict, List, Optional

from helix_transvoicer.frontend.styles.theme import HelixTheme
from helix_transvoicer.frontend.utils.api_client import APIClient
from helix_transvoicer.frontend.components.controls import DropdownSelect


class EmotionsPanel(ctk.CTkFrame):
    """
    Emotion coverage analysis dashboard.

    Shows:
    - Emotion spectrum visualization
    - Coverage per emotion
    - Gaps and recommendations
    """

    EMOTIONS = [
        ("neutral", "Neutral"),
        ("happy", "Happy"),
        ("sad", "Sad"),
        ("angry", "Angry"),
        ("fear", "Fear"),
        ("surprise", "Surprise"),
        ("disgust", "Disgust"),
        ("calm", "Calm"),
        ("excited", "Excited"),
    ]

    def __init__(self, parent, api_client: APIClient):
        super().__init__(
            parent,
            fg_color=HelixTheme.COLORS["bg_primary"],
            corner_radius=0,
        )

        self.api_client = api_client
        self._coverage_data: Dict = {}

        self._build_ui()
        self._refresh_models()

    def _build_ui(self):
        """Build the emotions panel UI."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        # Title row
        title_row = ctk.CTkFrame(self, fg_color="transparent")
        title_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))

        title = ctk.CTkLabel(
            title_row,
            text="EMOTION MAP",
            font=HelixTheme.FONTS["heading"],
            text_color=HelixTheme.COLORS["text_primary"],
        )
        title.pack(side="left")

        # Model selector
        self.model_select = DropdownSelect(
            title_row,
            label="",
            options=["Loading..."],
            width=200,
            command=self._on_model_change,
        )
        self.model_select.pack(side="right", padx=20)

        # Main visualization
        main_frame = ctk.CTkFrame(
            self,
            fg_color=HelixTheme.COLORS["bg_secondary"],
            corner_radius=HelixTheme.RADIUS["md"],
        )
        main_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 20))

        self._build_spectrum(main_frame)

        # Side panel
        side_frame = ctk.CTkFrame(
            self,
            fg_color=HelixTheme.COLORS["bg_secondary"],
            corner_radius=HelixTheme.RADIUS["md"],
            width=350,
        )
        side_frame.grid(row=1, column=1, sticky="ns")

        self._build_coverage_list(side_frame)
        self._build_recommendations(side_frame)

    def _build_spectrum(self, parent):
        """Build emotion spectrum visualization."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="both", expand=True, padx=30, pady=30)

        title = ctk.CTkLabel(
            section,
            text="EMOTION SPECTRUM ANALYSIS",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(anchor="w", pady=(0, 20))

        # Valence-Arousal grid (simplified)
        grid_frame = ctk.CTkFrame(
            section,
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            corner_radius=HelixTheme.RADIUS["md"],
            width=500,
            height=400,
        )
        grid_frame.pack(expand=True)
        grid_frame.pack_propagate(False)

        # Canvas for visualization
        self.spectrum_canvas = ctk.CTkCanvas(
            grid_frame,
            bg=HelixTheme.COLORS["bg_tertiary"],
            highlightthickness=0,
            width=480,
            height=380,
        )
        self.spectrum_canvas.pack(padx=10, pady=10)

        self._draw_spectrum_grid()

    def _draw_spectrum_grid(self):
        """Draw the valence-arousal grid."""
        canvas = self.spectrum_canvas
        w, h = 480, 380
        cx, cy = w // 2, h // 2

        # Draw axes
        canvas.create_line(
            0, cy, w, cy,
            fill=HelixTheme.COLORS["border"],
            width=1,
        )
        canvas.create_line(
            cx, 0, cx, h,
            fill=HelixTheme.COLORS["border"],
            width=1,
        )

        # Labels
        canvas.create_text(
            w - 10, cy - 10,
            text="Positive",
            fill=HelixTheme.COLORS["text_tertiary"],
            font=HelixTheme.FONTS["small"],
            anchor="e",
        )
        canvas.create_text(
            10, cy - 10,
            text="Negative",
            fill=HelixTheme.COLORS["text_tertiary"],
            font=HelixTheme.FONTS["small"],
            anchor="w",
        )
        canvas.create_text(
            cx + 10, 10,
            text="High Arousal",
            fill=HelixTheme.COLORS["text_tertiary"],
            font=HelixTheme.FONTS["small"],
            anchor="w",
        )
        canvas.create_text(
            cx + 10, h - 10,
            text="Low Arousal",
            fill=HelixTheme.COLORS["text_tertiary"],
            font=HelixTheme.FONTS["small"],
            anchor="w",
        )

        # Emotion positions (approximate)
        positions = {
            "neutral": (0.0, 0.0),
            "happy": (0.6, 0.4),
            "sad": (-0.5, -0.3),
            "angry": (-0.4, 0.6),
            "fear": (-0.6, 0.5),
            "surprise": (0.1, 0.6),
            "disgust": (-0.5, 0.1),
            "calm": (0.2, -0.5),
            "excited": (0.5, 0.7),
        }

        for emotion, (valence, arousal) in positions.items():
            x = cx + int(valence * (w // 2 - 40))
            y = cy - int(arousal * (h // 2 - 40))

            # Draw emotion bubble
            coverage = self._coverage_data.get(emotion, {}).get("coverage", 0)
            radius = 20 + int(coverage * 15)
            alpha = int(0.3 + coverage * 0.7)

            color = HelixTheme.COLORS.get(f"emotion_{emotion}", HelixTheme.COLORS["accent"])

            canvas.create_oval(
                x - radius, y - radius,
                x + radius, y + radius,
                fill=color,
                outline="",
            )

            canvas.create_text(
                x, y,
                text=emotion.upper()[:3],
                fill=HelixTheme.COLORS["text_primary"],
                font=HelixTheme.FONTS["small"],
            )

    def _build_coverage_list(self, parent):
        """Build detailed coverage list."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=20, pady=20)

        title = ctk.CTkLabel(
            section,
            text="DETAILED COVERAGE",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(anchor="w", pady=(0, 15))

        self.coverage_widgets = {}

        for key, label in self.EMOTIONS:
            row = ctk.CTkFrame(section, fg_color="transparent")
            row.pack(fill="x", pady=3)

            name = ctk.CTkLabel(
                row,
                text=label,
                font=HelixTheme.FONTS["small"],
                text_color=HelixTheme.COLORS["text_secondary"],
                width=80,
                anchor="w",
            )
            name.pack(side="left")

            bar = ctk.CTkProgressBar(
                row,
                width=150,
                height=12,
                fg_color=HelixTheme.COLORS["bg_tertiary"],
                progress_color=HelixTheme.COLORS["accent"],
            )
            bar.set(0)
            bar.pack(side="left", padx=5)

            pct = ctk.CTkLabel(
                row,
                text="0%",
                font=HelixTheme.FONTS["mono"],
                text_color=HelixTheme.COLORS["text_tertiary"],
                width=45,
                anchor="e",
            )
            pct.pack(side="left")

            self.coverage_widgets[key] = (bar, pct)

    def _build_recommendations(self, parent):
        """Build recommendations section."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="both", expand=True, padx=20, pady=20)

        title = ctk.CTkLabel(
            section,
            text="RECOMMENDATIONS",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(anchor="w", pady=(0, 10))

        self.recommendations_text = ctk.CTkTextbox(
            section,
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            text_color=HelixTheme.COLORS["text_secondary"],
            font=HelixTheme.FONTS["small"],
            corner_radius=HelixTheme.RADIUS["sm"],
            height=150,
        )
        self.recommendations_text.pack(fill="both", expand=True)
        self.recommendations_text.insert("1.0", "Select a model to see recommendations...")
        self.recommendations_text.configure(state="disabled")

        # Buttons
        btn_frame = ctk.CTkFrame(section, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(10, 0))

        refresh_btn = ctk.CTkButton(
            btn_frame,
            text="🔄 Re-analyze",
            **HelixTheme.get_button_style("secondary"),
            command=self._refresh_coverage,
        )
        refresh_btn.pack(side="left", padx=(0, 10))

        export_btn = ctk.CTkButton(
            btn_frame,
            text="📊 Export Report",
            **HelixTheme.get_button_style("ghost"),
        )
        export_btn.pack(side="left")

    def _refresh_models(self):
        """Refresh model list."""
        try:
            models = self.api_client.list_models()
            names = [m["id"] for m in models] if models else ["No models"]
            self.model_select.update_options(names)
        except Exception:
            self.model_select.update_options(["Error"])

    def _on_model_change(self, model_id: str):
        """Handle model selection change."""
        self._refresh_coverage()

    def _refresh_coverage(self):
        """Refresh emotion coverage data."""
        model_id = self.model_select.get()
        if not model_id or model_id in ["No models", "Loading...", "Error"]:
            return

        try:
            data = self.api_client.get_emotion_coverage(model_id)
            self._coverage_data = data.get("coverage", {})
            self._update_display()
        except Exception as e:
            self._set_recommendations([f"Error: {str(e)[:50]}"])

    def _update_display(self):
        """Update coverage display."""
        gaps = []

        for key, (bar, pct_label) in self.coverage_widgets.items():
            emotion_data = self._coverage_data.get(key, {})
            coverage = emotion_data.get("coverage", 0)

            bar.set(coverage)
            pct_label.configure(text=f"{int(coverage * 100)}%")

            if coverage < 0.3:
                bar.configure(progress_color=HelixTheme.COLORS["error"])
                gaps.append(key)
            elif coverage < 0.6:
                bar.configure(progress_color=HelixTheme.COLORS["warning"])
            else:
                bar.configure(progress_color=HelixTheme.COLORS["success"])

        # Update spectrum
        self.spectrum_canvas.delete("all")
        self._draw_spectrum_grid()

        # Update recommendations
        if gaps:
            recs = [f"• Add samples for: {', '.join(gaps)}"]
            recs.append(f"• {len(gaps)} emotion(s) need more coverage")
        else:
            recs = ["✓ Good emotion coverage across all categories"]

        self._set_recommendations(recs)

    def _set_recommendations(self, recommendations: List[str]):
        """Set recommendations text."""
        self.recommendations_text.configure(state="normal")
        self.recommendations_text.delete("1.0", "end")
        self.recommendations_text.insert("1.0", "\n".join(recommendations))
        self.recommendations_text.configure(state="disabled")
