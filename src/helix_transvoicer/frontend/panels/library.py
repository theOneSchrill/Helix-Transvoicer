"""
Helix Transvoicer - Model Library Panel.
"""

import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional

from helix_transvoicer.frontend.styles.theme import HelixTheme
from helix_transvoicer.frontend.utils.api_client import APIClient


class ModelCard(ctk.CTkFrame):
    """Individual model card in the library."""

    def __init__(
        self,
        parent,
        model_data: Dict,
        on_load: callable,
        on_delete: callable,
        **kwargs,
    ):
        super().__init__(
            parent,
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            corner_radius=HelixTheme.RADIUS["md"],
            **kwargs,
        )

        self.model_data = model_data
        self.on_load = on_load
        self.on_delete = on_delete

        self._build_ui()

    def _build_ui(self):
        """Build model card UI."""
        self.grid_columnconfigure(1, weight=1)

        # Icon - different for RVC vs Helix models
        is_rvc = self.model_data.get("model_type") == "rvc"
        icon_text = "🎤" if is_rvc else "🔊"

        icon = ctk.CTkLabel(
            self,
            text=icon_text,
            font=("", 24),
        )
        icon.grid(row=0, column=0, rowspan=2, padx=15, pady=15)

        # Info
        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        info_frame.grid(row=0, column=1, sticky="ew", padx=(0, 15), pady=(15, 5))

        name = ctk.CTkLabel(
            info_frame,
            text=self.model_data.get("name", self.model_data.get("id", "Unknown")),
            font=HelixTheme.FONTS["heading"],
            text_color=HelixTheme.COLORS["text_primary"],
        )
        name.pack(anchor="w")

        # Show model type badge for RVC models
        version_text = f"v{self.model_data.get('version', '1.0.0')}"
        if is_rvc:
            version_text = f"RVC · {version_text}"

        version = ctk.CTkLabel(
            info_frame,
            text=version_text,
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["accent"],
        )
        version.pack(anchor="w")

        # Details
        details_frame = ctk.CTkFrame(self, fg_color="transparent")
        details_frame.grid(row=1, column=1, sticky="ew", padx=(0, 15), pady=(0, 15))

        samples = self.model_data.get("total_samples", 0)
        duration = self.model_data.get("total_duration", 0)
        duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"

        details = ctk.CTkLabel(
            details_frame,
            text=f"Samples: {samples} | Duration: {duration_str}",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        details.pack(anchor="w")

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=0, column=2, rowspan=2, padx=15, pady=15)

        load_btn = ctk.CTkButton(
            btn_frame,
            text="Load",
            width=70,
            height=28,
            **HelixTheme.get_button_style("secondary"),
            command=lambda: self.on_load(self.model_data["id"]),
        )
        load_btn.pack(pady=2)

        delete_btn = ctk.CTkButton(
            btn_frame,
            text="Delete",
            width=70,
            height=28,
            **HelixTheme.get_button_style("ghost"),
            command=lambda: self.on_delete(self.model_data["id"]),
        )
        delete_btn.pack(pady=2)


class LibraryPanel(ctk.CTkFrame):
    """
    Model library panel.

    Shows all local voice models with management options.
    """

    def __init__(self, parent, api_client: APIClient):
        super().__init__(
            parent,
            fg_color=HelixTheme.COLORS["bg_primary"],
            corner_radius=0,
        )

        self.api_client = api_client
        self._models: List[Dict] = []

        self._build_ui()
        self._refresh_models()

    def _build_ui(self):
        """Build the library panel UI."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(1, weight=1)

        # Title row
        title_row = ctk.CTkFrame(self, fg_color="transparent")
        title_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))

        title = ctk.CTkLabel(
            title_row,
            text="MODEL LIBRARY",
            font=HelixTheme.FONTS["heading"],
            text_color=HelixTheme.COLORS["text_primary"],
        )
        title.pack(side="left")

        # Quick actions
        actions = ctk.CTkFrame(title_row, fg_color="transparent")
        actions.pack(side="right")

        new_btn = ctk.CTkButton(
            actions,
            text="+ New Model",
            width=120,
            **HelixTheme.get_button_style("primary"),
            command=self._on_new_model,
        )
        new_btn.pack(side="left", padx=5)

        import_btn = ctk.CTkButton(
            actions,
            text="📥 Import",
            width=100,
            **HelixTheme.get_button_style("secondary"),
            command=self._on_import,
        )
        import_btn.pack(side="left", padx=5)

        refresh_btn = ctk.CTkButton(
            actions,
            text="🔄 Refresh",
            width=100,
            **HelixTheme.get_button_style("ghost"),
            command=self._refresh_models,
        )
        refresh_btn.pack(side="left", padx=5)

        # Model list
        list_frame = ctk.CTkFrame(
            self,
            fg_color=HelixTheme.COLORS["bg_secondary"],
            corner_radius=HelixTheme.RADIUS["md"],
        )
        list_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 20))
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(0, weight=1)

        # Scrollable area
        self.scroll_frame = ctk.CTkScrollableFrame(
            list_frame,
            fg_color="transparent",
        )
        self.scroll_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        # Model details panel
        details_frame = ctk.CTkFrame(
            self,
            fg_color=HelixTheme.COLORS["bg_secondary"],
            corner_radius=HelixTheme.RADIUS["md"],
            width=300,
        )
        details_frame.grid(row=1, column=1, sticky="ns")

        self._build_details_panel(details_frame)

    def _build_details_panel(self, parent):
        """Build model details panel."""
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="both", expand=True, padx=20, pady=20)

        title = ctk.CTkLabel(
            section,
            text="MODEL DETAILS",
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        title.pack(anchor="w", pady=(0, 15))

        self.details_content = ctk.CTkTextbox(
            section,
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            text_color=HelixTheme.COLORS["text_secondary"],
            font=HelixTheme.FONTS["mono"],
            corner_radius=HelixTheme.RADIUS["sm"],
            height=300,
        )
        self.details_content.pack(fill="both", expand=True)
        self.details_content.insert("1.0", "Select a model to view details")
        self.details_content.configure(state="disabled")

        # Actions
        action_frame = ctk.CTkFrame(section, fg_color="transparent")
        action_frame.pack(fill="x", pady=(15, 0))

        export_btn = ctk.CTkButton(
            action_frame,
            text="📤 Export",
            **HelixTheme.get_button_style("secondary"),
            command=self._on_export,
        )
        export_btn.pack(fill="x", pady=2)

        duplicate_btn = ctk.CTkButton(
            action_frame,
            text="📋 Duplicate",
            **HelixTheme.get_button_style("ghost"),
        )
        duplicate_btn.pack(fill="x", pady=2)

    def _refresh_models(self):
        """Refresh model list."""
        # Clear existing cards
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        try:
            self._models = self.api_client.list_models()

            if not self._models:
                empty_label = ctk.CTkLabel(
                    self.scroll_frame,
                    text="No models found.\nCreate a new model or import one.",
                    font=HelixTheme.FONTS["body"],
                    text_color=HelixTheme.COLORS["text_tertiary"],
                )
                empty_label.pack(pady=50)
                return

            for model in self._models:
                card = ModelCard(
                    self.scroll_frame,
                    model,
                    on_load=self._on_load_model,
                    on_delete=self._on_delete_model,
                )
                card.pack(fill="x", pady=5)

        except Exception as e:
            error_label = ctk.CTkLabel(
                self.scroll_frame,
                text=f"Error loading models:\n{str(e)[:50]}",
                font=HelixTheme.FONTS["body"],
                text_color=HelixTheme.COLORS["error"],
            )
            error_label.pack(pady=50)

    def _on_load_model(self, model_id: str):
        """Load a model into memory."""
        try:
            self.api_client.load_model(model_id)
            self._show_model_details(model_id)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def _on_delete_model(self, model_id: str):
        """Delete a model."""
        if messagebox.askyesno(
            "Confirm Delete",
            f"Are you sure you want to delete model '{model_id}'?\n\nThis action cannot be undone.",
        ):
            try:
                self.api_client.delete_model(model_id)
                self._refresh_models()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete model: {e}")

    def _show_model_details(self, model_id: str):
        """Show model details."""
        try:
            model = self.api_client.get_model(model_id)

            details = [
                f"ID: {model.get('id', 'Unknown')}",
                f"Version: {model.get('version', '1.0.0')}",
                f"Created: {model.get('created_at', 'Unknown')[:10]}",
                f"Updated: {model.get('updated_at', 'Unknown')[:10]}",
                "",
                f"Samples: {model.get('total_samples', 0)}",
                f"Duration: {model.get('total_duration', 0):.1f}s",
                f"Quality: {model.get('quality_score', 0) * 100:.0f}%",
                "",
                "Emotion Coverage:",
            ]

            coverage = model.get("emotion_coverage", {})
            for emotion, data in coverage.items():
                if isinstance(data, dict):
                    pct = data.get("coverage", 0) * 100
                else:
                    pct = data * 100
                details.append(f"  {emotion}: {pct:.0f}%")

            self.details_content.configure(state="normal")
            self.details_content.delete("1.0", "end")
            self.details_content.insert("1.0", "\n".join(details))
            self.details_content.configure(state="disabled")

        except Exception:
            pass

    def _on_new_model(self):
        """Create new model."""
        # Simple dialog for model name
        dialog = ctk.CTkInputDialog(
            text="Enter model name:",
            title="New Model",
        )
        name = dialog.get_input()

        if name:
            try:
                self.api_client.create_model(name)
                self._refresh_models()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create model: {e}")

    def _on_import(self):
        """Import model from file."""
        path = filedialog.askopenfilename(
            title="Import Model",
            filetypes=[("Helix Model", "*.zip"), ("All files", "*.*")],
        )

        if path:
            messagebox.showinfo("Import", "Model import not yet implemented.")

    def _on_export(self):
        """Export selected model."""
        messagebox.showinfo("Export", "Model export not yet implemented.")
