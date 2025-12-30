"""
Helix Transvoicer - Status bar component.
"""

import customtkinter as ctk
from typing import Optional

from helix_transvoicer.frontend.styles.theme import HelixTheme
from helix_transvoicer.frontend.utils.api_client import APIClient


class StatusBar(ctk.CTkFrame):
    """
    Status bar showing system status, device info, and current model.
    """

    def __init__(self, parent, api_client: APIClient):
        super().__init__(
            parent,
            fg_color=HelixTheme.COLORS["bg_secondary"],
            height=30,
            corner_radius=0,
        )

        self.api_client = api_client
        self._build_ui()

    def _build_ui(self):
        """Build status bar UI."""
        self.grid_columnconfigure(3, weight=1)

        # Device info
        self.device_label = ctk.CTkLabel(
            self,
            text="Device: Checking...",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        self.device_label.grid(row=0, column=0, padx=20, pady=5)

        # Separator
        sep1 = ctk.CTkLabel(
            self,
            text="|",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        sep1.grid(row=0, column=1, padx=5)

        # Model info
        self.model_label = ctk.CTkLabel(
            self,
            text="Model: None",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        self.model_label.grid(row=0, column=2, padx=10, pady=5)

        # Separator
        sep2 = ctk.CTkLabel(
            self,
            text="|",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        sep2.grid(row=0, column=3, padx=5, sticky="w")

        # Spacer
        spacer = ctk.CTkFrame(self, fg_color="transparent")
        spacer.grid(row=0, column=4, sticky="ew")

        # Status indicator
        self.status_indicator = ctk.CTkLabel(
            self,
            text="●",
            font=HelixTheme.FONTS["body"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        self.status_indicator.grid(row=0, column=5, padx=5)

        self.status_label = ctk.CTkLabel(
            self,
            text="Connecting...",
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        self.status_label.grid(row=0, column=6, padx=(0, 20), pady=5)

    def update(self):
        """Update status bar with current system status."""
        try:
            status = self.api_client.get_status()

            if status.get("status") == "ready":
                self.status_indicator.configure(
                    text_color=HelixTheme.COLORS["success"]
                )
                self.status_label.configure(text="Ready")

                # Update device info
                device = status.get("device", {})
                device_text = f"{device.get('type', 'CPU').upper()}: {device.get('name', 'Unknown')}"

                if device.get("total_memory"):
                    total_gb = device["total_memory"] / (1024**3)
                    used_gb = (device.get("allocated_memory", 0)) / (1024**3)
                    device_text += f" | {used_gb:.1f}/{total_gb:.1f}GB"

                self.device_label.configure(text=device_text)

                # Update model count
                models_loaded = status.get("models_loaded", 0)
                models_total = status.get("models_total", 0)
                self.model_label.configure(
                    text=f"Models: {models_loaded}/{models_total} loaded"
                )

            else:
                self.status_indicator.configure(
                    text_color=HelixTheme.COLORS["error"]
                )
                self.status_label.configure(text="Offline")

        except Exception as e:
            self.status_indicator.configure(
                text_color=HelixTheme.COLORS["error"]
            )
            self.status_label.configure(text="Connection Error")
            self.device_label.configure(text="Device: Unknown")

    def set_model(self, model_name: Optional[str]):
        """Set the current model name."""
        if model_name:
            self.model_label.configure(text=f"Model: {model_name}")
        else:
            self.model_label.configure(text="Model: None")
