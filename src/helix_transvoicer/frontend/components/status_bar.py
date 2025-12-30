"""
Helix Transvoicer - Modern status bar component.
"""

import customtkinter as ctk
from typing import Optional

from helix_transvoicer.frontend.styles.theme import HelixTheme
from helix_transvoicer.frontend.utils.api_client import APIClient


class StatusBar(ctk.CTkFrame):
    """
    Modern status bar showing system status, device info, and metrics.
    """

    def __init__(self, parent, api_client: APIClient):
        super().__init__(
            parent,
            fg_color=HelixTheme.COLORS["bg_secondary"],
            height=36,
            corner_radius=0,
        )

        self.api_client = api_client
        self._build_ui()

    def _build_ui(self):
        """Build status bar UI."""
        self.grid_columnconfigure(2, weight=1)
        self.grid_propagate(False)

        # Left section - Status indicator
        left_frame = ctk.CTkFrame(self, fg_color="transparent")
        left_frame.grid(row=0, column=0, padx=16, pady=0, sticky="w")

        self.status_dot = ctk.CTkLabel(
            left_frame,
            text="●",
            font=("", 10),
            text_color=HelixTheme.COLORS["text_tertiary"],
            width=12,
        )
        self.status_dot.pack(side="left", padx=(0, 6))

        self.status_label = ctk.CTkLabel(
            left_frame,
            text="Connecting...",
            font=HelixTheme.FONTS["tiny"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        self.status_label.pack(side="left")

        # Divider
        self._add_divider(row=0, column=1)

        # Center section - Device info
        center_frame = ctk.CTkFrame(self, fg_color="transparent")
        center_frame.grid(row=0, column=2, sticky="w", padx=16)

        device_icon = ctk.CTkLabel(
            center_frame,
            text="⚡",
            font=("", 11),
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        device_icon.pack(side="left", padx=(0, 6))

        self.device_label = ctk.CTkLabel(
            center_frame,
            text="Detecting device...",
            font=HelixTheme.FONTS["tiny"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        self.device_label.pack(side="left")

        # Spacer
        spacer = ctk.CTkFrame(self, fg_color="transparent")
        spacer.grid(row=0, column=3, sticky="ew")
        self.grid_columnconfigure(3, weight=1)

        # Right section - Memory & Jobs
        right_frame = ctk.CTkFrame(self, fg_color="transparent")
        right_frame.grid(row=0, column=4, padx=16, sticky="e")

        # Memory usage
        self.memory_label = ctk.CTkLabel(
            right_frame,
            text="",
            font=HelixTheme.FONTS["tiny"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        self.memory_label.pack(side="left", padx=(0, 16))

        # Jobs indicator
        self.jobs_label = ctk.CTkLabel(
            right_frame,
            text="",
            font=HelixTheme.FONTS["tiny"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        self.jobs_label.pack(side="left")

    def _add_divider(self, row: int, column: int):
        """Add a subtle divider."""
        divider = ctk.CTkFrame(
            self,
            fg_color=HelixTheme.COLORS["border_subtle"],
            width=1,
            height=16,
        )
        divider.grid(row=row, column=column, padx=8, pady=10)

    def update(self):
        """Update status bar with current system status."""
        try:
            status = self.api_client.get_status()

            if status.get("status") == "ready":
                # Update status indicator
                self.status_dot.configure(text_color=HelixTheme.COLORS["success"])
                self.status_label.configure(
                    text="Online",
                    text_color=HelixTheme.COLORS["success"],
                )

                # Update device info
                device = status.get("device", {})
                device_type = device.get("type", "CPU").upper()
                device_name = device.get("name", "Unknown")

                # Shorten device name if too long
                if len(device_name) > 30:
                    device_name = device_name[:27] + "..."

                self.device_label.configure(
                    text=f"{device_type}: {device_name}",
                    text_color=HelixTheme.COLORS["text_secondary"],
                )

                # Update memory info
                if device.get("total_memory"):
                    total_gb = device["total_memory"] / (1024**3)
                    used_gb = (device.get("allocated_memory", 0)) / (1024**3)
                    free_gb = total_gb - used_gb
                    usage_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0

                    # Color code based on usage
                    if usage_pct > 80:
                        mem_color = HelixTheme.COLORS["error"]
                    elif usage_pct > 60:
                        mem_color = HelixTheme.COLORS["warning"]
                    else:
                        mem_color = HelixTheme.COLORS["text_secondary"]

                    self.memory_label.configure(
                        text=f"Memory: {used_gb:.1f}/{total_gb:.1f}GB ({usage_pct:.0f}%)",
                        text_color=mem_color,
                    )
                else:
                    self.memory_label.configure(text="")

                # Update jobs info
                jobs_running = status.get("jobs_running", 0)
                jobs_pending = status.get("jobs_pending", 0)

                if jobs_running > 0 or jobs_pending > 0:
                    self.jobs_label.configure(
                        text=f"Jobs: {jobs_running} running, {jobs_pending} pending",
                        text_color=HelixTheme.COLORS["accent"],
                    )
                else:
                    self.jobs_label.configure(text="")

            else:
                self._set_offline()

        except Exception:
            self._set_offline()

    def _set_offline(self):
        """Set status to offline."""
        self.status_dot.configure(text_color=HelixTheme.COLORS["error"])
        self.status_label.configure(
            text="Offline",
            text_color=HelixTheme.COLORS["error"],
        )
        self.device_label.configure(
            text="Backend not connected",
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        self.memory_label.configure(text="")
        self.jobs_label.configure(text="")

    def set_model(self, model_name: Optional[str]):
        """Set the current model name (for future use)."""
        pass
