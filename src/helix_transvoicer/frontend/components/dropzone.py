"""
Helix Transvoicer - Drag & Drop Zone Component.

A modern drop zone that supports drag & drop file input with visual feedback.
"""

import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog
from typing import Callable, List, Optional, Tuple

from helix_transvoicer.frontend.styles.theme import HelixTheme


# Try to import tkinterdnd2 for drag & drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False


class DropZone(ctk.CTkFrame):
    """
    A drag & drop zone for file input.

    Features:
    - Drag & drop file support (when tkinterdnd2 is available)
    - Click to browse fallback
    - Visual feedback on hover
    - File type filtering
    - Single or multiple file support
    """

    def __init__(
        self,
        parent,
        on_files_dropped: Callable[[List[Path]], None],
        filetypes: Optional[List[Tuple[str, str]]] = None,
        multiple: bool = True,
        width: int = 400,
        height: int = 120,
        title: str = "Drop files here",
        subtitle: str = "or click to browse",
        icon: str = "📁",
    ):
        """
        Initialize the drop zone.

        Args:
            parent: Parent widget
            on_files_dropped: Callback when files are dropped/selected
            filetypes: List of (description, pattern) tuples for file dialog
            multiple: Allow multiple file selection
            width: Width of the drop zone
            height: Height of the drop zone
            title: Main text shown in the drop zone
            subtitle: Secondary text shown below title
            icon: Icon to display
        """
        super().__init__(
            parent,
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            corner_radius=HelixTheme.RADIUS["md"],
            width=width,
            height=height,
        )

        self._on_files_dropped = on_files_dropped
        self._filetypes = filetypes or [("All files", "*.*")]
        self._multiple = multiple
        self._title = title
        self._subtitle = subtitle
        self._icon = icon
        self._is_hovering = False

        self._build_ui()
        self._setup_dnd()

    def _build_ui(self):
        """Build the drop zone UI."""
        # Make the frame not shrink to content
        self.pack_propagate(False)
        self.grid_propagate(False)

        # Main container with dashed border effect
        self.inner_frame = ctk.CTkFrame(
            self,
            fg_color="transparent",
            corner_radius=HelixTheme.RADIUS["sm"],
        )
        self.inner_frame.pack(fill="both", expand=True, padx=3, pady=3)

        # Center content
        self.content_frame = ctk.CTkFrame(
            self.inner_frame,
            fg_color="transparent",
        )
        self.content_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Icon
        self.icon_label = ctk.CTkLabel(
            self.content_frame,
            text=self._icon,
            font=("", 32),
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        self.icon_label.pack()

        # Title
        self.title_label = ctk.CTkLabel(
            self.content_frame,
            text=self._title,
            font=HelixTheme.FONTS["button"],
            text_color=HelixTheme.COLORS["text_secondary"],
        )
        self.title_label.pack(pady=(5, 0))

        # Subtitle
        self.subtitle_label = ctk.CTkLabel(
            self.content_frame,
            text=self._subtitle,
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_tertiary"],
        )
        self.subtitle_label.pack()

        # Bind click events to all components
        for widget in [self, self.inner_frame, self.content_frame,
                       self.icon_label, self.title_label, self.subtitle_label]:
            widget.bind("<Button-1>", self._on_click)
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)

        # Configure cursor
        self._set_cursor("hand2")

    def _setup_dnd(self):
        """Setup drag & drop if available."""
        if not DND_AVAILABLE:
            return

        try:
            # Register for drag & drop
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)
            self.dnd_bind("<<DragEnter>>", self._on_drag_enter)
            self.dnd_bind("<<DragLeave>>", self._on_drag_leave)
        except Exception:
            # DnD might not be fully available
            pass

    def _set_cursor(self, cursor: str):
        """Set cursor for all child widgets."""
        for widget in [self, self.inner_frame, self.content_frame,
                       self.icon_label, self.title_label, self.subtitle_label]:
            try:
                widget.configure(cursor=cursor)
            except Exception:
                pass

    def _on_click(self, event=None):
        """Handle click to browse."""
        if self._multiple:
            paths = filedialog.askopenfilenames(
                title="Select Files",
                filetypes=self._filetypes,
            )
        else:
            path = filedialog.askopenfilename(
                title="Select File",
                filetypes=self._filetypes,
            )
            paths = [path] if path else []

        if paths:
            self._on_files_dropped([Path(p) for p in paths])

    def _on_enter(self, event=None):
        """Handle mouse enter."""
        if not self._is_hovering:
            self._is_hovering = True
            self._set_hover_state(True)

    def _on_leave(self, event=None):
        """Handle mouse leave."""
        # Check if we're still within the widget bounds
        try:
            x, y = self.winfo_pointerxy()
            widget_x = self.winfo_rootx()
            widget_y = self.winfo_rooty()
            widget_w = self.winfo_width()
            widget_h = self.winfo_height()

            if not (widget_x <= x <= widget_x + widget_w and
                    widget_y <= y <= widget_y + widget_h):
                self._is_hovering = False
                self._set_hover_state(False)
        except Exception:
            self._is_hovering = False
            self._set_hover_state(False)

    def _set_hover_state(self, hovering: bool):
        """Update visual state for hover."""
        if hovering:
            self.configure(fg_color=HelixTheme.COLORS["bg_hover"])
            self.icon_label.configure(text_color=HelixTheme.COLORS["accent"])
            self.title_label.configure(text_color=HelixTheme.COLORS["text_primary"])
        else:
            self.configure(fg_color=HelixTheme.COLORS["bg_tertiary"])
            self.icon_label.configure(text_color=HelixTheme.COLORS["text_tertiary"])
            self.title_label.configure(text_color=HelixTheme.COLORS["text_secondary"])

    def _on_drag_enter(self, event=None):
        """Handle drag enter."""
        self._set_drag_state(True)
        return event.action if hasattr(event, 'action') else None

    def _on_drag_leave(self, event=None):
        """Handle drag leave."""
        self._set_drag_state(False)

    def _set_drag_state(self, dragging: bool):
        """Update visual state for drag."""
        if dragging:
            self.configure(
                fg_color=HelixTheme.COLORS["accent_dim"],
            )
            self.icon_label.configure(text_color=HelixTheme.COLORS["text_primary"])
            self.title_label.configure(
                text="Release to drop",
                text_color=HelixTheme.COLORS["text_primary"],
            )
        else:
            self.configure(fg_color=HelixTheme.COLORS["bg_tertiary"])
            self.icon_label.configure(text_color=HelixTheme.COLORS["text_tertiary"])
            self.title_label.configure(
                text=self._title,
                text_color=HelixTheme.COLORS["text_secondary"],
            )

    def _on_drop(self, event):
        """Handle file drop."""
        self._set_drag_state(False)

        # Parse dropped files
        data = event.data
        paths = []

        # Handle different formats
        if data.startswith("{"):
            # Windows format with braces for paths with spaces
            import re
            paths = re.findall(r'\{([^}]+)\}|(\S+)', data)
            paths = [Path(p[0] or p[1]) for p in paths]
        else:
            # Unix format or simple paths
            for path in data.split():
                paths.append(Path(path))

        # Filter to valid files
        valid_paths = [p for p in paths if p.exists() and p.is_file()]

        # Apply file type filter
        if self._filetypes and self._filetypes[0][1] != "*.*":
            extensions = set()
            for _, pattern in self._filetypes:
                for ext in pattern.split():
                    ext = ext.replace("*", "").lower()
                    if ext:
                        extensions.add(ext)

            if extensions:
                valid_paths = [p for p in valid_paths
                               if p.suffix.lower() in extensions]

        if valid_paths:
            if not self._multiple:
                valid_paths = valid_paths[:1]
            self._on_files_dropped(valid_paths)

    def set_title(self, title: str):
        """Update the title text."""
        self._title = title
        self.title_label.configure(text=title)

    def set_subtitle(self, subtitle: str):
        """Update the subtitle text."""
        self._subtitle = subtitle
        self.subtitle_label.configure(text=subtitle)

    def set_icon(self, icon: str):
        """Update the icon."""
        self._icon = icon
        self.icon_label.configure(text=icon)

    def flash_success(self):
        """Flash green to indicate successful drop."""
        original_color = HelixTheme.COLORS["bg_tertiary"]
        self.configure(fg_color=HelixTheme.COLORS["success"])
        self.after(200, lambda: self.configure(fg_color=original_color))


def enable_dnd_for_app(root):
    """
    Enable drag & drop for the entire application.

    Must be called before creating the root window.
    Returns a TkinterDnD.Tk instance if available, else regular Tk.
    """
    if DND_AVAILABLE:
        try:
            return TkinterDnD.Tk()
        except Exception:
            pass
    return None
