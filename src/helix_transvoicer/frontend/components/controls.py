"""
Helix Transvoicer - Custom control components.
"""

import customtkinter as ctk
from typing import Callable, List, Optional, Tuple

from helix_transvoicer.frontend.styles.theme import HelixTheme


class Slider(ctk.CTkFrame):
    """
    Labeled slider with value display.
    """

    def __init__(
        self,
        parent,
        label: str,
        from_: float = 0.0,
        to: float = 1.0,
        default: float = 0.5,
        format_str: str = "{:.2f}",
        width: int = 250,
        command: Optional[Callable[[float], None]] = None,
        **kwargs,
    ):
        super().__init__(
            parent,
            fg_color="transparent",
            **kwargs,
        )

        self._from = from_
        self._to = to
        self._format_str = format_str
        self._command = command

        self.grid_columnconfigure(1, weight=1)

        # Label
        self.label = ctk.CTkLabel(
            self,
            text=label,
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
            width=80,
            anchor="w",
        )
        self.label.grid(row=0, column=0, padx=(0, 10))

        # Slider
        self.slider = ctk.CTkSlider(
            self,
            from_=from_,
            to=to,
            width=width - 130,
            height=16,
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            progress_color=HelixTheme.COLORS["accent"],
            button_color=HelixTheme.COLORS["accent"],
            button_hover_color=HelixTheme.COLORS["accent_hover"],
            command=self._on_change,
        )
        self.slider.set(default)
        self.slider.grid(row=0, column=1, sticky="ew")

        # Value display
        self.value_label = ctk.CTkLabel(
            self,
            text=format_str.format(default),
            font=HelixTheme.FONTS["mono"],
            text_color=HelixTheme.COLORS["text_primary"],
            width=50,
            anchor="e",
        )
        self.value_label.grid(row=0, column=2, padx=(10, 0))

    def _on_change(self, value: float):
        """Handle slider value change."""
        self.value_label.configure(text=self._format_str.format(value))
        if self._command:
            self._command(value)

    def get(self) -> float:
        """Get current value."""
        return self.slider.get()

    def set(self, value: float):
        """Set slider value."""
        self.slider.set(value)
        self.value_label.configure(text=self._format_str.format(value))


class DropdownSelect(ctk.CTkFrame):
    """
    Labeled dropdown selector.
    """

    def __init__(
        self,
        parent,
        label: str,
        options: List[str],
        default: Optional[str] = None,
        width: int = 200,
        command: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(
            parent,
            fg_color="transparent",
            **kwargs,
        )

        self._command = command

        # Label
        self.label = ctk.CTkLabel(
            self,
            text=label,
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
            anchor="w",
        )
        self.label.grid(row=0, column=0, sticky="w", pady=(0, 5))

        # Dropdown
        self.dropdown = ctk.CTkOptionMenu(
            self,
            values=options or ["None"],
            width=width,
            height=32,
            font=HelixTheme.FONTS["body"],
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            button_color=HelixTheme.COLORS["bg_hover"],
            button_hover_color=HelixTheme.COLORS["bg_active"],
            dropdown_fg_color=HelixTheme.COLORS["bg_tertiary"],
            dropdown_hover_color=HelixTheme.COLORS["bg_hover"],
            corner_radius=HelixTheme.RADIUS["sm"],
            command=self._on_select,
        )
        if default and default in options:
            self.dropdown.set(default)
        elif options:
            self.dropdown.set(options[0])
        self.dropdown.grid(row=1, column=0, sticky="w")

    def _on_select(self, value: str):
        """Handle selection change."""
        if self._command:
            self._command(value)

    def get(self) -> str:
        """Get selected value."""
        return self.dropdown.get()

    def set(self, value: str):
        """Set selected value."""
        self.dropdown.set(value)

    def update_options(self, options: List[str], keep_selection: bool = True):
        """Update available options."""
        current = self.dropdown.get() if keep_selection else None
        self.dropdown.configure(values=options or ["None"])

        if keep_selection and current in options:
            self.dropdown.set(current)
        elif options:
            self.dropdown.set(options[0])


class Toggle(ctk.CTkFrame):
    """
    Labeled toggle switch.
    """

    def __init__(
        self,
        parent,
        label: str,
        default: bool = False,
        command: Optional[Callable[[bool], None]] = None,
        **kwargs,
    ):
        super().__init__(
            parent,
            fg_color="transparent",
            **kwargs,
        )

        self._command = command

        # Switch
        self.switch = ctk.CTkSwitch(
            self,
            text=label,
            font=HelixTheme.FONTS["small"],
            text_color=HelixTheme.COLORS["text_secondary"],
            fg_color=HelixTheme.COLORS["bg_tertiary"],
            progress_color=HelixTheme.COLORS["accent"],
            button_color=HelixTheme.COLORS["text_primary"],
            button_hover_color=HelixTheme.COLORS["accent_hover"],
            command=self._on_toggle,
        )
        if default:
            self.switch.select()
        self.switch.pack()

    def _on_toggle(self):
        """Handle toggle change."""
        if self._command:
            self._command(self.get())

    def get(self) -> bool:
        """Get current state."""
        return self.switch.get() == 1

    def set(self, value: bool):
        """Set toggle state."""
        if value:
            self.switch.select()
        else:
            self.switch.deselect()
