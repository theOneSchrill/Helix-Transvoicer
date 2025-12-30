"""
Helix Transvoicer - UI Theme configuration.

Industrial, dark, expert-oriented design.
"""

import customtkinter as ctk


class HelixTheme:
    """
    Helix Transvoicer theme configuration.

    A dark, industrial, professional theme for studio-grade software.
    """

    # Color palette
    COLORS = {
        # Backgrounds
        "bg_primary": "#0D0D0D",
        "bg_secondary": "#1A1A1A",
        "bg_tertiary": "#252525",
        "bg_hover": "#2D2D2D",
        "bg_active": "#353535",

        # Accent colors
        "accent": "#00D4AA",
        "accent_hover": "#00E5BB",
        "accent_dim": "#00A080",

        # Status colors
        "success": "#00D46A",
        "warning": "#FFB800",
        "error": "#FF4757",
        "info": "#3498DB",

        # Text colors
        "text_primary": "#FFFFFF",
        "text_secondary": "#A0A0A0",
        "text_tertiary": "#666666",
        "text_disabled": "#444444",

        # Border colors
        "border": "#333333",
        "border_active": "#00D4AA",

        # Progress colors
        "progress_bg": "#252525",
        "progress_fill": "#00D4AA",

        # Emotion colors (for visualization)
        "emotion_neutral": "#808080",
        "emotion_happy": "#FFD700",
        "emotion_sad": "#4169E1",
        "emotion_angry": "#DC143C",
        "emotion_fear": "#8B008B",
        "emotion_surprise": "#FF8C00",
        "emotion_disgust": "#556B2F",
        "emotion_calm": "#20B2AA",
        "emotion_excited": "#FF1493",
    }

    # Font configuration
    FONTS = {
        "title": ("JetBrains Mono", 16, "bold"),
        "heading": ("JetBrains Mono", 14, "bold"),
        "body": ("JetBrains Mono", 12),
        "button": ("JetBrains Mono", 11, "bold"),
        "small": ("JetBrains Mono", 10),
        "mono": ("JetBrains Mono", 11),
        "data": ("JetBrains Mono", 10),
    }

    # Spacing
    SPACING = {
        "xs": 4,
        "sm": 8,
        "md": 16,
        "lg": 24,
        "xl": 32,
    }

    # Border radius
    RADIUS = {
        "none": 0,
        "sm": 4,
        "md": 8,
        "lg": 12,
    }

    @classmethod
    def apply(cls):
        """Apply the Helix theme to customtkinter."""
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

    @classmethod
    def get_button_style(cls, variant: str = "primary") -> dict:
        """Get button style configuration."""
        styles = {
            "primary": {
                "fg_color": cls.COLORS["accent"],
                "hover_color": cls.COLORS["accent_hover"],
                "text_color": cls.COLORS["bg_primary"],
                "font": cls.FONTS["button"],
                "corner_radius": cls.RADIUS["sm"],
            },
            "secondary": {
                "fg_color": cls.COLORS["bg_tertiary"],
                "hover_color": cls.COLORS["bg_hover"],
                "text_color": cls.COLORS["text_primary"],
                "font": cls.FONTS["button"],
                "corner_radius": cls.RADIUS["sm"],
            },
            "ghost": {
                "fg_color": "transparent",
                "hover_color": cls.COLORS["bg_hover"],
                "text_color": cls.COLORS["text_secondary"],
                "font": cls.FONTS["button"],
                "corner_radius": cls.RADIUS["sm"],
            },
            "danger": {
                "fg_color": cls.COLORS["error"],
                "hover_color": "#FF6B7A",
                "text_color": cls.COLORS["text_primary"],
                "font": cls.FONTS["button"],
                "corner_radius": cls.RADIUS["sm"],
            },
        }
        return styles.get(variant, styles["primary"])

    @classmethod
    def get_input_style(cls) -> dict:
        """Get input field style configuration."""
        return {
            "fg_color": cls.COLORS["bg_tertiary"],
            "border_color": cls.COLORS["border"],
            "text_color": cls.COLORS["text_primary"],
            "placeholder_text_color": cls.COLORS["text_tertiary"],
            "font": cls.FONTS["mono"],
            "corner_radius": cls.RADIUS["sm"],
        }

    @classmethod
    def get_panel_style(cls) -> dict:
        """Get panel/frame style configuration."""
        return {
            "fg_color": cls.COLORS["bg_secondary"],
            "corner_radius": cls.RADIUS["md"],
        }

    @classmethod
    def get_label_style(cls, variant: str = "body") -> dict:
        """Get label style configuration."""
        return {
            "text_color": cls.COLORS["text_primary"],
            "font": cls.FONTS.get(variant, cls.FONTS["body"]),
        }
