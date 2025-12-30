"""
Helix Transvoicer - Modern UI Theme.

A sleek, modern dark theme with glassmorphism-inspired design.
"""

import customtkinter as ctk


class HelixTheme:
    """
    Helix Transvoicer modern theme configuration.

    Features a dark, sophisticated design with accent gradients,
    subtle glass effects, and refined typography.
    """

    # Modern color palette
    COLORS = {
        # Background layers (dark to light)
        "bg_primary": "#0A0A0F",      # Deep space black
        "bg_secondary": "#12121A",     # Card background
        "bg_tertiary": "#1A1A25",      # Input/elevated surfaces
        "bg_hover": "#222230",         # Hover state
        "bg_active": "#2A2A3A",        # Active/pressed state
        "bg_glass": "#1A1A2588",       # Glassmorphism overlay

        # Accent colors (vibrant cyan-green gradient spectrum)
        "accent": "#00E5CC",           # Primary accent (bright cyan)
        "accent_hover": "#00FFE0",     # Lighter on hover
        "accent_dim": "#00B8A3",       # Muted accent
        "accent_glow": "#00E5CC40",    # Glow effect

        # Secondary accent (purple for contrast)
        "accent_secondary": "#8B5CF6",
        "accent_secondary_hover": "#A78BFA",

        # Gradient stops
        "gradient_start": "#00E5CC",
        "gradient_end": "#8B5CF6",

        # Status colors (modern, slightly desaturated)
        "success": "#10B981",          # Emerald green
        "warning": "#F59E0B",          # Amber
        "error": "#EF4444",            # Red
        "info": "#3B82F6",             # Blue

        # Text colors
        "text_primary": "#FFFFFF",
        "text_secondary": "#9CA3AF",   # Muted gray
        "text_tertiary": "#6B7280",    # Subtle gray
        "text_disabled": "#4B5563",
        "text_accent": "#00E5CC",

        # Border colors
        "border": "#2D2D3A",
        "border_subtle": "#1F1F2A",
        "border_active": "#00E5CC",
        "border_glow": "#00E5CC30",

        # Progress colors
        "progress_bg": "#1A1A25",
        "progress_fill": "#00E5CC",
        "progress_glow": "#00E5CC50",

        # Emotion colors (refined palette)
        "emotion_neutral": "#6B7280",
        "emotion_happy": "#FBBF24",
        "emotion_sad": "#60A5FA",
        "emotion_angry": "#F87171",
        "emotion_fear": "#A78BFA",
        "emotion_surprise": "#FB923C",
        "emotion_disgust": "#84CC16",
        "emotion_calm": "#2DD4BF",
        "emotion_excited": "#F472B6",
    }

    # Typography with modern system fonts
    FONTS = {
        "title": ("Inter", 20, "bold"),
        "heading": ("Inter", 16, "bold"),
        "subheading": ("Inter", 14, "bold"),
        "body": ("Inter", 13),
        "button": ("Inter", 12, "bold"),
        "small": ("Inter", 11),
        "tiny": ("Inter", 10),
        "mono": ("JetBrains Mono", 12),
        "mono_small": ("JetBrains Mono", 11),
        "data": ("JetBrains Mono", 11),
    }

    # Spacing system (8px grid)
    SPACING = {
        "xs": 4,
        "sm": 8,
        "md": 16,
        "lg": 24,
        "xl": 32,
        "2xl": 48,
    }

    # Border radius (more rounded for modern feel)
    RADIUS = {
        "none": 0,
        "sm": 6,
        "md": 10,
        "lg": 14,
        "xl": 20,
        "full": 9999,
    }

    # Shadows (for depth)
    SHADOWS = {
        "sm": "0 1px 2px rgba(0, 0, 0, 0.3)",
        "md": "0 4px 6px rgba(0, 0, 0, 0.4)",
        "lg": "0 10px 15px rgba(0, 0, 0, 0.5)",
        "glow": "0 0 20px rgba(0, 229, 204, 0.2)",
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
                "corner_radius": cls.RADIUS["md"],
                "border_width": 0,
            },
            "secondary": {
                "fg_color": cls.COLORS["bg_tertiary"],
                "hover_color": cls.COLORS["bg_hover"],
                "text_color": cls.COLORS["text_primary"],
                "font": cls.FONTS["button"],
                "corner_radius": cls.RADIUS["md"],
                "border_width": 1,
                "border_color": cls.COLORS["border"],
            },
            "ghost": {
                "fg_color": "transparent",
                "hover_color": cls.COLORS["bg_hover"],
                "text_color": cls.COLORS["text_secondary"],
                "font": cls.FONTS["button"],
                "corner_radius": cls.RADIUS["md"],
                "border_width": 0,
            },
            "outline": {
                "fg_color": "transparent",
                "hover_color": cls.COLORS["accent_glow"],
                "text_color": cls.COLORS["accent"],
                "font": cls.FONTS["button"],
                "corner_radius": cls.RADIUS["md"],
                "border_width": 2,
                "border_color": cls.COLORS["accent"],
            },
            "danger": {
                "fg_color": cls.COLORS["error"],
                "hover_color": "#DC2626",
                "text_color": cls.COLORS["text_primary"],
                "font": cls.FONTS["button"],
                "corner_radius": cls.RADIUS["md"],
                "border_width": 0,
            },
            "success": {
                "fg_color": cls.COLORS["success"],
                "hover_color": "#059669",
                "text_color": cls.COLORS["text_primary"],
                "font": cls.FONTS["button"],
                "corner_radius": cls.RADIUS["md"],
                "border_width": 0,
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
            "font": cls.FONTS["body"],
            "corner_radius": cls.RADIUS["md"],
            "border_width": 1,
        }

    @classmethod
    def get_panel_style(cls) -> dict:
        """Get panel/frame style configuration."""
        return {
            "fg_color": cls.COLORS["bg_secondary"],
            "corner_radius": cls.RADIUS["lg"],
        }

    @classmethod
    def get_card_style(cls) -> dict:
        """Get card style configuration."""
        return {
            "fg_color": cls.COLORS["bg_secondary"],
            "corner_radius": cls.RADIUS["lg"],
            "border_width": 1,
            "border_color": cls.COLORS["border_subtle"],
        }

    @classmethod
    def get_label_style(cls, variant: str = "body") -> dict:
        """Get label style configuration."""
        text_colors = {
            "title": cls.COLORS["text_primary"],
            "heading": cls.COLORS["text_primary"],
            "body": cls.COLORS["text_primary"],
            "small": cls.COLORS["text_secondary"],
            "muted": cls.COLORS["text_tertiary"],
            "accent": cls.COLORS["accent"],
        }
        return {
            "text_color": text_colors.get(variant, cls.COLORS["text_primary"]),
            "font": cls.FONTS.get(variant, cls.FONTS["body"]),
        }

    @classmethod
    def get_checkbox_style(cls) -> dict:
        """Get checkbox style configuration."""
        return {
            "fg_color": cls.COLORS["accent"],
            "hover_color": cls.COLORS["accent_hover"],
            "border_color": cls.COLORS["border"],
            "checkmark_color": cls.COLORS["bg_primary"],
            "text_color": cls.COLORS["text_secondary"],
            "font": cls.FONTS["small"],
            "corner_radius": cls.RADIUS["sm"],
        }

    @classmethod
    def get_slider_style(cls) -> dict:
        """Get slider style configuration."""
        return {
            "fg_color": cls.COLORS["bg_tertiary"],
            "progress_color": cls.COLORS["accent"],
            "button_color": cls.COLORS["accent"],
            "button_hover_color": cls.COLORS["accent_hover"],
        }

    @classmethod
    def get_tab_style(cls) -> dict:
        """Get tabview style configuration."""
        return {
            "fg_color": cls.COLORS["bg_secondary"],
            "segmented_button_fg_color": cls.COLORS["bg_tertiary"],
            "segmented_button_selected_color": cls.COLORS["accent"],
            "segmented_button_selected_hover_color": cls.COLORS["accent_hover"],
            "segmented_button_unselected_color": cls.COLORS["bg_tertiary"],
            "segmented_button_unselected_hover_color": cls.COLORS["bg_hover"],
            "text_color": cls.COLORS["text_primary"],
            "text_color_disabled": cls.COLORS["text_disabled"],
            "corner_radius": cls.RADIUS["lg"],
        }
