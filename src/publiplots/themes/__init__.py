"""
Theming system for publiplots.

This module provides color palettes, styles, and marker definitions
for creating consistent, publication-ready visualizations.
"""

from publiplots.themes.colors import (
    color_palette,
    PALETTES,
)

from publiplots.themes.rcparams import (
    rcParams,
    resolve_param,
)

from publiplots.themes.styles import (
    reset_style,
    get_current_style,
    apply_custom_style,
)

from publiplots.themes.markers import (
    resolve_size_map,
    resolve_markers,
    resolve_marker_map,
    STANDARD_MARKERS,
    FILLED_UNFILLED_MARKERS,
    MARKER_SIZES,
)

from publiplots.themes.linestyles import (
    STANDARD_LINESTYLES,
    resolve_linestyle_map,
)

from publiplots.themes.hatches import (
    get_hatch_patterns,
    set_hatch_mode,
    get_hatch_mode,
    list_hatch_patterns,
    resolve_hatches,
    resolve_hatch_map,
    HATCH_PATTERNS,
)

__all__ = [
    # Parameter system
    "rcParams",
    "resolve_param",
    # Color functions
    "color_palette",
    # Color palettes
    "PALETTES",
    # Style functions
    "reset_style",
    "get_current_style",
    "apply_custom_style",
    # Marker functions
    "resolve_size_map",
    "resolve_markers",
    "resolve_marker_map",
    # Marker constants
    "STANDARD_MARKERS",
    "FILLED_UNFILLED_MARKERS",
    "MARKER_SIZES",
    # Line style functions
    "resolve_linestyle_map",
    "STANDARD_LINESTYLES",
    # Hatch functions
    "get_hatch_patterns",
    "set_hatch_mode",
    "get_hatch_mode",
    "list_hatch_patterns",
    "resolve_hatches",
    "resolve_hatch_map",
    # Hatch constants
    "HATCH_PATTERNS",
]
