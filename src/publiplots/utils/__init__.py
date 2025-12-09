"""
Utility functions for publiplots.

This module provides helper functions for file I/O, axis manipulation,
and input validation.
"""

from publiplots.utils.io import (
    savefig,
    save_multiple,
    close_all,
    get_figure_size,
    set_figure_size,
)

from publiplots.utils.axes import (
    adjust_spines,
    add_grid,
    remove_grid,
    set_axis_labels,
    set_axis_limits,
    rotate,
    invert_axis,
    add_reference_line,
    set_aspect_equal,
    tighten_layout,
)

from publiplots.utils.validation import (
    is_categorical,
    is_numeric,
    validate_data,
    validate_numeric,
    validate_colors,
    validate_dimensions,
    validate_positive,
    validate_range,
    coerce_to_numeric,
    check_required_columns,
)

from publiplots.utils.fonts import (
    _register_fonts,
    list_registered_fonts,
)

from publiplots.utils.legend import (
    HandlerRectangle,
    HandlerMarker,
    RectanglePatch,
    MarkerPatch,
    get_legend_handler_map,
    create_legend_handles,
    LegendBuilder,
    legend,
)

from publiplots.utils.offset import (
    offset_lines,
    offset_patches,
    offset_collections,
)

from publiplots.utils.text import (
    measure_text_dimensions,
    calculate_label_space,
    get_text_width_in_data_coords,
)

from publiplots.utils.transparency import (
    apply_transparency,
)

# Register fonts globally
_register_fonts()

__all__ = [
    # I/O functions
    "savefig",
    "save_multiple",
    "close_all",
    "get_figure_size",
    "set_figure_size",
    # Axes functions
    "adjust_spines",
    "add_grid",
    "remove_grid",
    "set_axis_labels",
    "set_axis_limits",
    "rotate",
    "invert_axis",
    "add_reference_line",
    "set_aspect_equal",
    "tighten_layout",
    # Validation functions
    "is_categorical",
    "is_numeric",
    "validate_data",
    "validate_numeric",
    "validate_colors",
    "validate_dimensions",
    "validate_positive",
    "validate_range",
    "coerce_to_numeric",
    "check_required_columns",
    # Fonts functions
    "list_registered_fonts",
    # Legend functions
    "HandlerRectangle",
    "HandlerMarker",
    "RectanglePatch",
    "MarkerPatch",
    "get_legend_handler_map",
    "create_legend_handles",
    "LegendBuilder",
    "legend",
    # Transparency functions
    "apply_transparency",
    # Offset functions
    "offset_lines",
    "offset_patches",
    "offset_collections",
    # Text measurement functions
    "measure_text_dimensions",
    "calculate_label_space",
    "get_text_width_in_data_coords",
]
