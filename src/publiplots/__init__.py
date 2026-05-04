"""
PubliPlots: Publication-ready plotting with a clean, modular API.

PubliPlots provides a seaborn-like interface for creating beautiful,
publication-ready visualizations with sensible defaults and extensive
customization options.

Basic usage:
    >>> import publiplots as pp
    >>> ax = pp.barplot(data=df, x='category', y='value')
    >>> pp.savefig('output.png')

publiplots applies its publication-grade rcParams on import. Use
:func:`publiplots.reset_style` to revert to matplotlib defaults.
"""

__version__ = "0.8.2"
__author__ = "Jorge Botas"
__license__ = "MIT"
__copyright__ = "Copyright 2025, Jorge Botas"
__url__ = "https://github.com/jorgebotas/publiplots"
__email__ = "jorgebotas.github@gmail.com"
__description__ = "Publication-ready plotting with a clean, modular API"

# Plotting functions
from publiplots.plot.bar import barplot
from publiplots.plot.scatter import scatterplot
from publiplots.plot.point import pointplot
from publiplots.plot.line import lineplot
from publiplots.plot.box import boxplot
from publiplots.plot.swarm import swarmplot
from publiplots.plot.strip import stripplot
from publiplots.plot.violin import violinplot
from publiplots.plot.raincloud import raincloudplot
from publiplots.plot.venn import venn
from publiplots.plot.upset import upsetplot
from publiplots.plot.heatmap import heatmap, complex_heatmap, dendrogram

# Utilities
from publiplots.utils.io import savefig, save_multiple, close_all
from publiplots.utils.display import show, suptitle

# Annotations
from publiplots.annotate import annotate
from publiplots.utils.axes import (
    adjust_spines,
    add_grid,
    set_axis_labels,
    add_reference_line,
    rotate,
    invert_axis,
)
from publiplots.utils.legend import (
    HandlerRectangle,
    HandlerMarker,
    HandlerLineMarker,
    RectanglePatch,
    MarkerPatch,
    LineMarkerPatch,
    get_legend_handler_map,
    create_legend_handles,
    LegendBuilder,
    legend,
)
from publiplots.utils.legend_group import MultiAxesLegendGroup, legend_group
from publiplots.layout import subplots
# Register custom fonts
from publiplots.utils.fonts import _register_fonts
_register_fonts()

# Theming
from publiplots.themes.colors import color_palette
from publiplots.themes.rcparams import rcParams, resolve_param, init_rcparams
from publiplots.themes.styles import reset_style
# Initialize publiplots rcParams defaults
init_rcparams()
from publiplots.themes.markers import (
    resolve_markers,
    resolve_marker_map,
    STANDARD_MARKERS,
)
from publiplots.themes.hatches import (
    set_hatch_mode,
    get_hatch_mode,
    get_hatch_patterns,
    list_hatch_patterns,
    resolve_hatches,
    resolve_hatch_map,
    HATCH_PATTERNS,
)

__all__ = [
    "__version__",
    "__author__",
    # Plots
    "barplot",
    "scatterplot",
    "pointplot",
    "lineplot",
    "boxplot",
    "swarmplot",
    "stripplot",
    "violinplot",
    "raincloudplot",
    "venn",
    "upsetplot",
    "heatmap",
    "complex_heatmap",
    "dendrogram",
    # I/O utilities
    "savefig",
    "save_multiple",
    "close_all",
    # Display utilities
    "show",
    "suptitle",
    # Axes utilities
    "adjust_spines",
    "add_grid",
    "set_axis_labels",
    "add_reference_line",
    "rotate",
    "invert_axis",
    # Legend utilities
    "HandlerRectangle",
    "HandlerMarker",
    "HandlerLineMarker",
    "RectanglePatch",
    "MarkerPatch",
    "LineMarkerPatch",
    "get_legend_handler_map",
    "create_legend_handles",
    "LegendBuilder",
    "legend",
    "MultiAxesLegendGroup",
    "legend_group",
    # Layout functions
    "subplots",
    # Color/palette functions
    "color_palette",
    # Parameter system
    "rcParams",
    "resolve_param",
    # Style functions
    "reset_style",
    # Marker functions
    "resolve_markers",
    "resolve_marker_map",
    # Hatch functions
    "set_hatch_mode",
    "get_hatch_mode",
    "get_hatch_patterns",
    "list_hatch_patterns",
    "resolve_hatches",
    "resolve_hatch_map",
    # Constants
    "STANDARD_MARKERS",
    "HATCH_PATTERNS",
]
