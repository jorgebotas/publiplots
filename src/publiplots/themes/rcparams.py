"""
Default rcParams for publiplots.

This module defines all default parameter values and provides the unified
rcParams interface for accessing both matplotlib and publiplots parameters.

Main components:
- MATPLOTLIB_RCPARAMS: Base matplotlib parameter defaults
- PUBLIPLOTS_RCPARAMS: Base publiplots custom parameter defaults
- PubliplotsRcParams: Unified dict-like interface
- rcParams: Global instance for parameter access
- resolve_param(): Helper for parameter resolution
"""

from typing import Dict, Any, Optional
import matplotlib.pyplot as plt


# =============================================================================
# Base Default Dictionaries
# =============================================================================
TEXT_COLOR = "black"
MATPLOTLIB_RCPARAMS: Dict[str, Any] = {
    # Figure settings - compact by default (publication-ready)
    "figure.figsize": [3, 1.8],
    "figure.dpi": 600,
    "figure.edgecolor": "none",
    "figure.subplot.hspace": 0.05,
    "figure.subplot.wspace": 0.05,

    # Font settings - optimized for readability
    "font.size": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "sans-serif"],

    # Text settings
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "xtick.labelcolor": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "ytick.labelcolor": TEXT_COLOR,
    "legend.labelcolor": TEXT_COLOR,

    # Axes settings
    "axes.linewidth": 1.0,
    "axes.edgecolor": "0.3",
    "axes.facecolor": "white",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,

    # Line settings
    "lines.linewidth": 1.0,
    "lines.markeredgewidth": 1.0,
    "lines.markersize": 6,

    # Patch settings (for bars, etc.)
    "patch.linewidth": 1.0,
    "patch.edgecolor": TEXT_COLOR,

    # Tick settings
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 0,
    "ytick.major.size": 0,

    # Grid settings
    "axes.grid": False,
    "grid.linewidth": 0.8,
    "grid.color": "0.8",
    "grid.alpha": 0.8,
    "grid.linestyle": "--",

    # Legend settings
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.edgecolor": "none",

    # Save settings - high quality for publications
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "savefig.transparent": True,
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",

    # PDF settings for vector graphics
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
"""
Base matplotlib rcParams defaults.

These are the fundamental matplotlib parameter values shared across all styles.
Individual styles (notebook, publication) compose these with their overrides.
"""


PUBLIPLOTS_RCPARAMS: Dict[str, Any] = {
    # Color and transparency
    "color": "#5d83c3",  # Default blue
    "alpha": 0.1,  # Default transparency for bars
    "edgecolor": None,  # Global edge color for patches and marker outlines; None = each plot's default auto behavior

    # Error bars
    "capsize": 0.0,  # Error bar cap size

    # Color palettes
    "palette": "pastel",  # Default color palette

    # Hatch patterns
    "hatch_mode": 2,  # Default hatch density mode (2=medium)

    # Scatter plot sizes
    "scatter.size_min": 50,  # Minimum marker size for size mapping
    "scatter.size_max": 1000,  # Maximum marker size for size mapping

    # Subplots layout (mm) — baseline is publication-grade; notebook style
    # overrides these in themes/styles.py.
    "subplots.title_space": 5,    # reserved above each row
    "subplots.xlabel_space": 8,   # reserved below each row
    "subplots.ylabel_space": 10,  # reserved left of each col
    "subplots.right": 2,          # reserved right of each col
    "subplots.hspace": 8,         # vertical gap between rows
    "subplots.wspace": 10,        # horizontal gap between cols
    "subplots.outer_pad": 2,      # figure outer margin (all sides)
}
"""
PubliPlots custom rcParams.

These are publiplots-specific parameters not part of matplotlib's rcParams.
They can be accessed via pp.rcParams just like matplotlib parameters.
"""


# Module-level mutable storage for custom parameters
# This gets updated by styles and user modifications
_PUBLIPLOTS_CUSTOM_DEFAULTS: Dict[str, Any] = PUBLIPLOTS_RCPARAMS.copy()


# =============================================================================
# Helper Functions
# =============================================================================

def _get_default(key: str) -> Any:
    """
    Get a default parameter value (internal use only).

    Checks custom publiplots params first, then matplotlib rcParams.
    Raises KeyError if parameter not found.

    Parameters
    ----------
    key : str
        Parameter name

    Returns
    -------
    Any
        Parameter value

    Raises
    ------
    KeyError
        If parameter not found in either custom or matplotlib params
    """
    # Check custom params first
    if key in _PUBLIPLOTS_CUSTOM_DEFAULTS:
        return _PUBLIPLOTS_CUSTOM_DEFAULTS[key]

    # Then check matplotlib rcParams
    if key in plt.rcParams:
        return plt.rcParams[key]

    raise KeyError(f"Parameter '{key}' not found in publiplots or matplotlib rcParams")


def resolve_param(key: str, value: Optional[Any] = None) -> Any:
    """
    Resolve a parameter value: use provided value if not None, otherwise get default.

    This helper eliminates the repetitive "if value is None: value = default" pattern
    throughout the codebase.

    Parameters
    ----------
    key : str
        Parameter name
    value : Any, optional
        User-provided value. If None, the default will be used.

    Returns
    -------
    Any
        The resolved parameter value (user value or default)

    Examples
    --------
    In a plotting function:
    >>> def barplot(color=None, alpha=None):
    ...     color = resolve_param('color', color)  # Uses color if provided, else default
    ...     alpha = resolve_param('alpha', alpha)
    ...     # Now color and alpha are guaranteed to have values

    User provides value:
    >>> color = resolve_param('color', '#ff0000')  # Returns '#ff0000'

    User doesn't provide value:
    >>> color = resolve_param('color', None)  # Returns default color '#5d83c3'
    >>> color = resolve_param('color')  # Same as above
    """
    return value if value is not None else _get_default(key)



# =============================================================================
# PubliPlots rcParams Wrapper
# =============================================================================

class PubliplotsRcParams:
    """
    Unified interface for publiplots parameters.

    This class provides a dict-like interface for accessing both standard
    matplotlib rcParams and custom publiplots parameters. It mimics the
    behavior of matplotlib's rcParams but includes publiplots-specific
    defaults.

    Examples
    --------
    Access parameters:
    >>> from publiplots.themes import rcParams
    >>> figsize = rcParams['figure.figsize']
    >>> color = rcParams['color']  # Custom publiplots param

    Set parameters:
    >>> rcParams['figure.figsize'] = (8, 6)
    >>> rcParams['color'] = '#ff0000'

    Use in functions with resolve_param:
    >>> from publiplots.themes.rcparams import resolve_param
    >>> color = resolve_param('color', user_color)  # Uses user_color if not None
    """

    def __getitem__(self, key: str) -> Any:
        """Get parameter value."""
        return _get_default(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter value with optional fallback."""
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: str, value: Any) -> None:
        """Set parameter value."""
        if key in PUBLIPLOTS_RCPARAMS:
            # Custom publiplots parameter
            _PUBLIPLOTS_CUSTOM_DEFAULTS[key] = value
        else:
            # Matplotlib parameter
            plt.rcParams[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if parameter exists."""
        return key in _PUBLIPLOTS_CUSTOM_DEFAULTS or key in plt.rcParams

    def keys(self):
        """Return all parameter keys."""
        return list(_PUBLIPLOTS_CUSTOM_DEFAULTS.keys()) + list(plt.rcParams.keys())


# =============================================================================
# Global rcParams Instance
# =============================================================================

rcParams = PubliplotsRcParams()
"""
Global publiplots rcParams instance.

This provides unified access to both matplotlib and publiplots parameters.
Use it just like matplotlib's rcParams:

>>> import publiplots as pp
>>> pp.rcParams['figure.figsize'] = [8, 6]  # matplotlib param
>>> pp.rcParams['alpha'] = 0.2              # publiplots param
>>> pp.rcParams['color']                    # Get default color
'#5d83c3'
"""


# =============================================================================
# Initialization
# =============================================================================

def init_rcparams() -> None:
    """
    Initialize publiplots default rcParams.

    This function is automatically called when publiplots is imported.
    It sets sensible defaults for matplotlib rcParams.

    Examples
    --------
    Manually reinitialize defaults:
    >>> import publiplots as pp
    >>> pp.themes.rcparams.init_rcparams()
    """
    for key, value in MATPLOTLIB_RCPARAMS.items():
        # Only set if key doesn't exist or is at matplotlib default
        # This preserves user customizations made before import
        if key not in plt.rcParams or plt.rcParams[key] == plt.rcParamsDefault.get(key):
            plt.rcParams[key] = value


# Initialize on import
init_rcparams()
