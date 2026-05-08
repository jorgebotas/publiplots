"""
Default rcParams for publiplots.

This module defines all default parameter values and provides the
unified rcParams interface for accessing both matplotlib and
publiplots-specific parameters.

Main components:

- :data:`MATPLOTLIB_RCPARAMS` — matplotlib parameter overrides installed
  when publiplots is imported.
- :data:`PUBLIPLOTS_RCPARAMS` — custom publiplots parameters not in
  matplotlib.
- :class:`PubliplotsRcParams` — unified dict-like interface exposed as
  :data:`publiplots.rcParams`.
- :func:`resolve_param` — helper for ``value if value is not None else
  default`` parameter resolution used throughout the plot functions.

publiplots-specific keys (not in matplotlib) include:

- ``edgecolor`` — global edge color for patches and marker outlines
  (``None`` = each plot's default).
- ``alpha`` — default fill transparency for bars (``0.1``).
- ``palette`` — default qualitative palette name (``"pastel"``).
- ``hatch_mode`` — global hatch density (``1`` – ``4``). Prefer
  :func:`publiplots.set_hatch_mode` to mutate it.
- ``scatter.size_min`` / ``scatter.size_max`` — size mapping bounds
  for :func:`publiplots.scatterplot` (points^2).
- ``subplots.axes_size`` — default ``(width_mm, height_mm)`` for
  :func:`publiplots.subplots`.
- ``subplots.title_space`` / ``xlabel_space`` / ``ylabel_space`` /
  ``right`` — initial per-side reservations in mm (auto-measured on
  first draw unless the user passes an explicit value).
- ``subplots.hspace`` / ``wspace`` / ``outer_pad`` — gaps and outer
  margin in mm (never auto-measured).
"""

from typing import Dict, Any, Optional
import matplotlib.pyplot as plt


# =============================================================================
# Base Default Dictionaries
# =============================================================================
TEXT_COLOR = "black"
MATPLOTLIB_RCPARAMS: Dict[str, Any] = {
    # Figure settings - compact by default (publication-ready)
    # NB: figure.figsize is intentionally not set. publiplots sizes figures
    # via pp.subplots(axes_size=...), which computes the canvas from axes
    # dimensions (mm) + reservations. See `subplots.axes_size` below.
    "figure.dpi": 150,      # screen rendering; savefig uses savefig.dpi below for print-quality
    "figure.edgecolor": "none",
    "figure.subplot.hspace": 0.05,
    "figure.subplot.wspace": 0.05,
    # Figure-level suptitle: one notch above axes.titlesize (10) so the
    # figure title reads as the outermost heading in the type hierarchy
    # (matplotlib's default 'large' resolves to 9.6pt, which is smaller
    # than the panel titles — flipped hierarchy).
    "figure.titlesize": 11,
    "figure.titleweight": "normal",

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
    "axes.linewidth": 0.75,
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
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
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
    # publiplots figures are already laid out to mm-precise margins via
    # FigureLayout; ``bbox='tight'`` would re-crop to the union of all
    # artist bboxes, shifting legends and nuking figure-anchored bands
    # (side='top'|'bottom'|'left'|'right' with no ``anchor=``). See the
    # sphinx-gallery scraper override in docs/source/conf.py.
    "savefig.bbox": "standard",
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
    "subplots.axes_size": (70.0, 50.0),  # default (width, height) of each axes in pp.subplots
    "subplots.title_space": 5,    # reserved above each row
    "subplots.xlabel_space": 8,   # reserved below each row
    "subplots.ylabel_space": 10,  # reserved left of each col
    "subplots.right": 2,          # reserved right of each col
    "subplots.hspace": 3,         # vertical gap between rows (on top of per-cell title/xlabel reservations)
    "subplots.wspace": 3,         # horizontal gap between cols (on top of per-cell ylabel/right reservations)
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
    Resolve a parameter value: return ``value`` if not ``None``, else
    the default for ``key``.

    This helper eliminates the repetitive ``if value is None: value =
    default`` pattern in plot functions. Lookup checks publiplots
    custom parameters first (e.g. ``"alpha"``, ``"palette"``,
    ``"subplots.axes_size"``), then falls back to
    :attr:`matplotlib.pyplot.rcParams`.

    Parameters
    ----------
    key : str
        Parameter name to look up when ``value`` is ``None``.
    value : Any, optional
        User-provided value. If not ``None``, it is returned
        unchanged.

    Returns
    -------
    Any
        Either the user-provided ``value`` (if not ``None``) or the
        default for ``key``.

    Raises
    ------
    KeyError
        If ``value`` is ``None`` and ``key`` is not defined in either
        publiplots or matplotlib rcParams.

    Examples
    --------
    Typical use inside a plotting function:

    >>> from publiplots.themes.rcparams import resolve_param
    >>> def my_plot(color=None, alpha=None):
    ...     color = resolve_param('color', color)
    ...     alpha = resolve_param('alpha', alpha)

    Explicit user value passes through unchanged:

    >>> resolve_param('color', '#ff0000')
    '#ff0000'

    ``None`` falls back to the default:

    >>> resolve_param('color', None)
    '#5d83c3'
    >>> resolve_param('color')
    '#5d83c3'

    See Also
    --------
    rcParams : Unified rcParams accessor for reading / writing defaults.
    """
    return value if value is not None else _get_default(key)



# =============================================================================
# PubliPlots rcParams Wrapper
# =============================================================================

class PubliplotsRcParams:
    """
    Unified interface for publiplots parameters.

    Dict-like accessor for both standard matplotlib rcParams and
    publiplots-specific parameters. Mimics matplotlib's rcParams but
    also exposes the custom keys listed in this module's docstring
    (``alpha``, ``edgecolor``, ``palette``, ``hatch_mode``,
    ``scatter.size_*``, ``subplots.*``). Use the module-level
    :data:`rcParams` instance — don't instantiate this class yourself.

    Writes are routed automatically: publiplots keys update the
    publiplots store; everything else is delegated to
    :attr:`matplotlib.pyplot.rcParams`. Prefer
    :func:`publiplots.set_hatch_mode` for mutating ``hatch_mode`` so
    that validation runs.

    Examples
    --------
    Read parameters:

    >>> import publiplots as pp
    >>> axes_size = pp.rcParams['subplots.axes_size']
    >>> color = pp.rcParams['color']  # publiplots-specific key

    Set parameters:

    >>> pp.rcParams['subplots.axes_size'] = (80, 50)  # mm
    >>> pp.rcParams['color'] = '#ff0000'

    Use with :func:`resolve_param` in a plotting function:

    >>> from publiplots.themes.rcparams import resolve_param
    >>> color = resolve_param('color', user_color)

    See Also
    --------
    resolve_param : Value-or-default helper used in plot signatures.
    publiplots.set_hatch_mode : Validated setter for ``hatch_mode``.
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
>>> pp.rcParams['subplots.axes_size'] = (80, 50)  # mm
>>> pp.rcParams['alpha'] = 0.2                     # publiplots param
>>> pp.rcParams['color']                           # Get default color
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
    # publiplots is opinionated about its publication-grade defaults — always
    # overwrite, so users get consistent styling (Arial fonts, 0.75pt strokes,
    # 8pt labels, etc.) regardless of their matplotlibrc. Per-parameter
    # overrides are still easy: assign to pp.rcParams after import, or call
    # pp.reset_style() to revert to matplotlib's own defaults entirely.
    plt.rcParams.update(MATPLOTLIB_RCPARAMS)


# Initialize on import
init_rcparams()
