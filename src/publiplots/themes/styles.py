"""
Matplotlib style presets for publiplots.

This module provides functions to apply consistent styling to matplotlib
plots, optimized for publication-ready visualizations.

Two main styles:
- set_notebook_style(): For interactive work in Jupyter notebooks
- set_publication_style(): For final publication figures (compact, high DPI)

Styles are composed from base defaults plus style-specific overrides.
"""

from typing import Dict, Any
import matplotlib.pyplot as plt

from .rcparams import (
    MATPLOTLIB_RCPARAMS,
    PUBLIPLOTS_RCPARAMS,
    _PUBLIPLOTS_CUSTOM_DEFAULTS,
)


# =============================================================================
# Style Compositions
# =============================================================================

# Notebook style: base defaults + larger sizes for interactive work
NOTEBOOK_STYLE = {
    **MATPLOTLIB_RCPARAMS,
    **PUBLIPLOTS_RCPARAMS,
    # Overrides for notebook/interactive work
    "figure.figsize": [6.0, 4.0],
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "lines.markeredgewidth": 2.0,
    "patch.linewidth": 2.0,
}
"""
Notebook-ready style optimized for interactive work and exploration.

Features:
- Readable font sizes for screens (12pt base)
- Larger figure sizes (6×4) for notebooks
- Medium DPI (300) for good quality
- Thicker lines for better visibility on screens
- Ideal for Jupyter notebooks and interactive analysis
"""

# Publication style: base defaults + compact publication settings
PUBLICATION_STYLE = {
    **MATPLOTLIB_RCPARAMS,
    **PUBLIPLOTS_RCPARAMS,
    # Overrides for publication
    "lines.linewidth": 1.0,
    "patch.linewidth": 1.0,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "figure.dpi": 600,
    "savefig.dpi": 600,
}
"""
Publication-ready style optimized for final publication figures.

Features:
- Small fonts (8pt base) for compact figures
- High DPI (600) for print quality
- Compact figure size (3×2) for publications
- Clean, minimal styling
- Perfect for creating figures for papers or Adobe Illustrator
"""


# =============================================================================
# Helper Functions
# =============================================================================

def _apply_style(style_dict: Dict[str, Any]) -> None:
    """
    Apply a complete style (both matplotlib and publiplots params).

    Parameters
    ----------
    style_dict : dict
        Complete style dictionary containing both matplotlib rcParams
        and publiplots-specific parameters.
    """
    # Separate matplotlib params from publiplots params
    matplotlib_params = {
        k: v for k, v in style_dict.items() if k not in PUBLIPLOTS_RCPARAMS
    }
    publiplots_params = {
        k: v for k, v in style_dict.items() if k in PUBLIPLOTS_RCPARAMS
    }

    # Apply to respective stores
    plt.rcParams.update(matplotlib_params)
    _PUBLIPLOTS_CUSTOM_DEFAULTS.clear()
    _PUBLIPLOTS_CUSTOM_DEFAULTS.update(publiplots_params)


# =============================================================================
# Public Style Functions
# =============================================================================

def set_notebook_style() -> None:
    """
    Apply notebook-ready style to all matplotlib plots.

    This style is optimized for interactive work in Jupyter notebooks with
    readable font sizes and larger figure dimensions.

    The style includes both matplotlib rcParams and publiplots parameters
    (color, alpha, capsize, palette, hatch_mode).

    Examples
    --------
    Apply notebook style for interactive work:
    >>> import publiplots as pp
    >>> pp.set_notebook_style()
    >>> fig, ax = pp.barplot(data=df, x='x', y='y')  # Uses notebook defaults

    Check current parameters:
    >>> pp.rcParams['font.size']  # 11
    >>> pp.rcParams['alpha']  # 0.1
    >>> pp.rcParams['figure.figsize']  # [6.0, 4.0]

    Notes
    -----
    This style sets:
    - Font size: 11pt (readable on screens)
    - Figure size: 6×4 inches
    - DPI: 300 (good quality for screens)
    - Line width: 2.0 (thicker for visibility)
    - All publiplots params: color, alpha, capsize, palette, hatch_mode
    """
    _apply_style(NOTEBOOK_STYLE)


def set_publication_style() -> None:
    """
    Apply publication-ready style to all matplotlib plots.

    This style is optimized for final publication figures with small fonts,
    high DPI (600), and compact dimensions. Perfect for creating figures that
    will be edited in Adobe Illustrator or directly included in papers.

    The style includes both matplotlib rcParams and publiplots parameters
    (color, alpha, capsize, palette, hatch_mode).

    Examples
    --------
    Apply publication style for final figures:
    >>> import publiplots as pp
    >>> pp.set_publication_style()
    >>> fig, ax = pp.barplot(data=df, x='x', y='y')  # Uses publication defaults

    Check current parameters:
    >>> pp.rcParams['font.size']  # 8
    >>> pp.rcParams['alpha']  # 0.15
    >>> pp.rcParams['figure.figsize']  # [3.5, 2.5]
    >>> pp.rcParams['savefig.dpi']  # 600

    Notes
    -----
    This style sets:
    - Font size: 8pt (compact for publications)
    - Figure size: 3.5×2.5 inches (fits journal columns)
    - DPI: 600 (print quality)
    - Line width: 1.2 (appropriate for small plots)
    - Alpha: 0.15 (more visible on compact figures)
    - All publiplots params: color, capsize, palette, hatch_mode
    """
    _apply_style(PUBLICATION_STYLE)


def reset_style() -> None:
    """
    Reset matplotlib rcParams to matplotlib defaults.

    Useful when you want to revert to matplotlib's default styling.
    Does not affect publiplots custom parameters.

    Examples
    --------
    >>> import publiplots as pp
    >>> pp.set_publication_style()
    >>> # ... create plots ...
    >>> pp.reset_style()  # Reset to matplotlib defaults
    """
    plt.rcdefaults()


def get_current_style() -> Dict[str, Any]:
    """
    Get current matplotlib rcParams as a dictionary.

    Useful for debugging or saving current style settings.
    Only returns matplotlib rcParams, not publiplots parameters.

    Returns
    -------
    Dict[str, Any]
        Dictionary of current matplotlib rcParams.

    Examples
    --------
    >>> import publiplots as pp
    >>> pp.set_publication_style()
    >>> current = pp.get_current_style()
    >>> print(current['font.size'])
    8
    """
    return dict(plt.rcParams)


def apply_custom_style(style_dict: Dict[str, Any]) -> None:
    """
    Apply a custom style dictionary to matplotlib.

    Only applies matplotlib rcParams. To set publiplots parameters,
    use pp.rcParams directly.

    Parameters
    ----------
    style_dict : Dict[str, Any]
        Dictionary of matplotlib rcParams to apply.

    Examples
    --------
    Apply custom matplotlib settings:
    >>> import publiplots as pp
    >>> custom = {'font.size': 14, 'lines.linewidth': 3}
    >>> pp.apply_custom_style(custom)

    For publiplots parameters, use rcParams directly:
    >>> pp.rcParams['alpha'] = 0.2
    >>> pp.rcParams['hatch_mode'] = 3
    """
    plt.rcParams.update(style_dict)
