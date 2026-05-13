"""
Color palettes for publiplots.

This module provides carefully curated color palettes optimized for
publication-ready visualizations, with seamless integration with seaborn.
"""

from typing import Dict, List, Optional, Union

from publiplots.themes.rcparams import resolve_param
from publiplots.utils import is_categorical

# =============================================================================
# Color Palettes
# =============================================================================

PALETTES = {
    "pastel": [
        "#75b375",  # Soft green
        "#8e8ec1",  # Soft purple
        "#eeaa58",  # Soft orange
        "#e67e7e",  # Soft red
        "#7ec5d9",  # Soft blue
        "#f0b0c4",  # Soft pink
        "#b8b88a",  # Soft olive
        "#c9a3cf",  # Soft lavender
        "#f4c896",  # Soft peach
        "#8fc9a8",  # Soft teal
        "#dba3a3",  # Soft rose
        "#9eb3d4",  # Soft periwinkle
    ],
    "RdBu": [
        "#355085",  # Deep blue
        "#5073b8",  # Medium blue
        "#95abd5",  # Light blue
        "#dbe3f1",  # Very light blue
        "#ffffff",  # White (center)
        "#efdae0",  # Very light red
        "#d296a8",  # Light red
        "#b4516f",  # Medium red
        "#80374d",  # Deep red
    ],
    "RdGyBu": [
        "#c96f76", 
        "#e4b6ba",
        "#a7a7a7", 
        "#b2c1df",
        "#6c89c3",
    ]
}
"""Dictionary of custom publiplots palettes."""

DIVERGENT_PALETTES = {"RdBu", "RdGyBu"}
"""Set of palette names that use divergent sampling (extremes first)."""


# =============================================================================
# Functions
# =============================================================================


def _sample_divergent(colors: List[str], n_colors: int) -> List[str]:
    """
    Sample colors from a divergent palette to preserve extremes.

    For divergent palettes, sampling should preserve the extreme values
    (endpoints) and sample evenly across the full range. This ensures
    that the divergent nature is maintained even with few colors.

    Parameters
    ----------
    colors : List[str]
        Full list of colors in the palette.
    n_colors : int
        Number of colors to sample.

    Returns
    -------
    List[str]
        Sampled colors with extremes preserved.

    Examples
    --------
    >>> palette = ['#000', '#111', '#222', '#333', '#444']
    >>> _sample_divergent(palette, 2)
    ['#000', '#444']  # First and last
    >>> _sample_divergent(palette, 3)
    ['#000', '#222', '#444']  # First, middle, last
    """
    import numpy as np

    if n_colors >= len(colors):
        return colors

    # Calculate evenly spaced indices across the full palette
    indices = np.round(np.linspace(0, len(colors) - 1, n_colors)).astype(int)
    return [colors[i] for i in indices]


def color_palette(palette=None, n_colors=None, desat=None, as_cmap=False, divergent=None):
    """
    Return a color palette as a list of hex colors or a colormap.

    Integrates with seaborn — custom publiplots palettes (see
    :data:`PALETTES`) take precedence over seaborn palette names.

    Parameters
    ----------
    palette : str or list, optional
        Palette name or explicit list of colors. Name resolution
        checks publiplots palettes first, then seaborn. If ``None``,
        uses ``pp.rcParams["palette"]`` (default ``"pastel"``). Append
        ``"_r"`` to reverse any named palette (e.g. ``"coolwarm_r"``,
        ``"viridis_r"``).
    n_colors : int, optional
        Number of colors to return. If ``None``, returns the full
        palette.
    desat : float, optional
        Proportion to desaturate each color, in ``[0, 1]``.
    as_cmap : bool, default ``False``
        If ``True``, return a :class:`matplotlib.colors.ListedColormap`
        instead of a list.
    divergent : bool, optional
        If ``True``, sample colors to preserve extremes (first / last
        / evenly spaced) — appropriate for diverging palettes. If
        ``None``, auto-detected for palettes listed in
        :data:`DIVERGENT_PALETTES`. Set ``False`` to disable sampling
        even for known divergent palettes.

    Returns
    -------
    list of str or matplotlib.colors.ListedColormap
        List of hex color codes, or a colormap when
        ``as_cmap=True``.

    Examples
    --------
    Get custom pastel palette:

    >>> colors = color_palette("pastel")
    >>> len(colors)
    12

    Get seaborn palette:

    >>> colors = color_palette("viridis", n_colors=5)

    Get as colormap:

    >>> cmap = color_palette("pastel", as_cmap=True)

    Divergent palette sampled at 3 colors (first, middle, last):

    >>> colors = color_palette("RdBu", n_colors=3)

    Force divergent sampling on any palette:

    >>> colors = color_palette("viridis", n_colors=5, divergent=True)

    Reverse any palette with the ``"_r"`` suffix (seaborn convention):

    >>> colors = color_palette("RdBu_r", n_colors=3)
    >>> colors = color_palette("viridis_r", n_colors=5)

    See Also
    --------
    PALETTES : Dictionary of built-in publiplots palettes.
    DIVERGENT_PALETTES : Names auto-detected as divergent.
    """
    import seaborn as sns

    # Handle None: use default palette from rcParams
    if palette is None:
        palette = resolve_param("palette", "pastel")

    # Handle "_r" suffix for reversed palettes (seaborn convention)
    reverse = False
    base_palette = palette
    if isinstance(palette, str) and palette.endswith("_r"):
        base_palette = palette[:-2]  # Remove "_r" suffix
        reverse = True

    # Auto-detect divergent palettes or use explicit parameter
    # Use base_palette for detection (before "_r" suffix)
    is_divergent = (
        divergent if divergent is not None
        else (isinstance(base_palette, str) and base_palette in DIVERGENT_PALETTES)
    )

    # Check publiplots PALETTES first (using base_palette name)
    if isinstance(base_palette, str) and base_palette in PALETTES:
        colors = PALETTES[base_palette]

        # Reverse colors if "_r" suffix was used
        if reverse:
            colors = list(reversed(colors))

        # Apply divergent sampling if needed and n_colors is specified
        if is_divergent and n_colors is not None and n_colors < len(colors):
            colors = _sample_divergent(colors, n_colors)
            # After sampling, use all sampled colors (don't resample in seaborn)
            n_colors = None

        if as_cmap:
            from matplotlib.colors import ListedColormap
            return ListedColormap(colors)
        return sns.color_palette(colors, n_colors=n_colors, desat=desat)

    # Delegate everything else to seaborn
    return sns.color_palette(palette, n_colors=n_colors, desat=desat, as_cmap=as_cmap)


def resolve_continuous_cmap(cmap=None, color=None):
    """
    Resolve a continuous-hue colormap for 2D plots.

    Returns ``cmap`` unchanged when set. Otherwise builds a light
    sequential gradient from ``color`` (or ``pp.rcParams["color"]`` when
    unset) via ``sns.color_palette("light:<color>", as_cmap=True)`` so
    2D primitives match the user's themed color.
    """
    if cmap is not None:
        return cmap
    import seaborn as sns
    base_color = resolve_param("color", color)
    return sns.color_palette(f"light:{base_color}", as_cmap=True)


def resolve_palette_map(
    values: Optional[List[str]] = None,
    palette: Optional[Union[str, Dict, List]] = None,
) -> Dict[str, str]:
    """
    Resolve a palette mapping to actual colors (internal helper).

    Maps categorical values to colors, handling various palette specifications.

    Parameters
    ----------
    values : List[str], optional
        List of categorical values to map to colors.
    palette : str, dict, or list, optional
        Palette specification:
        - None: Uses default palette
        - str: Palette name (checks publiplots first, then seaborn)
        - list: List of color hex codes
        - dict: Pre-mapped categories to colors (returned as-is)

    Returns
    -------
    Dict[str, str]
        Dictionary mapping each value to a color.
    """
    if values is None:
        return {}
    if isinstance(palette, dict):
        return palette
    if not is_categorical(values):
        return palette  # continuous mapping

    # Use color_palette internally
    palette = color_palette(palette, n_colors=len(values))
    return {value: palette[i % len(palette)] for i, value in enumerate(values)}
