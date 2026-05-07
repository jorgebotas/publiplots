"""
Marker definitions and utilities for publiplots.

This module provides standard marker sets and utilities for consistent
marker usage across visualizations.
"""

from typing import List, Dict, Tuple, Optional, Union
from publiplots.themes.rcparams import resolve_param


# =============================================================================
# Marker Sets
# =============================================================================

STANDARD_MARKERS: List[str] = [
    'o',   # Circle
    's',   # Square
    '^',   # Triangle up
    'D',   # Diamond
    'v',   # Triangle down
    'p',   # Pentagon
    '*',   # Star
    'h',   # Hexagon
    '<',   # Triangle left
    '>',   # Triangle right
]
"""
Standard set of distinguishable markers for categorical data.

Use case: when plotting multiple categories in scatter or line plots
where shape needs to distinguish groups in addition to (or instead of)
color. Used as the default pool by :func:`resolve_markers` and
:func:`resolve_marker_map`.

Example:
    >>> import publiplots as pp
    >>> pp.scatterplot(df, x='x', y='y', style='cat',
    ...                markers=pp.STANDARD_MARKERS)
"""

FILLED_UNFILLED_MARKERS: Dict[str, Tuple[str, str]] = {
    "circle": ('o', 'o'),          # Circle (same filled/unfilled)
    "square": ('s', 's'),          # Square
    "triangle_up": ('^', '^'),     # Triangle up
    "triangle_down": ('v', 'v'),   # Triangle down
    "diamond": ('D', 'd'),         # Diamond (filled, unfilled)
    "pentagon": ('p', 'p'),        # Pentagon
    "hexagon": ('h', 'h'),         # Hexagon
    "star": ('*', '*'),            # Star
}
"""
Mapping of marker names to (filled, unfilled) marker codes.

Use case: When you want to use the same shape but distinguish groups
by fill status (e.g., treatment vs control).

Example:
    >>> markers = pp.FILLED_UNFILLED_MARKERS
    >>> pp.scatterplot(treated, marker=markers['circle'][0])   # Filled
    >>> pp.scatterplot(control, marker=markers['circle'][1])   # Unfilled
"""


# =============================================================================
# Marker Size Recommendations
# =============================================================================

MARKER_SIZES: Dict[str, float] = {
    "small": 20,
    "medium": 50,
    "large": 100,
    "xlarge": 200,
}
"""
Recommended marker sizes for scatter plots (in points^2).

Use case: Provides consistent sizing across plots. These sizes are
calibrated to work well with publiplots's default styling.

Example:
    >>> pp.scatterplot(data, x='x', y='y', s=pp.MARKER_SIZES['medium'])
"""


# =============================================================================
# Functions
# =============================================================================

def resolve_markers(
    markers: Optional[List[str]] = None,
    n_markers: Optional[int] = None,
    reverse: bool = False
) -> List[str]:
    """
    Resolve marker patterns for plotting.

    Helper that standardises marker specifications into a concrete list.
    Handles marker cycling for arbitrary category counts and supports
    reversal.

    Parameters
    ----------
    markers : list of str, optional
        Marker symbols to use. If ``None``, uses
        :data:`STANDARD_MARKERS`.
    n_markers : int, optional
        Number of markers to return. If provided, the source list is
        cycled to reach this count. If ``None``, returns the full
        source list.
    reverse : bool, default ``False``
        Whether to reverse the final marker order. Useful for changing
        visual hierarchy.

    Returns
    -------
    list of str
        Resolved marker symbols.

    Examples
    --------
    Get default markers:

    >>> import publiplots as pp
    >>> markers = pp.resolve_markers()
    >>> len(markers)
    10

    Get exactly 5 markers with cycling:

    >>> markers = pp.resolve_markers(n_markers=5)
    >>> len(markers)
    5

    Use custom markers with cycling:

    >>> markers = pp.resolve_markers(markers=['o', '^', 's'], n_markers=7)

    Get reversed markers:

    >>> markers = pp.resolve_markers(n_markers=4, reverse=True)

    See Also
    --------
    resolve_marker_map : Create a mapping from category values to markers.
    STANDARD_MARKERS : Default marker pool.
    """
    # Get default markers if not provided
    if markers is None:
        markers = STANDARD_MARKERS

    # Cycle markers if n_markers specified
    if n_markers is not None:
        markers = [markers[i % len(markers)] for i in range(n_markers)]

    # Reverse if requested
    if reverse:
        markers = markers[::-1]

    return markers


def resolve_marker_map(
    values: Optional[List[str]] = None,
    marker_map: Optional[Union[Dict[str, str], List[str]]] = None,
    reverse: bool = False
) -> Dict[str, str]:
    """
    Create a mapping from category values to marker symbols.

    Returns a dictionary mapping category names to specific marker
    symbols. Useful for categorical plots such as
    :func:`publiplots.scatterplot` with a ``style=`` mapping — it
    ensures consistent marker assignment across multiple plots.

    Parameters
    ----------
    values : list of str, optional
        Category values to map to markers. If ``None``, returns an
        empty dict.
    marker_map : dict or list of str, optional
        Marker specification:

        - ``dict`` — explicit mapping from values to markers; returned
          as-is.
        - ``list`` — markers cycled through the ``values`` list.
        - ``None`` — defaults to :data:`STANDARD_MARKERS`.
    reverse : bool, default ``False``
        Whether to reverse the marker assignment order. Only applicable
        when ``marker_map`` is a list or ``None``.

    Returns
    -------
    dict of {str: str}
        Mapping from category values to marker symbols.

    Examples
    --------
    Create a mapping for categories:

    >>> import publiplots as pp
    >>> categories = ['A', 'B', 'C', 'D']
    >>> mapping = pp.resolve_marker_map(values=categories)
    >>> mapping['A']
    'o'
    >>> mapping['B']
    's'

    Use custom markers:

    >>> mapping = pp.resolve_marker_map(
    ...     values=['cat', 'dog', 'bird'],
    ...     marker_map=['o', '^', 's'],
    ... )

    Use an explicit mapping (returned unchanged):

    >>> mapping = pp.resolve_marker_map(
    ...     values=['A', 'B'],
    ...     marker_map={'A': 'o', 'B': '^'},
    ... )
    >>> mapping
    {'A': 'o', 'B': '^'}

    See Also
    --------
    resolve_markers : Resolve markers without creating a mapping.
    STANDARD_MARKERS : Default marker pool.
    """
    # Return empty dict if no values provided
    if values is None:
        return {}

    # If already a dict, return as-is
    if isinstance(marker_map, dict):
        return marker_map

    # Resolve markers and create mapping
    markers = resolve_markers(
        markers=marker_map,
        n_markers=len(values),
        reverse=reverse
    )

    return {value: marker for value, marker in zip(values, markers)}


def resolve_size_map(
    values: List[float],
    size_range: Optional[Tuple[float, float]] = None,
    method: str = "linear"
) -> List[float]:
    """
    Map data values to marker sizes.

    Parameters
    ----------
    values : List[float]
        Data values to map.
    size_range : Tuple[float, float], optional
        (min_size, max_size) in points^2. If None, reads from rcParams
        ('scatter.size_min', 'scatter.size_max').
    method : str, default='linear'
        Mapping method: 'linear' or 'log'.

    Returns
    -------
    List[float]
        Mapped marker sizes.

    Examples
    --------
    Map p-values to sizes:
    >>> pvalues = [0.001, 0.01, 0.05, 0.1]
    >>> neg_log_p = [-np.log10(p) for p in pvalues]
    >>> sizes = resolve_size_map(neg_log_p, size_range=(50, 500))

    Use default size range from rcParams:
    >>> sizes = resolve_size_map(neg_log_p)
    """
    import numpy as np

    values = np.array(values)

    # Get size range from rcParams if not provided
    if size_range is None:
        min_size = resolve_param('scatter.size_min', None)
        max_size = resolve_param('scatter.size_max', None)
        size_range = (min_size, max_size)

    min_size, max_size = size_range

    if method == "linear":
        # Linear scaling
        v_min, v_max = values.min(), values.max()
        if v_max == v_min:
            return [min_size] * len(values)
        normalized = (values - v_min) / (v_max - v_min)
        sizes = min_size + normalized * (max_size - min_size)

    elif method == "log":
        # Log scaling
        log_values = np.log10(values + 1)  # Add 1 to avoid log(0)
        v_min, v_max = log_values.min(), log_values.max()
        if v_max == v_min:
            return [min_size] * len(values)
        normalized = (log_values - v_min) / (v_max - v_min)
        sizes = min_size + normalized * (max_size - min_size)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'linear' or 'log'.")

    return sizes.tolist()
