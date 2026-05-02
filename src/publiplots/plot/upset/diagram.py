"""
Main UpSet plot function.

This module provides the main user-facing upsetplot() function for creating
UpSet plot visualizations of set intersections.

Portions of this implementation are based on concepts from UpSetPlot:
https://github.com/jnothman/UpSetPlot
Copyright (c) 2016, Joel Nothman
Licensed under BSD-3-Clause
"""

from typing import Dict, Optional, Set, Tuple, Union

from publiplots.themes.rcparams import resolve_param
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd

from .logic import process_upset_data
from .draw import (
    setup_upset_axes,
    draw_intersection_bars,
    draw_set_size_bars,
    draw_matrix,
    add_upset_labels,
)


def upsetplot(
    data: Union[pd.DataFrame, pd.Series, Dict[str, Set]],
    sort_by: str = "size",
    ascending: bool = False,
    min_subset_size: Optional[int] = None,
    max_subset_size: Optional[int] = None,
    min_degree: int = 1,
    max_degree: Optional[int] = None,
    show_counts: int = 20,
    color: Optional[str] = None,
    bar_linewidth: Optional[float] = None,
    matrix_linewidth: Optional[float] = None,
    alpha: Optional[float] = None,
    dotsize: Optional[float] = None,
    elementsize: Optional[float] = None,
    title: str = "",
    intersection_label: str = "",
    set_label: str = ""
) -> Dict[str, Axes]:
    """
    Create an UpSet plot for visualizing set intersections.

    UpSet plots [1]_ are an effective way to visualize intersections of multiple sets,
    providing more clarity than Venn diagrams when dealing with many sets or
    complex intersection patterns. This implementation is based on the UpSetPlot
    package [2]_.

    Parameters
    ----------
    data : DataFrame, Series, or dict of sets
        Input data in one of the following formats:

        - **DataFrame**: Each column represents a set, rows are elements.
          Values should be binary (0/1 or True/False) indicating membership.

        - **Series**: MultiIndex series where first level is elements and
          second level is sets, with binary values.

        - **dict**: Dictionary mapping set names (str) to sets of elements.
          Example: ``{'Set A': {1, 2, 3}, 'Set B': {2, 3, 4}}``

    sort_by : {'size', 'degree', 'name'}, default='size'
        How to sort intersections:

        - 'size': Sort by intersection size (largest first if ascending=False)
        - 'degree': Sort by number of sets in intersection
        - 'name': Sort alphabetically by set names

    ascending : bool, default=False
        Sort order. False shows largest/highest degree first.

    min_subset_size : int, optional
        Minimum size for an intersection to be displayed. Useful for
        filtering out small intersections.

    max_subset_size : int, optional
        Maximum size for an intersection to be displayed.

    min_degree : int, default=1
        Minimum number of sets in an intersection. Set to 2 to exclude
        individual sets.

    max_degree : int, optional
        Maximum number of sets in an intersection. Useful for focusing on
        simpler intersections.

    show_counts : int, default=20
        Maximum number of intersections to display in the plot.

    color : str, optional
        Color for bars (both intersection and set size bars).
        Supports any matplotlib color specification.

    bar_linewidth : float, optional
        Width of edges around bars.

    matrix_linewidth : float, optional
        Width of lines connecting dots in the matrix.

    alpha : float, optional
        Transparency level for bars (0=transparent, 1=opaque).

    dotsize : float, optional
        Size of dots in the membership matrix (area in points²).
        If not specified, calculated from elementsize using circle geometry:
        ``dotsize = π * (elementsize * 0.7 / 2)²``.
        Typical values: 400-900.

    elementsize : float, optional
        Width of each matrix cell/bar in figure points (1/72 inch).
        Controls the overall scale of the plot. If not specified,
        calculated from dotsize using circle geometry:
        ``diameter = 2 * sqrt(dotsize / π); elementsize = diameter / 0.7``
        or defaults to 48. Typical values: 32 (compact), 48 (default), 64 (spacious).
        The figure size is automatically calculated to maintain proper
        proportions based on this value.

    title : str, default=""
        Main plot title.

    intersection_label : str, default=""
        Label for the y-axis of intersection size bars.

    set_label : str, default=""
        Label for the x-axis of set size bars.

    Returns
    -------
    dict
        Dict with keys ``"intersections"``, ``"matrix"``, ``"sets"``, each
        mapping to an Axes:
        - ``"intersections"``: Intersection size bar plot (top)
        - ``"matrix"``: Set membership matrix (middle)
        - ``"sets"``: Set size bar plot (left)

    Notes
    -----
    UpSet plots consist of three main components:

    1. **Intersection size bars** (top): Show the number of elements in each
       intersection, sorted by the specified criterion.

    2. **Membership matrix** (middle): Visualizes which sets contribute to each
       intersection using dots and connecting lines. Each column represents one
       intersection, each row represents one set.

    3. **Set size bars** (left): Show the total size of each individual set.

    The implementation is inspired by the UpSetPlot package
    (https://github.com/jnothman/UpSetPlot) but redesigned to match the
    publiplots aesthetic with cleaner styling and integration with existing
    publiplots utilities.

    Examples
    --------
    Create an UpSet plot from a dictionary of sets:

    >>> data = {
    ...     'Set A': {1, 2, 3, 4, 5},
    ...     'Set B': {3, 4, 5, 6, 7},
    ...     'Set C': {5, 6, 7, 8, 9}
    ... }
    >>> axes = upsetplot(data, title='Set Intersections')

    Create from a DataFrame with binary membership:

    >>> df = pd.DataFrame({
    ...     'Set A': [1, 1, 1, 0, 0],
    ...     'Set B': [0, 1, 1, 1, 0],
    ...     'Set C': [0, 0, 1, 1, 1]
    ... })
    >>> axes = upsetplot(df, sort_by='degree', min_degree=2)

    Filter to show only intersections with at least 10 elements:

    >>> axes = upsetplot(data, min_subset_size=10, show_counts=15)

    Customize colors and styling:

    >>> axes = upsetplot(
    ...     data,
    ...     color='#ff6b6b',
    ...     elementsize=64  # Larger, more spacious plot
    ... )

    Control plot size via elementsize:

    >>> axes = upsetplot(
    ...     data,
    ...     elementsize=32  # Compact plot
    ... )
    >>> axes = upsetplot(
    ...     data,
    ...     elementsize=64  # Spacious plot
    ... )

    See Also
    --------
    venn : Create Venn diagrams for 2-5 sets
    barplot : Create bar plots with grouping and styling
    scatterplot : Create scatter plots with size and color encoding

    References
    ----------
    .. [1] Lex et al. (2014). "UpSet: Visualization of Intersecting Sets".
       IEEE Transactions on Visualization and Computer Graphics.
    .. [2] UpSetPlot package: https://github.com/jnothman/UpSetPlot
    """
    # Read defaults from rcParams if not provided
    color = resolve_param("color", color)
    bar_linewidth = resolve_param("lines.linewidth", bar_linewidth)
    matrix_linewidth = resolve_param("lines.linewidth") * 1.2 if matrix_linewidth is None else matrix_linewidth
    alpha = resolve_param("alpha", alpha)

    # Handle elementsize and dotsize relationship using proper circle geometry
    # elementsize: physical dimensions (bars/cells width in points)
    # dotsize: scatter marker area in points² (area = π * r²)
    # Relationship: dot diameter ≈ elementsize * 0.5 (dot takes 50% of cell width)
    import numpy as np

    DOT_TO_CELL_RATIO = 0.4  # Dot diameter as fraction of cell width

    if elementsize is None and dotsize is None:
        # Both unspecified: use sensible defaults
        elementsize = resolve_param("lines.markersize") * 4
        dot_diameter = elementsize * DOT_TO_CELL_RATIO
        dotsize = np.pi * (dot_diameter / 2) ** 2  # Area = π * r²
    elif elementsize is None:
        # Only dotsize specified: calculate elementsize from dot area
        dot_diameter = 2 * np.sqrt(dotsize / np.pi)  # diameter = 2 * sqrt(area/π)
        elementsize = dot_diameter / DOT_TO_CELL_RATIO
    elif dotsize is None:
        # Only elementsize specified: calculate dotsize from element width
        dot_diameter = elementsize * DOT_TO_CELL_RATIO
        dotsize = np.pi * (dot_diameter / 2) ** 2
    # else: both specified, use as-is

    # Process data
    processed = process_upset_data(
        data=data,
        sort_by=sort_by,
        ascending=ascending,
        min_subset_size=min_subset_size,
        max_subset_size=max_subset_size,
        min_degree=min_degree,
        max_degree=max_degree,
        show_counts=show_counts,
    )

    intersections = processed["intersections"]
    membership_matrix = processed["membership_matrix"]
    set_names = processed["set_names"]
    set_sizes = processed["set_sizes"]
    n_sets = processed["n_sets"]
    n_intersections = processed["n_intersections"]

    # Create a bare figure; setup_upset_axes() immediately resizes it (via
    # fig.set_figwidth/set_figheight) based on elementsize, so we don't
    # reserve space or install SubplotsAutoLayout here — UpSet manages its
    # own gridspec and figure dimensions.
    fig = plt.figure()

    # Setup axes with proper sizing that maintains proportions
    # Figure size is automatically calculated based on elementsize
    ax_intersections, ax_matrix, ax_sets, intersection_bar_width, set_bar_width = setup_upset_axes(
        fig=fig,
        set_names=set_names,
        n_intersections=n_intersections,
        elementsize=elementsize
    )

    # Draw intersection size bars
    intersection_sizes = intersections["size"].tolist()
    intersection_positions = list(range(n_intersections))

    draw_intersection_bars(
        ax=ax_intersections,
        sizes=intersection_sizes,
        positions=intersection_positions,
        width=intersection_bar_width,
        color=color,
        linewidth=bar_linewidth,
        alpha=alpha,
    )

    # Draw set size bars (reverse order for bottom-to-top display)
    set_positions = list(range(n_sets))

    draw_set_size_bars(
        ax=ax_sets,
        set_names=set_names,
        set_sizes=set_sizes,
        positions=set_positions,
        width=set_bar_width,
        color=color,
        linewidth=bar_linewidth,
        alpha=alpha,
    )

    # Draw membership matrix
    draw_matrix(
        ax=ax_matrix,
        membership_matrix=membership_matrix,
        set_names=set_names,
        dotsize=dotsize,
        linewidth=matrix_linewidth,
        active_color=color,
        alpha=alpha,
    )

    # Add labels and title
    add_upset_labels(
        fig=fig,
        ax_intersections=ax_intersections,
        ax_sets=ax_sets,
        title=title,
        intersection_label=intersection_label,
        set_label=set_label,
    )

    # Reserve vertical space for decorations (bar-top annotations, xlabel,
    # title, ticklabels) that would otherwise clip because the gridspec spans
    # the full figure (top=1/bottom=0). Draw once, measure overflow, grow the
    # figure by the measured amount, and shift the gridspec inward so panel
    # dimensions stay unchanged.
    _reserve_upset_decoration_space(fig, ax_intersections, ax_sets, ax_matrix)

    return {
        "intersections": ax_intersections,
        "matrix": ax_matrix,
        "sets": ax_sets,
    }


def _reserve_upset_decoration_space(fig, ax_intersections, ax_sets, ax_matrix):
    """Grow the figure and shift the gridspec so decorations don't clip.

    UpSet's gridspec uses ``top=1, bottom=0`` (zero margin). Bar-top
    annotations, xlabels, titles, and ticklabels therefore extend outside
    the figure canvas and clip on ``plt.show()`` or non-tight ``savefig``.

    Measurement-driven fix: draw once, compute overflow in inches at the
    top and bottom of the figure, grow ``figheight`` by that amount, then
    shift the gridspec ``top``/``bottom`` inward by the same fractional
    amount so the plotted panels keep their original dimensions.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    fig_bb = fig.bbox_inches

    # Overflow in inches; negative means safely inside the canvas.
    top_overflow = max(0.0, tight.y1 - fig_bb.y1)
    bottom_overflow = max(0.0, -tight.y0)

    if top_overflow == 0.0 and bottom_overflow == 0.0:
        return  # Nothing to reserve.

    # Small padding beyond the measured overflow so borders don't kiss the
    # canvas edge.
    pad_in = 4.0 / 72.0  # ~4 pt
    top_pad_in = top_overflow + pad_in if top_overflow > 0 else 0.0
    bottom_pad_in = bottom_overflow + pad_in if bottom_overflow > 0 else 0.0

    current_h_in = fig.get_figheight()
    new_h_in = current_h_in + top_pad_in + bottom_pad_in
    fig.set_figheight(new_h_in)

    # Recover the gridspec from any of its subplots and shift top/bottom
    # inward by the padding we just added.
    gs = ax_intersections.get_gridspec()
    gs.update(
        top=1.0 - top_pad_in / new_h_in,
        bottom=bottom_pad_in / new_h_in,
    )
