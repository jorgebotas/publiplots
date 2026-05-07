"""
Venn diagram visualizations for publiplots.

This module provides functions for creating Venn diagrams for 2-5 sets
using geometry based on the ggvenn R package by Yan Linlin.

The module supports:
- Venn diagrams for 2-5 sets using ellipses

Based on ggvenn by Yan Linlin: https://github.com/yanlinlin82/ggvenn
"""

from matplotlib.axes import Axes
from typing import Dict, List, Optional, Tuple, Union

from publiplots.themes.rcparams import resolve_param
import matplotlib.pyplot as plt

from publiplots.themes.colors import color_palette

from .draw import (
    init_axes,
    draw_ellipse,
    draw_text,
)
from .logic import get_n_sets, generate_petal_labels
from .geometry import get_geometry, get_coordinate_ranges
import numpy as np


def _prepare_colors(colors, n_sets: int) -> Union[List[str], List[Tuple[float, ...]]]:
    """
    Prepare colors for Venn diagram sets.

    Parameters
    ----------
    colors : list, str, or None
        Colors specification (list of colors, colormap name, or None)
    n_sets : int
        Number of sets (2-5)

    Returns
    -------
    colors : list of str or tuple of floats (RGBA)
        List of color strings or RGBA tuples
    """
    if colors is None:
        # Use default publiplots palette
        color_list = color_palette('pastel', n_colors=n_sets)
    elif isinstance(colors, str):
        # Use publiplots palette or colormap by name
        color_list = color_palette(colors, n_colors=n_sets)
    elif isinstance(colors, list):
        # Use provided color list
        color_list = colors[:n_sets]
    else:
        raise TypeError("colors must be None, a string (palette/colormap name), or a list of colors")

    # Convert to RGBA with specified alpha
    return color_list


def _venn(
    *,
    petal_labels: Dict[str, str],
    dataset_labels: List[str],
    colors: List[Tuple[float, ...]],
    alpha: float,
    ax: Optional[Axes],
    color_labels: bool = True,
) -> Axes:
    """
    Draw a true Venn diagram with ellipses (2-5 sets).

    Internal function that handles the actual drawing logic.
    Uses dynamic geometry calculation based on ggvenn approach.
    """
    n_sets = get_n_sets(petal_labels, dataset_labels)

    # Validate number of sets
    if n_sets < 2 or n_sets > 5:
        raise ValueError("Number of sets must be between 2 and 5. Consider using upset plot instead.")

    # Get dynamic geometry
    circles, label_positions, set_label_positions = get_geometry(n_sets)

    # Calculate coordinate ranges with padding
    x_range, y_range = get_coordinate_ranges(circles)
    padding = 0.15
    x_width = x_range[1] - x_range[0]
    y_height = y_range[1] - y_range[0]

    xlim = (x_range[0] - padding * x_width, x_range[1] + padding * x_width)
    ylim = (y_range[0] - padding * y_height, y_range[1] + padding * y_height)

    # Initialize axes with proper limits
    ax = init_axes(ax, xlim=xlim, ylim=ylim)

    # Draw all circles/ellipses
    for circle, color in zip(circles, colors):
        # Convert angle from radians to degrees for matplotlib
        angle_deg = np.degrees(circle.theta_offset)

        # Width and height are diameter (2 * radius)
        width = 2 * circle.radius_a
        height = 2 * circle.radius_b

        draw_ellipse(
            ax,
            circle.x_offset,
            circle.y_offset,
            width,
            height,
            angle_deg,
            color,
            alpha=alpha
        )

    # Draw labels for each petal (intersection region)
    # Font size controlled by rcParams['font.size']
    for logic, petal_label in petal_labels.items():
        if logic in label_positions:
            x, y = label_positions[logic]
            draw_text(ax, x, y, petal_label, fontsize=plt.rcParams['font.size'])

    # Draw set labels on diagram
    # Set label alignments based on position
    set_label_alignments = _get_set_label_alignments(n_sets, set_label_positions, circles)

    for i, label in enumerate(dataset_labels):
        x, y = set_label_positions[i]
        ha, va = set_label_alignments[i]
        color = colors[i] if color_labels else None
        draw_text(ax, x, y, label, fontsize=plt.rcParams['font.size'] * 1.2, color=color, ha=ha, va=va)

    return ax


def _get_set_label_alignments(
    n_sets: int,
    set_label_positions: List[Tuple[float, float]],
    circles: List,
) -> List[Tuple[str, str]]:
    """
    Calculate text alignments for set labels based on their position relative to circles.

    Parameters
    ----------
    n_sets : int
        Number of sets
    set_label_positions : List[Tuple[float, float]]
        Positions of set labels
    circles : List[Circle]
        Circle objects

    Returns
    -------
    alignments : List[Tuple[str, str]]
        List of (horizontal_alignment, vertical_alignment) tuples
    """
    alignments = []

    for i, (label_x, label_y) in enumerate(set_label_positions):
        circle = circles[i]

        # Determine alignment based on relative position
        # Horizontal alignment
        if label_x < circle.x_offset - 0.1:
            ha = "right"
        elif label_x > circle.x_offset + 0.1:
            ha = "left"
        else:
            ha = "center"

        if i == 0 and n_sets == 5:
            # Special case for first set in 5-way Venn diagram
            # located at the top center of the diagram
            # Take into account ellipse offset from center
            ha = "center"

        # Vertical alignment
        if label_y < circle.y_offset - 0.1:
            va = "top"
        elif label_y > circle.y_offset + 0.1:
            va = "bottom"
        else:
            va = "center"

        alignments.append((ha, va))

    return alignments


# =============================================================================
# Public API
# =============================================================================


def venn(
    sets: Union[List[set], Dict[str, set]],
    labels: Optional[List[str]] = None,
    colors: Optional[Union[List[str], str]] = None,
    alpha: Optional[float] = None,
    ax: Optional[Axes] = None,
    fmt: str = "{size}",
    color_labels: bool = True,
) -> Axes:
    """
    Create a Venn diagram for 2-5 sets.

    This function creates true Venn diagrams that show all possible intersections
    using ellipses (2-5 sets supported). Each region (petal) is labeled with the
    size of the intersection by default; use ``fmt`` to show logic strings or
    percentages instead. For more than 5 sets, prefer
    :func:`publiplots.upsetplot`.

    Parameters
    ----------
    sets : list of sets or dict
        Either a list of 2-5 sets, or a dictionary mapping labels to sets.
        Example: [set1, set2, set3] or {'A': set1, 'B': set2, 'C': set3}
    labels : list of str, optional
        Labels for each set. If sets is a dict, labels are taken from keys.
        Default: ['Set A', 'Set B', 'Set C', ...]
    colors : list of str, str, or None, optional
        Colors for each set. Can be:
        - List of color names/codes for each set
        - String name of a publiplots palette or seaborn palette
        - None (uses 'pastel' palette)
    alpha : float, optional
        Transparency of set regions (0=transparent, 1=opaque). When None,
        resolved from ``publiplots.rcParams["alpha"]``.
    ax : Axes, optional
        Matplotlib axes object. If None, creates new figure.
    fmt : str, default='{size}'
        Format string for region labels. Can include:
        - {size}: number of elements in the intersection
        - {logic}: binary string representing the intersection
        - {percentage}: percentage of total elements
    color_labels : bool, default=True
        Whether to color the set labels with the same color as the petals.

    Returns
    -------
    Axes
        The axes where the Venn diagram was drawn.

    Raises
    ------
    ValueError
        If the number of sets is not between 2 and 5
        Consider using upset plot instead.
    TypeError
        If sets is not a list of sets or dict of sets

    Examples
    --------
    Simple 2-way Venn diagram:

    >>> set1 = {1, 2, 3, 4, 5}
    >>> set2 = {4, 5, 6, 7, 8}
    >>> ax = pp.venn([set1, set2], labels=['Group A', 'Group B'])

    3-way Venn with custom colors:

    >>> sets_dict = {'A': set1, 'B': set2, 'C': set3}
    >>> colors = ['red', 'blue', 'green']
    >>> ax = pp.venn(sets_dict, colors=colors)

    4-way Venn with colormap:

    >>> ax = pp.venn([set1, set2, set3, set4], colors='Set1')

    5-way Venn diagram with percentage labels:

    >>> ax = pp.venn(
    ...     [set1, set2, set3, set4, set5],
    ...     fmt='{size} ({percentage:.1f}%)'
    ... )

    See Also
    --------
    publiplots.upsetplot : UpSet plot, preferred for 6+ sets or complex intersections.
    """
    # Read defaults from rcParams if not provided
    alpha = resolve_param("alpha", alpha)

    # Parse input sets
    if isinstance(sets, dict):
        labels = list(sets.keys())
        sets_list = [set(s) for s in sets.values()]
    else:
        sets_list = [set(s) for s in sets]
        if labels is None:
            labels = [f"Set {chr(65+i)}" for i in range(len(sets_list))]

    # Validate number of sets
    n_sets = len(sets_list)
    if n_sets < 2 or n_sets > 5:
        raise ValueError("Venn diagram supports 2 to 5 sets. Consider using upset plot instead.")

    # Validate that all inputs are sets
    for s in sets_list:
        if not isinstance(s, set):
            raise TypeError("All elements must be sets")

    # Prepare colors using publiplots color utilities
    colors = _prepare_colors(colors, n_sets)

    # Generate petal labels (intersection sizes)
    petal_labels = generate_petal_labels(sets_list, fmt=fmt)

    # Create figure via pp.subplots to install SubplotsAutoLayout; users who
    # want custom dimensions should compose with pp.subplots(axes_size=...)
    # before calling and pass ax=.
    if ax is None:
        from publiplots.layout.subplots import subplots as _pp_subplots
        fig, ax = _pp_subplots()
    else:
        fig = ax.get_figure()

    # Draw the Venn diagram
    ax = _venn(
        petal_labels=petal_labels,
        dataset_labels=labels,
        colors=colors,
        alpha=alpha,
        ax=ax,
        color_labels=color_labels,
    )

    return ax