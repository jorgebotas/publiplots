"""
Drawing utilities for UpSet plots.

This module handles the low-level rendering of UpSet plot components including
intersection bars, set size bars, and the membership matrix visualization.

Portions of this implementation are based on concepts from UpSetPlot:
https://github.com/jnothman/UpSetPlot
Copyright (c) 2016, Joel Nothman
Licensed under BSD-3-Clause
"""

from typing import Dict, List, Optional, Tuple

from publiplots.themes.rcparams import resolve_param
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator

GRID_LINEWIDTH = 1
BARWIDTH = 0.5


def draw_intersection_bars(
    ax: Axes,
    sizes: List[int],
    positions: List[int],
    width: float = BARWIDTH,
    color: Optional[str] = None,
    linewidth: Optional[float] = None,
    alpha: Optional[float] = None,
) -> None:
    """
    Draw bars showing intersection sizes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes for the intersection size plot
    sizes : list of int
        Intersection sizes
    positions : list of int
        X positions for each bar
    width : float
        Width of the bars
    color : str
        Bar color
    linewidth : float
        Edge line width
    alpha : float
        Bar transparency
    """
    # Read defaults from rcParams if not provided
    color = resolve_param("color", color)
    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)

    ax.bar(
        positions,
        sizes,
        width=width,
        color=to_rgba(color, alpha=alpha),
        edgecolor=color,
        linewidth=linewidth,
        zorder=2,
    )

    # Style axes
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.set_xticks([])
    ax.set_ylim(-0.02 * max(sizes), 1.05 * max(sizes))
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=GRID_LINEWIDTH)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))

    # Add value labels on top of bars
    for i, (pos, size) in enumerate(zip(positions, sizes)):
        ax.text(
            pos,
            size,
            str(size),
            ha="center",
            va="bottom",
        )


def draw_set_size_bars(
    ax: Axes,
    set_names: List[str],
    set_sizes: Dict[str, int],
    positions: List[int],
    width: float = BARWIDTH,
    color: Optional[str] = None,
    linewidth: Optional[float] = None,
    alpha: Optional[float] = None,
) -> None:
    """
    Draw horizontal bars showing set sizes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes for set size plot
    set_names : list
        Names of sets (bottom to top order)
    set_sizes : dict
        Mapping from set name to size
    positions : list
        Y positions for each bar
    width : float
        Width of the bars
    color : str
        Bar color
    linewidth : float
        Edge line width
    alpha : float
        Bar transparency
    """
    # Read defaults from rcParams if not provided
    color = resolve_param("color", color)
    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)

    sizes = [set_sizes[name] for name in set_names]

    ax.barh(
        positions,
        sizes,
        height=width,
        color=to_rgba(color, alpha=alpha),
        edgecolor=color,
        linewidth=linewidth,
        zorder=2,
    )

    # Style axes
    ax.set_xlim(-0.02 * max(sizes), 1.05 * max(sizes))
    ax.set_ylim(-0.5, len(positions) - 0.5)
    ax.set_yticks([])  # No tick marks or labels on set bars
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=GRID_LINEWIDTH)
    ax.set_axisbelow(True)
    ax.invert_xaxis()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))



def draw_matrix(
        ax: Axes,
        membership_matrix: List[Tuple[int, ...]],
        set_names: List[str],
        dotsize: float = 150,
        linewidth: Optional[float] = None,
        active_color: Optional[str] = None,
        inactive_color: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> None:
    """
    Draw the membership matrix showing which sets each intersection contains.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes for matrix
    membership_matrix : list of tuples
        Binary membership patterns (each tuple is one column)
    set_names : list
        Names of sets (corresponds to rows, bottom to top)
    dotsize : float
        Size of dots in the matrix
    linewidth : float
        Width of connecting lines
    active_color : str, optional
        Color for active set membership. If None, use DEFAULT_COLOR.
    inactive_color : str, optional
        Color for inactive dots. If None, use to_rgba(color, alpha=alpha).
    """
    import numpy as np

    # Read defaults from rcParams if not provided
    linewidth = resolve_param("lines.linewidth", linewidth)
    active_color = resolve_param("color", active_color)
    alpha = resolve_param("alpha", alpha)

    n_sets = len(set_names)
    n_intersections = len(membership_matrix)

    # Set positions (y-axis: one row per set)
    set_positions = list(range(n_sets))

    # Intersection positions (x-axis: one column per intersection)
    intersection_positions = list(range(n_intersections))

    # Set inactive color
    if inactive_color is None:
        inactive_color = to_rgba(active_color, alpha=alpha)

    # Set axis limits first (needed for transformation calculations)
    ax.set_xlim(-0.5, n_intersections - 0.5)
    ax.set_ylim(-0.5, n_sets - 0.5)

    # Draw dots for all positions
    for i, membership in enumerate(membership_matrix):
        active_sets = [j for j, is_member in enumerate(membership) if is_member]

        # Draw inactive dots (light gray)
        for j in range(n_sets):
            # Draw inactive dot
            ax.scatter(
                i,
                j,
                s=dotsize,
                color=inactive_color if j not in active_sets else "white",
                marker="o",
                zorder=2,
                linewidths=0,
            )

        x_coords = [i] * len(active_sets)
        y_coords = active_sets
        ax.plot(
            x_coords,
            y_coords,
            color=active_color,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=1,
        )

        # Draw active dots (dark)
        for j in active_sets:
            ax.scatter(
                i,
                j,
                s=dotsize,
                color=to_rgba(active_color, alpha=alpha),
                marker="o",
                edgecolors=active_color,
                linewidths=linewidth,
                zorder=4,
            )

    # Style axes
    ax.set_xlim(-0.5, n_intersections - 0.5)
    ax.set_ylim(-0.5, n_sets - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Add horizontal grid lines between sets
    for i in range(n_sets - 1):
        ax.axhline(
            i + 0.5,
            color="#e0e0e0",
            linewidth=GRID_LINEWIDTH,
            linestyle="-",
            zorder=0
        )

    # Add set name labels on the left side of the matrix
    for i, name in enumerate(set_names):
        ax.text(
            -0.6,  # Position to the left of the leftmost column
            i,
            str(name),
            ha="right",
            va="center",
            fontsize=resolve_param("ytick.labelsize"),
            fontweight="normal",
        )


def setup_upset_axes(
    fig: plt.Figure,
    set_names: List[str],
    n_intersections: int,
    elementsize: float,
) -> Tuple[Axes, Axes, Axes]:
    """
    Set up the three-panel layout for UpSet plot with proper sizing.

    This function calculates figure dimensions to ensure:
    1. Set label text has dedicated space between bars and matrix
    2. Matrix elements are square (width = height)
    3. Bar widths in both plots equal matrix element dimensions

    Parameters
    ----------
    fig : Figure
        Matplotlib figure (size will be adjusted)
    set_names : list
        Names of sets (for measuring text width)
    n_intersections : int
        Number of intersections to display
    elementsize : float
        Width of each element (bar/dot) in figure points. Controls the
        overall scale of the plot. Figure size is calculated to maintain
        proper proportions based on this value.

    Returns
    -------
    ax_intersections : Axes
        Axes for intersection size bars (top)
    ax_matrix : Axes
        Axes for membership matrix (middle)
    ax_sets : Axes
        Axes for set size bars (left)
    intersection_bar_width : float
        Optimal bar width for intersection bars in data coordinates
    set_bar_width : float
        Optimal bar width for set bars in data coordinates
    """
    from matplotlib import gridspec
    import numpy as np

    n_sets = len(set_names)

    # Measure text width needed for set labels
    text_kw = {"size": resolve_param("ytick.labelsize")}
    # Add "x" for margin
    t = fig.text(
        0,
        0,
        "\n".join(str(label) + "x" for label in set_names),
        **text_kw,
    )

    # Get text width in display coordinates
    fig.canvas.draw()  # Ensure renderer is available
    textw = t.get_window_extent(renderer=fig.canvas.get_renderer()).width
    t.remove()

    # Get figure width in display coordinates
    figw = fig.get_window_extent(renderer=fig.canvas.get_renderer()).width

    # Calculate figure size based on elementsize
    # Key constraint: bars should have same width as matrix elements
    # For square matrix elements: element_width = element_height
    # So: matrix_width / n_intersections = matrix_height / n_sets
    # Therefore: bars need width = colw for both intersection and set bars

    # Number of non-text elements (set bars + intersection bars)
    non_text_nelems = n_sets + n_intersections

    # Calculate element width in display coordinates from elementsize (in points)
    render_ratio = figw / fig.get_figwidth() if fig.get_figwidth() > 0 else 72
    colw = elementsize / 72 * render_ratio

    # Calculate figure width to fit: text + non-text elements
    # Add +1 to text columns for margin (like UpSetPlot does)
    figw = colw * (non_text_nelems + np.ceil(textw / colw) + 1)
    fig.set_figwidth(figw / render_ratio)
    fig.set_figheight((colw * (n_sets + 2)) / render_ratio)

    # Remeasure after resize
    figw = fig.get_window_extent(renderer=fig.canvas.get_renderer()).width

    # Calculate grid columns for each section
    # Use UpSetPlot's approach: text_cols = total_cols - non_text_cols
    total_cols_exact = figw / colw
    text_cols = int(np.ceil(total_cols_exact - non_text_nelems))
    text_cols = max(1, text_cols)  # Ensure at least 1 column for text

    set_bar_cols = n_sets  # Set bars occupy n_sets columns
    matrix_cols = n_intersections  # Matrix occupies n_intersections columns

    # Create width_ratios to ensure all bar/matrix columns have equal width
    # All set bar and matrix columns get ratio 1.0
    # Text columns get ratio proportional to their share of space
    width_ratios = []

    # Set bar columns: each gets ratio 1.0
    width_ratios.extend([1.0] * set_bar_cols)

    # Text columns: collectively need textw/colw worth of space
    # Each text column gets equal share
    if text_cols > 0:
        text_ratio_per_col = (textw / colw) / text_cols
        width_ratios.extend([text_ratio_per_col] * text_cols)

    # Matrix columns: each gets ratio 1.0 (same as set bars)
    width_ratios.extend([1.0] * matrix_cols)

    # Create GridSpec with calculated proportions
    # Total columns: set_bar_cols + text_cols + matrix_cols
    # Layout: [set bars][labels][matrix/intersection bars]
    gs = gridspec.GridSpec(
        2,  # 2 rows: intersection bars (top) and matrix section (bottom)
        set_bar_cols + text_cols + matrix_cols,
        figure=fig,
        height_ratios=[2, n_sets],  # Intersection bars + matrix
        width_ratios=width_ratios,  # Ensure equal bar widths
        hspace=0,  # No space between rows - intersection bars should connect to matrix
        wspace=0,  # No space between columns
        left=0,  # Use full figure width
        right=1,
        bottom=0,
        top=1,
    )

    # Top row: intersection bars (over the rightmost matrix_cols columns)
    ax_intersections = fig.add_subplot(gs[0, set_bar_cols + text_cols:])

    # Bottom row left: set size bars (over the leftmost set_bar_cols columns)
    ax_sets = fig.add_subplot(gs[1, :set_bar_cols])

    # Bottom row right: matrix (over the rightmost matrix_cols columns)
    ax_matrix = fig.add_subplot(gs[1, set_bar_cols + text_cols:])

    # Calculate appropriate bar widths for equal visual appearance
    # When figsize is specified, elements may not be square, so we need
    # different bar widths for intersection and set bars to appear equal

    # Get actual figure and axes dimensions
    figw_display = fig.get_figwidth() * fig.dpi
    figh_display = fig.get_figheight() * fig.dpi

    # Get axes dimensions (normalized coordinates → display pixels)
    int_bbox = ax_intersections.get_position()
    set_bbox = ax_sets.get_position()

    int_width_px = int_bbox.width * figw_display
    int_height_px = int_bbox.height * figh_display
    set_width_px = set_bbox.width * figw_display
    set_height_px = set_bbox.height * figh_display

    # Calculate pixels per data unit for each axis
    # Intersection bars: n_intersections span from -0.5 to n_intersections-0.5
    int_data_range = n_intersections  # Total range: from -0.5 to n-0.5 is n units
    int_px_per_unit = int_width_px / int_data_range

    # Set bars: n_sets span from -0.5 to n_sets-0.5
    set_data_range = n_sets
    set_px_per_unit = set_height_px / set_data_range

    # Choose target visual width (use the smaller px_per_unit as reference)
    target_visual_width = BARWIDTH * min(int_px_per_unit, set_px_per_unit)

    # Calculate bar widths in data coordinates to achieve target visual width
    intersection_bar_width = target_visual_width / int_px_per_unit
    set_bar_width = target_visual_width / set_px_per_unit

    return ax_intersections, ax_matrix, ax_sets, intersection_bar_width, set_bar_width


def add_upset_labels(
    fig: plt.Figure,
    ax_intersections: Axes,
    ax_sets: Axes,
    title: str = "",
    intersection_label: str = "",
    set_label: str = "",
) -> None:
    """
    Add labels and title to UpSet plot.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure
    ax_intersections : Axes
        Intersection size axes
    ax_sets : Axes
        Set size axes
    title : str
        Main plot title
    intersection_label : str
        Label for intersection size axis
    set_label : str
        Label for set size axis
    """
    if title:
        ax_intersections.set_title(title)
    if intersection_label:
        ax_intersections.set_ylabel(
            intersection_label
        )

    if set_label:
        ax_sets.set_xlabel(set_label)
