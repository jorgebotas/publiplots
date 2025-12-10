"""
Tick labels for complex heatmap margins.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from typing import Optional, List, Dict, Tuple

from publiplots.themes.rcparams import resolve_param
from publiplots.utils.text import calculate_label_space

# Conversion constants
MM2INCH = 1 / 25.4


def calculate_ticklabel_space(
    labels: List[str],
    position: str = 'left',
    rotation: float = 0,
    fontsize: Optional[float] = None,
    fig: Optional[plt.Figure] = None,
    **text_kws
) -> float:
    """
    Calculate space needed for tick labels in millimeters.

    Takes rotation into account when computing the required space.

    Parameters
    ----------
    labels : list of str
        The tick labels to measure.
    position : str, default='left'
        Position where labels will be placed: 'left', 'right', 'top', 'bottom'.
    rotation : float, default=0
        Text rotation angle in degrees.
    fontsize : float, optional
        Font size in points. Uses rcParams default if None.
    fig : Figure, optional
        Figure to use for measurement. If provided, ensures consistent DPI.
    **text_kws
        Additional text properties (color, weight, etc.)

    Returns
    -------
    float
        Required space in millimeters.
    """
    if not labels:
        return 0.0

    # Resolve fontsize
    if position in ('left', 'right'):
        fontsize = resolve_param("ytick.labelsize", fontsize)
    else:
        fontsize = resolve_param("xtick.labelsize", fontsize)

    # Use calculate_label_space with rotation support
    space_inches = calculate_label_space(
        labels=labels,
        fig=fig,
        position=position,
        rotation=rotation,
        fontsize=fontsize,
        tick_pad=0,  # No tick padding - we draw in dedicated axes
        unit="inches",
        safety_factor=1.0,
        **text_kws
    )

    # Convert to millimeters
    space_mm = space_inches / MM2INCH
    return space_mm


def ticklabels(
    labels: Optional[List[str]] = None,
    orientation: str = "vertical",
    position: str = "right",
    rotation: float = 0,
    fontsize: Optional[float] = None,
    color: Optional[str] = None,
    va: Optional[str] = None,
    ha: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """
    Draw tick labels in a dedicated axes for complex heatmap margins.

    This function is designed to be used with complex_heatmap's margin system,
    allowing labels to be added as regular margin plots.

    Parameters
    ----------
    labels : list of str, optional
        Labels to draw. If None, must be provided by the complex_heatmap builder.
    orientation : str, default="vertical"
        Label orientation: 'vertical' (for left/right) or 'horizontal' (for top/bottom).
    position : str, default="right"
        Position where labels are drawn: 'left', 'right', 'top', 'bottom'.
        Used to determine text alignment defaults.
    rotation : float, default=0
        Text rotation angle in degrees.
    fontsize : float, optional
        Font size in points. Uses rcParams default if None.
    color : str, optional
        Text color. Uses rcParams default if None.
    va : str, optional
        Vertical alignment. Auto-determined based on position if None.
    ha : str, optional
        Horizontal alignment. Auto-determined based on position if None.
    ax : Axes, optional
        Matplotlib axes. If None, creates new figure and axes.
    **kwargs
        Additional arguments passed to ax.text().

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if labels is None:
        raise ValueError("labels must be provided")

    # Turn off all axes decorations
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Resolve defaults
    if orientation == 'vertical':
        fontsize = resolve_param("ytick.labelsize", fontsize)
    else:
        fontsize = resolve_param("xtick.labelsize", fontsize)

    # Set default alignments based on position
    if va is None:
        if position in ('left', 'right'):
            va = 'center'
        elif position == 'bottom':
            va = 'top'
        else:  # top
            va = 'bottom'

    if ha is None:
        if position == 'right':
            ha = 'left'
        elif position == 'left':
            ha = 'right'
        else:  # top, bottom
            ha = 'center'

    # Build text kwargs
    text_kws = {
        'fontsize': fontsize,
        'rotation': rotation,
        'va': va,
        'ha': ha,
        **kwargs
    }
    if color is not None:
        text_kws['color'] = color

    n_labels = len(labels)

    if position in ['left', 'right']:
        # Vertical labels for y-axis
        # Note: xlim/ylim are set by ComplexHeatmapBuilder after all margin plots
        # to ensure consistent coordinate system across all aligned plots
        for i, label in enumerate(labels):
            y_pos = i + 0.5
            x_pos = 0 if ha == 'left' else 1
            ax.text(x_pos, y_pos, label,
                   transform=ax.get_yaxis_transform(),
                   **text_kws)

    else:  # top or bottom
        # Horizontal labels for x-axis
        # Note: xlim/ylim are set by ComplexHeatmapBuilder after all margin plots
        # to ensure consistent coordinate system across all aligned plots
        for i, label in enumerate(labels):
            x_pos = i + 0.5
            y_pos = 1 if va == 'top' else 0
            ax.text(x_pos, y_pos, label,
                   transform=ax.get_xaxis_transform(),
                   **text_kws)

    return fig, ax
