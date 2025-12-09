"""
Text measurement utilities for publiplots.

This module provides functions for measuring text dimensions,
useful for calculating space needed for labels in complex layouts.
"""

from typing import List, Optional, Tuple, Union, Literal
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text

from publiplots.themes.rcparams import resolve_param


def measure_text_dimensions(
    labels: List[str],
    fig: Optional[Figure] = None,
    fontsize: Optional[float] = None,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    rotation: float = 0,
    unit: Literal["pixels", "inches", "points"] = "pixels",
) -> Tuple[float, float]:
    """
    Measure the actual rendered dimensions needed for a list of text labels.

    This uses matplotlib's text rendering to get accurate measurements,
    accounting for font metrics, character widths, and any text transformations.

    Parameters
    ----------
    labels : list of str
        The text labels to measure.
    fig : Figure, optional
        Matplotlib figure to use for measurement. If None, creates a temporary
        figure which is cleaned up after measurement.
    fontsize : float, optional
        Font size in points. If None, uses rcParams default.
    orientation : {'horizontal', 'vertical'}, default='horizontal'
        How the labels are arranged:
        - 'horizontal': Labels laid out horizontally (e.g., x-axis)
        - 'vertical': Labels stacked vertically (e.g., y-axis)
    rotation : float, default=0
        Rotation angle in degrees for the text.
    unit : {'pixels', 'inches', 'points'}, default='pixels'
        Unit for the returned dimensions.

    Returns
    -------
    width : float
        Width of the bounding box in specified units.
    height : float
        Height of the bounding box in specified units.

    Examples
    --------
    Measure space needed for y-axis labels:

    >>> labels = ['Gene_1', 'Gene_2', 'Gene_3']
    >>> width, height = measure_text_dimensions(labels, orientation='vertical')

    Measure with specific font size:

    >>> width, height = measure_text_dimensions(labels, fontsize=12, unit='inches')
    """
    if not labels:
        return 0.0, 0.0

    # Determine if we need to create a temporary figure
    created_fig = False
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
        created_fig = True

    try:
        # Get font properties
        if fontsize is None:
            fontsize = resolve_param("font.size")

        # Create text string based on orientation
        if orientation == "vertical":
            # Stack labels vertically with newlines for height measurement
            # Add "x" margin character like upsetplot does
            text_str = "\n".join(str(label) for label in labels)
        else:
            # Join with spaces for horizontal layout
            # The maximum width label determines the space needed
            text_str = max((str(label) for label in labels), key=len)

        # Create temporary text element
        text_obj = fig.text(
            0, 0,
            text_str,
            fontsize=fontsize,
            rotation=rotation,
        )

        # Ensure renderer is available
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Get text bounding box in display (pixel) coordinates
        bbox = text_obj.get_window_extent(renderer=renderer)

        # Remove the temporary text
        text_obj.remove()

        # Convert to requested units
        width_px = bbox.width
        height_px = bbox.height

        if unit == "pixels":
            return width_px, height_px
        elif unit == "inches":
            dpi = fig.dpi
            return width_px / dpi, height_px / dpi
        elif unit == "points":
            dpi = fig.dpi
            return width_px / dpi * 72, height_px / dpi * 72
        else:
            raise ValueError(f"Invalid unit '{unit}'. Use 'pixels', 'inches', or 'points'.")

    finally:
        if created_fig:
            plt.close(fig)


def calculate_label_space(
    labels: List[str],
    fig: Optional[Figure] = None,
    position: Literal["left", "right", "top", "bottom"] = "left",
    fontsize: Optional[float] = None,
    tick_pad: Optional[float] = None,
    rotation: float = 0,
    unit: Literal["pixels", "inches", "points"] = "inches",
    safety_factor: float = 1.1,
) -> float:
    """
    Calculate the space needed for axis tick labels.

    This is a higher-level function that accounts for tick padding and
    applies a safety factor for reliable layout calculations.

    Parameters
    ----------
    labels : list of str
        The tick labels to measure.
    fig : Figure, optional
        Matplotlib figure for measurement. If None, creates temporary figure.
    position : {'left', 'right', 'top', 'bottom'}, default='left'
        Where the labels will be placed:
        - 'left', 'right': Vertical labels (y-axis style)
        - 'top', 'bottom': Horizontal labels (x-axis style)
    fontsize : float, optional
        Font size in points. If None, uses appropriate rcParams default.
    tick_pad : float, optional
        Padding between ticks and labels in points. If None, uses rcParams.
    rotation : float, default=0
        Rotation angle for labels in degrees.
    unit : {'pixels', 'inches', 'points'}, default='inches'
        Unit for the returned space.
    safety_factor : float, default=1.1
        Multiplier for extra margin (1.1 = 10% extra).

    Returns
    -------
    space : float
        Required space for labels in specified units.

    Examples
    --------
    Calculate space for y-axis labels on a heatmap:

    >>> row_labels = ['Gene_' + str(i) for i in range(20)]
    >>> ylabel_space = calculate_label_space(row_labels, position='right')

    Calculate space for rotated x-axis labels:

    >>> col_labels = ['Sample_' + str(i) for i in range(10)]
    >>> xlabel_space = calculate_label_space(
    ...     col_labels, position='bottom', rotation=45
    ... )
    """
    if not labels:
        return 0.0

    # Determine orientation and get appropriate rcParams
    if position in ("left", "right"):
        orientation = "vertical"
        if fontsize is None:
            fontsize = resolve_param("ytick.labelsize")
        if tick_pad is None:
            tick_pad = resolve_param("ytick.major.pad")
    else:  # top, bottom
        orientation = "horizontal"
        if fontsize is None:
            fontsize = resolve_param("xtick.labelsize")
        if tick_pad is None:
            tick_pad = resolve_param("xtick.major.pad")

    # Measure text dimensions
    width, height = measure_text_dimensions(
        labels=labels,
        fig=fig,
        fontsize=fontsize,
        orientation=orientation,
        rotation=rotation,
        unit="points",  # Work in points for tick_pad addition
    )

    # The width and height returned are already the dimensions of the
    # rotated bounding box (rotation is applied in measure_text_dimensions).
    # For vertical labels (left/right), we need the width
    # For horizontal labels (top/bottom), we need the height
    if position in ("left", "right"):
        space = width
    else:  # top, bottom
        space = height

    # Add tick padding
    space += tick_pad

    # Apply safety factor
    space *= safety_factor

    # Convert to requested unit
    if unit == "points":
        return space
    elif unit == "inches":
        return space / 72.0
    elif unit == "pixels":
        # Need figure dpi for this conversion
        dpi = 72  # Default if no figure
        if fig is not None:
            dpi = fig.dpi
        return space * dpi / 72.0
    else:
        raise ValueError(f"Invalid unit '{unit}'. Use 'pixels', 'inches', or 'points'.")


def get_text_width_in_data_coords(
    labels: List[str],
    ax,
    fontsize: Optional[float] = None,
) -> float:
    """
    Get text width converted to data coordinates for an axes.

    Useful when you need to know how much data-space a label will occupy.

    Parameters
    ----------
    labels : list of str
        The text labels to measure.
    ax : Axes
        Matplotlib axes for coordinate transformation.
    fontsize : float, optional
        Font size in points. If None, uses rcParams.

    Returns
    -------
    width : float
        Width in data coordinates.
    """
    if not labels:
        return 0.0

    fig = ax.figure

    # Measure in pixels
    width_px, _ = measure_text_dimensions(
        labels=labels,
        fig=fig,
        fontsize=fontsize,
        orientation="vertical",
        unit="pixels",
    )

    # Convert pixels to data coordinates
    # Get the axes transform
    inv_transform = ax.transData.inverted()
    # Transform from display to data coordinates
    p0 = inv_transform.transform((0, 0))
    p1 = inv_transform.transform((width_px, 0))

    return abs(p1[0] - p0[0])
