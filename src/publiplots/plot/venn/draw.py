"""
Core drawing functions for Venn diagrams.

This module provides the fundamental drawing primitives for creating Venn diagrams,
including functions to draw ellipses and text labels on matplotlib axes.
"""

from matplotlib.patches import Ellipse, Polygon
from matplotlib.colors import to_rgba
from matplotlib.axes import Axes
from typing import Tuple, Optional, Union

def draw_ellipse(
        ax: Axes,
        x: float,
        y: float,
        w: float,
        h: float,
        a: float,
        color: Union[str, Tuple[float, ...]],
        alpha: float,
    ) -> None:
    """
    Draw an ellipse on the given matplotlib axes.

    This function creates an ellipse patch with specified position, dimensions,
    rotation angle, and color. The ellipse is filled with the given color and
    outlined with a less transparent version of the same color.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the ellipse
    x : float
        X-coordinate of the ellipse center (data coordinates)
    y : float
        Y-coordinate of the ellipse center (data coordinates)
    w : float
        Width of the ellipse (data coordinates)
    h : float
        Height of the ellipse (data coordinates)
    a : float
        Rotation angle of the ellipse in degrees (counterclockwise from horizontal)
    color : color-like
        Fill color for the ellipse (any matplotlib-compatible color specification)
    alpha : float, default=1.0
        Alpha transparency value for the ellipse fill color
    Returns
    -------
    None
        Modifies the axes in-place by adding the ellipse patch

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> draw_ellipse(ax, 0, 0, 1.5, 1.5, 45, 'blue', 0.3)
    """
    ax.add_patch(
        Ellipse(
            xy=(x, y),
            width=w,
            height=h,
            angle=a,
            facecolor=to_rgba(color, alpha=alpha),
            edgecolor=color
        )
    )


def draw_text(
        ax: Axes,
        x: float,
        y: float,
        text: str,
        fontsize: int,
        color: Optional[Union[str, Tuple[float, ...]]] = None,
        ha: str = "center",
        va: str = "center",
    ) -> None:
    """
    Draw text at a specified position on the axes with configurable alignment.

    This function places text at the given coordinates. By default, it uses center
    alignment both horizontally and vertically, but alignment can be customized for
    optimal label positioning relative to shapes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the text
    x : float
        X-coordinate for text position (data coordinates)
    y : float
        Y-coordinate for text position (data coordinates)
    text : str
        The text string to display
    fontsize : int
        Font size in points
    color : str or tuple of floats (RGBA), optional
        Text color (any matplotlib-compatible color specification)
    ha : str, default='center'
        Horizontal alignment: 'left', 'center', or 'right'
    va : str, default='center'
        Vertical alignment: 'top', 'center', or 'bottom'

    Returns
    -------
    None
        Modifies the axes in-place by adding the text

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> draw_text(ax, 0, 0, '42', fontsize=12, color='black')
    >>> draw_text(ax, 0, 1.5, 'Label', fontsize=14, va='bottom')
    """
    ax.text(
        x, y, text,
        fontsize=fontsize,
        color=color,
        horizontalalignment=ha,
        verticalalignment=va
    )


def init_axes(
    ax: Optional[Axes],
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> Axes:
    """
    Initialize or configure axes for Venn diagram plotting.

    Creates new axes via ``pp.subplots`` if none are provided, or configures
    existing axes with the appropriate settings for Venn diagrams. Sets up an
    equal aspect ratio, removes frame and ticks, and establishes coordinate
    limits based on the geometry.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        Existing axes to configure, or None to create new axes via pp.subplots.
    xlim : tuple of float, optional
        X-axis limits. If None, uses automatic limits.
    ylim : tuple of float, optional
        Y-axis limits. If None, uses automatic limits.

    Returns
    -------
    matplotlib.axes.Axes
        Configured axes ready for Venn diagram plotting.
    """
    if ax is None:
        from publiplots.layout.subplots import subplots as _pp_subplots
        _, ax = _pp_subplots(nrows=1, ncols=1)

    # Set basic properties
    ax.set_aspect("equal")
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Set limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    return ax
