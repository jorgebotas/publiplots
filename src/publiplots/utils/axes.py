"""
Axis manipulation utilities for publiplots.

This module provides functions for manipulating axes appearance,
including spines, grids, labels, and limits.
"""

from typing import Optional, List, Union, Tuple, Literal
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def adjust_spines(
    ax: Axes,
    spines: Union[str, List[str]] = 'left-bottom',
    color: str = '0.2',
    linewidth: float = 1.5,
    offset: Optional[float] = None
) -> None:
    """
    Adjust which spines are visible and their appearance.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify in place.
    spines : str or list of str, default ``'left-bottom'``
        Which spines to show. Accepted values:

        - ``'all'`` — show all four spines.
        - ``'none'`` — hide all spines.
        - ``'left-bottom'`` — show only left and bottom (publiplots default).
        - ``'box'`` — equivalent to ``'all'``.
        - A list of spine names (subset of
          ``['left', 'bottom', 'right', 'top']``).
    color : str, default ``'0.2'``
        Color of visible spines.
    linewidth : float, default ``1.5``
        Width of visible spines (points).
    offset : float, optional
        Offset visible spines outward from the data by this many points.
        ``None`` leaves the spine position alone.

    Returns
    -------
    None
        ``ax`` is modified in place.

    Examples
    --------
    Show only left and bottom spines (publication style):

    >>> pp.adjust_spines(ax, spines='left-bottom')

    Show all spines:

    >>> pp.adjust_spines(ax, spines='all')

    Hide all spines:

    >>> pp.adjust_spines(ax, spines='none')

    Custom spine selection with an outward offset:

    >>> pp.adjust_spines(ax, spines=['left', 'bottom'], offset=3)
    """
    # Parse spines parameter
    if spines == 'all':
        visible_spines = ['left', 'bottom', 'right', 'top']
    elif spines == 'none':
        visible_spines = []
    elif spines == 'left-bottom':
        visible_spines = ['left', 'bottom']
    elif spines == 'box':
        visible_spines = ['left', 'bottom', 'right', 'top']
    elif isinstance(spines, list):
        visible_spines = spines
    else:
        raise ValueError(
            f"Invalid spines parameter: {spines}. "
            "Use 'all', 'none', 'left-bottom', 'box', or a list of spine names."
        )

    # Set spine visibility and properties
    for spine_name in ['left', 'bottom', 'right', 'top']:
        spine = ax.spines[spine_name]
        if spine_name in visible_spines:
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(linewidth)
            if offset is not None:
                spine.set_position(('outward', offset))
        else:
            spine.set_visible(False)

    # Adjust tick visibility
    if 'bottom' not in visible_spines:
        ax.xaxis.set_ticks_position('none')
    if 'left' not in visible_spines:
        ax.yaxis.set_ticks_position('none')


def add_grid(
    ax: Axes,
    which: str = 'major',
    axis: str = 'both',
    alpha: float = 0.3,
    linestyle: str = '--',
    linewidth: float = 0.5,
    color: str = '0.8',
    zorder: int = 0
) -> None:
    """
    Add customizable gridlines to axes.

    Also sets ``ax.set_axisbelow(True)`` so the grid renders beneath
    data artists.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify in place.
    which : {'major', 'minor', 'both'}, default ``'major'``
        Which tick gridlines to show.
    axis : {'x', 'y', 'both'}, default ``'both'``
        Which axis to add gridlines to.
    alpha : float, default ``0.3``
        Gridline transparency in ``[0, 1]``.
    linestyle : str, default ``'--'``
        Matplotlib linestyle spec.
    linewidth : float, default ``0.5``
        Gridline width (points).
    color : str, default ``'0.8'``
        Gridline color.
    zorder : int, default ``0``
        Gridline z-order (lower values render behind other elements).

    Examples
    --------
    Add default gridlines:

    >>> pp.add_grid(ax)

    Add only horizontal gridlines:

    >>> pp.add_grid(ax, axis='y')

    Customize grid appearance:

    >>> pp.add_grid(ax, alpha=0.5, linestyle=':', color='steelblue')
    """
    ax.grid(
        True,
        which=which,
        axis=axis,
        alpha=alpha,
        linestyle=linestyle,
        linewidth=linewidth,
        color=color,
        zorder=zorder
    )
    ax.set_axisbelow(True)  # Ensure grid is behind data


def remove_grid(ax: Axes, axis: str = 'both') -> None:
    """
    Remove gridlines from axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    axis : str, default='both'
        Which axis to remove grid from: 'x', 'y', or 'both'.

    Examples
    --------
    >>> pp.remove_grid(ax)
    """
    ax.grid(False, axis=axis)


def set_axis_labels(
    ax: Axes,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    fontsize: Optional[float] = None,
    fontweight: str = 'normal'
) -> None:
    """
    Set axis labels and title with consistent formatting.

    Only labels that are explicitly provided are set; ``None`` values
    leave the existing label / title untouched.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify in place.
    xlabel : str, optional
        X-axis label. ``None`` leaves it unchanged.
    ylabel : str, optional
        Y-axis label. ``None`` leaves it unchanged.
    title : str, optional
        Plot title. ``None`` leaves it unchanged.
    fontsize : float, optional
        Font size (points) for labels and title. ``None`` uses the
        active rcParams.
    fontweight : str, default ``'normal'``
        Font weight — e.g. ``'normal'``, ``'bold'``, ``'light'``.

    Examples
    --------
    >>> pp.set_axis_labels(ax, xlabel='Time (s)', ylabel='Signal',
    ...                    title='Results')
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontweight=fontweight)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, fontweight=fontweight)
    if title is not None:
        ax.set_title(title, fontsize=fontsize, fontweight=fontweight)


def set_axis_limits(
    ax: Axes,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    expand: float = 0.0
) -> None:
    """
    Set axis limits with optional expansion.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    xlim : Tuple[float, float], optional
        X-axis limits as (min, max).
    ylim : Tuple[float, float], optional
        Y-axis limits as (min, max).
    expand : float, default=0.0
        Fraction to expand limits beyond specified range (0-1).
        E.g., 0.1 adds 10% padding on each side.

    Examples
    --------
    Set specific limits:
    >>> pp.set_axis_limits(ax, xlim=(0, 10), ylim=(0, 100))

    Set limits with padding:
    >>> pp.set_axis_limits(ax, xlim=(0, 10), expand=0.05)
    """
    if xlim is not None:
        x_min, x_max = xlim
        if expand > 0:
            x_range = x_max - x_min
            x_min -= x_range * expand
            x_max += x_range * expand
        ax.set_xlim(x_min, x_max)

    if ylim is not None:
        y_min, y_max = ylim
        if expand > 0:
            y_range = y_max - y_min
            y_min -= y_range * expand
            y_max += y_range * expand
        ax.set_ylim(y_min, y_max)


def rotate(
    ax: Axes,
    axis: Literal["x", "y"] = "x",
    rotation: float = 45,
    ha: Optional[Literal["left", "center", "right"]] = None,
    va: Optional[Literal["top", "center", "bottom", "baseline"]] = None
) -> None:
    """
    Rotate axis tick labels.

    Commonly used when category labels are long or numerous.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify in place.
    axis : {'x', 'y'}, default ``'x'``
        Which axis's tick labels to rotate.
    rotation : float, default ``45``
        Rotation angle in degrees, counter-clockwise.
    ha : {'left', 'center', 'right'}, optional
        Horizontal alignment of tick labels. ``None`` leaves the
        existing alignment.
    va : {'top', 'center', 'bottom', 'baseline'}, optional
        Vertical alignment of tick labels. ``None`` leaves the existing
        alignment.

    Raises
    ------
    AssertionError
        If ``axis`` is not ``'x'`` or ``'y'``.

    Examples
    --------
    Rotate x-tick labels 45 degrees:

    >>> pp.rotate(ax, axis='x', rotation=45)

    Rotate y-tick labels 90 degrees with right-alignment:

    >>> pp.rotate(ax, axis='y', rotation=90, ha='right')
    """
    assert axis in ["x", "y"], ValueError(f"Invalid axis: {axis}. Use 'x' or 'y'.")
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()

    for lbl in labels:
        lbl.set_rotation(rotation)
        if ha is not None: lbl.set_ha(ha)
        if va is not None: lbl.set_va(va)


def invert_axis(ax: Axes, axis: str = 'y') -> None:
    """
    Invert an axis direction.

    Useful for heatmaps or other visualisations where reversed order is
    more intuitive (e.g., placing the origin at the top-left).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify in place.
    axis : {'x', 'y'}, default ``'y'``
        Which axis to invert.

    Raises
    ------
    ValueError
        If ``axis`` is not ``'x'`` or ``'y'``.

    Examples
    --------
    Invert y-axis (common for heatmaps):

    >>> pp.invert_axis(ax, axis='y')

    Invert x-axis:

    >>> pp.invert_axis(ax, axis='x')
    """
    if axis == 'y':
        ax.invert_yaxis()
    elif axis == 'x':
        ax.invert_xaxis()
    else:
        raise ValueError(f"Invalid axis '{axis}'. Use 'x' or 'y'.")


def add_reference_line(
    ax: Axes,
    value: float,
    axis: str = 'y',
    color: str = 'red',
    linestyle: str = '--',
    linewidth: float = 1.5,
    alpha: float = 0.7,
    label: Optional[str] = None,
    zorder: int = 1
) -> None:
    """
    Add a reference line to the plot.

    Useful for showing thresholds, means, or other reference values.
    Internally calls :meth:`~matplotlib.axes.Axes.axhline` (for
    ``axis='y'``) or :meth:`~matplotlib.axes.Axes.axvline` (for
    ``axis='x'``).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify in place.
    value : float
        Position of the reference line in data coordinates.
    axis : {'x', 'y'}, default ``'y'``
        Which axis to add the line to: ``'x'`` draws a vertical line at
        ``x=value``; ``'y'`` draws a horizontal line at ``y=value``.
    color : str, default ``'red'``
        Line color.
    linestyle : str, default ``'--'``
        Matplotlib linestyle spec.
    linewidth : float, default ``1.5``
        Line width (points).
    alpha : float, default ``0.7``
        Line transparency in ``[0, 1]``.
    label : str, optional
        Label for the legend. ``None`` leaves the line unlabelled.
    zorder : int, default ``1``
        Line z-order (higher values render on top).

    Raises
    ------
    ValueError
        If ``axis`` is not ``'x'`` or ``'y'``.

    Examples
    --------
    Add horizontal reference line at ``y=0``:

    >>> pp.add_reference_line(ax, value=0, axis='y', label='Baseline')

    Add vertical line at ``x=5``:

    >>> pp.add_reference_line(ax, value=5, axis='x', color='steelblue')
    """
    if axis == 'y':
        ax.axhline(
            y=value,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder
        )
    elif axis == 'x':
        ax.axvline(
            x=value,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder
        )
    else:
        raise ValueError(f"Invalid axis '{axis}'. Use 'x' or 'y'.")


def set_aspect_equal(ax: Axes) -> None:
    """
    Set equal aspect ratio for the axes.

    Forces x and y axes to have the same scale, useful for scatter plots
    where true distances matter.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.

    Examples
    --------
    >>> pp.set_aspect_equal(ax)
    """
    ax.set_aspect('equal', adjustable='box')


def tighten_layout(fig: Optional[plt.Figure] = None) -> None:
    """
    Apply tight layout to figure.

    Automatically adjusts subplot parameters to give specified padding.

    Parameters
    ----------
    fig : Figure, optional
        Matplotlib figure object. If None, uses current figure.

    Examples
    --------
    >>> pp.tighten_layout(fig)
    >>> pp.tighten_layout()  # Uses current figure
    """
    if fig is None:
        plt.tight_layout()
    else:
        fig.tight_layout()
