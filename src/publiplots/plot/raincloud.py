"""
Rain plot (raincloud plot) functions for publiplots.

This module provides publication-ready raincloud plot visualizations that
combine half-violin plots, box plots, and strip/swarm plots.
"""

from typing import Optional, List, Dict, Tuple, Union, Literal

from publiplots.themes.rcparams import resolve_param
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd

from publiplots.themes.colors import resolve_palette_map
from publiplots.utils.transparency import ArtistTracker
from publiplots.utils.validation import is_categorical
from publiplots.plot.violin import violinplot
from publiplots.plot.box import boxplot
from publiplots.plot.swarm import swarmplot
from publiplots.plot.strip import stripplot
from publiplots.utils.offset import offset_lines, offset_patches, offset_collections

def raincloudplot(
    data: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    order: Optional[List] = None,
    hue_order: Optional[List] = None,
    color: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    saturation: float = 1.0,
    dodge: bool = True,
    gap: float = 0.3,
    # Violin (cloud) parameters
    cloud_side: str = "right",
    cloud_alpha: Optional[float] = None,
    cut: float = 2,
    gridsize: int = 100,
    bw_method: str = "scott",
    bw_adjust: float = 1,
    density_norm: str = "area",
    # Box parameters
    box: bool = True,
    box_width: float = 0.15,
    box_kws: Optional[Dict] = None,
    # Points (rain) parameters
    rain: Literal["strip", "swarm"] = "strip",
    rain_kws: Optional[Dict] = dict(alpha=0.5, linewidth=0),
    # Offset control
    box_offset: float = 0.2,
    rain_offset: float = 0.2,
    # General styling
    edgecolor: Optional[str] = None,
    linewidth: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
) -> Tuple[plt.Figure, Axes]:
    """
    Create a publication-ready raincloud plot.

    A raincloud plot combines a half-violin plot (cloud), a box plot (umbrella),
    and a strip/swarm plot (rain) to show both the distribution and individual
    data points.

    Parameters
    ----------
    data : DataFrame
        Input data.
    x : str, optional
        Column name for x-axis variable.
    y : str, optional
        Column name for y-axis variable.
    hue : str, optional
        Column name for color grouping.
    order : list, optional
        Order for the categorical levels.
    hue_order : list, optional
        Order for the hue levels.
    color : str, optional
        Fixed color for all elements (only used when hue is None).
    palette : str, dict, or list, optional
        Color palette for hue grouping.
    saturation : float, default=1.0
        Proportion of the original saturation to draw colors at.
    dodge : bool, default=True
        Whether to dodge the violin and box plot.
    gap : float, default=0.3
        Gap between the violin and the box plot.
    cloud_side : str, default="right"
        Side of the cloud plot ("left" or "right").
    cloud_alpha : float, optional
        Transparency of violin fill (0-1). Defaults to rcParams alpha.
    cut : float, default=2
        Distance past extreme data points to extend density estimate.
    gridsize : int, default=100
        Number of points in the discrete grid used to evaluate KDE.
    bw_method : str, default="scott"
        Method for calculating smoothing bandwidth.
    bw_adjust : float, default=1
        Factor to adjust the bandwidth.
    density_norm : str, default="area"
        Method for normalizing density ("area", "count", "width").
    box : bool, default=True
        Whether to show the box plot.
    box_width : float, default=0.15
        Width of the box plot.
    box_kws : dict, optional
        Additional keyword arguments for box plot.
    rain : Literal["strip", "swarm"], default="strip"
        Type of rain plot: "strip" or "swarm".
    rain_kws : dict, optional
        Additional keyword arguments for rain plot.
    box_offset : float, default=0.0
        Offset for the box plot from center position.
    rain_offset : float, default=-0.15
        Offset for rain points from center position. Negative values move
        points to the left (for vertical) or down (for horizontal).
    linewidth : float, optional
        Width of edges.
    figsize : tuple, optional
        Figure size (width, height) if creating new figure.
    ax : Axes, optional
        Matplotlib axes object. If None, creates new figure.
    title : str, default=""
        Plot title.
    xlabel : str, default=""
        X-axis label.
    ylabel : str, default=""
        Y-axis label.
    legend : bool or dict, default=True
        Whether to show the legend. Accepts the same ``bool | dict[kind, bool]``
        form as the underlying violin plot; the value is forwarded to the
        inner ``violinplot`` call, which owns the legend for the raincloud.
    legend_kws : dict, optional
        Additional keyword arguments for legend.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Matplotlib axes object.

    Examples
    --------
    Simple raincloud plot:

    >>> import publiplots as pp
    >>> fig, ax = pp.raincloudplot(data=df, x="category", y="value")

    Raincloud plot with hue grouping:

    >>> fig, ax = pp.raincloudplot(
    ...     data=df, x="category", y="value", hue="group"
    ... )
    """
    # Read defaults from rcParams if not provided.
    # Deliberately do NOT fall back on rcParams["figure.figsize"] for the
    # figsize argument: when figsize is None, the inner violinplot call takes
    # the pp.subplots() path and SubplotsAutoLayout grows the figure to fit
    # the legend. Forcing a default figsize here would route us through
    # plt.subplots(figsize=...), skipping auto-layout and cropping the legend.
    linewidth = resolve_param("lines.linewidth", linewidth)
    cloud_alpha = resolve_param("alpha", cloud_alpha)
    color = resolve_param("color", color)
    edgecolor = resolve_param("edgecolor", edgecolor)

    # Resolve palette
    if hue is not None:
        palette = resolve_palette_map(
            values=data[hue].unique(),
            palette=palette,
        )

    # Determine orientation
    orientation = "vertical" if is_categorical(data[x]) else "horizontal"

    # 1. Draw the half-violin (cloud) using pp.violinplot with side parameter
    fig, ax = violinplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        color=color,
        palette=palette,
        saturation=saturation,
        inner=None,
        dodge=dodge,
        linewidth=linewidth,
        edgecolor=edgecolor,
        cut=cut,
        gap=gap,
        gridsize=gridsize,
        bw_method=bw_method,
        bw_adjust=bw_adjust,
        density_norm=density_norm,
        alpha=cloud_alpha,
        legend=legend,
        legend_kws=legend_kws,
        side=cloud_side,
        figsize=figsize,
        ax=ax,
    )

    # 2. Draw box plot (umbrella) if requested
    if box:
        box_kws = box_kws or {}
        box_kws.update(dict(
            data=data,
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=hue_order,
            color=color,
            palette=palette,
            edgecolor=edgecolor,
            ax=ax,
            legend=False,
            dodge=dodge,
            gap=(1 - box_width),
            fliersize=0
        ))
        box_tracker = ArtistTracker(ax)
        boxplot(**box_kws)

        # Apply box offset
        if box_offset != 0:
            box_offset = box_offset if cloud_side == "left" else -box_offset
            # Offset patches (box bodies)
            offset_patches(
                patches=box_tracker.get_new_patches(),
                offset=box_offset,
                orientation=orientation,
            )
            # Offset lines (whiskers, medians, caps)
            offset_lines(
                lines=box_tracker.get_new_lines(),
                offset=box_offset,
                orientation=orientation,
            )

    # 3. Draw rain (strip or swarm plot)
    if rain is not None:
        rain_kws = rain_kws or {}
        rain_kws.update(dict(
            data=data,
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=hue_order,
            color=color,
            palette=palette,
            edgecolor=edgecolor,
            ax=ax,
            legend=False,
            dodge=dodge,
            native_scale=True,
        ))
        rain_tracker = ArtistTracker(ax)
        rain_fn = swarmplot if rain == "swarm" else stripplot
        rain_fn(**rain_kws)

        # Apply rain offset
        if rain_offset != 0:
            rain_offset = rain_offset if cloud_side == "left" else -rain_offset
            offset_collections(
                collections=rain_tracker.get_new_collections(),
                offset=rain_offset,
                orientation=orientation,
                ax=ax,
            )

    # Set labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    return fig, ax