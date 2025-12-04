"""
Strip plot functions for publiplots.

This module provides publication-ready strip plot visualizations with
transparent fill and opaque edges.
"""

from typing import Optional, List, Dict, Tuple, Union

from publiplots.themes.rcparams import resolve_param
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
import pandas as pd

from publiplots.themes.colors import resolve_palette_map
from publiplots.utils.transparency import ArtistTracker
from publiplots.utils.legend import create_legend_handles, legend


def stripplot(
    data: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    order: Optional[List] = None,
    hue_order: Optional[List] = None,
    jitter: Union[bool, float] = True,
    dodge: bool = True,
    orient: Optional[str] = None,
    color: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    size: float = 5,
    edgecolor: Optional[str] = None,
    linewidth: Optional[float] = None,
    hue_norm: Optional[Union[Tuple[float, float], Normalize]] = None,
    alpha: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: bool = True,
    legend_kws: Optional[Dict] = None,
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """
    Create a publication-ready strip plot.

    This function creates strip plots with transparent fill and opaque edges,
    following the publiplots visual style.

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
    jitter : float or bool, default=True
        Amount of jitter (only along the categorical axis) to apply.
    dodge : bool, default=True
        Whether to separate points by hue along the categorical axis.
    orient : str, optional
        Orientation of the plot ('v' or 'h').
    color : str, optional
        Fixed color for all points (only used when hue is None).
    palette : str, dict, or list, optional
        Color palette for hue grouping.
    size : float, default=5
        Size of the markers.
    edgecolor : str, optional
        Color for marker edges. If None, uses face color.
    linewidth : float, optional
        Width of marker edges.
    hue_norm : tuple or Normalize, optional
        Normalization for continuous hue variable.
    alpha : float, optional
        Transparency of marker fill (0-1).
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
    legend : bool, default=True
        Whether to show the legend.
    legend_kws : dict, optional
        Additional keyword arguments for legend.
    **kwargs
        Additional keyword arguments passed to seaborn.stripplot.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Matplotlib axes object.

    Examples
    --------
    Simple strip plot:

    >>> import publiplots as pp
    >>> fig, ax = pp.stripplot(data=df, x="category", y="value")

    Strip plot with hue grouping:

    >>> fig, ax = pp.stripplot(
    ...     data=df, x="category", y="value", hue="group"
    ... )
    """
    # Read defaults from rcParams if not provided
    figsize = resolve_param("figure.figsize", figsize)
    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    color = resolve_param("color", color)

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Resolve palette
    if hue is not None:
        palette = resolve_palette_map(
            values=data[hue].unique(),
            palette=palette,
        )

    # Prepare kwargs for seaborn stripplot
    stripplot_kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "order": order,
        "hue_order": hue_order,
        "jitter": jitter,
        "dodge": dodge,
        "orient": orient,
        "color": color if hue is None else None,
        "palette": palette if hue else None,
        "size": size,
        "edgecolor": edgecolor,
        "linewidth": linewidth,
        "hue_norm": hue_norm,
        "ax": ax,
        "legend": False,  # Handle legend ourselves
    }

    # Merge with user-provided kwargs
    stripplot_kwargs.update(kwargs)

    # Track artists before plotting
    tracker = ArtistTracker(ax)

    # Create stripplot
    sns.stripplot(**stripplot_kwargs)

    # Set edge colors if not specified
    if edgecolor is None:
        for collection in tracker.get_new_collections():
            collection.set_edgecolors(collection.get_facecolors())

    # Apply transparency to new collections only
    tracker.apply_transparency(on="collections", face_alpha=alpha)

    # Add legend if hue is used
    if legend and hue is not None:
        _legend(
            ax=ax,
            hue=hue,
            color=color,
            palette=palette,
            hue_norm=hue_norm,
            alpha=alpha,
            linewidth=linewidth,
            kwargs=legend_kws,
        )

    # Set labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return fig, ax


def _legend(
    ax: Axes,
    hue: Optional[str],
    color: Optional[str],
    palette: Optional[Union[str, Dict, List]],
    hue_norm: Optional[Normalize],
    alpha: Optional[float] = None,
    linewidth: Optional[float] = None,
    kwargs: Optional[Dict] = None,
) -> None:
    """
    Create legend handles for strip plot.
    """
    # Read defaults from rcParams if not provided
    alpha = resolve_param("alpha", alpha)
    linewidth = resolve_param("lines.linewidth", linewidth)

    kwargs = kwargs or {}
    handle_kwargs = dict(alpha=alpha, linewidth=linewidth, color=color, style="circle")

    # Store legend data in collection for later retrieval
    legend_data = {}

    # Prepare hue legend data
    if hue is not None:
        hue_label = kwargs.pop("hue_label", hue)
        if isinstance(palette, dict):  # categorical legend
            hue_handles = create_legend_handles(
                labels=list(palette.keys()),
                colors=list(palette.values()),
                **handle_kwargs
            )
            legend_data["hue"] = {
                "handles": hue_handles,
                "label": hue_label,
            }
        else:
            # Continuous colorbar
            mappable = ScalarMappable(norm=hue_norm, cmap=palette)
            legend_data["hue"] = {
                "mappable": mappable,
                "label": hue_label,
                "height": kwargs.pop("hue_height", 0.2),
                "width": kwargs.pop("hue_width", 0.05),
                "type": "colorbar",
            }

    # Store metadata on collection
    if len(ax.collections) > 0:
        ax.collections[0]._legend_data = legend_data

    # Create legends using legend() API
    from publiplots.utils.legend import legend as pp_legend
    builder = pp_legend(ax=ax)
