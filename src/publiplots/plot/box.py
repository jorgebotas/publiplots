"""
Box plot functions for publiplots.

This module provides publication-ready box plot visualizations with
transparent fill and opaque edges.
"""

import warnings
from typing import Optional, List, Dict, Tuple, Union

from publiplots.themes.rcparams import resolve_param
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
import seaborn as sns
import pandas as pd
import numpy as np

from publiplots.themes.colors import resolve_palette_map
from publiplots.utils.transparency import ArtistTracker
from publiplots.utils import is_categorical
from publiplots.utils.plot_legend import stash_hue_legend


def boxplot(
    data: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    order: Optional[List] = None,
    hue_order: Optional[List] = None,
    orient: Optional[str] = None,
    color: Optional[str] = None,
    linecolor: Optional[str] = None,
    edgecolor: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    width: float = 0.8,
    gap: float = 0,
    whis: float = 1.5,
    showcaps: bool = False,
    fliersize: Optional[float] = None,
    linewidth: Optional[float] = None,
    alpha: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """
    Create a publication-ready box plot.

    This function creates box plots with transparent fill and opaque edges,
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
    orient : str, optional
        Orientation of the plot ('v' or 'h').
    color : str, optional
        Fixed color for all boxes (only used when hue is None).
    linecolor : str, optional
        Deprecated. Use edgecolor instead. Color of the box edges.
    edgecolor : str, optional
        Color of the box edges, whiskers, caps, and outlier marker edges.
        When None, edges match the palette face color for each group.
    palette : str, dict, or list, optional
        Color palette for hue grouping.
    width : float, default=0.8
        Width of the boxes.
    gap : float, default=0
        Gap between boxes when using hue.
    whis : float, default=1.5
        Proportion of IQR past low and high quartiles to extend whiskers.
    showcaps: bool, default=False
        Whether to show the caps.
    fliersize : float, optional
        Size of outlier markers.
    linewidth : float, optional
        Width of box edges.
    alpha : float, optional
        Transparency of box fill (0-1).
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
        Whether to show the legend. Accepts ``bool`` or ``dict[kind, bool]``
        for per-kind control (e.g., ``legend={"hue": False}``).
    legend_kws : dict, optional
        Additional keyword arguments for legend.
    **kwargs
        Additional keyword arguments passed to seaborn.boxplot.

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Matplotlib axes object.

    Examples
    --------
    Simple box plot:

    >>> import publiplots as pp
    >>> fig, ax = pp.boxplot(data=df, x="category", y="value")

    Box plot with hue grouping:

    >>> fig, ax = pp.boxplot(
    ...     data=df, x="category", y="value", hue="group"
    ... )
    """
    # Read defaults from rcParams if not provided
    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    color = resolve_param("color", color)
    edgecolor = resolve_param("edgecolor", edgecolor)

    # Resolve edgecolor vs linecolor (backward compat)
    if edgecolor is not None and linecolor is not None:
        warnings.warn(
            "linecolor is deprecated in favor of edgecolor. "
            "edgecolor takes precedence when both are provided.",
            FutureWarning,
            stacklevel=2,
        )
    resolved_edgecolor = edgecolor if edgecolor is not None else linecolor

    # Create figure if not provided
    # Only fall back to matplotlib's figsize when the user explicitly provides one;
    # otherwise use pp.subplots so axes_size comes from pp.rcParams["subplots.axes_size"].
    if ax is None:
        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            from publiplots.layout.subplots import subplots as _pp_subplots
            fig, ax = _pp_subplots()
    else:
        fig = ax.get_figure()

    # Resolve palette
    if hue is not None:
        palette = resolve_palette_map(
            values=data[hue].unique(),
            palette=palette,
        )

    # Determine categorical axis
    categorical_axis = "x"  # default
    if x is not None and y is not None:
        categorical_axis = "x" if is_categorical(data[x]) else "y"
    elif orient == "h":
        categorical_axis = "y"

    # Prepare kwargs for seaborn boxplot
    boxplot_kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "order": order,
        "hue_order": hue_order,
        "orient": orient,
        "color": color if hue is None else None,
        "linecolor": resolved_edgecolor,
        "palette": palette if hue else None,
        "width": width,
        "gap": gap,
        "whis": whis,
        "showcaps": showcaps,
        "fliersize": fliersize,
        "linewidth": linewidth,
        "fill": True,  # Need fill=True to get patches
        "ax": ax,
        "legend": False,  # Handle legend ourselves
    }

    # Merge with user-provided kwargs
    boxplot_kwargs.update(kwargs)

    # Track artists before plotting
    tracker = ArtistTracker(ax)

    # Create boxplot
    sns.boxplot(**boxplot_kwargs)

    # Get newly created patches and lines
    new_patches = tracker.get_new_patches()
    new_lines = tracker.get_new_lines()

    # Build a map of position -> color from patches
    # Position is on the categorical axis (x or y)
    patch_colors = {}
    for patch in new_patches:
        verts = patch.get_path().vertices
        if categorical_axis == "x":
            pos = round(np.mean(verts[:, 0]), 2)
        else:
            pos = round(np.mean(verts[:, 1]), 2)
        patch_colors[pos] = patch.get_facecolor()

    # Resolve markeredgewidth for outliers
    flierprops = kwargs.get("flierprops", {})
    markeredgewidth = flierprops.get("markeredgewidth", None)
    markeredgewidth = resolve_param("lines.markeredgewidth", markeredgewidth)

    # Recolor lines based on position and edgecolor
    for line in new_lines:
        line_data = line.get_xdata() if categorical_axis == "x" else line.get_ydata()
        if len(line_data) == 0:
            continue
        pos = np.mean(line_data)
        closest_pos = min(patch_colors.keys(), key=lambda p: abs(p - pos))
        base_color = patch_colors[closest_pos]

        if line.get_marker() and line.get_marker() != 'None':
            # Outlier markers: face = palette color, edge = edgecolor or palette
            line.set_markerfacecolor(base_color)
            line.set_markeredgecolor(resolved_edgecolor if resolved_edgecolor else base_color)
            line.set_markeredgewidth(markeredgewidth)
        else:
            # Structural lines (whiskers, caps, medians)
            line.set_color(resolved_edgecolor if resolved_edgecolor else base_color)
            line.set_linewidth(linewidth)

    # Set edge colors on box patches
    for patch in new_patches:
        patch.set_edgecolor(resolved_edgecolor if resolved_edgecolor else patch.get_facecolor())

    # Apply transparency to box patch faces
    tracker.apply_transparency(on="patches", face_alpha=alpha)
    # Apply transparency to outlier marker faces only
    for line in new_lines:
        if line.get_marker() and line.get_marker() != 'None':
            fc = line.get_markerfacecolor()
            line.set_markerfacecolor(to_rgba(fc, alpha))

    # Stash legend entries and optionally render per-axis legend.
    # legend may be bool or dict[str, bool]; False short-circuits both.
    _stash_legend(
        ax=ax,
        hue=hue,
        palette=palette,
        edgecolor=resolved_edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        legend=legend,
        legend_kws=legend_kws,
    )

    # Set labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    return fig, ax


def _stash_legend(
        ax: Axes,
        hue: Optional[str],
        palette: Optional[Union[str, Dict, List]],
        edgecolor: Optional[str],
        alpha: Optional[float],
        linewidth: Optional[float],
        legend: Union[bool, Dict],
        legend_kws: Optional[Dict],
    ) -> None:
    """Delegate to the shared hue-legend helper."""
    stash_hue_legend(
        ax,
        hue=hue,
        palette=palette,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        legend=legend,
        legend_kws=legend_kws,
    )
