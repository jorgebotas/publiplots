"""
Point plot visualization module for publiplots.

Provides point plot (line plot with markers and error bars) visualizations
with support for categorical grouping and custom marker styling.
"""

from publiplots.utils.validation import is_categorical
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, Dict, List

from publiplots.themes.rcparams import resolve_param
from publiplots.themes.colors import resolve_palette_map
from publiplots.themes.markers import resolve_marker_map
from publiplots.themes.linestyles import resolve_linestyle_map
from publiplots.utils import create_legend_handles
from publiplots.utils.legend import legend as legend_fn
from publiplots.utils.legend_entries import (
    LegendEntry,
    stash_entry,
    get_entries,
    resolve_legend_flags,
    entry_is_in_group,
    is_continuous_hue,
)


def pointplot(
    data: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    order: Optional[List] = None,
    hue_order: Optional[List] = None,
    estimator: str = "mean",
    errorbar: Optional[Union[str, Tuple]] = ("ci", 95),
    n_boot: int = 1000,
    seed: Optional[int] = None,
    units: Optional[str] = None,
    weights: Optional[str] = None,
    color: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    markers: Optional[Union[bool, List[str], Dict[str, str]]] = "o",
    linestyle: Optional[str] = None,
    linestyles: Optional[Union[str, List[str], Dict[str, str]]] = None,
    dodge: Union[bool, float] = False,
    orient: Optional[str] = None,
    capsize: float = 0,
    edgecolor: Optional[str] = None,
    err_kws: Optional[Dict] = None,
    alpha: Optional[float] = None,
    linewidth: Optional[float] = None,
    markersize: Optional[float] = None,
    markeredgewidth: Optional[float] = None,
    ax: Optional[Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    **kwargs
) -> Tuple[plt.Figure, Axes]:
    """
    Create a point plot with publiplots styling.

    Point plots show point estimates and error bars for numeric data
    grouped by categorical variables. Points are connected by lines to
    emphasize trends over categories. Markers use a distinctive double-layer
    style with white background and semi-transparent colored fill.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing x, y, and optional hue columns.
    x : str, optional
        Column name for x-axis (categorical).
    y : str, optional
        Column name for y-axis (numeric).
    hue : str, optional
        Column name for grouping variable that will produce lines with
        different colors and markers.
    order : list, optional
        Order to plot the categorical levels in.
    hue_order : list, optional
        Order to plot the hue levels in.
    estimator : str, default="mean"
        Statistical function to estimate within each categorical bin.
        Options: "mean", "median", etc.
    errorbar : str or tuple, default=("ci", 95)
        Method for computing error bars. Options include:
        - ("ci", confidence_level): Confidence interval
        - ("pi", confidence_level): Prediction interval
        - ("se", multiplier): Standard error with optional multiplier
        - "sd": Standard deviation
        - ("custom", (lower, upper)): Custom error values. Define precomputed
          lower and upper error bounds in the data.
          NOTE: If custom error values are provided, errorbar and estimator
          parameters are ignored.
        - None: No error bars
    n_boot : int, default=1000
        Number of bootstrap iterations for computing confidence intervals.
    seed : int, optional
        Seed for random number generator.
    units : str, optional
        Column for sampling units (for nested data).
    weights : str, optional
        Column for observation weights.
    color : str, optional
        Fixed color for all points (only used when hue is None).
    palette : str, dict, list, or None
        Color palette for hue values:
        - str: palette name (e.g., "pastel")
        - dict: mapping of hue values to colors
        - list: list of colors
        - None: uses default palette
    markers : bool, list, dict, optional. Default: "o"
        Markers to use for different hue levels:
        - True: use default marker set
        - list: list of marker symbols (e.g., ["o", "^", "s"])
        - dict: mapping of hue values to markers
        - None: uses default marker "o" for all points
    linestyles : str, list, dict, optional
        Line styles to use for different hue levels.
    dodge : bool or float, default=False
        Amount to separate points for different hue levels along the
        categorical axis.
    orient : str, optional
        Orientation of the plot ('v' or 'h').
    capsize : float, default=0
        Width of error bar caps.
    err_kws : dict, optional
        Additional keyword arguments for error bar styling.
        Example: {'linewidth': 1.5, 'alpha': 0.7}
    alpha : float, default=0.4
        Transparency level for marker fill (0-1).
    linewidth : float, default=2.0
        Width of marker edges and connecting lines.
    markersize : float, default=10
        Size of markers.
    markeredgewidth : float, default=1.0
        Width of marker edges.
    ax : Axes, optional
        Matplotlib axes object. If None, creates new figure.
    title : str, default=""
        Plot title.
    xlabel : str, default=""
        X-axis label. If empty, uses x column name.
    ylabel : str, default=""
        Y-axis label. If empty, uses y column name.
    legend : bool, default=True
        Whether to show legend.
    legend_kws : dict, optional
        Keyword arguments for legend builder:
        - hue_label : str, optional - Title for the hue legend.
    **kwargs
        Additional keyword arguments passed to seaborn.pointplot().

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    ax : Axes
        Matplotlib axes object.

    Examples
    --------
    Simple point plot:
    >>> fig, ax = pp.pointplot(data=df, x="time", y="value")

    Point plot with grouping:
    >>> fig, ax = pp.pointplot(
    ...     data=df, x="time", y="value", hue="group",
    ...     palette={'Control': '#8E8EC1', 'Treated': '#75B375'}
    ... )

    Point plot with custom markers:
    >>> fig, ax = pp.pointplot(
    ...     data=df, x="time", y="value", hue="group",
    ...     markers=["o", "D"], errorbar="se"
    ... )
    """
    # Read defaults from rcParams if not provided
    linewidth = resolve_param("lines.linewidth", linewidth)
    markersize = resolve_param("lines.markersize", markersize)
    markeredgewidth = resolve_param("lines.markeredgewidth", markeredgewidth)
    alpha = resolve_param("alpha", alpha)
    color = resolve_param("color", color)
    linestyle = resolve_param("lines.linestyle", linestyle)
    edgecolor = resolve_param("edgecolor", edgecolor)

    if kwargs.pop("join", None) is not None:
        raise DeprecationWarning(
            "join parameter is deprecated. Use linestyle='none' instead for no connecting lines."
        )

    # Create figure via pp.subplots to install SubplotsAutoLayout; users who
    # want custom dimensions should compose with pp.subplots(axes_size=...)
    # before calling and pass ax=.
    if ax is None:
        from publiplots.layout.subplots import subplots as _pp_subplots
        fig, ax = _pp_subplots()
    else:
        fig = ax.get_figure()

    # Resolve palette, markers and linestyles
    marker_map = {}
    linestyle_map = {}

    if linestyle == "none":
        linestyles = "none"

    if hue is not None:
        hue_values = data[hue].unique() if hue_order is None else hue_order
        palette = resolve_palette_map(
            values=hue_values,
            palette=palette,
        )
        # hue_order = list(palette.keys())

        if markers is not False:
            if markers is True:
                markers = None
            elif isinstance(markers, str):
                markers = [markers]
            # Get marker mapping
            marker_map = resolve_marker_map(values=list(hue_values), marker_map=markers)
            # Convert to list for seaborn
            markers = [marker_map[val] for val in data[hue].unique()]
        
        if linestyles is None:
            linestyles = [linestyle]
        elif isinstance(linestyles, str):
            linestyles = [linestyles]
        linestyle_map = resolve_linestyle_map(values=list(hue_values), linestyle_map=linestyles)
        linestyles = [linestyle_map[val] for val in data[hue].unique()]

    # Prepare err_kws with linewidth
    if err_kws is None:
        err_kws = {}
    if 'linewidth' not in err_kws:
        err_kws['linewidth'] = linewidth

    # Resolve errorbar if custom error values are provided
    if isinstance(errorbar, tuple) and errorbar[0] == "custom":
        data = _format_for_custom_errorbar(data, x, y, errorbar[1], orient)
        estimator = "median"   # middle value for estimator
        errorbar = ("pi", 100) # Use full range

    # Prepare kwargs for seaborn pointplot
    pointplot_kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "order": order,
        "hue_order": hue_order,
        "estimator": estimator,
        "errorbar": errorbar,
        "n_boot": n_boot,
        "seed": seed,
        "units": units,
        "weights": weights,
        "color": color if hue is None else None,
        "palette": palette if hue else None,
        "linewidth": linewidth,
        "markers": markers if hue else "o",
        "markersize": markersize,
        "markeredgewidth": markeredgewidth,
        "linestyle": "none",
        "linestyles": linestyles,
        "dodge": dodge,
        "orient": orient,
        "capsize": capsize,
        "err_kws": err_kws,
        "ax": ax,
        "legend": False,  # Handle legend ourselves
    }

    # Merge with user-provided kwargs
    pointplot_kwargs.update(kwargs)

    # Create pointplot
    sns.pointplot(**pointplot_kwargs)

    # Apply marker styling (double-layer effect)
    _apply_marker_styling(
        ax=ax,
        alpha=alpha,
        edgecolor=edgecolor,
    )

    # Set labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    # Stash legend entry (per-kind) and optionally render per-axis legend.
    # legend may be bool or dict[str, bool]; False short-circuits both stash
    # and render, True stashes/renders everything, and a dict filters per kind.
    if hue is not None:
        _legend(
            ax=ax,
            hue=hue,
            palette=palette,
            markers=list(marker_map.values()),
            linestyles=list(linestyle_map.values()),
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            kwargs=legend_kws,
            legend=legend,
        )

    return fig, ax

def _format_for_custom_errorbar(
    data: pd.DataFrame,
    x: str,
    y: str,
    custom_errorbar: Tuple[str, str],
    orient: Optional[str] = None,
) -> pd.DataFrame:
    """"
    Format data for custom errorbar.
    """
    # Figure out categorical variable
    value_col = None
    if is_categorical(data[y]) or orient.isin(["horizontal", "h", "x"]):
        value_col = x
    elif is_categorical(data[x]) or orient.isin(["vertical", "v", "y"]):
        value_col = y
    else:
        raise ValueError(
            "One of x or y must be categorical. "
            "Or orient must be specified."
        )

    lower, upper = custom_errorbar
    assert lower in data.columns and upper in data.columns, \
        "Custom errorbar must be a tuple of column names."
    
    data = pd.concat([
        data.assign(__value=data[lower]),
        data.assign(__value=data[value_col]),
        data.assign(__value=data[upper]),
    ], ignore_index=True)
    data[value_col] = data["__value"]
    data.drop(columns=["__value"], inplace=True)
    return data


def _apply_marker_styling(
    ax: Axes,
    alpha: float,
    edgecolor: Optional[str] = None,
) -> List[Dict]:
    """
    Apply double-layer marker styling to pointplot.

    Extracts markers from lines, removes them, and redraws with:
    1. White background layer
    2. Semi-transparent colored fill layer

    Parameters
    ----------
    ax : Axes
        Axes containing the pointplot.
    alpha : float
        Transparency for marker fill.
    markeredgewidth : float
        Width of marker edges.

    """
    # Extract markers and line info from lines
    markers = []
    for line in ax.lines:
        marker = line.get_marker()
        if marker != "None":
            markers.append({
                'x': line.get_xdata(),
                'y': line.get_ydata(),
                'marker': marker,
                'color': line.get_color(),
                'markersize': line.get_markersize(),
                'linestyle': line.get_linestyle(),
                'linewidth': line.get_linewidth(),
                'markeredgewidth': line.get_markeredgewidth(),
            })
            # Remove marker from original line (keep the line itself)
            line.set_markersize(0)

    # Redraw markers with double-layer effect
    for data in markers:
        x = data['x']
        y = data['y']
        marker = data['marker']
        color = data['color']
        size = data['markersize']
        markeredgewidth = data['markeredgewidth'] 

        # Layer 1: White background (no edge)
        ax.plot(
            x, y, marker,
            markeredgecolor=edgecolor if edgecolor else color,
            markerfacecolor="white",
            markersize=size,
            markeredgewidth=0,
            linestyle='none',
            zorder=99
        )

        # Layer 2: Semi-transparent colored fill with edge
        ax.plot(
            x, y, marker,
            markeredgecolor=edgecolor if edgecolor else color,
            markerfacecolor=to_rgba(color, alpha),
            markersize=size,
            markeredgewidth=markeredgewidth,
            linestyle='none',
            zorder=100
        )


def _legend(
    ax: Axes,
    hue: str,
    palette: Union[Dict, str],
    markers: Union[List, Dict],
    linestyles: Union[List, Dict],
    markersize: Optional[float] = None,
    markeredgewidth: Optional[float] = None,
    edgecolor: Optional[str] = None,
    alpha: Optional[float] = None,
    linewidth: Optional[float] = None,
    kwargs: Optional[Dict] = None,
    legend: Union[bool, Dict] = True,
) -> None:
    """
    Stash LegendEntry objects for point plot and optionally render per-axis legend.

    Parameters
    ----------
    legend : bool or dict, default=True
        - ``True``  -> stash all kinds and render per-axis (unless claimed by a group).
        - ``False`` -> stash nothing and render nothing.
        - ``dict``  -> per-kind flags (missing keys default to True).
    """
    # Read defaults from rcParams if not provided
    alpha = resolve_param("alpha", alpha)
    linewidth = resolve_param("lines.linewidth", linewidth)
    markersize = resolve_param("lines.markersize", markersize)
    markeredgewidth = resolve_param("lines.markeredgewidth", markeredgewidth)

    kwargs = kwargs or {}

    flags = resolve_legend_flags(legend)

    # Stash hue entry
    if flags["hue"] and isinstance(palette, dict):
        # Use palette keys as the source of truth for labels
        labels = list(palette.keys())
        colors = list(palette.values())

        # Create legend handles with line+marker style
        hue_handles = create_legend_handles(
            labels=[str(label) for label in labels],
            colors=colors,
            edgecolors=[edgecolor] * len(labels) if edgecolor else None,
            markers=markers,
            linestyles=linestyles,
            alpha=alpha,
            linewidth=linewidth,
            sizes=[markersize],
            markeredgewidth=markeredgewidth,
        )

        hue_label = kwargs.pop("hue_label", hue)
        stash_entry(
            ax,
            LegendEntry.build(
                name=hue_label,
                kind="hue",
                handles=hue_handles,
                labels=[str(label) for label in labels],
            ),
        )

    # Render per-axis legend for entries not claimed by a figure-level group.
    # legend=False short-circuits (all flags False, nothing stashed, nothing to render).
    if legend is False:
        return

    fig = ax.get_figure()
    entries_to_render = [
        e for e in get_entries(ax)
        if flags[e.kind] and not entry_is_in_group(fig, e)
    ]
    if not entries_to_render:
        return

    builder = legend_fn(ax=ax, auto=False)
    for entry in entries_to_render:
        if entry.kind == "hue" and is_continuous_hue(entry.handles):
            builder.add_colorbar(
                mappable=entry.handles[0],
                label=entry.name,
            )
        else:
            builder.add_legend(
                handles=list(entry.handles),
                label=entry.name,
            )
