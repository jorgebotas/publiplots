"""
Point plot visualization module for publiplots.

Provides point plot (line plot with markers and error bars) visualizations
with support for categorical grouping and custom marker styling.
"""

from publiplots.utils.validation import is_categorical
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, Dict, List

from publiplots.themes.rcparams import resolve_param
from publiplots.themes.colors import resolve_palette_map
from publiplots.utils import create_legend_handles
from publiplots.utils.errorbar import format_for_custom_errorbar
from publiplots.utils.legend_entries import (
    LegendEntry,
    stash_entry,
    get_entries,
    resolve_legend_flags,
    entry_is_in_group,
    is_continuous_hue,
)
from publiplots.utils.plot_legend import render_entries, resolve_style_maps


# Map seaborn's legacy orient tokens onto the modern {'x', 'y'} set used
# by :func:`publiplots.utils.errorbar.format_for_custom_errorbar`.
_MODERN_ORIENT = {
    "h": "y",
    "horizontal": "y",
    "v": "x",
    "vertical": "x",
    "x": "x",
    "y": "y",
}


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
    annotate: Union[bool, Dict, None] = None,
    **kwargs
) -> Axes:
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
        Error-bar specification. Accepts every seaborn-native form — see
        :func:`seaborn.pointplot` for the full list: ``("ci", level)``,
        ``("pi", level)``, ``("se", mult)``, ``"sd"``, ``None``.

        publiplots additionally supports ``("custom", (lower_col, upper_col))``
        to draw whisker endpoints at **precomputed** lower/upper bounds
        stored on the data (e.g. odds-ratio CIs, effect sizes with
        externally-computed bootstrap intervals). When this form is used,
        ``estimator`` is forced to ``"median"`` and any ``n_boot`` / ``seed``
        arguments are ignored.

        Example::

            pp.pointplot(
                data=df, x='gene', y='log2_or',
                errorbar=('custom', ('log2_lower', 'log2_upper')),
            )
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
    Axes
        The axes where the plot was drawn.

    Examples
    --------
    Simple point plot:
    >>> ax = pp.pointplot(data=df, x="time", y="value")

    Point plot with grouping:
    >>> ax = pp.pointplot(
    ...     data=df, x="time", y="value", hue="group",
    ...     palette={'Control': '#8E8EC1', 'Treated': '#75B375'}
    ... )

    Point plot with custom markers:
    >>> ax = pp.pointplot(
    ...     data=df, x="time", y="value", hue="group",
    ...     markers=["o", "D"], errorbar="se"
    ... )
    """
    from publiplots.layout.subplots import reject_figsize
    reject_figsize(kwargs)

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

    # Preserve the caller's original DataFrame identity for downstream
    # annotate builders that stash `source_frame` on the meta.
    _source_data = data

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

        # Normalize sentinel inputs before delegating to shared resolver.
        if markers is not False:
            if markers is True:
                markers = None
            elif isinstance(markers, str):
                markers = [markers]

        if linestyles is None:
            linestyles = [linestyle]
        elif isinstance(linestyles, str):
            linestyles = [linestyles]

        # Resolve marker + linestyle maps via shared helper. Pass True to
        # request defaults when no explicit list/dict is provided; pass False
        # to skip marker resolution entirely.
        marker_arg = False if markers is False else (markers if markers is not None else True)
        marker_map, linestyle_map = resolve_style_maps(
            style=hue,
            data=data,
            style_order=list(hue_values),
            markers=marker_arg,
            dashes=linestyles,
        )

        # Convert back to lists for seaborn (one entry per unique hue value).
        if markers is not False:
            markers = [marker_map[val] for val in data[hue].unique()]
        linestyles = [linestyle_map[val] for val in data[hue].unique()]

    # Prepare err_kws with linewidth
    if err_kws is None:
        err_kws = {}
    if 'linewidth' not in err_kws:
        err_kws['linewidth'] = linewidth

    # Resolve errorbar if custom error values are provided. Translate
    # seaborn's legacy orient tokens ('h'/'v') to the modern {'x','y'}
    # set that the shared helper expects; pass None through so the
    # helper can auto-detect from which axis is categorical.
    if isinstance(errorbar, tuple) and errorbar[0] == "custom":
        modern_orient = _MODERN_ORIENT.get(orient) if orient is not None else None
        data = format_for_custom_errorbar(data, x, y, errorbar[1], modern_orient)
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

    # Apply marker styling (double-layer effect — shared with lineplot)
    from publiplots.utils.transparency import apply_double_layer_markers
    apply_double_layer_markers(
        ax,
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

    # Determine categorical axis (same logic as before).
    if is_categorical(data[x]):
        categorical_axis = x
    elif is_categorical(data[y]):
        categorical_axis = y
    else:
        categorical_axis = x

    # Always attach the cache so follow-up pp.annotate(ax, ...) calls work.
    from publiplots.annotate._builders import build_from_pointplot_call
    ax._publiplots_point_meta = build_from_pointplot_call(
        ax=ax, data=data, x=x, y=y, hue=hue,
        categorical_axis=categorical_axis,
        palette=palette if isinstance(palette, dict) else None,
        errorbar=errorbar if isinstance(errorbar, str) else None,
        source_frame=_source_data,
    )
    if annotate:
        from publiplots.annotate import annotate as _annotate_fn
        opts = dict(annotate) if isinstance(annotate, dict) else {}
        kind = opts.pop("kind", "point_values")
        _annotate_fn(ax, kind=kind, **opts)

    return ax

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

    render_entries(ax, flags=flags, legend_kws=kwargs)
