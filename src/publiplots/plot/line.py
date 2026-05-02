"""
Line plot visualization module for publiplots.

Provides ``lineplot`` with full parity to :func:`seaborn.lineplot`. Supports
continuous and categorical x/y variables, hue (categorical or continuous),
size, and style encodings, aggregation with error bands/bars, and the
publiplots legend-entry pipeline (per-axis legends unless claimed by a
figure-level ``pp.legend_group``).
"""

from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from publiplots.themes.rcparams import resolve_param
from publiplots.themes.colors import resolve_palette_map
from publiplots.utils import is_categorical, is_numeric, create_legend_handles
from publiplots.utils.legend import legend as legend_fn
from publiplots.utils.legend_entries import (
    LegendEntry,
    stash_entry,
    get_entries,
    resolve_legend_flags,
    entry_is_in_group,
    is_continuous_hue,
)
from publiplots.utils.plot_legend import (
    get_size_ticks,
    stash_continuous_hue,
    resolve_style_maps,
)


def lineplot(
    data: pd.DataFrame,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    size: Optional[str] = None,
    style: Optional[str] = None,
    units: Optional[str] = None,
    weights: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    hue_order: Optional[List] = None,
    hue_norm: Optional[Union[Tuple[float, float], Normalize]] = None,
    sizes: Optional[Union[List, Dict, Tuple[float, float]]] = None,
    size_order: Optional[List] = None,
    size_norm: Optional[Union[Tuple[float, float], Normalize]] = None,
    dashes: Union[bool, List, Dict] = True,
    markers: Optional[Union[bool, List, Dict]] = None,
    style_order: Optional[List] = None,
    estimator: Optional[Union[str, "callable"]] = "mean",
    errorbar: Optional[Union[str, Tuple]] = ("ci", 95),
    n_boot: int = 1000,
    seed: Optional[int] = None,
    orient: str = "x",
    sort: bool = True,
    err_style: str = "band",
    err_kws: Optional[Dict] = None,
    color: Optional[str] = None,
    alpha: Optional[float] = None,
    linewidth: Optional[float] = None,
    markersize: Optional[float] = None,
    markeredgewidth: Optional[float] = None,
    edgecolor: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    **kwargs,
) -> Axes:
    """
    Create a line plot with publiplots styling and seaborn-parity kwargs.

    Line plots draw one or more line series over a shared x variable.
    Observations at each x are aggregated with ``estimator`` and error
    uncertainty is drawn as bands or bars per ``err_style``. Supports
    visual encoding of additional variables via ``hue`` (color),
    ``size`` (line width), and ``style`` (marker + dash pattern).

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing ``x``, ``y``, and any encoded columns.
    x : str, optional
        Column name for the x-axis variable.
    y : str, optional
        Column name for the y-axis variable.
    hue : str, optional
        Column name for color encoding. Can be categorical (uses
        ``palette`` mapping) or continuous (renders a colorbar legend).
    size : str, optional
        Column name for line-width encoding.
    style : str, optional
        Column name for marker/dash style encoding. Categorical only.
    units : str, optional
        Column name for sampling units (for nested data).
    weights : str, optional
        Column name for observation weights.
    palette : str, dict, list, or None
        Color palette for ``hue``:

        - str: palette name (e.g., "viridis", "pastel")
        - dict: mapping of hue values to colors (categorical only)
        - list: list of colors
        - None: uses default palette
    hue_order : list, optional
        Order of categorical hue levels.
    hue_norm : tuple of float or Normalize, optional
        ``(vmin, vmax)`` or a ``Normalize`` instance for continuous hue.
        If None with numeric hue, inferred from data.
    sizes : list, dict, or tuple of float, optional
        Width range or mapping for ``size`` encoding.
    size_order : list, optional
        Order of categorical size levels.
    size_norm : tuple of float or Normalize, optional
        Normalization for numeric ``size``.
    dashes : bool, list, or dict, default=True
        Dash patterns for ``style`` levels. ``True`` uses defaults, a
        list/dict specifies patterns, ``False`` disables dashing.
    markers : bool, list, or dict, optional
        Markers for ``style`` levels. ``True`` uses defaults, a list/dict
        specifies markers, ``None``/``False`` draws no markers.
    style_order : list, optional
        Order of categorical style levels.
    estimator : str or callable, default="mean"
        Aggregator applied within each x group. ``None`` disables
        aggregation (one line per observation).
    errorbar : str or tuple, default=("ci", 95)
        Error-bar specification. See :func:`seaborn.lineplot` for the
        full list: ``("ci", level)``, ``("pi", level)``, ``("se", mult)``,
        ``"sd"``, ``None``.
    n_boot : int, default=1000
        Number of bootstrap iterations for CI computation.
    seed : int, optional
        Seed for the random number generator used in bootstrapping.
    orient : str, default="x"
        Aggregation axis. "x" aggregates within x; "y" within y.
    sort : bool, default=True
        Sort by x before drawing. Set False to preserve input order.
    err_style : str, default="band"
        How to draw error uncertainty: ``"band"`` (shaded region) or
        ``"bars"`` (discrete error bars per estimate).
    err_kws : dict, optional
        Extra styling kwargs forwarded to the error-band/bar artists.
    color : str, optional
        Fixed color used when ``hue`` is None. Falls back to rcParams.
    alpha : float, optional
        Transparency for lines and fills. Falls back to rcParams.
    linewidth : float, optional
        Base line width. Falls back to rcParams.
    markersize : float, optional
        Marker size for style legend. Falls back to rcParams.
    markeredgewidth : float, optional
        Width of marker edges for style legend. Falls back to rcParams.
    edgecolor : str, optional
        Edge color for categorical legend swatches. Falls back to rcParams.
    ax : Axes, optional
        Matplotlib axes to draw on. If None, a new figure is created via
        :func:`publiplots.layout.subplots`. Recover the figure from the
        returned axes with ``ax.get_figure()``.
    title : str, default=""
        Plot title.
    xlabel : str, default=""
        X-axis label. Passed through only if non-empty.
    ylabel : str, default=""
        Y-axis label. Passed through only if non-empty.
    legend : bool or dict, default=True
        Legend control. ``True`` stashes and renders all legend kinds.
        ``False`` stashes and renders nothing. A dict enables/disables
        specific kinds, e.g. ``{"hue": False}`` hides only hue.
    legend_kws : dict, optional
        Legend builder tuning. Supported keys: ``hue_label``, ``size_label``,
        ``style_label`` (override kind titles); ``size_nbins``,
        ``size_min_n_ticks``, ``size_include_min_max`` (forwarded to
        :func:`get_size_ticks`).
    **kwargs
        Extra keyword arguments forwarded to :func:`seaborn.lineplot`.
        ``figsize`` is rejected — use ``pp.subplots(axes_size=...)`` first
        and pass ``ax=`` instead.

    Returns
    -------
    Axes
        The axes where the plot was drawn. Use ``ax.get_figure()`` to
        recover the figure handle.

    Examples
    --------
    Simple line plot with aggregation:

    >>> ax = pp.lineplot(data=df, x="time", y="value")

    Categorical hue with custom palette:

    >>> ax = pp.lineplot(
    ...     data=df, x="time", y="value",
    ...     hue="group", palette="pastel",
    ... )

    Continuous hue (renders a colorbar) with bars for uncertainty:

    >>> ax = pp.lineplot(
    ...     data=df, x="time", y="value",
    ...     hue="score", palette="viridis",
    ...     hue_norm=(0, 100), err_style="bars",
    ... )
    """
    from publiplots.layout.subplots import reject_figsize
    reject_figsize(kwargs)

    # Read defaults from rcParams if not provided
    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    color = resolve_param("color", color)
    edgecolor = resolve_param("edgecolor", edgecolor)
    markersize = resolve_param("lines.markersize", markersize)
    markeredgewidth = resolve_param("lines.markeredgewidth", markeredgewidth)

    # Create figure via pp.subplots to install SubplotsAutoLayout; users who
    # want custom dimensions should compose with pp.subplots(axes_size=...)
    # before calling and pass ax=.
    if ax is None:
        from publiplots.layout.subplots import subplots as _pp_subplots
        fig, ax = _pp_subplots()
    else:
        fig = ax.get_figure()

    # Palette resolution: categorical (dict mapping) vs continuous (str/cmap + norm).
    palette_resolved = None
    if hue is not None:
        if is_numeric(data[hue]) and not is_categorical(data[hue]):
            if hue_norm is None:
                hue_norm = (data[hue].min(), data[hue].max())
            if isinstance(hue_norm, tuple):
                hue_norm = Normalize(vmin=hue_norm[0], vmax=hue_norm[1])
            # seaborn accepts str or Colormap for continuous hue.
            palette_resolved = palette
        else:
            values = list(data[hue].unique() if hue_order is None else hue_order)
            palette_resolved = resolve_palette_map(values=values, palette=palette)

    # Resolve marker + linestyle maps via shared helper (used for legend stashing).
    # Note: seaborn's ``dashes`` dict expects its own dash format (``""`` or
    # ``(on, off)`` tuples), not matplotlib linestyle strings, so we pass the
    # user's ``dashes`` through untouched and only use ``linestyle_map`` for
    # the publiplots legend handles.
    marker_map, linestyle_map = resolve_style_maps(
        style=style,
        data=data,
        style_order=style_order,
        markers=markers,
        dashes=dashes,
    )

    # Size-legend defaults: when ``size`` is set but ``sizes``/``size_norm``
    # aren't, populate sensible defaults so the size legend can be built.
    if size is not None and is_numeric(data[size]):
        if sizes is None:
            sizes = (1.0, 4.0)
        if size_norm is None:
            size_norm = (data[size].min(), data[size].max())
        if isinstance(size_norm, tuple):
            size_norm = Normalize(vmin=size_norm[0], vmax=size_norm[1])

    # Prepare seaborn kwargs. publiplots handles the legend itself.
    sns_kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "size": size,
        "style": style,
        "units": units,
        "weights": weights,
        "palette": palette_resolved,
        "hue_order": hue_order,
        "hue_norm": hue_norm,
        "sizes": sizes,
        "size_order": size_order,
        "size_norm": size_norm,
        "dashes": dashes,
        "markers": marker_map if marker_map else markers,
        "style_order": style_order,
        "estimator": estimator,
        "errorbar": errorbar,
        "n_boot": n_boot,
        "seed": seed,
        "orient": orient,
        "sort": sort,
        "err_style": err_style,
        "err_kws": err_kws,
        "linewidth": linewidth,
        "ax": ax,
        "legend": False,
    }
    if hue is None:
        sns_kwargs["color"] = color

    sns_kwargs.update(kwargs)
    sns.lineplot(**sns_kwargs)

    # Labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Stash legend entries (per-kind) and render per-axis (unless grouped).
    _legend(
        ax=ax,
        data=data,
        hue=hue,
        size=size,
        style=style,
        palette=palette_resolved,
        marker_map=marker_map,
        linestyle_map=linestyle_map,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        edgecolor=edgecolor,
        hue_norm=hue_norm,
        size_norm=size_norm,
        sizes=sizes,
        legend_kws=legend_kws,
        legend=legend,
    )

    return ax


def _legend(
    ax: Axes,
    data: pd.DataFrame,
    hue: Optional[str],
    size: Optional[str],
    style: Optional[str],
    palette,
    marker_map: Dict,
    linestyle_map: Dict,
    color: Optional[str],
    alpha: Optional[float],
    linewidth: Optional[float],
    markersize: Optional[float],
    markeredgewidth: Optional[float],
    edgecolor: Optional[str],
    hue_norm,
    size_norm,
    sizes,
    legend_kws: Optional[Dict] = None,
    legend: Union[bool, Dict] = True,
) -> None:
    """Stash LegendEntry objects for lineplot; render per-axis unless grouped.

    Parameters
    ----------
    legend : bool or dict, default=True
        - ``True``  -> stash all kinds and render per-axis (unless claimed by a group).
        - ``False`` -> stash nothing and render nothing.
        - ``dict``  -> per-kind flags (missing keys default to True).
    """
    if legend is False:
        return

    flags = resolve_legend_flags(legend)
    legend_kws = dict(legend_kws or {})
    handle_kwargs = dict(alpha=alpha, linewidth=linewidth, color=color)

    # Stash hue entry
    if hue is not None and flags["hue"]:
        hue_label = legend_kws.pop("hue_label", hue)
        if isinstance(palette, dict):  # categorical
            labels = [str(k) for k in palette.keys()]
            hue_handles = create_legend_handles(
                labels=labels,
                colors=list(palette.values()),
                edgecolors=[edgecolor] * len(palette) if edgecolor else None,
                linestyles=["-"] * len(palette),
                markeredgewidth=markeredgewidth,
                **handle_kwargs,
            )
            stash_entry(
                ax,
                LegendEntry.build(
                    name=hue_label,
                    kind="hue",
                    handles=hue_handles,
                    labels=labels,
                ),
            )
        else:
            # continuous -> colorbar
            stash_continuous_hue(
                ax, name=hue_label, palette=palette, hue_norm=hue_norm,
            )

    # Stash size entry
    if size is not None and flags["size"]:
        tick_color = color if hue is None else "gray"
        size_handle_kwargs = dict(handle_kwargs)
        size_handle_kwargs["color"] = tick_color
        tick_labels, tick_sizes = get_size_ticks(
            values=data[size].dropna().values,
            sizes=sizes,
            size_norm=size_norm,
            nbins=legend_kws.pop("size_nbins", 4),
            min_n_ticks=legend_kws.pop("size_min_n_ticks", 3),
            include_min_max=legend_kws.pop("size_include_min_max", False),
        )
        size_handles = create_legend_handles(
            labels=tick_labels,
            sizes=tick_sizes,
            **size_handle_kwargs,
        )
        size_label = legend_kws.pop("size_label", size)
        stash_entry(
            ax,
            LegendEntry.build(
                name=size_label,
                kind="size",
                handles=size_handles,
                labels=tick_labels,
            ),
        )

    # Stash style entry
    if style is not None and flags["style"]:
        # Prefer linestyle_map's keys (fall back to marker_map's keys).
        style_values = list((linestyle_map or marker_map).keys())
        style_labels = [str(v) for v in style_values]
        style_color = color if hue is None else "gray"
        style_handle_kwargs = dict(handle_kwargs)
        style_handle_kwargs["color"] = style_color
        style_handles = create_legend_handles(
            labels=style_labels,
            markers=[marker_map[v] for v in style_values] if marker_map else None,
            linestyles=(
                [linestyle_map[v] for v in style_values]
                if linestyle_map
                else ["-"] * len(style_values)
            ),
            edgecolors=[edgecolor] * len(style_values) if edgecolor else None,
            sizes=[markersize] * len(style_values) if marker_map else None,
            markeredgewidth=markeredgewidth,
            **style_handle_kwargs,
        )
        stash_entry(
            ax,
            LegendEntry.build(
                name=legend_kws.pop("style_label", style),
                kind="style",
                handles=style_handles,
                labels=style_labels,
            ),
        )

    # Render per-axis legends for entries not claimed by a figure-level group.
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
