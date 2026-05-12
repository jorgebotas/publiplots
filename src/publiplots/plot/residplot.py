"""
Residual plot visualization module for publiplots.

Wraps :func:`seaborn.residplot` — a scatter of the residuals of a linear
(or polynomial / robust) regression — and adds support for ``hue=`` as a
per-group loop. Seaborn's own ``residplot`` has no hue dimension; we
loop and assign a palette color per level, routing the result through
the standard publiplots legend-entry pipeline (one ``LegendEntry`` for
the hue column, rendered per-axis unless claimed by a figure-level
``pp.legend_group``).
"""

from typing import Dict, List, Optional, Union
import warnings

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from publiplots.themes.rcparams import resolve_param
from publiplots.themes.colors import resolve_palette_map
from publiplots.utils import is_categorical, create_legend_handles
from publiplots.utils.legend_entries import (
    LegendEntry,
    stash_entry,
    resolve_legend_flags,
)
from publiplots.utils.plot_legend import render_entries


def residplot(
    data: Optional[pd.DataFrame] = None,
    *,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    color: Optional[str] = None,
    alpha: Optional[float] = None,
    marker: Optional[str] = None,
    linewidth: Optional[float] = None,
    edgecolor: Optional[str] = None,
    x_partial: Optional[Union[str, List[str]]] = None,
    y_partial: Optional[Union[str, List[str]]] = None,
    lowess: bool = False,
    order: int = 1,
    robust: bool = False,
    dropna: bool = True,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    scatter_kws: Optional[Dict] = None,
    line_kws: Optional[Dict] = None,
    **kwargs,
) -> Axes:
    """Plot the residuals of a linear regression, with optional hue grouping.

    Wraps :func:`seaborn.residplot` and adds a ``hue=`` dimension that
    seaborn doesn't natively support — when provided, the data is split
    by level and a separate ``residplot`` is drawn per group, each tinted
    by the resolved palette color. A single categorical hue
    :class:`~publiplots.utils.legend_entries.LegendEntry` is stashed on
    the axes and rendered through the publiplots legend reactor.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Input data containing ``x``, ``y``, and any ``hue`` column.
    x, y : str
        Column names for the predictor / response variables.
    hue : str, optional
        Column name for a categorical grouping. Each group gets its own
        residual computation and scatter; the palette mapping is one
        color per level. Numeric hue triggers a :class:`UserWarning`
        and falls back to a single residual plot.
    palette : str | dict | list, optional
        Palette specification for categorical hue. See
        :func:`publiplots.themes.colors.resolve_palette_map`.
    color : str, optional
        Single color used when ``hue`` is None. Falls back to
        ``pp.rcParams["color"]``.
    alpha : float, optional
        Scatter face transparency. Falls back to ``pp.rcParams["alpha"]``.
    marker : str, optional
        Matplotlib marker code for the scatter points. Forwarded into
        ``scatter_kws`` (seaborn's ``residplot`` has no top-level
        ``marker=`` kwarg).
    linewidth : float, optional
        Scatter marker edge width. Falls back to
        ``pp.rcParams["lines.linewidth"]``.
    edgecolor : str, optional
        Scatter marker edge color. Falls back to
        ``pp.rcParams["edgecolor"]``.
    x_partial, y_partial : str | list of str, optional
        Confounding columns regressed out before computing residuals.
        Forwarded unchanged to :func:`seaborn.residplot`.
    lowess : bool, default False
        Fit a LOWESS smoother on the residuals for visual diagnosis.
    order : int, default 1
        Polynomial order of the regression used to compute residuals.
    robust : bool, default False
        Use a robust linear regression.
    dropna : bool, default True
        Drop missing observations before fitting.
    ax : Axes, optional
        Target axes. When ``None``, a new figure is created via
        :func:`publiplots.subplots`.
    title : str, optional
        Plot title. ``None`` (default) leaves the axes title alone.
    xlabel, ylabel : str, optional
        Axis labels. ``None`` (default) keeps seaborn's inferred labels.
    legend : bool | dict, default True
        ``True`` stashes + renders the hue entry (if present). ``False``
        stashes nothing. A dict maps legend kinds to booleans
        (residplot only emits ``"hue"``).
    legend_kws : dict, optional
        Forwarded to the legend builder (e.g. ``{'inside': True}``).
        ``hue_label`` overrides the legend title.
    scatter_kws : dict, optional
        Extra keyword arguments forwarded to the scatter layer of
        :func:`seaborn.residplot`. User values override our
        publiplots defaults (linewidth, edgecolor, alpha).
    line_kws : dict, optional
        Extra keyword arguments forwarded to the LOWESS line layer.
    **kwargs
        Reserved for future expansion. ``figsize=`` is rejected —
        see :func:`publiplots.layout.subplots.reject_figsize`.

    Returns
    -------
    Axes
        The axes where the plot was drawn.

    Examples
    --------
    Simple residual plot (linear fit):

    >>> ax = pp.residplot(data=df, x="x", y="y")

    Residuals split by a categorical group (novelty over seaborn):

    >>> ax = pp.residplot(data=df, x="x", y="y", hue="treatment")

    Polynomial residuals with a LOWESS smoother:

    >>> ax = pp.residplot(data=df, x="x", y="y", order=2, lowess=True)
    """
    from publiplots.layout.subplots import reject_figsize, subplots as _pp_subplots
    reject_figsize(kwargs)

    # Read defaults from rcParams if not provided.
    linewidth = resolve_param("lines.linewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    color = resolve_param("color", color)
    edgecolor = resolve_param("edgecolor", edgecolor)

    # Validate required columns.
    if data is not None:
        required = [c for c in (x, y, hue) if c is not None]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")

    if ax is None:
        _, ax = _pp_subplots()

    # Common scatter_kws defaults (user values win via setdefault).
    scatter_kws = dict(scatter_kws or {})
    line_kws = dict(line_kws or {})
    scatter_kws.setdefault("linewidths", linewidth)
    if alpha is not None:
        scatter_kws.setdefault("alpha", alpha)
    if edgecolor is not None:
        scatter_kws.setdefault("edgecolor", edgecolor)
    if marker is not None:
        scatter_kws.setdefault("marker", marker)

    # Decide whether we take the per-group hue path or the single-call path.
    palette_map: Dict = {}
    fallback_to_single = False
    if hue is not None and data is not None:
        hue_values = data[hue].values
        if is_categorical(hue_values):
            # Preserve encounter order of unique hue levels for a stable
            # legend order (matches scatterplot conventions).
            unique_levels = list(pd.unique(data[hue]))
            palette_map = resolve_palette_map(
                values=unique_levels, palette=palette,
            )
        else:
            warnings.warn(
                "continuous hue not supported on pp.residplot; "
                "falling back to single residual plot",
                UserWarning,
                stacklevel=2,
            )
            fallback_to_single = True

    if hue is None or fallback_to_single or not palette_map:
        # Single-color path.
        sns.residplot(
            data=data, x=x, y=y,
            color=color,
            x_partial=x_partial, y_partial=y_partial,
            lowess=lowess, order=order, robust=robust, dropna=dropna,
            scatter_kws=scatter_kws, line_kws=line_kws,
            ax=ax,
        )
    else:
        # Per-group loop. Each call adds a scatter PathCollection (and,
        # for ``lowess=True``, an additional line). All calls overdraw
        # the same y=0 dotted reference line — we accept this as a visual
        # no-op (all copies render identically).
        for level, color_hex in palette_map.items():
            sub = data[data[hue] == level]
            if sub.empty:
                continue
            # Per-group scatter_kws copy so the per-call setdefaults don't
            # pollute the shared dict for the next iteration.
            group_scatter_kws = dict(scatter_kws)
            group_line_kws = dict(line_kws)
            sns.residplot(
                data=sub, x=x, y=y,
                color=color_hex,
                x_partial=x_partial, y_partial=y_partial,
                lowess=lowess, order=order, robust=robust, dropna=dropna,
                scatter_kws=group_scatter_kws, line_kws=group_line_kws,
                ax=ax,
            )

    # Labels / title. Only set when user passed an explicit value;
    # ``None`` preserves seaborn's inferred x / y names.
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    _legend(
        ax=ax,
        hue=hue,
        palette_map=palette_map,
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        marker=marker,
        legend=legend,
        legend_kws=legend_kws,
    )

    return ax


def _legend(
    ax: Axes,
    *,
    hue: Optional[str],
    palette_map: Dict,
    alpha: Optional[float],
    linewidth: Optional[float],
    edgecolor: Optional[str],
    marker: Optional[str],
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
) -> None:
    """Stash the (categorical) hue entry and render per-axis legends.

    No-op when ``hue`` is None, when the hue was continuous (no palette
    map was built), or when ``legend=False``.
    """
    if legend is False:
        return
    if hue is None or not palette_map:
        return

    flags = resolve_legend_flags(legend)
    legend_kws = dict(legend_kws or {})
    hue_label = legend_kws.pop("hue_label", hue)

    if flags["hue"]:
        labels = [str(k) for k in palette_map.keys()]
        handles = create_legend_handles(
            labels=labels,
            colors=list(palette_map.values()),
            edgecolors=[edgecolor] * len(palette_map) if edgecolor else None,
            markers=[marker] * len(palette_map) if marker is not None else None,
            alpha=alpha,
            linewidth=linewidth,
            style="circle",
        )
        stash_entry(
            ax,
            LegendEntry.build(
                name=hue_label,
                kind="hue",
                handles=handles,
                labels=labels,
            ),
        )

    render_entries(ax, flags=flags, legend_kws=legend_kws)
