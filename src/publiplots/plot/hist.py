"""
Histogram plot functions for publiplots.

This module provides publication-ready histogram visualizations with
flexible statistic, binning, element, and grouping options, plus
optional KDE overlay.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection, QuadMesh
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from publiplots.annotate._builders import build_from_histplot_call
from publiplots.themes.colors import resolve_palette_map
from publiplots.themes.hatches import resolve_hatch_map
from publiplots.themes.rcparams import resolve_param
from publiplots.utils import create_legend_handles
from publiplots.utils.legend_entries import (
    LegendEntry,
    resolve_legend_flags,
    stash_entry,
)
from publiplots.utils.plot_legend import render_entries, stash_continuous_hue
from publiplots.utils.transparency import ArtistTracker


def histplot(
    data: Optional[pd.DataFrame] = None,
    *,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    weights: Optional[str] = None,
    stat: str = "count",
    bins: Union[str, int, List] = "auto",
    binwidth: Optional[float] = None,
    binrange: Optional[Tuple[float, float]] = None,
    discrete: Optional[bool] = None,
    cumulative: bool = False,
    common_bins: bool = True,
    common_norm: bool = True,
    multiple: str = "layer",
    element: str = "bars",
    fill: bool = True,
    shrink: float = 1,
    kde: bool = False,
    kde_kws: Optional[Dict] = None,
    line_kws: Optional[Dict] = None,
    hue_order: Optional[List] = None,
    palette: Optional[Union[str, Dict, List]] = None,
    color: Optional[str] = None,
    edgecolor: Optional[str] = None,
    hatch: Optional[str] = None,
    hatch_map: Optional[Dict[str, str]] = None,
    hatch_order: Optional[List[str]] = None,
    alpha: Optional[float] = None,
    linewidth: Optional[float] = None,
    log_scale: Optional[Union[bool, float, Tuple]] = None,
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate: Union[bool, Dict, None] = None,
    ax: Optional[Axes] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    **kwargs,
) -> Axes:
    """
    Create a publication-ready histogram plot.

    Draws a univariate histogram with optional hue grouping, multiple
    rendering elements (bars, step, poly), and an optional KDE overlay.
    When BOTH ``x`` and ``y`` are passed, switches to a 2D bivariate
    heatmap-like mode (``QuadMesh`` rendered via :func:`seaborn.histplot`
    in 2D mode, with a continuous-hue colorbar routed through the
    publiplots legend reactor — mirrors :func:`pp.hexbinplot`'s
    convention). Exposes the full seaborn ``histplot`` API while
    applying publiplots' double-layer transparent-fill styling, palette
    resolution, and legend-entry pipeline.

    Parameters
    ----------
    data : DataFrame, optional
        Input data. Long-form frame containing ``x`` / ``y`` / ``hue``.
    x : str, optional
        Column name for the variable binned along the x-axis (vertical
        histogram). Pass either ``x`` alone (1D vertical), ``y`` alone
        (1D horizontal), or both (2D heatmap).
    y : str, optional
        Column name for the variable binned along the y-axis (horizontal
        histogram). See ``x`` for combined usage.
    hue : str, optional
        Column name for categorical color grouping.
    weights : str, optional
        Column name with per-observation weights; influences bin counts.
    stat : str, default="count"
        Aggregate statistic for each bin. One of ``"count"``,
        ``"frequency"``, ``"density"``, ``"probability"``, ``"percent"``.
    bins : str, int, or array-like, default="auto"
        Bin specification. Accepts every form :func:`seaborn.histplot`
        accepts (a numpy bin rule like ``"auto"``, an integer count, or
        an explicit sequence of bin edges).
    binwidth : float, optional
        Bin width in data units. Overrides ``bins``.
    binrange : (min, max), optional
        Lowest/highest data values considered. Falls back to the full
        range when omitted.
    discrete : bool, optional
        If True, center bins on integer values and default the binwidth
        to 1. Suitable for integer/categorical x.
    cumulative : bool, default=False
        Render the cumulative distribution instead of per-bin values.
    common_bins : bool, default=True
        Use the same bin edges across all hue levels.
    common_norm : bool, default=True
        When normalizing, normalize across all hue levels jointly
        (True) or per-level (False).
    multiple : {'layer', 'dodge', 'stack', 'fill'}, default='layer'
        How to handle overlapping hue levels.
    element : {'bars', 'step', 'poly'}, default='bars'
        Visual representation. ``"bars"`` draws Rectangle patches;
        ``"step"`` draws a piecewise-constant outline; ``"poly"`` draws
        a piecewise-linear outline at bin midpoints.
    fill : bool, default=True
        Fill the area under step/poly outlines. Ignored for ``"bars"``.
    shrink : float, default=1
        Bar width as a fraction of the bin width.
    kde : bool, default=False
        Overlay a kernel density estimate per hue level.
    kde_kws : dict, optional
        Extra keyword arguments forwarded to the KDE estimator (see
        :func:`seaborn.histplot`).
    line_kws : dict, optional
        Extra keyword arguments forwarded to step/poly/KDE line artists.
    hue_order : list, optional
        Order of the hue levels; determines palette assignment and
        legend order.
    palette : str, dict, or list, optional
        Color palette. Accepts a palette name (publiplots or seaborn),
        an explicit list of colors, or a ``{hue_level: color}`` dict.
    color : str, optional
        Fixed color used when ``hue`` is None.
    edgecolor : str, optional
        Edge color for the bar/outline stroke. When ``None`` (the
        default, also the default of ``rcParams["edgecolor"]``), the
        edge matches the face color at full opacity — matching
        publiplots' double-layer transparent-fill styling used by
        :func:`barplot`.
    hatch : str, optional
        Column name for a secondary categorical dimension rendered with
        hatch patterns (v1: supported only when ``multiple == "layer"``
        or when ``hue`` is None).
    hatch_map : dict, optional
        Explicit mapping from hatch-column values to matplotlib hatch
        pattern strings. See :data:`publiplots.HATCH_PATTERNS`.
    hatch_order : list, optional
        Order of hatch levels; determines pattern assignment.
    alpha : float, optional
        Transparency for the face fill (0-1). Falls back to rcParams.
    linewidth : float, optional
        Width of bar edges / step / poly / KDE line. Falls back to rcParams.
    log_scale : bool, number, or pair, optional
        If True, apply a log scale on the value axis. A number sets the
        base. A 2-tuple sets (x_log, y_log) independently — forwarded
        through to :func:`seaborn.histplot`.
    legend : bool or dict, default=True
        Legend control. ``True`` stashes and renders all legend kinds.
        ``False`` hides the legend. A dict enables/disables specific
        kinds (e.g. ``{"hue": False}``).
    legend_kws : dict, optional
        Additional kwargs for the legend builder. Supported keys include
        ``hue_label`` to override the title.
    cmap : str, optional
        2D-only. Matplotlib colormap name for the bivariate heatmap.
        When ``None`` (the default), falls back to
        ``rcParams["image.cmap"]`` (matches :func:`pp.hexbinplot`).
        Silently ignored in 1D and when ``hue`` is set in 2D (seaborn
        uses ``palette`` per hue level instead).
    vmin, vmax : float, optional
        2D-only. Color scale bounds. When both are ``None`` (the
        default), seaborn autoscales from the per-cell statistic.
        Silently ignored in 1D.
    annotate : bool or dict, optional
        If truthy, label each bar with its statistic (only supported
        for ``element="bars"``). Pass a dict to forward options to
        :func:`publiplots.annotate` (e.g. ``{"fmt": "d"}``).
    ax : Axes, optional
        Matplotlib axes to draw on. If None, a new figure is created via
        :func:`publiplots.layout.subplots`.
    title : str, default=""
        Plot title.
    xlabel : str, default=""
        X-axis label. Forwarded verbatim to :meth:`Axes.set_xlabel` when
        not ``None`` (use ``None`` to keep seaborn's inferred label).
    ylabel : str, default=""
        Y-axis label. Same semantics as ``xlabel``.
    **kwargs
        Extra keyword arguments forwarded to :func:`seaborn.histplot`.
        ``figsize`` is rejected — use ``pp.subplots(axes_size=...)``
        and pass ``ax=`` instead.

    Returns
    -------
    Axes
        The axes where the plot was drawn. Use ``ax.get_figure()`` to
        recover the figure handle.

    Examples
    --------
    Simple histogram of one variable:

    >>> ax = pp.histplot(data=df, x="value")

    Grouped histogram with KDE overlay:

    >>> ax = pp.histplot(
    ...     data=df, x="value", hue="group",
    ...     multiple="layer", kde=True, palette="pastel",
    ... )

    Step outline without fill:

    >>> ax = pp.histplot(data=df, x="value", hue="group",
    ...                  element="step", fill=False)

    2D bivariate heatmap (both ``x`` and ``y`` set):

    >>> ax = pp.histplot(data=df, x="x", y="y", cmap="magma")

    See Also
    --------
    seaborn.histplot : Underlying rendering primitive.
    publiplots.annotate : Add bar-value labels (see ``annotate=`` above).
    publiplots.hexbinplot : Hexagonal-binning bivariate density plot.
    """
    from publiplots.layout.subplots import reject_figsize
    reject_figsize(kwargs)

    if x is None and y is None:
        raise ValueError(
            "At least one of `x` or `y` must be provided to pp.histplot."
        )
    is_2d = x is not None and y is not None

    linewidth = resolve_param("lines.linewidth", linewidth)
    color = resolve_param("color", color)
    # 2D heatmap cells are solid density patches (mirrors hexbinplot):
    # alpha defaults to a literal 1.0 and edgecolor to "none". 1D paths
    # keep the existing rcParam-resolved defaults.
    if is_2d:
        alpha = alpha if alpha is not None else 1.0
        edgecolor_resolved = edgecolor if edgecolor is not None else "none"
        cmap_resolved = cmap if cmap is not None else plt.rcParams["image.cmap"]
    else:
        alpha = resolve_param("alpha", alpha)
        edgecolor_resolved = resolve_param("edgecolor", edgecolor)
        cmap_resolved = None

    if ax is None:
        from publiplots.layout.subplots import subplots as _pp_subplots
        fig, ax = _pp_subplots()
    else:
        fig = ax.get_figure()

    palette_map: Optional[Dict] = None
    if hue is not None:
        values = list(hue_order) if hue_order is not None else list(data[hue].unique())
        palette_map = resolve_palette_map(values=values, palette=palette)

    hatch_map_resolved: Optional[Dict[str, str]] = None
    if hatch is not None:
        if hue is not None and multiple == "dodge":
            raise NotImplementedError(
                "histplot: hatch= with hue + multiple='dodge' is not "
                "supported in this release. Use multiple='layer' or "
                "drop hue/hatch."
            )
        hatch_values = (
            list(hatch_order)
            if hatch_order is not None
            else list(data[hatch].unique())
        )
        hatch_map_resolved = resolve_hatch_map(
            values=hatch_values, hatch_map=hatch_map,
        )

    tracker = ArtistTracker(ax)

    if is_2d:
        # 2D bivariate histogram: seaborn renders one (or one-per-hue-level)
        # QuadMesh; we post-style it and route the legend either as a
        # continuous-hue colorbar (no hue) or via the existing 1D
        # categorical rectangle stash (hue set).
        if annotate:
            raise NotImplementedError(
                "histplot: annotate= is not supported in 2D mode "
                "(no Rectangle patches to label)."
            )
        if element != "bars":
            raise NotImplementedError(
                f"histplot: element={element!r} is not supported in 2D "
                "mode (use the default 'bars')."
            )

        sns_kwargs = {
            "data": data,
            "x": x,
            "y": y,
            "hue": hue,
            "weights": weights,
            "stat": stat,
            "bins": bins,
            "binwidth": binwidth,
            "binrange": binrange,
            "discrete": discrete,
            "cumulative": cumulative,
            "common_bins": common_bins,
            "common_norm": common_norm,
            "hue_order": hue_order,
            "log_scale": log_scale,
            "ax": ax,
            # Both routed through publiplots' legend reactor.
            "legend": False,
            "cbar": False,
        }
        if kde:
            sns_kwargs["kde"] = True
            if kde_kws:
                sns_kwargs["kde_kws"] = kde_kws
        # cmap is mutually exclusive with hue in seaborn 2D.
        if hue is None:
            sns_kwargs["cmap"] = cmap_resolved
            if vmin is not None:
                sns_kwargs["vmin"] = vmin
            if vmax is not None:
                sns_kwargs["vmax"] = vmax
        else:
            sns_kwargs["palette"] = palette_map if palette_map else palette
        sns_kwargs.update(kwargs)

        sns.histplot(**sns_kwargs)

        # Post-draw QuadMesh styling.
        new_collections = list(tracker.get_new_collections())
        meshes = [c for c in new_collections if isinstance(c, QuadMesh)]
        for mesh in meshes:
            if alpha is not None:
                mesh.set_alpha(alpha)
            if linewidth:
                mesh.set_linewidth(linewidth)
            if edgecolor_resolved not in (None, "none"):
                mesh.set_edgecolor(edgecolor_resolved)

        # Legend stash: continuous colorbar (no hue) or categorical
        # rectangles (hue).
        if hue is None:
            if legend is not False and meshes:
                flags = resolve_legend_flags(legend)
                _legend_kws = dict(legend_kws or {})
                hue_label = _legend_kws.pop("hue_label", stat)
                if flags["hue"]:
                    stash_continuous_hue(
                        ax,
                        name=hue_label,
                        palette=meshes[0].get_cmap(),
                        hue_norm=meshes[0].norm,
                    )
                render_entries(ax, flags=flags, legend_kws=_legend_kws)
        else:
            _legend(
                ax=ax,
                hue=hue,
                palette=palette_map,
                element=element,
                fill=fill,
                alpha=alpha,
                linewidth=linewidth,
                color=color,
                edgecolor=edgecolor_resolved,
                legend=legend,
                legend_kws=legend_kws,
            )
    else:
        sns_kwargs = {
            "data": data,
            "x": x,
            "y": y,
            "hue": hue,
            "weights": weights,
            "stat": stat,
            "bins": bins,
            "binwidth": binwidth,
            "binrange": binrange,
            "discrete": discrete,
            "cumulative": cumulative,
            "common_bins": common_bins,
            "common_norm": common_norm,
            "multiple": multiple,
            "element": element,
            "fill": fill,
            "shrink": shrink,
            "kde": kde,
            "kde_kws": kde_kws,
            "line_kws": line_kws,
            "palette": palette_map if palette_map else palette,
            "hue_order": hue_order,
            "log_scale": log_scale,
            "ax": ax,
            "legend": False,
        }
        if hue is None:
            sns_kwargs["color"] = color
        sns_kwargs.update(kwargs)

        sns.histplot(**sns_kwargs)

        if element == "bars":
            _paint_bars(
                patches=tracker.get_new_patches(),
                data=data,
                hue=hue,
                hue_order=hue_order,
                palette=palette_map,
                hatch_col=hatch,
                hatch_map=hatch_map_resolved,
                color=color,
                edgecolor=edgecolor_resolved,
                linewidth=linewidth,
            )
            tracker.apply_transparency(on="patches", face_alpha=alpha, edge_alpha=1.0)
        else:
            _paint_lines(
                new_lines=tracker.get_new_lines(),
                new_collections=tracker.get_new_collections(),
                palette=palette_map,
                hue_order=hue_order,
                color=color,
                edgecolor=edgecolor_resolved,
                linewidth=linewidth,
                alpha=alpha,
                fill=fill,
            )

        if kde:
            _paint_kde(
                new_lines=tracker.get_new_lines(),
                palette=palette_map,
                hue_order=hue_order,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )

        _legend(
            ax=ax,
            hue=hue,
            palette=palette_map,
            element=element,
            fill=fill,
            alpha=alpha,
            linewidth=linewidth,
            color=color,
            edgecolor=edgecolor_resolved,
            legend=legend,
            legend_kws=legend_kws,
        )

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if annotate:
        if element != "bars":
            raise NotImplementedError(
                "histplot: annotate= is only supported for element='bars' "
                f"in this release (got element={element!r})."
            )
        ax._publiplots_bar_meta = build_from_histplot_call(
            ax=ax, data=data, x=x, y=y, hue=hue,
            palette=palette_map, stat=stat, hue_order=hue_order,
        )
        from publiplots.annotate import annotate as _annotate_fn
        opts = annotate if isinstance(annotate, dict) else {}
        _annotate_fn(ax, kind="bar_values", **opts)

    return ax


# =============================================================================
# Helper Functions
# =============================================================================


def _hue_level_for_patch(
    patch: Rectangle,
    palette: Optional[Dict],
    color: Optional[str],
) -> Optional[str]:
    """Return the hue level whose palette color matches the patch's face."""
    if not palette:
        return None
    face = tuple(patch.get_facecolor())
    face_rgb = face[:3]
    best = None
    best_dist = float("inf")
    for level, col in palette.items():
        target = to_rgba(col)[:3]
        dist = sum((a - b) ** 2 for a, b in zip(face_rgb, target))
        if dist < best_dist:
            best_dist = dist
            best = level
    return best


def _paint_bars(
    patches: List,
    data: pd.DataFrame,
    hue: Optional[str],
    hue_order: Optional[List],
    palette: Optional[Dict],
    hatch_col: Optional[str],
    hatch_map: Optional[Dict[str, str]],
    color: Optional[str],
    edgecolor: Optional[str],
    linewidth: float,
) -> None:
    """Paint face, edge, hatch, and linewidth on each histogram bar.

    Hue assignment is inferred from the Rectangle's post-draw facecolor
    against the resolved palette — seaborn draws all patches of one hue
    level as a contiguous block, but ``multiple="dodge"`` etc. interleave
    per bin, so color-matching is the only ordering-agnostic inference.

    When ``hatch_col`` is given but ``hue`` is None, every bar carries a
    single hatch (v1 limitation; combined hue + hatch on histplot is
    deferred to a follow-up release).
    """
    bar_patches = [p for p in patches if isinstance(p, Rectangle)]

    single_hatch: Optional[str] = None
    if hatch_col is not None and hatch_map and hue is None:
        levels = list(hatch_map.keys())
        if levels:
            single_hatch = hatch_map[levels[0]]

    for patch in bar_patches:
        if palette is not None:
            level = _hue_level_for_patch(patch, palette, color)
            face_color = palette.get(level, color) if level is not None else color
        else:
            face_color = color

        patch.set_facecolor(face_color)
        patch.set_edgecolor(edgecolor if edgecolor is not None else face_color)
        patch.set_linewidth(linewidth)

        if hatch_col is not None and hatch_map is not None:
            if single_hatch is not None:
                patch.set_hatch(single_hatch)
            elif palette is not None:
                level = _hue_level_for_patch(patch, palette, color)
                if level is not None and level in hatch_map:
                    patch.set_hatch(hatch_map[level])
            patch.set_hatch_linewidth(linewidth)


def _match_collection_to_level(
    collection: PolyCollection,
    palette: Dict,
) -> Optional[str]:
    """Identify the hue level whose palette entry matches the collection face."""
    fc = collection.get_facecolor()
    if len(fc) == 0:
        ec = collection.get_edgecolor()
        if len(ec) == 0:
            return None
        sample = ec[0][:3]
    else:
        sample = fc[0][:3]
    best = None
    best_dist = float("inf")
    for level, col in palette.items():
        target = to_rgba(col)[:3]
        dist = sum((a - b) ** 2 for a, b in zip(sample, target))
        if dist < best_dist:
            best_dist = dist
            best = level
    return best


def _match_line_to_level(
    line: Line2D,
    palette: Dict,
) -> Optional[str]:
    """Identify the hue level whose palette entry matches the line's color."""
    sample = to_rgba(line.get_color())[:3]
    best = None
    best_dist = float("inf")
    for level, col in palette.items():
        target = to_rgba(col)[:3]
        dist = sum((a - b) ** 2 for a, b in zip(sample, target))
        if dist < best_dist:
            best_dist = dist
            best = level
    return best


def _paint_lines(
    new_lines: List[Line2D],
    new_collections: List,
    palette: Optional[Dict],
    hue_order: Optional[List],
    color: Optional[str],
    edgecolor: Optional[str],
    linewidth: float,
    alpha: float,
    fill: bool,
) -> None:
    """Style step / poly element artists.

    With ``fill=False`` seaborn draws one ``Line2D`` per hue level into
    ``ax.lines``; with ``fill=True`` it draws one ``FillBetweenPolyCollection``
    per hue level into ``ax.collections``. We colour them from the
    resolved palette and apply the publiplots alpha split (transparent
    fill, opaque edge).
    """
    for line in new_lines:
        if palette:
            level = _match_line_to_level(line, palette)
            line_color = palette.get(level, color) if level is not None else color
        else:
            line_color = color
        line.set_color(to_rgba(line_color, alpha=1.0))
        line.set_linewidth(linewidth)

    if not fill:
        return

    for collection in new_collections:
        if not isinstance(collection, PolyCollection):
            continue
        if palette:
            level = _match_collection_to_level(collection, palette)
            face_color = palette.get(level, color) if level is not None else color
        else:
            face_color = color
        edge = edgecolor if edgecolor is not None else face_color
        collection.set_facecolor(to_rgba(face_color, alpha=alpha))
        collection.set_edgecolor(to_rgba(edge, alpha=1.0))
        collection.set_linewidth(linewidth)


def _paint_kde(
    new_lines: List[Line2D],
    palette: Optional[Dict],
    hue_order: Optional[List],
    color: Optional[str],
    linewidth: float,
    alpha: float,
) -> None:
    """Style KDE overlay lines with a slightly bolder stroke.

    KDE lines live in ``ax.lines`` alongside any step/poly lines; we
    identify them by re-matching against the palette. When ``fill=True``
    on step/poly there are no competing lines, so any ``Line2D`` seaborn
    emitted is a KDE curve.
    """
    bold_lw = max(linewidth + 0.5, 1.5)
    for line in new_lines:
        if palette:
            level = _match_line_to_level(line, palette)
            line_color = palette.get(level, color) if level is not None else color
        else:
            line_color = color
        current_lw = line.get_linewidth()
        if current_lw < bold_lw:
            line.set_linewidth(bold_lw)
        line.set_color(to_rgba(line_color, alpha=1.0))


def _legend(
    ax: Axes,
    hue: Optional[str],
    palette: Optional[Dict],
    element: str,
    fill: bool,
    alpha: float,
    linewidth: float,
    color: Optional[str],
    edgecolor: Optional[str],
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
) -> None:
    """Stash a single hue LegendEntry (if applicable) and render.

    Handle shape follows the drawn geometry: rectangles for bars or for
    any filled step/poly, lines for unfilled step/poly outlines.
    """
    if legend is False or hue is None or not palette:
        if legend is not False:
            render_entries(
                ax,
                flags=resolve_legend_flags(legend),
                legend_kws=dict(legend_kws or {}),
            )
        return

    flags = resolve_legend_flags(legend)
    legend_kws = dict(legend_kws or {})
    hue_label = legend_kws.pop("hue_label", hue)

    if flags["hue"]:
        labels = list(palette.keys())
        colors = [palette[v] for v in labels]
        edgecolors = [edgecolor] * len(labels) if edgecolor else None
        if element == "bars" or fill:
            handles = create_legend_handles(
                labels=labels,
                colors=colors,
                edgecolors=edgecolors,
                alpha=alpha,
                linewidth=linewidth,
                style="rectangle",
            )
        else:
            handles = create_legend_handles(
                labels=labels,
                colors=colors,
                edgecolors=edgecolors,
                linestyles=["-"] * len(labels),
                alpha=alpha,
                linewidth=linewidth,
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
