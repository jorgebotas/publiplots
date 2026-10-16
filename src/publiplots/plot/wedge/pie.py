"""Pie and donut charts."""
from __future__ import annotations

import math
import numbers
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from matplotlib.axes import Axes
from matplotlib import colormaps
from matplotlib.colors import Colormap, Normalize, to_rgba

from publiplots.annotate import annotate as _annotate
from publiplots.plot.wedge.colour import derive_colors, resolve_hue_values
from publiplots.plot.wedge.draw import (
    apply_cell_limits,
    apply_gap,
    build_wedge_meta,
    draw_wedges,
    set_hatch_linewidth,
)
from publiplots.plot.wedge.geometry import (
    WedgeNode,
    build_nodes,
    layout,
    total_value,
)
from publiplots.themes.colors import resolve_continuous_cmap, resolve_palette_map
from publiplots.themes.hatches import resolve_hatch_map
from publiplots.themes.rcparams import resolve_param
from publiplots.utils import create_legend_handles
from publiplots.utils.legend_entries import (
    LegendEntry,
    resolve_legend_flags,
    stash_entry,
)
from publiplots.utils.plot_legend import render_entries, stash_continuous_hue

OTHER_LABEL = "Other"

# Neutral grey for the min_share bucket. "Other" is not a category of the
# data, so it must not consume a palette slot that a real category would
# otherwise have read as meaningful.
OTHER_COLOR = "#b4b4b4"


def _bucket_small(
    frame: pd.DataFrame,
    level: str,
    value: Optional[str],
    min_share: float,
) -> pd.DataFrame:
    """Merge categories below ``min_share`` of the total into one row.

    ``value=None`` counts rows, matching what ``build_nodes`` does with
    an omitted value column; grouping on a missing column would raise.
    """
    if not min_share:
        return frame
    grouped = frame.groupby(level, observed=True)
    sums = (grouped.size().astype(float) if value is None
            else grouped[value].sum())
    total = float(sums.sum())
    if total <= 0:
        return frame
    small = [k for k, v in sums.items() if (v / total) < min_share]
    # Bucketing one category renames it without merging anything, which
    # loses its label for no gain.
    if len(small) < 2:
        return frame
    out = frame.copy()
    column = out[level]
    if isinstance(column.dtype, pd.CategoricalDtype):
        # Casting a Categorical to object here would throw away its
        # declared category order, which build_nodes reads as the draw
        # order -- min_share would silently reorder the pie away from
        # the bar panel drawn from the same frame, which is the exact
        # correspondence #268 exists for. Append the bucket to the
        # categories instead and leave the rest where they were.
        if OTHER_LABEL not in column.cat.categories:
            column = column.cat.add_categories([OTHER_LABEL])
        out[level] = column
        out.loc[out[level].isin(small), level] = OTHER_LABEL
        out[level] = out[level].cat.remove_unused_categories()
        return out
    out[level] = column.astype(object)
    out.loc[out[level].isin(small), level] = OTHER_LABEL
    return out


def _numeric_hue_cmap(
    palette: Union[str, Colormap, Dict, List, None],
    *,
    color: Any,
    hue: str,
) -> Colormap:
    """Resolve ``palette=`` to the colormap a numeric hue maps through.

    The semantics are ``scatterplot``'s: a matplotlib colormap name or a
    :class:`~matplotlib.colors.Colormap` is used as given, and ``None``
    falls back to the light ramp anchored on ``color``. A name has to
    become a Colormap here, because the wedge colours are read by calling
    it (``cmap(norm(v))``), which a string cannot do.

    A dict or list assigns colours to discrete levels, which a numeric
    hue does not have. ``scatterplot`` raises ``TypeError`` on both too,
    from its dependencies (matplotlib's colormap lookup for a dict,
    seaborn's hue mapping for a list), with a message that names neither
    the parameter nor the cause; this raises the same type with one that
    does.
    """
    if palette is None:
        return resolve_continuous_cmap(None, color=color)
    if isinstance(palette, Colormap):
        return palette
    if isinstance(palette, (dict, list, tuple)):
        raise TypeError(
            f"pieplot: hue={hue!r} is numeric, so it maps through a "
            f"colormap and palette= must be a colormap name or a "
            f"matplotlib Colormap; got a {type(palette).__name__}. A "
            f"{type(palette).__name__} assigns colours to discrete levels "
            f"-- convert the column to strings or a Categorical to colour "
            f"it categorically."
        )
    if isinstance(palette, str):
        try:
            return colormaps[palette]
        except KeyError:
            raise ValueError(
                f"pieplot: hue={hue!r} is numeric, so palette= must name a "
                f"registered matplotlib colormap (e.g. 'viridis'); "
                f"{palette!r} is not one."
            ) from None
    raise TypeError(
        f"pieplot: palette= must be a colormap name, a matplotlib "
        f"Colormap or None for a numeric hue; got "
        f"{type(palette).__name__}."
    )


def pieplot(
    data: pd.DataFrame,
    x: Union[str, Sequence[str]],
    y: Optional[str] = None,
    hue: Optional[str] = None,
    *,
    ax: Optional[Axes] = None,
    donut: float = 0.0,
    start_angle: float = 90.0,
    direction: str = "clockwise",
    order: Optional[Sequence] = None,
    hue_order: Optional[Sequence] = None,
    sort: bool = False,
    pull: Union[float, Dict[Any, float], None] = None,
    gap: Optional[float] = None,
    min_share: float = 0.0,
    center_label: Union[str, Callable[[float], str], None] = None,
    aspect: str = "cell",
    palette: Optional[Union[str, Dict, List, Colormap]] = None,
    hatch: Optional[str] = None,
    hatch_map: Optional[Dict[str, str]] = None,
    hatch_order: Optional[List[str]] = None,
    color: Optional[str] = None,
    edgecolor: Optional[str] = None,
    linewidth: Optional[float] = None,
    alpha: Optional[float] = None,
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    annotate: Union[bool, Dict, None] = None,
    title: str = "",
    **kwargs,
) -> Axes:
    """Draw a pie or donut chart.

    A pie is the one-level case of a nested part-of-whole chart. ``x``
    names the category column, ``y`` the value (omitted, rows are
    counted), and ``hue`` what determines colour -- defaulting to ``x``
    itself, which is what every pie does. Only one level is supported:
    the machinery underneath is written for a tree, but this function
    draws a single ring.

    Parameters
    ----------
    data : pandas.DataFrame
        Tidy input. One row per observation, or one row per category when
        pre-aggregated.
    x : str
        Column naming each wedge. A sequence of length 1 is accepted;
        anything longer raises, because only one ring is drawn. Rows
        whose ``x`` is ``NaN`` are dropped and the remaining wedges
        renormalise to fill the circle, silently -- a pie cannot show a
        missing category, so drop or fill them yourself if the share
        they represent matters.
    y : str, optional
        Value column, summed within each category. When omitted, rows are
        counted.
    hue : str, optional
        Column determining colour. Defaults to ``x``. A categorical hue
        must be constant within a wedge.
    ax : Axes, optional
        Matplotlib axes to draw on. If ``None``, creates a new figure
        via :func:`publiplots.layout.subplots` so the publiplots layout
        manager can keep geometry stable. To control axes dimensions,
        pre-create with ``pp.subplots(axes_size=(w_mm, h_mm))`` and
        pass ``ax=``.
    donut : float, default 0.0
        Inner radius as a fraction of the outer radius. ``0.5`` leaves a
        hole of half the radius, which is where ``center_label`` goes.
    start_angle : float, default 90.0
        Angle of the first wedge's leading edge, in degrees. 90 is twelve
        o'clock.
    direction : {'clockwise', 'counterclockwise'}, default 'clockwise'
    order : sequence, optional
        Explicit wedge order. Categories absent from the data are dropped;
        categories absent from ``order`` are **filtered out, and the
        wedges that remain renormalise to a full circle** -- the chart
        then reads as 100% of a whole it is not showing all of. A
        :class:`UserWarning` names the dropped categories when that
        happens. Cannot be combined with ``min_share``, whose
        ``"Other"`` bucket is not a category of the data and so cannot
        be named in an order.
    hue_order : sequence, optional
        Running order of the legend's colour entries. Levels omitted from
        it keep their colour and are appended after the listed ones --
        ``hue_order`` reorders the key, it never decides which wedges are
        drawn. To drop categories, use ``order``.
    sort : bool, default False
        Order wedges by descending value. Cannot be combined with
        ``order``. Off by default so a pie keeps the category order of a
        bar panel drawn from the same frame.
    pull : float or dict, optional
        Radial offset as a fraction of the radius: a float for every
        wedge, or a dict keyed by category. Deliberately not a positional
        sequence. Each wedge moves along its own mid-ray, as in
        matplotlib's ``explode``, so the sideways gap it opens to a
        neighbour is ``pull * sin(half-span)``: a float pulls a wide
        wedge well clear of its neighbours and a 5% sliver barely at
        all. Use a dict to pop out one or two wedges, and ``gap`` to
        separate all of them evenly.
    gap : float, optional
        Gap between every pair of neighbouring wedges, in millimetres.
        Each wedge's two straight edges are inset by ``gap / 2``
        parallel to themselves, so the band between two neighbours is
        ``gap`` wide along its whole length, for wide and thin wedges
        alike, while the arcs keep their radii. The width is measured
        between the outlines' centre lines, so the unpainted band is
        ``gap`` less one outline width (``linewidth`` points). A wedge
        too thin to keep any shape after the inset draws nothing, but
        keeps its record, its label and its legend entry. Composes with
        ``pull``, which then moves each inset wedge as a whole. Defaults
        to ``rcParams["pie.gap"]`` (``0.0``, no gap).
    min_share : float, default 0.0
        Categories **strictly below** this fraction of the total merge
        into a single ``"Other"`` wedge; a category sitting exactly on the
        threshold is kept. Applies only when at least two categories fall
        below it. Cannot be combined with ``order``.
    center_label : str or callable, optional
        Text for the donut hole. A callable receives the total. Ignored
        when ``donut=0``: a full pie has no hole to write in.
    aspect : {'cell', 'equal'}, default 'cell'
        ``'cell'`` matches the data limits to the axes' millimetre
        ratio: the disc is sized to the shorter side of the cell and the
        longer side's limits are stretched to match, so the axes box
        keeps the whole allocation on both axes and the surplus stays
        free for outside labels. ``'equal'`` uses matplotlib's aspect
        constraint, which survives an interactive resize but shrinks the
        drawn box to a square, keeping only ``min(w, h) / max(w, h)`` of
        the allocation on whichever axis is longer. Both render a disc
        round to within a rendering pixel; measured ink ratios across
        70x50, 90x40, 40x70, 30x80 and 60x60mm cells run 0.9942 to
        1.0044.
    palette : str, dict, list, or Colormap, optional
        Colour palette. For a categorical ``hue``, can be:

        - ``str``: seaborn palette name or publiplots palette name
        - ``dict``: explicit ``{hue_level: color}`` mapping, overlaid on
          the resolved defaults, so a partial dict recolours only the
          levels it names
        - ``list``: list of colours assigned in hue-level order

        For a numeric ``hue``: a matplotlib colormap name
        (``"viridis"``) or a :class:`~matplotlib.colors.Colormap`, used
        for the wedges and the colorbar alike, as in ``scatterplot``; a
        dict or list raises ``TypeError``, as it does there, since a
        numeric hue has no discrete levels to assign colours to. Left as
        ``None``, ``pieplot`` builds its own light ramp running from
        near-white up to ``color`` -- unlike ``scatterplot``, whose
        ``None`` gives seaborn's default ramp whatever ``color`` is.
    hatch : str, optional
        Column name driving a **second categorical dimension** encoded
        via hatch textures -- the texture-based analogue of ``hue``.
        Independent of ``hue``: pass the same column to both
        (``hatch=x``) to pattern each wedge by its own colour level, or
        a different column for a second key. Must be constant within a
        wedge, as ``hue`` must.

        Patterns are drawn from :data:`publiplots.HATCH_PATTERNS` by
        default; density is controlled globally via
        :func:`publiplots.set_hatch_mode` or overridden per-plot with
        ``hatch_map``.
    hatch_map : dict, optional
        Mapping from hatch-column values to matplotlib hatch-pattern
        strings, overriding the per-``hatch_mode`` defaults, e.g.
        ``hatch_map={"control": "", "treated": "///"}``.
    hatch_order : list, optional
        Running order of the legend's hatch entries. Like ``hue_order``
        it reorders the key and never decides which wedges are drawn;
        levels omitted from it are appended after the listed ones.
    color : str, optional
        Fixed colour used where no hue level supplies one: the colormap
        anchor for a continuous ``hue`` when ``palette`` is ``None``, and
        the swatch colour for a
        standalone hatch key. Falls back to ``rcParams["color"]``.
    edgecolor : str, optional
        Explicit edge colour override. When ``None`` (the default, also
        the default of ``rcParams["edgecolor"]``), each wedge's edge
        matches its own face colour -- publiplots' double-layer
        transparent-fill styling. Set to a specific colour (``"black"``)
        to outline every wedge the same way regardless of fill.
    linewidth : float, optional
        Width of the wedge outlines and hatch strokes. Defaults to
        ``rcParams["edgewidth"]``.
    alpha : float, optional
        Transparency of the wedge *fill* (outlines stay fully opaque).
        ``0`` = outlined wedges only; ``1`` = solid fill. Defaults to
        ``rcParams["alpha"]``.
    legend : bool or dict, default=True
        Whether to render the legend. Accepts ``bool`` or
        ``dict[kind, bool]`` for per-kind control (e.g.
        ``legend={"hatch": False}`` hides just the hatch entry).
        Unlike ``barplot``, ``False`` suppresses only the rendering:
        entries are stashed either way, so a later figure-level
        :func:`publiplots.legend` group can still claim them.
    legend_kws : dict, optional
        Extra kwargs forwarded to the legend builder. Notable keys:
        ``inside=True`` with ``loc=...`` places the legend inside the
        axes using matplotlib's corner-based placement; ``hue_label`` /
        ``hatch_label`` override the title shown above each entry.
    annotate : bool or dict, optional
        If truthy, label each wedge with its share of the total as a
        percentage. Pass a ``dict`` to forward options to
        :func:`publiplots.annotate`: ``fmt`` (a format spec labels the
        raw value instead of the share), ``anchor``
        (``"auto"``/``"inside"``/``"outside"``), ``offset`` (mm),
        ``color``, ``rotation`` (degrees, or ``"tangential"`` /
        ``"radial"``), ``basis`` (``"total"``/``"parent"``).
    title : str or None, default=""
        Plot title. Pass ``None`` to leave the existing title unchanged.

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    ``hue`` means colour here and nothing else. In ``barplot`` it also
    controls how marks group, because grouping has no other parameter; in
    a wedge chart the structure lives in ``x``, so the two are separate.

    Examples
    --------
    >>> ax = pp.pieplot(data=df, x="kind", y="n")
    >>> ax = pp.pieplot(data=df, x="kind", y="n", donut=0.55,
    ...                 center_label="{:,.0f}".format)
    """
    from publiplots.layout.subplots import reject_figsize
    reject_figsize(kwargs)
    if kwargs:
        # Nothing is forwarded on from here, so an unconsumed keyword is a
        # typo that would otherwise be swallowed in silence.
        raise TypeError(
            "pieplot() got an unexpected keyword argument "
            + ", ".join(repr(k) for k in sorted(kwargs, key=str))
        )

    levels = [x] if isinstance(x, str) else list(x)
    if len(levels) > 1:
        raise ValueError(
            "pieplot: x= must name a single column; it draws one ring, "
            "and a nested composition is not supported."
        )
    if order is not None and min_share:
        # build_nodes filters against `order`, and the bucket label is not
        # a category of the data, so every category folded into it would
        # be dropped -- silently renormalising the pie to a false 100%.
        # Same shape as geometry.build_nodes' sort/order refusal.
        raise ValueError(
            f"pieplot: min_share= cannot be combined with order= -- the "
            f"{OTHER_LABEL!r} bucket min_share creates is not a category "
            f"of the data, so it cannot be named in an explicit order and "
            f"every category folded into it would be dropped from the "
            f"chart. Pass one or the other."
        )

    gap = resolve_param("pie.gap", gap)
    if isinstance(gap, bool) or not isinstance(gap, numbers.Real):
        raise TypeError(
            f"pieplot: gap= must be a number of millimetres; got "
            f"{type(gap).__name__} {gap!r}."
        )
    gap = float(gap)
    if not math.isfinite(gap) or gap < 0:
        raise ValueError(
            f"pieplot: gap= must be a finite, non-negative number of "
            f"millimetres; got {gap!r}."
        )

    hue_column = hue if hue is not None else levels[0]

    linewidth = resolve_param("edgewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    color = resolve_param("color", color)
    edgecolor = resolve_param("edgecolor", edgecolor)

    if ax is None:
        from publiplots.layout.subplots import subplots as _pp_subplots
        _, ax = _pp_subplots()

    frame = _bucket_small(data, levels[0], y, min_share) if min_share else data
    order_map = None
    if order is not None:
        order_map = {levels[0]: list(order)}
        listed = set(order_map[levels[0]])
        dropped = [k for k in dict.fromkeys(frame[levels[0]].dropna().tolist())
                   if k not in listed]
        if dropped:
            # The same hazard the min_share refusal above cites: the
            # wedges that survive renormalise to a full circle, so the
            # chart reads as 100% of a whole it is not showing. Dropping
            # a category is a legitimate thing to want, so this warns
            # rather than raising -- but it does not do it silently,
            # because annotate= would print the renormalised share.
            warnings.warn(
                f"pieplot: order= omits {len(dropped)} categor"
                f"{'y' if len(dropped) == 1 else 'ies'} present in the "
                f"data: {dropped!r}. They are dropped and the remaining "
                f"wedges renormalise to a full circle, so the chart -- "
                f"and any annotate= percentage -- reads as 100% of a "
                f"whole it is not showing all of.",
                UserWarning,
                stacklevel=2,
            )

    nodes = build_nodes(frame, levels, y, order=order_map, sort=sort)
    if min_share and any(n.path[0] == OTHER_LABEL for n in nodes):
        # The bucket sorts last whatever the resolver decided, then the
        # sibling indices are renumbered so the colour derivation sees a
        # consistent run. (Inert on one ring, but wrong indices here would
        # become wrong colours the moment a nested ring reuses this path.)
        ordered = ([n for n in nodes if n.path[0] != OTHER_LABEL]
                   + [n for n in nodes if n.path[0] == OTHER_LABEL])
        nodes = [
            WedgeNode(path=n.path, value=n.value, depth=n.depth,
                      parent=n.parent, sibling_index=i,
                      sibling_count=len(ordered))
            for i, n in enumerate(ordered)
        ]
    total = total_value(nodes)
    places = layout(nodes, start_angle=start_angle, direction=direction,
                    donut=donut, n_levels=1)

    hue_values, hue_continuous = resolve_hue_values(
        frame, levels, hue, nodes, param_name="hue",
        bucket_label=OTHER_LABEL if min_share else None)
    hue_levels = list(dict.fromkeys(
        hue_values[n.path] for n in nodes if n.path in hue_values))
    hue_norm = None
    # Resolved once, here, and read by both the wedge colours and the
    # colorbar below, so the two cannot drift onto different colormaps.
    # Resolved before the all-NaN check so a bad palette= is reported
    # whatever the data holds.
    hue_cmap = (_numeric_hue_cmap(palette, color=color, hue=hue_column)
                if hue_continuous else None)
    if hue_continuous and not hue_levels:
        # Every value in the column was NaN, so no wedge resolved one.
        # min()/max() over that raises "min() iterable argument is
        # empty", which names neither the column nor the cause. The
        # categorical path degrades to the fallback grey instead, so do
        # the same here and say which column it was.
        warnings.warn(
            f"pieplot: hue={hue_column!r} is numeric but has no non-NaN "
            f"value in any wedge, so there is no range to build a "
            f"colormap over. The wedges fall back to a neutral grey and "
            f"no colorbar is drawn.",
            UserWarning,
            stacklevel=2,
        )
        palette_map = {}
    elif hue_continuous:
        # A numeric hue maps through a colormap rather than a palette. Build
        # a value -> colour dict so derive_colors' lookup is unchanged.
        lo, hi = min(hue_levels), max(hue_levels)
        hue_norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1.0)
        palette_map = {v: hue_cmap(hue_norm(v)) for v in hue_levels}
    else:
        # resolve_palette_map returns a dict palette VERBATIM, so a partial
        # dict would leave every unnamed level without a colour. Resolve
        # the defaults first, then overlay the caller's entries, so
        # palette={"Other": "#123456"} recolours one wedge instead of
        # greying the rest.
        if isinstance(palette, dict):
            palette_map = {**resolve_palette_map(hue_levels, None), **palette}
        else:
            palette_map = resolve_palette_map(hue_levels, palette)
        named_other = isinstance(palette, dict) and OTHER_LABEL in palette
        if min_share and not named_other:
            palette_map = {**palette_map, OTHER_LABEL: OTHER_COLOR}
        if hue_order is not None:
            # Reorder, never narrow. barplot narrows the level list here
            # (bar.py:454) and gets away with it because it also filters
            # the drawn data, so an omitted level draws nothing. Every
            # wedge is drawn here, so narrowing before the palette is
            # resolved would leave the omitted ones with no entry and
            # derive_colors would paint them its grey fallback.
            listed = [v for v in hue_order if v in hue_levels]
            seen = set(listed)
            hue_levels = listed + [v for v in hue_levels if v not in seen]

    colors = derive_colors(nodes, hue_values, palette_map,
                           depth_range=0.0, sibling_range=0.0, n_levels=1,
                           hue_column=hue_column)

    # Hatch is resolved from its own column, the same way hue is, rather
    # than being keyed off the hue value -- the two are independent
    # parameters and only coincide in the common hatch=x case.
    # A continuous hue renders as a colorbar, which folds in no
    # patterns -- so even when hatch names that same column the hatch
    # needs a key of its own.
    hatch_shares_hue = (hatch is not None and hatch == hue_column
                        and not hue_continuous)
    if hatch is not None:
        hatch_values, _ = resolve_hue_values(
            frame, levels, hatch, nodes, param_name="hatch",
            bucket_label=OTHER_LABEL if min_share else None)
        hatch_levels = list(dict.fromkeys(
            hatch_values[n.path] for n in nodes if n.path in hatch_values))
        if hatch_order is not None:
            listed = [v for v in hatch_order if v in hatch_levels]
            seen = set(listed)
            hatch_levels = listed + [v for v in hatch_levels if v not in seen]
        hatch_map_resolved = resolve_hatch_map(values=hatch_levels,
                                               hatch_map=hatch_map)
        hatches = {n.path: hatch_map_resolved.get(hatch_values.get(n.path))
                   for n in nodes}
    else:
        hatch_levels = []
        hatch_map_resolved = {}
        hatches = {n.path: None for n in nodes}

    if pull is None:
        pull_map = {}
    elif isinstance(pull, dict):
        pull_map = {n.path: float(pull.get(n.path[0], 0.0)) for n in nodes}
    else:
        pull_map = {n.path: float(pull) for n in nodes}

    patches = draw_wedges(
        ax, places, colors, hatches, center=(0.0, 0.0), pull=pull_map,
        alpha=alpha, edgecolor=edgecolor, linewidth=linewidth,
    )
    _split_face_and_edge_alpha(patches, alpha=alpha, edgecolor=edgecolor)

    ax._publiplots_wedge_meta = build_wedge_meta(
        places, patches, colors, hatches, pull_map,
        total=total, donut=donut, center=(0.0, 0.0), levels=levels,
        hue_active=True, hue_continuous=hue_continuous, source_frame=data,
    )

    max_pull = max(pull_map.values(), default=0.0)
    apply_cell_limits(ax, radius=1.0 + max_pull + 0.05, aspect=aspect)
    # Only once the limits are framed does a data unit have a fixed size
    # in millimetres to convert the gap with.
    apply_gap(ax, patches, gap)

    if donut and center_label is not None:
        text = center_label(total) if callable(center_label) else center_label
        ax.text(0.0, 0.0, text, ha="center", va="center")

    # `is not None`, the house form (bar.py:639, box.py): title="" must
    # CLEAR a pre-existing title, and only title=None leaves it alone.
    if title is not None:
        ax.set_title(title)

    # Entries are stashed whether or not this call renders them: legend=
    # False still has to leave a deferred pp.legend(ax) something to find
    # (#268). barplot gates its stash on the flags; a pie deliberately
    # does not.
    flags = resolve_legend_flags(legend)
    if hue_continuous and hue_levels:
        stash_continuous_hue(
            ax,
            name=hue_column,
            palette=hue_cmap,
            hue_norm=hue_norm,
        )
    elif hue_levels:
        handles = create_legend_handles(
            labels=[str(v) for v in hue_levels],
            colors=[palette_map[v] for v in hue_levels],
            edgecolors=[edgecolor] * len(hue_levels) if edgecolor else None,
            # Only fold the patterns into the colour swatches when the two
            # columns coincide; otherwise the lookup is in the wrong key
            # space and the hatch gets an entry of its own below.
            hatches=([hatch_map_resolved.get(v) for v in hue_levels]
                     if hatch_shares_hue else None),
            alpha=alpha,
            linewidth=linewidth,
        )
        # The swatch carries the width the wedge's hatch is drawn with.
        for handle in handles:
            set_hatch_linewidth(handle, linewidth)
        stash_entry(ax, LegendEntry.build(
            name=hue_column,
            kind="hue",
            handles=handles,
            labels=[str(v) for v in hue_levels],
        ))

    if hatch is not None and not hatch_shares_hue and hatch_levels:
        # An independent hatch column needs its own key, keyed in its own
        # value space -- the same second-entry shape barplot uses at
        # bar.py:930. Neutral swatches, because the colour already means
        # the hue.
        hatch_color = "gray" if hue_levels else color
        hatch_handles = create_legend_handles(
            labels=[str(v) for v in hatch_levels],
            colors=[hatch_color] * len(hatch_levels),
            edgecolors=[edgecolor] * len(hatch_levels) if edgecolor else None,
            hatches=[hatch_map_resolved.get(v) for v in hatch_levels],
            alpha=alpha,
            linewidth=linewidth,
        )
        for handle in hatch_handles:
            set_hatch_linewidth(handle, linewidth)
        stash_entry(ax, LegendEntry.build(
            name=hatch,
            kind="hatch",
            handles=hatch_handles,
            labels=[str(v) for v in hatch_levels],
        ))

    if hue_levels or hatch_levels:
        render_entries(ax, flags=flags, legend_kws=legend_kws)

    if annotate:
        kws = annotate if isinstance(annotate, dict) else {}
        _annotate(ax, "wedge_values", **kws)

    return ax


def _split_face_and_edge_alpha(
    patches: Sequence[Any],
    *,
    alpha: float,
    edgecolor: Optional[Any],
) -> None:
    """Tint the fill and keep the outline opaque.

    ``draw_wedges`` takes a single ``alpha`` and matplotlib composites a
    patch-level alpha into face *and* edge alike, which at the themed
    ``rcParams["alpha"]`` of 0.1 leaves the whole chart nearly invisible.
    The house style -- ``barplot`` and ``histplot`` both do this, via
    ``apply_transparency(face_alpha=alpha, edge_alpha=1.0)`` -- is a
    tinted fill behind a full-strength outline, and an ``edgecolor`` of
    ``None`` means "match the face" rather than "draw nothing".
    """
    for patch in patches:
        face: Tuple[float, ...] = tuple(patch.get_facecolor())
        edge = edgecolor if edgecolor is not None else face
        # Clear the patch-level alpha first: set_facecolor/set_edgecolor
        # composite ``self._alpha`` over whatever they are handed.
        patch.set_alpha(None)
        patch.set_facecolor(to_rgba(face, alpha=alpha))
        patch.set_edgecolor(to_rgba(edge, alpha=1.0))
