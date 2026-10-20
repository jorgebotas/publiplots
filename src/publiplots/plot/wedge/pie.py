"""Pie and donut charts."""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from publiplots.annotate import annotate as _annotate
from publiplots.annotate._wedge import check_length
from publiplots.plot.wedge.render import render_wedge_chart
from publiplots.themes.rcparams import resolve_param

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
        How far wedges move out from the centre, as a fraction of the
        outer radius. Deliberately not a positional sequence.

        A **dict** keyed by category pops out just the wedges it names:
        each is translated along its own mid-ray by ``pull`` times the
        radius, as in matplotlib's ``explode``, keeping its shape. A key
        that names no wedge raises ``ValueError`` (a category merged into
        ``"Other"`` by ``min_share`` is no longer a wedge); a value must
        be a finite number of at least 0, else ``TypeError`` for a bool or
        a non-number and ``ValueError`` otherwise.

        A **float** explodes the whole chart evenly. With ``t`` that
        fraction of the radius, no wedge is translated: every wedge's
        inner boundary moves out to ``r_inner + t`` and its rim to
        ``r_outer + t``, and both its straight edges are inset so that
        every pair of neighbours is the same perpendicular distance
        ``2 * t`` apart, wide and thin wedges alike. Everything moves
        out by ``t`` and apart by ``2 * t``. A pie opens a central hole
        of radius ``t`` (a donut's hole grows by ``t``), and labels
        follow the drawn radii. The price is shape: a translation along
        each wedge's own mid-ray would keep it, but would open only
        ``t * sin(half-span)`` per side, so a thin wedge barely
        separates. Even for the two halves of a half-disc, where the
        edges and the gap match a translation exactly, the rim is an
        arc of radius ``r_outer + t`` about the chart's centre, and so
        reaches up to about ``t`` further than a translated half's at
        its ends. A wedge too narrow at its rim for the full inset is
        capped rather than dropped, as for ``gap``. A lone category's
        360-degree
        wedge has no edges to inset and draws as a full ring between
        ``r_inner + t`` and ``r_outer + t``. ``0`` draws exactly what
        no ``pull`` does; a negative or non-finite float raises
        ``ValueError``.
    gap : float, optional
        Gap between every pair of neighbouring wedges, in millimetres.
        Each wedge's two straight edges are inset by ``gap / 2``
        parallel to themselves, so the band between two neighbours is
        ``gap`` wide along its whole length, for wide and thin wedges
        alike, while the arcs keep their radii. The width is measured
        between the outlines' centre lines, so the unpainted band is
        ``gap`` less one outline width (``linewidth`` points). No
        wedge is ever dropped: a wedge of at most 180 degrees is inset
        by no more than ``r * sin(half-span / 2)``, ``r`` its drawn
        outer radius, so its rim arc keeps at least half its angular
        span -- half its rim length, and at least half its rim chord.
        A wedge thin enough to be capped is drawn narrowed, and the band
        beside it is narrower by what it was spared: ``h + h_i`` to a
        full-inset neighbour, where ``h`` is the full inset per edge and
        ``h_i <= h`` the capped wedge's, and ``h_i + h_j`` between two
        capped wedges; every other pair keeps the full width. With a float
        ``pull`` the two add up, so neighbours sit ``gap`` plus
        ``2 * t`` apart; a dict ``pull`` moves each inset wedge as a
        whole. Defaults to ``rcParams["pie.gap"]`` (``0.0``, no gap).
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
        ``"radial"``), ``basis`` (``"total"``/``"parent"``), ``leader``
        (``"radial"``/``"elbow"``/``None``), ``leader_gap`` and
        ``elbow_length`` (mm), ``avoid_overlap`` (move crowded outside
        labels apart) and ``label_spacing`` (mm between them).
        ``offset`` and the leader options default to the ``pie.*``
        rcParams.
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

    gap = check_length(resolve_param("pie.gap", gap), "gap", owner="pieplot")

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

    render_wedge_chart(
        ax, frame, levels,
        owner="pieplot", source_frame=data, y=y, hue=hue,
        order_map=order_map, sort=sort, hue_order=hue_order,
        start_angle=start_angle, direction=direction, donut=donut,
        pull=pull, gap=gap, palette=palette, hatch=hatch,
        hatch_map=hatch_map, hatch_order=hatch_order, color=color,
        edgecolor=edgecolor, linewidth=linewidth, alpha=alpha,
        center_label=center_label, aspect=aspect, legend=legend,
        legend_kws=legend_kws, title=title,
        bucket_label=OTHER_LABEL if min_share else None,
        bucket_color=OTHER_COLOR if min_share else None,
    )

    if annotate:
        kws = annotate if isinstance(annotate, dict) else {}
        _annotate(ax, "wedge_values", **kws)

    return ax
