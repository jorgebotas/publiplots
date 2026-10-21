"""Sunburst charts: part-of-whole hierarchies, one ring per level."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from publiplots.annotate._wedge import check_length
from publiplots.plot.wedge.colour import check_shading
from publiplots.plot.wedge.render import render_wedge_chart, warn_order_omissions
from publiplots.themes.rcparams import resolve_param


def sunburstplot(
    data: pd.DataFrame,
    x: Union[str, Sequence[str]],
    y: Optional[str] = None,
    hue: Optional[str] = None,
    *,
    ax: Optional[Axes] = None,
    donut: float = 0.0,
    start_angle: float = 90.0,
    direction: str = "clockwise",
    order: Union[Dict[str, Sequence], Sequence, None] = None,
    hue_order: Optional[Sequence] = None,
    sort: bool = False,
    pull: Union[float, Dict[Any, float], None] = None,
    gap: Optional[float] = None,
    depth_range: Optional[float] = None,
    sibling_range: Optional[float] = None,
    palette: Optional[Union[str, Dict, List, Colormap]] = None,
    hatch: Optional[str] = None,
    hatch_map: Optional[Dict[str, str]] = None,
    hatch_order: Optional[List[str]] = None,
    color: Optional[str] = None,
    edgecolor: Optional[str] = None,
    linewidth: Optional[float] = None,
    alpha: Optional[float] = None,
    center_label: Union[str, Callable[[float], str], None] = None,
    aspect: str = "cell",
    legend: Union[bool, Dict] = True,
    legend_kws: Optional[Dict] = None,
    annotate: Union[bool, Dict, None] = None,
    title: str = "",
    **kwargs,
) -> Axes:
    """Draw a sunburst: a part-of-whole hierarchy, one ring per level.

    The root ring splits the whole, and each outer ring splits its
    parent. ``x`` lists one column per ring, root first, ``y`` the value
    (omitted, rows are counted), and ``hue`` what determines colour --
    defaulting to the root column. A pie is the one-ring case:
    ``sunburstplot(data, x="kind")`` draws exactly the wedges
    :func:`publiplots.pieplot` draws, and a donut is the same with a
    hole.

    Parameters
    ----------
    data : pandas.DataFrame
        Tidy input. One row per observation, or one row per leaf when
        pre-aggregated.
    x : str or sequence of str
        One column per ring, root first. A string or a one-element
        sequence draws a single ring. A parent's value is the sum of its
        children plus the rows that end at it; nothing is read from the
        frame for an inner node. Only categories present in the data
        create wedges. Rows whose root column is ``NaN`` are dropped and
        the remaining wedges renormalise to fill the circle, silently.

        Hierarchies may be **ragged**. A row whose deeper columns are
        ``NaN`` ends at its last non-``NaN`` level: it adds to that
        wedge's own value, and the rings outside it stay empty over the
        row's share of its span. A wedge with children and rows of its
        own is a *parent-leaf*: its children are laid out first, from its
        leading edge, and its own share is an empty remainder at the
        trailing end. A row such as ``[a, NaN, c]``, with a ``NaN``
        followed by a deeper value, raises ``ValueError`` naming the
        row's index. Negative values raise ``ValueError``; zero-valued
        categories draw nothing.
    y : str, optional
        Value column, summed within each leaf group. When omitted, rows
        are counted.
    hue : str, optional
        Column determining colour. Defaults to the root column, ``x[0]``.
        It must be a **root attribute**: the root column itself, or a
        column constant within each root wedge. A column that varies
        within a root wedge raises ``ValueError`` naming the wedge and
        its values, and so does a later ring's column (``hue=x[k]`` for
        ``k > 0``). A numeric ``hue`` takes the unweighted mean over a
        root wedge's rows and colours through a colormap; the wedge's
        descendants inherit that colour, with ``depth_range`` and
        ``sibling_range`` disabled.
    ax : Axes, optional
        Matplotlib axes to draw on. If ``None``, creates a new figure
        via :func:`publiplots.layout.subplots` so the publiplots layout
        manager can keep geometry stable. To control axes dimensions,
        pre-create with ``pp.subplots(axes_size=(w_mm, h_mm))`` and
        pass ``ax=``.
    donut : float, default 0.0
        Hole radius as a fraction of the outer radius. The rest of the
        radius is divided evenly between the rings: with ``n`` rings
        each band is ``(1 - donut) / n`` wide. ``center_label`` goes in
        the hole.
    start_angle : float, default 90.0
        Angle of the first root wedge's leading edge, in degrees. 90 is
        twelve o'clock. Children start at their parent's leading edge.
    direction : {'clockwise', 'counterclockwise'}, default 'clockwise'
    order : dict or sequence, optional
        Explicit order. A **dict** ``{column: [...]}`` orders each column
        it names, within every parent; columns left out keep the order
        of the data (a categorical's declared order, else first
        appearance). A key that names no ``x`` column raises
        ``ValueError``. A flat sequence is accepted only when ``x`` is
        one column; with several it raises ``TypeError``. Categories
        absent from the data are dropped; categories present in the data
        but absent from ``order`` are filtered out, and what happens next
        depends on the column. Omit a **root** category and the wedges
        that remain renormalise to a full circle -- the chart then reads
        as 100% of a whole it is not showing all of. Omit a category of
        a **deeper** column and nothing renormalises: its parent still
        spans the full value of everything beneath it, so the omitted
        share stays as an empty arc inside the parent wedge and that ring
        sums to less than a full circle. Either way a
        :class:`UserWarning` names the dropped categories (and, for a
        deeper column, the column).
        Cannot be combined with ``sort``.
    hue_order : sequence, optional
        Running order of the legend's colour entries. Levels omitted from
        it keep their colour and are appended after the listed ones --
        ``hue_order`` reorders the key, it never reorders wedges. To
        drop categories, use ``order``.
    sort : bool, default False
        Order the children of each parent by descending value. A
        parent-leaf's own remainder stays last. Cannot be combined with
        ``order``.
    pull : float or dict, optional
        How far wedges move out from the centre, as a fraction of the
        outer radius.

        A **float** explodes the whole chart evenly. With ``t`` that
        fraction of the radius, ring ``k`` moves out to start at
        ``donut + (2k + 1)t + k * w``, ``w`` being the band width, so the
        hole grows by ``t`` and every seam between siblings and every
        boundary between rings opens to ``2t`` (plus ``gap``). Band
        widths are kept, and the outer rim sits at
        ``1 + t * (2n - 1)``. On one ring this is exactly the even
        explode of :func:`publiplots.pieplot`. ``0`` draws what no
        ``pull`` does; a negative or non-finite float raises
        ``ValueError``.

        A **dict** pops out just the nodes it names. Keys are path tuples
        (``("admin", "billing")``) or a root category (``"admin"``,
        shorthand for ``("admin",)``); a key equal to a root category
        exactly is taken as that category even if it is itself a tuple.
        A key matching no node raises ``ValueError``. Each named node
        moves along its own mid-ray by its value times the radius and
        takes its whole subtree with it, so the chart never tears; a
        descendant that is also named adds its own vector. A value must
        be a finite number of at least 0: ``TypeError`` for a bool or a
        non-number, ``ValueError`` for a negative or non-finite one.
    gap : float, optional
        Gap between every pair of neighbouring wedges, in millimetres:
        siblings and rings alike. Between siblings each straight edge is
        inset by ``gap / 2`` parallel to itself, so the band is ``gap``
        wide along its whole length and the seam runs straight through
        every ring. At each boundary between rings the inner ring's rim
        moves in by ``gap / 2`` and the outer ring's inner edge moves
        out by ``gap / 2``. The width is measured between outline centre
        lines, so the unpainted band is ``gap`` less one outline width.
        No wedge is dropped: an inset is capped so that a thin wedge
        keeps at least half its angular span, and a band is never inset
        radially to less than half its width. Where a very thin wedge
        caps its side of a ray, the gap on that side narrows, and every
        wedge on the same side of that ray, parent or child, takes the
        same inset, so the seam stays straight through the rings. A
        wedge wider than half the circle keeps a small hole of radius
        ``gap / 2`` (plus any explode) at the centre, so its edges keep
        the gap there too. A ragged leaf's rim, the outermost rim and the
        edge of the hole are otherwise never inset. Defaults to
        ``rcParams["pie.gap"]`` (``0.0``, no gap).
    depth_range : float, optional
        Total blend of the colour toward white at the outermost ring.
        Intermediate rings interpolate linearly and the root ring is the
        pure palette colour. A negative value blends toward black, giving
        a dark-rim variant. Both knobs are total spans, so adding a ring
        never washes the rim out to white. Defaults to
        ``rcParams["sunburst.depth_range"]`` (``0.45``).
    sibling_range : float, optional
        Total blend spread over the siblings under one parent, from 0 to
        the full amount in draw order. Negative values darken. Defaults
        to ``rcParams["sunburst.sibling_range"]`` (``0.14``). With both
        knobs at 0 each root category is one flat colour. The two must
        satisfy ``|depth_range| + |sibling_range| <= 1`` or
        ``ValueError`` is raised, whether given or read from rcParams.
    palette : str, dict, list, or Colormap, optional
        Colour palette for the root hue levels. For a categorical
        ``hue``, can be:

        - ``str``: seaborn palette name or publiplots palette name
        - ``dict``: explicit ``{hue_level: color}`` mapping, overlaid on
          the resolved defaults, so a partial dict recolours only the
          levels it names
        - ``list``: list of colours assigned in hue-level order

        For a numeric ``hue``: a matplotlib colormap name
        (``"viridis"``) or a :class:`~matplotlib.colors.Colormap`, used
        for the wedges and the colorbar alike; a dict or list raises
        ``TypeError``, since a numeric hue has no discrete levels to
        assign colours to. Left as ``None``, ``sunburstplot`` builds its
        own light ramp running from near-white up to ``color``.
    hatch : str, optional
        Column name driving a **second categorical dimension** encoded
        via hatch textures -- the texture-based analogue of ``hue``. A
        path column ``x[k]`` gives ring ``k``'s wedges their patterns and
        every descendant inherits them; rings inside ring ``k`` are
        unhatched. A column outside the path must be constant within each
        root wedge, and rings 0 and up then take the patterns, the same
        rule as ``hue``.

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
        anchor for a numeric ``hue`` when ``palette`` is ``None``, and the
        swatch colour for a standalone hatch key. Falls back to
        ``rcParams["color"]``.
    edgecolor : str, optional
        Explicit edge colour override. When ``None`` (the default, also
        the default of ``rcParams["edgecolor"]``), each wedge's outline
        matches its own shaded fill -- publiplots' double-layer
        transparent-fill styling. Set to a specific colour (``"black"``)
        to outline every wedge the same way regardless of fill.
    linewidth : float, optional
        Width of the wedge outlines and hatch strokes. Defaults to
        ``rcParams["edgewidth"]``.
    alpha : float, optional
        Transparency of the wedge *fill* (outlines stay fully opaque).
        ``0`` = outlined wedges only; ``1`` = solid fill. Defaults to
        ``rcParams["alpha"]``.
    center_label : str or callable, optional
        Text for the hole. A callable receives the total. Ignored when
        ``donut=0``: a chart without a hole has nowhere to write it.
    aspect : {'cell', 'equal'}, default 'cell'
        ``'cell'`` matches the data limits to the axes' millimetre
        ratio: the disc is sized to the shorter side of the cell and the
        longer side's limits are stretched to match, so the axes box
        keeps the whole allocation on both axes. ``'equal'`` uses
        matplotlib's aspect constraint, which survives an interactive
        resize but shrinks the drawn box to a square, keeping only
        ``min(w, h) / max(w, h)`` of the allocation on whichever axis is
        longer. Both render a round disc.
    legend : bool or dict, default=True
        Whether to render the legend. Accepts ``bool`` or
        ``dict[kind, bool]`` for per-kind control (e.g.
        ``legend={"hatch": False}`` hides just the hatch entry). Only
        the hue categories -- the root ring's -- get an entry; the
        shades of the outer rings are never listed. A numeric ``hue``
        gets a colorbar. ``False`` suppresses only the rendering:
        entries are stashed either way, so a later figure-level
        :func:`publiplots.legend` group can still claim them.
    legend_kws : dict, optional
        Extra kwargs forwarded to the legend builder. Notable keys:
        ``inside=True`` with ``loc=...`` places the legend inside the
        axes using matplotlib's corner-based placement; ``hue_label`` /
        ``hatch_label`` override the title shown above each entry.
    annotate : bool or dict, optional
        Not supported yet: any truthy value raises
        ``NotImplementedError``, as does
        ``pp.annotate(ax, "wedge_values")`` on a sunburst axes. Wedge
        labels for sunburst charts arrive in the next release.
    title : str or None, default=""
        Plot title. Pass ``None`` to leave the existing title unchanged.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    TypeError
        For an unknown keyword argument (``min_share`` is not offered:
        there is no bucketing inside a sunburst), for a flat ``order``
        with several ``x`` columns, and for the ``pull`` value errors
        above.
    ValueError
        For an unusable ``x``, ``order``, ``pull``, ``hue`` or ``hatch``,
        a negative value, a non-contiguous path, ``sort=True`` together
        with ``order``, and ``|depth_range| + |sibling_range| > 1``.
    NotImplementedError
        For a truthy ``annotate``.

    Notes
    -----
    ``hue`` means colour here and nothing else. In ``barplot`` it also
    controls how marks group, because grouping has no other parameter; in
    a wedge chart the structure lives in ``x``, so the two are separate.
    For the same reason ``hue`` must be the root column or constant
    within each root wedge: the colour of an outer wedge is its root's
    colour, shaded by ring and by sibling position, and is not set by
    the wedge's own category.

    Labelling wedges, through ``annotate=`` or :func:`publiplots.annotate`,
    is not supported yet.

    Examples
    --------
    >>> ax = pp.sunburstplot(data=df, x=["team", "role", "level"], y="n")
    >>> ax = pp.sunburstplot(data=df, x=["team", "role"], y="n",
    ...                      donut=0.3, gap=1.0, depth_range=-0.3,
    ...                      pull={"design": 0.05})
    """
    from publiplots.layout.subplots import reject_figsize
    reject_figsize(kwargs)
    if kwargs:
        raise TypeError(
            "sunburstplot() got an unexpected keyword argument "
            + ", ".join(repr(k) for k in sorted(kwargs, key=str)))
    if annotate:
        raise NotImplementedError(
            "sunburstplot: annotate= is not supported yet; wedge labels for "
            "sunburst charts arrive in the next release. Draw the chart "
            "without it for now.")

    levels = [x] if isinstance(x, str) else list(x)
    if not levels:
        raise ValueError("sunburstplot: x= must name at least one column.")

    order_map = None
    if order is not None:
        if isinstance(order, Mapping):
            unknown = [k for k in order if k not in levels]
            if unknown:
                raise ValueError(
                    f"sunburstplot: order= names {unknown!r}, which are not "
                    f"x= columns {levels!r}.")
            order_map = {k: list(v) for k, v in order.items()}
        elif len(levels) == 1:
            order_map = {levels[0]: list(order)}
        else:
            raise TypeError(
                "sunburstplot: with several x= columns, order= must be a "
                "dict keyed by column, e.g. {'kind': [...], 'subkind': [...]}.")

    gap = check_length(resolve_param("pie.gap", gap), "gap", owner="sunburstplot")
    depth_range = float(resolve_param("sunburst.depth_range", depth_range))
    sibling_range = float(resolve_param("sunburst.sibling_range", sibling_range))
    check_shading(depth_range, sibling_range, owner="sunburstplot")

    linewidth = resolve_param("edgewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    color = resolve_param("color", color)
    edgecolor = resolve_param("edgecolor", edgecolor)

    if ax is None:
        from publiplots.layout.subplots import subplots as _pp_subplots
        _, ax = _pp_subplots()

    if order_map is not None:
        warn_order_omissions(data, order_map, owner="sunburstplot",
                             root=levels[0])

    render_wedge_chart(
        ax, data, levels, owner="sunburstplot", chart="sunburst",
        source_frame=data, y=y,
        hue=hue, order_map=order_map, sort=sort, hue_order=hue_order,
        start_angle=start_angle, direction=direction, donut=donut,
        pull=pull, gap=gap, palette=palette, hatch=hatch,
        hatch_map=hatch_map, hatch_order=hatch_order, color=color,
        edgecolor=edgecolor, linewidth=linewidth, alpha=alpha,
        center_label=center_label, aspect=aspect, legend=legend,
        legend_kws=legend_kws, title=title,
        depth_range=depth_range, sibling_range=sibling_range,
    )
    return ax
