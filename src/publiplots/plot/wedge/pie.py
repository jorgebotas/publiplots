"""Pie and donut charts."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, to_rgba

from publiplots.annotate import annotate as _annotate
from publiplots.plot.wedge.colour import derive_colors, resolve_hue_values
from publiplots.plot.wedge.draw import (
    apply_cell_limits,
    build_wedge_meta,
    draw_wedges,
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
    out[level] = out[level].astype(object)
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
    min_share: float = 0.0,
    center_label: Union[str, Callable[[float], str], None] = None,
    aspect: str = "cell",
    palette: Optional[Union[str, Dict, List]] = None,
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

    A pie is a one-level sunburst. ``x`` names the category column, ``y``
    the value (omitted, rows are counted), and ``hue`` what determines
    colour -- defaulting to ``x`` itself, which is what every pie does.
    For more than one level use :func:`publiplots.sunburstplot`.

    Parameters
    ----------
    data : pandas.DataFrame
        Tidy input. One row per observation, or one row per category when
        pre-aggregated.
    x : str
        Column naming each wedge. A sequence of length 1 is accepted;
        anything longer raises, naming ``sunburstplot``.
    y : str, optional
        Value column, summed within each category. When omitted, rows are
        counted.
    hue : str, optional
        Column determining colour. Defaults to ``x``. A categorical hue
        must be constant within a wedge.
    donut : float, default 0.0
        Inner radius as a fraction of the outer radius. ``0.5`` leaves a
        hole of half the radius, which is where ``center_label`` goes.
    start_angle : float, default 90.0
        Angle of the first wedge's leading edge, in degrees. 90 is twelve
        o'clock.
    direction : {'clockwise', 'counterclockwise'}, default 'clockwise'
    order : sequence, optional
        Explicit wedge order. Categories absent from the data are dropped;
        categories absent from ``order`` are filtered out.
    sort : bool, default False
        Order wedges by descending value. Cannot be combined with
        ``order``. Off by default so a pie keeps the category order of a
        bar panel drawn from the same frame.
    pull : float or dict, optional
        Radial offset as a fraction of the radius: a float for every
        wedge, or a dict keyed by category. Deliberately not a positional
        sequence.
    min_share : float, default 0.0
        Categories below this fraction of the total merge into a single
        ``"Other"`` wedge. Applies only when at least two categories fall
        below the threshold.
    center_label : str or callable, optional
        Text for the donut hole. A callable receives the total.
    aspect : {'cell', 'equal'}, default 'cell'
        ``'cell'`` matches the data limits to the axes' millimetre ratio,
        keeping the full cell width; ``'equal'`` uses matplotlib's aspect
        constraint, which survives an interactive resize but abandons part
        of the cell.

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

    levels = [x] if isinstance(x, str) else list(x)
    if len(levels) > 1:
        raise ValueError(
            "pieplot: x= must name a single column; for a nested "
            "composition use pp.sunburstplot(x=[...])."
        )

    linewidth = resolve_param("edgewidth", linewidth)
    alpha = resolve_param("alpha", alpha)
    color = resolve_param("color", color)
    edgecolor = resolve_param("edgecolor", edgecolor)

    if ax is None:
        from publiplots.layout.subplots import subplots as _pp_subplots
        _, ax = _pp_subplots()

    frame = _bucket_small(data, levels[0], y, min_share) if min_share else data
    order_map = {levels[0]: list(order)} if order is not None else None

    nodes = build_nodes(frame, levels, y, order=order_map, sort=sort)
    if min_share and any(n.path[0] == OTHER_LABEL for n in nodes):
        # The bucket sorts last whatever the resolver decided, then the
        # sibling indices are renumbered so the colour derivation sees a
        # consistent run. (Inert on one ring, but wrong indices here would
        # become wrong colours the moment the sunburst reuses this path.)
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

    hue_values, hue_continuous = resolve_hue_values(frame, levels, hue, nodes)
    hue_levels = list(dict.fromkeys(
        hue_values[n.path] for n in nodes if n.path in hue_values))
    hue_norm = None
    if hue_continuous:
        # A numeric hue maps through a colormap rather than a palette. Build
        # a value -> colour dict so derive_colors' lookup is unchanged.
        cmap = resolve_continuous_cmap(None, color=color)
        lo, hi = min(hue_levels), max(hue_levels)
        hue_norm = Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1.0)
        palette_map = {v: cmap(hue_norm(v)) for v in hue_levels}
    else:
        if hue_order is not None:
            hue_levels = [v for v in hue_order if v in hue_levels]
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

    colors = derive_colors(nodes, hue_values, palette_map,
                           depth_range=0.0, sibling_range=0.0, n_levels=1,
                           hue_column=hue if hue is not None else levels[0])

    # Hatch is resolved from its own column, the same way hue is, rather
    # than being keyed off the hue value -- the two are independent
    # parameters and only coincide in the common hatch=x case.
    if hatch is not None:
        hatch_values, _ = resolve_hue_values(frame, levels, hatch, nodes)
        hatch_levels = list(dict.fromkeys(
            hatch_values[n.path] for n in nodes if n.path in hatch_values))
        if hatch_order is not None:
            hatch_levels = [v for v in hatch_order if v in hatch_levels]
        hatch_map_resolved = resolve_hatch_map(values=hatch_levels,
                                               hatch_map=hatch_map)
        hatches = {n.path: hatch_map_resolved.get(hatch_values.get(n.path))
                   for n in nodes}
    else:
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

    if donut and center_label is not None:
        text = center_label(total) if callable(center_label) else center_label
        ax.text(0.0, 0.0, text, ha="center", va="center")

    if title:
        ax.set_title(title)

    flags = resolve_legend_flags(legend)
    if hue_continuous and hue_levels:
        stash_continuous_hue(
            ax,
            name=hue if hue is not None else levels[0],
            palette=resolve_continuous_cmap(None, color=color),
            hue_norm=hue_norm,
        )
        render_entries(ax, flags=flags, legend_kws=legend_kws)
    elif hue_levels:
        handles = create_legend_handles(
            labels=[str(v) for v in hue_levels],
            colors=[palette_map[v] for v in hue_levels],
            edgecolors=[edgecolor] * len(hue_levels) if edgecolor else None,
            hatches=([hatch_map_resolved.get(v) for v in hue_levels]
                     if hatch else None),
            alpha=alpha,
            linewidth=linewidth,
        )
        stash_entry(ax, LegendEntry.build(
            name=hue if hue is not None else levels[0],
            kind="hue",
            handles=handles,
            labels=[str(v) for v in hue_levels],
        ))
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
