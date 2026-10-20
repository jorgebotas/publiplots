"""The drawing pipeline shared by pieplot and sunburstplot."""
from __future__ import annotations

import math
import numbers
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize, to_rgba

from publiplots.annotate._cache import WedgeMeta
from publiplots.plot.wedge.colour import derive_colors, resolve_hue_values
from publiplots.plot.wedge.draw import (
    apply_cell_limits,
    apply_gap,
    build_wedge_meta,
    draw_wedges,
    even_explode,
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
from publiplots.utils import create_legend_handles
from publiplots.utils.legend_entries import (
    LegendEntry,
    resolve_legend_flags,
    stash_entry,
)
from publiplots.utils.plot_legend import render_entries, stash_continuous_hue


def check_scalar_pull(pull: Any, *, owner: str) -> float:
    """A scalar ``pull`` must be a finite, non-negative fraction."""
    if isinstance(pull, bool) or not isinstance(pull, numbers.Real):
        raise TypeError(
            f"{owner}: pull= must be a number (a fraction of the radius) "
            f"or a dict keyed by category; got {type(pull).__name__} "
            f"{pull!r}."
        )
    pull = float(pull)
    if not math.isfinite(pull) or pull < 0:
        raise ValueError(
            f"{owner}: pull= must be a finite, non-negative fraction of "
            f"the radius; got {pull!r}."
        )
    return pull


def numeric_hue_cmap(
    palette: Union[str, Colormap, Dict, List, None],
    *,
    color: Any,
    hue: str,
    owner: str,
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
            f"{owner}: hue={hue!r} is numeric, so it maps through a "
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
                f"{owner}: hue={hue!r} is numeric, so palette= must name a "
                f"registered matplotlib colormap (e.g. 'viridis'); "
                f"{palette!r} is not one."
            ) from None
    raise TypeError(
        f"{owner}: palette= must be a colormap name, a matplotlib "
        f"Colormap or None for a numeric hue; got "
        f"{type(palette).__name__}."
    )


def render_wedge_chart(
    ax: Axes,
    frame: pd.DataFrame,
    levels: List[str],
    *,
    owner: str,
    source_frame: pd.DataFrame,
    y: Optional[str],
    hue: Optional[str],
    order_map: Optional[Dict[str, List[Any]]],
    sort: bool,
    hue_order: Optional[Sequence],
    start_angle: float,
    direction: str,
    donut: float,
    pull: Union[float, Dict[Any, float], None],
    gap: float,
    palette: Any,
    hatch: Optional[str],
    hatch_map: Optional[Dict[str, str]],
    hatch_order: Optional[List[str]],
    color: Any,
    edgecolor: Any,
    linewidth: float,
    alpha: float,
    center_label: Union[str, Callable[[float], str], None],
    aspect: str,
    legend: Union[bool, Dict],
    legend_kws: Optional[Dict],
    title: Optional[str],
    depth_range: float = 0.0,
    sibling_range: float = 0.0,
    bucket_label: Optional[str] = None,
    bucket_color: Optional[str] = None,
) -> WedgeMeta:
    """Draw a wedge chart on ``ax`` and stash its legend entries.

    ``frame`` is the data to draw, already bucketed by the caller;
    ``source_frame`` is the caller's original ``data``, recorded on the
    meta. ``owner`` names the calling function in every message. Every
    style argument arrives resolved (``gap`` in millimetres, already
    validated). ``bucket_label`` names the caller's ``min_share`` bucket,
    which sorts last and is painted ``bucket_color`` unless ``palette``
    names it; ``None`` means nothing was bucketed.

    Sets ``ax._publiplots_wedge_meta`` and returns it. Does not annotate.
    """
    hue_column = hue if hue is not None else levels[0]

    nodes = build_nodes(frame, levels, y, order=order_map, sort=sort,
                        owner=owner)
    if bucket_label is not None and any(n.path[0] == bucket_label
                                        for n in nodes):
        # The bucket sorts last whatever the resolver decided, then the
        # sibling indices are renumbered so the colour derivation sees a
        # consistent run. (Inert on one ring, but wrong indices here would
        # become wrong colours the moment a nested ring reuses this path.)
        ordered = ([n for n in nodes if n.path[0] != bucket_label]
                   + [n for n in nodes if n.path[0] == bucket_label])
        nodes = [
            WedgeNode(path=n.path, value=n.value, depth=n.depth,
                      parent=n.parent, sibling_index=i,
                      sibling_count=len(ordered))
            for i, n in enumerate(ordered)
        ]
    total = total_value(nodes)
    places = layout(nodes, start_angle=start_angle, direction=direction,
                    donut=donut, n_levels=len(levels))

    hue_values, hue_continuous = resolve_hue_values(
        frame, levels, hue, nodes, param_name="hue",
        bucket_label=bucket_label)
    hue_levels = list(dict.fromkeys(
        hue_values[n.path] for n in nodes if n.path in hue_values))
    hue_norm = None
    # Resolved once, here, and read by both the wedge colours and the
    # colorbar below, so the two cannot drift onto different colormaps.
    # Resolved before the all-NaN check so a bad palette= is reported
    # whatever the data holds.
    hue_cmap = (numeric_hue_cmap(palette, color=color, hue=hue_column,
                                 owner=owner)
                if hue_continuous else None)
    if hue_continuous and not hue_levels:
        # Every value in the column was NaN, so no wedge resolved one.
        # min()/max() over that raises "min() iterable argument is
        # empty", which names neither the column nor the cause. The
        # categorical path degrades to the fallback grey instead, so do
        # the same here and say which column it was.
        warnings.warn(
            f"{owner}: hue={hue_column!r} is numeric but has no non-NaN "
            f"value in any wedge, so there is no range to build a "
            f"colormap over. The wedges fall back to a neutral grey and "
            f"no colorbar is drawn.",
            UserWarning,
            stacklevel=3,
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
        named_other = isinstance(palette, dict) and bucket_label in palette
        if bucket_label is not None and not named_other:
            palette_map = {**palette_map, bucket_label: bucket_color}
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
                           depth_range=depth_range,
                           sibling_range=sibling_range,
                           n_levels=len(levels), hue_column=hue_column)

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
            bucket_label=bucket_label)
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

    # A dict pops the wedges it names out along their own mid-rays, as
    # matplotlib's `explode` does. A scalar is the even explode: no wedge
    # is translated, every one grows outward by the same distance and is
    # inset by it, so all neighbours part by the same gap (`even_explode`).
    r_outer = max((p.r_outer for p in places), default=1.0)
    if pull is None:
        pull_map = {}
        explode = 0.0
    elif isinstance(pull, dict):
        pull_map = {n.path: float(pull.get(n.path[0], 0.0)) for n in nodes}
        explode = 0.0
    else:
        pull = check_scalar_pull(pull, owner=owner)
        # The translation map stays empty: records still carry the value
        # asked for, while the geometry takes it from `explode`.
        pull_map = {}
        explode = even_explode(pull, r_outer)

    patches = draw_wedges(
        ax, places, colors, hatches, center=(0.0, 0.0), pull=pull_map,
        alpha=alpha, edgecolor=edgecolor, linewidth=linewidth,
        explode=explode,
    )
    split_face_and_edge_alpha(patches, alpha=alpha, edgecolor=edgecolor)

    record_pull = (pull_map if not explode
                   else {n.path: float(pull) for n in nodes})

    # Room for the furthest rim: a translated wedge reaches `pull` past
    # the radius, and an evenly exploded one has its rim at r + t.
    max_pull = max(pull_map.values(), default=0.0)
    apply_cell_limits(ax, radius=r_outer + max(max_pull, explode) + 0.05,
                      aspect=aspect)
    # Only once the limits are framed does a data unit have a fixed size
    # in millimetres to convert the gap with.
    apply_gap(ax, patches, gap, explode=explode)

    # Built last, so the records are read off the patches as they are
    # finally drawn: after the gap, and after any explode.
    ax._publiplots_wedge_meta = build_wedge_meta(
        places, patches, colors, hatches, record_pull,
        total=total, donut=donut, center=(0.0, 0.0), levels=levels,
        hue_active=True, hue_continuous=hue_continuous,
        source_frame=source_frame,
    )

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

    return ax._publiplots_wedge_meta


def split_face_and_edge_alpha(
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
