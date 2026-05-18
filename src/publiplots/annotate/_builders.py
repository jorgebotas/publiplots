"""Owned-axes meta builders.

Plot functions call these to construct a strategy-specific meta object
pre-paired with the drawn artists, so the annotator doesn't have to
re-introspect the axes. The foreign-axes equivalent is `_introspect` in
`_cache.py`.

One builder per plot kind. Each takes the plot function's resolved
inputs (data, column names, palette map, errorbar spec) and returns a
`*Meta` dataclass with `owner_is_publiplots=True`.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

from matplotlib.patches import PathPatch

from publiplots.annotate._cache import (
    BarRecord,
    BarValueMeta,
    BoxStatsMeta,
    BoxStatsRecord,
    PointRecord,
    PointValueMeta,
    _is_bar_rect,
    _iter_error_segments,
    _match_errorbars,
)
from publiplots.annotate._splits import BarSplitSpec, _categories_in_draw_order


def _aggregate_means(
    data,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    hatch: Optional[str] = None,
    source_frame=None,
) -> List[Dict]:
    """Group by (categorical_axis [, hue [, hatch]]) and return means.

    Each row carries its draw-order category/hue/hatch keys and the
    frame row index of the first matching source row, so downstream
    label lookups can align by group without re-deriving the spec.

    ``frame_row_index`` is a *position* (integer offset) into
    ``source_frame``, not a pandas label, so ``source_frame.iloc[idx]``
    resolves the caller's row for any index kind (RangeIndex, string
    index, sliced, MultiIndex, ...). When ``source_frame`` is omitted
    the field is ``None``.
    """
    value_col = y if categorical_axis == x else x
    spec = BarSplitSpec.resolve(
        x=x, y=y, hue=hue, hatch=hatch, categorical_axis=categorical_axis,
    )

    rows: List[Dict] = []
    for cat, h_val, ht_val in spec.iter_draw_order(data):
        parts = [data[categorical_axis] == cat]
        if spec.split_hue is not None:
            parts.append(data[spec.split_hue] == h_val)
        if spec.split_hatch is not None:
            parts.append(data[spec.split_hatch] == ht_val)
        mask = parts[0]
        for p in parts[1:]:
            mask = mask & p
        vals = data.loc[mask, value_col].to_numpy()
        # Position within source_frame of the first row matching this group.
        # We resolve against source_frame (not the prepared `data`) because the
        # meta's `source_frame` is the pre-copy caller frame; `iloc` on the
        # caller's frame must land on the same semantic row.
        matching = data.index[mask]
        frame_row_index: Optional[int] = None
        if source_frame is not None and len(matching):
            # Map the matching label back to a position in source_frame's index.
            first_label = matching[0]
            try:
                frame_row_index = int(source_frame.index.get_loc(first_label))
            except (KeyError, TypeError):
                # Prepared data diverged from source_frame's index (rare).
                frame_row_index = None
        row: Dict = {
            "mean": float(np.mean(vals)) if len(vals) else float("nan"),
            "category": cat,
            "hue_value": h_val,
            "hatch_value": ht_val,
            "frame_row_index": frame_row_index,
        }
        rows.append(row)
    return rows


def _match_point_errorbars(
    ax: Axes,
    points_xy: List,
    orient: str,
    tol: float,
) -> List:
    """For each (x, y) point, find its errorbar segment.

    Match by (a) segment aligned on the categorical axis at px within `tol`,
    then (b) segment midpoint closest to py on the value axis. The second
    step disambiguates overlapping hue series that share an x-position.
    Returns [(err_low, err_high), ...] in point order.
    """
    segments = list(_iter_error_segments(ax))
    results: List = []
    for (px, py) in points_xy:
        best = None
        best_dist = float("inf")
        for xs, ys in segments:
            if orient == "v":
                if max(xs) - min(xs) >= 1e-6 or abs(xs[0] - px) > tol:
                    continue
                seg_low = float(min(ys))
                seg_high = float(max(ys))
                seg_mid = (seg_low + seg_high) / 2.0
                dist = abs(seg_mid - py)
            else:
                if max(ys) - min(ys) >= 1e-6 or abs(ys[0] - py) > tol:
                    continue
                seg_low = float(min(xs))
                seg_high = float(max(xs))
                seg_mid = (seg_low + seg_high) / 2.0
                dist = abs(seg_mid - px)
            if dist < best_dist:
                best = (seg_low, seg_high)
                best_dist = dist
        results.append(best if best is not None else (None, None))
    return results


def build_from_pointplot_call(
    ax: Axes,
    data,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    palette: Optional[Dict],
    errorbar: Optional[str],
    *,
    source_frame,
) -> PointValueMeta:
    """Build a `PointValueMeta` paired with the pointplot's marker positions.

    ``source_frame`` is a required keyword-only: must be the caller's
    pre-copy DataFrame so ``meta.source_frame is df`` holds and
    ``source_frame.iloc[record.frame_row_index]`` resolves correctly.
    """
    spec = BarSplitSpec.resolve(
        x=x, y=y, hue=hue, hatch=None, categorical_axis=categorical_axis,
    )
    orient = spec.orient
    cat_categories = _categories_in_draw_order(data[categorical_axis])
    cat_to_pos = {cat: i for i, cat in enumerate(cat_categories)}
    value_col = y if categorical_axis == x else x

    points_xy: List[Tuple[float, float]] = []
    aggregated: List[Dict] = []
    for cat, h_val, _ht_val in spec.iter_draw_order(data):
        mask = data[categorical_axis] == cat
        if spec.split_hue is not None:
            mask = mask & (data[spec.split_hue] == h_val)
        vals = data.loc[mask, value_col].to_numpy()
        if len(vals) == 0:
            continue
        pos = cat_to_pos[cat]
        mean = float(np.mean(vals))
        points_xy.append((pos, mean) if orient == "v" else (mean, pos))
        matching_labels = data.index[mask]
        frame_row_index: Optional[int] = None
        if source_frame is not None and len(matching_labels):
            try:
                frame_row_index = int(
                    source_frame.index.get_loc(matching_labels[0])
                )
            except (KeyError, TypeError):
                frame_row_index = None
        aggregated.append({
            "category": cat,
            "hue_value": h_val,
            "frame_row_index": frame_row_index,
        })

    # Tolerance on the categorical axis: 0.5 is conservative (integer positions)
    tol = 0.5
    err_by_point = _match_point_errorbars(ax, points_xy, orient, tol)

    points: List[PointRecord] = []
    for i, (xy, row, (err_low, err_high)) in enumerate(
        zip(points_xy, aggregated, err_by_point)
    ):
        hue_color = None
        hv = row["hue_value"]
        if hv is not None and palette is not None and hv in palette:
            hue_color = to_rgba(palette[hv])
        value = xy[1] if orient == "v" else xy[0]
        points.append(PointRecord(
            xy=xy,
            value=float(value),
            err_low=err_low,
            err_high=err_high,
            hue_color=hue_color,
            category=row["category"],
            hue_value=row["hue_value"],
            hatch_value=None,
            draw_index=i,
            frame_row_index=row["frame_row_index"],
        ))

    keys: List[str] = [categorical_axis]
    dims: List[str] = ["cat"]
    if spec.split_hue is not None:
        keys.append(spec.split_hue)
        dims.append("hue")

    return PointValueMeta(
        orient=orient,
        points=points,
        errorbar_kind=errorbar if isinstance(errorbar, str) else None,
        hue_active=spec.split_hue is not None,
        owner_is_publiplots=True,
        source_frame=source_frame,
        group_keys=tuple(keys),
        group_dims=tuple(dims),
    )


def build_from_barplot_call(
    ax: Axes,
    data,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    palette: Optional[Dict],
    errorbar: Optional[str],
    hatch: Optional[str] = None,
    *,
    source_frame,
) -> BarValueMeta:
    """Build a `BarValueMeta` paired with the barplot's Rectangles.

    Means come from a groupby of the raw data; errorbar extents are pulled
    from the drawn errorbar artists via `_match_errorbars`, so they match
    seaborn exactly for every errorbar spec (`"ci"`, `"pi"`, tuples,
    callables, `None`). Records are paired with Rectangles in draw order,
    which under hue + hatch double-split is hue-outer / hatch-middle /
    cat-inner. Hue colors come from the resolved palette map.

    Parameters
    ----------
    source_frame : pandas.DataFrame
        The caller's pre-copy DataFrame. Required keyword-only; must be
        the exact object the user passed to ``pp.barplot``, so
        ``meta.source_frame is df`` holds and
        ``source_frame.iloc[record.frame_row_index]`` resolves correctly
        regardless of the caller's index kind.
    """
    orient = "v" if categorical_axis == x else "h"

    agg = _aggregate_means(data, x=x, y=y, hue=hue, hatch=hatch,
                           categorical_axis=categorical_axis,
                           source_frame=source_frame)
    # `_is_bar_rect` treats signed extents as valid so negative-valued bars
    # (matplotlib emits them as rects with negative height/width) are kept.
    rects = [p for p in ax.patches if _is_bar_rect(p)]
    err_by_bar = _match_errorbars(ax, rects, orient)

    bars: List[BarRecord] = []
    for i, (rect, row, (err_low, err_high)) in enumerate(
        zip(rects, agg, err_by_bar)
    ):
        hue_color = None
        if hue is not None and palette is not None:
            key = row.get("hue_value")
            if key is not None and key in palette:
                hue_color = to_rgba(palette[key])
        if hue_color is None:
            # Strip face alpha (publiplots' default bar style is alpha=0.1
            # outline) so labels referencing this color render opaquely
            # rather than inheriting the bar's translucency.
            r, g, b, _ = rect.get_facecolor()
            hue_color = (r, g, b, 1.0)
        bars.append(BarRecord(
            patch=rect,
            value=float(row["mean"]),
            err_low=err_low,
            err_high=err_high,
            hue_color=hue_color,
            anchor_override=None,
            category=row["category"],
            hue_value=row["hue_value"],
            hatch_value=row["hatch_value"],
            draw_index=i,
            frame_row_index=row["frame_row_index"],
        ))

    spec = BarSplitSpec.resolve(
        x=x, y=y, hue=hue, hatch=hatch, categorical_axis=categorical_axis,
    )
    keys: List[str] = [categorical_axis]
    dims: List[str] = ["cat"]
    if spec.split_hue is not None:
        keys.append(spec.split_hue)
        dims.append("hue")
    if spec.split_hatch is not None:
        keys.append(spec.split_hatch)
        dims.append("hatch")

    return BarValueMeta(
        orient=orient,
        bars=bars,
        errorbar_kind=errorbar,
        hue_active=hue is not None,
        owner_is_publiplots=True,
        source_frame=source_frame,
        group_keys=tuple(keys),
        group_dims=tuple(dims),
    )


def build_from_stacked_barplot_call(
    ax: Axes,
    data,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    palette: Optional[Dict],
    hatch: Optional[str] = None,
    multiple: Literal["stack", "fill", "gain"] = "stack",
    *,
    source_frame,
) -> BarValueMeta:
    """Build a ``BarValueMeta`` paired with a stacked bar plot's Rectangles.

    The stacked path draws with ``ax.bar(bottom=cum)`` (or ``ax.barh(left=cum)``)
    so each Rectangle's ``get_height()`` / ``get_width()`` is the per-segment
    value — not the cumulative total. We pair each drawn patch with an
    aggregate row yielded by ``BarSplitSpec.iter_draw_order`` in the same order
    the plotter painted them, so hue color resolution is deterministic and
    does not rely on color-matching heuristics.

    For ``multiple="gain"``, the drawing path at `_draw_stacked` in
    `plot/bar.py` emits the base segment with ``bottom=0`` and the top
    segment with ``bottom=lo``. This function inverts that geometry:
    when ``rect.get_y() > 0`` the rect is a top segment and the value
    labeled is ``y + height`` (the winner's absolute total); otherwise
    (base segment or tie) the label value is ``rect.get_height()``
    itself. Horizontal orient mirrors with ``x`` / ``width``.

    Parameters
    ----------
    source_frame : pandas.DataFrame
        The caller's pre-copy DataFrame. Required keyword-only; must be
        the exact object the user passed to ``pp.barplot``, so
        ``meta.source_frame is df`` holds and
        ``source_frame.iloc[record.frame_row_index]`` resolves correctly
        regardless of the caller's index kind.
    """
    orient: Literal["v", "h"] = "v" if categorical_axis == x else "h"

    agg = _aggregate_means(
        data, x=x, y=y, hue=hue, hatch=hatch,
        categorical_axis=categorical_axis,
        source_frame=source_frame,
    )
    # No > 0 filter on height / width: the stacked path emits one
    # Rectangle per aggregated row via ax.bar, including legitimate
    # zero-valued segments (e.g. a group mean that happens to be 0).
    # Pairing with agg is 1:1 in deterministic draw order.
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]

    bars: List[BarRecord] = []
    for i, (rect, row) in enumerate(zip(rects, agg)):
        anchor_override: Optional[str] = None
        if multiple == "gain":
            # Absolute values: base segment's height is the loser's total;
            # top segment's cumulative (y + height) is the winner's total.
            # For ties we get a single rect whose height is already the
            # absolute value.
            if orient == "v":
                if rect.get_y() > 0:
                    value = rect.get_y() + rect.get_height()  # top: winner abs
                    anchor_override = "outside"                # winner label floats above
                else:
                    value = rect.get_height()                  # base or tie
                    anchor_override = "inside"                 # loser label inside base
            else:
                if rect.get_x() > 0:
                    value = rect.get_x() + rect.get_width()
                    anchor_override = "outside"
                else:
                    value = rect.get_width()
                    anchor_override = "inside"
        else:
            value = rect.get_height() if orient == "v" else rect.get_width()

        hue_color: Optional[Tuple[float, float, float, float]] = None
        if palette is not None:
            key = row.get("hue_value")
            if key is not None and key in palette:
                hue_color = to_rgba(palette[key])
        if hue_color is None:
            # Strip face alpha — see note on the dodged-builder fallback.
            r, g, b, _ = rect.get_facecolor()
            hue_color = (r, g, b, 1.0)
        bars.append(BarRecord(
            patch=rect,
            value=float(value),
            err_low=None,
            err_high=None,
            hue_color=hue_color,
            anchor_override=anchor_override,
            category=row["category"],
            hue_value=row["hue_value"],
            hatch_value=row["hatch_value"],
            draw_index=i,
            frame_row_index=row["frame_row_index"],
        ))

    spec = BarSplitSpec.resolve(
        x=x, y=y, hue=hue, hatch=hatch, categorical_axis=categorical_axis,
    )
    keys: List[str] = [categorical_axis]
    dims: List[str] = ["cat"]
    if spec.split_hue is not None:
        keys.append(spec.split_hue)
        dims.append("hue")
    if spec.split_hatch is not None:
        keys.append(spec.split_hatch)
        dims.append("hatch")

    return BarValueMeta(
        orient=orient,
        bars=bars,
        errorbar_kind=None,
        hue_active=hue is not None,
        owner_is_publiplots=True,
        source_frame=source_frame,
        group_keys=tuple(keys),
        group_dims=tuple(dims),
    )


def build_from_histplot_call(
    ax: Axes,
    data,
    x: Optional[str],
    y: Optional[str],
    hue: Optional[str],
    palette: Optional[Dict],
    stat: str,
    hue_order: Optional[List],
) -> BarValueMeta:
    """Build a ``BarValueMeta`` paired with the histplot's Rectangles.

    Histogram bars have no errorbars and the "value" of each bar is its
    height (or width, for horizontal). Hue assignment is inferred from
    each Rectangle's facecolor against the resolved palette map, same
    strategy the paint pass uses — ordering-agnostic under
    ``multiple="dodge"`` / ``"stack"``.
    """
    orient = "v" if x is not None else "h"

    rects = [p for p in ax.patches if _is_bar_rect(p)]

    bars: List[BarRecord] = []
    for rect in rects:
        value = rect.get_height() if orient == "v" else rect.get_width()
        hue_color: Optional[Tuple[float, float, float, float]] = None
        if palette is not None and palette:
            face = tuple(rect.get_facecolor())[:3]
            best_level = None
            best_dist = float("inf")
            for level, col in palette.items():
                target = to_rgba(col)[:3]
                dist = sum((a - b) ** 2 for a, b in zip(face, target))
                if dist < best_dist:
                    best_dist = dist
                    best_level = level
            if best_level is not None:
                hue_color = to_rgba(palette[best_level])
        if hue_color is None:
            # Strip face alpha — see note on the dodged-builder fallback.
            r, g, b, _ = rect.get_facecolor()
            hue_color = (r, g, b, 1.0)
        bars.append(BarRecord(
            patch=rect,
            value=float(value),
            err_low=None,
            err_high=None,
            hue_color=hue_color,
        ))
    return BarValueMeta(
        orient=orient,
        bars=bars,
        errorbar_kind=None,
        hue_active=hue is not None,
        owner_is_publiplots=True,
    )


VALID_BOX_STATS = {"median", "q1", "q3", "whisker_low", "whisker_high", "mean"}


def _compute_box_stats(values: np.ndarray, whis: float) -> Dict[str, float]:
    """Compute box-plot statistics from raw samples.

    Matches seaborn's convention: Q1 = 25th percentile, Q3 = 75th, whiskers
    extend to the most extreme sample within `whis * IQR` of Q1/Q3.
    """
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    low_cut = q1 - whis * iqr
    high_cut = q3 + whis * iqr
    inside = values[(values >= low_cut) & (values <= high_cut)]
    whisker_low = float(inside.min()) if len(inside) else q1
    whisker_high = float(inside.max()) if len(inside) else q3
    return {
        "median": float(np.median(values)),
        "q1": q1,
        "q3": q3,
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
        "mean": float(np.mean(values)),
    }


def _aggregate_box_stats(
    data,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    whis: float,
    *,
    source_frame,
) -> List[Dict]:
    """Group by (categorical_axis [, hue]) and compute box stats per group.

    Row ordering matches seaborn's dodge draw order: hue-outer, cat-inner.

    Each row carries its draw-order ``category`` / ``hue_value`` /
    ``hatch_value`` (boxes don't dodge on hatch, always ``None``) plus
    ``frame_row_index``: a *position* (integer offset) into
    ``source_frame``, not a pandas label, so
    ``source_frame.iloc[frame_row_index]`` resolves the caller's row for
    any index kind. When ``source_frame`` is ``None`` or the label does
    not map (rare), ``frame_row_index`` is ``None``.
    """
    value_col = y if categorical_axis == x else x
    spec = BarSplitSpec.resolve(
        x=x, y=y, hue=hue, hatch=None, categorical_axis=categorical_axis,
    )

    rows: List[Dict] = []
    for cat, h_val, _ht_val in spec.iter_draw_order(data):
        mask = data[categorical_axis] == cat
        if spec.split_hue is not None:
            mask = mask & (data[spec.split_hue] == h_val)
        vals = data.loc[mask, value_col].to_numpy()
        if len(vals) == 0:
            continue
        stats = _compute_box_stats(vals, whis)
        # Position within source_frame of the first row matching this group.
        matching = data.index[mask]
        frame_row_index: Optional[int] = None
        if source_frame is not None and len(matching):
            try:
                frame_row_index = int(source_frame.index.get_loc(matching[0]))
            except (KeyError, TypeError):
                frame_row_index = None
        stats["category"] = cat
        stats["hue_value"] = h_val
        stats["hatch_value"] = None
        stats["frame_row_index"] = frame_row_index
        rows.append(stats)
    return rows


def _cat_extents_from_artist(artist, ax: Axes, orient: str) -> Tuple[float, float]:
    """Return (center, half_width) on the categorical axis, in data coords."""
    tr = artist.get_transform() if hasattr(artist, "get_transform") else ax.transData
    # Works for both Patch (get_path) and Collection (get_paths()[0])
    if hasattr(artist, "get_path"):
        extents = artist.get_path().get_extents(tr)
    else:
        extents = artist.get_paths()[0].get_extents(tr)
    extents_data = extents.transformed(ax.transData.inverted())
    if orient == "v":
        lo, hi = extents_data.x0, extents_data.x1
    else:
        lo, hi = extents_data.y0, extents_data.y1
    return float((lo + hi) / 2.0), float(abs(hi - lo) / 2.0)


def _facecolor_of(artist) -> Tuple[float, float, float, float]:
    fc = artist.get_facecolor()
    # Collection.get_facecolor returns an array of shape (N, 4); patches return (4,)
    fc = np.asarray(fc)
    if fc.ndim == 2:
        fc = fc[0]
    return tuple(float(c) for c in fc[:4])


def _build_box_stats_meta(
    ax: Axes,
    data,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    palette: Optional[Dict],
    whis: float,
    artists: List,
    *,
    source_frame,
) -> BoxStatsMeta:
    """Shared builder for pp.boxplot / pp.violinplot.

    Stats are computed from raw data (exact, matching seaborn's whis rule
    for boxplot; violinplot shows the same underlying stats). Categorical
    positions come from the drawn artists so dodge groupings are honored.

    ``source_frame`` is required keyword-only: the caller's pre-copy
    DataFrame. The returned meta carries it so downstream annotate
    strategies can look up per-box source rows.
    """
    orient = "v" if categorical_axis == x else "h"
    agg = _aggregate_box_stats(
        data, x=x, y=y, hue=hue,
        categorical_axis=categorical_axis, whis=whis,
        source_frame=source_frame,
    )
    if len(artists) != len(agg):
        n = min(len(artists), len(agg))
        artists = artists[:n]
        agg = agg[:n]

    boxes: List[BoxStatsRecord] = []
    for i, (artist, row) in enumerate(zip(artists, agg)):
        center_pos, cat_half_width = _cat_extents_from_artist(artist, ax, orient)
        hue_color = None
        if hue is not None and palette is not None:
            key = row.get("hue_value")
            if key is not None and key in palette:
                hue_color = to_rgba(palette[key])
        if hue_color is None:
            hue_color = _facecolor_of(artist)
        boxes.append(BoxStatsRecord(
            center_pos=center_pos,
            cat_half_width=cat_half_width,
            stats={k: v for k, v in row.items() if k in VALID_BOX_STATS},
            hue_color=hue_color,
            category=row["category"],
            hue_value=row["hue_value"],
            hatch_value=row["hatch_value"],
            draw_index=i,
            frame_row_index=row["frame_row_index"],
        ))

    spec = BarSplitSpec.resolve(
        x=x, y=y, hue=hue, hatch=None, categorical_axis=categorical_axis,
    )
    keys: List[str] = [categorical_axis]
    dims: List[str] = ["cat"]
    if spec.split_hue is not None:
        keys.append(spec.split_hue)
        dims.append("hue")

    return BoxStatsMeta(
        orient=orient,
        boxes=boxes,
        hue_active=hue is not None,
        owner_is_publiplots=True,
        source_frame=source_frame,
        group_keys=tuple(keys),
        group_dims=tuple(dims),
    )


def build_from_boxplot_call(
    ax: Axes,
    data,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    palette: Optional[Dict],
    whis: float,
    *,
    source_frame,
) -> BoxStatsMeta:
    """Build a BoxStatsMeta paired with the boxplot's PathPatches.

    ``source_frame`` is required keyword-only; see ``_build_box_stats_meta``.
    """
    patches = [p for p in ax.patches if isinstance(p, PathPatch)]
    return _build_box_stats_meta(
        ax, data, x, y, hue, categorical_axis, palette, whis, patches,
        source_frame=source_frame,
    )


def build_from_violinplot_call(
    ax: Axes,
    data,
    x: str,
    y: str,
    hue: Optional[str],
    categorical_axis: str,
    palette: Optional[Dict],
    whis: float = 1.5,
    *,
    source_frame,
) -> BoxStatsMeta:
    """Build a BoxStatsMeta paired with the violinplot's fill collections.

    Violin draws one FillBetweenPolyCollection per violin; stats are the
    same as for boxplot (computed from raw data with matching whis rule).

    ``source_frame`` is required keyword-only; see ``_build_box_stats_meta``.
    """
    from matplotlib.collections import PolyCollection
    artists = [c for c in ax.collections if isinstance(c, PolyCollection)]
    return _build_box_stats_meta(
        ax, data, x, y, hue, categorical_axis, palette, whis, artists,
        source_frame=source_frame,
    )
