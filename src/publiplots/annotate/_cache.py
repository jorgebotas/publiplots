"""Private cache contract between plot functions and annotation strategies.

`BarValueMeta` is attached to an Axes as `ax._publiplots_bar_meta` by
`pp.barplot(..., annotate=...)`. When absent (foreign axes), `_introspect`
reconstructs an equivalent meta by walking `ax.patches` and `ax.lines`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, List, Literal, Optional, Tuple

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


RGBA = Tuple[float, float, float, float]


@dataclass(frozen=True)
class BarRecord:
    """One drawn bar plus the metadata annotate strategies need.

    Fields:
        patch: the matplotlib Rectangle drawn for this bar.
        value: aggregated bar height (orient='v') or width ('h'); NaN-possible.
        err_low, err_high: errorbar extents on the value axis, if present.
        hue_color: the palette color assigned to this bar (RGBA).
        anchor_override: if set, overrides the caller's anchor for this bar
            (used by stacked/gain barplots to pin labels inside-vs-outside).
        category: the x-axis group key (None on foreign axes).
        hue_value: the hue group key, if hue is active (else None).
        hatch_value: the hatch group key, if hatch is active (else None).
        draw_index: 0-based position in the draw order.
        frame_row_index: index of a representative source-DataFrame row for
            this group (first row in the group); None on foreign axes.
    """
    patch: Rectangle
    value: float
    err_low: Optional[float]
    err_high: Optional[float]
    hue_color: Optional[RGBA]
    anchor_override: Optional[str] = None
    category: Optional[Hashable] = None
    hue_value: Optional[Hashable] = None
    hatch_value: Optional[Hashable] = None
    draw_index: int = 0
    frame_row_index: Optional[int] = None


@dataclass
class BarValueMeta:
    orient: Literal["v", "h"]
    bars: List[BarRecord]
    errorbar_kind: Optional[str]
    hue_active: bool
    owner_is_publiplots: bool
    source_frame: Optional[Any] = None          # pandas DataFrame (kept Any to avoid import)
    group_keys: Optional[Tuple[str, ...]] = None
    # Companion to group_keys identifying each key's dimension: one of
    # "cat", "hue", "hatch". Parallel tuple with the same length and order as
    # `group_keys`. Populated by publiplots-owned builders; left `None` by
    # `_introspect` (foreign axes have no semantic dims to report).
    group_dims: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class PointRecord:
    """A single point's anchor + aggregated value + errorbar extents.

    Unlike BarRecord, a point has no matplotlib artist for the *mark itself*
    (seaborn merges all points of one hue into a single Line2D). We track
    the data-coord position directly and the palette color that was used.

    Fields:
        xy: (x, y) in data coords; y is the value for orient="v".
        value: aggregated estimate (mean/median); NaN-possible.
        err_low, err_high: errorbar extents on the value axis, if present.
        hue_color: palette color assigned by pointplot.
        category: x-axis group key (None on foreign axes).
        hue_value: hue group key if hue is active (else None).
        hatch_value: hatch group key if hatch is active (else None). Points
            don't dodge on hatch today, so this is always None; the field
            exists for shape-symmetry with BarRecord.
        draw_index: 0-based position in the draw order.
        frame_row_index: position of the first matching source-DataFrame row.
    """
    xy: Tuple[float, float]
    value: float
    err_low: Optional[float]
    err_high: Optional[float]
    hue_color: Optional[RGBA]
    category: Optional[Hashable] = None
    hue_value: Optional[Hashable] = None
    hatch_value: Optional[Hashable] = None
    draw_index: int = 0
    frame_row_index: Optional[int] = None


@dataclass
class PointValueMeta:
    orient: Literal["v", "h"]
    points: List[PointRecord]
    errorbar_kind: Optional[str]
    hue_active: bool
    owner_is_publiplots: bool
    source_frame: Optional[Any] = None
    group_keys: Optional[Tuple[str, ...]] = None
    group_dims: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class BoxStatsRecord:
    """Statistics computed for a single box. Same field set as PointRecord
    plus box-specific center_pos / cat_half_width / stats dict.

    Fields:
        center_pos: categorical-axis coordinate of the box's center.
        cat_half_width: half-extent on the categorical axis in data coords.
        stats: dict mapping stat name to its value-axis coordinate.
        hue_color: palette color assigned by boxplot/violinplot.
        category/hue_value/hatch_value/draw_index/frame_row_index: same
            semantics as in BarRecord; populated by the builder.
        face_color: the artist's drawn facecolor, alpha included — what a
            label inside the mark actually sits on. Differs from hue_color,
            the palette colour at full opacity.
        artist: the drawn mark — a violin's PolyCollection, a filled box's
            PathPatch, or an unfilled box's closed Line2D outline. Pairs
            per-mark decorations to their mark explicitly.
        half: for a half-body (split violin, side= violin), which side of
            its seam it occupies on the categorical axis, in data
            coordinates; None for a full body.
    """
    center_pos: float
    cat_half_width: float
    stats: dict
    hue_color: Optional[RGBA]
    category: Optional[Hashable] = None
    hue_value: Optional[Hashable] = None
    hatch_value: Optional[Hashable] = None
    draw_index: int = 0
    frame_row_index: Optional[int] = None
    face_color: Optional[RGBA] = None
    artist: Optional[Any] = None
    half: Optional[Literal["low", "high"]] = None


@dataclass
class BoxStatsMeta:
    orient: Literal["v", "h"]
    boxes: List[BoxStatsRecord]
    hue_active: bool
    owner_is_publiplots: bool
    source_frame: Optional[Any] = None
    group_keys: Optional[Tuple[str, ...]] = None
    group_dims: Optional[Tuple[str, ...]] = None
    # True when some category has two or more full bodies side by side.
    dodged: bool = False


def _is_bar_rect(p) -> bool:
    """True for a Rectangle that is a real bar rather than an empty slot.

    Seaborn/matplotlib draw negative-valued bars as Rectangles with a
    negative ``get_height()`` (or ``get_width()`` for horizontal orient):
    ``x, y=0, height=-12.3`` instead of ``x, y=-12.3, height=12.3``.
    A strict ``> 0`` check would silently drop those bars from meta,
    causing annotate strategies to skip every negative bar.

    Only *fully* degenerate rects are rejected — zero on **both** axes.
    That is the shape *raw seaborn* emits for a ``(category, hue)``
    combination with no observations: ``x=0, y=0, width=0, height=0``.
    ``BarSplitSpec.iter_draw_order`` skips those combos too, so the rect
    and group lists stay aligned on either path. Note ``pp.barplot`` does
    not itself produce such rects (its category preparation means seaborn
    sees no empty combo), so this branch only bites via `_introspect` on a
    foreign ``sns.barplot`` axes — which is where the test for it lives.

    A single zero extent is NOT degenerate. A group whose aggregate is
    exactly 0 is a real group that happens to be zero, and matplotlib
    draws it with a full-size categorical extent and a zero value extent.
    Rejecting it desynced the positional pairing between rects and group
    keys in `_builders.build_from_barplot_call`, shifting every label onto
    its neighbour's bar and dropping the last category — see issue #199.
    """
    # Lazy import to avoid a cycle: utils.rounding pulls in publiplots
    # internals at import time.
    from publiplots.utils.rounding import _RoundedBarPatch

    if not isinstance(p, (Rectangle, _RoundedBarPatch)):
        return False
    return p.get_width() != 0 or p.get_height() != 0


def _iter_error_segments(ax: Axes, artists: Optional[List] = None):
    """Yield (xs, ys) arrays for every candidate errorbar segment on `ax`.

    ``artists`` restricts the scan to a known set of Line2D / LineCollection
    objects — the ones the current drawing call produced. Pass it whenever
    the axes may hold errorbars from an earlier call: the matcher below takes
    the *first* aligned segment, so an earlier call's errorbar at the same
    categorical position wins and anchors this call's label at the wrong
    value. Defaults to everything on ``ax``.

    Matplotlib/seaborn render errorbars as either:
      - Line2D artists in `ax.lines` (e.g., seaborn's bar/point errorbars), or
      - a LineCollection in `ax.collections` (matplotlib's `ax.bar(yerr=...)`
        since ~3.6 — segments packed as (2, 2) arrays).

    When ``capsize > 0``, seaborn packs the whole errorbar (bottom cap +
    vertical stem + top cap) into a SINGLE Line2D whose data is separated by
    ``nan`` between sub-segments. We split each line on those ``nan`` breaks
    and yield each finite contiguous run independently, so the vertical stem
    run survives for the matcher. A run shorter than two points (or an
    all-``nan`` line, as seaborn emits for single-sample groups) yields
    nothing, leaving ``err_low/err_high`` as ``None``.
    """
    import math as _math

    def _isnan(v) -> bool:
        try:
            return _math.isnan(float(v))
        except (TypeError, ValueError):
            return True  # non-numeric → treat as a break, never emit

    def _finite_runs(xs, ys):
        """Split (xs, ys) into contiguous runs free of nan on either axis."""
        run_xs: List[float] = []
        run_ys: List[float] = []
        for x, y in zip(xs, ys):
            if _isnan(x) or _isnan(y):
                if len(run_xs) >= 2:
                    yield run_xs, run_ys
                run_xs, run_ys = [], []
            else:
                run_xs.append(x)
                run_ys.append(y)
        if len(run_xs) >= 2:
            yield run_xs, run_ys

    candidates = (list(ax.lines) + list(ax.collections)
                  if artists is None else list(artists))
    for ln in candidates:
        if not isinstance(ln, Line2D):
            continue
        yield from _finite_runs(ln.get_xdata(), ln.get_ydata())
    for col in candidates:
        if not isinstance(col, LineCollection):
            continue
        for seg in col.get_segments():
            if len(seg) < 2:
                continue
            yield from _finite_runs(seg[:, 0], seg[:, 1])


def _match_errorbars(
    ax: Axes,
    rects: List[Rectangle],
    orient: Literal["v", "h"],
    artists: Optional[List] = None,
) -> List[Tuple[Optional[float], Optional[float]]]:
    """For each rect, find an errorbar line whose midpoint aligns with the bar's center.

    Matplotlib/seaborn errorbars are drawn as short Line2D segments — either
    whole vertical/horizontal bars or individual cap segments. We match by
    proximity of the segment's midpoint to each rect's center on the
    categorical axis.

    ``artists`` scopes the candidate segments to the current call's own
    errorbars; see `_iter_error_segments`.
    """
    if not rects:
        return []
    if orient == "v":
        widths = [r.get_width() for r in rects]
        tol = 0.5 * min(widths) if widths else 0.0
    else:
        heights = [r.get_height() for r in rects]
        tol = 0.5 * min(heights) if heights else 0.0

    segments = list(_iter_error_segments(ax, artists))

    result: List[Tuple[Optional[float], Optional[float]]] = []
    for r in rects:
        if orient == "v":
            center = r.get_x() + r.get_width() / 2.0
            # Find vertical segment whose x is near bar center (single vertical
            # bar, not the caps, which are horizontal).
            low = high = None
            for xs, ys in segments:
                if max(xs) - min(xs) < 1e-6 and abs(xs[0] - center) <= tol:
                    low = min(ys)
                    high = max(ys)
                    break
            result.append((low, high))
        else:
            center = r.get_y() + r.get_height() / 2.0
            low = high = None
            for xs, ys in segments:
                if max(ys) - min(ys) < 1e-6 and abs(ys[0] - center) <= tol:
                    low = min(xs)
                    high = max(xs)
                    break
            result.append((low, high))
    return result


def _infer_orient(rects: List[Rectangle]) -> Literal["v", "h"]:
    """All bars share width → vertical; all share height → horizontal."""
    if not rects:
        return "v"
    widths = [r.get_width() for r in rects]
    heights = [r.get_height() for r in rects]
    w_spread = max(widths) - min(widths)
    h_spread = max(heights) - min(heights)
    if w_spread != h_spread:
        return "v" if w_spread < h_spread else "h"

    # Extents tie. That happens when every bar's value is identical — and in
    # particular when every value is 0, where the varying extent carries no
    # signal at all. Fall back to where the bars are *positioned*: they march
    # along the categorical axis, so the axis whose origins spread is the
    # categorical one. Without this, all-zero horizontal bars inferred "v"
    # and reported each bar's thickness as its value.
    xs = [r.get_x() for r in rects]
    ys = [r.get_y() for r in rects]
    x_spread = max(xs) - min(xs)
    y_spread = max(ys) - min(ys)
    if x_spread != y_spread:
        return "v" if x_spread > y_spread else "h"
    return "v"  # single bar, or genuinely square — no signal either way


def _introspect(ax: Axes) -> BarValueMeta:
    """Build a BarValueMeta from an already-drawn Axes."""
    # Two-step, because the value axis is not known until the orientation is.
    # `_is_bar_rect` keeps a bar whose value is exactly 0 (#199), which is
    # right for a bar chart but wrong for a histogram: an empty bin is a
    # full-width rect with a zero value extent, and labelling every gap in a
    # sparse histogram with "0" is noise. Infer the orientation from every
    # bar-shaped rect, then drop the ones with no value extent.
    candidates = [p for p in ax.patches if _is_bar_rect(p)]
    orient = _infer_orient(candidates)
    rects = [
        p for p in candidates
        if (p.get_height() if orient == "v" else p.get_width()) != 0
    ]
    err_by_bar = _match_errorbars(ax, rects, orient)
    bars: List[BarRecord] = []
    for i, (r, (err_low, err_high)) in enumerate(zip(rects, err_by_bar)):
        value = r.get_height() if orient == "v" else r.get_width()
        bars.append(BarRecord(
            patch=r,
            value=float(value),
            err_low=err_low,
            err_high=err_high,
            hue_color=tuple(r.get_facecolor()),
            anchor_override=None,
            category=None,
            hue_value=None,
            hatch_value=None,
            draw_index=i,
            frame_row_index=None,
        ))
    return BarValueMeta(
        orient=orient,
        bars=bars,
        errorbar_kind=None,
        hue_active=False,
        owner_is_publiplots=False,
        source_frame=None,
        group_keys=None,
        group_dims=None,
    )


@dataclass(frozen=True)
class WedgeRecord:
    """One drawn wedge plus the metadata annotate strategies need.

    Unlike BarRecord, `patch` is held from construction rather than
    recovered: publiplots draws every Wedge itself, so there is no step
    that scans the axes and pairs records to artists by index. That step
    is where #103, #199, #236, #241, #254 and #262 all lived.

    Fields:
        patch: the matplotlib Wedge drawn for this node.
        path: keys from the root ring to this node.
        value: aggregated value; for an internal node, the sum of children.
        fraction_of_total: share of the whole chart.
        fraction_of_parent: share of this node's parent; equal to
            fraction_of_total at depth 0.
        theta0, theta1: angular extent in degrees, theta0 < theta1.
        r_inner, r_outer: radial band in axes data units, as drawn: read
            off the patch after the gap and any explode.
        depth: 0 for the innermost ring.
        hue_color: the RGB the wedge was filled with. Named to match
            BarRecord/PointRecord so `_color.resolve_color` accepts a
            WedgeRecord unchanged -- it reads `.patch` and `.hue_color`,
            and its `anchor in ("inside", "center", "base")` branch already
            covers the wedge anchor vocabulary.
        hatch: the hatch pattern, or None.
        pull: the ``pull`` the wedge was drawn with, as a fraction of
            the radius. A dict ``pull`` translates the wedge, so
            ``patch.center`` carries it; a scalar one is an even explode
            that leaves ``patch.center`` on the chart's centre and is
            already folded into ``r_inner`` and ``r_outer``. Either way,
            geometry is read from ``patch.center`` and the radii, never
            recomputed from this.
    """

    patch: Any
    path: Tuple[Any, ...]
    value: float
    fraction_of_total: float
    fraction_of_parent: float
    theta0: float
    theta1: float
    r_inner: float
    r_outer: float
    depth: int
    hue_color: Tuple[float, float, float]
    hatch: Optional[str]
    pull: float


@dataclass
class WedgeMeta:
    """Everything the wedge_values strategy needs about one plot call."""

    wedges: List[WedgeRecord]
    total: float
    donut: float
    center: Tuple[float, float]
    hue_active: bool
    hue_continuous: bool
    levels: Tuple[str, ...]
    source_frame: Optional[Any] = None
