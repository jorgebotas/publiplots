"""Value-label strategy for boxplots.

Labels box statistics (median by default; optionally Q1/Q3/whiskers/mean)
at each box. Anchor vocabulary is directional (top/bottom/left/right/center)
since boxes are small and "outside" vs "inside" is ambiguous for median-on-
the-median-line labels. Default anchor is "right" (beside the box), which
matches the most common journal style.
"""
from __future__ import annotations

import warnings
from typing import Iterable, List, Tuple

from matplotlib.axes import Axes
from matplotlib.text import Text

from publiplots.annotate._cache import BoxStatsMeta, BoxStatsRecord
from publiplots.annotate._color import resolve_color
from publiplots.annotate._positioning import (
    make_offset_transform,
    mm_to_data,
)


_VALID_ANCHORS = {"top", "bottom", "left", "right", "center"}
_DEFAULT_ANCHOR = "right"
_DEFAULT_STATS: Tuple[str, ...] = ("median",)



def _format_value(value: float, fmt: str) -> str:
    if "{" in fmt and "}" in fmt:
        return fmt.format(value)
    return format(value, fmt)


def _ensure_renderer(ax: Axes):
    canvas = ax.figure.canvas
    renderer = canvas.get_renderer() if hasattr(canvas, "get_renderer") else None
    if renderer is None:
        canvas.draw()
        renderer = canvas.get_renderer()
    return renderer


def _resolve_box_anchor(
    box: BoxStatsRecord,
    stat_name: str,
    stat_value: float,
    anchor: str,
    orient: str,
    offset_mm: float,
    ax: Axes,
) -> Tuple[float, float, float, float, str, str]:
    """Return (x, y, dx_mm, dy_mm, ha, va) for a label on one box statistic.

    (x, y) lives in data coords on the box edge or at the stat line —
    no offset applied. (dx_mm, dy_mm) is the mm displacement the caller
    should apply via `make_offset_transform`, so the pixel gap survives
    later limit changes. ``box.cat_half_width`` stays in data coords
    because the categorical axis uses integer category positions that
    don't rescale with ``sharey=True`` — only the mm padding needs to
    live in display space.
    """
    if orient == "v":
        px = box.center_pos
        py = stat_value
        if anchor == "top":
            return px, py, 0.0, +offset_mm, "center", "bottom"
        if anchor == "bottom":
            return px, py, 0.0, -offset_mm, "center", "top"
        if anchor == "right":
            return px + box.cat_half_width, py, +offset_mm, 0.0, "left", "center"
        if anchor == "left":
            return px - box.cat_half_width, py, -offset_mm, 0.0, "right", "center"
        if anchor == "center":
            return px, py, 0.0, 0.0, "center", "center"
    else:
        # orient == "h": value on x, categorical on y
        px = stat_value
        py = box.center_pos
        if anchor == "top":
            return px, py + box.cat_half_width, 0.0, +offset_mm, "center", "bottom"
        if anchor == "bottom":
            return px, py - box.cat_half_width, 0.0, -offset_mm, "center", "top"
        if anchor == "right":
            return px, py, +offset_mm, 0.0, "left", "center"
        if anchor == "left":
            return px, py, -offset_mm, 0.0, "right", "center"
        if anchor == "center":
            return px, py, 0.0, 0.0, "center", "center"
    raise ValueError(f"unreachable: unknown anchor {anchor!r}")


def _box_stats_strategy(
    ax: Axes,
    *,
    fmt: str,
    anchor,
    offset: float,
    color,
    pad: float,
    rotation: float = 0.0,
    **text_kws,
) -> List[Text]:
    # Pop strategy-specific opts before forwarding the rest to ax.text.
    stats_opt = text_kws.pop("stats", None)
    # Top-level `rotation` wins over one smuggled via text_kws; pop silently.
    text_kws.pop("rotation", None)

    if anchor is None:
        anchor = _DEFAULT_ANCHOR
    if anchor not in _VALID_ANCHORS:
        raise ValueError(
            f"box_stats anchor must be one of {sorted(_VALID_ANCHORS)}; "
            f"got {anchor!r}"
        )
    if stats_opt is None:
        stats_tuple = _DEFAULT_STATS
    else:
        stats_tuple = tuple(stats_opt)
    for s in stats_tuple:
        if s not in {"median", "q1", "q3", "whisker_low", "whisker_high", "mean"}:
            raise ValueError(
                f"unknown box stat {s!r}; known: "
                "['median', 'q1', 'q3', 'whisker_low', 'whisker_high', 'mean']"
            )

    meta = getattr(ax, "_publiplots_box_meta", None)
    if not isinstance(meta, BoxStatsMeta):
        warnings.warn(
            "pp.annotate: kind='box_stats' needs pp.boxplot- or "
            "pp.violinplot-owned axes; call pp.boxplot(..., annotate=True) "
            "or pp.violinplot(..., annotate=True) instead of annotating a "
            "foreign axes.",
            UserWarning,
            stacklevel=3,
        )
        return []
    if not meta.boxes:
        return []

    _ensure_renderer(ax)
    texts: List[Text] = []
    for box in meta.boxes:
        for stat_name in stats_tuple:
            if stat_name not in box.stats:
                continue
            stat_value = box.stats[stat_name]
            x, y, dx_mm, dy_mm, ha, va = _resolve_box_anchor(
                box, stat_name, stat_value, anchor, meta.orient, offset, ax,
            )
            rgba = resolve_color(
                _BoxStandIn(box),
                color, "outside", ax,
                hue_active=meta.hue_active,
            )
            label = _format_value(stat_value, fmt)
            t = ax.text(
                x, y, label, ha=ha, va=va, color=rgba,
                rotation=rotation,
                transform=make_offset_transform(ax, dx_mm, dy_mm),
                **text_kws,
            )
            texts.append(t)

    _maybe_expand_box_limits(ax, texts, anchor, meta.orient, pad_mm=pad,
                             owner_is_publiplots=meta.owner_is_publiplots,
                             rotation=rotation)
    return texts


class _BoxStandIn:
    """Same shim pattern as point_values — resolve_color only needs
    `hue_color` and `patch.get_facecolor/get_edgecolor`; for the 'outside'
    anchor used here, neither face/edge path is read."""

    def __init__(self, box: BoxStatsRecord):
        self.patch = _PatchShim(box.hue_color)
        self.hue_color = box.hue_color


class _PatchShim:
    def __init__(self, rgba):
        self._rgba = rgba or (0, 0, 0, 1)

    def get_facecolor(self):
        return self._rgba

    def get_edgecolor(self):
        return self._rgba


_MAX_EXPAND_ITERS = 4


def _maybe_expand_box_limits(
    ax: Axes,
    texts: List[Text],
    anchor: str,
    orient: str,
    pad_mm: float,
    owner_is_publiplots: bool,
    rotation: float = 0.0,
) -> None:
    if not texts:
        return

    # Which axis matches the unrotated anchor direction?
    # top/bottom → y; left/right → x; center → neither.
    if anchor in ("top", "bottom"):
        primary_axis = "y"
    elif anchor in ("left", "right"):
        primary_axis = "x"
    else:
        primary_axis = None

    rotated = rotation % 360.0 != 0.0

    # Under rotation, a rotated label can spill onto either axis regardless of
    # anchor direction; expand both sides.
    if primary_axis is None and not rotated:
        return

    axes_to_expand = []
    if primary_axis is not None:
        axes_to_expand.append(primary_axis)
    if rotated:
        other = "x" if primary_axis == "y" else "y"
        if primary_axis is None:
            axes_to_expand.extend(["x", "y"])
        elif other not in axes_to_expand:
            axes_to_expand.append(other)

    renderer = _ensure_renderer(ax)
    for _ in range(_MAX_EXPAND_ITERS):
        # Re-fetch inverted transform each iter — set_xlim/ylim invalidates it.
        ax.figure.canvas.draw()
        inv = ax.transData.inverted()
        extents = [t.get_window_extent(renderer).transformed(inv) for t in texts]
        any_changed = False
        for ax_name in axes_to_expand:
            if _expand_box_axis(
                ax, extents, axis=ax_name, pad_mm=pad_mm,
                owner_is_publiplots=owner_is_publiplots,
            ):
                any_changed = True
        if not any_changed:
            break


def _expand_box_axis(
    ax: Axes,
    extents: list,
    axis: str,
    pad_mm: float,
    owner_is_publiplots: bool,
) -> bool:
    """Expand one axis to fit extents. Return True if limits changed."""
    autoscale_on = (ax.get_autoscaley_on() if axis == "y"
                    else ax.get_autoscalex_on())
    should_expand = owner_is_publiplots or autoscale_on

    if axis == "y":
        need_min = min(e.y0 for e in extents)
        need_max = max(e.y1 for e in extents)
        get_lim, set_lim = ax.get_ylim, ax.set_ylim
    else:
        need_min = min(e.x0 for e in extents)
        need_max = max(e.x1 for e in extents)
        get_lim, set_lim = ax.get_xlim, ax.set_xlim

    cur_lo, cur_hi = get_lim()
    pad_data = mm_to_data(pad_mm, ax, axis=axis)

    inverted = cur_lo > cur_hi
    cur_min = min(cur_lo, cur_hi)
    cur_max = max(cur_lo, cur_hi)

    if should_expand:
        new_min = min(cur_min, need_min - pad_data)
        new_max = max(cur_max, need_max + pad_data)
        if new_min == cur_min and new_max == cur_max:
            return False
        if inverted:
            set_lim(new_max, new_min)
        else:
            set_lim(new_min, new_max)
        return True
    else:
        if need_min < cur_min or need_max > cur_max:
            warnings.warn(
                "pp.annotate: labels clipped; autoscale is off on this axis",
                UserWarning,
                stacklevel=3,
            )
        return False
