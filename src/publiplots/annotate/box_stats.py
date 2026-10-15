"""Value-label strategy for boxplots.

Labels box statistics (median by default; optionally Q1/Q3/whiskers/mean)
at each box. Anchor vocabulary is directional (top/bottom/left/right/center),
plus "inside": on the forward side of the stat line, within the body, with
half-bodies anchored against their seam. Default anchor is "right" (beside
the box), which matches the most common journal style.
"""
from __future__ import annotations

import warnings
from typing import ClassVar, FrozenSet, Iterable, List, Tuple

from matplotlib.axes import Axes
from matplotlib.text import Text

from publiplots.annotate._cache import BoxStatsMeta, BoxStatsRecord
from publiplots.annotate._color import resolve_color
from publiplots.annotate._positioning import (
    AnchorTuple,
    make_offset_transform,
)
from publiplots.annotate._shared import (
    compute_axes_to_expand_directional,
    ensure_renderer as _ensure_renderer,
    format_value as _format_value,
    maybe_expand_limits,
)


_VALID_ANCHORS = {"top", "bottom", "left", "right", "center", "inside"}
_DEFAULT_ANCHOR = "right"
_DEFAULT_STATS: Tuple[str, ...] = ("median",)


def _toward_display(ax: Axes, axis: str, half: str) -> float:
    """+1 or -1: the display direction from a half-body's seam into it."""
    inverted = ax.xaxis_inverted() if axis == "x" else ax.yaxis_inverted()
    toward_data = 1.0 if half == "high" else -1.0
    return -toward_data if inverted else toward_data


def _resolve_inside(
    box: BoxStatsRecord,
    stat_value: float,
    orient: str,
    offset_mm: float,
    ax: Axes,
    rotated: bool,
) -> Tuple[float, float, float, float, str, str]:
    """``anchor="inside"``: offset_mm on the forward side of the stat line
    (above it for orient="v", right of it for "h"), never flipping. A full
    body centres the label; a half-body puts it against its seam, on its
    own side, offset_mm from the seam.

    The seam is derived from the body's own extents, never from the
    category's integer position: a split violin without hue, for one, has
    its seam at the slot edge.
    """
    if box.half is None:
        if orient == "v":
            # A label turned 90 deg aligns by its rotated bounding box, so
            # va="bottom" stands that box on the line.
            va = "bottom" if rotated else "baseline"
            return box.center_pos, stat_value, 0.0, +offset_mm, "center", va
        return stat_value, box.center_pos, +offset_mm, 0.0, "left", "center"

    seam = (box.center_pos - box.cat_half_width if box.half == "high"
            else box.center_pos + box.cat_half_width)
    if orient == "v":
        s = _toward_display(ax, "x", box.half)
        return (seam, stat_value, s * offset_mm, +offset_mm,
                "left" if s > 0 else "right", "baseline")
    s = _toward_display(ax, "y", box.half)
    return (stat_value, seam, +offset_mm, s * offset_mm,
            "left", "bottom" if s > 0 else "top")


def _resolve_box_anchor(
    box: BoxStatsRecord,
    stat_name: str,
    stat_value: float,
    anchor: str,
    orient: str,
    offset_mm: float,
    ax: Axes,
    rotated: bool = False,
) -> Tuple[float, float, float, float, str, str]:
    """Return (x, y, dx_mm, dy_mm, ha, va) for a label on one box statistic.

    (x, y) lives in data coords on the box edge or at the stat line —
    no offset applied. (dx_mm, dy_mm) is the mm displacement the caller
    should apply via `make_offset_transform`, so the pixel gap survives
    later limit changes. ``box.cat_half_width`` stays in data coords
    because the categorical axis uses integer category positions that
    don't rescale with ``sharey=True`` — only the mm padding needs to
    live in display space.

    ``rotated`` (label turned 90 deg) only affects ``anchor="inside"``.
    """
    if anchor == "inside":
        return _resolve_inside(box, stat_value, orient, offset_mm, ax, rotated)
    # va="baseline" for "above-mark" anchors: see _positioning.resolve_anchor
    # for the rationale (digit-only labels have invisible descender slack
    # under va="bottom"; va="baseline" matches the horizontal-anchor gap).
    if orient == "v":
        px = box.center_pos
        py = stat_value
        if anchor == "top":
            return px, py, 0.0, +offset_mm, "center", "baseline"
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
            return px, py + box.cat_half_width, 0.0, +offset_mm, "center", "baseline"
        if anchor == "bottom":
            return px, py - box.cat_half_width, 0.0, -offset_mm, "center", "top"
        if anchor == "right":
            return px, py, +offset_mm, 0.0, "left", "center"
        if anchor == "left":
            return px, py, -offset_mm, 0.0, "right", "center"
        if anchor == "center":
            return px, py, 0.0, 0.0, "center", "center"
    raise ValueError(f"unreachable: unknown anchor {anchor!r}")


class BoxAnchorResolver:
    """Adapter to _resolve_box_anchor. Box-stats records carry per-stat
    values inside `record.stats`; the custom strategy labels a single
    chosen stat (default: "median") whose value is passed to the resolver
    as stat_value."""
    VALID_ANCHORS: ClassVar[FrozenSet[str]] = frozenset({
        "top", "bottom", "left", "right", "center", "inside",
    })
    DEFAULT_ANCHOR: ClassVar[str] = "top"

    # The custom strategy always uses the median by default; a callable
    # label path can compute from record.stats directly. Pin the stat name
    # the resolver receives.
    STAT_FOR_POSITIONING: ClassVar[str] = "median"

    def resolve(self, record, anchor, orient, offset_mm, ax) -> AnchorTuple:
        stat_value = record.stats[self.STAT_FOR_POSITIONING]
        return _resolve_box_anchor(
            record, self.STAT_FOR_POSITIONING, stat_value,
            anchor, orient, offset_mm, ax,
        )


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

    rotated = abs(rotation) % 180 == 90
    # "inside" sits on the fill, so it alone takes the contrast path; every
    # other anchor keeps "outside" and boxplot output stays byte-identical.
    color_anchor = "inside" if anchor == "inside" else "outside"
    expand_anchor = (("top" if meta.orient == "v" else "right")
                     if anchor == "inside" else anchor)

    _ensure_renderer(ax)
    texts: List[Text] = []
    for box in meta.boxes:
        for stat_name in stats_tuple:
            if stat_name not in box.stats:
                continue
            stat_value = box.stats[stat_name]
            x, y, dx_mm, dy_mm, ha, va = _resolve_box_anchor(
                box, stat_name, stat_value, anchor, meta.orient, offset, ax,
                rotated=rotated,
            )
            rgba = resolve_color(
                _BoxStandIn(box),
                color, color_anchor, ax,
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

    maybe_expand_limits(
        ax, texts,
        axes_to_expand=compute_axes_to_expand_directional(
            expand_anchor, meta.orient, rotation,
        ),
        pad_mm=pad, owner_is_publiplots=meta.owner_is_publiplots,
    )
    return texts


class _BoxStandIn:
    """Same shim pattern as point_values. resolve_color reads `hue_color`
    and `patch.get_facecolor/get_edgecolor`; the face is the drawn one, so
    a label inside the mark contrasts with what is actually under it."""

    def __init__(self, box: BoxStatsRecord):
        face = box.face_color if box.face_color is not None else box.hue_color
        self.patch = _PatchShim(face, box.hue_color)
        self.hue_color = box.hue_color


class _PatchShim:
    def __init__(self, face, edge):
        self._face = face or (0, 0, 0, 1)
        self._edge = edge or self._face

    def get_facecolor(self):
        return self._face

    def get_edgecolor(self):
        return self._edge
