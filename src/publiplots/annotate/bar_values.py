"""Value-label strategy for barplots."""
from __future__ import annotations

import logging
import math
import warnings
from typing import List

from matplotlib.axes import Axes
from matplotlib.text import Text

from publiplots.annotate._cache import BarValueMeta, _introspect
from publiplots.annotate._color import resolve_color
from publiplots.annotate._positioning import (
    fit_check,
    mm_to_data,
    remap_alignment_for_rotation,
    resolve_anchor,
)


logger = logging.getLogger(__name__)


_VALID_ANCHORS = {"outside", "inside", "base", "center"}
_DEFAULT_ANCHOR = "outside"


def _get_or_introspect(ax: Axes) -> BarValueMeta:
    meta = getattr(ax, "_publiplots_bar_meta", None)
    if isinstance(meta, BarValueMeta):
        return meta
    return _introspect(ax)


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


def _bar_values_strategy(
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
    if anchor is None:
        anchor = _DEFAULT_ANCHOR
    if anchor not in _VALID_ANCHORS:
        raise ValueError(
            f"bar_values anchor must be one of {sorted(_VALID_ANCHORS)}; "
            f"got {anchor!r}"
        )

    # Top-level `rotation` wins over one smuggled via text_kws; pop silently.
    text_kws.pop("rotation", None)

    meta = _get_or_introspect(ax)
    if not meta.bars:
        warnings.warn("pp.annotate: no bars found on axes", UserWarning, stacklevel=3)
        return []

    renderer = _ensure_renderer(ax)
    texts: List[Text] = []

    for bar in meta.bars:
        if math.isnan(bar.value):
            continue
        x, y, ha, va = resolve_anchor(bar, anchor, meta.orient, offset, ax)
        ha, va = remap_alignment_for_rotation(ha, va, rotation)
        rgba = resolve_color(bar, color, anchor, ax, hue_active=meta.hue_active)
        label = _format_value(bar.value, fmt)
        t = ax.text(x, y, label, ha=ha, va=va, color=rgba,
                    rotation=rotation, **text_kws)

        if anchor != "outside":
            bbox = bar.patch.get_window_extent(renderer)
            if fit_check(t, bbox, meta.orient, anchor, renderer) == "reanchor_outside":
                x2, y2, ha2, va2 = resolve_anchor(bar, "outside", meta.orient, offset, ax)
                ha2, va2 = remap_alignment_for_rotation(ha2, va2, rotation)
                rgba2 = resolve_color(bar, color, "outside", ax, hue_active=meta.hue_active)
                t.set_position((x2, y2))
                t.set_ha(ha2)
                t.set_va(va2)
                t.set_color(rgba2)
                logger.debug(
                    "pp.annotate: bar value=%s label re-anchored to 'outside'",
                    bar.value,
                )
        texts.append(t)

    _maybe_expand_limits(ax, texts, meta.orient, pad_mm=pad,
                        owner_is_publiplots=meta.owner_is_publiplots,
                        rotation=rotation)
    return texts


def _maybe_expand_limits(
    ax: Axes,
    texts: List[Text],
    orient: str,
    pad_mm: float,
    owner_is_publiplots: bool,
    rotation: float = 0.0,
) -> None:
    if not texts:
        return

    value_axis = "y" if orient == "v" else "x"
    cat_axis = "x" if value_axis == "y" else "y"

    # Force a fresh draw so the just-created Text artists have accurate
    # window extents; without this, get_window_extent returns a stale bbox
    # based on the pre-layout state, which undershoots `need_max` and leaves
    # labels clipping the axis frame on already-drawn figures.
    ax.figure.canvas.draw()
    renderer = _ensure_renderer(ax)
    inv = ax.transData.inverted()
    extents = [t.get_window_extent(renderer).transformed(inv) for t in texts]

    _expand_axis(
        ax, extents, axis=value_axis, pad_mm=pad_mm,
        owner_is_publiplots=owner_is_publiplots,
    )
    # Rotated labels can extend beyond the categorical axis too. Only touch
    # that axis when rotation actually changes the bbox shape; at rotation=0
    # the bbox width is the text width, which is usually much smaller than a
    # bar's categorical slot — expanding would be noise.
    if rotation % 360.0 != 0.0:
        _expand_axis(
            ax, extents, axis=cat_axis, pad_mm=pad_mm,
            owner_is_publiplots=owner_is_publiplots,
        )


def _expand_axis(
    ax: Axes,
    extents: list,
    axis: str,
    pad_mm: float,
    owner_is_publiplots: bool,
) -> None:
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

    if should_expand:
        set_lim(min(cur_lo, need_min - pad_data),
                max(cur_hi, need_max + pad_data))
    else:
        if need_min < cur_lo or need_max > cur_hi:
            warnings.warn(
                "pp.annotate: labels clipped; autoscale is off on this axis",
                UserWarning,
                stacklevel=3,
            )
