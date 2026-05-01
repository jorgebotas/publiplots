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
    resolve_anchor,
)


logger = logging.getLogger(__name__)


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
    anchor: str,
    offset: float,
    color,
    pad: float,
    **text_kws,
) -> List[Text]:
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
        rgba = resolve_color(bar, color, anchor, ax)
        label = _format_value(bar.value, fmt)
        t = ax.text(x, y, label, ha=ha, va=va, color=rgba, **text_kws)

        if anchor != "outside":
            bbox = bar.patch.get_window_extent(renderer)
            if fit_check(t, bbox, meta.orient, anchor, renderer) == "reanchor_outside":
                x2, y2, ha2, va2 = resolve_anchor(bar, "outside", meta.orient, offset, ax)
                rgba2 = resolve_color(bar, color, "outside", ax)
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
                        owner_is_publiplots=meta.owner_is_publiplots)
    return texts


def _maybe_expand_limits(
    ax: Axes,
    texts: List[Text],
    orient: str,
    pad_mm: float,
    owner_is_publiplots: bool,
) -> None:
    if not texts:
        return

    value_axis = "y" if orient == "v" else "x"
    autoscale_on = (ax.get_autoscaley_on() if value_axis == "y"
                    else ax.get_autoscalex_on())
    should_expand = owner_is_publiplots or autoscale_on

    renderer = _ensure_renderer(ax)
    inv = ax.transData.inverted()
    extents = [t.get_window_extent(renderer).transformed(inv) for t in texts]
    if value_axis == "y":
        need_min = min(e.y0 for e in extents)
        need_max = max(e.y1 for e in extents)
        get_lim, set_lim = ax.get_ylim, ax.set_ylim
    else:
        need_min = min(e.x0 for e in extents)
        need_max = max(e.x1 for e in extents)
        get_lim, set_lim = ax.get_xlim, ax.set_xlim

    cur_lo, cur_hi = get_lim()
    pad_data = mm_to_data(pad_mm, ax, axis=value_axis)

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
