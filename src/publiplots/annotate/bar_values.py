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
    make_offset_transform,
    mm_to_data,
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
        bar_anchor = bar.anchor_override if bar.anchor_override is not None else anchor
        x, y, dx_mm, dy_mm, ha, va = resolve_anchor(
            bar, bar_anchor, meta.orient, offset, ax,
        )
        rgba = resolve_color(bar, color, bar_anchor, ax, hue_active=meta.hue_active)
        label = _format_value(bar.value, fmt)
        t = ax.text(
            x, y, label, ha=ha, va=va, color=rgba,
            rotation=rotation,
            transform=make_offset_transform(ax, dx_mm, dy_mm),
            **text_kws,
        )

        if bar_anchor != "outside":
            bbox = bar.patch.get_window_extent(renderer)
            if fit_check(t, bbox, meta.orient, bar_anchor, renderer) == "reanchor_outside":
                x2, y2, dx2, dy2, ha2, va2 = resolve_anchor(
                    bar, "outside", meta.orient, offset, ax,
                )
                rgba2 = resolve_color(bar, color, "outside", ax, hue_active=meta.hue_active)
                t.set_position((x2, y2))
                t.set_transform(make_offset_transform(ax, dx2, dy2))
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


# Expanding a limit changes the data→display scale, which shifts every text
# artist's bbox in data coords even though its pixel extent is unchanged.
# One pass always undershoots by a bounded geometric factor; 4 iterations
# is overkill in practice — most plots converge in 2.
_MAX_EXPAND_ITERS = 4


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

    # Wide labels (long text at rotation=0, tall text at rotation=90) can
    # spill onto the categorical axis too. `_expand_axis` is a no-op when
    # labels already fit, so we always check both axes.
    axes_to_expand = [value_axis, cat_axis]

    renderer = _ensure_renderer(ax)
    for _ in range(_MAX_EXPAND_ITERS):
        # Force a fresh draw so get_window_extent returns a bbox consistent
        # with the most recent limits; without this we'd loop on stale data.
        # Re-fetch `inv` every iteration: set_xlim/ylim invalidates transData,
        # and a cached inverted transform silently returns wrong data coords.
        ax.figure.canvas.draw()
        inv = ax.transData.inverted()
        extents = [t.get_window_extent(renderer).transformed(inv) for t in texts]
        any_changed = False
        for ax_name in axes_to_expand:
            if _expand_axis(
                ax, extents, axis=ax_name, pad_mm=pad_mm,
                owner_is_publiplots=owner_is_publiplots,
            ):
                any_changed = True
        if not any_changed:
            break


def _expand_axis(
    ax: Axes,
    extents: list,
    axis: str,
    pad_mm: float,
    owner_is_publiplots: bool,
) -> bool:
    """Expand one axis to fit the given extents. Return True if limits changed."""
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
