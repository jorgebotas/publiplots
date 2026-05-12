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
    resolve_anchor,
)
from publiplots.annotate._shared import (
    ensure_renderer as _ensure_renderer,
    format_value as _format_value,
    maybe_expand_limits,
)


logger = logging.getLogger(__name__)


_VALID_ANCHORS = {"outside", "inside", "base", "center"}
_DEFAULT_ANCHOR = "outside"


def _get_or_introspect(ax: Axes) -> BarValueMeta:
    meta = getattr(ax, "_publiplots_bar_meta", None)
    if isinstance(meta, BarValueMeta):
        return meta
    return _introspect(ax)


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


def _maybe_expand_limits(
    ax, texts, orient, pad_mm, owner_is_publiplots, rotation: float = 0.0,
):
    """Back-compat wrapper: bar_custom imports this by name. New strategies
    should call `maybe_expand_limits` directly."""
    _ = rotation  # bars always expand both axes; rotation is informational
    maybe_expand_limits(
        ax, texts,
        axes_to_expand=["y", "x"] if orient == "v" else ["x", "y"],
        pad_mm=pad_mm, owner_is_publiplots=owner_is_publiplots,
    )
