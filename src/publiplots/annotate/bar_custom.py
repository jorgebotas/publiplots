"""Custom-text label strategy for barplots.

Same placement machinery as bar_values (resolve_anchor, resolve_color,
fit_check, _maybe_expand_limits); the only difference is where the label
string comes from: a DataFrame column aligned by (x, hue, hatch) group
keys, or a user-supplied callable that receives each BarRecord.
"""
from __future__ import annotations

import logging
import math
import warnings
from typing import Callable, Dict, List, Optional, Union

from matplotlib.axes import Axes
from matplotlib.text import Text

from publiplots.annotate._cache import BarRecord, BarValueMeta, _introspect
from publiplots.annotate._color import resolve_color
from publiplots.annotate._positioning import (
    fit_check,
    make_offset_transform,
    resolve_anchor,
)
from publiplots.annotate.bar_values import (
    _ensure_renderer,
    _maybe_expand_limits,
    _VALID_ANCHORS,
    _DEFAULT_ANCHOR,
)


_DIM_TO_FIELD = {"cat": "category", "hue": "hue_value", "hatch": "hatch_value"}


logger = logging.getLogger(__name__)


def _get_or_introspect(ax: Axes) -> BarValueMeta:
    meta = getattr(ax, "_publiplots_bar_meta", None)
    if isinstance(meta, BarValueMeta):
        return meta
    return _introspect(ax)


def _build_column_label_table(
    meta: BarValueMeta,
    column: str,
    data,                              # pandas DataFrame
) -> Dict[int, object]:
    """Return {draw_index: cell_value} for each bar in meta.bars.

    Aligns by meta.group_keys. Emits a UserWarning when the column varies
    within any group (uses `.first()`).
    """
    if column not in data.columns:
        raise KeyError(
            f"column {column!r} not found in data; available: "
            f"{list(data.columns)}"
        )
    groupby = data.groupby(list(meta.group_keys), observed=True)
    firsts = groupby[column].first()
    nunique = groupby[column].nunique(dropna=False)
    varies = nunique[nunique > 1]
    if len(varies):
        warnings.warn(
            f"pp.annotate: column {column!r} varies within group(s) "
            f"{list(varies.index)}; using .first()",
            UserWarning,
            stacklevel=4,
        )

    # meta.group_dims identifies each group_keys entry's semantic dimension:
    # position 0 is always "cat", but position 1 may be "hue" OR "hatch"
    # (under hatch-only splits). Reading that companion tuple — instead of
    # assuming positional (cat, hue, hatch) order — keeps key lookups aligned
    # regardless of which split dimensions are active. Foreign axes never
    # reach this function (group_keys is None there), but we keep a defensive
    # fallback to all-"cat" so a hypothetical caller doesn't silently KeyError.
    dims = meta.group_dims or ("cat",) * len(meta.group_keys)
    table: Dict[int, object] = {}
    for bar in meta.bars:
        key_parts = [getattr(bar, _DIM_TO_FIELD[d]) for d in dims]
        key = key_parts[0] if len(key_parts) == 1 else tuple(key_parts)
        try:
            table[bar.draw_index] = firsts.loc[key]
        except KeyError:
            table[bar.draw_index] = None
    return table


def _format_column_value(value, fmt: str) -> Optional[str]:
    """Apply fmt to a column cell. Return None for NaN/None so caller skips."""
    if value is None:
        return None
    # pandas NA / float NaN
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except TypeError:
        pass
    if "{" in fmt and "}" in fmt:
        return fmt.format(value)
    return format(value, fmt) if fmt != "{}" else str(value)


def _bar_custom_strategy(
    ax: Axes,
    *,
    fmt: str,
    anchor,
    offset: float,
    color,
    pad: float,
    rotation: float = 0.0,
    labels: Union[str, Callable[[BarRecord], str], None] = None,
    data=None,
    **text_kws,
) -> List[Text]:
    if labels is None:
        raise ValueError("bar_custom requires a labels= argument")
    if not isinstance(labels, str) and not callable(labels):
        raise TypeError(
            f"labels must be a column name (str) or a callable; "
            f"got {type(labels).__name__}"
        )

    if anchor is None:
        anchor = _DEFAULT_ANCHOR
    if anchor not in _VALID_ANCHORS:
        raise ValueError(
            f"bar_custom anchor must be one of {sorted(_VALID_ANCHORS)}; "
            f"got {anchor!r}"
        )

    # Top-level `rotation` wins over one smuggled via text_kws; pop silently.
    text_kws.pop("rotation", None)

    meta = _get_or_introspect(ax)
    if not meta.bars:
        warnings.warn("pp.annotate: no bars found on axes", UserWarning,
                      stacklevel=3)
        return []

    # -------- resolve label source -------------------------------------------
    is_callable = callable(labels)
    if is_callable and fmt != "{}":
        # Per-strategy default fmt is "{}" (neutral passthrough) when the
        # caller did not override; warn if it did.
        warnings.warn(
            "pp.annotate: fmt is ignored when labels is a callable",
            UserWarning,
            stacklevel=3,
        )

    label_table: Optional[Dict[int, object]] = None
    if isinstance(labels, str):
        frame = data if data is not None else meta.source_frame
        if frame is None:
            raise ValueError(
                "pp.annotate: column-based labels require either a "
                "publiplots-owned axes (via pp.barplot) or an explicit "
                "data= DataFrame"
            )
        if meta.group_keys is None:
            raise NotImplementedError(
                "pp.annotate: column-based labels on foreign axes are not "
                "supported; use a callable (fn(record) -> str) instead"
            )
        label_table = _build_column_label_table(meta, labels, frame)

    # -------- per-record loop (mirrors bar_values._bar_values_strategy) ------
    renderer = _ensure_renderer(ax)
    texts: List[Text] = []

    for bar in meta.bars:
        if math.isnan(bar.value):
            continue

        # Stacked/gain bars may pin themselves to a specific anchor.
        bar_anchor = bar.anchor_override if bar.anchor_override is not None else anchor

        if is_callable:
            label_obj = labels(bar)
            if label_obj is None:
                continue
            if not isinstance(label_obj, str):
                raise TypeError(
                    f"labels callable must return str; got "
                    f"{type(label_obj).__name__} at draw_index={bar.draw_index} "
                    f"(category={bar.category!r})"
                )
            label = label_obj
        else:
            cell = label_table[bar.draw_index]
            formatted = _format_column_value(cell, fmt)
            if formatted is None:
                continue
            label = formatted

        x, y, dx_mm, dy_mm, ha, va = resolve_anchor(
            bar, bar_anchor, meta.orient, offset, ax,
        )
        rgba = resolve_color(bar, color, bar_anchor, ax, hue_active=meta.hue_active)
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
                rgba2 = resolve_color(bar, color, "outside", ax,
                                       hue_active=meta.hue_active)
                t.set_position((x2, y2))
                t.set_transform(make_offset_transform(ax, dx2, dy2))
                t.set_ha(ha2)
                t.set_va(va2)
                t.set_color(rgba2)
                logger.debug(
                    "pp.annotate: bar_custom draw_index=%d re-anchored to 'outside'",
                    bar.draw_index,
                )
        texts.append(t)

    _maybe_expand_limits(
        ax, texts, meta.orient, pad_mm=pad,
        owner_is_publiplots=meta.owner_is_publiplots,
        rotation=rotation,
    )
    return texts
