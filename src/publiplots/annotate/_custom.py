"""Generic 'labels = column-or-callable' strategy over any mark type.

Instantiated once per mark in bar_custom.py / point_custom.py / etc.
Parameterized by:

- `resolver`: an AnchorResolver for this mark type.
- `record_type`: the record class, used for isinstance checks in callable
  labels and for the NaN / anchor-override attribute probes.
- `meta_attr`: attribute name on `ax` where pp.barplot / pp.pointplot /
  etc. stashed the *Meta (e.g., "_publiplots_bar_meta").
- `records_attr`: attribute on the meta holding the list of records
  ("bars" / "points" / "boxes").
- `introspect`: foreign-axes fallback; may be None for strategies that
  don't support foreign axes (point, box, violin).
- `has_fit_check`: whether this mark supports inside->outside fallback
  when a label doesn't fit (bar only; points/boxes have no "interior").
- `value_source`: callable (record) -> float returning the record's value
  for the NaN skip. Most record types use `record.value`; box-stats uses
  `record.stats["median"]`.
- `axes_to_expand`: callable (anchor, orient, rotation) -> list of "x"/"y"
  telling the limit-expansion logic which axes to grow to fit labels.
"""
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

from matplotlib.axes import Axes
from matplotlib.text import Text

from publiplots.annotate._color import resolve_color
from publiplots.annotate._positioning import (
    AnchorResolver,
    fit_check,
    make_offset_transform,
)
from publiplots.annotate._shared import (
    build_column_label_table,
    ensure_renderer,
    format_column_value,
    maybe_expand_limits,
)


logger = logging.getLogger(__name__)


def _default_value_source(r: Any) -> float:
    return r.value


def _default_axes_to_expand(anchor: str, orient: str, rotation: float) -> List[str]:
    return ["y", "x"] if orient == "v" else ["x", "y"]


@dataclass(frozen=True)
class _CustomStrategy:
    """Strategy-kind-agnostic custom-label implementation."""
    resolver: AnchorResolver
    record_type: Type
    meta_attr: str
    records_attr: str
    introspect: Optional[Callable[[Axes], Any]] = None
    has_fit_check: bool = False
    value_source: Callable[[Any], float] = field(default=_default_value_source)
    # Which axes to expand. For bars always both; for point/box it's
    # computed from anchor + rotation.
    axes_to_expand: Callable[[str, str, float], List[str]] = field(
        default=_default_axes_to_expand
    )

    def __call__(
        self,
        ax: Axes,
        *,
        fmt: str,
        anchor,
        offset: float,
        color,
        pad: float,
        rotation: float = 0.0,
        labels: Union[str, Callable[[Any], str], None] = None,
        data=None,
        **text_kws,
    ) -> List[Text]:
        kind = self._kind_name()

        # --- validate labels ---
        if labels is None:
            raise ValueError(f"{kind} requires a labels= argument")
        if not isinstance(labels, str) and not callable(labels):
            raise TypeError(
                f"labels must be a column name (str) or a callable; "
                f"got {type(labels).__name__}"
            )

        # --- validate anchor ---
        valid = self.resolver.VALID_ANCHORS
        default = self.resolver.DEFAULT_ANCHOR
        if anchor is None:
            anchor = default
        if anchor not in valid:
            raise ValueError(
                f"{kind} anchor must be one of {sorted(valid)}; got {anchor!r}"
            )

        text_kws.pop("rotation", None)

        # --- fetch meta ---
        meta = getattr(ax, self.meta_attr, None)
        if meta is None:
            if self.introspect is not None:
                meta = self.introspect(ax)
            else:
                warnings.warn(
                    f"pp.annotate: kind='{kind}' needs publiplots-owned axes; "
                    f"call the matching pp.*plot(..., annotate=True) first "
                    "instead of annotating a foreign axes.",
                    UserWarning,
                    stacklevel=3,
                )
                return []
        records = getattr(meta, self.records_attr)
        if not records:
            warnings.warn(
                f"pp.annotate: no {self.records_attr} found on axes",
                UserWarning,
                stacklevel=3,
            )
            return []

        # --- label source resolution ---
        is_callable = callable(labels)
        if is_callable and fmt != "{}":
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
                    f"pp.annotate: column-based labels require either a "
                    f"publiplots-owned axes (via pp.{self._plotter_name()}) "
                    f"or an explicit data= DataFrame"
                )
            if meta.group_keys is None:
                raise NotImplementedError(
                    "pp.annotate: column-based labels on foreign axes are not "
                    "supported; use a callable (fn(record) -> str) instead"
                )
            label_table = build_column_label_table(
                records, meta.group_keys,
                getattr(meta, "group_dims", None),
                labels, frame,
            )

        # --- per-record loop ---
        renderer = ensure_renderer(ax)
        texts: List[Text] = []
        for rec in records:
            val = self.value_source(rec)
            if val is not None and isinstance(val, float) and math.isnan(val):
                continue

            rec_anchor = getattr(rec, "anchor_override", None) or anchor

            if is_callable:
                label_obj = labels(rec)
                if label_obj is None:
                    continue
                if not isinstance(label_obj, str):
                    raise TypeError(
                        f"labels callable must return str; got "
                        f"{type(label_obj).__name__} at "
                        f"draw_index={rec.draw_index} "
                        f"(category={getattr(rec, 'category', None)!r})"
                    )
                label = label_obj
            else:
                cell = label_table[rec.draw_index]
                formatted = format_column_value(cell, fmt)
                if formatted is None:
                    continue
                label = formatted

            x, y, dx_mm, dy_mm, ha, va = self.resolver.resolve(
                rec, rec_anchor, meta.orient, offset, ax,
            )
            rgba = resolve_color(
                _ColorShim.wrap(rec), color, rec_anchor, ax,
                hue_active=meta.hue_active,
            )
            t = ax.text(
                x, y, label, ha=ha, va=va, color=rgba,
                rotation=rotation,
                transform=make_offset_transform(ax, dx_mm, dy_mm),
                **text_kws,
            )

            if self.has_fit_check and rec_anchor != "outside":
                bbox = rec.patch.get_window_extent(renderer)
                if fit_check(t, bbox, meta.orient, rec_anchor, renderer) == "reanchor_outside":
                    x2, y2, dx2, dy2, ha2, va2 = self.resolver.resolve(
                        rec, "outside", meta.orient, offset, ax,
                    )
                    rgba2 = resolve_color(
                        _ColorShim.wrap(rec), color, "outside", ax,
                        hue_active=meta.hue_active,
                    )
                    t.set_position((x2, y2))
                    t.set_transform(make_offset_transform(ax, dx2, dy2))
                    t.set_ha(ha2)
                    t.set_va(va2)
                    t.set_color(rgba2)
                    logger.debug(
                        "pp.annotate: %s draw_index=%d re-anchored to 'outside'",
                        kind, rec.draw_index,
                    )
            texts.append(t)

        maybe_expand_limits(
            ax, texts,
            axes_to_expand=self.axes_to_expand(anchor, meta.orient, rotation),
            pad_mm=pad, owner_is_publiplots=meta.owner_is_publiplots,
        )
        return texts

    def _kind_name(self) -> str:
        """e.g. '_publiplots_bar_meta' -> 'bar_custom'."""
        core = self.meta_attr.removeprefix("_publiplots_").removesuffix("_meta")
        return f"{core}_custom"

    def _plotter_name(self) -> str:
        """e.g. '_publiplots_bar_meta' -> 'barplot'."""
        core = self.meta_attr.removeprefix("_publiplots_").removesuffix("_meta")
        return f"{core}plot"


class _ColorShim:
    """Wrap any record so resolve_color can read .hue_color and .patch.

    Bars have .patch (Rectangle) directly. Points / boxes have neither -
    but resolve_color only reads hue_color + facecolor/edgecolor on the
    "outside" paths, and when `color='hue'` the record's hue_color is
    used directly. For our purposes a simple facade that forwards
    hue_color and fabricates a dummy patch is sufficient.
    """
    def __init__(self, record):
        self._record = record
        self.hue_color = getattr(record, "hue_color", None)
        self.patch = getattr(record, "patch", None) or _DummyPatch(self.hue_color)

    @classmethod
    def wrap(cls, record):
        # Bars have .patch - pass the record through unchanged so resolve_color
        # gets the real patch (needed for the "auto" luminance-contrast path).
        if getattr(record, "patch", None) is not None:
            return record
        return cls(record)


class _DummyPatch:
    def __init__(self, rgba):
        self._rgba = rgba or (0, 0, 0, 1)

    def get_facecolor(self):
        return self._rgba

    def get_edgecolor(self):
        return self._rgba
