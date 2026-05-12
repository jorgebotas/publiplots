"""Custom-text label strategy for barplots.

Bars can be labeled with either a DataFrame column (aligned by (x, hue,
hatch) group keys) or a callable returning a string per BarRecord.
Wires BarAnchorResolver into the generic _CustomStrategy machinery.
"""
from __future__ import annotations

from publiplots.annotate._cache import BarRecord, _introspect as _bar_introspect
from publiplots.annotate._custom import _CustomStrategy
from publiplots.annotate._positioning import BarAnchorResolver


_bar_custom_strategy = _CustomStrategy(
    resolver=BarAnchorResolver(),
    record_type=BarRecord,
    meta_attr="_publiplots_bar_meta",
    records_attr="bars",
    introspect=_bar_introspect,
    has_fit_check=True,
    # BarRecord.value is the canonical per-bar value; default value_source works.
    # Bars always expand both axes regardless of anchor/rotation (default).
)
