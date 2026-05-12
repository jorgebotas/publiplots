"""Custom-text label strategy for violinplots.

Shares BoxAnchorResolver + BoxStatsRecord with box_custom (violins and
boxes both produce BoxStatsMeta). The distinction exists so users can
request kind="violin_custom" for discoverability; future KDE-extent-
aware anchoring can diverge this strategy without a breaking rename.
"""
from __future__ import annotations

from publiplots.annotate._cache import BoxStatsRecord
from publiplots.annotate._custom import _CustomStrategy
from publiplots.annotate._shared import compute_axes_to_expand_directional
from publiplots.annotate.box_stats import BoxAnchorResolver


_violin_custom_strategy = _CustomStrategy(
    resolver=BoxAnchorResolver(),
    record_type=BoxStatsRecord,
    meta_attr="_publiplots_box_meta",
    records_attr="boxes",
    introspect=None,
    has_fit_check=False,
    value_source=lambda r: r.stats.get("median", float("nan")),
    axes_to_expand=(
        lambda anchor, orient, rotation:
            compute_axes_to_expand_directional(anchor, orient, rotation)
    ),
)
