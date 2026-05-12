"""Custom-text label strategy for pointplots.

Parallels bar_custom; uses PointAnchorResolver. No foreign-axes
introspection today — callers must plot via pp.pointplot to use
kind="point_custom".
"""
from __future__ import annotations

from publiplots.annotate._cache import PointRecord
from publiplots.annotate._custom import _CustomStrategy
from publiplots.annotate._shared import compute_axes_to_expand_directional
from publiplots.annotate.point_values import PointAnchorResolver


_point_custom_strategy = _CustomStrategy(
    resolver=PointAnchorResolver(),
    record_type=PointRecord,
    meta_attr="_publiplots_point_meta",
    records_attr="points",
    introspect=None,
    has_fit_check=False,
    axes_to_expand=(
        lambda anchor, orient, rotation:
            compute_axes_to_expand_directional(anchor, orient, rotation)
    ),
)
