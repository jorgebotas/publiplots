"""
Complex heatmap annotations module.

Provides annotation functions for complex heatmaps:
- block: Simple colored blocks for categorical/continuous variables
- label: Text labels with optional arrows
- spacer: Empty space for visual separation
"""

from publiplots.plot.complex_heatmap.annotations import block, label, spacer

__all__ = [
    "block",
    "label",
    "spacer",
]
