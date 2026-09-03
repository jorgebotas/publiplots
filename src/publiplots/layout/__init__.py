"""publiplots layout engine — fixed-axes, flexible-canvas helpers."""

from publiplots.layout.figure_layout import FigureLayout
from publiplots.layout.auto_layout import (
    LayoutConvergenceWarning,
    SubplotsAutoLayout,
)
from publiplots.layout.subplots import subplots
from publiplots.layout.label_outer import label_outer

__all__ = [
    "FigureLayout",
    "LayoutConvergenceWarning",
    "SubplotsAutoLayout",
    "subplots",
    "label_outer",
]
