"""
Pure-geometry legend layout tracking for publiplots.

This module provides LegendLayout — an mm-based cursor that tracks column
positions and overflow for legend placement. It contains no matplotlib
imports; LegendBuilder is responsible for converting mm positions to
matplotlib coordinates and creating artists.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LegendLayout:
    """
    Mm-based cursor for legend column/row positioning.

    All dimensions are in millimeters. Pure geometry — no matplotlib imports.
    LegendBuilder uses this to track where each element should go and when
    to overflow into a new column.

    Parameters
    ----------
    x_offset : float, default=2
        Horizontal distance from anchor edge (mm).
    y_offset : float, optional
        Explicit starting y position (mm). If None, reset_to() uses
        (axes_height - vpad) instead.
    gap : float, default=2
        Vertical space between elements (mm).
    column_spacing : float, default=5
        Horizontal space between columns (mm).
    vpad : float, default=5
        Padding from top of axes when y_offset is None (mm).
    max_width : float, optional
        Maximum column width hint (mm). Currently informational only.
    """

    x_offset: float = 2
    y_offset: Optional[float] = None
    gap: float = 2
    column_spacing: float = 5
    vpad: float = 5
    max_width: Optional[float] = None

    # Mutable cursor state (init=False so they aren't constructor args)
    current_x: float = field(init=False, default=0.0)
    current_y: float = field(init=False, default=0.0)
    current_column_width: float = field(init=False, default=0.0)
    columns: List[float] = field(init=False, default_factory=list)
    _column_top_y: float = field(init=False, default=0.0)

    def reset_to(self, axes_height_mm: float) -> None:
        """Reset the cursor. Called at builder init and after axes resize."""
        self.current_x = self.x_offset
        top_y = self.y_offset if self.y_offset is not None else (
            axes_height_mm - self.vpad
        )
        self.current_y = top_y
        self._column_top_y = top_y
        self.current_column_width = 0.0
        self.columns = []

    def check_overflow(self, required_height: float) -> bool:
        """True if an element of this height would overflow the current column."""
        return self.current_y < required_height

    def start_new_column(self) -> None:
        """Record current column's width, shift cursor right, reset y."""
        self.columns.append(self.current_column_width)
        self.current_x += self.current_column_width + self.column_spacing
        self.current_column_width = 0.0
        self.current_y = self._column_top_y

    def advance_y(self, element_height: float) -> None:
        """Move cursor down by element_height + gap."""
        self.current_y -= element_height + self.gap

    def update_width(self, element_width: float) -> None:
        """Grow current column width to at least this value (never shrinks)."""
        if element_width > self.current_column_width:
            self.current_column_width = element_width