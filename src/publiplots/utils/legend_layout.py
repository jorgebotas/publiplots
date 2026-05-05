"""
Pure-geometry legend layout tracking for publiplots.

This module provides LegendLayout — an mm-based cursor that tracks
positions and overflow for legend placement. It contains no matplotlib
imports; LegendBuilder is responsible for converting mm positions to
matplotlib coordinates and creating artists.

Terminology (orientation-neutral):

- **outward**: distance from the anchor edge. For ``side='right'`` this
  is rightward; for ``'bottom'`` it's downward.
- **along**: remaining space along the anchor edge. For ``side='right'``
  this is the remaining height below the cursor; for ``'bottom'`` it's
  the remaining width to the right of the cursor.
- **row/column**: abstract names for orthogonal bands of legend entries.
  In vertical orientation (default for right/left), overflow starts a
  new *column* further outward; in horizontal orientation (default for
  top/bottom), overflow starts a new *row* further outward.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LegendLayout:
    """
    Mm-based cursor for legend row/column positioning.

    All dimensions are in millimeters. Pure geometry — no matplotlib imports.
    LegendBuilder uses this to track where each element should go and when
    to overflow into a new band (column for vertical orientation, row for
    horizontal).

    Parameters
    ----------
    x_offset : float, default=2
        Outward distance from anchor edge (mm).
    y_offset : float, optional
        Explicit starting position along the edge (mm). If None,
        ``reset_to()`` uses ``(edge_length - vpad)`` instead.
    gap : float, default=2
        Space between successive elements along the edge (mm).
    column_spacing : float, default=5
        Spacing between parallel bands (columns for vertical, rows for
        horizontal) in mm.
    vpad : float, default=5
        Padding from the starting corner along the edge when
        ``y_offset`` is None (mm).
    max_width : float, optional
        Maximum band width hint (mm). Currently informational only.
    orientation : {'vertical', 'horizontal'}, default='vertical'
        Primary stacking direction. Informational — LegendBuilder reads
        it to decide whether overflow checks use estimated height
        (vertical) or width (horizontal) and whether a new band runs
        outward as a column or row.
    """

    x_offset: float = 2
    y_offset: Optional[float] = None
    gap: float = 2
    column_spacing: float = 5
    vpad: float = 5
    max_width: Optional[float] = None
    orientation: str = "vertical"

    # Mutable cursor state (init=False so they aren't constructor args).
    # Field names use orientation-neutral semantics ("outward" +
    # "along-edge") so the same geometry serves both vertical and
    # horizontal legends.
    current_outward: float = field(init=False, default=0.0)
    current_along: float = field(init=False, default=0.0)
    current_band_width: float = field(init=False, default=0.0)
    bands: List[float] = field(init=False, default_factory=list)
    _band_start_along: float = field(init=False, default=0.0)
    _along_from_start: float = field(init=False, default=0.0)

    def reset_to(self, edge_length_mm: float) -> None:
        """Reset the cursor. Called at builder init and after anchor resize.

        ``edge_length_mm`` is the along-edge extent available (height
        for vertical orientation, width for horizontal).
        """
        self.current_outward = self.x_offset
        start_along = self.y_offset if self.y_offset is not None else (
            edge_length_mm - self.vpad
        )
        self.current_along = start_along
        self._band_start_along = start_along
        self.current_band_width = 0.0
        self.bands = []
        # mm offset from the starting corner along the edge. Starts at
        # vpad for the first element and does NOT depend on
        # edge_length_mm, so it's stable across anchor resizes.
        self._along_from_start = self.vpad

    def check_overflow(self, required_along: float) -> bool:
        """True if an element of this along-edge size would overflow the current band."""
        return self.current_along < required_along

    def start_new_band(self) -> None:
        """Record current band's width, shift cursor outward, reset along-cursor."""
        self.bands.append(self.current_band_width)
        self.current_outward += self.current_band_width + self.column_spacing
        self.current_band_width = 0.0
        self.current_along = self._band_start_along
        self._along_from_start = self.vpad

    def advance_along(self, element_along: float) -> None:
        """Move cursor along the edge by ``element_along + gap``."""
        self.current_along -= element_along + self.gap
        self._along_from_start += element_along + self.gap

    def update_width(self, element_width: float) -> None:
        """Grow current band's outward width to at least this value (never shrinks)."""
        if element_width > self.current_band_width:
            self.current_band_width = element_width

    @property
    def along_from_start(self) -> float:
        """mm offset of the current cursor from the starting corner (stable)."""
        return self._along_from_start
