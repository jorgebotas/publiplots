"""
Pure-geometry layout for publiplots subplot grids.

FigureLayout computes figure size and axes positions from declared mm
dimensions. No matplotlib imports — this module is pure math and is
testable in isolation.
"""

from dataclasses import dataclass, replace
from typing import Tuple


@dataclass
class FigureLayout:
    """
    Millimeter-based grid geometry for a uniform subplot layout.

    Every cell has the same ``axes_size`` and the same per-side
    reservations. Gaps between cells are ``hspace`` (vertical) and
    ``wspace`` (horizontal). An optional ``legend_column`` reserves
    space on the far right of the whole figure, outside the grid.

    All values are in millimeters.

    Parameters
    ----------
    nrows, ncols : int
        Grid shape (>= 1).
    axes_size : tuple of (width, height) in mm
        The declared axes bbox for every cell. Inviolate after
        construction — never changes.
    title_space : float
        Space reserved above each row for titles.
    xlabel_space : float
        Space reserved below each row for x-axis labels and tick labels.
    ylabel_space : float
        Space reserved left of each column for y-axis labels and tick labels.
    right : float
        Space reserved right of each column (breathing room past the spine).
    hspace : float
        Vertical gap between rows.
    wspace : float
        Horizontal gap between columns.
    outer_pad : float
        Figure outer margin (same on all four sides).
    legend_column : float
        Extra width reserved on the far right of the figure for a
        legend_group anchored outside the grid. Never auto-measured.
    """

    nrows: int
    ncols: int
    axes_size: Tuple[float, float]
    title_space: float
    xlabel_space: float
    ylabel_space: float
    right: float
    hspace: float
    wspace: float
    outer_pad: float
    legend_column: float

    def figure_size(self) -> Tuple[float, float]:
        """Total figure size in mm as (width, height)."""
        w_ax, h_ax = self.axes_size
        W = (
            self.outer_pad
            + self.ncols * (self.ylabel_space + w_ax + self.right)
            + max(self.ncols - 1, 0) * self.wspace
            + self.legend_column
            + self.outer_pad
        )
        H = (
            self.outer_pad
            + self.nrows * (self.title_space + h_ax + self.xlabel_space)
            + max(self.nrows - 1, 0) * self.hspace
            + self.outer_pad
        )
        return W, H

    def axes_position(self, row: int, col: int) -> Tuple[float, float, float, float]:
        """
        Figure-fraction (x0, y0, w, h) of the axes at (row, col).

        Row 0 is the top row. y is measured from the bottom, matching
        matplotlib's convention.
        """
        W, H = self.figure_size()
        w_ax, h_ax = self.axes_size
        x0_mm = (
            self.outer_pad
            + col * (self.ylabel_space + w_ax + self.right + self.wspace)
            + self.ylabel_space
        )
        rows_below = self.nrows - 1 - row
        y0_mm = (
            self.outer_pad
            + rows_below * (self.title_space + h_ax + self.xlabel_space + self.hspace)
            + self.xlabel_space
        )
        return (x0_mm / W, y0_mm / H, w_ax / W, h_ax / H)

    def with_updated_reservations(self, **overrides) -> "FigureLayout":
        """Return a copy with the given fields updated. ``axes_size`` must not change."""
        if "axes_size" in overrides:
            raise ValueError("axes_size is inviolate; cannot be overridden")
        return replace(self, **overrides)