"""
Pure-geometry layout for publiplots subplot grids.

FigureLayout computes figure size and axes positions from declared mm
dimensions. No matplotlib imports — this module is pure math.

Per-side reservations are TUPLES indexed by position (length nrows for
title_space / xlabel_space; length ncols for ylabel_space / right).
Scalar-to-tuple broadcast happens at the pp.subplots() boundary.
"""

from dataclasses import dataclass, replace
from typing import Tuple


@dataclass
class FigureLayout:
    """
    Millimeter-based grid geometry with per-row / per-column reservations.

    Parameters
    ----------
    nrows, ncols : int
        Grid shape (>= 1).
    axes_size : tuple of (width, height) in mm
        Declared axes bbox for every cell. Inviolate after construction.
    title_space : tuple[float, ...] of length nrows
        Space reserved above each row for titles.
    xlabel_space : tuple[float, ...] of length nrows
        Space reserved below each row for x-axis labels and tick labels.
    ylabel_space : tuple[float, ...] of length ncols
        Space reserved left of each column for y-axis labels and tick labels.
    right : tuple[float, ...] of length ncols
        Space reserved right of each column.
    hspace, wspace : float
        Inter-row and inter-column gaps (scalar; gaps are global).
    outer_pad : float
        Figure outer margin (same on all four sides).
    legend_column : float
        Extra width on the far right, outside the grid. Never auto-measured.
    """

    nrows: int
    ncols: int
    axes_size: Tuple[float, float]
    title_space: Tuple[float, ...]
    xlabel_space: Tuple[float, ...]
    ylabel_space: Tuple[float, ...]
    right: Tuple[float, ...]
    hspace: float
    wspace: float
    outer_pad: float
    legend_column: float

    def __post_init__(self) -> None:
        for side in ("title_space", "xlabel_space", "ylabel_space", "right"):
            val = getattr(self, side)
            if not isinstance(val, tuple):
                raise TypeError(
                    f"{side} must be a tuple of floats; got {type(val).__name__}. "
                    f"pp.subplots() broadcasts scalars — construct FigureLayout with tuples."
                )
        for side in ("title_space", "xlabel_space"):
            val = getattr(self, side)
            if len(val) != self.nrows:
                raise ValueError(
                    f"{side} must have length nrows={self.nrows}, got length {len(val)}"
                )
        for side in ("ylabel_space", "right"):
            val = getattr(self, side)
            if len(val) != self.ncols:
                raise ValueError(
                    f"{side} must have length ncols={self.ncols}, got length {len(val)}"
                )
        for side in ("title_space", "xlabel_space", "ylabel_space", "right"):
            for i, v in enumerate(getattr(self, side)):
                if v < 0:
                    raise ValueError(f"{side}[{i}] must be non-negative, got {v}")

    def figure_size(self) -> Tuple[float, float]:
        """Total figure size in mm as (width, height)."""
        w_ax, h_ax = self.axes_size
        W = (
            self.outer_pad
            + sum(self.ylabel_space)
            + self.ncols * w_ax
            + sum(self.right)
            + max(self.ncols - 1, 0) * self.wspace
            + self.legend_column
            + self.outer_pad
        )
        H = (
            self.outer_pad
            + sum(self.title_space)
            + self.nrows * h_ax
            + sum(self.xlabel_space)
            + max(self.nrows - 1, 0) * self.hspace
            + self.outer_pad
        )
        return W, H

    def axes_position(self, row: int, col: int) -> Tuple[float, float, float, float]:
        """Figure-fraction (x0, y0, w, h) for the cell at (row, col)."""
        W, H = self.figure_size()
        w_ax, h_ax = self.axes_size
        x0_mm = self.outer_pad
        for c in range(col):
            x0_mm += self.ylabel_space[c] + w_ax + self.right[c] + self.wspace
        x0_mm += self.ylabel_space[col]
        y0_mm = self.outer_pad
        for r in range(self.nrows - 1, row, -1):
            y0_mm += self.xlabel_space[r] + h_ax + self.title_space[r] + self.hspace
        y0_mm += self.xlabel_space[row]
        return (x0_mm / W, y0_mm / H, w_ax / W, h_ax / H)

    def with_updated_reservations(self, **overrides) -> "FigureLayout":
        """Return a copy with the given fields updated. ``axes_size`` must not change."""
        if "axes_size" in overrides:
            raise ValueError("axes_size is inviolate; cannot be overridden")
        return replace(self, **overrides)
