"""
Pure-geometry layout for publiplots subplot grids.

FigureLayout computes figure size and axes positions from declared mm
dimensions. No matplotlib imports — this module is pure math.

Per-side reservations are TUPLES indexed by position (length nrows for
title_space / xlabel_space; length ncols for ylabel_space / right).
Per-cell dimensions are also TUPLES (length nrows for ``row_heights``,
length ncols for ``col_widths``) so rows and columns can have
heterogeneous sizes — a JointGrid-shaped figure, for instance, has a
large main cell and two thin marginal cells. Scalar-to-tuple broadcast
happens at the pp.subplots() boundary.
"""

from dataclasses import dataclass, field, replace
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
        Uniform axes bbox fallback. When ``col_widths`` / ``row_heights``
        are empty tuples, they are broadcast from this field. Kept for
        backward compatibility with code that constructs ``FigureLayout``
        with only ``axes_size``.
    col_widths : tuple[float, ...] of length ncols
        Per-column axes widths in mm. Broadcast from ``axes_size[0]``
        when ``()``.
    row_heights : tuple[float, ...] of length nrows
        Per-row axes heights in mm. Broadcast from ``axes_size[1]``
        when ``()``.
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
        Extra width on the far right, outside the grid. Used by
        figure-anchored ``pp.legend_group(side='right')``.
    legend_band_bottom, legend_band_top, legend_band_left : float
        Extra mm on the bottom / top / left of the figure, outside the
        grid. Used by figure-anchored ``pp.legend_group(side=...)``.
        Default ``0.0``. All four scalars only kick in when a
        figure-anchored legend group asks for space on that side.
    suptitle_space : float
        Extra mm above ``legend_band_top`` reserved for a figure-level
        ``pp.suptitle``. Default ``0.0``; auto-measured by
        :class:`SubplotsAutoLayout` when a suptitle is present.
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
    legend_band_bottom: float = 0.0
    legend_band_top: float = 0.0
    legend_band_left: float = 0.0
    suptitle_space: float = 0.0
    col_widths: Tuple[float, ...] = field(default_factory=tuple)
    row_heights: Tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Broadcast from axes_size when caller didn't supply per-cell tuples.
        if not self.col_widths:
            self.col_widths = (float(self.axes_size[0]),) * self.ncols
        if not self.row_heights:
            self.row_heights = (float(self.axes_size[1]),) * self.nrows

        for side in ("title_space", "xlabel_space", "ylabel_space", "right",
                     "col_widths", "row_heights"):
            val = getattr(self, side)
            if not isinstance(val, tuple):
                raise TypeError(
                    f"{side} must be a tuple of floats; got {type(val).__name__}. "
                    f"pp.subplots() broadcasts scalars — construct FigureLayout with tuples."
                )
        for side in ("title_space", "xlabel_space", "row_heights"):
            val = getattr(self, side)
            if len(val) != self.nrows:
                raise ValueError(
                    f"{side} must have length nrows={self.nrows}, got length {len(val)}"
                )
        for side in ("ylabel_space", "right", "col_widths"):
            val = getattr(self, side)
            if len(val) != self.ncols:
                raise ValueError(
                    f"{side} must have length ncols={self.ncols}, got length {len(val)}"
                )
        for side in ("title_space", "xlabel_space", "ylabel_space", "right"):
            for i, v in enumerate(getattr(self, side)):
                if v < 0:
                    raise ValueError(f"{side}[{i}] must be non-negative, got {v}")
        for side in ("col_widths", "row_heights"):
            for i, v in enumerate(getattr(self, side)):
                if v <= 0:
                    raise ValueError(f"{side}[{i}] must be positive, got {v}")

    def figure_size(self) -> Tuple[float, float]:
        """Total figure size in mm as (width, height)."""
        W = (
            self.outer_pad
            + self.legend_band_left
            + sum(self.ylabel_space)
            + sum(self.col_widths)
            + sum(self.right)
            + max(self.ncols - 1, 0) * self.wspace
            + self.legend_column
            + self.outer_pad
        )
        H = (
            self.outer_pad
            + self.suptitle_space
            + self.legend_band_top
            + sum(self.title_space)
            + sum(self.row_heights)
            + sum(self.xlabel_space)
            + max(self.nrows - 1, 0) * self.hspace
            + self.legend_band_bottom
            + self.outer_pad
        )
        return W, H

    def axes_position(self, row: int, col: int) -> Tuple[float, float, float, float]:
        """Figure-fraction (x0, y0, w, h) for the cell at (row, col)."""
        W, H = self.figure_size()
        x0_mm = self.outer_pad + self.legend_band_left
        for c in range(col):
            x0_mm += self.ylabel_space[c] + self.col_widths[c] + self.right[c] + self.wspace
        x0_mm += self.ylabel_space[col]
        y0_mm = self.outer_pad + self.legend_band_bottom
        for r in range(self.nrows - 1, row, -1):
            y0_mm += self.xlabel_space[r] + self.row_heights[r] + self.title_space[r] + self.hspace
        y0_mm += self.xlabel_space[row]
        return (x0_mm / W, y0_mm / H, self.col_widths[col] / W, self.row_heights[row] / H)

    def with_updated_reservations(self, **overrides) -> "FigureLayout":
        """Return a copy with the given fields updated.

        ``axes_size``, ``col_widths``, and ``row_heights`` are inviolate
        cell dimensions and cannot be overridden via this method.
        """
        for protected in ("axes_size", "col_widths", "row_heights"):
            if protected in overrides:
                raise ValueError(
                    f"{protected} is inviolate; cannot be overridden"
                )
        return replace(self, **overrides)
