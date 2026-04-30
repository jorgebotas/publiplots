"""
Shared legend column across multiple subplots.

MultiAxesLegendGroup composes one unified legend column anchored to a
chosen axes even when individual legends/colorbars are attached to other
axes in the same figure. This is the primary tool for complex subplot
layouts.
"""

from typing import List, Optional

from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar

from publiplots.utils.legend import LegendBuilder


class MultiAxesLegendGroup:
    """
    Unified legend column across multiple axes.

    All elements share a single mm-based layout anchored to `anchor`. Each
    element can be attached to a different axes (via `ax=` on add_* calls)
    for hit-testing and picking; its POSITION is always computed against
    the anchor's right edge regardless of which axes owns the artist.

    Parameters
    ----------
    anchor : Axes
        The axes whose right edge defines x=0 for the shared column.
    x_offset, y_offset, gap, column_spacing, vpad, max_width
        Same meaning as LegendBuilder — all in millimeters.

    Examples
    --------
    >>> fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    >>> group = pp.legend_group(anchor=axes[0])
    >>> group.add_legend(handles_a, label="Treatment", ax=axes[0])
    >>> group.add_legend(handles_b, label="Dose",      ax=axes[1])
    >>> group.add_colorbar(mappable, ax=axes[2])
    """

    def __init__(
        self,
        anchor: Axes,
        x_offset: float = 2,
        y_offset: Optional[float] = None,
        gap: float = 2,
        column_spacing: float = 5,
        vpad: float = 5,
        max_width: Optional[float] = None,
    ):
        self.anchor = anchor
        # anchor_ax=anchor pins position math/reactor to the anchor regardless
        # of self.ax swaps during add_* calls.
        self._builder = LegendBuilder(
            ax=anchor,
            anchor_ax=anchor,
            x_offset=x_offset,
            y_offset=y_offset,
            gap=gap,
            column_spacing=column_spacing,
            vpad=vpad,
            max_width=max_width,
        )

    def add_legend(
        self,
        handles: List,
        label: str = "",
        *,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Legend:
        """Add a legend to the shared column.

        The artist is attached to ax (defaults to anchor); position is
        always computed against the anchor's right edge.
        """
        target_ax = ax if ax is not None else self.anchor
        original_ax = self._builder.ax
        try:
            self._builder.ax = target_ax
            legend = self._builder.add_legend(handles=handles, label=label, **kwargs)
        finally:
            self._builder.ax = original_ax
        return legend

    def add_colorbar(
        self,
        mappable: Optional[ScalarMappable] = None,
        *,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Colorbar:
        """Add a colorbar to the shared column. See add_legend for ax semantics."""
        target_ax = ax if ax is not None else self.anchor
        original_ax = self._builder.ax
        try:
            self._builder.ax = target_ax
            cbar = self._builder.add_colorbar(mappable=mappable, **kwargs)
        finally:
            self._builder.ax = original_ax
        return cbar


def legend_group(
    anchor: Axes,
    *,
    x_offset: float = 2,
    y_offset: Optional[float] = None,
    gap: float = 2,
    column_spacing: float = 5,
    vpad: float = 5,
    max_width: Optional[float] = None,
) -> MultiAxesLegendGroup:
    """Create a shared legend column anchored to `anchor`. See MultiAxesLegendGroup."""
    return MultiAxesLegendGroup(
        anchor=anchor,
        x_offset=x_offset,
        y_offset=y_offset,
        gap=gap,
        column_spacing=column_spacing,
        vpad=vpad,
        max_width=max_width,
    )