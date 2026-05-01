"""
Shared legend column across multiple subplots.

MultiAxesLegendGroup composes one unified legend column anchored to a
chosen axes even when individual legends/colorbars are attached to other
axes in the same figure. This is the primary tool for complex subplot
layouts.
"""

import warnings
from typing import List, Optional, Sequence

from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar

from publiplots.utils.legend import LegendBuilder
from publiplots.utils.legend_entries import (
    LegendEntry,
    get_entries,
    is_continuous_hue,
)


class MultiAxesLegendGroup:
    """
    Unified legend column across multiple axes.

    All elements share a single mm-based layout anchored to ``anchor``. Each
    element can be attached to a different axes (via ``ax=`` on ``add_*``
    calls) for hit-testing and picking; its POSITION is always computed
    against the anchor's right edge regardless of which axes owns the artist.

    Parameters
    ----------
    anchor : Axes
        The axes whose right edge defines x=0 for the shared column.
    collect : list or tuple of str, optional
        Names of entries to auto-collect from across the grid's stashed
        ``LegendEntry`` objects. ``None`` (default) collects everything.
        A list filters and orders — e.g. ``collect=['treatment', 'dose']``
        renders only those two names, in that order.
    x_offset, y_offset, gap, column_spacing, vpad, max_width
        Same meaning as :class:`LegendBuilder` — all in millimeters.
    """

    def __init__(
        self,
        anchor: Axes,
        collect: Optional[Sequence[str]] = None,
        x_offset: float = 2,
        y_offset: Optional[float] = None,
        gap: float = 2,
        column_spacing: float = 5,
        vpad: float = 5,
        max_width: Optional[float] = None,
    ):
        self.anchor = anchor
        if collect is not None:
            if isinstance(collect, str) or not hasattr(collect, "__iter__"):
                raise TypeError(
                    "collect must be None or a list/tuple of names; "
                    "got a bare string. Wrap in a list: collect=['name']"
                )
            collect = list(collect)
        self._collect = collect
        # Track whether _materialize has already run (set by Task 3).
        self._materialized = False
        self._warned_mismatch = False
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
            external_to_axis=True,
        )
        # Register on the figure so plot functions can check claims.
        anchor.get_figure()._publiplots_legend_group = self

    def claims(self, name: str) -> bool:
        """True if the group will render an entry with this name."""
        if self._collect is None:
            return True
        return name in self._collect

    def _materialize(self) -> None:
        """Collect stashed entries from every grid axes and render them.

        Called by SubplotsAutoLayout during the settle pass (Task 5).
        Idempotent — subsequent calls after the first return immediately.
        """
        if self._materialized:
            return
        self._materialized = True

        fig = self.anchor.get_figure()
        axes_matrix = getattr(fig, "_publiplots_axes", None)
        if axes_matrix is None:
            return

        seen = {}  # (name, kind) -> LegendEntry
        order = []  # list of (name, kind) in collection order
        for row in axes_matrix:
            for ax in row:
                for entry in get_entries(ax):
                    if self._collect is not None and entry.name not in self._collect:
                        continue
                    key = (entry.name, entry.kind)
                    if key in seen:
                        if seen[key].signature != entry.signature and not self._warned_mismatch:
                            warnings.warn(
                                f"legend entry {entry.name!r} ({entry.kind}) "
                                "differs between axes; group uses first occurrence",
                                UserWarning,
                                stacklevel=2,
                            )
                            self._warned_mismatch = True
                        continue
                    seen[key] = entry
                    order.append(key)

        if self._collect is not None:
            # Stable sort by the user's collect order; ties (same name with
            # different kinds) stay in the order they were encountered.
            order.sort(key=lambda k: (self._collect.index(k[0]), 0))

        for key in order:
            self._render_entry(seen[key])

    def _render_entry(self, entry: LegendEntry) -> None:
        """Route to add_legend (categorical) or add_colorbar (continuous)."""
        if entry.kind == "hue" and is_continuous_hue(entry.handles):
            mappable = entry.handles[0]
            self.add_colorbar(mappable=mappable, label=entry.name)
        else:
            self.add_legend(
                handles=list(entry.handles),
                label=entry.name,
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
    collect: Optional[Sequence[str]] = None,
    x_offset: float = 2,
    y_offset: Optional[float] = None,
    gap: float = 2,
    column_spacing: float = 5,
    vpad: float = 5,
    max_width: Optional[float] = None,
) -> MultiAxesLegendGroup:
    """Create a shared legend column anchored to ``anchor``.

    See :class:`MultiAxesLegendGroup` for parameter docs.
    """
    return MultiAxesLegendGroup(
        anchor=anchor,
        collect=collect,
        x_offset=x_offset,
        y_offset=y_offset,
        gap=gap,
        column_spacing=column_spacing,
        vpad=vpad,
        max_width=max_width,
    )