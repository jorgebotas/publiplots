"""
Shared legend band across multiple subplots.

MultiAxesLegendGroup composes one unified legend band anchored to a
chosen side of the figure (or a single axes) even when individual
legends/colorbars are attached to other axes in the same figure. This
is the primary tool for complex subplot layouts.
"""

import warnings
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.transforms import Bbox

from publiplots.utils.legend import LegendBuilder
from publiplots.utils.legend_entries import (
    LegendEntry,
    get_entries,
    is_continuous_hue,
)


class _GridAnchor:
    """Virtual Axes proxy spanning the whole publiplots subplot grid
    **including its decorations** (tick labels, axis labels, titles).

    Duck-types the small slice of ``matplotlib.axes.Axes`` that
    ``LegendBuilder`` and ``LayoutReactor`` use (``get_position()`` +
    ``get_figure()`` + ``get_window_extent()``). Lets figure-anchored
    ``pp.legend_group`` share exactly the same placement machinery as
    axes-anchored groups — the anchor just happens to be the decorated
    grid rectangle (the outer boundary of everything *except* legend
    bands) rather than a single axes' rectangle.

    The decorated-grid bbox is computed from the ``FigureLayout``:

        x0 = (outer_pad + legend_band_left)     / W
        x1 = (W - outer_pad - legend_column)    / W
        y0 = (outer_pad + legend_band_bottom)   / H
        y1 = (H - outer_pad - legend_band_top)  / H

    Anchoring here (rather than the axes-rectangle union) guarantees
    that side='bottom' sits BELOW all xlabel_space, side='top' sits
    ABOVE all title_space, side='left' sits LEFT of all ylabel_space,
    etc. — i.e., the legend doesn't overlap with axis decorations.
    """

    def __init__(self, fig: Figure) -> None:
        self._fig = fig

    def get_figure(self) -> Figure:
        return self._fig

    def get_position(self) -> Bbox:
        """Decorated-grid bbox in figure fractions."""
        layout = getattr(self._fig, "_publiplots_layout", None)
        if layout is not None:
            W, H = layout.figure_size()
            # Sanity guard: avoid division by zero if the layout hasn't
            # measured yet (e.g., during the first draw before sizes are
            # known).
            if W > 0 and H > 0:
                x0 = (layout.outer_pad + layout.legend_band_left) / W
                x1 = (W - layout.outer_pad - layout.legend_column) / W
                y0 = (layout.outer_pad + layout.legend_band_bottom) / H
                y1 = (H - layout.outer_pad - layout.legend_band_top) / H
                return Bbox.from_extents(x0, y0, x1, y1)
        # Fallback when no publiplots layout is installed (figure built
        # by raw matplotlib): union the axes rectangles only.
        matrix = getattr(self._fig, "_publiplots_axes", None)
        if matrix is None:
            return Bbox.from_extents(0.0, 0.0, 1.0, 1.0)
        x0 = y0 = 1.0
        x1 = y1 = 0.0
        for row in matrix:
            for ax in row:
                pos = ax.get_position()
                x0, y0 = min(x0, pos.x0), min(y0, pos.y0)
                x1, y1 = max(x1, pos.x1), max(y1, pos.y1)
        if x0 >= x1 or y0 >= y1:
            return Bbox.from_extents(0.0, 0.0, 1.0, 1.0)
        return Bbox.from_extents(x0, y0, x1, y1)

    def get_window_extent(self, renderer=None):
        """Pixel extent of the decorated-grid bbox."""
        pos = self.get_position()
        fig_w = self._fig.get_window_extent().width
        fig_h = self._fig.get_window_extent().height
        return Bbox.from_extents(
            pos.x0 * fig_w, pos.y0 * fig_h,
            pos.x1 * fig_w, pos.y1 * fig_h,
        )


class MultiAxesLegendGroup:
    """
    Unified legend band across multiple axes.

    All elements share a single mm-based layout anchored to the chosen
    side of ``anchor``. Each element can be attached to a different
    axes (via ``ax=`` on ``add_*`` calls) for hit-testing and picking;
    its POSITION is always computed against the anchor's chosen edge
    regardless of which axes owns the artist.

    Parameters
    ----------
    anchor : Axes, optional
        The axes whose chosen edge defines the origin for the shared
        band. When ``None`` (default), the group is **figure-anchored**:
        it spans the full subplot grid on the chosen side (e.g.,
        ``side='right'`` anchors to the rightmost cells' right edge and
        extends vertically across every row). Pass an explicit axes to
        pin the band to that single cell instead (axes-anchored).
    side : {'right', 'bottom', 'left', 'top'}, default 'right'
        Which edge of the anchor the band grows outward from. ``'right'``
        is the classic publiplots outside-right column.
    figure : Figure, optional
        Figure to attach a figure-anchored group to. Defaults to
        ``plt.gcf()``; only needed when no current figure exists.
    collect : list or tuple of str, optional
        Names of entries to auto-collect from across the grid's stashed
        ``LegendEntry`` objects. ``None`` (default) collects everything.
        A list filters and orders — e.g. ``collect=['treatment', 'dose']``
        renders only those two names, in that order.
    x_offset, y_offset, gap, column_spacing, vpad, max_width
        Same meaning as :class:`LegendBuilder` — all in millimeters.
    """

    # Default outward gap (mm) between the anchor's edge and the
    # legend's near side. For side='right' the gap sits in otherwise
    # empty space, so 2 mm suffices. For the other sides the gap sits
    # adjacent to tick labels / titles and needs more air for the legend
    # not to crowd them.
    _DEFAULT_X_OFFSET_MM = {"right": 2, "left": 5, "bottom": 5, "top": 5}

    # side → default orientation. Horizontal makes sense on top/bottom
    # where there's plenty of figure width; vertical stays the default
    # for right/left where width is narrow.
    _DEFAULT_ORIENTATION = {
        "right": "vertical", "left": "vertical",
        "bottom": "horizontal", "top": "horizontal",
    }

    # side → default along-edge alignment. Horizontal bands center to
    # balance the figure; vertical bands start at the top (current
    # 'upper' behaviour).
    _DEFAULT_ALIGN = {
        "right": "start", "left": "start",
        "bottom": "center", "top": "center",
    }

    def __init__(
        self,
        anchor: Optional[Axes] = None,
        collect: Optional[Sequence[str]] = None,
        *,
        side: str = "right",
        figure: Optional[Figure] = None,
        orientation: str = "auto",
        align: str = "auto",
        x_offset: Optional[float] = None,
        y_offset: Optional[float] = None,
        gap: float = 2,
        column_spacing: float = 5,
        vpad: float = 5,
        max_width: Optional[float] = None,
    ):
        if side not in ("right", "left", "bottom", "top"):
            raise ValueError(
                f"side must be 'right' | 'left' | 'bottom' | 'top', got {side!r}"
            )
        if orientation not in ("auto", "vertical", "horizontal"):
            raise ValueError(
                f"orientation must be 'auto' | 'vertical' | 'horizontal', "
                f"got {orientation!r}"
            )
        if align not in ("auto", "start", "center", "end"):
            raise ValueError(
                f"align must be 'auto' | 'start' | 'center' | 'end', got {align!r}"
            )
        self._side = side
        self._orientation = (
            self._DEFAULT_ORIENTATION[side] if orientation == "auto" else orientation
        )
        self._align = (
            self._DEFAULT_ALIGN[side] if align == "auto" else align
        )
        if x_offset is None:
            x_offset = self._DEFAULT_X_OFFSET_MM[side]

        # Decide anchor kind. When the caller passes an explicit axes we
        # pin the band to that one axes (and its per-cell reservation
        # tuple grows). Otherwise span the whole grid via _GridAnchor.
        if anchor is None:
            fig = figure if figure is not None else plt.gcf()
            self.anchor = _GridAnchor(fig)
            self._anchor_kind = "figure"
        else:
            self.anchor = anchor
            self._anchor_kind = "axes"

        if collect is not None:
            if isinstance(collect, str) or not hasattr(collect, "__iter__"):
                raise TypeError(
                    "collect must be None or a list/tuple of names; "
                    "got a bare string. Wrap in a list: collect=['name']"
                )
            collect = list(collect)
        self._collect = collect
        self._materialized = False
        self._warned_mismatch = False
        self._align_connected = False
        self._aligning = False  # re-entrancy guard for _on_draw_align
        # anchor_ax=self.anchor pins position math/reactor to the anchor
        # regardless of self.ax swaps during add_* calls. For
        # figure-anchored groups anchor_ax is the _GridAnchor proxy,
        # which exposes the same get_position()/get_figure() API.
        builder_ax = self.anchor if self._anchor_kind == "axes" \
                     else self._pick_builder_ax()
        self._builder = LegendBuilder(
            ax=builder_ax,
            anchor_ax=self.anchor,
            x_offset=x_offset,
            y_offset=y_offset,
            gap=gap,
            column_spacing=column_spacing,
            vpad=vpad,
            max_width=max_width,
            external_to_axis=True,
            side=side,
            orientation=self._orientation,
        )
        # Register on the figure so plot functions can check claims.
        self.anchor.get_figure()._publiplots_legend_group = self

    def _pick_builder_ax(self) -> Axes:
        """Pick a real axes from the figure to attach Legend artists to.

        Figure-anchored groups use ``_GridAnchor`` only for position
        math; the actual matplotlib ``Legend`` artist still needs a
        real Axes parent. Default to the corner cell closest to the
        legend's edge so hit-testing lines up with where the user sees
        the band sitting.
        """
        fig = self.anchor.get_figure()
        matrix = getattr(fig, "_publiplots_axes", None)
        if matrix is None:
            return fig.axes[0]
        if self._side == "right":
            return matrix[0][-1]
        if self._side == "left":
            return matrix[0][0]
        if self._side == "bottom":
            return matrix[-1][0]
        return matrix[0][0]  # "top"

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

        # Align pass runs on each draw via the reactor hook so it uses
        # the current (possibly still-settling) figure geometry.
        # Absolute positioning → idempotent as geometry converges.
        if self._align != "start" and not self._align_connected:
            # Wrap the reactor's _refresh_all to run alignment AFTER it
            # repositions every registration. That way align sees the
            # post-refresh anchor position but can still override the
            # along-edge offsets before matplotlib renders.
            reactor = self._builder._reactor
            if not hasattr(reactor, "_align_callbacks"):
                reactor._align_callbacks = []
                _orig_refresh = reactor._refresh_all

                def _refresh_then_align():
                    result = _orig_refresh()
                    for cb in reactor._align_callbacks:
                        cb()
                    return result

                reactor._refresh_all = _refresh_then_align
            reactor._align_callbacks.append(self._on_draw_align_cb)
            self._align_connected = True

    def _on_draw_align_cb(self) -> None:
        """Post-reactor-refresh alignment hook. Runs inside each draw
        after LayoutReactor has repositioned every registration; our
        alignment override fires on the registrations this group owns,
        then those overrides reach the canvas via the reactor's next
        refresh. Guarded against re-entrancy (measuring triggers draws).
        """
        if self._aligning:
            return
        self._aligning = True
        try:
            self._apply_along_alignment()
            # Re-run reactor refresh with the updated mm_y_from_top
            # values so matplotlib renders the new positions in this
            # draw (not the next one).
            self._builder._reactor._refresh_all()
        finally:
            self._aligning = False

    def _apply_along_alignment(self) -> None:
        """Re-position all placed reactor registrations along the anchor edge
        to honor ``self._align``.

        Each row (legends sharing an outward offset) is aligned
        independently so a wrapped block stays visually balanced. We
        recompute each registration's **absolute** along-edge offset
        (``mm_y_from_top`` — historical field name, semantically
        "along-edge mm from the starting corner") from the known legend
        extents, then re-run the reactor so the legends snap to the new
        figure-fraction positions.

        The starting corner of the along-edge axis is ``ax_pos.y1`` for
        side=right/left (top of anchor) and ``ax_pos.x0`` for
        side=top/bottom (left of anchor). "Along-edge mm from start"
        increases as we move AWAY from that corner.
        """
        if self._align == "start":
            return
        if not self._builder.elements:
            return

        reactor = self._builder._reactor
        element_ids = {id(a) for _, a in self._builder.elements}
        regs = [r for r in reactor._registrations if id(r.artist) in element_ids]
        if not regs:
            return

        orient = self._orientation
        edge_length_mm = self._builder._get_edge_length()
        gap_mm = self._builder._layout.gap

        # Group regs into rows sharing the same outward offset.
        rows = {}
        for reg in regs:
            key = round(reg.mm_x_from_right, 3)
            rows.setdefault(key, []).append(reg)

        for row_regs in rows.values():
            # Measure each legend's along-edge extent.
            extents = []
            for reg in row_regs:
                w, h = self._builder._measure_object_dimensions(reg.artist)
                extents.append(w if orient == "horizontal" else h)
            total = sum(extents) + gap_mm * (len(row_regs) - 1)
            if total >= edge_length_mm:
                # Block already fills the edge — no room to align.
                continue
            if self._align == "center":
                start = (edge_length_mm - total) / 2
            elif self._align == "end":
                start = edge_length_mm - total - self._builder._layout.vpad
            else:
                start = self._builder._layout.vpad
            cursor = start
            for reg, extent in zip(row_regs, extents):
                reg.mm_y_from_top = cursor
                cursor += extent + gap_mm

        reactor._refresh_all()

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

    def _default_target_ax(self) -> Axes:
        """Axes to attach a Legend artist to when no explicit ax= is passed.

        For axes-anchored groups, that's the anchor. For figure-anchored
        groups the anchor is a ``_GridAnchor`` proxy (not a real Axes);
        the builder's pre-picked corner cell is used instead.
        """
        if self._anchor_kind == "axes":
            return self.anchor
        return self._builder.ax  # corner cell picked in __init__

    def add_legend(
        self,
        handles: List,
        label: str = "",
        *,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Legend:
        """Add a legend to the shared band.

        The artist is attached to ``ax`` (defaults to a sensible corner
        cell for figure-anchored groups, or the anchor for axes-anchored);
        position is always computed against the anchor's chosen edge.
        """
        target_ax = ax if ax is not None else self._default_target_ax()
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
        """Add a colorbar to the shared band. See add_legend for ax semantics."""
        target_ax = ax if ax is not None else self._default_target_ax()
        original_ax = self._builder.ax
        try:
            self._builder.ax = target_ax
            cbar = self._builder.add_colorbar(mappable=mappable, **kwargs)
        finally:
            self._builder.ax = original_ax
        return cbar


def legend_group(
    anchor: Optional[Axes] = None,
    *,
    side: str = "right",
    figure: Optional[Figure] = None,
    orientation: str = "auto",
    align: str = "auto",
    collect: Optional[Sequence[str]] = None,
    x_offset: Optional[float] = None,
    y_offset: Optional[float] = None,
    gap: float = 2,
    column_spacing: float = 5,
    vpad: float = 5,
    max_width: Optional[float] = None,
) -> MultiAxesLegendGroup:
    """Create a shared legend band.

    Pass ``anchor=None`` (default) for a **figure-anchored** band that
    spans the full subplot grid on the chosen side. Pass
    ``anchor=axes[r, c]`` to pin the band to a single cell; the
    corresponding per-cell reservation (``right[c]`` for ``side='right'``,
    ``xlabel_space[r]`` for ``side='bottom'``, etc.) absorbs the band
    width.

    ``orientation`` defaults to ``'horizontal'`` for ``side='top'|'bottom'``
    (entries run along the edge left-to-right, multiple legends sit
    side-by-side, overflow spills outward in new rows) and
    ``'vertical'`` for ``'right'|'left'`` (current behaviour).
    ``align`` defaults to ``'center'`` for top/bottom and ``'start'``
    for left/right.

    See :class:`MultiAxesLegendGroup` for all parameter docs.
    """
    return MultiAxesLegendGroup(
        anchor=anchor,
        side=side,
        figure=figure,
        orientation=orientation,
        align=align,
        collect=collect,
        x_offset=x_offset,
        y_offset=y_offset,
        gap=gap,
        column_spacing=column_spacing,
        vpad=vpad,
        max_width=max_width,
    )