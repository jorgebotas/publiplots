"""
Draw-event hook that keeps declared axes sizes fixed while the figure
grows to fit auto-measured decorations.

Reservations are per-row (title_space, xlabel_space) or per-column
(ylabel_space, right). Measurement excludes LayoutReactor-managed
artists flagged external_to_axis (e.g., pp.legend_group) — those are
handled by the reactor's own anchoring geometry plus the user's
legend_column reservation, not by axis-level tightbbox.

Cooperates with LayoutReactor (utils/layout_reactor.py): both react to
draw_event, but SubplotsAutoLayout is registered first (during
pp.subplots()) and therefore fires first, so LayoutReactor sees the
repositioned axes and re-anchors legends correctly.
"""

from typing import Dict, Set, Tuple

from publiplots.layout.figure_layout import FigureLayout


_MM2INCH = 1 / 25.4
_UPDATE_THRESHOLD_MM = 0.1
_ALL_SIDES = {
    "title_space", "xlabel_space", "ylabel_space", "right",
    "legend_column", "legend_band_bottom", "legend_band_top", "legend_band_left",
    "suptitle_space",
}
# Cap on settle() draws; 1-3 is typical.
_MAX_CONVERGENCE_ITERS = 5

# side_name -> (axis_kind, bbox_fn)
#   axis_kind: "row" (result length == nrows) or "col" (length == ncols)
#   bbox_fn:   float (ax_bbox, tight_bbox) -> px
_SIDE_CALCULATORS = {
    "title_space":  ("row", lambda ax_bb, t: t.y1 - ax_bb.y1),
    "xlabel_space": ("row", lambda ax_bb, t: ax_bb.y0 - t.y0),
    "ylabel_space": ("col", lambda ax_bb, t: ax_bb.x0 - t.x0),
    "right":        ("col", lambda ax_bb, t: t.x1 - ax_bb.x1),
}


class SubplotsAutoLayout:
    """Per-figure draw-event listener that resizes the figure to fit decorations."""

    def __init__(self, fig, layout: FigureLayout, locked: Set[str]):
        self._fig = fig
        self._layout = layout
        self._locked = set(locked)
        self._updating = False

        fig._publiplots_layout = layout
        fig._publiplots_auto_layout = self

        if _ALL_SIDES.issubset(self._locked):
            self._cid = None
        else:
            self._cid = fig.canvas.mpl_connect("draw_event", self._on_draw)

        # Wrap savefig so that by the time it renders, the figure has
        # been resized to fit its current decorations. draw_event fires
        # AFTER the renderer has written to its buffer, so resizing
        # during draw_event is too late — the saved file captures the
        # pre-resize buffer. Running a settlement draw BEFORE savefig's
        # internal draw ensures the renderer is allocated at the final
        # size from the start.
        self._install_savefig_wrapper()

    def _install_savefig_wrapper(self) -> None:
        fig = self._fig
        if getattr(fig, "_publiplots_savefig_wrapped", False):
            return
        original_savefig = fig.savefig

        def _wrapped_savefig(*args, **kwargs):
            self.settle()
            return original_savefig(*args, **kwargs)

        fig.savefig = _wrapped_savefig
        fig._publiplots_savefig_wrapped = True

    def settle(self) -> None:
        """Drive the layout to convergence without leaving stale state.

        Runs canvas.draw() up to a small number of times. Each draw
        fires our _on_draw, which measures and (if needed) resizes.
        Once _needs_update returns False, we stop. This is the safe
        settlement primitive — unlike in-event iteration, each draw is
        a complete matplotlib pass with its own renderer, avoiding the
        reentrancy hazards of draw_without_rendering inside draw_event.
        """
        fig = self._fig
        for _ in range(_MAX_CONVERGENCE_ITERS):
            fig.canvas.draw()
            if not self._needs_update(self._measure()):
                return

    def _on_draw(self, event) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            new = self._measure()
            if self._needs_update(new):
                self._apply(new)
        finally:
            self._updating = False

    def _measure(self) -> Dict[str, Tuple[float, ...]]:
        fig = self._fig
        dpi = fig.dpi
        if dpi <= 0:
            return {}
        axes_matrix = self._axes_matrix()
        if not axes_matrix or not axes_matrix[0]:
            return {}

        managed = self._externally_managed_artist_ids()
        measured: Dict[str, Tuple[float, ...]] = {}

        for side, (axis_kind, calc) in _SIDE_CALCULATORS.items():
            if side in self._locked:
                continue
            if axis_kind == "row":
                per = []
                for row in axes_matrix:
                    max_px = 0.0
                    for ax in row:
                        max_px = max(max_px, self._side_extent(ax, calc, managed))
                    per.append(max(max_px / dpi * 25.4, 0.0))
                measured[side] = tuple(per)
            else:  # "col"
                ncols = len(axes_matrix[0])
                per = []
                for c in range(ncols):
                    max_px = 0.0
                    for row in axes_matrix:
                        ax = row[c]
                        max_px = max(max_px, self._side_extent(ax, calc, managed))
                    per.append(max(max_px / dpi * 25.4, 0.0))
                measured[side] = tuple(per)

        # Measure the single figure's legend_group (if any) and dispatch
        # its overhang to the right reservation field based on (side,
        # anchor_kind).
        self._apply_legend_band_measurement(measured, axes_matrix)
        # Measure pp.suptitle (if any) and write its mm height into
        # ``measured["suptitle_space"]``. Ordering is irrelevant —
        # suptitle_space is a dedicated scalar, independent of legend
        # bands.
        self._apply_suptitle_measurement(measured, dpi)
        return measured

    def _side_extent(self, ax, calc, managed_artist_ids) -> float:
        """Measure ax's tight-vs-window extent for one side, excluding managed overlays.

        Also accounts for reactor-managed artists that are NOT children of
        ``ax`` but are pinned to it — e.g. colorbar titles added via
        ``fig.text`` which live under the Figure, not the Axes.
        ``ax.get_tightbbox()`` misses those, so we manually union their
        window extents into the tight bbox before computing the side extent.
        """
        ax_bbox = ax.get_window_extent()
        # Temporarily drop managed overlay artists (legend_group's legends)
        # from layout consideration so they don't inflate the per-axis
        # reservations.
        toggled = []
        for child in ax.get_children():
            if id(child) in managed_artist_ids and child.get_in_layout():
                child.set_in_layout(False)
                toggled.append(child)
        try:
            tight = ax.get_tightbbox()
        finally:
            for child in toggled:
                child.set_in_layout(True)
        if tight is None:
            return 0.0

        # Union with pinned-but-not-child artists (per-axis colorbar
        # titles are fig.text artists registered to this axes via the
        # reactor). These sit inside the reserved side but are invisible
        # to ax.get_tightbbox() — without this union, the reservation
        # shrinks below what the title actually needs and the title gets
        # clipped on save.
        tight = self._union_pinned_artists(ax, tight, managed_artist_ids)
        return calc(ax_bbox, tight)

    def _union_pinned_artists(self, ax, tight, managed_artist_ids):
        """Union `tight` with extents of reactor-managed artists pinned to `ax`
        that are NOT among the excluded managed overlays."""
        reactor = getattr(self._fig, "_publiplots_layout_reactor", None)
        if reactor is None:
            return tight
        from matplotlib.transforms import Bbox
        for reg in reactor._registrations:
            if reg.ax is not ax:
                continue
            if id(reg.artist) in managed_artist_ids:
                continue  # external overlay — already excluded
            extent = self._artist_window_extent(reg.artist)
            if extent is None:
                continue
            tight = Bbox.union([tight, extent])
        return tight

    def _externally_managed_artist_ids(self) -> set:
        """IDs of LayoutReactor registrations flagged external_to_axis=True.

        The flag lives on _Registration in utils/layout_reactor.py (added by
        Task 4 of this amendment). Before Task 4 lands, getattr returns
        False and nothing is excluded — equivalent to pre-amendment
        behavior.
        """
        reactor = getattr(self._fig, "_publiplots_layout_reactor", None)
        if reactor is None:
            return set()
        return {
            id(reg.artist)
            for reg in reactor._registrations
            if getattr(reg, "external_to_axis", False)
        }

    _SCALAR_SIDES = {
        "legend_column", "legend_band_bottom", "legend_band_top", "legend_band_left",
        "suptitle_space",
    }

    def _needs_update(self, measured: Dict[str, Tuple[float, ...]]) -> bool:
        for side, new_val in measured.items():
            current = getattr(self._layout, side)
            if side in self._SCALAR_SIDES:
                if abs(new_val - current) >= _UPDATE_THRESHOLD_MM:
                    return True
            else:
                # tuple comparison (per-row / per-col reservations)
                if len(new_val) != len(current):
                    return True
                for nv, cv in zip(new_val, current):
                    if abs(nv - cv) >= _UPDATE_THRESHOLD_MM:
                        return True
        return False

    # Per-side overhang calculators. Each returns the signed distance
    # (in pixels) that 'obj' projects past the 'anchor_bb' edge in the
    # chosen direction. Distances are non-negative after the max() below.
    _OVERHANG_BY_SIDE = {
        "right":  lambda anchor_bb, obj_bb: obj_bb.x1 - anchor_bb.x1,
        "left":   lambda anchor_bb, obj_bb: anchor_bb.x0 - obj_bb.x0,
        "bottom": lambda anchor_bb, obj_bb: anchor_bb.y0 - obj_bb.y0,
        "top":    lambda anchor_bb, obj_bb: obj_bb.y1 - anchor_bb.y1,
    }

    # side → (figure-anchored FigureLayout field, axes-anchored per-cell field)
    _FIELD_BY_SIDE = {
        "right":  ("legend_column",       "right",        "col"),
        "left":   ("legend_band_left",    "ylabel_space", "col"),
        "bottom": ("legend_band_bottom",  "xlabel_space", "row"),
        "top":    ("legend_band_top",     "title_space",  "row"),
    }

    def _apply_suptitle_measurement(self, measured: dict, dpi: float) -> None:
        """Measure the pixel height of ``fig._publiplots_suptitle`` (if any)
        and write it into ``measured["suptitle_space"]`` in mm.

        A ``+1 mm`` safety margin is added (same convention as
        :meth:`_measure_one_group` below at the
        ``overhang_mm = max_overhang_px / dpi * 25.4 + 1.0`` line).
        When there is no suptitle, writes ``0.0`` so the reservation
        collapses back to zero.
        """
        if "suptitle_space" in self._locked:
            return
        artist = getattr(self._fig, "_publiplots_suptitle", None)
        if artist is None:
            measured["suptitle_space"] = 0.0
            return
        try:
            extent = artist.get_window_extent()
        except Exception:
            measured["suptitle_space"] = 0.0
            return
        if extent is None or extent.height <= 0:
            measured["suptitle_space"] = 0.0
            return
        measured["suptitle_space"] = extent.height / dpi * 25.4 + 1.0

    def _apply_legend_band_measurement(self, measured: dict, axes_matrix) -> None:
        """Measure every pp.legend_group's overhang and write it into
        the correct FigureLayout reservation based on each group's
        ``side`` and anchor kind.

        Multiple groups may coexist on the same figure (each scoped via
        ``axes=``); each contributes its own measurement. Per-cell
        reservations accumulate via ``max()`` so two axes-anchored
        groups targeting different cells both get room.
        """
        groups = getattr(self._fig, "_publiplots_legend_groups", None)
        if not groups:
            return
        dpi = self._fig.dpi
        if dpi <= 0:
            return

        for group in groups:
            self._measure_one_group(group, measured, axes_matrix, dpi)

    def _measure_one_group(self, group, measured, axes_matrix, dpi) -> None:
        # Force materialization so artists exist to measure.
        group._materialize()
        if not group._builder.elements:
            return

        # Single-axes, in-frame groups (external_to_axis=False) are measured
        # by ax.get_tightbbox() in _side_extent — no overhang write needed.
        # This guard prevents double-counting against the per-cell reservation
        # when commit 4 routes pp.legend(ax) through the same group machinery.
        if not getattr(group, "_external_to_axis", True):
            return

        side = group._side
        overhang_fn = self._OVERHANG_BY_SIDE[side]
        figure_field, cell_field, axis_kind = self._FIELD_BY_SIDE[side]

        anchor_bb = group.anchor.get_window_extent()
        max_overhang_px = 0.0
        for _, obj in group._builder.elements:
            extent = self._artist_window_extent(obj)
            if extent is None:
                continue
            max_overhang_px = max(max_overhang_px, overhang_fn(anchor_bb, extent))
        if max_overhang_px <= 0:
            # Band doesn't overhang yet (first-draw before reactor has
            # repositioned). Still re-bake the decoration offset so a
            # group constructed BEFORE plots picks up the offset as soon
            # as its entries materialize.
            self._bake_decoration_offset(group, measured, axes_matrix)
            return
        overhang_mm = max_overhang_px / dpi * 25.4 + 1.0

        if group._anchor_kind == "figure":
            if figure_field in self._locked:
                return
            # Figure-anchored: _GridAnchor already places the band past
            # all per-cell decorations, so no extra outward offset is
            # needed. Clear any stale offset from a prior layout pass.
            group._band_contribution_mm = overhang_mm
            group._set_decoration_offset(0.0)
            # Multiple figure-anchored groups on the same side compete
            # for the same band; take the tallest so neither clips.
            existing_scalar = measured.get(figure_field, 0.0)
            measured[figure_field] = max(existing_scalar, overhang_mm)
            return

        # Axes-anchored: grow the per-cell reservation for the anchor's
        # row/column. Merge with whatever the auto-measurement already
        # produced (so label/title space co-exists with legend space).
        if cell_field in self._locked:
            return
        ax = group.anchor
        r, c = self._find_ax_indices(ax, axes_matrix)
        # Read the pure-decoration reservation BEFORE we write our
        # overhang into it. _measure() already filled ``measured[cell_field]``
        # from _side_extent (which uses set_in_layout(False) on managed
        # overlays, i.e., OUR legend, so the measured slot is the
        # anchor's decoration size WITHOUT our band).
        existing = list(measured.get(cell_field, getattr(self._layout, cell_field)))
        idx = c if axis_kind == "col" else r
        pure_decoration_mm = existing[idx]
        # For side='right' there is no decoration past ax.x1 (tick labels
        # live inside ax), so this is already 0. For side='top' it equals
        # the title height above ax.y1; for 'bottom' the xlabel+ticks
        # below ax.y0; for 'left' the ylabel+ticks left of ax.x0. The
        # band must step past that amount to avoid overlap.
        group._band_contribution_mm = overhang_mm
        group._set_decoration_offset(pure_decoration_mm)
        # Grow the reservation by overhang_mm (capped against prior value
        # so multiple overlapping groups don't shrink it).
        existing[idx] = max(existing[idx], overhang_mm + pure_decoration_mm)
        measured[cell_field] = tuple(existing)

    def _bake_decoration_offset(self, group, measured, axes_matrix) -> None:
        """Write the decoration offset onto the group's registrations
        without touching its reservation. Used on first draw when the
        band hasn't rendered yet (no overhang to measure) but we still
        want the band to land past decorations once it materializes.
        """
        if group._anchor_kind != "axes":
            return
        side = group._side
        _, cell_field, axis_kind = self._FIELD_BY_SIDE[side]
        if cell_field in self._locked:
            return
        r, c = self._find_ax_indices(group.anchor, axes_matrix)
        existing = measured.get(cell_field, getattr(self._layout, cell_field))
        idx = c if axis_kind == "col" else r
        pure_decoration_mm = existing[idx] - group._band_contribution_mm
        if pure_decoration_mm < 0:
            pure_decoration_mm = 0.0
        group._set_decoration_offset(pure_decoration_mm)

    def _find_ax_indices(self, ax, axes_matrix):
        for r, row in enumerate(axes_matrix):
            for c, a in enumerate(row):
                if a is ax:
                    return r, c
        return 0, 0

    def _find_scope_indices(self, scope_axes, axes_matrix):
        """Return (row_indices, col_indices) touched by any axes in scope_axes.

        scope_axes is a list of matplotlib Axes. Returns two sorted lists of
        unique row/col indices within axes_matrix. Used by commit 4's
        multi-axes scope path to aggregate per-cell reservations via max().
        """
        rows, cols = set(), set()
        for ax in scope_axes:
            for r, row in enumerate(axes_matrix):
                for c, a in enumerate(row):
                    if a is ax:
                        rows.add(r)
                        cols.add(c)
        return sorted(rows), sorted(cols)

    def _artist_window_extent(self, obj):
        """Duck-typed tight-bbox accessor (Legend/Colorbar/Text).

        Returns the *tight* bbox, not the bare window extent — the tight
        bbox includes decorations attached outside the artist's
        rectangle, like colorbar tick labels and titles sitting past the
        color strip. Without that, the reactor measures only the narrow
        color strip and the tick labels get clipped on save.

        Legend.get_window_extent() already equals its tightbbox (the
        legend packs its own frame internally), so this call is
        idempotent there.
        """
        # Colorbar-like: geometry lives on a child Axes, which exposes
        # get_tightbbox. Use it so tick labels / titles are included.
        if hasattr(obj, "ax") and hasattr(obj.ax, "get_tightbbox"):
            return obj.ax.get_tightbbox()
        if hasattr(obj, "get_tightbbox"):
            try:
                return obj.get_tightbbox()
            except TypeError:
                # Some artists require a renderer arg.
                pass
        if hasattr(obj, "get_window_extent"):
            return obj.get_window_extent()
        if hasattr(obj, "ax"):
            return obj.ax.get_window_extent()
        return None

    def _apply(self, measured: Dict[str, Tuple[float, ...]]) -> None:
        new_layout = self._layout.with_updated_reservations(**measured)
        self._layout = new_layout
        self._fig._publiplots_layout = new_layout

        W, H = new_layout.figure_size()
        # forward=True propagates the new size to the GUI canvas so plt.show()
        # renders at the resized dimensions. Without it, the canvas keeps its
        # initial size and decorations that grew into the extra reservation get
        # cropped. The re-entrance guard (_updating) plus the 0.1 mm threshold
        # in _needs_update() keep this loop-safe.
        self._fig.set_size_inches(W * _MM2INCH, H * _MM2INCH, forward=True)

        for r, row in enumerate(self._axes_matrix()):
            for c, ax in enumerate(row):
                ax.set_position(new_layout.axes_position(r, c))

        # Reposition pp.suptitle (if any) to the vertical midpoint of
        # its reserved band. Runs after set_size_inches so the figure's
        # final height is known; the next draw renders the title at
        # the fixed fraction inside the grown canvas.
        suptitle = getattr(self._fig, "_publiplots_suptitle", None)
        if suptitle is not None and new_layout.suptitle_space > 0:
            y_mm = H - new_layout.outer_pad - new_layout.suptitle_space / 2
            suptitle.set_position((0.5, y_mm / H))
            suptitle.set_verticalalignment("center")
            suptitle.set_horizontalalignment("center")

    def _axes_matrix(self):
        stored = getattr(self._fig, "_publiplots_axes", None)
        if stored is not None:
            return stored
        flat = list(self._fig.axes)
        nrows, ncols = self._layout.nrows, self._layout.ncols
        if len(flat) < nrows * ncols:
            return [[]]
        return [flat[r * ncols:(r + 1) * ncols] for r in range(nrows)]
