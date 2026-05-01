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
_ALL_SIDES = {"title_space", "xlabel_space", "ylabel_space", "right", "legend_column"}
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

        if "legend_column" not in self._locked:
            measured["legend_column"] = self._measure_legend_column()
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

    def _needs_update(self, measured: Dict[str, Tuple[float, ...]]) -> bool:
        for side, new_val in measured.items():
            current = getattr(self._layout, side)
            if side == "legend_column":
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

    def _measure_legend_column(self) -> float:
        """mm width past the anchor axes' right edge, plus 1 mm padding."""
        group = getattr(self._fig, "_publiplots_legend_group", None)
        if group is None:
            return 0.0

        # Force collection so artists exist to measure.
        group._materialize()
        if not group._builder.elements:
            return 0.0

        dpi = self._fig.dpi
        if dpi <= 0:
            return 0.0

        anchor_bb = group.anchor.get_window_extent()
        max_x1 = anchor_bb.x1
        for _, obj in group._builder.elements:
            extent = self._artist_window_extent(obj)
            if extent is None:
                continue
            max_x1 = max(max_x1, extent.x1)
        overhang_px = max_x1 - anchor_bb.x1
        if overhang_px <= 0:
            return 0.0
        return overhang_px / dpi * 25.4 + 1.0

    def _artist_window_extent(self, obj):
        """Duck-typed window-extent accessor (Legend/Colorbar/Text)."""
        if hasattr(obj, "get_window_extent"):
            return obj.get_window_extent()
        if hasattr(obj, "ax"):  # Colorbar stores geometry on .ax
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

    def _axes_matrix(self):
        stored = getattr(self._fig, "_publiplots_axes", None)
        if stored is not None:
            return stored
        flat = list(self._fig.axes)
        nrows, ncols = self._layout.nrows, self._layout.ncols
        if len(flat) < nrows * ncols:
            return [[]]
        return [flat[r * ncols:(r + 1) * ncols] for r in range(nrows)]
