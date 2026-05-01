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
_ALL_SIDES = {"title_space", "xlabel_space", "ylabel_space", "right"}

# Maximum measure/apply iterations per draw event. Convergence in 1-3 is
# typical; the cap protects against tick-locator feedback on degenerate
# axes (e.g., empty numeric axes whose right-tick overhang grows each
# time the figure widens).
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

    def _on_draw(self, event) -> None:
        """Iterate measure/apply to convergence inside a single draw event.

        Measurement depends on the figure's current size (tick locators pick
        different numbers of ticks at different widths; wrapped tick labels
        change height as columns narrow). A single measure/apply pair would
        leave the canvas visibly out of sync with the logical layout — the
        symptom is plt.show() rendering at the pre-apply size (titles /
        legends cropped) and successive fig.canvas.draw() calls oscillating
        between locally stable sizes.

        We resolve both by iterating until _needs_update returns False, using
        ``fig.draw_without_rendering()`` between iterations so tick locators
        and wrapped-text artists update their positions against the latest
        canvas size. All intermediate set_size_inches calls use
        ``forward=False`` to avoid emitting spurious resize events; the final
        converged size is pushed with ``forward=True`` so plt.show() and
        savefig see the canonical dimensions.

        The re-entrance guard prevents the inner draw_without_rendering from
        recursing back into this method.
        """
        if self._updating:
            return
        self._updating = True
        try:
            for _ in range(_MAX_CONVERGENCE_ITERS):
                new = self._measure()
                if not self._needs_update(new):
                    self._sync_canvas_size()
                    return
                self._apply(new, forward=False)
                # Let matplotlib recompute tick positions, text layout, and
                # artist bboxes against the new figure size before we
                # measure again. Without this the next _measure sees stale
                # tight-bboxes and the loop would be a no-op.
                self._fig.draw_without_rendering()
            # Did not converge: push the current logical size to the canvas
            # so at least the GUI / savefig sees a self-consistent state.
            self._sync_canvas_size()
        finally:
            self._updating = False

    def _sync_canvas_size(self) -> None:
        """Ensure the canvas size matches the layout's figure_size (forward=True)."""
        W, H = self._layout.figure_size()
        target_w, target_h = W * _MM2INCH, H * _MM2INCH
        cur_w, cur_h = self._fig.get_size_inches()
        if abs(cur_w - target_w) > 1e-4 or abs(cur_h - target_h) > 1e-4:
            self._fig.set_size_inches(target_w, target_h, forward=True)

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
        return measured

    def _side_extent(self, ax, calc, managed_artist_ids) -> float:
        """Measure ax's tight-vs-window extent for one side, excluding managed overlays."""
        ax_bbox = ax.get_window_extent()
        # Temporarily drop managed artists from layout consideration.
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
        return calc(ax_bbox, tight)

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
        for side, new_tuple in measured.items():
            current = getattr(self._layout, side)
            if len(new_tuple) != len(current):
                return True
            for new_v, cur_v in zip(new_tuple, current):
                if abs(new_v - cur_v) >= _UPDATE_THRESHOLD_MM:
                    return True
        return False

    def _apply(
        self,
        measured: Dict[str, Tuple[float, ...]],
        *,
        forward: bool = True,
    ) -> None:
        """Commit new reservations to the layout and resize the figure.

        ``forward`` is False during intra-draw convergence iterations so the
        canvas does not emit resize events between measurements; the final
        converged size is propagated by ``_sync_canvas_size()`` with
        ``forward=True``.
        """
        new_layout = self._layout.with_updated_reservations(**measured)
        self._layout = new_layout
        self._fig._publiplots_layout = new_layout

        W, H = new_layout.figure_size()
        self._fig.set_size_inches(W * _MM2INCH, H * _MM2INCH, forward=forward)

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
