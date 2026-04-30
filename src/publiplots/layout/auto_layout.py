"""
Draw-event hook that keeps declared axes sizes fixed while the figure
grows to fit auto-measured decorations (titles, axis labels, tick labels).

Only the four per-cell reservations are auto-measured:
title_space, xlabel_space, ylabel_space, right.

Gaps (hspace, wspace) and outer_pad are never remeasured — users set
them explicitly or inherit rcParams defaults.

Cooperates with LayoutReactor (utils/layout_reactor.py): both react to
draw_event, but SubplotsAutoLayout is registered first (during
pp.subplots()) and therefore fires first, so LayoutReactor sees the
repositioned axes and re-anchors legends correctly.
"""

from typing import Set

from publiplots.layout.figure_layout import FigureLayout


_MM2INCH = 1 / 25.4
_UPDATE_THRESHOLD_MM = 0.1
_ALL_SIDES = {"title_space", "xlabel_space", "ylabel_space", "right"}


class SubplotsAutoLayout:
    """Per-figure draw-event listener that resizes the figure to fit decorations."""

    def __init__(self, fig, layout: FigureLayout, locked: Set[str]):
        self._fig = fig
        self._layout = layout
        self._locked = set(locked)
        self._updating = False

        # Expose the layout on the figure for introspection and the
        # future composer PR. Public-ish but underscore-prefixed until
        # the composer API ships.
        fig._publiplots_layout = layout
        fig._publiplots_auto_layout = self

        # If every auto-measurable side is locked, no hook is needed —
        # figure size is fully deterministic.
        if _ALL_SIDES.issubset(self._locked):
            self._cid = None
        else:
            self._cid = fig.canvas.mpl_connect("draw_event", self._on_draw)

    def _on_draw(self, event):
        if self._updating:
            return
        self._updating = True
        try:
            new = self._measure()
            if self._needs_update(new):
                self._apply(new)
        finally:
            self._updating = False

    def _measure(self) -> dict:
        """Measure max decoration size per side across all axes, in mm."""
        fig = self._fig
        dpi = fig.dpi
        if dpi <= 0:
            return {}
        # Collect axes in grid order
        axes_matrix = self._axes_matrix()
        measured: dict = {}

        def _mm(px: float) -> float:
            return px / dpi * 25.4

        if "title_space" not in self._locked:
            max_top_px = 0.0
            for row in axes_matrix:
                for ax in row:
                    ax_bbox = ax.get_window_extent()
                    tight = ax.get_tightbbox()
                    if tight is None:
                        continue
                    max_top_px = max(max_top_px, tight.y1 - ax_bbox.y1)
            measured["title_space"] = max(_mm(max_top_px), 0.0)

        if "xlabel_space" not in self._locked:
            max_bot_px = 0.0
            for row in axes_matrix:
                for ax in row:
                    ax_bbox = ax.get_window_extent()
                    tight = ax.get_tightbbox()
                    if tight is None:
                        continue
                    max_bot_px = max(max_bot_px, ax_bbox.y0 - tight.y0)
            measured["xlabel_space"] = max(_mm(max_bot_px), 0.0)

        if "ylabel_space" not in self._locked:
            max_left_px = 0.0
            for row in axes_matrix:
                for ax in row:
                    ax_bbox = ax.get_window_extent()
                    tight = ax.get_tightbbox()
                    if tight is None:
                        continue
                    max_left_px = max(max_left_px, ax_bbox.x0 - tight.x0)
            measured["ylabel_space"] = max(_mm(max_left_px), 0.0)

        if "right" not in self._locked:
            max_right_px = 0.0
            for row in axes_matrix:
                for ax in row:
                    ax_bbox = ax.get_window_extent()
                    tight = ax.get_tightbbox()
                    if tight is None:
                        continue
                    max_right_px = max(max_right_px, tight.x1 - ax_bbox.x1)
            measured["right"] = max(_mm(max_right_px), 0.0)

        return measured

    def _needs_update(self, measured: dict) -> bool:
        for side, new_val in measured.items():
            current = getattr(self._layout, side)
            # Only update if the new measurement requires more space
            if new_val > current + _UPDATE_THRESHOLD_MM:
                return True
        return False

    def _apply(self, measured: dict) -> None:
        # Only grow reservations, never shrink them
        updates = {}
        for side, new_val in measured.items():
            current = getattr(self._layout, side)
            if new_val > current + _UPDATE_THRESHOLD_MM:
                updates[side] = new_val

        if not updates:
            return

        new_layout = self._layout.with_updated_reservations(**updates)
        self._layout = new_layout
        self._fig._publiplots_layout = new_layout

        W, H = new_layout.figure_size()
        self._fig.set_size_inches(W * _MM2INCH, H * _MM2INCH, forward=False)

        for r, row in enumerate(self._axes_matrix()):
            for c, ax in enumerate(row):
                ax.set_position(new_layout.axes_position(r, c))

    def _axes_matrix(self):
        """
        Return the axes in row-major order.

        Stored by pp.subplots() as ``fig._publiplots_axes`` (list of lists).
        For figures built manually in tests, this falls back to walking
        ``fig.axes`` in insertion order.
        """
        stored = getattr(self._fig, "_publiplots_axes", None)
        if stored is not None:
            return stored
        # Fallback: reshape fig.axes to (nrows, ncols)
        flat = list(self._fig.axes)
        nrows, ncols = self._layout.nrows, self._layout.ncols
        if len(flat) < nrows * ncols:
            return [[]]
        return [flat[r * ncols:(r + 1) * ncols] for r in range(nrows)]