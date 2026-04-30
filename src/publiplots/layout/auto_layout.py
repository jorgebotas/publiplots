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

    # At class scope, define the per-side bbox calculators as a constant:
    _SIDE_CALCULATORS = {
        "title_space": lambda ax_bbox, tight: tight.y1 - ax_bbox.y1,
        "xlabel_space": lambda ax_bbox, tight: ax_bbox.y0 - tight.y0,
        "ylabel_space": lambda ax_bbox, tight: ax_bbox.x0 - tight.x0,
        "right": lambda ax_bbox, tight: tight.x1 - ax_bbox.x1,
    }

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
        """Measure max decoration size per unlocked side across all axes, in mm."""
        fig = self._fig
        dpi = fig.dpi
        if dpi <= 0:
            return {}
        axes_matrix = self._axes_matrix()
        measured: dict = {}
        for side, calc in self._SIDE_CALCULATORS.items():
            if side in self._locked:
                continue
            max_px = 0.0
            for row in axes_matrix:
                for ax in row:
                    ax_bbox = ax.get_window_extent()
                    tight = ax.get_tightbbox()
                    if tight is None:
                        continue
                    max_px = max(max_px, calc(ax_bbox, tight))
            measured[side] = max(max_px / dpi * 25.4, 0.0)
        return measured

    def _needs_update(self, measured: dict) -> bool:
        for side, new_val in measured.items():
            current = getattr(self._layout, side)
            if abs(new_val - current) >= _UPDATE_THRESHOLD_MM:
                return True
        return False

    def _apply(self, measured: dict) -> None:
        new_layout = self._layout.with_updated_reservations(**measured)
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