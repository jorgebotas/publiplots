"""
Per-figure reactor that keeps publiplots legends/colorbars positioned
correctly across layout changes (tight_layout, subplots_adjust,
constrained_layout passes, and downstream axes repositioning).

The reactor stores mm offsets relative to the anchor axes. On every draw,
it re-reads ax.get_position() and updates each registered artist's
bbox_to_anchor to match the current axes edge.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox


_MM2INCH = 1 / 25.4

_DISPLACEMENT_WARNING = (
    "A LegendBuilder element was displaced by a layout change "
    "(likely plt.tight_layout() or fig.subplots_adjust). "
    "publiplots enables constrained_layout in set_notebook_style() and "
    "set_publication_style() — using those avoids this issue. The element "
    "was re-anchored automatically; rendered output is correct."
)


@dataclass
class _Registration:
    ax: Axes
    artist: object  # Legend or Colorbar; duck-typed via set_bbox_to_anchor
    mm_x_from_right: float
    mm_y_from_top: float
    mm_width: Optional[float] = None
    mm_height: Optional[float] = None


class LayoutReactor:
    """
    Per-figure singleton that refreshes anchored element positions on each draw.

    Obtain via LayoutReactor.get(fig); the reactor attaches itself to the
    figure's draw_event and stays alive for the lifetime of the figure.
    """

    _ATTR = "_publiplots_layout_reactor"

    @classmethod
    def get(cls, fig: Figure) -> "LayoutReactor":
        existing = getattr(fig, cls._ATTR, None)
        if existing is not None:
            return existing
        reactor = cls(fig)
        setattr(fig, cls._ATTR, reactor)
        fig.canvas.mpl_connect("draw_event", reactor._on_draw)
        return reactor

    def __init__(self, fig: Figure) -> None:
        self._fig = fig
        self._registrations: List[_Registration] = []
        self._last_positions: dict = {}  # id(ax) -> Bbox
        self._warned: bool = False
        self._updating: bool = False  # re-entrancy guard

    def register(
        self,
        ax: Axes,
        artist: object,
        mm_x_from_right: float,
        mm_y_from_top: float,
        mm_width: Optional[float] = None,
        mm_height: Optional[float] = None,
    ) -> None:
        """Track this element; its bbox_to_anchor will be refreshed every draw.

        For legends, leave mm_width/mm_height as None. For colorbars, pass the
        colorbar's mm dimensions so the colorbar axes can be repositioned via
        set_position (colorbars do not respond to bbox_to_anchor).
        """
        self._registrations.append(_Registration(
            ax=ax,
            artist=artist,
            mm_x_from_right=mm_x_from_right,
            mm_y_from_top=mm_y_from_top,
            mm_width=mm_width,
            mm_height=mm_height,
        ))

    def _on_draw(self, event) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            any_displaced = self._refresh_all()
            if any_displaced and not self._warned and not self._has_constrained_layout():
                warnings.warn(_DISPLACEMENT_WARNING, UserWarning, stacklevel=2)
                self._warned = True
        finally:
            self._updating = False

    def _refresh_all(self) -> bool:
        """Refresh every registration's bbox_to_anchor. Return True if any moved."""
        any_displaced = False
        for reg in self._registrations:
            pos = reg.ax.get_position()
            last = self._last_positions.get(id(reg.ax))
            if last is not None and not _bboxes_equal(pos, last):
                any_displaced = True
            self._last_positions[id(reg.ax)] = pos.frozen()
            self._update_artist_anchor(reg)
        return any_displaced

    def _update_artist_anchor(self, reg: _Registration) -> None:
        from matplotlib.text import Text

        fig = self._fig
        ax_pos = reg.ax.get_position()
        fig_extent = fig.get_window_extent()
        # Convert mm offsets to figure fractions
        x_frac = (reg.mm_x_from_right * _MM2INCH * fig.dpi) / fig_extent.width
        y_frac = (reg.mm_y_from_top * _MM2INCH * fig.dpi) / fig_extent.height
        new_x = ax_pos.x1 + x_frac
        new_y = ax_pos.y1 - y_frac

        # Colorbar path — artist has .ax and we stored mm dimensions.
        if reg.mm_width is not None and reg.mm_height is not None:
            w_frac = (reg.mm_width * _MM2INCH * fig.dpi) / fig_extent.width
            h_frac = (reg.mm_height * _MM2INCH * fig.dpi) / fig_extent.height
            reg.artist.ax.set_position([new_x, new_y - h_frac, w_frac, h_frac])
            return

        # Text path — figure-level text (e.g., colorbar titles added via fig.text).
        # set_position takes (x, y) in the text's current transform; since we pin
        # these via transform=fig.transFigure at creation time, figure fractions
        # are correct here.
        if isinstance(reg.artist, Text):
            reg.artist.set_position((new_x, new_y))
            return

        # Legend path.
        if hasattr(reg.artist, "set_bbox_to_anchor"):
            reg.artist.set_bbox_to_anchor((new_x, new_y), transform=fig.transFigure)

    def _has_constrained_layout(self) -> bool:
        engine = self._fig.get_layout_engine()
        if engine is None:
            return False
        return type(engine).__name__ == "ConstrainedLayoutEngine"


def _bboxes_equal(a, b, tol_frac: float = 3e-4) -> bool:
    """Compare two Bbox objects in figure-fraction space with 0.5-pixel tolerance."""
    ap = a.get_points()
    bp = b.get_points()
    return bool(
        abs(ap[0, 0] - bp[0, 0]) < tol_frac
        and abs(ap[0, 1] - bp[0, 1]) < tol_frac
        and abs(ap[1, 0] - bp[1, 0]) < tol_frac
        and abs(ap[1, 1] - bp[1, 1]) < tol_frac
    )