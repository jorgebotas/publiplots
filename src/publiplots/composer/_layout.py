"""Pure-geometry multi-row canvas layout.

Computes canvas dimensions and per-panel mm-rects for a multi-row
canvas. No matplotlib imports — pure math.

The canvas is laid out top-to-bottom: row 0 at the top, row R-1 at the
bottom. y-coordinates use matplotlib's bottom-left origin convention,
so row 0's y_mm > row 1's y_mm > ... > row (R-1)'s y_mm.

A row carries its own panel widths (post-flex resolution), a single
row_height (max of its panel heights), and a vpad (mm above this row;
0 for the first row).

Decoration reservations (title_space, xlabel_space, ylabel_space, right)
are scalar in PR 3 — they apply uniformly to every row/column. This
matches what `pp.subplots`' `SubplotsAutoLayout` will measure at draw
time. Per-row variation is left for a future PR if user feedback
demands it.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class CanvasGeometry:
    """Result of :func:`compute_canvas_geometry`.

    Attributes
    ----------
    canvas_width_mm, canvas_height_mm : float
        Overall canvas dimensions.
    row_axes_rects_mm : list of list of (x, y, w, h)
        ``row_axes_rects_mm[r][c]`` is the mm rect for panel c in row r.
        Coordinates use bottom-left origin. ``(x, y)`` is the bottom-
        left corner of the panel's axes (NOT including ylabel/xlabel
        decoration space).
    """

    canvas_width_mm: float
    canvas_height_mm: float
    row_axes_rects_mm: List[List[Tuple[float, float, float, float]]]


def compute_canvas_geometry(
    *,
    rows: Sequence[Dict],
    canvas_width_mm: float,
    outer_pad: float,
    ylabel_space: float,
    right: float,
    wspace: float,
    title_space: float,
    xlabel_space: float,
) -> CanvasGeometry:
    """Compute canvas dimensions + per-panel mm rects for multi-row layouts.

    Parameters
    ----------
    rows : sequence of dict
        Each row dict has keys:
        - ``panel_widths_mm`` (tuple of float, after flex resolution)
        - ``row_height_mm`` (float — max of panel heights in the row)
        - ``vpad_mm`` (float — mm above this row; 0 for first row)
    canvas_width_mm : float
        The canvas's declared width (NOT necessarily what the figure
        ends up — that depends on whether any row had flex panels).
    outer_pad, ylabel_space, right, wspace, title_space, xlabel_space : float
        rcParams-derived reservations in mm.

    Returns
    -------
    :class:`CanvasGeometry`
    """
    if not rows:
        raise ValueError("compute_canvas_geometry requires at least one row")

    # Canvas width = max of any row's required width. Each row's required
    # width = outer_pad + sum(ylabel + panel_w + right) + (n-1)*wspace + outer_pad.
    row_required_widths = []
    for row in rows:
        widths = row["panel_widths_mm"]
        n = len(widths)
        if n == 0:
            raise ValueError("each row must have at least one panel")
        rw = (
            2 * outer_pad
            + n * ylabel_space
            + sum(widths)
            + n * right
            + max(n - 1, 0) * wspace
        )
        row_required_widths.append(rw)
    canvas_width = max(row_required_widths)

    # The user-supplied canvas_width_mm is informational context (e.g. the
    # journal's column width) but does NOT inflate the actual figure
    # width — we use the max-row required width. This matches PR 1's
    # behavior (figure width = panels + decorations) and treats the
    # canvas budget as a CAP/HINT rather than a TARGET.
    canvas_width_mm = canvas_width

    # Canvas height = outer_pad + sum_rows(title + row_height + xlabel + vpad) + outer_pad.
    canvas_height = 2 * outer_pad
    for row in rows:
        canvas_height += row["vpad_mm"]
        canvas_height += title_space
        canvas_height += row["row_height_mm"]
        canvas_height += xlabel_space

    # Compute per-panel rects. Iterate top-down (row 0 at top), tracking
    # the running y-cursor from the top of the canvas downward.
    row_axes_rects: List[List[Tuple[float, float, float, float]]] = []
    y_cursor_from_top = outer_pad  # mm from the top of the canvas
    for row in rows:
        widths = row["panel_widths_mm"]
        row_h = row["row_height_mm"]

        y_cursor_from_top += row["vpad_mm"]
        y_cursor_from_top += title_space

        # The panel's axes-rect y_mm (bottom-left origin) = canvas_height - y_cursor - row_h
        y_mm = canvas_height - y_cursor_from_top - row_h

        # Per-panel x positions: left-justify (justify='start' default)
        x_cursor = outer_pad
        rects = []
        for w in widths:
            x_cursor += ylabel_space
            rects.append((x_cursor, y_mm, w, row_h))
            x_cursor += w
            x_cursor += right
            x_cursor += wspace
        row_axes_rects.append(rects)

        y_cursor_from_top += row_h
        y_cursor_from_top += xlabel_space

    return CanvasGeometry(
        canvas_width_mm=canvas_width_mm,
        canvas_height_mm=canvas_height,
        row_axes_rects_mm=row_axes_rects,
    )
