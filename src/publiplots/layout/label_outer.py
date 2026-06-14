"""
Outer-only axis labeling for shared-axes grids (seaborn parity).

publiplots builds its subplot grid with ``fig.add_axes()`` (no
``SubplotSpec``), so matplotlib's :meth:`~matplotlib.axes.Axes.label_outer`
is unavailable. This module computes the "outer edge" from publiplots' own
axes matrix and hides interior tick labels, offset text, and axis labels.

Hiding is persistence-safe across the figure's multi-draw ``settle()`` loop:
it uses ``ax.tick_params(labelbottom/labelleft=False)`` (which survives
locator-driven tick-label regeneration) plus ``set_visible(False)`` on the
stable offset-text and axis-label artists.
"""

from typing import Set, Tuple

import numpy as np

# Share-mode vocabulary (mirrors layout/subplots.py::_resolve_shared).
_X_HIDE_MODES = frozenset({True, "all", "col"})
_Y_HIDE_MODES = frozenset({True, "all", "row"})
_VALID_SHARE = frozenset({True, False, "all", "col", "row", "none"})


def _resolve_outer_edges(
    nrows: int, ncols: int, sharex, sharey
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Return ``(hide_x, hide_y)`` sets of ``(r, c)`` positions to hide.

    x labels are hidden on non-bottom axes when ``sharex`` is in
    ``{True, "all", "col"}``; y labels on non-left axes when ``sharey`` is in
    ``{True, "all", "row"}``. All other share modes hide nothing on that axis.
    """
    if sharex not in _VALID_SHARE:
        raise ValueError(
            f"sharex must be one of {sorted(_VALID_SHARE, key=str)}, got {sharex!r}"
        )
    if sharey not in _VALID_SHARE:
        raise ValueError(
            f"sharey must be one of {sorted(_VALID_SHARE, key=str)}, got {sharey!r}"
        )

    hide_x: Set[Tuple[int, int]] = set()
    hide_y: Set[Tuple[int, int]] = set()
    if sharex in _X_HIDE_MODES:
        for r in range(nrows - 1):  # every row except the bottom
            for c in range(ncols):
                hide_x.add((r, c))
    if sharey in _Y_HIDE_MODES:
        for r in range(nrows):
            for c in range(1, ncols):  # every col except the left
                hide_y.add((r, c))
    return hide_x, hide_y
