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

from typing import List, Set, Tuple

import numpy as np
from matplotlib.axes import Axes

# Share-mode vocabulary (mirrors layout/subplots.py::_resolve_shared).
_X_HIDE_MODES = frozenset({True, "all", "col"})
_Y_HIDE_MODES = frozenset({True, "all", "row"})
_VALID_SHARE = frozenset({True, False, "all", "col", "row", "none"})


def _as_matrix(axes) -> List[List["Axes"]]:
    """Normalize ``axes`` to a list-of-lists grid (``mat[r][c]``).

    Preference order:
      1. If the axes belong to a publiplots figure, use the authoritative
         unsqueezed matrix at ``figure._publiplots_axes``.
      2. A 2D numpy array → its rows.
      3. A 1D numpy array → a single row (shape ``1 x len``). Documented:
         foreign callers wanting a column should pass a 2D / ``(n, 1)`` array.
      4. A single Axes → ``[[ax]]``.
    """
    # Single Axes object (has plotting API, not array-like).
    if hasattr(axes, "get_figure") and not isinstance(axes, np.ndarray):
        fig = axes.get_figure()
        stored = getattr(fig, "_publiplots_axes", None)
        if stored is not None:
            return [list(row) for row in stored]
        return [[axes]]

    arr = np.asarray(axes, dtype=object)
    # Recover the publiplots grid from any contained axes (handles squeeze).
    if arr.size:
        first = arr.flat[0]
        fig = getattr(first, "get_figure", lambda: None)()
        stored = getattr(fig, "_publiplots_axes", None) if fig is not None else None
        if stored is not None:
            return [list(row) for row in stored]

    if arr.ndim == 2:
        return [list(row) for row in arr]
    if arr.ndim == 1:
        return [list(arr)]
    if arr.ndim == 0:
        return [[arr.item()]]
    raise ValueError(f"axes array must be 0/1/2-dimensional, got ndim={arr.ndim}")


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


def label_outer(axes, *, sharex=True, sharey=True) -> None:
    """Hide interior tick labels, offset text, and axis labels on a grid.

    Leaves x labels only on the bottom row and y labels only on the left
    column — the publiplots equivalent of :meth:`matplotlib.axes.Axes.label_outer`
    for ``fig.add_axes``-built grids (which lack a ``SubplotSpec``).

    Parameters
    ----------
    axes : ndarray of Axes or Axes
        The grid to operate on. A publiplots grid is recovered from
        ``figure._publiplots_axes`` regardless of squeeze; a 2D array is used
        as-is; a 1D foreign array is treated as a single **row** (pass a 2D /
        ``(n, 1)`` array for a column).
    sharex : bool or {"all", "col", "row", "none"}, default True
        Hide x labels on non-bottom axes when in ``{True, "all", "col"}``.
    sharey : bool or {"all", "col", "row", "none"}, default True
        Hide y labels on non-left axes when in ``{True, "all", "row"}``.

    Returns
    -------
    None
        Operates in place, matching ``ax.label_outer()``.
    """
    mat = _as_matrix(axes)
    nrows = len(mat)
    ncols = len(mat[0]) if nrows else 0
    if nrows == 0 or ncols == 0:
        return
    hide_x, hide_y = _resolve_outer_edges(nrows, ncols, sharex, sharey)
    for (r, c) in hide_x:
        ax = mat[r][c]
        ax.tick_params(axis="x", labelbottom=False)
        ax.xaxis.offsetText.set_visible(False)
        ax.xaxis.label.set_visible(False)
    for (r, c) in hide_y:
        ax = mat[r][c]
        ax.tick_params(axis="y", labelleft=False)
        ax.yaxis.offsetText.set_visible(False)
        ax.yaxis.label.set_visible(False)
