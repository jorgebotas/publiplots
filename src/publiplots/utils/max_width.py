"""Cap the categorical-axis width of bars / boxes / violins to a mm value.

Used by :func:`publiplots.barplot`, :func:`publiplots.boxplot`, and
:func:`publiplots.violinplot` to translate the rcParams ``bar.max_width``
/ ``box.max_width`` / ``violin.max_width`` into a per-artist clamp. The
clamp preserves each artist's center on the categorical axis — naively
shrinking width via ``set_width`` would shift the visual center.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import FillBetweenPolyCollection
from matplotlib.patches import PathPatch, Rectangle


def clamp_patch_widths_mm(
    artists: Iterable,
    max_width_mm: float | None,
    ax: Axes,
    *,
    axis: Literal["x", "y"] = "x",
) -> None:
    """Clamp each artist's extent on ``axis`` to ``max_width_mm``, center-preserving.

    No-op when ``max_width_mm`` is ``None`` or non-positive. Dispatches by
    artist type: :class:`Rectangle` (bar), ``_RoundedBarPatch`` (rounded
    bar/box), :class:`PathPatch` (raw box), and
    :class:`FillBetweenPolyCollection` (violin). Other artist types are
    skipped. Pass ``axis="y"`` for horizontal bars / boxes / violins.
    """
    if max_width_mm is None or max_width_mm <= 0:
        return

    # Lazy import to avoid a cycle: utils.rounding may import from this
    # package. Mirrors the pattern in utils/offset.py.
    from publiplots.utils.rounding import _RoundedBarPatch
    from publiplots.annotate._positioning import mm_to_data

    max_data = abs(mm_to_data(max_width_mm, ax, axis))
    if max_data <= 0:
        return

    for artist in artists:
        if isinstance(artist, _RoundedBarPatch):
            _clamp_rounded(artist, max_data, axis)
        elif isinstance(artist, Rectangle):
            _clamp_rectangle(artist, max_data, axis)
        elif isinstance(artist, PathPatch):
            _clamp_pathpatch(artist, max_data, axis)
        elif isinstance(artist, FillBetweenPolyCollection):
            _clamp_violin_collection(artist, max_data, axis)


def _clamp_rectangle(r: Rectangle, max_data: float, axis: str) -> None:
    if axis == "x":
        w = r.get_width()
        if abs(w) <= max_data:
            return
        new_w = max_data if w >= 0 else -max_data
        r.set_x(r.get_x() + (w - new_w) / 2.0)
        r.set_width(new_w)
    else:
        h = r.get_height()
        if abs(h) <= max_data:
            return
        new_h = max_data if h >= 0 else -max_data
        r.set_y(r.get_y() + (h - new_h) / 2.0)
        r.set_height(new_h)


def _clamp_rounded(p, max_data: float, axis: str) -> None:
    """Clamp a _RoundedBarPatch using its public Rectangle-like API."""
    if axis == "x":
        w = p.get_width()
        if abs(w) <= max_data:
            return
        new_w = max_data if w >= 0 else -max_data
        x, y = p.get_xy()
        p.set_xy((x + (w - new_w) / 2.0, y))
        # Public attribute carrying the data-coord width (see _RoundedBarPatch).
        p._rounded_width = float(new_w)
        p.stale = True
    else:
        h = p.get_height()
        if abs(h) <= max_data:
            return
        new_h = max_data if h >= 0 else -max_data
        x, y = p.get_xy()
        p.set_xy((x, y + (h - new_h) / 2.0))
        p._rounded_height = float(new_h)
        p.stale = True


def _clamp_pathpatch(p: PathPatch, max_data: float, axis: str) -> None:
    """Clamp a rectangular PathPatch by mutating its path vertices."""
    path = p.get_path()
    verts = path.vertices
    col = 0 if axis == "x" else 1
    coords = verts[:, col]
    cmin, cmax = float(coords.min()), float(coords.max())
    cur = cmax - cmin
    if cur <= max_data:
        return
    center = (cmin + cmax) / 2.0
    half = max_data / 2.0
    # Map every vertex coord on this axis to its clamped equivalent,
    # preserving its side relative to the center.
    new_coords = np.where(
        coords > center,
        np.minimum(coords, center + half),
        np.maximum(coords, center - half),
    )
    # For pure rectangles (vertices sit on cmin or cmax), the result is
    # exactly [center-half, center+half]; for off-rect vertices the sided
    # min/max keeps them on their side without overshooting.
    verts[:, col] = new_coords
    p.stale = True


def _clamp_violin_collection(
    coll: FillBetweenPolyCollection, max_data: float, axis: str
) -> None:
    """Scale violin paths toward each path's center on ``axis``.

    For two-sided violins the cap is the full width; for half-violins
    (side="left"|"right" — many vertices stack at the clipped edge) the
    cap is the half-width, and the path is scaled toward that edge.
    """
    col = 0 if axis == "x" else 1
    new_verts = []
    for path in coll.get_paths():
        verts = path.vertices.copy()
        coords = verts[:, col]
        vmin, vmax = float(coords.min()), float(coords.max())
        n = len(coords)
        # Side-clipped half-violins stack many vertices at the clipped edge
        # (≥25% of vertices, well above 1-2 from a normal cap).
        thresh = max(0.25 * n, 5)
        n_at_max = int(np.sum(np.isclose(coords, vmax, atol=1e-9)))
        n_at_min = int(np.sum(np.isclose(coords, vmin, atol=1e-9)))
        if n_at_max >= thresh and n_at_max > n_at_min:
            # Half-violin clipped on the right side of the spike.
            center, half, target_half = vmax, vmax - vmin, max_data
        elif n_at_min >= thresh and n_at_min > n_at_max:
            center, half, target_half = vmin, vmax - vmin, max_data
        else:
            center = (vmin + vmax) / 2.0
            half = (vmax - vmin) / 2.0
            target_half = max_data / 2.0
        if half <= 0 or target_half >= half:
            new_verts.append(verts)
            continue
        scale = target_half / half
        verts[:, col] = center + (coords - center) * scale
        new_verts.append(verts)
    coll.set_verts(new_verts)


__all__ = ["clamp_patch_widths_mm"]
