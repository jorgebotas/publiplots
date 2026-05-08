"""Rounded-corner rendering for Rectangle-based plot primitives.

This module provides a plot-agnostic helper that swaps
:class:`matplotlib.patches.Rectangle` patches (e.g. from
``seaborn.barplot``) for :class:`_RoundedBarPatch` instances with
independent top / bottom corner radii.

**Units**: radii are specified in millimeters and resolved to data
coordinates **at draw time** using the parent axes' ``transData``,
separately for x and y. This keeps corners visually circular on
screen/print regardless of the data-axis range and aspect ratio — the
central benefit over building a pre-baked path in data coords (which
would render elliptical rounding on non-square aspect ratios).

The intended consumer is :func:`publiplots.barplot`, but the helper is
designed to generalize to any Rectangle-based primitive.
"""

from typing import Optional, Sequence, Tuple, Union

from matplotlib.axes import Axes
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform

# Users supply radii in mm; the draw-time resolver needs points.
_MM_TO_POINTS = 72.0 / 25.4


BorderRadiusLike = Union[float, int, Tuple[float, float], None]


def normalize_border_radius(
    value: BorderRadiusLike,
) -> Tuple[float, float]:
    """Coerce user input to a canonical ``(top_mm, bottom_mm)`` tuple.

    Parameters
    ----------
    value : float, int, 2-tuple of float, or None
        User-facing input:
        - ``None`` → ``(0.0, 0.0)`` (flat corners, default).
        - scalar number → symmetric: ``(value, value)``.
        - 2-tuple → ``(top_mm, bottom_mm)`` passthrough (cast to float).

    Returns
    -------
    tuple of float
        ``(top_mm, bottom_mm)``.

    Raises
    ------
    TypeError
        If the value is neither a number, a 2-tuple/list, nor ``None``.
    """
    if value is None:
        return (0.0, 0.0)
    if isinstance(value, bool):
        # Guard: bool is a subclass of int; reject to avoid surprise.
        raise TypeError(
            f"border_radius must be a number (symmetric) or (top, bottom) "
            f"tuple, got {value!r}"
        )
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise TypeError(
        f"border_radius must be a number (symmetric) or (top, bottom) "
        f"tuple, got {value!r}"
    )


class _RoundedBarPatch(Patch):
    """Rectangle-shaped patch with independent top / bottom corner radii.

    Radii are carried as **points**, not data units. Each call to
    :meth:`get_path` (triggered by the renderer every draw) converts
    the stored point radii to data coords using the parent axes'
    ``transData``, applying different conversions to x and y so that
    on-screen corners remain circular on any aspect ratio.

    Corners are rendered with quadratic Bezier curves
    (``Path.CURVE3``), which matplotlib's Path API represents as
    ``(control_vert, end_vert)`` pairs. The control vertex sits at
    the literal rectangle corner, the end vertex on the adjacent edge
    — the same convention matplotlib's built-in
    :class:`matplotlib.patches.BoxStyle.Round` uses internally.

    Parameters
    ----------
    xy : tuple of float
        Lower-left corner ``(x, y)`` in data coords.
    width, height : float
        Rectangle width and height in data coords.
    top_pts, bottom_pts : float
        Corner radii in points. Use
        :func:`normalize_border_radius` + the ``_MM_TO_POINTS``
        constant to convert from user-facing mm input.
    **kwargs
        Forwarded to :class:`matplotlib.patches.Patch` (e.g. facecolor,
        edgecolor, linewidth, hatch, alpha, zorder, transform,
        clip_on, label).
    """

    def __init__(
        self,
        xy: Tuple[float, float],
        width: float,
        height: float,
        top_pts: float,
        bottom_pts: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._rounded_xy = (float(xy[0]), float(xy[1]))
        self._rounded_width = float(width)
        self._rounded_height = float(height)
        self._top_pts = float(top_pts)
        self._bottom_pts = float(bottom_pts)

    # Bar-like getters for compatibility with code that walks
    # ax.patches and checks hasattr(p, "get_height") to filter bars
    # from other patches (e.g. `_paint_bars`, annotate builders).
    def get_x(self) -> float:  # noqa: D401 — mirror Rectangle API
        """Return the lower-left x in data coords."""
        return self._rounded_xy[0]

    def get_y(self) -> float:
        """Return the lower-left y in data coords."""
        return self._rounded_xy[1]

    def get_xy(self) -> Tuple[float, float]:
        """Return the lower-left ``(x, y)`` in data coords."""
        return self._rounded_xy

    def get_width(self) -> float:
        """Return the width in data coords."""
        return self._rounded_width

    def get_height(self) -> float:
        """Return the height in data coords."""
        return self._rounded_height

    def get_patch_transform(self):
        """Return identity — :meth:`get_path` already emits data coords.

        Patch.get_transform composes ``get_patch_transform() +
        Artist.get_transform()``. We want the final transform to equal
        ``ax.transData`` (the Artist transform), so the patch-local
        part must be identity. Otherwise a default ``BboxTransformTo``
        would map our already-in-data-coords path through a second
        transform and render at the wrong scale.
        """
        return IdentityTransform()

    def _data_per_point(self) -> Tuple[float, float]:
        """Return ``(data_per_pt_x, data_per_pt_y)`` via the axes transform.

        Falls back to ``(0, 0)`` (which degenerates to a flat rect in
        :meth:`get_path`) when the patch is unparented or the axes has
        no working transform yet.
        """
        ax = self.axes
        if ax is None:
            return (0.0, 0.0)
        fig = ax.figure
        if fig is None:
            return (0.0, 0.0)
        dpi = fig.dpi
        if not dpi:
            return (0.0, 0.0)
        pts_to_px = dpi / 72.0

        # Use a small reference segment at the bar's origin so we pick
        # up per-axis scaling (handles log scales near the bar, not
        # globally — close enough for the kind of bars users draw).
        x0, y0 = self._rounded_xy
        try:
            p0 = ax.transData.transform((x0, y0))
            px = ax.transData.transform((x0 + 1.0, y0))
            py = ax.transData.transform((x0, y0 + 1.0))
        except Exception:
            return (0.0, 0.0)

        dx_px = abs(px[0] - p0[0])
        dy_px = abs(py[1] - p0[1])
        dpx = pts_to_px / dx_px if dx_px else 0.0
        dpy = pts_to_px / dy_px if dy_px else 0.0
        return (dpx, dpy)

    def get_path(self) -> Path:
        """Build the rounded-rectangle path in data coords."""
        x0, y0 = self._rounded_xy
        w = self._rounded_width
        h = self._rounded_height
        x1, y1 = x0 + w, y0 + h

        dpx, dpy = self._data_per_point()
        t_rx = self._top_pts * dpx
        t_ry = self._top_pts * dpy
        b_rx = self._bottom_pts * dpx
        b_ry = self._bottom_pts * dpy

        # Clamp to half the shorter side so curves never overshoot.
        half_w = abs(w) / 2.0
        half_h = abs(h) / 2.0
        t_rx = min(max(t_rx, 0.0), half_w)
        t_ry = min(max(t_ry, 0.0), half_h)
        b_rx = min(max(b_rx, 0.0), half_w)
        b_ry = min(max(b_ry, 0.0), half_h)

        # A corner is only drawn when BOTH x- and y-radii are positive.
        draw_b = b_rx > 0 and b_ry > 0
        draw_t = t_rx > 0 and t_ry > 0

        verts: list = []
        codes: list = []

        # Start on the left edge, just above the bottom-left corner.
        start = (x0, y0 + b_ry) if draw_b else (x0, y0)
        verts.append(start)
        codes.append(Path.MOVETO)

        # Bottom-left corner.
        if draw_b:
            verts.extend([(x0, y0), (x0 + b_rx, y0)])
            codes.extend([Path.CURVE3, Path.CURVE3])

        # Bottom edge → bottom-right.
        verts.append((x1 - b_rx if draw_b else x1, y0))
        codes.append(Path.LINETO)

        # Bottom-right corner.
        if draw_b:
            verts.extend([(x1, y0), (x1, y0 + b_ry)])
            codes.extend([Path.CURVE3, Path.CURVE3])

        # Right edge → top-right.
        verts.append((x1, y1 - t_ry if draw_t else y1))
        codes.append(Path.LINETO)

        # Top-right corner.
        if draw_t:
            verts.extend([(x1, y1), (x1 - t_rx, y1)])
            codes.extend([Path.CURVE3, Path.CURVE3])

        # Top edge → top-left.
        verts.append((x0 + t_rx if draw_t else x0, y1))
        codes.append(Path.LINETO)

        # Top-left corner.
        if draw_t:
            verts.extend([(x0, y1), (x0, y1 - t_ry)])
            codes.extend([Path.CURVE3, Path.CURVE3])

        # Close back to start.
        verts.append(start)
        codes.append(Path.CLOSEPOLY)

        return Path(verts, codes)


def apply_border_radius(
    patches: Sequence[Rectangle],
    radius_mm: Tuple[float, float],
    ax: Axes,
) -> None:
    """Swap ``Rectangle`` bars for rounded-corner patches.

    For each :class:`~matplotlib.patches.Rectangle` in ``patches``:

    1. Capture bounds, facecolor, edgecolor, linewidth, hatch, alpha,
       zorder, transform, clip_on, and label (plus publiplots-specific
       ``hatch_linewidth``).
    2. Construct a :class:`_RoundedBarPatch` with the same bounds and
       point-space corner radii derived from ``radius_mm``.
    3. ``rect.remove()`` + ``ax.add_patch(new)`` — swap in place so
       downstream code that walks ``ax.patches`` still sees one patch
       per bar (and a subsequent
       :func:`~publiplots.utils.transparency.apply_transparency` call
       picks up the new artist via the tracker snapshot diff).

    No-op when ``radius_mm`` is ``(0, 0)`` (or effectively zero on both
    sides) — callers can use the same call site for the default flat
    case without paying the swap cost.

    Non-``Rectangle`` patches in ``patches`` are skipped. This is
    deliberate so callers don't have to pre-filter error-bar caps or
    other artists that happen to live in ``ax.patches``.

    Parameters
    ----------
    patches : sequence of Rectangle
        Patches to convert. Typically
        ``ArtistTracker.get_new_patches()`` post-``sns.barplot``.
    radius_mm : tuple of float
        ``(top_mm, bottom_mm)`` — corner radii in millimeters. Use
        :func:`normalize_border_radius` to coerce user input to this
        canonical form.
    ax : Axes
        Axes to re-add the new patches to.

    Returns
    -------
    None
        Mutates ``ax.patches`` in place.
    """
    top_mm, bottom_mm = radius_mm
    if top_mm <= 0 and bottom_mm <= 0:
        return

    top_pts = max(top_mm, 0.0) * _MM_TO_POINTS
    bottom_pts = max(bottom_mm, 0.0) * _MM_TO_POINTS

    # Iterate over a materialized copy — we mutate ax.patches via
    # rect.remove() + ax.add_patch() below.
    for rect in list(patches):
        if not isinstance(rect, Rectangle):
            continue

        x, y = rect.get_xy()
        w = rect.get_width()
        h = rect.get_height()

        # NOTE: we deliberately do NOT copy rect.get_transform() here.
        # Rectangle.get_transform() returns `get_patch_transform() +
        # Artist.get_transform()` — a composite that includes the
        # Rectangle's bbox scaling (unit path [0,1]×[0,1] → bar bbox).
        # Our patch emits its path in data coords directly, so we only
        # want the ax.transData part. Letting matplotlib default the
        # transform when the patch is added via ax.add_patch(new)
        # wires it to ax.transData cleanly.
        new = _RoundedBarPatch(
            xy=(x, y),
            width=w,
            height=h,
            top_pts=top_pts,
            bottom_pts=bottom_pts,
            facecolor=rect.get_facecolor(),
            edgecolor=rect.get_edgecolor(),
            linewidth=rect.get_linewidth(),
            hatch=rect.get_hatch(),
            alpha=rect.get_alpha(),
            zorder=rect.get_zorder(),
            clip_on=rect.get_clip_on(),
            label=rect.get_label(),
        )
        # Preserve publiplots-specific hatch linewidth when present.
        if hasattr(rect, "get_hatch_linewidth") and hasattr(
            new, "set_hatch_linewidth"
        ):
            try:
                new.set_hatch_linewidth(rect.get_hatch_linewidth())
            except Exception:
                # Older mpl without the attr — harmless to skip.
                pass

        rect.remove()
        ax.add_patch(new)


__all__ = [
    "apply_border_radius",
    "normalize_border_radius",
]
