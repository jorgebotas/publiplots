"""Rounded-corner rendering for Rectangle-based plot primitives.

This module provides a plot-agnostic helper that swaps
:class:`matplotlib.patches.Rectangle` patches (e.g. from
``seaborn.barplot``) for :class:`_RoundedBarPatch` instances with
independent per-corner radii. The user-facing API is
``(top_mm, bottom_mm)`` — interpreted as (free-end, base-end) and
rotated by orientation so that horizontal bars round their end caps
rather than their long edges.

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
from matplotlib.patches import Patch, PathPatch, Rectangle
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
    """Rectangle-shaped patch with independent per-corner radii.

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
        Rectangle width and height in data coords (signed; matplotlib
        stores negative values for negative-valued bars).
    top_pts, bottom_pts : float, optional
        Convenience pair: top corners (TL+TR) and bottom corners
        (BL+BR) get the same radius respectively. Used by callers
        that don't care about per-corner control. When
        ``corner_pts`` is also given, ``corner_pts`` wins.
    corner_pts : tuple of 4 floats, keyword-only, optional
        Per-corner radii in points, in order
        ``(top_left, top_right, bottom_right, bottom_left)``. Use
        :func:`apply_border_radius` to translate user-facing
        ``(top_mm, bottom_mm)`` + orientation into this 4-tuple.
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
        top_pts: float = 0.0,
        bottom_pts: float = 0.0,
        *,
        corner_pts: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._rounded_xy = (float(xy[0]), float(xy[1]))
        self._rounded_width = float(width)
        self._rounded_height = float(height)
        if corner_pts is not None:
            tl, tr, br, bl = corner_pts
            self._corner_pts = (
                float(tl), float(tr), float(br), float(bl),
            )
        else:
            t = float(top_pts)
            b = float(bottom_pts)
            self._corner_pts = (t, t, b, b)
        # Kept for back-compat with code reading these attrs.
        self._top_pts = self._corner_pts[0]
        self._bottom_pts = self._corner_pts[3]

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

    def set_xy(self, xy: Tuple[float, float]) -> None:
        """Set the lower-left ``(x, y)`` in data coords.

        Mirrors :meth:`matplotlib.patches.Rectangle.set_xy`. Needed so
        ``offset_patches`` (and any caller that shifts bar positions
        post-hoc) can move a rounded patch — the path is rebuilt at
        draw time from ``_rounded_xy``, so mutating the vertices of
        the path returned by :meth:`get_path` would be a no-op.
        """
        self._rounded_xy = (float(xy[0]), float(xy[1]))
        self.stale = True

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
        """Build the rounded-rectangle path in data coords.

        Corner radii are stored per-corner (TL, TR, BR, BL); each
        corner's x- and y-radii in data coords are computed
        independently and clamped to half of the (unsigned) side
        length so curves never overshoot. Width/height are taken with
        sign — for matplotlib bars built with negative values, this
        means ``y1 < y0`` and the "top" corners (TL/TR) end up at the
        visually-lower edge, which is exactly what we want: the
        free-end of the bar.
        """
        x0, y0 = self._rounded_xy
        w = self._rounded_width
        h = self._rounded_height
        x1, y1 = x0 + w, y0 + h

        dpx, dpy = self._data_per_point()
        half_w = abs(w) / 2.0
        half_h = abs(h) / 2.0

        def _xy(pts: float) -> Tuple[float, float]:
            rx = min(max(pts * dpx, 0.0), half_w)
            ry = min(max(pts * dpy, 0.0), half_h)
            return rx, ry

        tl_rx, tl_ry = _xy(self._corner_pts[0])
        tr_rx, tr_ry = _xy(self._corner_pts[1])
        br_rx, br_ry = _xy(self._corner_pts[2])
        bl_rx, bl_ry = _xy(self._corner_pts[3])

        # A corner is only drawn when BOTH x- and y-radii are positive.
        draw_tl = tl_rx > 0 and tl_ry > 0
        draw_tr = tr_rx > 0 and tr_ry > 0
        draw_br = br_rx > 0 and br_ry > 0
        draw_bl = bl_rx > 0 and bl_ry > 0

        # Signed radii: when w<0 (negative horizontal bar) or h<0
        # (negative vertical bar), the "right" of the bbox is
        # numerically less than its "left" / etc., so we walk along
        # the edge using signed deltas instead of plain subtraction.
        sx = 1.0 if w >= 0 else -1.0
        sy = 1.0 if h >= 0 else -1.0

        verts: list = []
        codes: list = []

        # Start on the left edge, just above the bottom-left corner.
        start = (x0, y0 + sy * bl_ry) if draw_bl else (x0, y0)
        verts.append(start)
        codes.append(Path.MOVETO)

        # Bottom-left corner.
        if draw_bl:
            verts.extend([(x0, y0), (x0 + sx * bl_rx, y0)])
            codes.extend([Path.CURVE3, Path.CURVE3])

        # Bottom edge → bottom-right.
        verts.append((x1 - sx * br_rx if draw_br else x1, y0))
        codes.append(Path.LINETO)

        # Bottom-right corner.
        if draw_br:
            verts.extend([(x1, y0), (x1, y0 + sy * br_ry)])
            codes.extend([Path.CURVE3, Path.CURVE3])

        # Right edge → top-right.
        verts.append((x1, y1 - sy * tr_ry if draw_tr else y1))
        codes.append(Path.LINETO)

        # Top-right corner.
        if draw_tr:
            verts.extend([(x1, y1), (x1 - sx * tr_rx, y1)])
            codes.extend([Path.CURVE3, Path.CURVE3])

        # Top edge → top-left.
        verts.append((x0 + sx * tl_rx if draw_tl else x0, y1))
        codes.append(Path.LINETO)

        # Top-left corner.
        if draw_tl:
            verts.extend([(x0, y1), (x0, y1 - sy * tl_ry)])
            codes.extend([Path.CURVE3, Path.CURVE3])

        # Close back to start.
        verts.append(start)
        codes.append(Path.CLOSEPOLY)

        return Path(verts, codes)


def apply_border_radius(
    patches: Sequence[Patch],
    radius_mm: Tuple[float, float],
    ax: Axes,
    orient: str = "v",
) -> None:
    """Swap ``Rectangle`` / ``PathPatch`` bars for rounded-corner patches.

    For each :class:`~matplotlib.patches.Rectangle` (or rectangular
    :class:`~matplotlib.patches.PathPatch`) in ``patches``:

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

    :class:`~matplotlib.patches.PathPatch` inputs are also accepted:
    seaborn 0.13+ draws ``boxplot`` IQR boxes as ``PathPatch`` with
    rectangular vertex sets, so the helper derives ``(x, y, w, h)``
    from ``patch.get_path().get_extents()`` (axis-aligned bounding
    box) and swaps them for ``_RoundedBarPatch`` instances too.
    Degenerate patches (zero or negative width/height) are skipped.

    Non-``Rectangle``, non-``PathPatch`` patches in ``patches`` are
    skipped. This is deliberate so callers don't have to pre-filter
    error-bar caps or other artists that happen to live in
    ``ax.patches``.

    Parameters
    ----------
    patches : sequence of Patch
        Patches to convert. Typically
        ``ArtistTracker.get_new_patches()`` post-``sns.barplot`` /
        ``sns.boxplot``. Rectangle and rectangular PathPatch inputs
        are handled; other types are skipped.
    radius_mm : tuple of float
        ``(top_mm, bottom_mm)`` in user-facing terms: ``top_mm`` is
        the radius of the **free end** of the bar (visually at the
        end opposite the baseline / zero line) and ``bottom_mm`` is
        the radius of the **base end**. For vertical bars the free
        end is the y-extreme (visually top for positive values,
        bottom for negative); for horizontal bars it is the x-extreme
        (right for positive, left for negative). Sign-awareness comes
        for free from matplotlib's signed width/height storage.
    ax : Axes
        Axes to re-add the new patches to.
    orient : {"v", "h"}, default "v"
        Plot orientation. ``"v"`` rounds the y-extreme corners with
        ``top_mm`` and the y-baseline corners with ``bottom_mm``;
        ``"h"`` rotates the mapping 90° so ``top_mm`` lands on the
        x-extreme corners and ``bottom_mm`` on the x-baseline corners.

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

    # Map (top_mm, bottom_mm) onto (TL, TR, BR, BL) per orientation.
    # Sign-awareness then comes from the patch's signed width/height.
    if orient == "v":
        corner_pts = (top_pts, top_pts, bottom_pts, bottom_pts)
    elif orient == "h":
        corner_pts = (bottom_pts, top_pts, top_pts, bottom_pts)
    else:
        raise ValueError(
            f"orient must be 'v' or 'h', got {orient!r}"
        )

    # Iterate over a materialized copy — we mutate ax.patches via
    # rect.remove() + ax.add_patch() below.
    for patch in list(patches):
        if isinstance(patch, Rectangle):
            x, y = patch.get_xy()
            w = patch.get_width()
            h = patch.get_height()
        elif isinstance(patch, PathPatch):
            # Seaborn 0.13+ draws boxplot IQR boxes as PathPatch with a
            # rectangular vertex set. Derive (x, y, w, h) from the path's
            # axis-aligned bounding box.
            bbox = patch.get_path().get_extents()
            if bbox.width <= 0 or bbox.height <= 0:
                # Degenerate; nothing to round.
                continue
            x, y = bbox.x0, bbox.y0
            w, h = bbox.width, bbox.height
        else:
            # Unknown patch kind (e.g. error-bar caps); leave untouched.
            continue

        # NOTE: we deliberately do NOT copy patch.get_transform() here.
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
            corner_pts=corner_pts,
            facecolor=patch.get_facecolor(),
            edgecolor=patch.get_edgecolor(),
            linewidth=patch.get_linewidth(),
            hatch=patch.get_hatch(),
            alpha=patch.get_alpha(),
            zorder=patch.get_zorder(),
            clip_on=patch.get_clip_on(),
            label=patch.get_label(),
        )
        # Preserve publiplots-specific hatch linewidth when present.
        if hasattr(patch, "get_hatch_linewidth") and hasattr(
            new, "set_hatch_linewidth"
        ):
            try:
                new.set_hatch_linewidth(patch.get_hatch_linewidth())
            except Exception:
                # Older mpl without the attr — harmless to skip.
                pass

        patch.remove()
        ax.add_patch(new)


__all__ = [
    "apply_border_radius",
    "normalize_border_radius",
]
