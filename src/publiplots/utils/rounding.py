"""Rounded-corner rendering for Rectangle-based plot primitives.

This module provides a plot-agnostic helper that swaps
:class:`matplotlib.patches.Rectangle` patches (e.g. from
``seaborn.barplot``) for :class:`matplotlib.patches.FancyBboxPatch`
instances with independent top / bottom corner radii. Radii are
specified in millimeters and converted to points via
``72 / 25.4``, so the rendered corner radius is scale-invariant
(constant on screen/print regardless of data-axis ranges).

The intended consumer is :func:`publiplots.barplot`, but the helper is
designed to generalize to any Rectangle-based primitive (histograms,
stacked bars, future ``pp.histplot``).
"""

from typing import Optional, Sequence, Tuple, Union

from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Rectangle

# Units: user supplies radii in millimeters. FancyBboxPatch evaluates
# the callable BoxStyle in point space, so the conversion is a constant.
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


class _AsymmetricRoundBoxStyle:
    """Callable BoxStyle with independent top and bottom corner radii.

    Usable anywhere :class:`matplotlib.patches.FancyBboxPatch` accepts
    a ``boxstyle=`` argument. Renders in **point space** (via
    ``mutation_size``), giving scale-invariant rounding on screens and
    print regardless of the axes' data ranges or aspect ratio.

    Each corner is a quadratic Bezier (``Path.CURVE3``) with the
    control vertex at the rectangle corner itself and the end vertex
    on the adjacent edge. Radii are clamped to half the shorter side
    so the curves never overshoot.

    Parameters
    ----------
    top_pts : float
        Top corner radius in points. ``0`` means square top corners.
    bottom_pts : float
        Bottom corner radius in points. ``0`` means square bottom
        corners.
    """

    def __init__(self, top_pts: float, bottom_pts: float):
        self.top = float(top_pts)
        self.bottom = float(bottom_pts)

    def __call__(
        self,
        x0: float,
        y0: float,
        width: float,
        height: float,
        mutation_size: float,
    ):
        """Build the rounded-rectangle Path.

        Signature matches matplotlib's documented callable-BoxStyle
        contract: ``(x0, y0, width, height, mutation_size)``. All four
        position/size arguments are in the same units (points when
        called by :class:`FancyBboxPatch`).
        """
        from matplotlib.path import Path

        # Clamp to half the shorter side so curves never overshoot.
        half_w = width / 2.0
        half_h = height / 2.0
        t = min(self.top, half_w, half_h)
        b = min(self.bottom, half_w, half_h)
        # Don't let tiny numerical noise create degenerate curves.
        if t < 0:
            t = 0.0
        if b < 0:
            b = 0.0

        x1 = x0 + width
        y1 = y0 + height

        verts = []
        codes = []

        # Start on the left edge, just above the bottom-left corner.
        start = (x0, y0 + b) if b > 0 else (x0, y0)
        verts.append(start)
        codes.append(Path.MOVETO)

        # Bottom-left corner.
        if b > 0:
            # CURVE3 uses 2 verts per curve: (control, end).
            verts.extend([(x0, y0), (x0 + b, y0)])
            codes.extend([Path.CURVE3, Path.CURVE3])

        # Bottom edge → bottom-right.
        verts.append((x1 - b if b > 0 else x1, y0))
        codes.append(Path.LINETO)

        # Bottom-right corner.
        if b > 0:
            verts.extend([(x1, y0), (x1, y0 + b)])
            codes.extend([Path.CURVE3, Path.CURVE3])

        # Right edge → top-right.
        verts.append((x1, y1 - t if t > 0 else y1))
        codes.append(Path.LINETO)

        # Top-right corner.
        if t > 0:
            verts.extend([(x1, y1), (x1 - t, y1)])
            codes.extend([Path.CURVE3, Path.CURVE3])

        # Top edge → top-left.
        verts.append((x0 + t if t > 0 else x0, y1))
        codes.append(Path.LINETO)

        # Top-left corner.
        if t > 0:
            verts.extend([(x0, y1), (x0, y1 - t)])
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
    """Swap ``Rectangle`` bars for ``FancyBboxPatch``es with rounded corners.

    For each :class:`~matplotlib.patches.Rectangle` in ``patches``:

    1. Capture bounds, facecolor, edgecolor, linewidth, hatch, alpha,
       zorder, transform, clip_on, and label (plus publiplots-specific
       ``hatch_linewidth``).
    2. Construct a :class:`~matplotlib.patches.FancyBboxPatch` with the
       same bounds and an :class:`_AsymmetricRoundBoxStyle` derived
       from ``radius_mm``.
    3. ``rect.remove()`` + ``ax.add_patch(new)`` — swap in place so
       downstream code that walks ``ax.patches`` still sees one patch
       per bar (and a subsequent :func:`~publiplots.utils.transparency`
       call picks up the new artist).

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
        Axes to re-add the new FancyBboxPatches to.

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
    boxstyle = _AsymmetricRoundBoxStyle(top_pts, bottom_pts)

    # Iterate over a materialized copy — we mutate ax.patches via
    # rect.remove() + ax.add_patch() below.
    for rect in list(patches):
        if not isinstance(rect, Rectangle):
            continue

        x, y = rect.get_xy()
        w = rect.get_width()
        h = rect.get_height()

        new = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=boxstyle,
            facecolor=rect.get_facecolor(),
            edgecolor=rect.get_edgecolor(),
            linewidth=rect.get_linewidth(),
            hatch=rect.get_hatch(),
            alpha=rect.get_alpha(),
            zorder=rect.get_zorder(),
            transform=rect.get_transform(),
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
