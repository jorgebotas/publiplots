"""Text color resolution: auto (contrast), hue (palette), or literal."""
from __future__ import annotations

import colorsys
from typing import Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba

from publiplots.annotate._cache import BarRecord


RGBA = Tuple[float, float, float, float]
LUMINANCE_THRESHOLD = 0.408  # W3C / seaborn heatmap convention


def _effective_rgb(face_rgba: RGBA, bg_rgba: RGBA) -> Tuple[float, float, float]:
    """Composite a translucent face over an opaque background using its alpha."""
    fr, fg, fb, fa = face_rgba
    br, bg_, bb, ba = bg_rgba
    # Assume background is opaque (common case); if bg alpha < 1, recurse to fig bg later.
    r = fr * fa + br * (1 - fa)
    g = fg * fa + bg_ * (1 - fa)
    b = fb * fa + bb * (1 - fa)
    return r, g, b


def _background_rgba(ax: Axes) -> RGBA:
    bg = to_rgba(ax.get_facecolor())
    if bg[3] >= 1.0:
        return bg
    fig_bg = to_rgba(ax.figure.get_facecolor())
    return fig_bg


def resolve_color(
    bar: BarRecord,
    color: Union[str, tuple],
    anchor: str,
    ax: Axes,
    hue_active: bool = True,
) -> RGBA:
    """Return RGBA for the label text."""
    if isinstance(color, str) and color == "auto":
        if anchor in ("inside", "center", "base"):
            face = to_rgba(bar.patch.get_facecolor())
            bg = _background_rgba(ax)
            r, g, b = _effective_rgb(face, bg)
            _, lightness, _ = colorsys.rgb_to_hls(r, g, b)
            if lightness < LUMINANCE_THRESHOLD:
                return to_rgba("#ffffff")
            return to_rgba(plt.rcParams["text.color"])
        # outside — nothing's under the label
        return to_rgba(plt.rcParams["text.color"])

    if isinstance(color, str) and color == "hue":
        if not hue_active:
            import warnings
            warnings.warn(
                "pp.annotate: color='hue' requested but plot has no hue; "
                "falling back to color='auto'",
                UserWarning,
                stacklevel=3,
            )
            return resolve_color(bar, "auto", anchor, ax, hue_active=hue_active)
        if bar.hue_color is not None:
            return to_rgba(bar.hue_color)
        edge = to_rgba(bar.patch.get_edgecolor())
        if edge[3] > 0:
            return edge
        return to_rgba(plt.rcParams["text.color"])

    # Literal color
    return to_rgba(color)
