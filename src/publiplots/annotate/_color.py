"""Text color resolution: auto (contrast), hue (palette), or literal."""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba

from publiplots.annotate._cache import BarRecord


RGBA = Tuple[float, float, float, float]
_WHITE: RGBA = (1.0, 1.0, 1.0, 1.0)


def relative_luminance(rgb: Sequence[float]) -> float:
    """WCAG 2.x relative luminance of an sRGB colour with components in 0-1."""
    def _linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (_linear(float(c)) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def white_wins(face_rgba: RGBA, bg_rgba: RGBA) -> bool:
    """True when white has a higher WCAG contrast ratio than black against
    ``face_rgba`` composited over ``bg_rgba``.

    Black stands in for publiplots' INK. White wins below relative
    luminance ~0.179, where the two ratios cross. The answer does not
    depend on ``rcParams["text.color"]``, so callers can tell "the fill is
    dark" apart from "the ink happens to be white".
    """
    y = relative_luminance(_effective_rgb(face_rgba, bg_rgba))
    on_ink = (y + 0.05) / 0.05
    on_white = 1.05 / (y + 0.05)
    return on_white > on_ink


def contrast_ink(face_rgba: RGBA, bg_rgba: RGBA) -> RGBA:
    """Ink with the higher WCAG contrast ratio against ``face_rgba``
    composited over ``bg_rgba``: white, or ``rcParams["text.color"]``.

    See `white_wins` for the rule.
    """
    if white_wins(face_rgba, bg_rgba):
        return _WHITE
    return to_rgba(plt.rcParams["text.color"])


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
            return contrast_ink(face, _background_rgba(ax))
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
