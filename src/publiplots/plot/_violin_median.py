"""The median-line inner mark for ``pp.violinplot(inner="median")``.

seaborn rejects ``inner="median"``, so violinplot passes ``inner=None`` down
and calls `draw_median_inner` once the bodies are final — after ``side=``
clipping, after the edge colour override, after the box-stats meta is
built. Each line is read off its body's drawn polygon, so it spans the
body's thickness *at the median* (on a skewed distribution that is not the
widest point), and a split or ``side=`` half yields a half-width line.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cbook
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from publiplots.annotate._cache import BoxStatsMeta
from publiplots.annotate._color import _background_rgba, contrast_ink
from publiplots.annotate._positioning import mm_to_data

_PT_TO_MM = 25.4 / 72.0
# A large inset on a narrow body degrades to a short tick, never to nothing.
_MIN_SPAN_FRACTION = 0.2
_WHITE = (1.0, 1.0, 1.0, 1.0)


def local_span(body, value: float, val_axis: int) -> Optional[Tuple[float, float]]:
    """Where the scan line at ``value`` crosses ``body``'s outline.

    Returns ``(lo, hi)`` on the position axis, or None when the line misses
    the body. ``val_axis`` is 1 for vertical violins (value on y), 0 for
    horizontal ones.
    """
    pos_axis = 1 - val_axis
    hits: List[float] = []
    for path in body.get_paths():
        for poly in path.to_polygons():
            a, b = poly, np.roll(poly, -1, axis=0)
            va, vb = a[:, val_axis], b[:, val_axis]
            crosses = (va != vb) & ((va - value) * (vb - value) <= 0)
            t = (value - va[crosses]) / (vb[crosses] - va[crosses])
            hits.extend(a[crosses, pos_axis]
                        + t * (b[crosses, pos_axis] - a[crosses, pos_axis]))
    if len(hits) < 2:
        return None
    return float(min(hits)), float(max(hits))


def _validate_inset(inset: Any) -> Union[str, float]:
    if isinstance(inset, str) and inset == "auto":
        return inset
    if (isinstance(inset, bool) or not isinstance(inset, (int, float))
            or not 0 <= inset < 1):
        raise ValueError(
            "inner_kws['inset'] must be 'auto' or a number in [0, 1); "
            f"got {inset!r}"
        )
    return float(inset)


def _pad(inset: Union[str, float], half: float, linewidth: float,
         ax: Axes, pos_name: str) -> float:
    """Data-unit length to trim from each end of a line of half-length ``half``."""
    if inset == "auto":
        # Half the stroke: a round cap then ends on the outline's centreline,
        # where the outline's own stroke covers it while the widths match.
        pad = abs(mm_to_data(linewidth / 2 * _PT_TO_MM, ax, pos_name))
    else:
        pad = inset * half
    return min(pad, (1 - _MIN_SPAN_FRACTION) * half)


def _auto_color(body, ax: Axes):
    """The body's edge colour, or white where its fill is too dark for it."""
    edges = np.asarray(body.get_edgecolor()).reshape(-1, 4)
    edge = tuple(edges[0]) if len(edges) else to_rgba(plt.rcParams["text.color"])
    faces = np.asarray(body.get_facecolor()).reshape(-1, 4)
    if len(faces) and contrast_ink(tuple(faces[0]), _background_rgba(ax)) == _WHITE:
        return _WHITE
    return edge


def draw_median_inner(
    ax: Axes,
    meta: BoxStatsMeta,
    inner_kws: Dict[str, Any],
    *,
    linewidth: float,
) -> List[Line2D]:
    """Draw one median line per violin body in ``meta``.

    ``inner_kws`` keys: ``inset`` (``"auto"`` or a number in ``[0, 1)``:
    the fraction of the local half-thickness trimmed from each end),
    ``color`` (``"auto"`` or any matplotlib colour), ``linewidth``
    (defaults to the violin outline's ``linewidth``), ``solid_capstyle``,
    ``zorder`` (defaults to one above the body). Line2D aliases such as
    ``c`` and ``lw`` are accepted. Anything else is passed to
    :meth:`~matplotlib.axes.Axes.plot`.
    """
    kws = cbook.normalize_kwargs(dict(inner_kws), Line2D)
    # The line sits inside an already-drawn body, so it must never rescale
    # the axes, whatever the caller asks for.
    kws.pop("scalex", None)
    kws.pop("scaley", None)
    inset = _validate_inset(kws.pop("inset", "auto"))
    color = kws.pop("color", "auto")
    lw = kws.pop("linewidth", linewidth)
    capstyle = kws.pop("solid_capstyle", "butt" if inset == 0 else "round")
    zorder = kws.pop("zorder", None)
    auto_color = isinstance(color, str) and color == "auto"
    val_axis = 1 if meta.orient == "v" else 0
    pos_name = "x" if meta.orient == "v" else "y"

    lines: List[Line2D] = []
    for rec in meta.boxes:
        median = rec.stats.get("median")
        if rec.artist is None or median is None or not np.isfinite(median):
            continue
        span = local_span(rec.artist, median, val_axis)
        if span is None:
            continue
        lo, hi = span
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        half -= _pad(inset, half, lw, ax, pos_name)
        pos, val = [mid - half, mid + half], [median, median]
        xs, ys = (pos, val) if meta.orient == "v" else (val, pos)
        (line,) = ax.plot(
            xs, ys,
            color=_auto_color(rec.artist, ax) if auto_color else color,
            linewidth=lw,
            solid_capstyle=capstyle,
            zorder=rec.artist.get_zorder() + 1 if zorder is None else zorder,
            scalex=False, scaley=False,
            **kws,
        )
        lines.append(line)
    return lines
