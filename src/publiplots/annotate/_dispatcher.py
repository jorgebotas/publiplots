"""Public entry point: annotate(ax, kind=..., ...).

Validates inputs and dispatches to a registered strategy. Adding a new
strategy means: write the `_<kind>_strategy` function in its own file and
register it in `_STRATEGIES` below.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Union

from matplotlib.axes import Axes
from matplotlib.text import Text

from publiplots.annotate.bar_values import _bar_values_strategy


_STRATEGIES: dict[str, Callable] = {
    "bar_values": _bar_values_strategy,
}

_VALID_ANCHORS = {"outside", "inside", "base", "center"}


def annotate(
    ax: Axes,
    kind: str = "bar_values",
    *,
    fmt: str = ".2f",
    anchor: str = "outside",
    offset: float = 1.0,
    color: Union[str, tuple] = "auto",
    pad: Optional[float] = None,
    **text_kws,
) -> List[Text]:
    """Add value labels to plot marks on `ax`.

    Parameters
    ----------
    ax : Axes
        The axes to annotate. Must already have marks drawn on it.
    kind : str, default="bar_values"
        Which strategy to use. v1 ships 'bar_values'.
    fmt : str, default=".2f"
        Either a bare format spec (e.g. ".2f") or a format-string template
        containing {} (e.g. "{:,.1f}%").
    anchor : {"outside", "inside", "base", "center"}, default="outside"
        Where to place the label relative to the bar.
    offset : float, default=1.0
        Gap in millimeters between the bar (or errorbar cap) and the label,
        applied in the outward direction of the anchor.
    color : str or tuple, default="auto"
        "auto" = contrast-aware (compositing for translucent fills); "hue" =
        use the bar's palette color; any matplotlib color passes through.
    pad : float, optional
        Extra margin in millimeters between the label and the axis edge when
        auto-expanding limits. Defaults to `offset` so the geometry reads
        bar → offset → label → pad → axis edge.
    **text_kws
        Forwarded to `ax.text` (fontsize, fontweight, etc.).

    Returns
    -------
    list of Text
        The Text artists created, in bar order.
    """
    if kind not in _STRATEGIES:
        raise ValueError(
            f"unknown kind={kind!r}; known: {sorted(_STRATEGIES)}"
        )
    if anchor not in _VALID_ANCHORS:
        raise ValueError(
            f"anchor must be one of {sorted(_VALID_ANCHORS)}; got {anchor!r}"
        )
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if pad is None:
        pad = offset
    if pad < 0:
        raise ValueError("pad must be >= 0")

    return _STRATEGIES[kind](
        ax, fmt=fmt, anchor=anchor, offset=offset, color=color, pad=pad,
        **text_kws,
    )
