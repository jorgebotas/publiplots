"""Panel constructors and result types for the Composer.

PR 1 ships only :class:`PanelAxes`. PR 3 adds :class:`PanelGrid` /
:class:`PanelText`; PR 5 adds :class:`PanelImage`.

Pure-Python module — no matplotlib imports. The result type
:class:`Panel` carries an opaque ``ax`` reference that the canvas
populates; that's the only object that crosses the Python/matplotlib
boundary, and it's typed as ``Any`` here for layering purity.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple


PanelKind = Literal["axes", "axesgrid", "image", "text"]  # only "axes" used in PR 1


@dataclass(frozen=True)
class PanelAxes:
    """Input record for an axes panel.

    Pass this to :meth:`Canvas.add_row` to declare a panel containing a
    single matplotlib Axes. The canvas allocates the Axes at the panel's
    mm rect and stores the result as a :class:`Panel`.

    Parameters
    ----------
    label : str
        Caption-addressable identifier (``'A'``, ``'B'``, ``'a.i'``, ...).
        PR 1 requires an explicit string; PR 2 adds auto-letter
        sequencing for ``label=None``.
    size : tuple of (width_mm, height_mm)
        Panel mm rect. Both dimensions must be positive numerics. PR 2
        adds ``'flex'`` for the width to absorb leftover row space.

    Notes
    -----
    Frozen dataclass — instances are immutable and hashable.
    """

    label: str
    size: Tuple[float, float]

    def __post_init__(self) -> None:
        # Validate label
        if not isinstance(self.label, str):
            raise TypeError(
                f"label must be a string, got {type(self.label).__name__}"
            )

        # Validate size: must be a 2-tuple of positive numerics
        try:
            n = len(self.size)
        except TypeError:
            raise ValueError(
                f"size must be a 2-tuple of (width_mm, height_mm), got {self.size!r}"
            )
        if n != 2:
            raise ValueError(
                f"size must be a 2-tuple of (width_mm, height_mm), got length {n}"
            )
        w, h = self.size
        for name, v in (("width", w), ("height", h)):
            if isinstance(v, str):
                raise ValueError(
                    f"size {name} must be numeric (PR 2 will add 'flex'), got {v!r}"
                )
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise ValueError(
                    f"size {name} must be numeric, got {v!r}"
                )
            if fv <= 0:
                raise ValueError(f"size {name} must be positive, got {fv}")

        # Coerce to floats. dataclass(frozen=True) requires object.__setattr__.
        object.__setattr__(self, "size", (float(w), float(h)))


@dataclass(frozen=True)
class Panel:
    """Result type returned by ``canvas[label]``.

    Carries the resolved axes reference and the panel's mm geometry.
    Frozen so users don't accidentally mutate cached metadata.

    For ``kind='axes'``, ``ax`` is a single :class:`matplotlib.axes.Axes`.
    PR 3 adds ``kind='axesgrid'`` (where ``axes`` is a numpy array) and
    other panel kinds.

    Attributes
    ----------
    label : str
        The caption-addressable identifier.
    kind : str
        One of ``'axes'``, ``'axesgrid'``, ``'image'``, ``'text'``.
        PR 1 only emits ``'axes'``.
    ax : matplotlib.axes.Axes or None
        The underlying axes for ``kind='axes'``. ``None`` for non-axes
        panels (PR 5+).
    size_mm : tuple of (width_mm, height_mm)
        Panel mm rect dimensions.
    bbox_mm : tuple of (x_mm, y_mm, w_mm, h_mm)
        Panel mm rect position relative to the canvas. ``(x, y)`` is the
        bottom-left corner; ``(w, h)`` matches ``size_mm``.
    """

    label: str
    kind: PanelKind
    ax: Optional[Any]  # matplotlib.axes.Axes | None — typed Any for layering
    size_mm: Tuple[float, float]
    bbox_mm: Tuple[float, float, float, float]
